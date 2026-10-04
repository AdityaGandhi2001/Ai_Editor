"""
Thin, budget-aware LLM client.

* Primary transport is LiteLLM (as in the starter code). If LiteLLM cannot be
  imported, it falls back to a plain OpenAI-compatible HTTP call via `requests`,
  so a broken install never stops `rectify-all`.
* Every response is cached on disk (keyed by model + messages + params), so
  re-running the pipeline never pays twice for an identical request.
* Token usage of every non-cached call is appended to logs/usage.jsonl.
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    import litellm
    from litellm import completion as _litellm_completion

    litellm.suppress_debug_info = True
    litellm.drop_params = True
except Exception:  # pragma: no cover - fallback path
    _litellm_completion = None

def env_num(name, default, cast=float):
    """Read a numeric setting; an empty or malformed value falls back to the default instead of crashing."""
    try:
        return cast(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return cast(default)


MODEL = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")
API_KEY = os.getenv("LLM_API_KEY")
if API_KEY and ("PASTE" in API_KEY or not API_KEY.strip()):
    API_KEY = None  # placeholder left in .env
API_BASE = os.getenv("LLM_API_BASE")

CACHE_DIR = Path(os.getenv("RECTIFIER_CACHE_DIR", ".llm_cache"))
USE_CACHE = os.getenv("RECTIFIER_NO_CACHE", "") == ""
USAGE_LOG = Path("logs/usage.jsonl")
MAX_RETRIES = env_num("RECTIFIER_MAX_RETRIES", 4, int)
TIMEOUT = env_num("RECTIFIER_TIMEOUT", 300, int)
MAX_TOKENS = env_num("RECTIFIER_MAX_TOKENS", 12000, int)

# ---- Budget guards -------------------------------------------------------
# Conservative (slightly above list) gpt-oss-120b prices, USD per 1M tokens.
PRICE_IN = env_num("RECTIFIER_PRICE_IN_PER_M", 0.15)
PRICE_OUT = env_num("RECTIFIER_PRICE_OUT_PER_M", 0.75)
# Hard cap on what a single process (one `rectify-all` / `test`) may spend.
MAX_RUN_USD = env_num("RECTIFIER_MAX_RUN_USD", 0.80)
# Refuse to call the LLM when the proxy reports less than this remaining.
MIN_REMAINING_USD = env_num("RECTIFIER_MIN_REMAINING_USD", 0.50)
BUDGET_RECHECK_EVERY = env_num("RECTIFIER_BUDGET_RECHECK_EVERY", 15, int)


class BudgetGuardError(RuntimeError):
    """Raised instead of calling the LLM when a budget limit would be crossed."""


_lock = threading.Lock()
usage_totals = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "est_usd": 0.0}
_budget = {"stopped": None, "last_check_call": None, "remaining": None}
# Circuit breaker: after this many consecutive requests fail every retry, assume the API is down
# and pause calling it, so a true outage finishes quickly with fallback instead of hanging.
# The breaker is RECOVERABLE: after a cooldown it lets one probe through, so a brief blip
# (a few seconds of 5xx) only pauses the run rather than dumping every later article to fallback.
BREAKER_LIMIT = env_num("RECTIFIER_BREAKER_LIMIT", 6, int)
BREAKER_COOLDOWN = env_num("RECTIFIER_BREAKER_COOLDOWN", 20, int)
_breaker = {"consecutive_failures": 0, "opened_at": 0.0}


def remote_budget():
    """(spend, max_budget) from the LiteLLM proxy /key/info, or None if unavailable."""
    if not API_KEY or not API_BASE:
        return None
    try:
        import requests

        r = requests.get(
            API_BASE.rstrip("/") + "/key/info",
            headers={"Authorization": f"Bearer {API_KEY}"},
            params={"key": API_KEY},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        info = r.json()
        info = info.get("info", info)
        return float(info.get("spend") or 0.0), info.get("max_budget")
    except Exception:
        return None


def _est_cost(prompt_tokens, completion_tokens):
    return prompt_tokens * PRICE_IN / 1e6 + completion_tokens * PRICE_OUT / 1e6


def _check_budget(messages):
    """Called under _lock before every paid request."""
    if _budget["stopped"]:
        raise BudgetGuardError(_budget["stopped"])
    # Worst case for this call: full prompt + max_tokens of output.
    prompt_est = sum(len(m["content"]) for m in messages) / 3.0
    worst = _est_cost(prompt_est, MAX_TOKENS)
    if usage_totals["est_usd"] + worst > MAX_RUN_USD:
        _budget["stopped"] = (
            f"per-run cap reached: est. ${usage_totals['est_usd']:.4f} spent + ${worst:.4f} worst case "
            f"> RECTIFIER_MAX_RUN_USD=${MAX_RUN_USD:.2f}"
        )
        raise BudgetGuardError(_budget["stopped"])
    n = usage_totals["calls"]
    last = _budget["last_check_call"]
    if last is None or n - last >= BUDGET_RECHECK_EVERY:
        _budget["last_check_call"] = n
        rb = remote_budget()
        if rb and rb[1] is not None:
            remaining = float(rb[1]) - rb[0]
            _budget["remaining"] = remaining
            if remaining < MIN_REMAINING_USD:
                _budget["stopped"] = (
                    f"proxy reports ${remaining:.4f} remaining < RECTIFIER_MIN_REMAINING_USD=${MIN_REMAINING_USD:.2f}"
                )
                raise BudgetGuardError(_budget["stopped"])


def _cache_key(messages, params) -> str:
    blob = json.dumps({"model": MODEL, "messages": messages, "params": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _http_completion(messages, params):
    """OpenAI-compatible fallback when LiteLLM is unavailable."""
    import requests

    # Mirror LiteLLM routing: the leading provider segment is not sent upstream
    # ("openai/gpt-oss-120b" -> "gpt-oss-120b", "groq/openai/gpt-oss-120b" -> "openai/gpt-oss-120b").
    provider, _, rest = MODEL.partition("/")
    model = rest if rest and provider in ("openai", "groq", "litellm_proxy", "hosted_vllm") else MODEL
    base = (API_BASE or "").rstrip("/")
    url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
    body = {"model": model, "messages": messages}
    body.update({k: v for k, v in params.items() if k != "extra_body"})
    body.update(params.get("extra_body", {}))
    r = requests.post(url, headers={"Authorization": f"Bearer {API_KEY}"}, json=body, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"].get("content") or "", data.get("usage") or {}


def _litellm_call(messages, params):
    resp = _litellm_completion(
        model=MODEL,
        messages=messages,
        api_key=API_KEY,
        api_base=API_BASE,
        timeout=TIMEOUT,
        **params,
    )
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    usage = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }
    return content, usage


def _record(tag, usage):
    with _lock:
        usage_totals["calls"] += 1
        usage_totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
        usage_totals["completion_tokens"] += usage.get("completion_tokens", 0)
        cost = _est_cost(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        usage_totals["est_usd"] += cost
        try:  # logging is best-effort: a read-only disk must never cost us a good answer
            USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with USAGE_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "tag": tag, **usage, "est_usd": round(cost, 6)}) + "\n")
        except OSError:
            pass


_reserved = {"usd": 0.0}


def chat(messages, tag: str = "", reasoning_effort: str = "medium", max_tokens: int = None) -> str:
    """Send a chat request and return the assistant text. Raises on failure / budget guard."""
    params = {
        "temperature": 0,
        "max_tokens": max_tokens or MAX_TOKENS,
        "extra_body": {"reasoning_effort": reasoning_effort},
    }
    key = _cache_key(messages, params)
    cache_file = CACHE_DIR / f"{key}.json"
    if USE_CACHE and cache_file.exists():
        try:
            content = json.loads(cache_file.read_text(encoding="utf-8"))["content"]
            with _lock:
                usage_totals["cached"] += 1
            return content
        except Exception:
            pass

    if not API_KEY:
        raise RuntimeError("LLM_API_KEY is not set (fill it in .env)")

    with _lock:
        if _breaker["consecutive_failures"] >= BREAKER_LIMIT:
            # Open: refuse fast, but after a cooldown let ONE probe through (half-open).
            if time.time() - _breaker["opened_at"] < BREAKER_COOLDOWN:
                raise RuntimeError("API unavailable (circuit breaker open); using fallback")
            _breaker["consecutive_failures"] = BREAKER_LIMIT - 1  # allow a single probe

    paid_failures = 0  # billed responses that were unusable (e.g. empty)
    last_err = None
    for attempt in range(MAX_RETRIES):
        # Reserve worst-case cost so parallel workers cannot jointly overshoot the cap.
        worst = _est_cost(sum(len(m["content"]) for m in messages) / 3.0, params["max_tokens"])
        with _lock:
            usage_totals["est_usd"] += _reserved["usd"]
            try:
                _check_budget(messages)
            finally:
                usage_totals["est_usd"] -= _reserved["usd"]
            _reserved["usd"] += worst
        try:
            if _litellm_completion is not None:
                content, usage = _litellm_call(messages, params)
            else:
                content, usage = _http_completion(messages, params)
        except Exception as e:  # rate limits, timeouts, transient 5xx: not billed
            last_err = e
            msg = str(e).lower()
            if any(s in msg for s in ("budget", "exceeded", "invalid api key", "authentication", "401", "403")):
                with _lock:
                    _budget["stopped"] = f"provider refused: {e}"
                raise BudgetGuardError(_budget["stopped"])
            time.sleep(min(60, 4 * 2 ** attempt))
            continue
        finally:
            with _lock:
                _reserved["usd"] -= worst
        _record(tag, usage)
        with _lock:
            _breaker["consecutive_failures"] = 0
        if content.strip():
            if USE_CACHE:
                try:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
                except OSError:
                    pass
            return content
        paid_failures += 1
        last_err = RuntimeError("empty completion (reasoning used the whole token budget)")
        if paid_failures >= 2:
            break  # do not keep paying for unusable responses
        # Same settings would loop the same way at temperature 0: step the effort down.
        lower = {"high": "medium", "medium": "low"}.get(params["extra_body"]["reasoning_effort"])
        if not lower:
            break
        params = {**params, "extra_body": {"reasoning_effort": lower}}
    if paid_failures == 0:  # never got a response at all: count towards the breaker
        with _lock:
            _breaker["consecutive_failures"] += 1
            if _breaker["consecutive_failures"] >= BREAKER_LIMIT:
                _breaker["opened_at"] = time.time()  # start the cooldown; recoverable, not permanent
    raise RuntimeError(f"LLM call failed: {last_err}")
