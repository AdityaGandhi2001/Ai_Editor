"""
Deterministic text utilities for the rectification pipeline.

Nothing in here calls an LLM. These helpers:
  * split an AI-generated article into its body and the trailing
    "**Error Annotations:**" block that most files carry,
  * parse that block leniently (two files contain malformed JSON),
  * apply LLM-proposed find/replace edits surgically, so every character the
    LLM did not explicitly target is preserved byte-for-byte.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# The annotation block appears as "**Error Annotations:**" or "## Error Annotations:".
_ANNOTATION_RE = re.compile(r"\n[ \t]*(?:\*\*|#+[ \t]*)?Error Annotations?:?(?:\*\*)?:?[ \t]*\n", re.IGNORECASE)


def split_article(raw: str) -> Tuple[str, str]:
    """Return (body, annotation_block). annotation_block is '' if absent."""
    m = _ANNOTATION_RE.search(raw)
    if not m:
        return raw.rstrip(), ""
    return raw[: m.start()].rstrip(), raw[m.end():].strip()


def clean_output(text: str) -> str:
    """Final hygiene: no code fences, no annotation residue, no trailing whitespace."""
    text, _ = split_article(text)
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.rstrip()


def _unescape(s: str) -> str:
    try:
        return json.loads('"' + s + '"')
    except Exception:
        return s.replace('\\"', '"').replace("\\n", "\n")


def parse_annotations(block: str) -> List[dict]:
    """Parse the annotation block into a list of {error, correction, error_type}."""
    if not block:
        return []
    start, end = block.find("["), block.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(block[start : end + 1])
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except Exception:
            pass
    # Lenient fallback: regex out the individual string fields per object.
    items = []
    for obj in re.findall(r"\{(.*?)\}", block, flags=re.S):
        item = {}
        for key in ("location", "error", "correction", "error_type"):
            m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % key, obj, flags=re.S)
            if m:
                item[key] = _unescape(m.group(1))
        if item.get("error"):
            items.append(item)
    return items


def format_hints(annotations: List[dict]) -> str:
    if not annotations:
        return "(none provided - you must find the errors yourself by comparing with the SOURCE)"
    lines = []
    for i, a in enumerate(annotations, 1):
        lines.append(
            f"{i}. flagged text: {a.get('error', '').strip()}\n"
            f"   suggested fix: {a.get('correction', '').strip()}\n"
            f"   type: {a.get('error_type', '').strip()}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Surgical edit application
# --------------------------------------------------------------------------

_QUOTE_CLASS = {
    '"': "[\"“”„″]",
    "“": "[\"“”„″]",
    "”": "[\"“”„″]",
    "'": "['‘’‚′`]",
    "‘": "['‘’‚′`]",
    "’": "['‘’‚′`]",
    "-": "[-‐‑‒–—−]",
    "–": "[-‐‑‒–—−]",
    "—": "[-‐‑‒–—−]",
}


def _loose_pattern(snippet: str) -> str:
    """Regex that tolerates whitespace and quote/dash variants."""
    out = []
    for ch in snippet.strip():
        if ch.isspace():
            if not out or out[-1] != r"\s+":
                out.append(r"\s+")
        elif ch in _QUOTE_CLASS:
            out.append(_QUOTE_CLASS[ch])
        else:
            out.append(re.escape(ch))
    return "".join(out)


@dataclass
class EditResult:
    text: str
    applied: List[dict] = field(default_factory=list)
    failed: List[dict] = field(default_factory=list)


def _on_boundaries(text: str, s: int, t: int) -> bool:
    """Reject matches that start/end inside a word or number ("201" inside "2016")."""
    if s > 0 and text[s].isalnum() and text[s - 1].isalnum():
        return False
    if t < len(text) and text[t - 1].isalnum() and text[t].isalnum():
        return False
    return True


def _find_spans(text: str, original: str) -> List[Tuple[int, int]]:
    spans = []
    if original in text:
        i = text.find(original)
        while i != -1:
            spans.append((i, i + len(original)))
            i = text.find(original, i + len(original))
    else:
        pat = _loose_pattern(original)
        if pat:
            spans = [(m.start(), m.end()) for m in re.finditer(pat, text)]
    return [(s, t) for s, t in spans if _on_boundaries(text, s, t)]


def _fix_spacing(text: str, start: int, end: int, replacement: str) -> Tuple[int, int, str]:
    """When an edit deletes a span, avoid leaving doubled spaces or ' ,' artefacts."""
    if replacement.strip():
        return start, end, replacement
    before = text[:start]
    after = text[end:]
    if before.endswith(" ") and (after.startswith(" ") or after[:1] in ",.;:!?)"):
        start -= 1
    elif not before or before.endswith("\n"):
        while end < len(text) and text[end] == " ":
            end += 1
    return start, end, replacement


def apply_edits(text: str, edits: List[dict]) -> EditResult:
    """
    Apply a list of {"original": str, "corrected": str} edits.

    Every occurrence of an `original` span is replaced (the LLM is told to add
    context when only some occurrences are wrong). Overlapping edits are
    resolved in favour of the one listed first.
    """
    result = EditResult(text=text)
    planned: List[Tuple[int, int, str, dict]] = []
    for e in edits:
        orig = e.get("original")
        corr = e.get("corrected")
        if not isinstance(orig, str) or not isinstance(corr, str) or not orig.strip():
            result.failed.append(e)
            continue
        if orig == corr:
            continue
        spans = _find_spans(text, orig)
        if not spans:
            result.failed.append(e)
            continue
        ok = False
        for s, t in spans:
            if any(not (t <= ps or s >= pt) for ps, pt, _, _ in planned):
                continue
            planned.append((s, t, corr, e))
            ok = True
        if ok:
            result.applied.append(e)
    new = text
    for s, t, corr, _ in sorted(planned, key=lambda p: p[0], reverse=True):
        s, t, corr = _fix_spacing(new, s, t, corr)
        # a/an agreement only in the window touching the edit (preceding article + replacement)
        lo = max(0, s - 4)
        window = fix_articles(new[lo:s] + corr) if corr.strip() else new[lo:s] + corr
        new = new[:lo] + window + new[t:]
    result.text = new
    return result


def extract_json(content: Optional[str]):
    """Pull the first JSON object/array out of an LLM response."""
    if not content:
        return None
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except Exception:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        s, t = content.find(open_ch), content.rfind(close_ch)
        if s != -1 and t > s:
            try:
                return json.loads(content[s : t + 1])
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------
# Deterministic edit filters (precision guards)
# --------------------------------------------------------------------------

_STYLE_SUBS = [
    (r"\brs\.?\s*", "₹"), (r"\binr\s*", "₹"), (r"colour", "color"), (r"aluminium", "aluminum"),
    (r"vapour", "vapor"), (r"centre", "center"), (r"theatre", "theater"), (r"favour", "favor"),
    (r"honour", "honor"), (r"labour", "labor"), (r"defence", "defense"), (r"programme", "program"),
    (r"([a-z])is(e|ed|es|ing|ation)\b", r"\1iz\2"), (r"\bper cent\b", "%"), (r"\bpercent\b", "%"),
]


def _canon(s: str) -> str:
    s = s.lower()
    s = re.sub(r"(\d)\.(\d)", r"\1p\2", s)  # 4.1 != 41: decimal points are content
    s = re.sub(r"\b(not|no|never)\b", r"_\1_", s)
    for pat, rep in _STYLE_SUBS:
        s = re.sub(pat, rep, s)
    return re.sub(r"[^\w₹$%]|_", "", s)  # \w is Unicode-aware: Devanagari etc. count as content


def is_style_only(original: str, corrected: str) -> bool:
    """True if the edit changes only spelling variant, currency notation, spacing or punctuation."""
    return bool(corrected.strip()) and _canon(original) == _canon(corrected)


def is_pure_insertion(original: str, corrected: str) -> bool:
    """True if `corrected` merely adds words around an unchanged `original` (e.g. name expansion)."""
    o, c = words_of(original), words_of(corrected)
    if not o or len(c) <= len(o):
        return False
    it = iter(c)
    return all(any(w == x for x in it) for w in o)


def words_of(s: str) -> List[str]:
    return re.findall(r"[0-9a-z₹$%]+", s.lower())


def _norm_for_quote(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(re.findall(r"[0-9a-z₹$%]+", s))


def quote_in_source(quote: str, source: str, min_cover: float = 0.7) -> bool:
    """Check that the evidence quote really comes from the SOURCE (tolerant to formatting/ellipses)."""
    q, s = _norm_for_quote(quote or "").split(), " " + _norm_for_quote(source or "") + " "
    if not q:
        return False
    if len(q) < 3:
        return (" " + " ".join(q) + " ") in s
    grams = [" ".join(q[i:i + 3]) for i in range(len(q) - 2)]
    hit = sum(1 for g in grams if (" " + g + " ") in s)
    return hit / len(grams) >= min_cover


# --------------------------------------------------------------------------
# Make a correction follow the article's own conventions
# --------------------------------------------------------------------------

_AN_EXCEPT_A = re.compile(r"^(one|once|uni|use|usu|euro|eu|ur[aeiou])", re.I)
_AN_EXCEPT_AN = re.compile(r"^(hour|honest|honou?r|heir)", re.I)


def _needs_an(word: str) -> bool:
    if not word:
        return False
    if word.isupper() and len(word) > 1:  # acronym: go by letter name
        return word[0] in "AEFHILMNORSX"
    if _AN_EXCEPT_AN.match(word):
        return True
    if _AN_EXCEPT_A.match(word):
        return False
    if word[0].isdigit():
        return word.startswith(("8", "11", "18"))
    return word[0].lower() in "aeiou"


def conform_style(original: str, corrected: str, article: str) -> str:
    """Rewrite source-style notation in `corrected` into the article's notation."""
    c = corrected
    if "₹" in article or "₹" in original:
        c = re.sub(r"\b(?:Rs\.?|INR)\s?(?=\d)", "₹", c)
    # explanatory parentheticals copied from a hint ("(after its theatrical run)") are not article prose
    def _strip_paren(m):
        inner = m.group(1)
        # keep it if the edited text already had a bracket (e.g. a quote's translation being corrected)
        if "(" in original or len(inner.split()) < 3:
            return m.group(0)  # keep pre-existing text and short items such as acronyms "(FWICE)"
        return ""
    c = re.sub(r"\s*\(([^()]*)\)", _strip_paren, c)
    # number-unit spacing: article writes 1,000cc / 48MP -> do not introduce "1,200 cc"
    if re.search(r"\d[A-Za-z]", original) and not re.search(r"\d [A-Za-z]", original):
        c = re.sub(r"(\d) ([A-Za-z]{1,4})\b", r"\1\2", c)
    return c


def fix_articles(text: str) -> str:
    """Correct a/an agreement before words that an edit may have changed."""

    def rep(m):
        art, space, word = m.group(1), m.group(2), m.group(3)
        want = "an" if _needs_an(word) else "a"
        if art.lower() == want:
            return m.group(0)
        if art[0].isupper():
            want = want.capitalize()
        return want + space + word

    return re.sub(r"\b([Aa]n?)(\s+)([A-Za-z0-9][\w-]*)", rep, text)


# --------------------------------------------------------------------------
# Minimal-edit enforcement
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+(?:[.,'’]\w+)*|\s+|[^\w\s]", re.UNICODE)


def minimize_edit(original: str, corrected: str) -> str:
    """
    Undo the parts of an edit that change only punctuation, capitalisation or
    notation (e.g. ". When" -> " — when"), keeping the content changes.
    Returns the corrected text with those cosmetic sub-changes reverted.
    """
    if not corrected.strip() or not original.strip():
        return corrected
    import difflib

    a, b = _TOKEN_RE.findall(original), _TOKEN_RE.findall(corrected)
    if "".join(a) != original or "".join(b) != corrected:
        return corrected  # tokenisation not lossless; leave untouched
    # Group neighbouring changes (separated only by whitespace) into regions.
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    regions, cur = [], None
    for op, i1, i2, j1, j2 in ops:
        if op == "equal" and not ("".join(a[i1:i2]).isspace() and cur):
            if cur:
                regions.append(cur)
                cur = None
            regions.append(("equal", i1, i2, j1, j2))
        elif cur:
            cur = ("change", cur[1], i2, cur[3], j2)
        else:
            cur = ("change", i1, i2, j1, j2)
    if cur:
        regions.append(cur)
    out = []
    for op, i1, i2, j1, j2 in regions:
        old, new = "".join(a[i1:i2]), "".join(b[j1:j2])
        # Revert a cosmetic-only change (". When" -> " — when", added trailing "."), but never
        # a pure removal of punctuation: that usually belongs to a clause the edit deleted.
        if op != "equal" and new.strip() and _canon(old) == _canon(new):
            out.append(old)
        else:
            out.append(new)
    result = "".join(out)
    # Safety: never let reverting create a broken result (doubled spaces, etc.)
    return result if _canon(result) == _canon(corrected) else corrected


def is_morphological_variant(original: str, corrected: str) -> bool:
    """
    True if the edit only swaps a word for a grammatical variant of itself
    (plural/singular, verb tense): "promise" <-> "promises", "block" <-> "blocked".
    These are never injected facts, so in no-hint mode they are over-edits.
    """
    ao, bo = words_of(original), words_of(corrected)
    # Exactly one differing word, same count.
    if len(ao) != len(bo):
        return False
    diff = [(a, b) for a, b in zip(ao, bo) if a != b]
    if len(diff) != 1:
        return False
    a, b = diff[0]
    if a[0].isdigit() or b[0].isdigit():
        return False  # numbers are hard facts, never "variants"
    short, long = sorted((a, b), key=len)
    # share a 4+ char stem and differ only by a short grammatical suffix
    return len(short) >= 4 and long.startswith(short[:4]) and long[:len(short)] == short and len(long) - len(short) <= 3


# --------------------------------------------------------------------------
# Candidate recall booster (ported/adapted from the two-tier design):
# flag numbers / proper nouns in the article that are ABSENT from the source,
# so the LLM can verify spans the annotation list never flagged. Purely
# lexical: it only focuses attention, it never edits text itself.
# --------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"(?:[₹$€£]|Rs\.?\s?)?\d[\d,]*(?:\.\d+)?"
    r"(?:%|\s?(?:nits|kg|km|cm|mm|cc|mp|billion|million|lakh|crore))?"
)
_CAPPED_RE = re.compile(r"\b(?:[A-Z][a-zA-Z&'-]+(?:[^\S\n]+[A-Z][a-zA-Z&'-]+)*)\b")
_MONTHS_DAYS = {
    "india", "the", "and", "apple", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "november", "december",
}


def _cnorm(text: str) -> str:
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), (",", ""), ("₹", ""), ("$", "")):
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip().lower()


def _similar_numbers(token: str, norm_source: str, limit: int = 3):
    digits = re.sub(r"\D", "", token)
    if not digits:
        return []
    hits = []
    for m in re.finditer(r"\d+", norm_source):
        cand = m.group(0)
        if cand == digits or cand in hits or abs(len(cand) - len(digits)) > 1:
            continue
        if len(cand) == len(digits):
            if sum(x != y for x, y in zip(cand, digits)) <= 1:
                hits.append(cand)
        else:
            short, lng = sorted((digits, cand), key=len)
            i = j = diffs = 0
            while i < len(short) and j < len(lng):
                if short[i] == lng[j]:
                    i += 1; j += 1
                else:
                    diffs += 1; j += 1
                if diffs > 1:
                    break
            else:
                hits.append(cand)
        if len(hits) >= limit:
            break
    return hits


def extract_candidates(body, source, limit=30, covered=None, high_confidence=False):
    """
    Spans in the article whose text is absent from the source (suspect errors).
    high_confidence=True keeps only the strongest signals: numbers that have a
    near-identical number in the source (a digit was almost certainly swapped),
    which keeps the recall pass precise. Proper-noun candidates (noisier) are
    included only in the normal mode.
    """
    if not source:
        return []
    nsrc = _cnorm(source)
    raw_src_lower = re.sub(r"\s+", " ", source).replace("’", "'").replace("“", '"').replace("”", '"').lower()
    cov = [_cnorm(c) for c in (covered or []) if c and c.strip()]
    seen, out = set(), []
    for m in _NUMBER_RE.finditer(body):
        tok = m.group(0).strip()
        # Expand to the whole token so a trailing unit the regex missed ("145g")
        # is included — otherwise the bare number can't be word-boundary matched.
        s, e = m.start(), m.end()
        while e < len(body) and (body[e].isalpha()):
            e += 1
        tok = body[s:e].strip()
        k = tok.lower()
        if k in seen or len(tok) < 2:
            continue
        seen.add(k)
        nt = _cnorm(tok)
        if nt in nsrc or any(nt in c for c in cov):
            continue
        if tok.endswith("%") and re.search(r"\b" + re.escape(_cnorm(tok.rstrip("%"))) + r"\b", nsrc):
            continue
        sim = _similar_numbers(tok, nsrc)
        if high_confidence and not sim:
            continue  # only numbers with a near-match in the source
        cnt = len(re.findall(re.escape(tok), body))
        label = f"{tok!r} (x{cnt})" if cnt > 1 else repr(tok)
        if sim:
            label += f" — source has similar: {', '.join(sim)}"
        out.append(label)
        if len(out) >= limit:
            return out
    if high_confidence:
        return out[:limit]
    for m in _CAPPED_RE.finditer(body):
        tok = m.group(0)
        if len(tok) < 4 or tok.lower() in seen or re.fullmatch(r"[A-Z]{2,}", tok):
            continue
        if len(tok.split()) == 1 and tok.lower() in _MONTHS_DAYS:
            continue
        if any(_cnorm(tok) in c for c in cov):
            continue
        seen.add(tok.lower())
        tn = re.sub(r"\s+", " ", tok).replace("’", "'").replace("“", '"').replace("”", '"').lower()
        if tn not in raw_src_lower:
            out.append(repr(tok))
        if len(out) >= limit:
            break
    return out[:limit]


def sweep_value_consistency(text: str, applied_edits: List[dict]) -> str:
    """
    After an edit changes a single value token (a number, or a Capitalised name
    token), update any OTHER standalone occurrences of the old value to the new
    one, so the article never ends up internally inconsistent (e.g. "145g" fixed
    in one place but left in another). Only fires for an unambiguous one-token
    change; never touches substrings inside larger words.
    """
    for e in applied_edits:
        o, c = e.get("original", ""), e.get("corrected", "")
        ow, cw = o.split(), c.split()
        # find the single differing token between original and corrected
        diff = [(x, y) for x, y in zip(ow, cw) if x != y]
        if len(ow) != len(cw) or len(diff) != 1:
            continue
        old_tok, new_tok = diff[0]
        old_core = old_tok.strip(".,;:!?()\"'")
        new_core = new_tok.strip(".,;:!?()\"'")
        if not old_core or old_core == new_core:
            continue
        # Numbers only: a single name token can be a shared surname (swapping it
        # everywhere is unsafe); numeric consistency is the real, safe win.
        has_digit = any(ch.isdigit() for ch in old_core) and any(ch.isdigit() for ch in new_core)
        if not has_digit:
            continue
        # word-boundary, whole-token replacement of any remaining old_core
        pat = r"(?<![\w.,])" + re.escape(old_core) + r"(?![\w])"
        text = re.sub(pat, new_core, text)
    return text


def recall_edit_source_backed(corrected: str, source: str) -> bool:
    """
    For the high-confidence recall pass: accept a numeric fix only when the
    corrected number actually occurs in the source. This rejects the LLM's
    guesses (e.g. "19"->"one", "5am"->"5:20am") while keeping real digit
    corrections ("145g"->"165g", "$240M"->"$24 million").
    """
    cd = re.findall(r"\d+", corrected)
    if not cd:
        return False  # a number candidate whose "fix" has no digits is a hallucination
    sd = set(re.findall(r"\d+", source.replace(",", "")))
    return all(d in sd for d in cd)
