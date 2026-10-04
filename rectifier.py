import json
import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from rectification_system import run
from article_utils import clean_output, split_article
import llm_client
from llm_client import env_num

# Resolve all relative paths (article_mapping.json, data folders, .env) from this file's folder,
# so the command works no matter which directory it is launched from.
os.chdir(Path(__file__).resolve().parent)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
WORKERS = env_num("RECTIFIER_WORKERS", 4, int)

def get_article_mapping(article_id: str):
    # Load article mapping to get file paths
    with open('article_mapping.json', 'r') as f:
        articles = json.load(f)
    
    # Find the article by ID
    article_data = next((a for a in articles if a['article_id'] == article_id), None)
    if not article_data:
        raise ValueError(f"Article {article_id} not found in mapping")
    
    return article_data

def get_ai_generated_article(article_id: str):
    # Read the AI-generated article
    _mapping = get_article_mapping(article_id)
    fpath = _mapping['ai_generated_file']
    with open(fpath, 'r', encoding='utf-8') as f:
        article = f.read()
    return article

def get_source_article(article_id: str):
    fpath = get_article_mapping(article_id)['source_file']
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            return f.read()
    except OSError:
        return ""

def save_rectified_article(article_id: str, rectified_content: str):
    mapping = get_article_mapping(article_id)
    fpath = mapping['rectified_file']
    
    # Ensure output directory exists
    output_path = Path(fpath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(rectified_content)

def rectify_article(article_id: str):
    """
    Rectify an AI-generated article.
    
    Args:
        article_id: ID of the article (e.g., 'article_001')
    
    Returns:
        str: The rectified article content
    """
    
    ai_generated_content = get_ai_generated_article(article_id)
    source_content = get_source_article(article_id)
    
    # PLUG YOUR CUSTOM RECTIFIER HERE
    try:
        rectified_content = run(ai_generated_content, source_content, article_id)
    except Exception as e:
        # Never leave a file missing: fall back to the article minus its annotation block.
        print(f"✗ {article_id}: rectifier error ({e}); saving cleaned original")
        rectified_content = clean_output(split_article(ai_generated_content)[0])
    ###################################
    
    save_rectified_article(article_id, rectified_content)
    
    print(f"✓ Rectified {article_id}")
    return rectified_content


def _process(articles):
    """Rectify the given mapping entries in parallel; every article gets a file."""
    # Refuse to run degraded: no key AND no cache would silently emit low-quality output.
    err = llm_client.preflight()
    if err:
        print("ERROR: " + err)
        sys.exit(1)
    total = len(articles)
    done = 0
    rb = llm_client.remote_budget()
    if rb and rb[1] is not None:
        print(f"Budget: ${rb[0]:.4f} spent of ${float(rb[1]):.2f} "
              f"(run cap ${llm_client.MAX_RUN_USD:.2f}, reserve ${llm_client.MIN_REMAINING_USD:.2f})")
    with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
        futures = {pool.submit(rectify_article, a['article_id']): a['article_id'] for a in articles}
        for fut in as_completed(futures):
            article_id = futures[fut]
            done += 1
            try:
                fut.result()
            except Exception as e:
                print(f"✗ Error processing {article_id}: {str(e)}")
                try:
                    raw = get_ai_generated_article(article_id)
                    save_rectified_article(article_id, clean_output(split_article(raw)[0]))
                except Exception as e2:
                    print(f"✗ Could not write fallback for {article_id}: {e2}")
            print(f"  progress {done}/{total}")
    u = llm_client.usage_totals
    print(f"\nLLM usage: {u['calls']} calls ({u['cached']} cache hits), "
          f"{u['prompt_tokens']:,} prompt + {u['completion_tokens']:,} completion tokens, "
          f"est. ${u['est_usd']:.4f}")
    if llm_client._budget["stopped"]:
        print(f"⚠ BUDGET GUARD STOPPED LLM CALLS: {llm_client._budget['stopped']}")
        print("  Articles after that point used the no-LLM hint fallback. Re-run later; cached results are reused for free.")
    # Loud warning if the whole batch degraded (e.g. an invalid key with no cache):
    # no LLM calls succeeded and nothing was served from cache.
    if u['calls'] == 0 and u['cached'] == 0 and total > 0:
        print("\n" + "=" * 60)
        print("⚠ WARNING: output was produced by the DEGRADED hint-only fallback")
        print("  (no successful LLM calls and no cache hits — likely a bad LLM_API_KEY).")
        print("  This is NOT the full-quality result. Fix the key or restore .llm_cache/ and re-run.")
        print("=" * 60)


def test_rectifier(count: int):
    """
    Test the rectification system on a subset of articles.
    
    Args:
        count: Number of articles to test (default: 16)
    """
    with open('article_mapping.json', 'r') as f:
        articles = json.load(f)
    _process(articles[:count])


def rectify_all():
    """
    Generate rectified articles for all 104 articles.
    """
    with open('article_mapping.json', 'r') as f:
        articles = json.load(f)
    
    total = len(articles)
    _process(articles)
    
    print(f"\n{'='*50}")
    print(f"Completed! Processed {total} articles.")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rectify AI-generated articles by fixing errors and inaccuracies."
    )
    parser.add_argument(
        'command',
        choices=['test', 'rectify-all'],
        help='Command to execute: "test" to process first 16 articles, "rectify-all" to process all 104 articles'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=16,
        help='Number of articles to test (only applicable for "test" command, default: 16)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'test':
        print(f"Testing rectification system on first {args.count} articles...")
        test_rectifier(count=args.count)
    elif args.command == 'rectify-all':
        print("Processing all 104 articles...")
        rectify_all()