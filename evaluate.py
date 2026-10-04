"""
Offline evaluation against the 10 human-rectified references (eval/gold/).

For each article we compare three texts at word level:
  AI   = AI-generated body (annotation block removed)
  OURS = our rectified output
  GOLD = human reference

Metrics
  similarity   : difflib ratio(OURS, GOLD)              (1.0 = identical)
  fixed        : gold edit regions (AI->GOLD) that OURS reproduces exactly
  touched      : gold edit regions OURS changed at all (exactly or not)
  spurious     : regions OURS changed that GOLD left untouched (over-editing)
  residual     : word-level distance OURS->GOLD vs AI->GOLD (lower is better)

Usage:
  python evaluate.py                 # evaluate rectified_articles/
  python evaluate.py --dir some/dir  # evaluate another output folder
  python evaluate.py --fallback      # evaluate the no-LLM hint fallback
  python evaluate.py --verbose       # print every missed / spurious edit
"""

import argparse
import difflib
import os
import re

from article_utils import parse_annotations, split_article

GOLD_DIR = "eval/gold"


def words(t):
    return re.findall(r"\S+", t)


def ranges(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [(i1, i2, j1, j2) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]


def dist(a, b):
    return sum(max(i2 - i1, j2 - j1) for i1, i2, j1, j2 in ranges(a, b))


def overlaps(r, rs):
    i1, i2 = r
    return any((i1 < s2 and s1 < i2) or (i1 == i2 and s1 <= i1 <= s2) or (s1 == s2 and i1 <= s1 <= i2) for s1, s2 in rs)


def evaluate_one(ai, ours, gold, verbose=False, name=""):
    A, O, G = words(ai), words(ours), words(gold)
    gold_edits = ranges(A, G)
    our_edits = ranges(A, O)
    our_spans = [(i1, i2) for i1, i2, _, _ in our_edits]
    gold_spans = [(i1, i2) for i1, i2, _, _ in gold_edits]

    # Map AI token positions to OURS so we can read our text over a gold edit region.
    sm = difflib.SequenceMatcher(None, A, O, autojunk=False)
    fixed = touched = 0
    for i1, i2, j1, j2 in gold_edits:
        want = " ".join(G[j1:j2])
        if overlaps((i1, i2), our_spans):
            touched += 1
        # our text for the same AI window (with 2 words of context either side)
        ctx_gold = " ".join(G[max(0, j1 - 2): j2 + 2])
        if ctx_gold in " ".join(O):
            fixed += 1
        elif verbose:
            print(f"   MISSED {name}: AI={' '.join(A[i1:i2])!r} -> GOLD={want!r}")
    spurious = 0
    for i1, i2, j1, j2 in our_edits:
        if not overlaps((i1, i2), gold_spans):
            spurious += 1
            if verbose:
                print(f"   SPURIOUS {name}: AI={' '.join(A[i1:i2])!r} -> OURS={' '.join(O[j1:j2])!r}")
    sim = difflib.SequenceMatcher(None, O, G, autojunk=False).ratio()
    return dict(n_gold=len(gold_edits), fixed=fixed, touched=touched, spurious=spurious,
                sim=sim, base_sim=difflib.SequenceMatcher(None, A, G, autojunk=False).ratio(),
                residual=dist(O, G), base_residual=dist(A, G))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="rectified_articles")
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    tot = dict(n_gold=0, fixed=0, touched=0, spurious=0, residual=0, base_residual=0)
    sims, base_sims = [], []
    print(f"{'article':<13}{'gold':>5}{'fixed':>6}{'touch':>6}{'spur':>5}{'sim':>8}{'base':>8}{'resid':>7}{'base':>6}")
    for fname in sorted(os.listdir(GOLD_DIR)):
        aid = fname[:-4]
        ai, block = split_article(open(f"ai_generated_articles/{fname}", encoding="utf-8").read())
        gold = open(f"{GOLD_DIR}/{fname}", encoding="utf-8").read()
        if args.fallback:
            from rectification_system import fallback_from_hints
            ours = fallback_from_hints(ai, parse_annotations(block))
        else:
            path = os.path.join(args.dir, fname)
            if not os.path.exists(path):
                print(f"{aid:<13} (missing output)")
                continue
            ours = open(path, encoding="utf-8").read()
        r = evaluate_one(ai, ours, gold, args.verbose, aid)
        for k in tot:
            tot[k] += r[k]
        sims.append(r["sim"]); base_sims.append(r["base_sim"])
        print(f"{aid:<13}{r['n_gold']:>5}{r['fixed']:>6}{r['touched']:>6}{r['spurious']:>5}"
              f"{r['sim']:>8.4f}{r['base_sim']:>8.4f}{r['residual']:>7}{r['base_residual']:>6}")
    if not sims:
        return
    print("-" * 64)
    print(f"{'TOTAL':<13}{tot['n_gold']:>5}{tot['fixed']:>6}{tot['touched']:>6}{tot['spurious']:>5}"
          f"{sum(sims)/len(sims):>8.4f}{sum(base_sims)/len(base_sims):>8.4f}{tot['residual']:>7}{tot['base_residual']:>6}")
    print(f"\nexact-fix rate  {tot['fixed']/max(1,tot['n_gold']):.1%}   "
          f"touch rate {tot['touched']/max(1,tot['n_gold']):.1%}   spurious edits {tot['spurious']}   "
          f"residual reduction {1 - tot['residual']/max(1,tot['base_residual']):.1%}")


if __name__ == "__main__":
    main()
