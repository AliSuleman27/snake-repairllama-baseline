#!/usr/bin/env python3
"""
metrics.py
----------
Score generated patches against ground truth.

Metrics (per bug, then aggregated):
  - exact_match : normalize_for_match(pred) == normalize_for_match(gold)
  - ast_match   : ast.dump(pred) == ast.dump(gold)  [parsed inside the
                  buggy function so indentation works]
  - compile     : the function with `pred` substituted for <FILL_ME>
                  parses without SyntaxError
  - buried_fix  : gold appears anywhere inside the lenient generation
                  (used to detect "model knows it but can't isolate it")

Aggregated reports:
  - Top-1 / Top-3 / Top-10 exact / ast / compile
  - Top-10 buried_fix
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .postprocess import extract_patch, normalize_for_match


# ─────────────────────────────── per-patch scoring ───────────────────────────


def _normalize_ast(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
        return ast.dump(tree, annotate_fields=False, include_attributes=False)
    except SyntaxError:
        return None
    except Exception:
        return None


def _splice_into_function(buggy_function: str, patch: str) -> str:
    """
    Replace the '# Buggy code:' block + <FILL_ME> region in buggy_function with
    `patch`. Used to check that the spliced function parses.

    We don't have the IR4 here directly — but we have buggy_function, and we
    know the patch is meant to replace some contiguous slice. For a syntax
    check we do a simple thing: replace the entire buggy function body with
    patch lines, indented to match the function. This is a coarse but
    sufficient compile check.
    """
    # Coarsest-but-safe: just attempt to parse the patch on its own, wrapped
    # inside a stub function to give it scope. This catches syntax errors that
    # are independent of context.
    indent = "    "
    wrapped = "def _stub():\n" + "".join(
        indent + ln if ln.strip() else ln
        for ln in patch.splitlines(keepends=True)
    )
    if not wrapped.endswith("\n"):
        wrapped += "\n"
    # If patch is empty, the stub still needs a body
    body_present = any(ln.strip() for ln in patch.splitlines())
    if not body_present:
        wrapped = "def _stub():\n    pass\n"
    try:
        ast.parse(wrapped)
        return "OK"
    except SyntaxError:
        return "FAIL"


def score_patch(
    raw_generation: str,
    gold: str,
    buggy_function: str = "",
) -> Dict:
    """Score one generated patch."""
    pred_strict = extract_patch(raw_generation, mode="strict")
    pred_lenient = extract_patch(raw_generation, mode="lenient")

    pred_norm = normalize_for_match(pred_strict)
    gold_norm = normalize_for_match(gold)

    exact = pred_norm == gold_norm

    pred_ast = _normalize_ast(pred_strict)
    gold_ast = _normalize_ast(gold)
    ast_match = (
        pred_ast is not None and gold_ast is not None and pred_ast == gold_ast
    )

    compile_ok = _splice_into_function(buggy_function, pred_strict) == "OK"

    # "Buried" = gold (normalized) appears as a substring in lenient generation
    buried = gold_norm in normalize_for_match(pred_lenient)

    return {
        "exact":   bool(exact),
        "ast":     bool(ast_match),
        "compile": bool(compile_ok),
        "buried":  bool(buried),
        "pred_strict": pred_strict,
    }


# ─────────────────────────────── aggregation ─────────────────────────────────


def evaluate_file(
    inference_jsonl: str,
    eval_jsonl: str,
    plausibility_jsonl: Optional[str] = None,
) -> Dict:
    """
    Score an inference file. Returns a dict of metric counts and rates.

    inference_jsonl  : produced by inference.run_inference (1 row per bug)
    eval_jsonl       : original eval set (for buggy_function lookup)
    plausibility_jsonl: optional path produced by runners.{quixbugs,bugsinpy}
                       (1 row per (bug, gen_idx) with test_pass field).
                       If provided, plausible@K metrics are added.
    """
    # Load eval index by bug_id
    eval_index: Dict[str, Dict] = {}
    with open(eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            eval_index[r["bug_id"]] = r

    # Load plausibility results if provided. Map (bug_id, gen_idx) -> test_pass.
    plaus_index: Dict[tuple, bool] = {}
    if plausibility_jsonl:
        with open(plausibility_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                p = json.loads(line)
                plaus_index[(p["bug_id"], p["gen_idx"])] = bool(p.get("test_pass", False))

    # Initialize counters
    n = 0
    counts = {
        "top1_exact": 0, "top1_ast": 0, "top1_compile": 0, "top1_plausible": 0,
        "top3_exact": 0, "top3_ast": 0, "top3_compile": 0, "top3_buried": 0, "top3_plausible": 0,
        "top10_exact": 0, "top10_ast": 0, "top10_compile": 0, "top10_buried": 0, "top10_plausible": 0,
    }
    per_bug_records: List[Dict] = []
    n_with_plaus = 0  # bugs that have at least one plausibility row

    with open(inference_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            bug_id = rec["bug_id"]
            gold = rec["gold_output"]
            buggy_function = eval_index.get(bug_id, {}).get("buggy_function", "")
            gens = rec["generations"]

            scores = [
                score_patch(g, gold, buggy_function) for g in gens
            ]

            # Attach plausibility flag (None if unknown)
            plaus_flags: List[Optional[bool]] = []
            for gi in range(len(gens)):
                key = (bug_id, gi)
                plaus_flags.append(plaus_index.get(key))

            has_any_plaus = any(p is not None for p in plaus_flags)
            if has_any_plaus:
                n_with_plaus += 1

            # Top-K logic
            def any_in(k: int, key: str) -> bool:
                return any(s[key] for s in scores[:k])

            def any_plausible(k: int) -> bool:
                return any(p is True for p in plaus_flags[:k])

            top1 = scores[0] if scores else {"exact": False, "ast": False, "compile": False, "buried": False}

            if top1["exact"]:    counts["top1_exact"] += 1
            if top1["ast"]:      counts["top1_ast"] += 1
            if top1["compile"]:  counts["top1_compile"] += 1
            if has_any_plaus and plaus_flags[0] is True:
                counts["top1_plausible"] += 1

            if any_in(3, "exact"):    counts["top3_exact"] += 1
            if any_in(3, "ast"):      counts["top3_ast"] += 1
            if any_in(3, "compile"):  counts["top3_compile"] += 1
            if any_in(3, "buried"):   counts["top3_buried"] += 1
            if has_any_plaus and any_plausible(3):
                counts["top3_plausible"] += 1

            if any_in(10, "exact"):    counts["top10_exact"] += 1
            if any_in(10, "ast"):      counts["top10_ast"] += 1
            if any_in(10, "compile"):  counts["top10_compile"] += 1
            if any_in(10, "buried"):   counts["top10_buried"] += 1
            if has_any_plaus and any_plausible(10):
                counts["top10_plausible"] += 1

            per_bug_records.append({
                "bug_id": bug_id,
                "scores": scores,
                "plausibility": plaus_flags,
            })
            n += 1

    rates = {k: (v / n if n else 0.0) for k, v in counts.items()}
    # plausible rates use n_with_plaus as denominator (only bugs we tested)
    if n_with_plaus:
        for k in ("top1_plausible", "top3_plausible", "top10_plausible"):
            rates[k + "_of_tested"] = counts[k] / n_with_plaus

    return {
        "n":     n,
        "n_with_plausibility": n_with_plaus,
        "counts": counts,
        "rates":  rates,
        "per_bug": per_bug_records,
    }


def print_report(name: str, result: Dict):
    n = result["n"]
    c = result["counts"]
    r = result["rates"]
    n_p = result.get("n_with_plausibility", 0)
    print()
    print("=" * 64)
    print(f"  {name}")
    print(f"  Total bugs: {n}    Bugs with plausibility data: {n_p}")
    print("=" * 64)
    print(f"  Top-1  Exact     : {c['top1_exact']:>4} / {n} ({r['top1_exact']*100:5.1f}%)")
    print(f"  Top-1  AST       : {c['top1_ast']:>4} / {n} ({r['top1_ast']*100:5.1f}%)")
    print(f"  Top-1  Compile   : {c['top1_compile']:>4} / {n} ({r['top1_compile']*100:5.1f}%)")
    if n_p:
        denom = n_p
        print(f"  Top-1  Plausible : {c['top1_plausible']:>4} / {denom} ({r.get('top1_plausible_of_tested',0)*100:5.1f}%) [tests passed]")
    print()
    print(f"  Top-3  Exact     : {c['top3_exact']:>4} / {n} ({r['top3_exact']*100:5.1f}%)")
    print(f"  Top-3  AST       : {c['top3_ast']:>4} / {n} ({r['top3_ast']*100:5.1f}%)")
    print(f"  Top-3  Compile   : {c['top3_compile']:>4} / {n} ({r['top3_compile']*100:5.1f}%)")
    print(f"  Top-3  Buried    : {c['top3_buried']:>4} / {n} ({r['top3_buried']*100:5.1f}%)")
    if n_p:
        print(f"  Top-3  Plausible : {c['top3_plausible']:>4} / {n_p} ({r.get('top3_plausible_of_tested',0)*100:5.1f}%)")
    print()
    print(f"  Top-10 Exact     : {c['top10_exact']:>4} / {n} ({r['top10_exact']*100:5.1f}%)")
    print(f"  Top-10 AST       : {c['top10_ast']:>4} / {n} ({r['top10_ast']*100:5.1f}%)")
    print(f"  Top-10 Compile   : {c['top10_compile']:>4} / {n} ({r['top10_compile']*100:5.1f}%)")
    print(f"  Top-10 Buried    : {c['top10_buried']:>4} / {n} ({r['top10_buried']*100:5.1f}%)")
    if n_p:
        print(f"  Top-10 Plausible : {c['top10_plausible']:>4} / {n_p} ({r.get('top10_plausible_of_tested',0)*100:5.1f}%)")
    print("=" * 64)
