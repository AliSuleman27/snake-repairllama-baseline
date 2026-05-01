#!/usr/bin/env python3
"""
select_plausibility_subset.py
-----------------------------
Pick a 50-bug subset of BugsInPy worth running plausibility on.

Selection policy (in priority order):
  Tier 1 — bugs where any of top-10 generations was BURIED and at least one
           generation COMPILES. These are the bugs where the model both
           "knew" the fix (gold appears in some generation) AND produced at
           least one syntactically valid patch. Highest signal — take all.
  Tier 2 — bugs where at least one of top-10 COMPILES but none were buried.
           Sample stratified by project (proportional, with min 1 per project
           where possible) up to the target.
  Tier 3 — bugs with 0/10 compile. Skip — guaranteed plausibility failure,
           wastes hours.

Usage:
    python scripts/select_plausibility_subset.py \
        --eval data/bugsinpy_eval_verified.jsonl \
        --inference results/bugsinpy_codellama_baseline.jsonl \
        --target 50 \
        --out scripts/plausibility_subset.json

Prints a per-project breakdown and writes the picked bug_ids + indices to JSON.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import score_patch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--inference", required=True)
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # Load eval (preserve file order, since bug indices = position in file)
    eval_records = []
    with open(args.eval, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_records.append(json.loads(line))
    eval_index = {r["bug_id"]: r for r in eval_records}
    bug_id_to_idx = {r["bug_id"]: i for i, r in enumerate(eval_records)}
    print(f"[select] Loaded {len(eval_records)} eval bugs")

    # Score every generation
    scored = {}  # bug_id -> {"any_compile": bool, "any_buried": bool, "project": str}
    with open(args.inference, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            bug_id = rec["bug_id"]
            if bug_id not in eval_index:
                continue
            gold = rec["gold_output"]
            buggy_function = eval_index[bug_id].get("buggy_function", "")
            any_compile = False
            any_buried = False
            for g in rec["generations"]:
                s = score_patch(g, gold, buggy_function)
                any_compile |= s["compile"]
                any_buried |= s["buried"]
                if any_compile and any_buried:
                    break  # short-circuit
            scored[bug_id] = {
                "any_compile": any_compile,
                "any_buried": any_buried,
                "project": eval_index[bug_id]["project"],
                "idx": bug_id_to_idx[bug_id],
            }

    print(f"[select] Scored {len(scored)} bugs")

    # Categorize into tiers
    tier1 = [b for b, s in scored.items() if s["any_buried"] and s["any_compile"]]
    tier2 = [b for b, s in scored.items() if s["any_compile"] and not s["any_buried"]]
    tier3 = [b for b, s in scored.items() if not s["any_compile"]]
    print(f"[select] Tier 1 (compile + buried): {len(tier1)}")
    print(f"[select] Tier 2 (compile only):     {len(tier2)}")
    print(f"[select] Tier 3 (no compile, skip): {len(tier3)}")

    # Take all of Tier 1
    picked = list(tier1)
    target = args.target
    remaining = target - len(picked)
    if remaining < 0:
        # Tier 1 alone exceeds target — keep all anyway, can't go below it without losing the most signal
        print(f"[select] Tier 1 size ({len(tier1)}) exceeds target ({target}); keeping all of Tier 1")
        remaining = 0

    # Stratified sample from Tier 2 by project
    if remaining > 0:
        by_project = defaultdict(list)
        for b in tier2:
            by_project[scored[b]["project"]].append(b)

        # Ignore projects already covered by Tier 1 if we want max project diversity?
        # No — projects with Tier 1 bugs may also have many Tier 2 bugs (e.g. pandas).
        # Proportional allocation: project_quota = round(remaining * len(project_t2) / total_t2),
        # with min(1) for each project that has any t2 bugs (until budget is hit).
        total_t2 = sum(len(v) for v in by_project.values())
        quotas = {}
        for p, bugs in by_project.items():
            q = max(1, round(remaining * len(bugs) / total_t2))
            quotas[p] = min(q, len(bugs))
        # Trim or expand to hit exactly `remaining`
        while sum(quotas.values()) > remaining:
            # Trim from the largest quota
            p_max = max(quotas, key=lambda p: quotas[p])
            quotas[p_max] -= 1
        while sum(quotas.values()) < remaining:
            # Expand: give one more to project with highest unmet ratio
            candidates = [
                (p, len(by_project[p]) - quotas[p])
                for p in quotas
                if quotas[p] < len(by_project[p])
            ]
            if not candidates:
                break
            p_best = max(candidates, key=lambda x: x[1])[0]
            quotas[p_best] += 1

        for p, q in quotas.items():
            chosen = random.sample(by_project[p], q)
            picked.extend(chosen)

    # Sort picked by file order so the runner processes them in eval-file sequence
    picked.sort(key=lambda b: scored[b]["idx"])

    # Per-project summary
    summary = defaultdict(lambda: {"tier1": 0, "tier2": 0})
    for b in picked:
        proj = scored[b]["project"]
        if scored[b]["any_buried"]:
            summary[proj]["tier1"] += 1
        else:
            summary[proj]["tier2"] += 1

    print()
    print(f"=== Picked {len(picked)} bugs ===")
    print(f"{'project':20s} {'T1(buried)':>11s} {'T2(compile)':>12s} {'total':>6s}")
    print("-" * 55)
    grand_t1, grand_t2 = 0, 0
    for proj in sorted(summary):
        s = summary[proj]
        print(f"{proj:20s} {s['tier1']:>11d} {s['tier2']:>12d} {s['tier1']+s['tier2']:>6d}")
        grand_t1 += s['tier1']
        grand_t2 += s['tier2']
    print("-" * 55)
    print(f"{'TOTAL':20s} {grand_t1:>11d} {grand_t2:>12d} {grand_t1+grand_t2:>6d}")

    # Write summary JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "target": args.target,
        "selected": [
            {"bug_id": b, "project": scored[b]["project"], "idx": scored[b]["idx"],
             "tier": 1 if scored[b]["any_buried"] else 2}
            for b in picked
        ],
        "indices": [scored[b]["idx"] for b in picked],
        "tier1_count": grand_t1,
        "tier2_count": grand_t2,
        "skipped_tier3_count": len(tier3),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print()
    print(f"[select] Wrote summary: {out_path}")

    # Also write a filtered eval JSONL so notebook 03 can use a simple
    # contiguous start_bug=0, end_bug=N slice without any runner changes.
    picked_ids = set(picked)
    filtered_eval = out_path.parent.parent / "data" / (
        Path(args.eval).stem + "_subset50.jsonl"
    )
    with open(filtered_eval, "w", encoding="utf-8") as f:
        for r in eval_records:
            if r["bug_id"] in picked_ids:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[select] Wrote filtered eval: {filtered_eval}")
    print(f"[select] In notebook 03, set EVAL_FILE = '{filtered_eval.as_posix().split('/', 1)[-1] if '/' in filtered_eval.as_posix() else filtered_eval.name}'")
    print(f"[select] Then START_BUG=0, END_BUG={len(picked)} runs the whole subset.")


if __name__ == "__main__":
    main()
