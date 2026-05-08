#!/usr/bin/env python3
"""
prefilter_for_plausibility.py
-----------------------------
Optimization wrapper for BugsInPy plausibility testing.

For a given inference JSONL (e.g. results/bugsinpy_snakellama_run3.jsonl),
this script analyzes each bug's 10 generations and pre-computes:

  1. Generations whose `extract_patch(strict)` doesn't pass our `compile`
     metric — these will fail bugsinpy-test by definition (SyntaxError),
     no point running the real test. Mark them as 'compile_fail_skipped'.

  2. Within-bug duplicate generations (same `pred_strict`). The test
     result for a duplicate is identical to its first-occurrence sibling.
     Test the first, copy the result to duplicates.

Outputs:

  - <out_dir>/skipped_results.jsonl
        Pre-computed plausibility rows for all skipped generations
        (compile_fail and duplicates). The test runner doesn't need to
        touch them.

  - <out_dir>/dedup_map.json
        Mapping from duplicate (bug_id, gen_idx) -> representative
        gen_idx, used at merge time to copy real test results onto
        their duplicate slots.

  - <out_dir>/filtered_inference.jsonl
        Inference JSONL containing ONLY the unique compiling generations
        per bug. Feed this to the existing
        `src.runners.bugsinpy.run_plausibility` runner — it'll only
        test what's worth testing.

Bugs with zero unique compiling patches don't even need bugsinpy-checkout
or bugsinpy-compile (the slowest step). They're recorded as all-fail
without touching the framework.

Typical reduction on snakellama_run3 BugsInPy generations:
  1610 patches -> ~500-800 actually tested (50-70% time savings)
  ~30 bugs need no checkout/compile at all

Usage:
    python scripts/prefilter_for_plausibility.py \\
        --eval data/bugsinpy_eval_verified_subset50.jsonl \\
        --inference results/bugsinpy_snakellama_run3.jsonl \\
        --out-dir results/bugsinpy_snakellama_run3_prefilter

Then on the runner side:
    from src.runners.bugsinpy import run_plausibility
    run_plausibility(
        eval_jsonl="<your eval>",
        inference_jsonl="<out_dir>/filtered_inference.jsonl",
        output_jsonl="<final>/plausibility_unfiltered_part.jsonl",
        ...
    )

And to merge the final results back together:
    python scripts/prefilter_for_plausibility.py --merge \\
        --tested <final>/plausibility_unfiltered_part.jsonl \\
        --skipped <out_dir>/skipped_results.jsonl \\
        --dedup-map <out_dir>/dedup_map.json \\
        --out <final>/plausibility_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import score_patch  # noqa: E402


def cmd_filter(args):
    """Generate filtered inference + skipped results."""
    eval_records = []
    with open(args.eval, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_records.append(json.loads(line))
    eval_index = {r["bug_id"]: r for r in eval_records}
    print(f"[prefilter] Eval set: {len(eval_index)} bugs")

    gens_by_bug = {}
    with open(args.inference, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                gens_by_bug[r["bug_id"]] = r
    print(f"[prefilter] Inference: {len(gens_by_bug)} bugs with generations")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered_path = out_dir / "filtered_inference.jsonl"
    skipped_path = out_dir / "skipped_results.jsonl"
    dedup_path = out_dir / "dedup_map.json"

    n_total_gens = 0
    n_skip_compile = 0
    n_skip_dup = 0
    n_to_test = 0
    bugs_with_zero_unique = 0
    dedup_map = {}  # "bug_id|gen_idx" -> representative gen_idx
    project_counts = defaultdict(lambda: {"total": 0, "tested": 0, "skipped_compile": 0, "skipped_dup": 0})

    with open(filtered_path, "w", encoding="utf-8") as f_filt, \
         open(skipped_path, "w", encoding="utf-8") as f_skip:

        for bug_id, rec in gens_by_bug.items():
            if bug_id not in eval_index:
                continue
            project = rec.get("project") or eval_index[bug_id].get("project", "?")
            gold = rec["gold_output"]
            buggy_function = eval_index[bug_id].get("buggy_function", "")
            ir4_input = eval_index[bug_id].get("input")
            n_total_gens += len(rec["generations"])
            project_counts[project]["total"] += len(rec["generations"])

            seen_strict = {}      # pred_strict_norm -> first gen_idx
            kept_indices = []     # generation indices to actually test
            kept_gens = []        # full generation strings (aligned with kept_indices)

            for gi, gen in enumerate(rec["generations"]):
                s = score_patch(gen, gold, buggy_function, ir4_input=ir4_input)
                if not args.skip_compile_filter and not s["compile"]:
                    f_skip.write(json.dumps({
                        "bug_id": bug_id,
                        "gen_idx": gi,
                        "compile_pass": False,
                        "test_pass": False,
                        "test_status": "compile_fail_skipped",
                        "stderr": "patch fails standalone-compile metric; skipped without testing",
                    }) + "\n")
                    n_skip_compile += 1
                    project_counts[project]["skipped_compile"] += 1
                    continue

                key = s["pred_strict"]
                if key in seen_strict:
                    rep_gi = seen_strict[key]
                    dedup_map[f"{bug_id}|{gi}"] = rep_gi
                    n_skip_dup += 1
                    project_counts[project]["skipped_dup"] += 1
                    continue

                seen_strict[key] = gi
                kept_indices.append(gi)
                kept_gens.append(gen)

            n_to_test += len(kept_indices)
            project_counts[project]["tested"] += len(kept_indices)
            if len(kept_indices) == 0:
                bugs_with_zero_unique += 1

            # Emit a filtered inference row. We keep the original gen positions
            # via "orig_indices" so the runner's "gen_idx" lines up with the
            # original generation slot in the source inference file.
            f_filt.write(json.dumps({
                "bug_id": bug_id,
                "project": project,
                "input": rec["input"],
                "gold_output": rec["gold_output"],
                "generations": kept_gens,            # only unique compiling
                "orig_indices": kept_indices,         # for merge time
            }) + "\n")

    with open(dedup_path, "w", encoding="utf-8") as f:
        json.dump(dedup_map, f, indent=2)

    # Summary
    print()
    print("====================  PRE-FILTER SUMMARY  ====================")
    print(f"  Total generations:                {n_total_gens}")
    print(f"  Skipped (failed compile metric):  {n_skip_compile}  ({100*n_skip_compile/max(n_total_gens,1):.1f}%)")
    print(f"  Skipped (duplicate within bug):   {n_skip_dup}  ({100*n_skip_dup/max(n_total_gens,1):.1f}%)")
    print(f"  To actually test:                 {n_to_test}  ({100*n_to_test/max(n_total_gens,1):.1f}%)")
    print(f"  Bugs with 0 unique compiling:     {bugs_with_zero_unique}  (no checkout/compile needed)")
    print()
    print("  Per-project breakdown:")
    print(f"    {'project':20s} {'total':>6s} {'test':>6s} {'compfail':>9s} {'dup':>6s}")
    print(f"    {'-'*20:20s} {'-'*6:>6s} {'-'*6:>6s} {'-'*9:>9s} {'-'*6:>6s}")
    for proj in sorted(project_counts):
        c = project_counts[proj]
        print(f"    {proj:20s} {c['total']:>6d} {c['tested']:>6d} {c['skipped_compile']:>9d} {c['skipped_dup']:>6d}")
    print("==============================================================")
    print()
    print(f"Wrote: {filtered_path}")
    print(f"Wrote: {skipped_path}")
    print(f"Wrote: {dedup_path}")
    print()
    print("Next steps:")
    print(f"  1. Run plausibility runner against {filtered_path.name}")
    print(f"     (it will skip checkout+compile entirely for bugs with empty 'generations')")
    print(f"  2. After it finishes, merge results:")
    print(f"     python {Path(__file__).name} --merge \\")
    print(f"         --tested <runner_output.jsonl> \\")
    print(f"         --skipped {skipped_path} \\")
    print(f"         --dedup-map {dedup_path} \\")
    print(f"         --out <final_plausibility.jsonl>")


def cmd_merge(args):
    """Merge tested + skipped + duplicate-propagated results into one final JSONL."""
    # Load tested rows (real test results from the runner)
    tested = {}  # (bug_id, orig_gen_idx) -> row
    # The filtered_inference rows had `orig_indices` aligned with `generations`.
    # The runner output uses the runner's gen_idx (0..N-1 within filtered).
    # We need a way to translate runner gen_idx back to original gen_idx.
    # Easiest: re-read filtered_inference to recover orig_indices per bug.
    if not args.filtered_inference:
        sys.exit(
            "--filtered-inference required for --merge (the filtered_inference.jsonl "
            "from the prefilter step). It has the orig_indices mapping."
        )

    orig_idx_by_bug = {}  # bug_id -> [orig_idx_for_runner_gen_idx_0, orig_idx_for_runner_gen_idx_1, ...]
    with open(args.filtered_inference, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            orig_idx_by_bug[r["bug_id"]] = r["orig_indices"]

    with open(args.tested, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            bid = r["bug_id"]
            runner_gi = r["gen_idx"]
            mapping = orig_idx_by_bug.get(bid)
            if mapping is None or runner_gi >= len(mapping):
                # Fallback: use runner_gi directly (shouldn't happen)
                orig_gi = runner_gi
            else:
                orig_gi = mapping[runner_gi]
            r2 = dict(r)
            r2["gen_idx"] = orig_gi
            tested[(bid, orig_gi)] = r2

    # Load skipped rows
    skipped = {}
    with open(args.skipped, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            skipped[(r["bug_id"], r["gen_idx"])] = r

    # Load dedup map and propagate tested results onto duplicates
    with open(args.dedup_map, "r", encoding="utf-8") as f:
        dedup_map = json.load(f)

    duplicates = {}
    for key, rep_gi in dedup_map.items():
        bid, gi = key.rsplit("|", 1)
        gi = int(gi)
        rep = tested.get((bid, rep_gi))
        if rep is None:
            # Representative wasn't in tested set (maybe runner failed on it). Mark as error.
            duplicates[(bid, gi)] = {
                "bug_id": bid,
                "gen_idx": gi,
                "compile_pass": False,
                "test_pass": False,
                "test_status": "error",
                "stderr": f"duplicate of gen {rep_gi} but representative result is missing",
            }
        else:
            r2 = dict(rep)
            r2["gen_idx"] = gi
            r2["test_status"] = rep["test_status"] + "_via_duplicate"
            duplicates[(bid, gi)] = r2

    # Merge: tested wins; then duplicates; then skipped
    final = {}
    final.update(skipped)
    final.update(duplicates)
    final.update(tested)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for key in sorted(final):
            f.write(json.dumps(final[key], ensure_ascii=False) + "\n")
    print(f"Wrote merged plausibility: {out_path}")
    print(f"  tested:     {len(tested)} rows")
    print(f"  duplicates: {len(duplicates)} rows")
    print(f"  skipped:    {len(skipped)} rows")
    print(f"  total:      {len(final)} rows")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    f = sub.add_parser("filter", help="Pre-filter inference JSONL (default action if no subcommand)")
    f.add_argument("--eval", required=True, help="Path to bugsinpy_eval_verified*.jsonl")
    f.add_argument("--inference", required=True, help="Path to inference JSONL with generations")
    f.add_argument("--out-dir", required=True, help="Output directory for filtered + skipped + dedup-map")
    f.add_argument("--skip-compile-filter", action="store_true",
                   help="Skip the compile pre-filter (dedup-only mode). Useful when the "
                        "compile metric wrongly drops valid patches — e.g. for class-method "
                        "IR4s or other context-dependent shapes. Lets the actual test runner "
                        "be the source of truth for whether a patch works.")

    m = sub.add_parser("merge", help="Merge tested + skipped + duplicate results")
    m.add_argument("--tested", required=True)
    m.add_argument("--skipped", required=True)
    m.add_argument("--dedup-map", required=True)
    m.add_argument("--filtered-inference", required=True)
    m.add_argument("--out", required=True)

    # Allow legacy positional-style for backward-compat
    args, extra = ap.parse_known_args()
    if args.cmd is None:
        # Try to interpret legacy --merge flag
        if "--merge" in extra:
            extra.remove("--merge")
            sys.argv = [sys.argv[0], "merge"] + extra
            args = ap.parse_args()
        else:
            ap.print_help()
            sys.exit(1)

    if args.cmd == "filter":
        cmd_filter(args)
    elif args.cmd == "merge":
        cmd_merge(args)


if __name__ == "__main__":
    main()
