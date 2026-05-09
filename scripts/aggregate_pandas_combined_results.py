#!/usr/bin/env python3
"""
aggregate_pandas_combined_results.py
=====================================
Splits the combined pandas plausibility output back into per-model files
(matching the rest of the repo's bugsinpy_<model>_pandas_plausibility.jsonl
naming) so it slots into the comparison notebook without further work.

Reads:
  results/pandas/bugsinpy_combined_pandas_gen.jsonl
  results/pandas/bugsinpy_combined_pandas_gold.jsonl

Writes (one per model):
  results/snakellama/bugsinpy_snakellama_pandas_plausibility.jsonl
  results/codellama-baseline/bugsinpy_codellama_pandas_plausibility.jsonl
  results/kimi-moonshot/bugsinpy_kimi_pandas_plausibility.jsonl
  results/gemini-2.5-flash/bugsinpy_gemini_pandas_plausibility.jsonl
  results/pandas/bugsinpy_combined_pandas_gold.jsonl    (already merged; copied
                                                         to per-model folders)

Each per-model gen row gets gen_idx remapped from the combined-space
(0..39) back to per-model-space (0..9): orig_gen_idx = combined_gen_idx % 10.

Also prints a per-model pass@10 summary.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
PANDAS_DIR = RESULTS / "pandas"

MODEL_ORDER = ["snakellama", "codellama", "kimi", "gemini"]
MODEL_FOLDER = {
    "snakellama": "snakellama",
    "codellama":  "codellama-baseline",
    "kimi":       "kimi-moonshot",
    "gemini":     "gemini-2.5-flash",
}

GEN_IN  = PANDAS_DIR / "bugsinpy_combined_pandas_gen.jsonl"
GOLD_IN = PANDAS_DIR / "bugsinpy_combined_pandas_gold.jsonl"


def main():
    if not GEN_IN.exists() or not GOLD_IN.exists():
        raise SystemExit(f"missing combined outputs; expected {GEN_IN} and {GOLD_IN}")

    # Pre-open per-model gen output files
    out_files = {}
    for m in MODEL_ORDER:
        out_path = RESULTS / MODEL_FOLDER[m] / f"bugsinpy_{m}_pandas_plausibility.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_files[m] = open(out_path, "w", encoding="utf-8")
    gold_dest = {m: RESULTS / MODEL_FOLDER[m] / f"bugsinpy_{m}_pandas_gold.jsonl"
                 for m in MODEL_ORDER}

    n_per_model: Counter = Counter()
    pass_per_model: dict[str, dict[str, int]] = {m: defaultdict(int) for m in MODEL_ORDER}
    bugs_per_model: dict[str, set[str]] = {m: set() for m in MODEL_ORDER}

    try:
        with open(GEN_IN, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                combined_gi = r["gen_idx"]
                model = MODEL_ORDER[combined_gi // 10]
                orig_gi = combined_gi % 10

                # Remap dedup_rep too (so dedup chains stay model-internal).
                # If the dedup rep is from another model's slice, this just means
                # the patch coincidentally matched one of that model's generations
                # — we re-tag to this model's slice.
                dedup_rep = r.get("dedup_rep")
                if isinstance(dedup_rep, int):
                    dedup_rep = dedup_rep % 10

                out_row = {
                    "bug_id":      r["bug_id"],
                    "gen_idx":     orig_gi,
                    "dedup_rep":   dedup_rep,
                    "test_pass":   r["test_pass"],
                    "test_status": r["test_status"],
                    "stderr_tail": r["stderr_tail"],
                    "stage":       r["stage"],
                    "model":       model,
                }
                out_files[model].write(json.dumps(out_row) + "\n")
                n_per_model[model] += 1
                bugs_per_model[model].add(r["bug_id"])
                if r["test_pass"]:
                    pass_per_model[model][r["bug_id"]] += 1
    finally:
        for fh in out_files.values():
            fh.close()

    # Copy gold to each model's folder (gold is identical across models)
    for m in MODEL_ORDER:
        shutil.copy(GOLD_IN, gold_dest[m])

    # Print summary
    print(f"\n=== Pandas plausibility — per-model pass@10 (combined-run aggregation) ===\n")
    for m in MODEL_ORDER:
        pa10 = sum(1 for b in bugs_per_model[m] if pass_per_model[m][b] > 0)
        n_b = len(bugs_per_model[m])
        print(f"  {m:<11s} {n_per_model[m]:>5d} gen rows | {n_b:>2d} bugs | "
              f"pass@10 = {pa10}/{n_b} = {100*pa10/max(n_b,1):.1f}%")
        out_path = RESULTS / MODEL_FOLDER[m] / f"bugsinpy_{m}_pandas_plausibility.jsonl"
        print(f"               -> {out_path}")
    print(f"\n  gold (45 rows, identical across models) copied to each results/<model>/ folder.")


if __name__ == "__main__":
    main()
