#!/usr/bin/env python3
"""
build_pandas_combined_inference.py
==================================
Concatenates the four models' pandas-bug generations into a single JSONL with
40 generations per bug:

  gens[ 0..9 ] = snakellama
  gens[10..19] = codellama
  gens[20..29] = kimi
  gens[30..39] = gemini

The order is fixed; aggregation reverses it via `model = MODEL_ORDER[gen_idx // 10]`.

Output: results/pandas/bugsinpy_combined_pandas_generations.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

MODEL_ORDER = ["snakellama", "codellama", "kimi", "gemini"]
MODEL_FILES = {
    "snakellama": RESULTS / "snakellama"         / "bugsinpy_snakellama_generations.jsonl",
    "codellama":  RESULTS / "codellama-baseline" / "bugsinpy_codellama_generations.jsonl",
    "kimi":       RESULTS / "kimi-moonshot"      / "bugsinpy_kimi_generations_aligned.jsonl",
    "gemini":     RESULTS / "gemini-2.5-flash"   / "bugsinpy_gemini_generations_aligned.jsonl",
}

OUT_DIR = RESULTS / "pandas"
OUT_FILE = OUT_DIR / "bugsinpy_combined_pandas_generations.jsonl"


def load_pandas_only(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["bug_id"].startswith("pandas/"):
                out[r["bug_id"]] = r
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_model = {m: load_pandas_only(p) for m, p in MODEL_FILES.items()}
    bug_ids = sorted(set.intersection(*[set(d.keys()) for d in per_model.values()]),
                     key=lambda x: int(x.split("/")[1]))
    print(f"[build] {len(bug_ids)} pandas bugs common to all {len(MODEL_ORDER)} models")

    for m in MODEL_ORDER:
        only_m = set(per_model[m].keys()) - set(bug_ids)
        if only_m:
            print(f"  WARNING: {m} has {len(only_m)} pandas bugs not in intersection: {sorted(only_m)}")

    n_written = 0
    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for bid in bug_ids:
            ref = per_model[MODEL_ORDER[0]][bid]
            combined_gens: list[str] = []
            for m in MODEL_ORDER:
                gens = per_model[m][bid]["generations"]
                if len(gens) != 10:
                    raise SystemExit(f"{m} {bid}: expected 10 gens, got {len(gens)}")
                combined_gens.extend(gens)

            row = {
                "bug_id":      ref["bug_id"],
                "project":     ref["project"],
                "input":       ref["input"],
                "gold_output": ref["gold_output"],
                "generations": combined_gens,
                "_model_order": MODEL_ORDER,
            }
            fout.write(json.dumps(row) + "\n")
            n_written += 1

    print(f"[build] wrote {n_written} bugs ({n_written * 40} total gens) -> {OUT_FILE}")


if __name__ == "__main__":
    main()
