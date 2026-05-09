#!/usr/bin/env python3
"""
run_pandas_combined_4workers.py
================================
Pandas plausibility for ALL FOUR models in a single docker run by exploiting
env-compile cache reuse. Splits the 45 pandas bugs round-robin across 4
docker workers; each worker runs `bugsinpy_run_eval.py` against the COMBINED
inference file (40 gens per bug = 10 snakellama + 10 codellama + 10 kimi
+ 10 gemini).

Output (under results/pandas/):
  bugsinpy_combined_pandas_gen_part{1..4}.jsonl       (40 rows per bug)
  bugsinpy_combined_pandas_gold_part{1..4}.jsonl      (1 row per bug)
  bugsinpy_combined_pandas_worker{1..4}.log
  bugsinpy_combined_pandas_gen.jsonl                  (merged gen, post-run)
  bugsinpy_combined_pandas_gold.jsonl                 (merged gold, post-run)

Aggregation back to per-model plausibility files happens in
aggregate_pandas_combined_results.py.

Why 4 workers (not 6): pandas conda envs need ~8 GB peak each during compile;
4 concurrent compiles is the safe ceiling on a typical WSL2 6 GB-default.

Usage (host):
  python scripts/run_pandas_combined_4workers.py
  python scripts/run_pandas_combined_4workers.py --merge-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
OUT_DIR = RESULTS / "pandas"

DEFAULT_INFERENCE = "/work/results/pandas/bugsinpy_combined_pandas_generations.jsonl"
DEFAULT_EVAL = "/work/data/bugsinpy_eval_verified.jsonl"
DEFAULT_EVAL_HOST = REPO_ROOT / "data" / "bugsinpy_eval_verified.jsonl"

PREFIX = "bugsinpy_combined_pandas"


def windows_repo_path() -> str:
    p = str(REPO_ROOT).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        return "/" + p[0].lower() + p[2:]
    return p


def stream_to_file(stream, path: Path):
    with open(path, "ab") as f:
        for chunk in iter(lambda: stream.readline(), b""):
            f.write(chunk); f.flush()


def launch_worker(worker_id: int, bug_ids: list[str], eval_path: str,
                  inference_path: str, gen_part: str, gold_part: str,
                  log_path: Path) -> subprocess.Popen:
    name = f"bugsinpy_combined_pandas_w{worker_id}"
    mount_src = windows_repo_path()
    cmd = [
        "docker", "run", "--rm",
        "--name", name,
        "-v", f"{mount_src}:/work",
        "-v", "bugsinpy_envs:/opt/conda/envs",
        "-v", "bugsinpy_work:/tmp/bugsinpy_work",
        "bugsinpy-setup:latest",
        "python", "/work/scripts/bugsinpy_run_eval.py",
        "--eval", eval_path,
        "--inference", inference_path,
        "--output", gen_part,
        "--gold-output", gold_part,
        "--projects", "pandas",
        "--bug-ids", ",".join(bug_ids),
    ]
    env = os.environ.copy()
    env["MSYS_NO_PATHCONV"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    open(log_path, "wb").close()
    p = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1,
    )
    threading.Thread(target=stream_to_file, args=(p.stdout, log_path),
                     daemon=True).start()
    return p


def merge_jsonl(parts: list[Path], out: Path) -> int:
    rows = []
    for p in parts:
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(line.rstrip("\n"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-host", default=str(DEFAULT_EVAL_HOST))
    ap.add_argument("--eval", default=DEFAULT_EVAL)
    ap.add_argument("--inference", default=DEFAULT_INFERENCE)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n = args.n_workers

    pandas_bugs: list[str] = []
    with open(args.eval_host, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                if r["project"] == "pandas":
                    pandas_bugs.append(r["bug_id"])
    pandas_bugs.sort(key=lambda x: int(x.split("/")[1]))
    print(f"[orch] {len(pandas_bugs)} pandas bugs to split across {n} workers "
          f"(combined inference: 40 gens/bug across snakellama/codellama/kimi/gemini)")

    groups = [pandas_bugs[i::n] for i in range(n)]
    for i, g in enumerate(groups, 1):
        print(f"  W{i}: {len(g)} bugs ({g[0]} ... {g[-1]})")

    gen_parts = [out_dir / f"{PREFIX}_gen_part{i+1}.jsonl" for i in range(n)]
    gold_parts = [out_dir / f"{PREFIX}_gold_part{i+1}.jsonl" for i in range(n)]
    log_paths = [out_dir / f"{PREFIX}_worker{i+1}.log" for i in range(n)]

    rel_out = out_dir.resolve().relative_to(REPO_ROOT).as_posix()

    if not args.merge_only:
        procs = []
        t0 = time.time()
        for i, g in enumerate(groups, 1):
            p = launch_worker(
                i, g, args.eval, args.inference,
                f"/work/{rel_out}/{PREFIX}_gen_part{i}.jsonl",
                f"/work/{rel_out}/{PREFIX}_gold_part{i}.jsonl",
                log_paths[i-1],
            )
            procs.append(p)
            print(f"  -> W{i} started (pid {p.pid}, log: {log_paths[i-1]})")

        rcs = []
        for i, p in enumerate(procs, 1):
            rc = p.wait()
            rcs.append(rc)
            print(f"W{i} exited rc={rc} (after {(time.time()-t0)/60:.1f} min)")
        if any(rc != 0 for rc in rcs):
            print("[WARN] some workers exited non-zero", file=sys.stderr)

    n_gen = merge_jsonl(gen_parts, out_dir / f"{PREFIX}_gen.jsonl")
    n_gold = merge_jsonl(gold_parts, out_dir / f"{PREFIX}_gold.jsonl")
    print(f"\n[merge] gen rows:  {n_gen} -> {out_dir / f'{PREFIX}_gen.jsonl'}")
    print(f"[merge] gold rows: {n_gold} -> {out_dir / f'{PREFIX}_gold.jsonl'}")
    print()
    print("Next step (morning):")
    print("  python scripts/aggregate_pandas_combined_results.py")


if __name__ == "__main__":
    main()
