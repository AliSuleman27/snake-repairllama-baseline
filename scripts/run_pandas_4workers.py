#!/usr/bin/env python3
"""
run_pandas_4workers.py
======================
Pandas-only 4-way parallel runner. Splits the 45 pandas bugs across 4
docker workers (round-robin by bug_id), runs them in parallel, then
merges results into:
  results/snakellama_full/pandas_gen_part{1..4}.jsonl
  results/snakellama_full/pandas_gold_part{1..4}.jsonl
  results/snakellama_full/pandas_merged_gen.jsonl
  results/snakellama_full/pandas_merged_gold.jsonl

Usage (host):
  python scripts/run_pandas_4workers.py
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
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_INFERENCE = "/work/results/snakellama_model_generations/bugsinpy_snakellama_run3.jsonl"
DEFAULT_EVAL = "/work/data/bugsinpy_eval_verified.jsonl"
DEFAULT_OUTPUT_SUBDIR = RESULTS_DIR / "snakellama_full"


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
    name = f"bugsinpy_pandas_w{worker_id}"
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
    threading.Thread(target=stream_to_file, args=(p.stdout, log_path), daemon=True).start()
    return p


def merge_jsonl(parts: list[Path], out: Path) -> int:
    rows = []
    for p in parts:
        if not p.exists(): continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(line.rstrip("\n"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows: f.write(r + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-host", default=str(REPO_ROOT / "data" / "bugsinpy_eval_verified.jsonl"))
    ap.add_argument("--eval", default=DEFAULT_EVAL)
    ap.add_argument("--inference", default=DEFAULT_INFERENCE)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_SUBDIR))
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n = args.n_workers

    # Find pandas bugs in eval
    pandas_bugs = []
    with open(args.eval_host, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                if r["project"] == "pandas":
                    pandas_bugs.append(r["bug_id"])
    pandas_bugs.sort(key=lambda x: int(x.split("/")[1]))
    print(f"[orch] {len(pandas_bugs)} pandas bugs to split across {n} workers")

    # Round-robin split for balanced bug-id distribution
    groups = [pandas_bugs[i::n] for i in range(n)]
    for i, g in enumerate(groups, 1):
        print(f"  W{i}: {len(g)} bugs ({g[0]} ... {g[-1]})")

    gen_parts = [out_dir / f"pandas_gen_part{i+1}.jsonl" for i in range(n)]
    gold_parts = [out_dir / f"pandas_gold_part{i+1}.jsonl" for i in range(n)]
    log_paths = [out_dir / f"pandas_worker{i+1}.log" for i in range(n)]

    if not args.merge_only:
        procs = []
        t0 = time.time()
        for i, g in enumerate(groups, 1):
            p = launch_worker(
                i, g, args.eval, args.inference,
                f"/work/results/snakellama_full/pandas_gen_part{i}.jsonl",
                f"/work/results/snakellama_full/pandas_gold_part{i}.jsonl",
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

    # Merge
    n_gen = merge_jsonl(gen_parts, out_dir / "pandas_merged_gen.jsonl")
    n_gold = merge_jsonl(gold_parts, out_dir / "pandas_merged_gold.jsonl")
    print(f"\n[merge] gen rows:  {n_gen} -> {out_dir / 'pandas_merged_gen.jsonl'}")
    print(f"[merge] gold rows: {n_gold} -> {out_dir / 'pandas_merged_gold.jsonl'}")


if __name__ == "__main__":
    main()
