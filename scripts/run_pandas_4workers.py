#!/usr/bin/env python3
"""
run_pandas_4workers.py
======================
Pandas-only 4-way parallel runner. Splits the 45 pandas bugs across 4
docker workers (round-robin by bug_id), runs them in parallel, then
merges results.

Output layout (matches the restructured results/ tree):
  results/all_docker_runs_result/                          <- per-worker
    bugsinpy_<model>_pandas_gen_part{1..4}.jsonl
    bugsinpy_<model>_pandas_gold_part{1..4}.jsonl
    bugsinpy_<model>_pandas_worker{1..4}.log
    bugsinpy_<model>_pandas_gold.jsonl                     <- merged gold
  results/<model_folder>/
    bugsinpy_<model>_pandas_plausibility.jsonl             <- merged gen

Default --model is snakellama; pass --model {kimi,gemini,codellama} to
run pandas against another model's generations.

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
DOCKER_ARTIFACTS_DIR = RESULTS_DIR / "all_docker_runs_result"
DEFAULT_EVAL = "/work/data/bugsinpy_eval_verified.jsonl"

# Maps --model to (per-model folder, generation filename) — must stay in sync
# with run_bugsinpy_4workers.py::MODEL_LAYOUT.
MODEL_LAYOUT = {
    "snakellama":      ("snakellama",         "bugsinpy_snakellama_generations.jsonl"),
    "codellama":       ("codellama-baseline", "bugsinpy_codellama_generations.jsonl"),
    "kimi":            ("kimi-moonshot",      "bugsinpy_kimi_generations_aligned.jsonl"),
    "gemini":          ("gemini-2.5-flash",   "bugsinpy_gemini_generations_aligned.jsonl"),
}


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
    ap.add_argument("--model", default="snakellama", choices=sorted(MODEL_LAYOUT.keys()),
                    help="Which model's generations to evaluate. Drives default "
                         "--inference, --prefix, and merged-output paths.")
    ap.add_argument("--eval-host", default=str(REPO_ROOT / "data" / "bugsinpy_eval_verified.jsonl"))
    ap.add_argument("--eval", default=DEFAULT_EVAL)
    ap.add_argument("--inference", default=None,
                    help="Container-side path to the model's generations JSONL. "
                         "Defaults to /work/results/<folder>/<file> based on --model.")
    ap.add_argument("--out-dir", default=str(DOCKER_ARTIFACTS_DIR),
                    help="Directory for per-worker part files, logs, and merged "
                         "GOLD jsonl. Default: results/all_docker_runs_result/.")
    ap.add_argument("--prefix", default=None,
                    help="Prefix for per-worker artifacts. "
                         "Default: bugsinpy_<model>_pandas.")
    ap.add_argument("--gen-out", default=None,
                    help="Override merged gen (plausibility) path. "
                         "Default: results/<folder>/bugsinpy_<model>_pandas_plausibility.jsonl.")
    ap.add_argument("--gold-out", default=None,
                    help="Override merged gold path. "
                         "Default: <out-dir>/<prefix>_gold.jsonl.")
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    model_folder, gen_filename = MODEL_LAYOUT[args.model]
    if args.prefix is None:
        args.prefix = f"bugsinpy_{args.model}_pandas"
    if args.inference is None:
        args.inference = f"/work/results/{model_folder}/{gen_filename}"

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

    gen_parts = [out_dir / f"{args.prefix}_gen_part{i+1}.jsonl" for i in range(n)]
    gold_parts = [out_dir / f"{args.prefix}_gold_part{i+1}.jsonl" for i in range(n)]
    log_paths = [out_dir / f"{args.prefix}_worker{i+1}.log" for i in range(n)]

    # Container-side paths to part files (out_dir is mounted at /work/<rel> inside docker).
    rel_out = out_dir.resolve().relative_to(REPO_ROOT).as_posix()

    if not args.merge_only:
        procs = []
        t0 = time.time()
        for i, g in enumerate(groups, 1):
            p = launch_worker(
                i, g, args.eval, args.inference,
                f"/work/{rel_out}/{args.prefix}_gen_part{i}.jsonl",
                f"/work/{rel_out}/{args.prefix}_gold_part{i}.jsonl",
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
    default_gen_out = (RESULTS_DIR / model_folder
                       / f"bugsinpy_{args.model}_pandas_plausibility.jsonl")
    gen_out = Path(args.gen_out) if args.gen_out else default_gen_out
    gold_out = Path(args.gold_out) if args.gold_out else out_dir / f"{args.prefix}_gold.jsonl"
    gen_out.parent.mkdir(parents=True, exist_ok=True)
    gold_out.parent.mkdir(parents=True, exist_ok=True)

    n_gen = merge_jsonl(gen_parts, gen_out)
    n_gold = merge_jsonl(gold_parts, gold_out)
    print(f"\n[merge] gen rows:  {n_gen} -> {gen_out}")
    print(f"[merge] gold rows: {n_gold} -> {gold_out}")


if __name__ == "__main__":
    main()
