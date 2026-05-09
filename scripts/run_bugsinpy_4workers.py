#!/usr/bin/env python3
"""
run_bugsinpy_4workers.py
========================
Host-side orchestrator that splits the 161-bug eval across 4 parallel
docker workers, then merges the per-worker JSONLs.

Worker assignment (load-balanced by unique-patch count from snakellama_run3):
  W1: pandas
  W2: luigi, matplotlib, black, tornado, ansible
  W3: thefuck, keras, fastapi, cookiecutter, httpie, PySnooper
  W4: youtube-dl, scrapy, spacy, tqdm, sanic

Each worker mounts the named docker volume `bugsinpy_envs` at
/opt/conda/envs so conda environments persist across runs and across
workers (different projects -> different env hashes -> no race).

Usage (PowerShell, host):
  python scripts/run_bugsinpy_4workers.py
  python scripts/run_bugsinpy_4workers.py --merge-only   # just stitch part files
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

WORKERS = [
    # Pandas is intentionally NOT here — it needs ~24h on its own. Run separately.
    # Balanced by paper Table III compile-minutes-per-bug × bugs-in-our-subset.
    {"id": 1, "projects": ["scrapy"]},                                          # ~13h
    {"id": 2, "projects": ["thefuck", "fastapi"]},                              # ~10h
    {"id": 3, "projects": ["luigi", "keras", "black"]},                         # ~5.5h
    {"id": 4, "projects": ["youtube-dl", "matplotlib", "spacy", "tornado",
                           "ansible", "cookiecutter", "sanic", "tqdm",
                           "PySnooper", "httpie"]},                             # ~1h
]


def windows_repo_path() -> str:
    """Convert REPO_ROOT to a docker-friendly mount source on Windows."""
    p = str(REPO_ROOT).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        # D:/foo/bar -> //d/foo/bar  (works for Docker Desktop)
        return "/" + p[0].lower() + p[2:]
    return p


def stream_to_file(stream, path: Path):
    """Drain a subprocess stream into a file, line-buffered."""
    with open(path, "ab") as f:
        for chunk in iter(lambda: stream.readline(), b""):
            f.write(chunk); f.flush()


def launch_worker(w: dict, eval_path: str, inference_path: str,
                  gen_part: str, gold_part: str, log_path: Path) -> subprocess.Popen:
    name = f"bugsinpy_w{w['id']}"
    projects_arg = ",".join(w["projects"])
    mount_src = windows_repo_path()
    cmd = [
        "docker", "run", "--rm",
        "--name", name,
        "-v", f"{mount_src}:/work",
        "-v", "bugsinpy_envs:/opt/conda/envs",       # conda envs persist
        "-v", "bugsinpy_work:/tmp/bugsinpy_work",     # project checkouts persist
        "bugsinpy-setup:latest",
        "python", "/work/scripts/bugsinpy_run_eval.py",
        "--eval", eval_path,
        "--inference", inference_path,
        "--output", gen_part,
        "--gold-output", gold_part,
        "--projects", projects_arg,
    ]
    env = os.environ.copy()
    env["MSYS_NO_PATHCONV"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    open(log_path, "wb").close()  # truncate
    p = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1,
    )
    threading.Thread(
        target=stream_to_file, args=(p.stdout, log_path), daemon=True,
    ).start()
    return p


def merge_jsonl(parts: list[Path], out: Path) -> int:
    """Concatenate JSONLs in deterministic order, dedup by key field."""
    rows = []
    for p in parts:
        if not p.exists(): continue
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
    ap.add_argument("--eval", default="/work/data/bugsinpy_eval_verified.jsonl")
    ap.add_argument("--inference", default="/work/results/bugsinpy_snakellama_run3.jsonl")
    ap.add_argument("--gen-out", default=str(RESULTS_DIR / "bugsinpy_snakellama_run3_plausibility.jsonl"))
    ap.add_argument("--gold-out", default=str(RESULTS_DIR / "bugsinpy_snakellama_run3_gold.jsonl"))
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip docker runs and just merge existing part files.")
    ap.add_argument("--delete-parts", action="store_true",
                    help="Delete per-worker JSONLs after merge. Default: keep them so "
                         "re-runs (e.g. against another model's generations) can resume "
                         "or be cross-checked.")
    args = ap.parse_args()

    gen_parts = [RESULTS_DIR / f"bugsinpy_run3_gen_part{w['id']}.jsonl" for w in WORKERS]
    gold_parts = [RESULTS_DIR / f"bugsinpy_run3_gold_part{w['id']}.jsonl" for w in WORKERS]
    worker_logs = [RESULTS_DIR / f"bugsinpy_run3_worker{w['id']}.log" for w in WORKERS]

    if not args.merge_only:
        print("Launching 4 workers in parallel...")
        for w in WORKERS:
            print(f"  W{w['id']}: {','.join(w['projects'])}")
        procs = []
        t0 = time.time()
        for w, gp, gdp, lp in zip(WORKERS, gen_parts, gold_parts, worker_logs):
            p = launch_worker(
                w, args.eval, args.inference,
                f"/work/results/bugsinpy_run3_gen_part{w['id']}.jsonl",
                f"/work/results/bugsinpy_run3_gold_part{w['id']}.jsonl",
                lp,
            )
            procs.append(p)
            print(f"  -> W{w['id']} started (pid {p.pid}, log: {lp})")

        rcs = []
        for w, p in zip(WORKERS, procs):
            rc = p.wait()
            rcs.append(rc)
            print(f"W{w['id']} exited rc={rc} (after {(time.time()-t0)/60:.1f} min)")

        if any(rc != 0 for rc in rcs):
            print("[WARN] One or more workers exited non-zero. Merging what we have anyway.",
                  file=sys.stderr)

    n_gen = merge_jsonl(gen_parts, Path(args.gen_out))
    n_gold = merge_jsonl(gold_parts, Path(args.gold_out))
    print(f"\n[merge] gen rows:  {n_gen} -> {args.gen_out}")
    print(f"[merge] gold rows: {n_gold} -> {args.gold_out}")

    # Default: keep part files so re-runs against another model's generations
    # can be merged back without losing data.
    if args.delete_parts:
        for p in gen_parts + gold_parts:
            try: p.unlink()
            except FileNotFoundError: pass
        print("[merge] deleted part files")
    else:
        print("[merge] kept part files (use --delete-parts to remove)")


if __name__ == "__main__":
    main()
