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
                  gen_part: str, gold_part: str, log_path: Path,
                  bug_ids: str | None = None) -> subprocess.Popen:
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
    if bug_ids:
        cmd += ["--bug-ids", bug_ids]
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
    ap.add_argument("--out-dir", default=str(RESULTS_DIR),
                    help="Directory for per-worker parts, worker logs, and final "
                         "merged JSONLs. Defaults to repo's results/. For Kimi/Gemini "
                         "etc., point to results/kimi-moonshot/ or similar.")
    ap.add_argument("--prefix", default="bugsinpy_run3",
                    help="Filename prefix used for ALL outputs in --out-dir. "
                         "E.g. prefix='bugsinpy_kimi' yields "
                         "bugsinpy_kimi_gen_part{1..4}.jsonl, "
                         "bugsinpy_kimi_worker{1..4}.log, and the merged "
                         "bugsinpy_kimi_plausibility.jsonl + bugsinpy_kimi_gold.jsonl.")
    ap.add_argument("--gen-out", default=None,
                    help="Override path for merged gen JSONL. Default: "
                         "<out-dir>/<prefix>_plausibility.jsonl")
    ap.add_argument("--gold-out", default=None,
                    help="Override path for merged gold JSONL. Default: "
                         "<out-dir>/<prefix>_gold.jsonl")
    ap.add_argument("--bug-ids", default=None,
                    help="Comma-separated bug_ids to restrict ALL workers to "
                         "(e.g. only the 71 reproducible bugs). Each worker still "
                         "filters by its assigned projects, so this further narrows "
                         "to (project ∩ bug-ids). Used for apples-to-apples "
                         "Kimi/Gemini reruns over the snakellama-reproducible set.")
    ap.add_argument("--bug-ids-file", default=None,
                    help="File containing newline-separated bug_ids; combined with --bug-ids.")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip docker runs and just merge existing part files.")
    ap.add_argument("--delete-parts", action="store_true",
                    help="Delete per-worker JSONLs after merge. Default: keep them so "
                         "re-runs (e.g. against another model's generations) can resume "
                         "or be cross-checked.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    gen_parts = [out_dir / f"{args.prefix}_gen_part{w['id']}.jsonl" for w in WORKERS]
    gold_parts = [out_dir / f"{args.prefix}_gold_part{w['id']}.jsonl" for w in WORKERS]
    worker_logs = [out_dir / f"{args.prefix}_worker{w['id']}.log" for w in WORKERS]
    gen_out = Path(args.gen_out) if args.gen_out else out_dir / f"{args.prefix}_plausibility.jsonl"
    gold_out = Path(args.gold_out) if args.gold_out else out_dir / f"{args.prefix}_gold.jsonl"

    # Compose bug-ids list from --bug-ids and --bug-ids-file
    bug_ids_set: set[str] = set()
    if args.bug_ids:
        bug_ids_set.update(b.strip() for b in args.bug_ids.split(",") if b.strip())
    if args.bug_ids_file and Path(args.bug_ids_file).exists():
        with open(args.bug_ids_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    bug_ids_set.add(line.strip())
    bug_ids_arg = ",".join(sorted(bug_ids_set)) if bug_ids_set else None
    if bug_ids_arg:
        print(f"[orch] restricting to {len(bug_ids_set)} bug_ids")

    if not args.merge_only:
        print("Launching 4 workers in parallel...")
        for w in WORKERS:
            print(f"  W{w['id']}: {','.join(w['projects'])}")
        procs = []
        t0 = time.time()
        # Translate host part-file paths to /work/... container paths
        # (out_dir on host is mounted at /work/<rel_path> inside the container)
        rel_out = out_dir.relative_to(REPO_ROOT).as_posix()
        for w, gp, gdp, lp in zip(WORKERS, gen_parts, gold_parts, worker_logs):
            p = launch_worker(
                w, args.eval, args.inference,
                f"/work/{rel_out}/{args.prefix}_gen_part{w['id']}.jsonl",
                f"/work/{rel_out}/{args.prefix}_gold_part{w['id']}.jsonl",
                lp,
                bug_ids=bug_ids_arg,
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

    n_gen = merge_jsonl(gen_parts, gen_out)
    n_gold = merge_jsonl(gold_parts, gold_out)
    print(f"\n[merge] gen rows:  {n_gen} -> {gen_out}")
    print(f"[merge] gold rows: {n_gold} -> {gold_out}")

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
