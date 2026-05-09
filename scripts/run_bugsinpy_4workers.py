#!/usr/bin/env python3
"""
run_bugsinpy_4workers.py
========================
Host-side orchestrator that splits the 161-bug eval across 4 parallel
docker workers, then merges the per-worker JSONLs.

Output layout (matches the restructured results/ tree):
  results/all_docker_runs_result/                 <- per-worker artifacts
    bugsinpy_<model>_gen_part{1..4}.jsonl
    bugsinpy_<model>_gold_part{1..4}.jsonl
    bugsinpy_<model>_worker{1..4}.log
    bugsinpy_<model>_gold.jsonl                   <- merged gold
  results/<model_folder>/                         <- per-model results
    bugsinpy_<model>_plausibility.jsonl           <- merged gen (plausibility)

Worker assignment (pandas excluded — see run_pandas_4workers.py):
  W1: scrapy                                      ~13h
  W2: thefuck, fastapi                            ~10h
  W3: luigi, keras, black                         ~5.5h
  W4: youtube-dl, matplotlib, spacy, tornado,
      ansible, cookiecutter, sanic, tqdm,
      PySnooper, httpie                           ~1h

Each worker mounts the named docker volume `bugsinpy_envs` at
/opt/conda/envs so conda environments persist across runs and across
workers (different projects -> different env hashes -> no race).

Usage (PowerShell, host):
  # Default: run snakellama generations end-to-end
  python scripts/run_bugsinpy_4workers.py
  # Other models pick up their --inference + paths from --model:
  python scripts/run_bugsinpy_4workers.py --model kimi
  python scripts/run_bugsinpy_4workers.py --model gemini
  python scripts/run_bugsinpy_4workers.py --model codellama
  # Re-merge without re-running docker:
  python scripts/run_bugsinpy_4workers.py --merge-only
  # Restrict to the 71 snakellama-reproducible bugs (apples-to-apples):
  python scripts/run_bugsinpy_4workers.py --model kimi \\
      --bug-ids-file results/reproducible_bug_ids.txt
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

# Maps the --model flag to (per-model folder under results/, generation file).
# The model folder holds the inference output and the merged plausibility file.
# Per-worker part files, logs, and merged gold all land in DOCKER_ARTIFACTS_DIR.
MODEL_LAYOUT = {
    "snakellama":      ("snakellama",         "bugsinpy_snakellama_generations.jsonl"),
    "codellama":       ("codellama-baseline", "bugsinpy_codellama_generations.jsonl"),
    "kimi":            ("kimi-moonshot",      "bugsinpy_kimi_generations_aligned.jsonl"),
    "gemini":          ("gemini-2.5-flash",   "bugsinpy_gemini_generations_aligned.jsonl"),
}

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
    ap.add_argument("--model", default="snakellama", choices=sorted(MODEL_LAYOUT.keys()),
                    help="Which model's generations to evaluate. Drives the default "
                         "--inference, --prefix, and merged-output paths per the "
                         "results/<folder>/ layout. Override individual paths below "
                         "if needed.")
    ap.add_argument("--eval", default="/work/data/bugsinpy_eval_verified.jsonl")
    ap.add_argument("--inference", default=None,
                    help="Container-side path to the model's generations JSONL. "
                         "Default: /work/results/<folder>/<bugsinpy_<model>_generations[_aligned].jsonl> "
                         "(varies per --model).")
    ap.add_argument("--out-dir", default=str(DOCKER_ARTIFACTS_DIR),
                    help="Directory for per-worker part files, worker logs, and the "
                         "merged GOLD jsonl. Defaults to results/all_docker_runs_result/.")
    ap.add_argument("--prefix", default=None,
                    help="Filename prefix used for per-worker artifacts in --out-dir. "
                         "Default: 'bugsinpy_<model>' (e.g. bugsinpy_kimi). Yields "
                         "bugsinpy_kimi_gen_part{1..4}.jsonl, bugsinpy_kimi_gold_part*, "
                         "bugsinpy_kimi_worker{1..4}.log, and the merged "
                         "bugsinpy_kimi_gold.jsonl.")
    ap.add_argument("--gen-out", default=None,
                    help="Override path for merged gen (plausibility) JSONL. "
                         "Default: results/<folder>/bugsinpy_<model>_plausibility.jsonl.")
    ap.add_argument("--gold-out", default=None,
                    help="Override path for merged gold JSONL. Default: "
                         "<out-dir>/<prefix>_gold.jsonl.")
    ap.add_argument("--bug-ids", default=None,
                    help="Comma-separated bug_ids to restrict ALL workers to "
                         "(e.g. only the 71 reproducible bugs). Each worker still "
                         "filters by its assigned projects, so this further narrows "
                         "to (project ∩ bug-ids). Used for apples-to-apples "
                         "Kimi/Gemini reruns over the snakellama-reproducible set "
                         "(see results/reproducible_bug_ids.txt).")
    ap.add_argument("--bug-ids-file", default=None,
                    help="File containing newline-separated bug_ids; combined with --bug-ids.")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip docker runs and just merge existing part files.")
    ap.add_argument("--delete-parts", action="store_true",
                    help="Delete per-worker JSONLs after merge. Default: keep them so "
                         "re-runs (e.g. against another model's generations) can resume "
                         "or be cross-checked.")
    args = ap.parse_args()

    model_folder, gen_filename = MODEL_LAYOUT[args.model]
    if args.prefix is None:
        args.prefix = f"bugsinpy_{args.model}"
    if args.inference is None:
        args.inference = f"/work/results/{model_folder}/{gen_filename}"

    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    gen_parts = [out_dir / f"{args.prefix}_gen_part{w['id']}.jsonl" for w in WORKERS]
    gold_parts = [out_dir / f"{args.prefix}_gold_part{w['id']}.jsonl" for w in WORKERS]
    worker_logs = [out_dir / f"{args.prefix}_worker{w['id']}.log" for w in WORKERS]
    default_gen_out = (RESULTS_DIR / model_folder
                       / f"bugsinpy_{args.model}_plausibility.jsonl")
    gen_out = Path(args.gen_out) if args.gen_out else default_gen_out
    gold_out = Path(args.gold_out) if args.gold_out else out_dir / f"{args.prefix}_gold.jsonl"
    gen_out.parent.mkdir(parents=True, exist_ok=True)

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
