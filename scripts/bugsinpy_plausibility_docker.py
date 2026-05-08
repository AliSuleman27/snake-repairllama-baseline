#!/usr/bin/env python3
"""
bugsinpy_plausibility_docker.py
================================
Docker-based plausibility runner for the BugsInPy benchmark.

Designed to run INSIDE the BugsInPy Docker container (built from
the local ``BugsInPy/`` repo's Dockerfile) where:

  - miniconda3 is at /opt/conda
  - the BugsInPy framework is at /home/user/BugsInPy with bin/ on PATH
  - the snake-repairllama-baseline repo is bind-mounted at /work

For each bug whose generations need testing, this runner:

  1. ``bugsinpy-checkout -p <project> -v 0 -i <bug_num> -w /tmp/work``
     to materialize the buggy version of the project.
  2. Reads bug.info + requirements.txt and computes a conda env hash
     (md5 of ``python_version\\n<requirements_content>``) — same
     pattern as the upstream ``bugsinpy-testall`` script.
  3. Creates the conda env once (cached for siblings with the same hash).
  4. Runs ``bugsinpy-compile`` once per bug inside the env.
  5. For each unique compiling generation in the pre-filtered inference:
     - Splices our predicted patch into the source file at the function's
       known line range.
     - Runs ``bugsinpy-test``.
     - Classifies the output (pass / fail / error) using the same keyword
       logic as ``bugsinpy-testall``.
     - Restores the original source file.
  6. Records one JSONL row per (bug_id, original_gen_idx).

Launch from host PowerShell:

  cd D:\\snake-repairllama-baseline\\BugsInPy
  docker run --rm `
      -v "D:\\snake-repairllama-baseline:/work" `
      bugsinpy-setup:latest `
      python /work/scripts/bugsinpy_plausibility_docker.py `
          --eval /work/data/bugsinpy_eval_verified_reproducible.jsonl `
          --inference /work/results/bugsinpy_snakellama_run3_prefilter_reproducible/filtered_inference.jsonl `
          --output /work/results/bugsinpy_snakellama_run3_plausibility_tested.jsonl

The ``--rm`` flag is fine: conda envs persist for the duration of a single
``docker run`` (which executes the whole loop), then are discarded with
the container. If you want envs to persist across multiple runs, add
``-v conda_envs_volume:/opt/conda/envs``.

Resume support: re-running the same command picks up where it left off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Make our own src package importable when running inside the container at /work
SCRIPT_DIR = Path(__file__).resolve().parent
WORK_ROOT = SCRIPT_DIR.parent  # /work
sys.path.insert(0, str(WORK_ROOT))

from src.patcher import reconstruct_patched_function, splice_function_into_file  # noqa: E402
from src.postprocess import extract_patch  # noqa: E402


# ─────────────────────── classification (mirrors testall) ─────────────────

ERROR_KEYWORDS = (
    "= ERRORS =",
    "ImportError while loading",
    ": command not found",
    "You have not compile this project",
)
FAIL_KEYWORDS = ("= FAILURES =", "FAILED (")
PASS_PATTERN = re.compile(r"\bpassed\b|\bOK\b")


def classify(output: str) -> str:
    """Map combined stdout+stderr from bugsinpy-test to {pass,fail,error}."""
    if any(k in output for k in ERROR_KEYWORDS):
        return "error"
    if any(k in output for k in FAIL_KEYWORDS):
        return "fail"
    if PASS_PATTERN.search(output):
        return "pass"
    return "error"


# ─────────────────────── conda env management ─────────────────────────────


def conda_env_hash(python_version: str, requirements_content: str) -> str:
    """Same hash function as bugsinpy-testall.

        cat <(echo $bug_python_version) "$requirements" | md5sum | cut -d' ' -f 1
    """
    h = hashlib.md5()
    h.update(f"{python_version}\n{requirements_content}".encode("utf-8"))
    return h.hexdigest()


def conda_env_exists(env_name: str) -> bool:
    r = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    return env_name in (r.stdout or "")


def conda_env_create(env_name: str, python_version: str, log_file: Path) -> bool:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["conda", "create", "-n", env_name, "-y", f"python={python_version}", "pytest"]
    with open(log_file, "ab") as f:
        f.write(f"\n[conda create {env_name}]\n".encode())
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode == 0


def run_in_env(env_name: str, cmd: str, cwd: Path, timeout: int = 180) -> Tuple[int, str, str]:
    """Run a bash command inside an activated conda env. Returns (rc, stdout, stderr)."""
    bash = (
        ". /opt/conda/etc/profile.d/conda.sh\n"
        f"conda activate {env_name}\n"
        f"cd {cwd}\n"
        f"{cmd}\n"
    )
    try:
        r = subprocess.run(
            ["bash", "-c", bash],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        err = (e.stderr or "")
        if isinstance(out, bytes): out = out.decode("utf-8", errors="replace")
        if isinstance(err, bytes): err = err.decode("utf-8", errors="replace")
        return -9, out, err + f"\n[TIMEOUT after {timeout}s]"


# ─────────────────────── bug pipeline helpers ─────────────────────────────


def parse_bug_info(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r'^\s*(\w+)\s*=\s*"?(.*?)"?\s*$', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def checkout_buggy(project: str, bug_num: int, work_root: Path, log_file: Path) -> Optional[Path]:
    """Run bugsinpy-checkout -v 0. Returns project_dir if successful, None on failure."""
    work_dir = work_root / f"{project}_{bug_num}"
    project_dir = work_dir / project
    if project_dir.exists() and (project_dir / ".git").exists():
        return project_dir  # cached (resume case)
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"bugsinpy-checkout -p {project} -v 0 -i {bug_num} -w {work_dir}"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "ab") as f:
        f.write(f"\n[checkout {project}/{bug_num}]\n".encode())
        r = subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, timeout=300)
    if not project_dir.exists():
        return None
    # Ensure requirements.txt exists (testall does this too)
    req = work_dir / "bugsinpy_requirements.txt"
    if not req.exists():
        req.write_text("")
    return project_dir


def setup_env_and_compile(
    project: str, bug_num: int, work_dir: Path, project_dir: Path, log_file: Path,
) -> Tuple[bool, str]:
    """Ensure conda env exists, run bugsinpy-compile in it. Returns (ok, env_name)."""
    bug_info_path = work_dir / "bugsinpy_bug.info"
    requirements_path = work_dir / "bugsinpy_requirements.txt"

    if not bug_info_path.exists():
        return False, ""

    bug_info_text = bug_info_path.read_text(encoding="utf-8", errors="replace")
    py_match = re.search(r"3\.\d+(?:\.\d+)?", bug_info_text)
    if not py_match:
        return False, ""
    python_version = py_match.group(0)
    if python_version.count(".") == 1:
        python_version += ".0"  # conda wants major.minor.patch sometimes

    requirements = requirements_path.read_text(encoding="utf-8", errors="replace") if requirements_path.exists() else ""
    env_name = conda_env_hash(python_version, requirements)

    if not conda_env_exists(env_name):
        ok = conda_env_create(env_name, python_version, log_file)
        if not ok:
            return False, env_name

    rc, out, err = run_in_env(env_name, "bugsinpy-compile", cwd=project_dir, timeout=1200)
    with open(log_file, "ab") as f:
        f.write(f"\n[bugsinpy-compile rc={rc}]\n".encode())
        f.write((out + "\n" + err).encode("utf-8", errors="replace"))

    # Light sanity check: bugsinpy-compile usually echoes "Compile completed" or
    # at least doesn't say "You have not compile". Don't be too strict — some
    # projects produce odd output but still leave a working venv.
    if "You have not compile" in (out + err):
        return False, env_name
    return True, env_name


def run_one_test(
    eval_record: dict,
    raw_generation: str,
    project_dir: Path,
    env_name: str,
    timeout: int,
    stderr_max_chars: int,
) -> dict:
    """Splice patch -> bugsinpy-test -> classify -> restore source. Returns result dict."""
    file_path = project_dir / eval_record["file_path"]
    if not file_path.exists():
        return {
            "compile_pass": False, "test_pass": False,
            "test_status": "error",
            "stderr": f"source file missing: {file_path}",
        }
    original_bytes = file_path.read_bytes()
    try:
        patch = extract_patch(raw_generation, mode="strict")
        patched_fn = reconstruct_patched_function(eval_record["input"], patch)
        original_source = original_bytes.decode("utf-8", errors="replace")
        patched_source = splice_function_into_file(
            file_source=original_source,
            func_start_line=eval_record["func_start_line"],
            func_end_line=eval_record["func_end_line"],
            patched_function=patched_fn,
        )
        file_path.write_text(patched_source, encoding="utf-8")

        rc, out, err = run_in_env(env_name, "bugsinpy-test", cwd=project_dir, timeout=timeout)
        full = (out or "") + "\n" + (err or "")
        if rc == -9:
            status = "timeout"
        else:
            status = classify(full)
        compile_pass = status not in ("error",)  # error often means import failure
        return {
            "compile_pass": bool(compile_pass),
            "test_pass": status == "pass",
            "test_status": status,
            "stderr": full[-stderr_max_chars:] if len(full) > stderr_max_chars else full,
        }
    finally:
        try:
            file_path.write_bytes(original_bytes)
        except Exception:
            pass


# ─────────────────────── main loop ────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="bugsinpy_eval_verified_reproducible.jsonl")
    ap.add_argument("--inference", required=True, help="filtered_inference.jsonl from prefilter step")
    ap.add_argument("--output", required=True, help="output JSONL for tested rows")
    ap.add_argument("--work-root", default="/tmp/bugsinpy_work")
    ap.add_argument("--log-dir", default="/tmp/bugsinpy_logs")
    ap.add_argument("--timeout-test", type=int, default=180)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit to first N bugs (for smoke testing)")
    ap.add_argument("--projects", default=None,
                    help="Comma-separated project names to restrict to (e.g. thefuck,youtube-dl)")
    ap.add_argument("--stderr-max-chars", type=int, default=1000)
    ap.add_argument("--keep-work-dirs", action="store_true",
                    help="Don't delete /tmp/bugsinpy_work/<bug>/ after testing (for debugging)")
    args = ap.parse_args()

    work_root = Path(args.work_root); work_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load eval index by bug_id
    eval_index = {}
    with open(args.eval, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line); eval_index[r["bug_id"]] = r

    # Load filtered inference (only contains bugs/gens that need real testing)
    filtered = []
    with open(args.inference, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                filtered.append(json.loads(line))

    # Bugs with empty generations after prefilter -> skip entirely
    todo = [r for r in filtered if r["generations"]]

    # Apply project restriction if requested
    if args.projects:
        wanted = set(p.strip() for p in args.projects.split(","))
        todo = [r for r in todo if r.get("project") in wanted]

    # Resume support
    done = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r = json.loads(line); done.add((r["bug_id"], r["gen_idx"]))
                except Exception:
                    pass
        if done:
            print(f"[runner] Resuming: {len(done)} (bug, gen) pairs already done", flush=True)

    if args.limit:
        todo = todo[:args.limit]

    n_total_patches = sum(len(r["generations"]) for r in todo)
    print(f"[runner] {len(todo)} bugs, {n_total_patches} patches to (re-)test", flush=True)
    print(f"[runner] Output: {out_path}", flush=True)
    print(f"[runner] Work root (in-container): {work_root}", flush=True)

    out_f = open(out_path, "a", encoding="utf-8")
    t0 = time.time()
    n_run = n_pass = n_fail = n_err = 0

    try:
        for filt_rec in todo:
            bug_id = filt_rec["bug_id"]
            if bug_id not in eval_index:
                print(f"  [SKIP] {bug_id}: not in eval index", flush=True)
                continue

            eval_rec = eval_index[bug_id]
            project = eval_rec["project"]
            bug_num = int(eval_rec["bug_num"])
            orig_indices = filt_rec["orig_indices"]

            # Skip bug entirely if all gens already done
            if all((bug_id, gi) in done for gi in orig_indices):
                continue

            log_file = log_dir / f"{project}_{bug_num}.log"
            print(f"\n[{bug_id}] checkout ...", flush=True)
            project_dir = checkout_buggy(project, bug_num, work_root, log_file)
            if project_dir is None:
                for gi in orig_indices:
                    if (bug_id, gi) in done: continue
                    out_f.write(json.dumps({
                        "bug_id": bug_id, "gen_idx": gi,
                        "compile_pass": False, "test_pass": False,
                        "test_status": "checkout_failed",
                        "stderr": f"see log: {log_file}",
                    }, ensure_ascii=False) + "\n")
                out_f.flush()
                continue

            work_dir = work_root / f"{project}_{bug_num}"
            print(f"[{bug_id}] env + compile ...", flush=True)
            ok, env_name = setup_env_and_compile(project, bug_num, work_dir, project_dir, log_file)
            if not ok:
                for gi in orig_indices:
                    if (bug_id, gi) in done: continue
                    out_f.write(json.dumps({
                        "bug_id": bug_id, "gen_idx": gi,
                        "compile_pass": False, "test_pass": False,
                        "test_status": "compile_failed",
                        "stderr": f"see log: {log_file}",
                    }, ensure_ascii=False) + "\n")
                out_f.flush()
                if not args.keep_work_dirs:
                    shutil.rmtree(work_dir, ignore_errors=True)
                continue

            print(f"[{bug_id}] testing {len(orig_indices)} unique compiling patches ...", flush=True)
            for runner_gi, gen in enumerate(filt_rec["generations"]):
                orig_gi = orig_indices[runner_gi]
                if (bug_id, orig_gi) in done:
                    continue

                result = run_one_test(
                    eval_rec, gen, project_dir, env_name,
                    timeout=args.timeout_test,
                    stderr_max_chars=args.stderr_max_chars,
                )
                row = {"bug_id": bug_id, "gen_idx": orig_gi, **result}
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                n_run += 1
                if result["test_pass"]: n_pass += 1
                elif result["test_status"] == "fail": n_fail += 1
                else: n_err += 1
                marker = "PASS" if result["test_pass"] else ("FAIL" if result["test_status"] == "fail" else "ERR ")
                print(f"  [{marker}] gen[{orig_gi}] -> {result['test_status']}", flush=True)

            if not args.keep_work_dirs:
                try:
                    shutil.rmtree(work_dir)
                except Exception:
                    pass
    finally:
        out_f.close()

    dt = time.time() - t0
    print(f"\n[runner] Done in {dt:.1f}s ({dt / 60:.1f} min)", flush=True)
    print(f"[runner] Tested {n_run} new patches: {n_pass} pass, {n_fail} fail, {n_err} error", flush=True)
    print(f"[runner] Output: {out_path}", flush=True)


if __name__ == "__main__":
    main()
