#!/usr/bin/env python3
"""
runners/bugsinpy.py
-------------------
BugsInPy plausibility runner.

Approach:
  - Use BugsInPy's own framework (`bugsinpy-checkout`, `bugsinpy-compile`,
    `bugsinpy-test`) — clone the repo, run the bash scripts via subprocess.
  - Cache the (checkout + compile) work per bug_id; iterate patches by just
    rewriting the source file in-place.
  - Restore original source between runs so 10 different patches all start
    from the same buggy baseline.

Per bug, cost breakdown:
  - bugsinpy-checkout: 5-60s (git clone if not cached, then checkout + apply
    metadata)
  - bugsinpy-compile : 30s - 20min (creates venv, pip-installs project deps —
    pandas/scipy projects are the slow ones)
  - bugsinpy-test    : 5s - 2min (runs the specific test file)

We hard-cap compile and test with subprocess timeouts so one bad bug can't
stall an overnight run.

This runner expects to run on Linux (Colab, WSL, or a Linux VM). It calls
`bash` and uses pyenv-managed Python interpreters for each bug.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..patcher import reconstruct_patched_function, splice_function_into_file
from ..postprocess import extract_patch


# ─────────────────────────────── helpers ─────────────────────────────────────


def _read_bug_info(bugsinpy_dir: Path, project: str, bug_num: int) -> Dict:
    """Parse projects/PROJECT/bugs/N/bug.info into a dict."""
    info = bugsinpy_dir / "projects" / project / "bugs" / str(bug_num) / "bug.info"
    text = info.read_text(encoding="utf-8", errors="replace")
    out = {}
    for line in text.splitlines():
        m = re.match(r'^\s*(\w+)\s*=\s*"?(.*?)"?\s*$', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _bash(cmd: str, cwd: Optional[Path] = None, timeout: int = 600,
          extra_env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """Run a bash command, return (returncode, stdout, stderr). Truncates output."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            cwd=str(cwd) if cwd else None,
            env=env,
            capture_output=True, text=True,
            timeout=timeout,
        )
        return proc.returncode, (proc.stdout or "")[-4000:], (proc.stderr or "")[-4000:]
    except subprocess.TimeoutExpired as e:
        so = e.stdout or b""
        se = e.stderr or b""
        if isinstance(so, (bytes, bytearray)):
            so = so.decode("utf-8", errors="replace")
        if isinstance(se, (bytes, bytearray)):
            se = se.decode("utf-8", errors="replace")
        return -9, so[-4000:], (se + f"\n[TIMEOUT after {timeout}s]")[-4000:]


# ─────────────────────────────── checkout + compile ─────────────────────────


def checkout_bug(
    bugsinpy_dir: Path,
    work_root: Path,
    project: str,
    bug_num: int,
    timeout_sec: int = 300,
) -> Tuple[bool, Path, str]:
    """
    Run `bugsinpy-checkout` to materialize the buggy version at
    work_root/<project>_<bug_num>/<project>/

    Returns (ok, project_dir, log).
    """
    work_dir = work_root / f"{project}_{bug_num}"
    project_dir = work_dir / project

    if project_dir.exists() and (project_dir / ".git").exists():
        return True, project_dir, "[cached]"

    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"export PATH=\"{bugsinpy_dir}/framework/bin:$PATH\" && "
        f"bugsinpy-checkout -p {project} -v 0 -i {bug_num} -w {work_dir}"
    )
    rc, out, err = _bash(cmd, timeout=timeout_sec)
    if rc != 0 or not project_dir.exists():
        return False, project_dir, f"[checkout failed rc={rc}]\n{err}"
    return True, project_dir, out + err


def compile_bug(
    bugsinpy_dir: Path,
    project_dir: Path,
    pyenv_root: Optional[Path] = None,
    timeout_sec: int = 1200,
) -> Tuple[bool, str]:
    """
    Run `bugsinpy-compile` in the checked-out project dir.
    Creates a venv and installs deps. Idempotent — checks for env/bin/python.
    """
    venv_python = project_dir / "env" / "bin" / "python"
    if venv_python.exists():
        return True, "[cached venv]"

    pyenv_init = ""
    if pyenv_root:
        pyenv_init = (
            f"export PYENV_ROOT={pyenv_root} && "
            f"export PATH=\"$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH\" && "
            f'eval "$(pyenv init -)" && '
        )

    cmd = (
        f"{pyenv_init}"
        f"export PATH=\"{bugsinpy_dir}/framework/bin:$PATH\" && "
        f"cd {project_dir} && bugsinpy-compile"
    )
    rc, out, err = _bash(cmd, timeout=timeout_sec)
    if not venv_python.exists():
        return False, f"[compile failed rc={rc}]\n{out[-2000:]}\n{err[-2000:]}"
    return True, "[ok]"


def run_bug_test(
    bugsinpy_dir: Path,
    project_dir: Path,
    timeout_sec: int = 180,
) -> Tuple[bool, str, str]:
    """
    Run `bugsinpy-test` in the project dir. Returns (passed, status, log).

    status in {"pass", "fail", "timeout", "error"}.
    """
    cmd = (
        f"export PATH=\"{bugsinpy_dir}/framework/bin:$PATH\" && "
        f"cd {project_dir} && bugsinpy-test"
    )
    rc, out, err = _bash(cmd, timeout=timeout_sec)
    log = (out + "\n" + err)[-3000:]

    if rc == -9:
        return False, "timeout", log
    if rc == 0 and not re.search(r"FAILED|ERROR\b|AssertionError", out + err):
        return True, "pass", log
    if re.search(r"\bSyntaxError\b|\bIndentationError\b", out + err):
        return False, "compile", log
    return False, "fail", log


# ─────────────────────────────── per-patch test ─────────────────────────────


def test_patch_against_bug(
    bug_record: Dict,
    raw_generation: str,
    bugsinpy_dir: Path,
    project_dir: Path,
    timeout_sec_test: int = 180,
) -> Dict:
    """
    Splice the patch into the source file, run bugsinpy-test, restore source.
    Assumes checkout+compile already done.
    """
    source_path = project_dir / bug_record["file_path"]
    if not source_path.exists():
        return {
            "compile_pass": False, "test_pass": False,
            "test_status": "error",
            "stderr": f"source file missing: {source_path}",
        }

    # Snapshot original buggy source
    original_bytes = source_path.read_bytes()

    try:
        # Build patched file source
        patch = extract_patch(raw_generation, mode="strict")
        patched_fn = reconstruct_patched_function(bug_record["input"], patch)

        original_source = original_bytes.decode("utf-8", errors="replace")
        patched_source = splice_function_into_file(
            file_source=original_source,
            func_start_line=bug_record["func_start_line"],
            func_end_line=bug_record["func_end_line"],
            patched_function=patched_fn,
        )
        source_path.write_text(patched_source, encoding="utf-8")

        passed, status, log = run_bug_test(
            bugsinpy_dir, project_dir, timeout_sec=timeout_sec_test
        )
        return {
            "compile_pass": status != "compile",
            "test_pass":    passed,
            "test_status":  status,
            "stderr":       log,
        }
    finally:
        # Always restore the original source
        try:
            source_path.write_bytes(original_bytes)
        except Exception:
            pass


# ─────────────────────────────── batch runner ────────────────────────────────


def run_plausibility(
    eval_jsonl: str,
    inference_jsonl: str,
    output_jsonl: str,
    bugsinpy_dir: str,
    work_root: str,
    pyenv_root: Optional[str] = None,
    start_bug: int = 0,
    end_bug: int | None = None,
    timeout_sec_compile: int = 1200,
    timeout_sec_test: int = 180,
    resume: bool = True,
    skip_compile_failures: bool = True,
):
    """
    Run plausibility for bugs in [start_bug, end_bug). For each target bug:
      checkout (cached) -> compile (cached) -> for each generation: splice + test

    Output: one JSONL row per (bug_id, gen_idx).

    Setup precondition: bugsinpy_dir exists with framework/bin in it; pyenv has
    the Python versions needed by the targeted bugs (see scripts/setup_bugsinpy.sh).
    """
    eval_path = Path(eval_jsonl)
    inf_path  = Path(inference_jsonl)
    out_path  = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bugsinpy_dir = Path(bugsinpy_dir)
    work_root    = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    pyenv_root_p = Path(pyenv_root) if pyenv_root else None

    # Load eval records (preserve file order)
    eval_records = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_records.append(json.loads(line))

    if end_bug is None:
        end_bug = len(eval_records)
    sliced = eval_records[start_bug:end_bug]
    print(f"[bugsinpy] Targeting bugs {start_bug}..{end_bug} ({len(sliced)} bugs)")

    # Load generations
    gens_by_bug = {}
    with open(inf_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            gens_by_bug[rec["bug_id"]] = rec["generations"]

    # Resume index
    done = set()
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    done.add((r["bug_id"], r["gen_idx"]))
                except Exception:
                    pass
        if done:
            print(f"[bugsinpy] Resuming: {len(done)} (bug, gen) pairs already done")

    out_f = open(out_path, "a", encoding="utf-8")
    t0 = time.time()
    n_run, n_pass = 0, 0

    try:
        for rec in sliced:
            bug_id = rec["bug_id"]
            project = rec["project"]
            bug_num = rec["bug_num"]

            if bug_id not in gens_by_bug:
                print(f"  [SKIP] {bug_id}: no generations available")
                continue

            generations = gens_by_bug[bug_id]
            # Skip bug entirely if all gens already done
            if all((bug_id, gi) in done for gi in range(len(generations))):
                continue

            print(f"\n[{bug_id}] checkout ...", flush=True)
            ok, project_dir, log = checkout_bug(
                bugsinpy_dir, work_root, project, bug_num,
                timeout_sec=300,
            )
            if not ok:
                err_msg = log[-1500:]
                for gi in range(len(generations)):
                    if (bug_id, gi) in done:
                        continue
                    out_f.write(json.dumps({
                        "bug_id": bug_id, "gen_idx": gi,
                        "compile_pass": False, "test_pass": False,
                        "test_status": "checkout_failed",
                        "stderr": err_msg,
                    }) + "\n")
                out_f.flush()
                continue

            print(f"[{bug_id}] compile ...", flush=True)
            ok, log = compile_bug(
                bugsinpy_dir, project_dir, pyenv_root=pyenv_root_p,
                timeout_sec=timeout_sec_compile,
            )
            if not ok:
                err_msg = log[-1500:]
                for gi in range(len(generations)):
                    if (bug_id, gi) in done:
                        continue
                    out_f.write(json.dumps({
                        "bug_id": bug_id, "gen_idx": gi,
                        "compile_pass": False, "test_pass": False,
                        "test_status": "compile_failed",
                        "stderr": err_msg,
                    }) + "\n")
                out_f.flush()
                if skip_compile_failures:
                    continue

            print(f"[{bug_id}] testing {len(generations)} patches ...", flush=True)
            for gi, gen in enumerate(generations):
                if (bug_id, gi) in done:
                    continue
                result = test_patch_against_bug(
                    rec, gen, bugsinpy_dir, project_dir,
                    timeout_sec_test=timeout_sec_test,
                )
                row = {"bug_id": bug_id, "gen_idx": gi, **result}
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                n_run += 1
                if result["test_pass"]:
                    n_pass += 1
                marker = "✓" if result["test_pass"] else "✗"
                print(f"  {marker} gen[{gi}] -> {result['test_status']}")
    finally:
        out_f.close()

    dt = time.time() - t0
    print(f"\n[bugsinpy] Ran {n_run} new (bug, gen) tests in {dt/60:.1f} min ({n_pass} passed)")
    print(f"[bugsinpy] Output: {out_path}")
