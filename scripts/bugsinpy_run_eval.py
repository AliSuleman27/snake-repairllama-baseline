#!/usr/bin/env python3
"""
bugsinpy_run_eval.py
====================
Targeted plausibility runner for BugsInPy.

For each bug in `--eval` x `--inference`:
  1. `bugsinpy-checkout -v 0` into /tmp/bugsinpy_work/<project>_<bug>/
  2. md5(python_version + cleaned requirements) -> conda env (cached)
  3. `bugsinpy-compile`
  4. Splice the GOLD fix (eval row's `output` / spliced reference patch)
     into the source, run `bugsinpy-test`, write one row to --gold-output.
  5. Dedup the bug's generations; for each unique compiling-or-not patch:
     splice into source, `bugsinpy-test`, write one row to --output.
     Propagate to dedup duplicates.

Failure modes are RECORDED, not skipped:
  - checkout_failed: write checkout_failed rows to BOTH gold and gen JSONLs.
  - compile_failed:  write compile_failed rows to BOTH gold and gen JSONLs.
  - test errors during gold OR gen runs are kept as-is (status='error').

Designed to run inside the bugsinpy-setup docker image with this repo
bind-mounted at /work. Resumable on both files.

Host launch (PowerShell):
  docker run --rm `
      -v "D:\\snake-repairllama-baseline:/work" `
      -v bugsinpy_envs:/opt/conda/envs `
      bugsinpy-setup:latest `
      python /work/scripts/bugsinpy_run_eval.py `
          --eval /work/data/bugsinpy_eval_verified.jsonl `
          --inference /work/results/bugsinpy_snakellama_run3.jsonl `
          --output /work/results/bugsinpy_run3_gen.jsonl `
          --gold-output /work/results/bugsinpy_run3_gold.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
WORK_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORK_ROOT))

from src.patcher import reconstruct_patched_function, splice_function_into_file  # noqa: E402
from src.postprocess import extract_patch  # noqa: E402

BUGSINPY_DIR = Path("/home/user/BugsInPy")
ERROR_KEYWORDS = (
    "= ERRORS =",
    "ImportError while loading",
    ": command not found",
    "You have not compile this project",
)
FAIL_KEYWORDS = ("= FAILURES =", "FAILED (")
PASS_RE = re.compile(r"\b(?:passed|OK)\b", re.IGNORECASE)


def classify(output: str) -> str:
    if any(k in output for k in ERROR_KEYWORDS):
        return "error"
    if any(k in output for k in FAIL_KEYWORDS):
        return "fail"
    if PASS_RE.search(output):
        return "pass"
    return "error"


def env_hash_via_bash(req_path: Path, python_version: str) -> str:
    """Replicate bugsinpy-testall's hash byte-for-byte by running the same
    bash pipeline. Some BugsInPy requirements.txt files are UTF-16 LE with
    BOM (Windows-edited), and decoding them in Python before hashing produces
    a different digest than `cat | md5sum` on the raw bytes. Shelling out
    guarantees identity.
    """
    bash_cmd = f"""
        TMP=$(mktemp)
        if [ -f "{req_path}" ]; then cp "{req_path}" "$TMP"; else : > "$TMP"; fi
        sed -i -e '/^\\s*#.*$/d' -e '/^\\s*$/d' "$TMP"
        dos2unix "$TMP" >/dev/null 2>&1 || true
        cat <(echo "{python_version}") "$TMP" | md5sum | cut -d' ' -f 1
        rm -f "$TMP"
    """
    r = subprocess.run(["bash", "-c", bash_cmd], capture_output=True, text=True, timeout=30)
    return (r.stdout or "").strip()


def conda_env_exists(env_name: str) -> bool:
    """Check by directory existence on disk — more reliable than parsing
    `conda env list`, whose output isn't always immediately consistent after
    a fresh create. Race-prone substring match was causing false negatives →
    `conda create` then fails with `CondaValueError: prefix already exists`."""
    return Path(f"/opt/conda/envs/{env_name}").is_dir()


def run_bash(cmd: str, cwd: Path | None = None, timeout: int = 180,
             env_name: str | None = None) -> Tuple[int, str]:
    prefix = ". /opt/conda/etc/profile.d/conda.sh\n"
    if env_name:
        prefix += f"conda activate {env_name}\n"
    if cwd:
        prefix += f"cd {cwd}\n"
    script = prefix + cmd + "\n"
    try:
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        err = e.stderr or ""
        if isinstance(out, bytes): out = out.decode("utf-8", "replace")
        if isinstance(err, bytes): err = err.decode("utf-8", "replace")
        return -9, (out or "") + (err or "") + f"\n[TIMEOUT after {timeout}s]"


def parse_python_version(bug_info_text: str) -> str | None:
    m = re.search(r"3\.\d+(?:\.\d+)?", bug_info_text)
    if not m:
        return None
    v = m.group(0)
    return v if v.count(".") == 2 else f"{v}.0"


def splice(ev: dict, file_source: str, patch_or_gen: str) -> str | None:
    """Splice an extracted patch or model gen into the file source.
    Returns the patched-file source, or None if reconstruction failed."""
    try:
        patch = extract_patch(patch_or_gen, mode="strict")
        patched_fn = reconstruct_patched_function(ev["input"], patch)
        return splice_function_into_file(
            file_source=file_source,
            func_start_line=ev["func_start_line"],
            func_end_line=ev["func_end_line"],
            patched_function=patched_fn,
        )
    except Exception:
        return None


def splice_gold(ev: dict, file_source: str) -> str | None:
    """Gold output is just the OR2 line(s) — no fences, no extraction needed.
    Wrap it in the IR4 framing and splice."""
    try:
        patched_fn = reconstruct_patched_function(ev["input"], ev["output"])
        return splice_function_into_file(
            file_source=file_source,
            func_start_line=ev["func_start_line"],
            func_end_line=ev["func_end_line"],
            patched_function=patched_fn,
        )
    except Exception:
        return None


def dedup_generations(generations, ev: dict, file_source: str):
    """Group generations by spliced-file content. Returns:
      reps:    list of (rep_gen_idx, patched_source_or_None)
      dup_map: list aligned with `generations`; dup_map[i] = rep_gen_idx for gen i."""
    seen: dict = {}
    reps = []
    dup_map = [None] * len(generations)
    for gi, gen in enumerate(generations):
        patched_source = splice(ev, file_source, gen)
        key = ("OK", patched_source) if patched_source is not None else ("ERR", gi)
        if key in seen:
            dup_map[gi] = seen[key]
        else:
            seen[key] = gi
            dup_map[gi] = gi
            reps.append((gi, patched_source))
    return reps, dup_map


def checkout(project: str, bug_num: int, work_root: Path, log_f) -> Tuple[Path | None, str]:
    """Returns (project_dir, last_log_tail). project_dir=None if checkout failed."""
    work_dir = work_root / f"{project}_{bug_num}"
    project_dir = work_dir / project
    if project_dir.exists() and (project_dir / ".git").exists():
        return project_dir, ""
    work_dir.mkdir(parents=True, exist_ok=True)
    rc, out = run_bash(
        f"bugsinpy-checkout -p {project} -v 0 -i {bug_num} -w {work_dir}",
        timeout=300,
    )
    log_f.write(f"\n[checkout {project}/{bug_num} rc={rc}]\n{out[-2000:]}\n"); log_f.flush()
    if project_dir.exists():
        return project_dir, ""
    return None, out[-1000:]


def setup_env_and_compile(project: str, bug_num: int, project_dir: Path, log_f) -> Tuple[bool, str, str]:
    """Returns (ok, env_name, last_log_tail).

    Skips bugsinpy-compile entirely if a `bugsinpy_compile_flag` from a previous
    run is already present in project_dir — this is what makes re-running on a
    different inference file (e.g., another model's generations) fast.
    """
    bug_meta = BUGSINPY_DIR / "projects" / project / "bugs" / str(bug_num)
    bi_path = bug_meta / "bug.info"
    if not bi_path.exists():
        return False, "", f"bug.info missing at {bi_path}"
    bug_info = bi_path.read_text(encoding="utf-8", errors="replace")
    py = parse_python_version(bug_info)
    if not py:
        return False, "", f"no 3.x.y in bug.info:\n{bug_info[:500]}"
    req_path = bug_meta / "requirements.txt"
    env_name = env_hash_via_bash(req_path, py)
    if not env_name or len(env_name) != 32:
        return False, "", f"env_hash failed: got {env_name!r}"
    log_f.write(f"\n[env] python={py} hash={env_name}\n"); log_f.flush()

    flag = project_dir / "bugsinpy_compile_flag"
    skip_compile = flag.exists() and conda_env_exists(env_name)
    if skip_compile:
        log_f.write(f"\n[compile cached] flag exists; skipping bugsinpy-compile\n"); log_f.flush()

    if not conda_env_exists(env_name):
        rc, out = run_bash(
            f"conda create -n {env_name} -y python={py} pytest",
            timeout=600,
        )
        log_f.write(f"\n[conda create rc={rc}]\n{out[-2000:]}\n"); log_f.flush()
        if rc != 0:
            return False, env_name, out[-1000:]

    if not skip_compile:
        # Compile timeout: pandas in particular spends ~30 min building C extensions
        # via `python setup.py install`. 1 hour gives plenty of headroom; smaller
        # projects finish in seconds and don't pay this cost.
        rc, out = run_bash("bugsinpy-compile", cwd=project_dir, timeout=3600, env_name=env_name)
        log_f.write(f"\n[bugsinpy-compile rc={rc}]\n{out[-3000:]}\n"); log_f.flush()

        # The authoritative signal that bugsinpy-compile finished is the
        # bugsinpy_compile_flag file it writes at the very end. If that file is
        # missing, bugsinpy-test will refuse to run with "You have not compile this
        # project". Treat its absence as a hard compile failure regardless of rc.
        if rc != 0 or not flag.exists():
            reason = f"bugsinpy-compile rc={rc}, flag_exists={flag.exists()}"
            log_f.write(f"\n[compile failed] {reason}\n"); log_f.flush()
            return False, env_name, reason + "\n" + out[-1000:]

    # ---------------- pinned-dependencies installation ----------------
    # 48/161 of the BugsInPy requirements.txt files are UTF-16 LE with BOM
    # (edited on Windows). bugsinpy-compile's `cat | xargs -I {} pip install`
    # silently mangles those — none of the pinned versions get installed.
    # We re-do the install in Python: detect encoding, decode, pip install -r.
    # This is idempotent for the UTF-8 majority (already installed by
    # bugsinpy-compile -> no-op).
    src_req = bug_meta / "requirements.txt"
    has_pinned_deps = False
    if src_req.exists():
        raw = src_req.read_bytes()
        text = None
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or raw[:3] == b"\xef\xbb\xbf":
            try:
                text = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig")
            except Exception:
                text = raw.decode("utf-8", "replace")
        else:
            text = raw.decode("utf-8", "replace")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [l.strip() for l in text.split("\n") if l.strip() and not l.strip().startswith("#")]
        if lines:
            has_pinned_deps = True
            converted = project_dir / "bugsinpy_requirements_utf8.txt"
            converted.write_text("\n".join(lines) + "\n", encoding="utf-8")
            # Pass 1 — with-deps: bring in any transitive dep that the
            # requirements file forgot to list explicitly. Without this
            # scrapy/2 breaks because its Twisted needs deps not pinned.
            rc_r1, out_r1 = run_bash(
                f"cat {converted} | xargs -I {{}} pip install {{}} 2>&1 | tail -30",
                cwd=project_dir, timeout=1800, env_name=env_name,
            )
            log_f.write(f"\n[pip install pinned pass1 rc={rc_r1}, n={len(lines)}]\n{out_r1[-800:]}\n")

            # Pass 2 — force-reinstall --no-deps: needed when the requirements.txt
            # has pins that pip's resolver upgrades away during pass 1. Affected
            # projects (so far):
            #   fastapi: pins starlette==0.12.9 alongside fastapi==0.55.1, but
            #            fastapi 0.55.1 declares starlette>=0.13.4 → upgraded.
            #   scrapy:  pins Twisted==20.3.0, attrs==19.3.0, but transitive
            #            deps from later packages pull in Twisted 25.x / attrs 25.x.
            #   sanic:   has `-e git+https://...#egg=sanic` for the project itself
            #            plus version pins that get upgraded by deps.
            # Pass 2 snaps every package back to its exact pin ignoring resolver
            # constraints. Skipped for other projects to avoid the reinstall cost.
            if project in {"fastapi", "scrapy", "sanic"}:
                rc_r2, out_r2 = run_bash(
                    f"cat {converted} | xargs -I {{}} pip install --force-reinstall --no-deps {{}} 2>&1 | tail -30",
                    cwd=project_dir, timeout=1800, env_name=env_name,
                )
                log_f.write(f"\n[pip install pinned pass2 (fastapi force --no-deps) rc={rc_r2}]\n{out_r2[-800:]}\n")

            # scrapy ships pinned attrs==19.3.0 in some bugs, but modern Twisted
            # (which transitive deps pull in even with pin) imports `from attrs
            # import frozen` — that module didn't exist until attrs 22.2.0.
            # Force-upgrade attrs for scrapy to make Twisted import.
            if project == "scrapy":
                rc_a, out_a = run_bash(
                    "pip install --upgrade --no-deps --force-reinstall 'attrs>=22.2.0' 2>&1 | tail -10",
                    cwd=project_dir, timeout=300, env_name=env_name,
                )
                log_f.write(f"\n[scrapy attrs>=22.2.0 force-install rc={rc_a}]\n{out_a[-500:]}\n")

            # luigi/sanic — modern setuptools' vendored typeguard imports
            # `is_typeddict` from typing_extensions (added in 4.1.0) AND uses
            # `entry_points(group=...)` which needs importlib_metadata>=4.0.
            # Pinned older versions of either break ALL pytest invocations at
            # plugin-load time. Force-upgrade both.
            if project in {"luigi", "sanic"}:
                rc_t, out_t = run_bash(
                    "pip install --upgrade --no-deps --force-reinstall "
                    "'typing_extensions>=4.1.0' 'importlib_metadata>=4.0' 2>&1 | tail -10",
                    cwd=project_dir, timeout=300, env_name=env_name,
                )
                log_f.write(f"\n[{project} typing_extensions+importlib_metadata upgrade rc={rc_t}]\n{out_t[-500:]}\n")
            log_f.flush()
            # NB: we keep going even on failure — some packages have been deleted from
            # PyPI / yanked. The ones that DID install give us the closest reproducible
            # environment we can achieve.

    # ---------------- editable install of the project under test ----------------
    # bugsinpy-compile may have installed the package via setup.sh's
    # `python setup.py install` (writing an .egg into site-packages that
    # SHADOWS our source-tree patches). Force-reinstall as editable so source
    # edits go live. If we just installed pinned deps, pass --no-deps so we
    # don't disturb them; otherwise let pip pull setup.py's install_requires
    # (matters for projects with empty requirements like matplotlib that need
    # numpy from setup.py).
    no_deps = "--no-deps" if has_pinned_deps else ""
    rc_e, out_e = run_bash(
        f"pip install -e . --force-reinstall {no_deps} 2>&1 | tail -30",
        cwd=project_dir, timeout=900, env_name=env_name,
    )
    log_f.write(f"\n[pip install -e . rc={rc_e}, no_deps={bool(no_deps)}]\n{out_e[-1500:]}\n"); log_f.flush()
    # If editable install failed, keep going — source-tree edits MAY still take
    # effect via PYTHONPATH override at test time, since the test's cwd will
    # have <project_dir>/<package>/ visible before site-packages.

    # matplotlib has C extensions (ft2font, _path, etc.) that must be compiled
    # in-place per project_dir. `pip install -e .` runs setup.py develop which
    # USUALLY does this, but when the conda env is shared across multiple
    # matplotlib bugs the previously-built extensions in a SIBLING work_dir
    # confuse the import (ft2font.so lives next to a different __init__.py).
    # Force a fresh build_ext --inplace so each project_dir has its own .so
    # files ready for tests run from that dir.
    if project == "matplotlib":
        rc_b, out_b = run_bash(
            "python setup.py build_ext --inplace 2>&1 | tail -20",
            cwd=project_dir, timeout=1500, env_name=env_name,
        )
        log_f.write(f"\n[matplotlib build_ext --inplace rc={rc_b}]\n{out_b[-1500:]}\n"); log_f.flush()

    return True, env_name, ""


def run_test(project_dir: Path, file_path: Path, original_bytes: bytes,
             patched_source: str | None, env_name: str, timeout: int) -> dict:
    if patched_source is None:
        return {"test_pass": False, "test_status": "extract_failed",
                "stderr_tail": "extract_patch / reconstruct raised"}
    file_path.write_text(patched_source, encoding="utf-8")
    try:
        # Force PYTHONPATH so source-tree changes win over any stale egg in
        # site-packages. The editable reinstall in setup_env_and_compile is the
        # primary fix; this is a belt-and-braces second line of defense.
        rc, out = run_bash(
            f"PYTHONPATH={project_dir}:${{PYTHONPATH:-}} bugsinpy-test",
            cwd=project_dir, timeout=timeout, env_name=env_name,
        )
        status = "timeout" if rc == -9 else classify(out)
        return {"test_pass": status == "pass", "test_status": status,
                "stderr_tail": out[-1000:]}
    finally:
        file_path.write_bytes(original_bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--inference", required=True)
    ap.add_argument("--output", required=True, help="Per-(bug,gen_idx) gen test results JSONL")
    ap.add_argument("--gold-output", required=True, help="Per-bug gold test results JSONL")
    ap.add_argument("--work-root", default="/tmp/bugsinpy_work")
    ap.add_argument("--log-dir", default="/tmp/bugsinpy_logs")
    ap.add_argument("--timeout-test", type=int, default=180)
    ap.add_argument("--projects", default=None,
                    help="Comma-separated project names (e.g. thefuck,httpie)")
    ap.add_argument("--bug-ids", default=None,
                    help="Comma-separated bug_ids to restrict to (e.g. pandas/1,pandas/2). "
                         "Applied AFTER --projects filter.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N bugs (after project filter)")
    ap.add_argument("--no-keep-work-dirs", action="store_true",
                    help="Delete /tmp/bugsinpy_work/<bug>/ after each bug. Default is "
                         "to KEEP, so re-runs against other inference files reuse the "
                         "checkout + compiled state.")
    args = ap.parse_args()

    work_root = Path(args.work_root); work_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path = Path(args.gold_output); gold_path.parent.mkdir(parents=True, exist_ok=True)

    eval_idx = {}
    with open(args.eval, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line); eval_idx[r["bug_id"]] = r

    inf_rows = []
    with open(args.inference, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                inf_rows.append(json.loads(line))

    todo = [r for r in inf_rows if r["bug_id"] in eval_idx]
    if args.projects:
        wanted = {p.strip() for p in args.projects.split(",")}
        todo = [r for r in todo if eval_idx[r["bug_id"]]["project"] in wanted]
    if args.bug_ids:
        wanted_ids = {b.strip() for b in args.bug_ids.split(",")}
        todo = [r for r in todo if r["bug_id"] in wanted_ids]
    if args.limit:
        todo = todo[: args.limit]

    gen_done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r = json.loads(line); gen_done.add((r["bug_id"], r["gen_idx"]))
                except Exception:
                    pass

    gold_done = set()
    if gold_path.exists():
        with open(gold_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r = json.loads(line); gold_done.add(r["bug_id"])
                except Exception:
                    pass

    n_total_gens = sum(len(r["generations"]) for r in todo)
    print(f"[runner] {len(todo)} bugs | {n_total_gens} generations | "
          f"{len(gen_done)} gen rows + {len(gold_done)} gold rows already done", flush=True)
    print(f"[runner] gen output:  {out_path}", flush=True)
    print(f"[runner] gold output: {gold_path}", flush=True)

    out_f = open(out_path, "a", encoding="utf-8")
    gold_f = open(gold_path, "a", encoding="utf-8")
    t0 = time.time()
    n_tested = n_pass = n_fail = n_err = 0
    n_gold_pass = n_gold_fail = n_gold_err = 0

    def write_gen_row(bug_id, gi, dedup_rep, status, test_pass, stderr_tail, stage):
        out_f.write(json.dumps({
            "bug_id": bug_id, "gen_idx": gi, "dedup_rep": dedup_rep,
            "test_pass": bool(test_pass), "test_status": status,
            "stderr_tail": stderr_tail, "stage": stage,
        }) + "\n")

    def write_gold_row(bug_id, project, bug_num, status, test_pass, stderr_tail, stage):
        gold_f.write(json.dumps({
            "bug_id": bug_id, "project": project, "bug_num": bug_num,
            "gold_test_pass": bool(test_pass), "gold_test_status": status,
            "stderr_tail": stderr_tail, "stage": stage,
        }) + "\n")

    try:
        for ir, inf in enumerate(todo):
            bug_id = inf["bug_id"]
            ev = eval_idx[bug_id]
            project = ev["project"]
            bug_num = int(ev["bug_num"])
            gens = inf["generations"]
            n_gens = len(gens)

            need_gold = bug_id not in gold_done
            need_gen_idxs = [gi for gi in range(n_gens) if (bug_id, gi) not in gen_done]
            if not need_gold and not need_gen_idxs:
                continue

            log_path = log_dir / f"{project}_{bug_num}.log"
            log_f = open(log_path, "a", encoding="utf-8")
            print(f"\n[{ir+1}/{len(todo)}] {bug_id}", flush=True)

            project_dir, ck_err = checkout(project, bug_num, work_root, log_f)
            if project_dir is None:
                tail = f"checkout failed; see {log_path}\n{ck_err}"
                if need_gold:
                    write_gold_row(bug_id, project, bug_num,
                                   "checkout_failed", False, tail, "checkout_failed")
                for gi in need_gen_idxs:
                    write_gen_row(bug_id, gi, gi, "checkout_failed", False, tail, "checkout_failed")
                out_f.flush(); gold_f.flush(); log_f.close()
                continue

            file_path = project_dir / ev["file_path"]
            if not file_path.exists():
                tail = f"{ev['file_path']} not in checkout"
                if need_gold:
                    write_gold_row(bug_id, project, bug_num,
                                   "file_missing", False, tail, "file_missing")
                for gi in need_gen_idxs:
                    write_gen_row(bug_id, gi, gi, "file_missing", False, tail, "file_missing")
                out_f.flush(); gold_f.flush(); log_f.close()
                continue
            file_source = file_path.read_text(encoding="utf-8", errors="replace")
            original_bytes = file_path.read_bytes()

            reps, dup_map = dedup_generations(gens, ev, file_source)
            print(f"  dedup: {n_gens} gens -> {len(reps)} unique", flush=True)
            log_f.write(f"\n[dedup] {n_gens} -> {len(reps)} unique. dup_map={dup_map}\n")

            ok, env_name, comp_err = setup_env_and_compile(project, bug_num, project_dir, log_f)
            if not ok:
                tail = f"compile failed; see {log_path}\n{comp_err}"
                if need_gold:
                    write_gold_row(bug_id, project, bug_num,
                                   "compile_failed", False, tail, "compile_failed")
                for gi in need_gen_idxs:
                    write_gen_row(bug_id, gi, dup_map[gi], "compile_failed", False, tail, "compile_failed")
                out_f.flush(); gold_f.flush(); log_f.close()
                if args.no_keep_work_dirs:
                    shutil.rmtree(work_root / f"{project}_{bug_num}", ignore_errors=True)
                continue

            if need_gold:
                gold_patched = splice_gold(ev, file_source)
                gold_res = run_test(project_dir, file_path, original_bytes,
                                    gold_patched, env_name, timeout=args.timeout_test)
                write_gold_row(bug_id, project, bug_num,
                               gold_res["test_status"], gold_res["test_pass"],
                               gold_res["stderr_tail"], "tested")
                gold_f.flush()
                gm = "PASS" if gold_res["test_pass"] else (
                    "FAIL" if gold_res["test_status"] == "fail" else "ERR ")
                print(f"  [{gm}] gold -> {gold_res['test_status']}", flush=True)
                if gold_res["test_pass"]: n_gold_pass += 1
                elif gold_res["test_status"] == "fail": n_gold_fail += 1
                else: n_gold_err += 1

            rep_results: dict = {}
            for rep_gi, patched_source in reps:
                if (bug_id, rep_gi) in gen_done:
                    continue
                result = run_test(
                    project_dir, file_path, original_bytes,
                    patched_source, env_name, timeout=args.timeout_test,
                )
                rep_results[rep_gi] = result
                marker = "PASS" if result["test_pass"] else (
                    "FAIL" if result["test_status"] == "fail" else "ERR ")
                print(f"  [{marker}] gen[{rep_gi}] -> {result['test_status']}", flush=True)
                n_tested += 1
                if result["test_pass"]: n_pass += 1
                elif result["test_status"] == "fail": n_fail += 1
                else: n_err += 1

            for gi in need_gen_idxs:
                rep_gi = dup_map[gi]
                rep = rep_results.get(rep_gi)
                if rep is None:
                    continue
                write_gen_row(
                    bug_id, gi, rep_gi,
                    rep["test_status"] if gi == rep_gi else rep["test_status"] + "_via_dedup",
                    rep["test_pass"],
                    rep["stderr_tail"] if gi == rep_gi else "",
                    "tested",
                )
            out_f.flush(); log_f.close()

            if args.no_keep_work_dirs:
                shutil.rmtree(work_root / f"{project}_{bug_num}", ignore_errors=True)
    finally:
        out_f.close(); gold_f.close()

    dt = time.time() - t0
    print(f"\n[runner] Done in {dt:.1f}s ({dt/60:.1f} min)", flush=True)
    print(f"[runner] Gen patches: {n_tested} tested -> {n_pass} pass / {n_fail} fail / {n_err} error", flush=True)
    print(f"[runner] Gold tests: {n_gold_pass} pass / {n_gold_fail} fail / {n_gold_err} error", flush=True)


if __name__ == "__main__":
    main()
