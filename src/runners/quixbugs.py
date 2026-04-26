#!/usr/bin/env python3
"""
runners/quixbugs.py
-------------------
QuixBugs plausibility runner.

For each generated patch:
  1. Reconstruct the patched function via patcher.reconstruct_patched_function
  2. Build a self-contained Python script: patched_function + tests (assertions)
  3. Run it in a subprocess with a hard timeout
  4. Return {compile_pass, test_pass, test_status, stderr_excerpt}

QuixBugs functions are pure stdlib + (occasionally) `node.py` helper for the
graph/linkedlist algorithms. We provide a minimal `node.py` shim for those.

Status values:
  "pass"     -> all assertions passed (return code 0)
  "fail"     -> assertion failure or wrong output
  "compile"  -> SyntaxError / IndentationError when running the script
  "timeout"  -> process exceeded timeout_sec
  "error"    -> other runtime exception
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Dict, List

from ..patcher import reconstruct_patched_function
from ..postprocess import extract_patch


# Minimal Node class needed by some QuixBugs programs (e.g. depth_first_search,
# breadth_first_search, reverse_linked_list, detect_cycle). The official
# QuixBugs repo ships this; we inline a copy so users don't need to clone it.
NODE_PY = textwrap.dedent(
    '''
    class Node:
        def __init__(self, value=None, successor=None, successors=[],
                     predecessors=[], incoming_nodes=[], outgoing_nodes=[]):
            self.value = value
            self.successor = successor
            self.successors = list(successors) if successors else []
            self.predecessors = list(predecessors) if predecessors else []
            self.incoming_nodes = list(incoming_nodes) if incoming_nodes else []
            self.outgoing_nodes = list(outgoing_nodes) if outgoing_nodes else []

        def successor(self):
            return self.successor

        def successors(self):
            return self.successors

        def predecessors(self):
            return self.predecessors
    '''
).strip()


def _build_test_script(
    patched_function: str,
    tests: str,
) -> str:
    """
    Build a runnable script: imports + Node shim + patched function + tests.
    """
    # Some QuixBugs (sqrt, etc.) need imports already inside `solution`. The
    # patched_function we build is just the function body — for things that
    # need `from heapq import *` etc., the assertions usually still work
    # without imports because they call the function with concrete types. If a
    # bug needs more imports, it's typically `from collections import deque`
    # which the function defines internally. We add a few common imports
    # defensively.

    preamble = textwrap.dedent(
        """
        import sys, math
        from collections import deque
        """
    ).strip()

    return (
        preamble + "\n\n"
        + NODE_PY + "\n\n"
        + patched_function + "\n\n"
        + "# --- inline assertions ---\n"
        + tests + "\n"
        + "print('__OK__')\n"
    )


def _classify_outcome(returncode: int, stderr: str, stdout: str) -> str:
    """Map subprocess result to a status string."""
    if returncode == 0 and "__OK__" in stdout:
        return "pass"
    if any(
        s in stderr
        for s in ("SyntaxError", "IndentationError")
    ):
        return "compile"
    if "AssertionError" in stderr:
        return "fail"
    return "error"


def test_patch(
    bug_record: Dict,
    raw_generation: str,
    timeout_sec: int = 10,
    python_exec: str = sys.executable,
) -> Dict:
    """
    Run one patch attempt. Returns a result dict.

    bug_record: a row from data/quixbugs_eval.jsonl
    raw_generation: raw model output for this patch (we extract OR2 here)
    """
    ir4 = bug_record["input"]
    tests = bug_record.get("tests", "")
    if not tests:
        return {
            "compile_pass": False,
            "test_pass": False,
            "test_status": "skip",
            "stderr": "no tests in eval record",
        }

    patch = extract_patch(raw_generation, mode="strict")
    patched_fn = reconstruct_patched_function(ir4, patch)

    script = _build_test_script(patched_fn, tests)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(script)
        tf_path = tf.name

    try:
        try:
            proc = subprocess.run(
                [python_exec, tf_path],
                capture_output=True, text=True,
                timeout=timeout_sec,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            status = _classify_outcome(proc.returncode, stderr, stdout)
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")
            stderr = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or "")
            status = "timeout"
    finally:
        try:
            Path(tf_path).unlink()
        except OSError:
            pass

    test_pass = status == "pass"
    compile_pass = status not in ("compile",)

    return {
        "compile_pass": bool(compile_pass),
        "test_pass":    bool(test_pass),
        "test_status":  status,
        "stderr":       (stderr or "")[-800:],   # truncate
        "patched_fn":   patched_fn,
    }


# ─────────────────────────────── batch runner ────────────────────────────────


def run_plausibility(
    eval_jsonl: str,
    inference_jsonl: str,
    output_jsonl: str,
    start_bug: int = 0,
    end_bug: int | None = None,
    timeout_sec: int = 10,
    python_exec: str = sys.executable,
    resume: bool = True,
):
    """
    Run plausibility tests for bugs in [start_bug, end_bug) (slice of the eval
    set, in eval-file order).

    Output: one JSON line per (bug_id, generation_idx) with the test result.

    Schema:
        {
          "bug_id": str,
          "gen_idx": int,
          "compile_pass": bool,
          "test_pass":    bool,
          "test_status":  str,
          "stderr":       str (truncated)
        }
    """
    eval_path = Path(eval_jsonl)
    inf_path  = Path(inference_jsonl)
    out_path  = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load eval records keyed by bug_id, preserving file order
    eval_records = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_records.append(json.loads(line))
    eval_index = {r["bug_id"]: r for r in eval_records}

    # Slice
    if end_bug is None:
        end_bug = len(eval_records)
    sliced = eval_records[start_bug:end_bug]
    target_ids = {r["bug_id"] for r in sliced}
    print(f"[quixbugs] Targeting bugs {start_bug}..{end_bug} ({len(target_ids)} bugs)")

    # Load generations index by bug_id
    gens_by_bug = {}
    with open(inf_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            gens_by_bug[rec["bug_id"]] = rec["generations"]

    # Resume support
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
            print(f"[quixbugs] Resuming: {len(done)} (bug, gen) pairs already done")

    out_f = open(out_path, "a", encoding="utf-8")
    t0 = time.time()
    n_run = 0
    n_pass = 0

    try:
        for rec in sliced:
            bug_id = rec["bug_id"]
            if bug_id not in gens_by_bug:
                print(f"  [WARN] no generations for {bug_id}, skipping")
                continue
            generations = gens_by_bug[bug_id]
            for gi, gen in enumerate(generations):
                if (bug_id, gi) in done:
                    continue
                result = test_patch(
                    rec, gen, timeout_sec=timeout_sec, python_exec=python_exec
                )
                # Drop bulky patched_fn before writing
                result.pop("patched_fn", None)
                row = {
                    "bug_id": bug_id,
                    "gen_idx": gi,
                    **result,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                n_run += 1
                if result["test_pass"]:
                    n_pass += 1
            print(
                f"  {bug_id:35s}  pass={sum(1 for _ in range(0))}"
                if False else
                f"  {bug_id:35s}  done"
            )
    finally:
        out_f.close()

    dt = time.time() - t0
    print(f"[quixbugs] Ran {n_run} new (bug, gen) tests in {dt:.1f}s ({n_pass} passed)")
    print(f"[quixbugs] Output: {out_path}")
