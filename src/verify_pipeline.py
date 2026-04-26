#!/usr/bin/env python3
"""
verify_pipeline.py
------------------
Verify the patcher + splicer pipeline against ground truth.

For each (sampled) BugsInPy bug:
  1. Read buggy file source from git at buggy_commit
  2. Read fixed file source from git at fixed_commit
  3. Reconstruct the patched function from (IR4 input, gold OR2 output)
  4. Splice it into the buggy file
  5. Diff the spliced result vs the actual fixed file
  6. Report match / near-match / mismatch

A "match" means: if the runner spliced the gold patch into the buggy file at
the test stage, the resulting file would be byte-equivalent to what the
project's CI saw when the bug was actually fixed. That's the contract
test_patch_against_bug relies on.

Usage:
    python -m src.verify_pipeline                       # 8 sampled bugs
    python -m src.verify_pipeline --all                  # all 196 bugs
    python -m src.verify_pipeline --bug-id thefuck/12    # one specific bug
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .patcher import reconstruct_patched_function, splice_function_into_file


def git_show(repo_dir: Path, commit: str, file_path: str) -> Optional[str]:
    """Return file contents at a given commit, or None if missing."""
    proc = subprocess.run(
        ["git", "show", f"{commit}:{file_path}"],
        cwd=str(repo_dir),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def normalize(text: str) -> str:
    """Strip trailing whitespace per line and trailing blank lines."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def diff_summary(actual: str, expected: str, context: int = 1) -> str:
    """Compact unified diff (only changed regions)."""
    import difflib
    a = actual.splitlines()
    e = expected.splitlines()
    diff = list(difflib.unified_diff(
        e, a, lineterm="",
        fromfile="actual_fixed", tofile="our_spliced",
        n=context,
    ))
    return "\n".join(diff[:80])  # cap


def verify_one(bug: dict, repos_dir: Path) -> dict:
    project = bug["project"]
    repo_dir = repos_dir / project
    if not (repo_dir / ".git").exists():
        return {"bug_id": bug["bug_id"], "status": "no_repo"}

    file_path = bug["file_path"]
    buggy_source = git_show(repo_dir, bug["buggy_commit"], file_path)
    fixed_source = git_show(repo_dir, bug["fixed_commit"], file_path)
    if buggy_source is None or fixed_source is None:
        return {"bug_id": bug["bug_id"], "status": "git_show_failed"}

    # Normalize line endings as the converter did
    buggy_source = buggy_source.replace("\r\n", "\n").replace("\r", "\n")
    fixed_source = fixed_source.replace("\r\n", "\n").replace("\r", "\n")

    # Reconstruct patched function from IR4 + gold OR2
    patched_fn = reconstruct_patched_function(bug["input"], bug["output"])

    # Splice into the buggy file
    spliced = splice_function_into_file(
        file_source=buggy_source,
        func_start_line=bug["func_start_line"],
        func_end_line=bug["func_end_line"],
        patched_function=patched_fn,
    )

    spliced_n = normalize(spliced)
    fixed_n = normalize(fixed_source)

    if spliced_n == fixed_n:
        return {"bug_id": bug["bug_id"], "status": "exact"}

    # Maybe the diff is purely whitespace within the patched function region
    return {
        "bug_id":  bug["bug_id"],
        "status":  "mismatch",
        "diff":    diff_summary(spliced, fixed_source),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", default="data/bugsinpy_eval.jsonl")
    ap.add_argument("--repos-dir", default="D:/BugsInPy/repos")
    ap.add_argument("--all", action="store_true",
                    help="verify all bugs (default: random sample of 8)")
    ap.add_argument("--bug-id", default=None,
                    help="verify only this bug_id (e.g. 'thefuck/12')")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    eval_path = Path(args.eval_jsonl)
    repos_dir = Path(args.repos_dir)

    bugs = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                bugs.append(json.loads(line))

    if args.bug_id:
        bugs = [b for b in bugs if b["bug_id"] == args.bug_id]
    elif not args.all:
        # Stratified sample: at least one bug per project, capped at 12
        by_project = {}
        for b in bugs:
            by_project.setdefault(b["project"], []).append(b)
        rng = random.Random(args.seed)
        sample = []
        for proj, lst in sorted(by_project.items()):
            sample.append(rng.choice(lst))
        bugs = sample

    print(f"Verifying {len(bugs)} bug(s) ...\n")
    counts = {"exact": 0, "mismatch": 0, "no_repo": 0, "git_show_failed": 0}
    mismatches: List[dict] = []

    for b in bugs:
        result = verify_one(b, repos_dir)
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        marker = {
            "exact": "OK ", "mismatch": "FAIL", "no_repo": "SKIP",
            "git_show_failed": "SKIP",
        }.get(result["status"], "??")
        print(f"  [{marker}] {result['bug_id']}")
        if result["status"] == "mismatch":
            mismatches.append(result)

    print()
    print("=" * 60)
    print(f"  EXACT match    : {counts.get('exact', 0)}")
    print(f"  MISMATCH       : {counts.get('mismatch', 0)}")
    print(f"  SKIPPED        : {counts.get('no_repo', 0) + counts.get('git_show_failed', 0)}")
    print("=" * 60)

    for m in mismatches[:5]:
        print()
        print(f"--- DIFF for {m['bug_id']} ---")
        print(m["diff"])
        print()

    sys.exit(0 if counts.get("mismatch", 0) == 0 else 1)


if __name__ == "__main__":
    main()
