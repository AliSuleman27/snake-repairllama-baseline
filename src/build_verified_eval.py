#!/usr/bin/env python3
"""
build_verified_eval.py
----------------------
Filter `bugsinpy_eval.jsonl` to bugs whose gold OR2 patch round-trips correctly:

    splice( reconstruct(IR4, gold_OR2), buggy_file ) == fixed_file   (modulo whitespace)

The original `filter_bugsinpy.py` (in the other repo) has a known bug in its
`build_ir4_or2` reconstruction:
  - For multi-hunk diffs, kept-context lines BETWEEN hunks are dropped from OR2
  - For single-hunk diffs with complex `+`/`-` interleaving, kept-context lines
    INSIDE the hunk can also be dropped

This script verifies each row against ground truth (the actual fixed-commit file
in the locally-cloned BugsInPy repo) and writes only the verified rows to a
new JSONL. The dropped rows are listed in a sidecar text file so we know which
bugs were excluded and why.

Usage:
    python -m src.build_verified_eval                                      # default paths
    python -m src.build_verified_eval --eval-jsonl data/bugsinpy_eval.jsonl \\
        --output data/bugsinpy_eval_verified.jsonl --repos-dir D:/BugsInPy/repos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .patcher import reconstruct_patched_function, splice_function_into_file
from .verify_pipeline import git_show, normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", default="data/bugsinpy_eval.jsonl")
    ap.add_argument("--output",     default="data/bugsinpy_eval_verified.jsonl")
    ap.add_argument("--repos-dir",  default="D:/BugsInPy/repos")
    args = ap.parse_args()

    eval_path = Path(args.eval_jsonl)
    out_path  = Path(args.output)
    repos_dir = Path(args.repos_dir)

    rows = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    kept = []
    dropped = []

    for b in rows:
        repo = repos_dir / b["project"]
        if not (repo / ".git").exists():
            dropped.append((b["bug_id"], "no_repo"))
            continue

        bs = git_show(repo, b["buggy_commit"], b["file_path"])
        fs = git_show(repo, b["fixed_commit"], b["file_path"])
        if bs is None or fs is None:
            dropped.append((b["bug_id"], "git_show_failed"))
            continue

        bs = bs.replace("\r\n", "\n").replace("\r", "\n")
        fs = fs.replace("\r\n", "\n").replace("\r", "\n")

        pf = reconstruct_patched_function(b["input"], b["output"])
        sp = splice_function_into_file(
            bs, b["func_start_line"], b["func_end_line"], pf
        )

        if normalize(sp) == normalize(fs):
            kept.append(b)
        else:
            dropped.append((b["bug_id"], "or2_mismatch"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sidecar = out_path.with_suffix(".dropped.txt")
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(f"# Bugs dropped during verification: {len(dropped)} / {len(rows)}\n")
        f.write(f"# Reason 'or2_mismatch' = gold OR2 in eval JSONL doesn't reconstruct\n")
        f.write(f"# the actual fixed file. Pre-existing bug in upstream filter_bugsinpy.py.\n")
        f.write("#\n")
        for bug_id, reason in sorted(dropped):
            f.write(f"{reason}\t{bug_id}\n")

    print(f"Total bugs   : {len(rows)}")
    print(f"Kept         : {len(kept)}    -> {out_path}")
    print(f"Dropped      : {len(dropped)} -> {sidecar}")


if __name__ == "__main__":
    main()
