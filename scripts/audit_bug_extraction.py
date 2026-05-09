#!/usr/bin/env python3
"""
audit_bug_extraction.py
=======================
Sanity-check that the 161-bug eval represents single-function intra-procedural
bugs. For each bug, parse the BugsInPy `bug_patch.txt` and flag:

  - multi_file:        the patch touches >1 file
  - multi_func_ctx:    hunks have different enclosing functions
  - adds_new_funcs:    patch introduces +def lines naming a function other
                       than the eval row's function_name
  - removes_other_fn:  patch removes -def lines for a function other than
                       function_name (rare; usually means whole-fn rewrite)
  - hunk_outside_span: at least one hunk's NEW line range falls outside the
                       eval's [func_start_line, func_end_line]
  - file_mismatch:     the patch's modified file != eval row's file_path

A bug with NONE of these flags is a clean single-function intra-procedural
patch. Otherwise it is mis-extracted for our IR4 fill-in framing and should
be dropped or re-extracted.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DIFF_FILE_RE = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)")
HUNK_RE = re.compile(
    r"^@@ -(?P<a_start>\d+)(?:,(?P<a_count>\d+))? "
    r"\+(?P<b_start>\d+)(?:,(?P<b_count>\d+))? @@(?P<ctx>.*)$"
)
DEF_LINE_RE = re.compile(r"^[+-]\s*def\s+(?P<name>[A-Za-z_]\w*)\s*\(")


def fn_name_from_ctx(ctx: str) -> str | None:
    """`@@ ... @@ def foo(self, x):` -> `foo`."""
    m = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", ctx)
    if m:
        return m.group(1)
    m = re.search(r"\bclass\s+([A-Za-z_]\w*)", ctx)
    return m.group(1) if m else None


def parse_patch(text: str):
    """Parse unified diff. Returns hunks_by_file with `-` and `+` line ranges
    tracked at the actual (not context) line level."""
    hunks_by_file: dict[tuple[str, str], list[dict]] = {}
    cur_hunk = None
    cur_key = None
    a_lineno = 0  # current a-side line number while walking hunk body
    b_lineno = 0
    for line in text.splitlines():
        m = DIFF_FILE_RE.match(line)
        if m:
            cur_key = (m.group("a"), m.group("b"))
            hunks_by_file.setdefault(cur_key, [])
            cur_hunk = None
            continue
        m = HUNK_RE.match(line)
        if m and cur_key is not None:
            cur_hunk = {
                "a_start": int(m.group("a_start")),
                "a_count": int(m.group("a_count") or 1),
                "b_start": int(m.group("b_start")),
                "b_count": int(m.group("b_count") or 1),
                "ctx": m.group("ctx").strip(),
                "ctx_fn": fn_name_from_ctx(m.group("ctx")),
                "added_fns": [],          # +def names
                "removed_fns": [],         # -def names
                "removed_a_lines": [],     # absolute a-side line numbers of `-`
                "added_b_lines": [],       # absolute b-side line numbers of `+`
                "added_count": 0,
                "removed_count": 0,
            }
            hunks_by_file[cur_key].append(cur_hunk)
            a_lineno = cur_hunk["a_start"]
            b_lineno = cur_hunk["b_start"]
            continue
        if cur_hunk is None or cur_key is None:
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            cur_hunk["added_count"] += 1
            cur_hunk["added_b_lines"].append(b_lineno)
            dm = DEF_LINE_RE.match(line)
            if dm:
                cur_hunk["added_fns"].append(dm.group("name"))
            b_lineno += 1
        elif line.startswith("-"):
            cur_hunk["removed_count"] += 1
            cur_hunk["removed_a_lines"].append(a_lineno)
            dm = DEF_LINE_RE.match(line)
            if dm:
                cur_hunk["removed_fns"].append(dm.group("name"))
            a_lineno += 1
        elif line.startswith(" "):
            a_lineno += 1
            b_lineno += 1
        elif line.startswith("\\"):
            # "\ No newline at end of file" — no line counter change
            pass
    return hunks_by_file


def audit_bug(eval_row: dict, bugsinpy_root: Path) -> dict:
    proj = eval_row["project"]
    bug = eval_row["bug_num"]
    expected_fn = eval_row["function_name"]
    expected_file = eval_row["file_path"]
    fs, fe = eval_row["func_start_line"], eval_row["func_end_line"]

    patch_path = bugsinpy_root / "projects" / proj / "bugs" / str(bug) / "bug_patch.txt"
    if not patch_path.exists():
        return {"bug_id": eval_row["bug_id"], "flags": ["patch_missing"], "patch_path": str(patch_path)}

    text = patch_path.read_text(encoding="utf-8", errors="replace")
    hunks_by_file = parse_patch(text)

    flags = []
    n_files = len(hunks_by_file)
    if n_files > 1:
        flags.append("multi_file")

    # Pick the file the eval row points at (or first if only one)
    best_key = None
    for (a, b) in hunks_by_file:
        # Try to match by suffix — file_path may be "httpie/models.py" while diff has "httpie/models.py"
        if expected_file in (a, b) or a.endswith(expected_file) or b.endswith(expected_file):
            best_key = (a, b)
            break
    if best_key is None and hunks_by_file:
        best_key = next(iter(hunks_by_file))

    if best_key is not None:
        a_path, b_path = best_key
        if expected_file not in (a_path, b_path) and not (
            a_path.endswith(expected_file) or b_path.endswith(expected_file)
        ):
            flags.append("file_mismatch")

    hunks = hunks_by_file.get(best_key, []) if best_key else []
    n_hunks = len(hunks)

    ctx_fns = {h["ctx_fn"] for h in hunks if h["ctx_fn"]}
    if len(ctx_fns) > 1:
        flags.append("multi_func_ctx")

    added_fns = []
    removed_fns = []
    for h in hunks:
        added_fns.extend(h["added_fns"])
        removed_fns.extend(h["removed_fns"])

    extra_added = [f for f in added_fns if f != expected_fn]
    extra_removed = [f for f in removed_fns if f != expected_fn]
    if extra_added:
        flags.append("adds_new_funcs")
    if extra_removed:
        flags.append("removes_other_fn")

    # Real spillover: any ACTUALLY removed line outside [fs, fe] (context lines
    # don't count — only `-` lines we're truly editing)
    removed_outside = []
    for h in hunks:
        for ln in h["removed_a_lines"]:
            if ln < fs or ln > fe:
                removed_outside.append(ln)
    if removed_outside:
        flags.append("removed_outside_span")

    return {
        "bug_id": eval_row["bug_id"],
        "expected_fn": expected_fn,
        "func_span": [fs, fe],
        "n_files": n_files,
        "n_hunks_in_target_file": n_hunks,
        "ctx_fns": sorted(ctx_fns),
        "added_fns": added_fns,
        "removed_fns": removed_fns,
        "removed_outside_span": removed_outside,
        "flags": flags,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/bugsinpy_eval_verified.jsonl")
    ap.add_argument("--bugsinpy-root", default="BugsInPy")
    ap.add_argument("--out", default="results/bugsinpy_eval_audit.jsonl")
    args = ap.parse_args()

    bugsinpy_root = Path(args.bugsinpy_root)
    rows = []
    with open(args.eval, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    audits = [audit_bug(r, bugsinpy_root) for r in rows]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for a in audits:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    flag_counts: Counter = Counter()
    proj_dirty: dict[str, list[str]] = {}
    n_clean = 0
    for r, a in zip(rows, audits):
        proj = r["project"]
        if a["flags"]:
            for fl in a["flags"]:
                flag_counts[fl] += 1
            proj_dirty.setdefault(proj, []).append(a["bug_id"])
        else:
            n_clean += 1

    print(f"Audited {len(audits)} bugs")
    print(f"  clean (single-function intra-proc): {n_clean}")
    print(f"  flagged (mis-extracted candidates): {len(audits) - n_clean}")
    print()
    print("Flag breakdown:")
    for fl, c in flag_counts.most_common():
        print(f"  {fl:25s} {c}")
    print()
    print("Per-project flagged bug counts:")
    for proj in sorted(proj_dirty):
        ids = proj_dirty[proj]
        print(f"  {proj:15s} {len(ids):3d}  ({', '.join(ids[:5])}{'...' if len(ids) > 5 else ''})")
    print()
    print(f"Full audit: {out_path}")


if __name__ == "__main__":
    main()
