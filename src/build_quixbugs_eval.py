#!/usr/bin/env python3
"""
build_quixbugs_eval.py
----------------------
Download Muennighoff/quixbugs (40 single-function Python bugs) and convert to
the SAME IR4/OR2 JSONL schema used by bugsinpy_eval.jsonl, so a single
inference pipeline works for both benchmarks.

IR4 input format (must match training data exactly):
    <pre-buggy-lines>
    # Buggy code:
    # <buggy line 1>
    # <buggy line 2>
    <FILL_ME>
    <post-buggy-lines>

OR2 output format:
    The fixed lines that replace <FILL_ME>.

Each row of QuixBugs has buggy_program (full buggy fn) + solution (full fix).
We diff them line-by-line, find the contiguous run of changed lines, and
build IR4/OR2 from that.

Usage:
    python src/build_quixbugs_eval.py --output data/quixbugs_eval.jsonl
"""

import argparse
import difflib
import json
from pathlib import Path
from typing import List, Tuple, Optional

FILL_ME = "<FILL_ME>"


def find_change_range(
    buggy_lines: List[str], fixed_lines: List[str]
) -> Optional[Tuple[int, int, List[str]]]:
    """
    Return (first_buggy_idx, last_buggy_idx, replacement_lines).
    Indices are 0-based into buggy_lines (inclusive on both ends).
    replacement_lines is the list of lines that should appear in OR2.

    Returns None if buggy == fixed (no bug detected).

    Strategy: SequenceMatcher opcodes -> collect all 'replace'/'delete'/'insert'
    spans, take the bounding range on the buggy side, and the corresponding
    bounding range on the fixed side as replacement.
    """
    sm = difflib.SequenceMatcher(a=buggy_lines, b=fixed_lines, autojunk=False)
    opcodes = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not opcodes:
        return None

    buggy_starts = [op[1] for op in opcodes]
    buggy_ends = [op[2] for op in opcodes]   # exclusive
    fixed_starts = [op[3] for op in opcodes]
    fixed_ends = [op[4] for op in opcodes]   # exclusive

    b_start = min(buggy_starts)
    b_end = max(buggy_ends)                  # exclusive
    f_start = min(fixed_starts)
    f_end = max(fixed_ends)                  # exclusive

    # Pure-insertion bug: no buggy lines were removed, only added.
    # Anchor on the preceding context line (must exist; QuixBugs all start
    # with `def name(...):`). Treat that line as the "buggy" region; the
    # fix is that same line followed by the inserted lines.
    if b_end <= b_start:
        if b_start == 0:
            # Insertion at the very top — no preceding line to anchor on.
            return None
        anchor_idx = b_start - 1
        replacement = [buggy_lines[anchor_idx]] + fixed_lines[f_start:f_end]
        return anchor_idx, anchor_idx, replacement

    return b_start, b_end - 1, fixed_lines[f_start:f_end]


def build_ir4_or2(
    buggy_program: str, solution: str
) -> Optional[Tuple[str, str, str]]:
    """Return (ir4_input, or2_output, original_buggy_function) or None."""
    buggy_lines = buggy_program.splitlines(keepends=True)
    fixed_lines = solution.splitlines(keepends=True)

    # Ensure all lines end with \n for clean concatenation
    if buggy_lines and not buggy_lines[-1].endswith("\n"):
        buggy_lines[-1] += "\n"
    if fixed_lines and not fixed_lines[-1].endswith("\n"):
        fixed_lines[-1] += "\n"

    rng = find_change_range(buggy_lines, fixed_lines)
    if rng is None:
        return None
    first_idx, last_idx, replacement = rng

    # IR4
    ir4_parts = []
    ir4_parts.extend(buggy_lines[:first_idx])
    ir4_parts.append("# Buggy code:\n")
    for idx in range(first_idx, last_idx + 1):
        line = buggy_lines[idx].rstrip("\n").rstrip("\r")
        ir4_parts.append("# \n" if line == "" else "# " + line + "\n")
    ir4_parts.append(FILL_ME + "\n")
    ir4_parts.extend(buggy_lines[last_idx + 1:])
    ir4 = "".join(ir4_parts)

    # OR2
    or2 = "".join(replacement)

    return ir4, or2, "".join(buggy_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/quixbugs_eval.jsonl")
    parser.add_argument(
        "--dataset", default="Muennighoff/quixbugs",
        help="HuggingFace dataset id (default: Muennighoff/quixbugs)"
    )
    args = parser.parse_args()

    from datasets import load_dataset
    print(f"Loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train")
    print(f"Loaded {len(ds)} rows.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    rows = []

    for i, row in enumerate(ds):
        name = row["name"]
        buggy = row["buggy_program"]
        fixed = row["solution"]
        tests = row.get("tests", "")

        result = build_ir4_or2(buggy, fixed)
        if result is None:
            print(f"  [SKIP] {name}: no diff or pure-insertion")
            skipped += 1
            continue
        ir4, or2, buggy_func = result

        # Function start line is line 1 in QuixBugs (each row is one function)
        func_end = len(buggy_func.splitlines())

        rows.append({
            "bug_id":          f"quixbugs/{name}",
            "project":         "quixbugs",
            "bug_num":         i,
            "file_path":       f"{name}.py",
            "buggy_commit":    "",
            "fixed_commit":    "",
            "test_command":    "",   # tests are inline assertions, see `tests` field
            "function_name":   name,
            "func_start_line": 1,
            "func_end_line":   func_end,
            "num_hunks":       1,
            "num_removed":     0,    # filled below
            "input":           ir4,
            "output":          or2,
            "buggy_function":  buggy_func,
            "tests":           tests,  # extra field for plausible-pass evaluation
        })
        kept += 1

    # Fill num_removed by counting `# ` lines under "# Buggy code:"
    for r in rows:
        ir4_lines = r["input"].splitlines()
        in_block = False
        n = 0
        for ln in ir4_lines:
            if ln.strip() == "# Buggy code:":
                in_block = True
                continue
            if in_block:
                if ln.strip() == FILL_ME:
                    break
                if ln.startswith("#"):
                    n += 1
        r["num_removed"] = n

    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print("=" * 50)
    print(f"Kept    : {kept}")
    print(f"Skipped : {skipped}")
    print(f"Output  : {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
