#!/usr/bin/env python3
"""
patcher.py
----------
Reconstruct a patched function/file from the IR4 prompt + a generated patch.

Two ops:

reconstruct_patched_function(ir4_input, patch) -> str
    Replace the "# Buggy code:" block + <FILL_ME> in IR4 with the patch.
    Returns the full patched function source.
    Used for QuixBugs (single-function, the function IS the program).

splice_function_into_file(file_source, func_start, func_end, patched_function) -> str
    Replace lines [func_start, func_end] (1-indexed, inclusive) in file_source
    with patched_function.
    Used for BugsInPy (function lives inside a larger file).

Both ops are independent of any HF dataset format — they only need the eval
JSONL row schema (input, buggy_function, func_start_line, func_end_line).
"""

from __future__ import annotations

from typing import List


FILL_ME = "<FILL_ME>"


def reconstruct_patched_function(ir4_input: str, patch: str) -> str:
    """
    Walk through IR4 line-by-line:
      - drop the "# Buggy code:" header
      - drop subsequent lines starting with "#" (commented buggy lines)
      - replace the <FILL_ME> line with the patch
      - keep everything else as-is

    The patch may or may not end with \\n; we ensure it does.
    """
    out: List[str] = []
    in_buggy_block = False

    for line in ir4_input.splitlines(keepends=True):
        stripped = line.strip()

        if stripped == "# Buggy code:":
            in_buggy_block = True
            continue

        if in_buggy_block:
            if FILL_ME in stripped:
                p = patch if patch.endswith("\n") else patch + "\n"
                out.append(p)
                in_buggy_block = False
                continue
            # commented buggy line: "# something" or just "#"
            if stripped.startswith("#"):
                continue
            # Unexpected: end of buggy block without seeing FILL_ME
            in_buggy_block = False
            out.append(line)
        else:
            out.append(line)

    return "".join(out)


def splice_function_into_file(
    file_source: str,
    func_start_line: int,
    func_end_line: int,
    patched_function: str,
) -> str:
    """
    Replace lines [func_start_line, func_end_line] (1-indexed, inclusive) in
    file_source with patched_function.
    """
    lines = file_source.splitlines(keepends=True)

    # Convert to 0-indexed slice indices
    start_idx = max(0, func_start_line - 1)
    end_idx = min(len(lines), func_end_line)

    pf = patched_function
    if pf and not pf.endswith("\n"):
        pf += "\n"

    new_lines = lines[:start_idx] + [pf] + lines[end_idx:]
    return "".join(new_lines)
