#!/usr/bin/env python3
"""
postprocess.py
--------------
Extract the OR2 patch from a raw model generation.

CodeLlama base (no fine-tuning) tends to:
  - Echo part of the input back
  - Continue generating arbitrary code after the fix
  - Sometimes emit a fenced code block

This module trims the raw generation down to "the lines that should replace
<FILL_ME>". Multiple heuristics are tried in order.

The same module also defines a strict-vs-lenient extraction policy:
  - strict: take only what we're confident replaces <FILL_ME>
  - lenient: a wider window (used for "buried fix" detection)
"""

from __future__ import annotations

import re
from typing import List


_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)
    return text


def _take_until_blank(lines: List[str]) -> List[str]:
    """Stop at the first fully blank line after a non-blank line."""
    out: List[str] = []
    seen_nonblank = False
    for ln in lines:
        if ln.strip() == "":
            if seen_nonblank:
                break
            else:
                continue
        seen_nonblank = True
        out.append(ln)
    return out


def _take_indented_block(lines: List[str]) -> List[str]:
    """
    Keep lines that share the indentation of the first non-blank line, plus
    any lines indented MORE deeply. Stop at the first line indented LESS.
    Useful when the OR2 fix is a small indented block.
    """
    out: List[str] = []
    base_indent = None
    for ln in lines:
        if ln.strip() == "":
            if not out:
                continue
            out.append(ln)
            continue
        indent = len(ln) - len(ln.lstrip())
        if base_indent is None:
            base_indent = indent
            out.append(ln)
            continue
        if indent < base_indent:
            break
        out.append(ln)
    # trim trailing blanks
    while out and out[-1].strip() == "":
        out.pop()
    return out


def extract_patch(generation: str, mode: str = "strict") -> str:
    """
    Extract the predicted OR2 (the fix) from a raw model generation.

    mode='strict':
        Conservative — first non-empty indented block, stop at dedent or blank.
        This is what you compare to gold for exact_match / ast_match.

    mode='lenient':
        Wider — entire fenced code block if present, else the first ~30 lines
        of generation. Useful for "is the correct fix BURIED in the output?"
    """
    text = generation
    text = _strip_fences(text)

    # Drop a leading newline (common — model starts with \n after <FILL_ME>)
    text = text.lstrip("\n")

    lines = text.splitlines(keepends=True)

    if mode == "lenient":
        # Just cap to ~30 lines, keep most of the output
        return "".join(lines[:30])

    # strict
    block = _take_indented_block(lines)
    if block:
        return "".join(block)
    # fall back to take_until_blank
    block = _take_until_blank(lines)
    return "".join(block)


def normalize_for_match(s: str) -> str:
    """Normalize whitespace for exact-match comparison."""
    # Strip trailing whitespace per line, drop trailing blank lines
    lines = [ln.rstrip() for ln in s.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)