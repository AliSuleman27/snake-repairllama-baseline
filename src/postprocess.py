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
from typing import List, Optional


_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)

# Top-level constructs that suggest the model started a NEW definition
# (i.e. moved past the fix). Indent must be 0.
_NEW_TOP_RE = re.compile(
    r"^(def |class |import |from |@|if __name__)"
)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)
    return text


def _strict_extract(text: str) -> str:
    """
    Take lines from `text` until one of these stop conditions:
      - 2+ consecutive blank lines (signals "ran out of fix content")
      - A column-0 line that begins a NEW top-level construct
        (def/class/import/from/@/if __name__)
      - End of text

    This deliberately allows lines that dedent below the first line's indent —
    real OR2 patches often include both deeper-indented inner-loop content and
    shallower-indented sibling lines (e.g. shunting_yard, knapsack).
    """
    lines = text.splitlines(keepends=True)
    out: List[str] = []
    blank_run = 0

    for ln in lines:
        is_blank = ln.strip() == ""
        if is_blank:
            blank_run += 1
            if blank_run >= 2 and out:
                break
            out.append(ln)
            continue
        blank_run = 0

        # New top-level construct after we already have content -> stop
        if (
            out
            and not ln.startswith((" ", "\t"))
            and _NEW_TOP_RE.match(ln)
        ):
            break

        out.append(ln)

    # Trim trailing blank lines
    while out and out[-1].strip() == "":
        out.pop()
    return "".join(out)


def extract_patch(
    generation: str,
    mode: str = "strict",
    post_context_anchor: Optional[str] = None,
) -> str:
    """
    Extract the predicted OR2 (the fix) from a raw model generation.

    mode='strict':
        Conservative — stop at 2 blank lines or a new top-level construct.
        Used for exact_match / ast_match scoring.

    mode='lenient':
        Wider — fenced code block if present, else first ~30 lines.
        Used for "buried fix" detection.

    post_context_anchor (optional):
        A line that should appear immediately AFTER the patch. If provided,
        we cut the generation at the first occurrence of this line. This is
        the most reliable way to delimit the patch when the model echoes the
        post-context after the fix.
    """
    text = generation
    text = _strip_fences(text)
    text = text.lstrip("\n")

    if post_context_anchor:
        anchor = post_context_anchor.rstrip("\n")
        if anchor.strip():
            idx = text.find(anchor)
            if idx >= 0:
                text = text[:idx]

    lines = text.splitlines(keepends=True)

    if mode == "lenient":
        return "".join(lines[:30])

    return _strict_extract(text)


def normalize_for_match(s: str) -> str:
    """Normalize whitespace for exact-match comparison."""
    # Strip trailing whitespace per line, drop trailing blank lines
    lines = [ln.rstrip() for ln in s.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)