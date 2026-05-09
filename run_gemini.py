"""
Generate 10 patches per bug using Google Gemini 2.5 Flash for Snake-RepairLLaMA evaluation.

Produces output JSONL in the SAME schema as the existing baseline / run3 / kimi result
files in results/, so it can be scored by the same metrics module:

    {
      "bug_id":      "...",
      "project":     "...",
      "input":       "<IR4 with <FILL_ME>>",
      "gold_output": "<gold OR2 (the lines that should replace <FILL_ME>)>",
      "generations": ["candidate_1", "candidate_2", ..., "candidate_10"]
    }

Defaults match the existing CodeLlama baseline + Snake-run3 + Kimi runs:
  - QuixBugs:  data/quixbugs_eval.jsonl                (40 bugs)
  - BugsInPy:  data/bugsinpy_eval_verified.jsonl       (161 bugs — same set used
                                                        for baseline + run3 + kimi
                                                        so numbers are comparable)

Outputs:
  - results/quixbugs_gemini.jsonl
  - results/bugsinpy_gemini.jsonl

Resumes automatically — if the output file already has all 10 candidates for a
bug, that bug is skipped. Safe to Ctrl-C and rerun.

Inference protocol matches the RepairLLaMA paper / our other runs:
  - 10 candidates per bug
  - temperature   = 1.0   (sampling for diversity)
  - top_p         = 0.95
  - maxOutputTokens = 256
  - thinkingBudget = 0    (Gemini 2.5 Flash defaults to thinking-on, which wastes
                          tokens and money for short pattern-completion tasks like
                          OR2 fix generation. Disabled.)
  - one HTTP call per candidate (Gemini's `candidateCount` is unreliable across
    tiers and capped at 1 on free tier, so we just loop 10 times like Kimi)

Usage:
    python run_gemini.py                       # both datasets, default paths
    python run_gemini.py --only quixbugs       # quixbugs only
    python run_gemini.py --only bugsinpy       # bugsinpy only
    python run_gemini.py --bugsinpy data/bugsinpy_eval.jsonl   # use the 196-bug set
"""

import argparse
import json
import time
from pathlib import Path

import requests



GEMINI_API_KEY = "hi guys this is a fake key, put your real Gemini API key here to run the script"
MODEL          = "gemini-2.5-flash"
API_URL        = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_API_KEY}"

# ---------------------------------------------------------------------------
# Generation hyperparameters — match kimi / snake-run3 / baseline inference protocol
# ---------------------------------------------------------------------------
NUM_CANDIDATES    = 10
TEMPERATURE       = 1.0
TOP_P             = 0.95
MAX_OUTPUT_TOKENS = 256
THINKING_BUDGET   = 0    # disable Gemini 2.5 Flash thinking — see module docstring
REQUEST_TIMEOUT_S = 180

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR    = Path(__file__).resolve().parent
DATA_DIR    = REPO_DIR / "data"
RESULTS_DIR = REPO_DIR / "results"

DEFAULT_QUIXBUGS_INPUT  = DATA_DIR / "quixbugs_eval.jsonl"
DEFAULT_BUGSINPY_INPUT  = DATA_DIR / "bugsinpy_eval_verified.jsonl"
DEFAULT_QUIXBUGS_OUTPUT = RESULTS_DIR / "quixbugs_gemini.jsonl"
DEFAULT_BUGSINPY_OUTPUT = RESULTS_DIR / "bugsinpy_gemini.jsonl"


# ---------------------------------------------------------------------------
# Prompting (same instructions as Kimi run for fair comparison)
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are an expert Python programmer specializing in bug repair. "
    "You will receive a buggy Python function in which the buggy lines are kept as comments "
    "(prefixed with `# Buggy code:` and `# `) and a single `<FILL_ME>` token marks the location "
    "where the fix must be inserted.\n\n"
    "Your task: output ONLY the lines of code that should replace `<FILL_ME>` — nothing else.\n"
    "Rules:\n"
    "  - Do NOT include any explanation, markdown formatting, or code fences.\n"
    "  - Do NOT repeat the rest of the function.\n"
    "  - Do NOT include the `# Buggy code:` comments or the `<FILL_ME>` token.\n"
    "  - Preserve the indentation level that matches the surrounding code.\n"
    "  - Output raw Python code only."
)


def build_user_prompt(ir4: str) -> str:
    return (
        "Fix the bug in this Python function. The buggy lines are shown as comments. "
        "Return ONLY the code that should replace `<FILL_ME>` — no explanation, no markdown.\n\n"
        f"```\n{ir4}\n```\n\n"
        "Replacement code for `<FILL_ME>`:"
    )


# ---------------------------------------------------------------------------
# Response cleaning — strip markdown fences and stray tokens the model may add
# ---------------------------------------------------------------------------
def clean_response(text: str) -> str:
    if not text:
        return ""
    stripped = text.strip()

    # Strip leading/trailing markdown fences
    if stripped.startswith("```python"):
        stripped = stripped[len("```python"):].lstrip("\n")
    elif stripped.startswith("```"):
        stripped = stripped[3:].lstrip("\n")
    if stripped.endswith("```"):
        stripped = stripped[:-3].rstrip("\n")

    # Drop any line that is the FILL_ME marker itself or a buggy-code comment
    cleaned_lines = []
    for line in stripped.split("\n"):
        s = line.strip()
        if s == "<FILL_ME>":
            continue
        if s.startswith("# Buggy code:"):
            continue
        cleaned_lines.append(line)

    out = "\n".join(cleaned_lines)
    if out and not out.endswith("\n"):
        out += "\n"
    return out


# ---------------------------------------------------------------------------
# Gemini HTTP call
# ---------------------------------------------------------------------------
def call_gemini_once(user_prompt: str) -> tuple[int, dict]:
    body = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": user_prompt}]}
        ],
        "generationConfig": {
            "temperature":     TEMPERATURE,
            "topP":            TOP_P,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "thinkingConfig":  {"thinkingBudget": THINKING_BUDGET},
        },
    }
    try:
        resp = requests.post(API_URL, json=body, timeout=REQUEST_TIMEOUT_S)
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, {"error": {"message": f"non-JSON body: {resp.text[:200]}"}}
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        return -1, {"error": {"message": f"network: {e}"}}


def extract_gemini_text(body: dict) -> str:
    """Pull the model output text from a Gemini response, robust to missing fields."""
    try:
        cand = body["candidates"][0]
    except (KeyError, IndexError):
        return ""
    # Some responses (safety blocks, MAX_TOKENS-on-thinking) lack 'parts'
    content = cand.get("content", {}) or {}
    parts   = content.get("parts", []) or []
    chunks  = [p.get("text", "") for p in parts if isinstance(p, dict)]
    return "".join(chunks)


def generate_one_candidate(user_prompt: str, max_attempts: int = 8) -> str:
    """Call Gemini once with retry-on-rate-limit. Returns the cleaned candidate string,
    or empty string if all attempts fail."""
    for attempt in range(max_attempts):
        status, body = call_gemini_once(user_prompt)

        if status == 200:
            raw = extract_gemini_text(body)
            return clean_response(raw)

        if status == 429:
            # Gemini free tier: 15 RPM for 2.5 Flash. Sleep aggressively.
            wait = min(20 * (attempt + 1), 60)
            msg  = body.get("error", {}).get("message", "")[:60]
            print(f" [429 rate-limit; sleep {wait}s; {msg}]", end="", flush=True)
            time.sleep(wait)
            continue

        if status == -1:
            backoff = min(5 * (attempt + 1), 30)
            print(f" [net err; sleep {backoff}s]", end="", flush=True)
            time.sleep(backoff)
            continue

        if status >= 500 and attempt < max_attempts - 1:
            backoff = min(5 * (attempt + 1), 30)
            print(f" [{status} server err; sleep {backoff}s]", end="", flush=True)
            time.sleep(backoff)
            continue

        # Non-retriable client error
        err = body.get("error", {}).get("message", "unknown")[:120]
        print(f" [http {status}: {err}]", end="", flush=True)
        return ""

    return ""


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def load_done_bug_ids(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = rec.get("bug_id")
            if bid and len(rec.get("generations", [])) == NUM_CANDIDATES:
                done.add(bid)
    return done


# ---------------------------------------------------------------------------
# Main per-dataset loop
# ---------------------------------------------------------------------------
def run_dataset(input_path: Path, output_path: Path, label: str) -> None:
    if not input_path.exists():
        print(f"[{label}] ERROR: input file not found: {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    done_ids = load_done_bug_ids(output_path)
    total = len(samples)

    print()
    print(f"=== {label} ===")
    print(f"Input file : {input_path}")
    print(f"Output file: {output_path}")
    print(f"Total bugs : {total}    Already done: {len(done_ids)}    Remaining: {total - len(done_ids)}")
    print(f"Candidates per bug: {NUM_CANDIDATES}    Model: {MODEL}    Temp: {TEMPERATURE}    Top-p: {TOP_P}    Thinking: {THINKING_BUDGET}")
    print()

    quick_exact = 0
    t0 = time.time()

    for idx, sample in enumerate(samples, start=1):
        bug_id  = sample["bug_id"]
        project = sample.get("project", "")
        ir4     = sample["input"]
        gold    = sample["output"]

        if bug_id in done_ids:
            print(f"[{idx:3d}/{total}] {bug_id:40s}  [cached]")
            continue

        user_prompt = build_user_prompt(ir4)
        gens = []
        print(f"[{idx:3d}/{total}] {bug_id:40s}", end="", flush=True)

        for c in range(NUM_CANDIDATES):
            patch = generate_one_candidate(user_prompt)
            gens.append(patch)
            mark = "*" if patch.strip() == gold.strip() else "."
            print(mark, end="", flush=True)
            time.sleep(0.3)

        if any(g.strip() == gold.strip() for g in gens):
            quick_exact += 1

        record = {
            "bug_id":      bug_id,
            "project":     project,
            "input":       ir4,
            "gold_output": gold,
            "generations": gens,
        }
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        done_ids.add(bug_id)
        elapsed = time.time() - t0
        print(f"  done ({elapsed:.0f}s elapsed)")

    print()
    print(f"=== {label} complete ===")
    print(f"Wrote      : {output_path}")
    print(f"Quick exact (any of {NUM_CANDIDATES} candidates == gold): {quick_exact} this-run")
    print(f"Total time : {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["quixbugs", "bugsinpy", "both"], default="both",
                        help="Which dataset(s) to run (default: both).")
    parser.add_argument("--quixbugs", type=Path, default=DEFAULT_QUIXBUGS_INPUT,
                        help=f"QuixBugs input JSONL (default: {DEFAULT_QUIXBUGS_INPUT.relative_to(REPO_DIR)})")
    parser.add_argument("--bugsinpy", type=Path, default=DEFAULT_BUGSINPY_INPUT,
                        help=f"BugsInPy input JSONL (default: {DEFAULT_BUGSINPY_INPUT.relative_to(REPO_DIR)})")
    parser.add_argument("--quixbugs-out", type=Path, default=DEFAULT_QUIXBUGS_OUTPUT,
                        help=f"QuixBugs output JSONL (default: {DEFAULT_QUIXBUGS_OUTPUT.relative_to(REPO_DIR)})")
    parser.add_argument("--bugsinpy-out", type=Path, default=DEFAULT_BUGSINPY_OUTPUT,
                        help=f"BugsInPy output JSONL (default: {DEFAULT_BUGSINPY_OUTPUT.relative_to(REPO_DIR)})")
    args = parser.parse_args()

    if args.only in ("quixbugs", "both"):
        run_dataset(args.quixbugs, args.quixbugs_out, "QuixBugs")

    if args.only in ("bugsinpy", "both"):
        run_dataset(args.bugsinpy, args.bugsinpy_out, "BugsInPy")


if __name__ == "__main__":
    main()
