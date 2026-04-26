# Snake-RepairLLaMA — Baseline Evaluation

Baseline numbers for **vanilla `codellama/CodeLlama-7b-Python-hf`** (no LoRA, no fine-tuning) on the same QuixBugs / BugsInPy IR4-OR2 evaluation sets used to benchmark our trained adapter. Establishes the "did fine-tuning actually help?" reference point.

## Why this repo exists

The first Snake-RepairLLaMA training run produced poor numbers on QuixBugs/BugsInPy. We cannot tell whether that means **(a)** training hurt the model or **(b)** the task is just hard for a 7B model — without knowing what the **untrained base model** scores on the same prompts. This repo measures **(b)**.

## Eval sets

Both are converted to the **same IR4 (input) / OR2 (output) JSONL schema** so a single inference + metrics pipeline works for both.

| File | Bugs | Source | Notes |
|---|---|---|---|
| `data/quixbugs_eval.jsonl` | 40 | [`Muennighoff/quixbugs`](https://huggingface.co/datasets/Muennighoff/quixbugs) | Single-function algorithmic bugs. 4 are pure-insertion → anchored on preceding context line. |
| `data/bugsinpy_eval.jsonl` | 196 | BugsInPy, filtered to single-function intra-procedural bugs | Re-used from the prior eval run for direct comparability. |

### Schema (per row)

```json
{
  "bug_id": "quixbugs/bitcount",
  "project": "quixbugs",
  "function_name": "bitcount",
  "input": "<IR4 prompt with # Buggy code: + <FILL_ME>>",
  "output": "<OR2: the lines that replace <FILL_ME>>",
  "buggy_function": "<full original buggy fn>",
  "tests": "..."         // QuixBugs only — assertions, for plausible-pass eval
}
```

### Rebuild QuixBugs from scratch

```bash
python src/build_quixbugs_eval.py --output data/quixbugs_eval.jsonl
```

## Running the baseline

**Use Colab Free.** Three notebooks, run in order:

1. **`01_baseline_codellama.ipynb`** — T4 **GPU** runtime. Generates 10 patches/bug for both datasets. CodeLlama-7B fits in ~7 GB with 8-bit. Add `HF_TOKEN` Colab Secret (CodeLlama is gated). Save outputs to Drive when done.
2. **`02_plausibility_quixbugs.ipynb`** — **CPU** runtime. Reads generations from Drive, runs inline assertions, writes plausibility JSONL. ~3 min total.
3. **`03_plausibility_bugsinpy.ipynb`** — **CPU** runtime. Run `setup_bugsinpy.sh` once per session (~20 min), then test bugs in chunks via `START_BUG`/`END_BUG`. Save plausibility JSONL back to Drive between sessions.

Inference protocol (matches the RepairLLaMA paper):
- 10 candidate patches per bug
- `do_sample=True, temperature=1.0, top_p=0.95`
- `max_new_tokens=256`
- 8-bit via bitsandbytes

## Metrics

Per bug, we score each of the 10 patches on:

- **exact** — generation matches gold OR2 after whitespace normalization
- **ast** — `ast.dump()` of generation == `ast.dump()` of gold (catches reformatting)
- **compile** — generation parses without `SyntaxError`
- **buried** — gold appears as a substring inside the (lenient-extracted) generation. Flags "model knows the fix but can't isolate it cleanly."
- **plausible** — patched program **passes the actual test suite**. This is the gold-standard repair metric — a patch is "plausible" if the project's tests accept it. Computed by separate notebooks (see below).

Aggregated as **Top-1 / Top-3 / Top-10** (Top-K passes if any of the first K patches passes).

## Plausibility testing (the gold-standard metric)

Inference says "model produced text that matches gold". Plausibility says "the patched program actually works". The two notebooks below run the real tests:

| Notebook | Dataset | Test mechanism | Time |
|---|---|---|---|
| `notebooks/02_plausibility_quixbugs.ipynb` | 40 algorithmic bugs | Inline `assert` cases, run in subprocess with timeout | ~3 min total (CPU only) |
| `notebooks/03_plausibility_bugsinpy.ipynb` | 196 real-project bugs | BugsInPy framework: `bugsinpy-checkout` → splice patch → `bugsinpy-compile` → `bugsinpy-test`, with pyenv-managed Pythons | **~1-2 hrs per 25 bugs** (CPU only) |

Both notebooks expose **`START_BUG`, `END_BUG` parameters** so you can chunk the work across multiple Colab sessions:

```python
START_BUG = 0    # i (inclusive)
END_BUG   = 25   # j (exclusive). Resume-on-rerun is automatic.
```

For BugsInPy, run `scripts/setup_bugsinpy.sh` once per fresh Colab session — it installs pyenv + Python 3.6/3.7/3.8 and clones the BugsInPy framework (~15-25 min one-time).

Output schema (`results/<dataset>_<model>_plausibility.jsonl`, one row per (bug, gen)):

```json
{
  "bug_id": "quixbugs/bitcount",
  "gen_idx": 0,
  "compile_pass": true,
  "test_pass":    true,
  "test_status":  "pass",          // pass | fail | compile | timeout | error | checkout_failed | compile_failed
  "stderr":       "..."            // truncated test output
}
```

`evaluate_file(plausibility_jsonl=...)` ingests this and emits Top-K plausible alongside exact/AST/compile.

## Layout

```
snake-repairllama-baseline/
├── data/                                   # IR4/OR2 eval sets (committed)
│   ├── quixbugs_eval.jsonl                 # 40 bugs
│   └── bugsinpy_eval.jsonl                 # 196 bugs
├── src/
│   ├── build_quixbugs_eval.py              # HF dataset → IR4/OR2 JSONL
│   ├── inference.py                        # generate N patches/bug
│   ├── patcher.py                          # splice generated patch back into source
│   ├── postprocess.py                      # extract OR2 from raw model output
│   ├── metrics.py                          # exact / ast / compile / buried / plausible
│   └── runners/
│       ├── quixbugs.py                     # inline-assertion plausibility runner
│       └── bugsinpy.py                     # bugsinpy-framework plausibility runner
├── notebooks/
│   ├── 01_baseline_codellama.ipynb         # T4 GPU: generate patches
│   ├── 02_plausibility_quixbugs.ipynb      # CPU: run QuixBugs tests, slice with i,j
│   └── 03_plausibility_bugsinpy.ipynb      # CPU: run BugsInPy tests, slice with i,j
├── scripts/
│   └── setup_bugsinpy.sh                   # one-time pyenv + framework setup for Colab
├── results/                                # generations + plausibility (gitignored)
└── requirements.txt
```

## Next step (after this baseline)

Retrain the adapter on Vertex AI with paper-aligned config — LoRA r=16, alpha=32, all 7 linear modules, `eval_strategy="steps"`, `load_best_model_at_end=True`. Then re-run this exact notebook with the adapter loaded on top of CodeLlama and compare to the baseline numbers.
