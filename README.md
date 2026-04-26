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

**Use Colab Free (T4, 16 GB).** CodeLlama-7B fits in ~7 GB with 8-bit quantization.

1. Open `notebooks/01_baseline_codellama.ipynb` in Colab
2. Add `HF_TOKEN` as a Colab Secret (CodeLlama is gated)
3. Run cells top-to-bottom

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

Aggregated as **Top-1 / Top-3 / Top-10** (Top-K passes if any of the first K patches passes).

## Layout

```
snake-repairllama-baseline/
├── data/                              # IR4/OR2 eval sets (committed)
│   ├── quixbugs_eval.jsonl            # 40 bugs
│   └── bugsinpy_eval.jsonl            # 196 bugs
├── src/
│   ├── build_quixbugs_eval.py         # HF dataset → IR4/OR2 JSONL
│   ├── inference.py                   # generate N patches/bug, save JSONL
│   ├── postprocess.py                 # extract OR2 from raw model output
│   └── metrics.py                     # exact / ast / compile / buried
├── notebooks/
│   └── 01_baseline_codellama.ipynb    # Colab T4 entry point
├── results/                           # generations + scores (gitignored)
└── requirements.txt
```

## Next step (after this baseline)

Retrain the adapter on Vertex AI with paper-aligned config — LoRA r=16, alpha=32, all 7 linear modules, `eval_strategy="steps"`, `load_best_model_at_end=True`. Then re-run this exact notebook with the adapter loaded on top of CodeLlama and compare to the baseline numbers.
