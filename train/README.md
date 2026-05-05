# Snake-RepairLLaMA Training

Self-contained pipeline for fine-tuning a LoRA adapter on top of `codellama/CodeLlama-7b-hf` using the IR4/OR2 dataset.

## What's here

| File | Purpose |
|------|---------|
| `train_codellama_lora.ipynb` | **Main training notebook** — run this top-to-bottom |
| `data/train.parquet` | 80,597 training examples (IR4 → OR2) |
| `data/validation.parquet` | 8,956 validation examples |
| `data/metadata.json` | Dataset stats and provenance |
| `prepare_data.py` | Re-copy data from a source dir if needed |

## Why `CodeLlama-7b-hf`, not `-Python-hf`

The eval pipeline (notebooks 01–04 in this repo) uses `<FILL_ME>` IR4 prompts. The fast tokenizer expands `<FILL_ME>` into FIM token IDs 32007 / 32008 / 32009. Those IDs only exist in the **base** variant's 32016-row embedding table — the **Python** variant has 32000 rows and crashes on them. Training and eval must use the same base for the prompt format to be consistent.

## How to run (RunPod recommended)

```bash
# 1. Spin up an A100 80GB pod (or 40GB / RTX 4090 — see Hardware below)
# 2. Pick the PyTorch 2.x template, 50 GB ephemeral disk

# Inside the pod:
git clone https://github.com/AliSuleman27/snake-repairllama-baseline.git
cd snake-repairllama-baseline

# 3. Open train/train_codellama_lora.ipynb in JupyterLab
#    Set HF_USERNAME at the top, run all cells.
```

The notebook handles dependencies, data loading, tokenization, training, validation, save, push to HF Hub, and a final sanity-check generation — all visible step by step.

## Hardware / time / cost

| GPU | VRAM | Time/epoch | RunPod $/hr | $ for 1 epoch |
|-----|------|-----------|-------------|---------------|
| A100 80GB | 80 GB | ~60 min | $1.49 | ~$1.50 |
| A100 40GB | 40 GB | ~75 min | $1.39 | ~$1.75 |
| RTX A6000 | 48 GB | ~80 min | $0.49 | ~$0.65 |
| RTX 4090 | 24 GB | ~3 hr | $0.69 | ~$2.10 |

4-bit QLoRA fits everything ≥ 16 GB. The notebook auto-detects but defaults to `PER_DEVICE_BATCH_SIZE=4, GRAD_ACCUM=8` (effective batch 32).

## Output

After running, the adapter is saved to:
- **Local**: `train/output/<adapter_name>/` (gitignored)
- **HF Hub**: `<HF_USERNAME>/snake-repairllama-7b-fim-r16` (private by default)

To evaluate it, edit `notebooks/04_run1_snakellama.ipynb`:
```python
ADAPTER_NAME = "<your-hf-username>/snake-repairllama-7b-fim-r16"
```

and run that notebook to get full QuixBugs + BugsInPy metrics.

## Hyperparameters at a glance

| | This run | Ahsan's prior run | Why changed |
|--|----------|-------------------|-------------|
| Base | `7b-hf` | `7b-Python-hf` | FIM token compatibility |
| LoRA r | 16 | 8 | More capacity, ~$0 cost |
| LoRA alpha | 32 | 16 | Standard 2× rank |
| Target modules | all linear | q_proj, v_proj | Better fine-tune quality |
| LR | 2e-4 | 5e-4 | Standard QLoRA value (5e-4 was high for 2 modules, way too high for 7) |
| Epochs | 1 | 3 | 1 epoch on 80k usually enough; iterate if underfit |
| Quantization | 4-bit NF4 | fp16 | Memory headroom |
| Eval | every 200 steps | every 500 steps | More signal during run |

If the first run looks under-fit (val loss still falling at the end), bump epochs to 2 and re-run.
