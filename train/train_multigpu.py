#!/usr/bin/env python3
"""
train_multigpu.py
-----------------
Multi-GPU LoRA training script for Snake-RepairLLaMA on CodeLlama-7b-hf.

This is the same logic as train_codellama_lora.ipynb, but as a runnable
.py script so we can launch it with torchrun for true Distributed Data
Parallel (DDP) across multiple GPUs.

Launch:
    bash train/launch_multigpu.sh
    # or directly:
    torchrun --nproc_per_node=4 train/train_multigpu.py

Resume from checkpoint:
    RESUME=1 bash train/launch_multigpu.sh

Single-GPU smoke test (for debugging):
    python train/train_multigpu.py
"""
from __future__ import annotations

import os

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


# =============================================================================
# Configuration — edit these and re-launch
# =============================================================================

# Model
BASE_MODEL = "codellama/CodeLlama-7b-hf"

# HF Hub destination for the trained adapter
HF_USERNAME = os.environ.get("HF_USERNAME", "alisuleman525")
HF_ADAPTER_REPO = f"{HF_USERNAME}/snake-repairllama-7b-fim-r16"
LOCAL_OUTPUT_DIR = os.environ.get(
    "LOCAL_OUTPUT_DIR",
    "./train/output/snake-repairllama-7b-fim-r16",
)

# Data
TRAIN_PARQUET = "train/data/train.parquet"
VAL_PARQUET = "train/data/validation.parquet"

# Per-GPU batch. Total effective batch = per_device * grad_accum * num_gpus.
# 4x A5000 (24 GB each): per_device=4 fits comfortably with seq=1024 + no checkpoint.
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM = 2

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Optim
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024
WARMUP_STEPS = 100
LR_SCHEDULER_TYPE = "cosine"

# Eval / save / log
EVAL_STEPS = 200
SAVE_STEPS = 400  # must be a multiple of EVAL_STEPS for load_best_model_at_end
LOGGING_STEPS = 50
SAVE_TOTAL_LIMIT = 2

USE_4BIT = True

# Resume support — set RESUME=1 to continue from latest checkpoint in output dir
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME", "").lower() in ("1", "true", "yes")


# =============================================================================
# Distributed setup
# =============================================================================

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_MAIN = LOCAL_RANK in (-1, 0)


def log(*args, **kwargs):
    """Print only on the main process to keep logs clean."""
    if IS_MAIN:
        print(*args, **kwargs, flush=True)


log(f"World size: {WORLD_SIZE}")
log(f"Local rank: {LOCAL_RANK}")
log(f"Effective batch: {PER_DEVICE_BATCH_SIZE} per-dev × {GRAD_ACCUM} accum × "
    f"{max(WORLD_SIZE, 1)} GPUs = {PER_DEVICE_BATCH_SIZE * GRAD_ACCUM * max(WORLD_SIZE, 1)}")
log(f"Resume from checkpoint: {RESUME_FROM_CHECKPOINT}")


# =============================================================================
# Tokenizer
# =============================================================================

log(f"Loading tokenizer: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Sanity check FIM expansion (catches Python-hf-by-mistake at startup)
sample = "def f():\n# bug\n#  return 1\n<FILL_ME>\n    return 2\n"
ids = tokenizer(sample, add_special_tokens=True).input_ids
assert max(ids) < tokenizer.vocab_size, (
    f"OOB token ID detected (max={max(ids)} vs vocab={tokenizer.vocab_size}). "
    f"Wrong base? Make sure BASE_MODEL is the FIM-supporting variant."
)
log(f"Tokenizer FIM check OK (vocab={tokenizer.vocab_size}, FIM tokens 32007-32009 present)")


# =============================================================================
# Data
# =============================================================================

log("Loading datasets ...")
ds_train = load_dataset("parquet", data_files=TRAIN_PARQUET, split="train")
ds_val = load_dataset("parquet", data_files=VAL_PARQUET, split="train")
log(f"  train: {len(ds_train):,}  val: {len(ds_val):,}")


def tokenize_example(ex):
    in_ids = tokenizer(ex["input"], add_special_tokens=True).input_ids
    out_ids = tokenizer(ex["output"], add_special_tokens=False).input_ids
    eos = tokenizer.eos_token_id

    full = in_ids + out_ids + [eos]
    labels = [-100] * len(in_ids) + out_ids + [eos]

    if len(full) > MAX_SEQ_LENGTH:
        excess = len(full) - MAX_SEQ_LENGTH
        full = full[excess:]
        labels = labels[excess:]

    return {"input_ids": full, "labels": labels, "attention_mask": [1] * len(full)}


# All ranks tokenize but HF's content-addressed cache means only one actually
# computes; the others read from disk. num_proc=4 parallelizes within each rank.
log("Tokenizing ...")
ds_train_tok = ds_train.map(
    tokenize_example, remove_columns=ds_train.column_names,
    num_proc=4, desc="tok train",
)
ds_val_tok = ds_val.map(
    tokenize_example, remove_columns=ds_val.column_names,
    num_proc=4, desc="tok val",
)


# =============================================================================
# Model — 4-bit base + LoRA on top
# =============================================================================

bnb = (
    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    if USE_4BIT
    else None
)

# In DDP, each rank loads the model on its OWN GPU only. Setting
# device_map={"": local_rank} pins the model to this rank's GPU.
# Single-GPU fallback uses device_map="auto".
device_map = {"": LOCAL_RANK} if LOCAL_RANK >= 0 else "auto"

log(f"Loading base model on rank {LOCAL_RANK} (device_map={device_map}) ...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
try:
    model.gradient_checkpointing_disable()
except Exception:
    pass
model.config.use_cache = False  # required during training

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=LORA_TARGET_MODULES,
)
model = get_peft_model(model, lora_cfg)
if IS_MAIN:
    model.print_trainable_parameters()


# =============================================================================
# Training
# =============================================================================

training_args = TrainingArguments(
    output_dir=LOCAL_OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=False,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_steps=WARMUP_STEPS,
    optim="paged_adamw_8bit",
    bf16=True,
    fp16=False,
    tf32=True,                       # Ampere supports it; free matmul speedup

    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    logging_steps=LOGGING_STEPS,
    logging_first_step=True,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="tensorboard",
    push_to_hub=False,               # we push manually after training

    dataloader_pin_memory=True,
    dataloader_num_workers=4,

    # DDP-specific
    ddp_find_unused_parameters=False,  # LoRA has no unused params
    local_rank=LOCAL_RANK,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
    return_tensors="pt",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train_tok,
    eval_dataset=ds_val_tok,
    processing_class=tokenizer,
    data_collator=data_collator,
)


# =============================================================================
# Run
# =============================================================================

log("Starting training ...")
if RESUME_FROM_CHECKPOINT:
    log("  (resuming from latest checkpoint in output_dir)")
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()


# =============================================================================
# Save + push (main process only — pushing from multiple ranks would conflict)
# =============================================================================

if IS_MAIN:
    log(f"Saving adapter to {LOCAL_OUTPUT_DIR} ...")
    trainer.save_model(LOCAL_OUTPUT_DIR)
    tokenizer.save_pretrained(LOCAL_OUTPUT_DIR)

    log(f"Pushing to HF Hub: {HF_ADAPTER_REPO} ...")
    try:
        trainer.model.push_to_hub(HF_ADAPTER_REPO, private=True)
        tokenizer.push_to_hub(HF_ADAPTER_REPO, private=True)
        log(f"Pushed: https://huggingface.co/{HF_ADAPTER_REPO}")
    except Exception as e:
        log(f"Push failed (saved locally is still safe): {e}")

log("Done.")
