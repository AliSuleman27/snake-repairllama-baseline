#!/usr/bin/env python3
"""
inference.py
------------
Generate N candidate patches per bug from a base CausalLM (no adapter).

Following the RepairLLaMA paper inference protocol:
  - 8-bit quantization (BitsAndBytesConfig)
  - Sampling: do_sample=True, temperature=1.0, top_p=0.95  (NOT beam search)
  - max_new_tokens = 256
  - 10 candidates per bug

Usage (from notebook or CLI):
    from src.inference import run_inference
    run_inference(
        eval_jsonl="data/quixbugs_eval.jsonl",
        output_jsonl="results/quixbugs_codellama_baseline.jsonl",
        model_name="codellama/CodeLlama-7b-Python-hf",
        n_samples=10,
    )
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm.auto import tqdm


def _load_model_and_tokenizer(model_name: str, load_in_8bit: bool):
    """Return (model, tokenizer) ready for inference."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    print(f"[inference] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    dtype = torch.float16
    if load_in_8bit:
        print("[inference] Using 8-bit quantization (bitsandbytes)")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

    print(f"[inference] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=dtype if quant_config is None else None,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def run_inference(
    eval_jsonl: str,
    output_jsonl: str,
    model_name: str = "codellama/CodeLlama-7b-Python-hf",
    n_samples: int = 10,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    load_in_8bit: bool = True,
    limit: Optional[int] = None,
    resume: bool = True,
):
    """
    Generate `n_samples` patches per bug from `eval_jsonl` and append each result
    as one JSON line to `output_jsonl`.

    Each output record:
        {
          "bug_id": ...,
          "input": <IR4 prompt>,
          "gold_output": <ground truth OR2>,
          "generations": [str, str, ..., str]    # length n_samples
        }

    `resume=True` skips bug_ids already present in the output file.
    """
    eval_path = Path(eval_jsonl)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load eval set
    with open(eval_path, "r", encoding="utf-8") as f:
        bugs = [json.loads(line) for line in f if line.strip()]
    print(f"[inference] Loaded {len(bugs)} bugs from {eval_path}")

    if limit is not None:
        bugs = bugs[:limit]
        print(f"[inference] Limited to first {len(bugs)} bugs")

    # Resume support
    done_ids = set()
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    done_ids.add(rec["bug_id"])
                except Exception:
                    pass
        if done_ids:
            print(f"[inference] Resuming: {len(done_ids)} bugs already done")

    todo = [b for b in bugs if b["bug_id"] not in done_ids]
    if not todo:
        print("[inference] Nothing to do.")
        return

    model, tokenizer = _load_model_and_tokenizer(model_name, load_in_8bit)

    t0 = time.time()
    out_f = open(out_path, "a", encoding="utf-8")

    try:
        for bug in tqdm(todo, desc="bugs"):
            ir4 = bug["input"]
            gold = bug["output"]

            inputs = tokenizer(ir4, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            generations = []
            with torch.no_grad():
                for _ in range(n_samples):
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    new_tokens = out[0, input_len:]
                    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    generations.append(text)

            rec = {
                "bug_id":      bug["bug_id"],
                "project":     bug.get("project"),
                "input":       ir4,
                "gold_output": gold,
                "generations": generations,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
    finally:
        out_f.close()

    dt = time.time() - t0
    print(f"[inference] Done in {dt:.1f}s ({dt / max(len(todo), 1):.1f}s/bug)")
    print(f"[inference] Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--model", default="codellama/CodeLlama-7b-Python-hf")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--no-8bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run_inference(
        eval_jsonl=args.eval_jsonl,
        output_jsonl=args.output_jsonl,
        model_name=args.model,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_8bit=not args.no_8bit,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
