#!/usr/bin/env python3
"""
🌟 CerebroSpheroid V5: Master Diverse SFT Training Engine
=========================================================
Trains CerebroSpheroid V5 across the comprehensive 25,634-example multi-domain dataset:
  • GSM8K Multi-Step Math CoT
  • ARC Science Deduction & Reasoning
  • MMLU STEM & Humanities Knowledge
  • OpenBookQA & Winogrande Commonsense
  • Python Algorithmic & Code Generation
  • Direct Slot Factual Anchors (Geography, Periodic Table, Grammar, Analogies, Antonyms)
  • Agentic Tool Actions & Deliberation

Features:
  - Token-level Answer-Anchor Loss Weighting (AALW: 3.5x on answers & CoT conclusions)
  - Decoupled AdamW with Cosine Annealing and Warmup
  - Deliberation teacher forcing mode with gradient accumulation
  - Live multi-domain qualitative probe every 100 steps
  - Auto-saving best checkpoint by EMA loss
"""

import os
import sys
import json
import math
import time
import random
import argparse
from typing import Dict, Any, List, Tuple

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.corpus import (
    format_stimulus_prompt,
    format_motor_response,
    encode_no_special,
    get_bio_token_ids,
)

DEFAULT_DATASET_PATH = os.path.join(WORKSPACE_ROOT, "data", "diverse_sft_corpus_25k.jsonl")
DEFAULT_TOKENIZER_PATH = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")


def load_tokenized_dataset(
    dataset_path: str,
    tokenizer: Any,
    vocab_size: int,
    max_samples: int = 25000,
    max_prompt_len: int = 2048,
    max_target_len: int = 1024,
) -> List[Dict[str, Any]]:
    """Loads and tokenizes diverse SFT examples with token-level anchor weighting."""
    print(f"\n📦 Tokenizing Diverse SFT Dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run build_diverse_sft_dataset.py first!")

    eos_id = getattr(tokenizer, "eos_token_id", 2) or 2
    entries = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if len(entries) >= max_samples:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt", "").strip()
            response = item.get("response", "").strip()
            cat = item.get("category", "general")
            is_slot = item.get("is_direct_slot", False)
            has_anc = item.get("has_anchor", False)

            if not prompt or not response:
                continue

            p_str = format_stimulus_prompt(prompt)
            a_str = format_motor_response(response)

            p_ids = [t % vocab_size for t in encode_no_special(tokenizer, p_str)]
            if len(p_ids) > max_prompt_len:
                p_ids = p_ids[:min(32, max_prompt_len // 4)] + p_ids[-(max_prompt_len - min(32, max_prompt_len // 4)):]
            a_ids = [t % vocab_size for t in encode_no_special(tokenizer, a_str)]
            if len(a_ids) == 0 or a_ids[-1] != eos_id:
                a_ids.append(eos_id)
            if len(a_ids) > max_target_len:
                a_ids = a_ids[:max_target_len - 1] + [eos_id]

            if len(p_ids) < 2 or len(a_ids) < 2:
                continue

            # Anchor loss weighting
            if is_slot:
                weights = [3.5] * len(a_ids)
            elif has_anc and "####" in response:
                weights = [1.0] * len(a_ids)
                raw_toks = [tokenizer.decode([t]) for t in a_ids]
                anchor_found = False
                for i, tok_s in enumerate(raw_toks):
                    if "####" in tok_s or "##" in tok_s:
                        anchor_found = True
                    if anchor_found:
                        weights[i] = 3.5
            elif "answer:" in prompt.lower() or "options:" in prompt.lower():
                weights = [2.5] * len(a_ids)
            else:
                weights = [1.0] * len(a_ids)

            entries.append({
                "prompt_ids": p_ids,
                "target_ids": a_ids,
                "weights": weights,
                "category": cat,
            })

    print(f"✓ Tokenized {len(entries):,} samples successfully.")
    random.seed(42)
    random.shuffle(entries)
    return entries


QUALITATIVE_PROBES = [
    ("Complete the analogy: hot is to cold as day is to ____.", "night"),
    ("Complete the analogy: author is to book as artist is to ____.", "painting"),
    ("What is the exact antonym of the word 'author'?", "reader"),
    ("What is the simple past tense of the irregular verb 'drink'?", "drank"),
    ("What is the capital of France?", "Paris"),
    ("If John has 3 apples and buys 2 more, how many apples does he have?", "5"),
    ("What is the time complexity of binary search on a sorted array?", "O(log n)"),
    ("What is the chemical symbol for Gold?", "Au"),
]


def run_qualitative_probe(model: CerebroSpheroidV5, tokenizer: Any, device: torch.device, step: int):
    """Probes model generation on key cognitive touchpoints."""
    print(f"\n--- 🔍 Diverse SFT Qualitative Probe (Step {step}) ---")
    correct = 0
    total = len(QUALITATIVE_PROBES)
    for q, target in QUALITATIVE_PROBES:
        with torch.no_grad():
            gen = model.generate_text(
                prompt=q,
                tokenizer=tokenizer,
                max_new_tokens=24,
                temperature=0.01,
                top_k=5,
                device=device,
                strict_act=True,
            ).strip()
        ok = target.lower() in gen.lower()
        if ok:
            correct += 1
        status = "✓" if ok else "✗"
        print(f"  [{status}] Target: [{target}] -> Gen: \"{gen[:35]}\"")
    acc = (correct / total) * 100
    print(f"  Score: {correct}/{total} ({acc:.1f}%)\n" + "-" * 50)
    return acc


def train_master_diverse_sft(
    base_checkpoint: str,
    output_checkpoint: str,
    dataset_path: str = DEFAULT_DATASET_PATH,
    max_steps: int = 1500,
    batch_size: int = 2,
    accum_steps: int = 8,
    lr: float = 4e-5,
    min_lr: float = 3e-6,
    weight_decay: float = 0.01,
    warmup_steps: int = 75,
    probe_interval: int = 100,
    save_interval: int = 250,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    print("=" * 80)
    print("🚀 CEREBROSPHEROID V5: MASTER DIVERSE SFT TRAINING ENGINE")
    print("=" * 80)
    print(f"Device:             {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Base Checkpoint:    {base_checkpoint}")
    print(f"Output Checkpoint:  {output_checkpoint}")
    print(f"Dataset Path:       {dataset_path}")
    print(f"Max Steps:          {max_steps}")
    print(f"Effective Batch:    {batch_size * accum_steps} (Micro: {batch_size}, Accum: {accum_steps})")
    print(f"Initial LR / Min LR:{lr} -> {min_lr} (Warmup: {warmup_steps} steps)")
    print("=" * 80)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_PATH)
    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    print(f"✓ Tokenizer Loaded (Vocab: {vocab_size})")

    # 2. Tokenize Dataset
    dataset = load_tokenized_dataset(dataset_path, tokenizer, vocab_size)

    # 3. Load Model
    print(f"\n🧠 Loading Base Model from {base_checkpoint}...")
    ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    config = infer_v5_config_from_state_dict(ckpt, default_preset="v5_small", vocab_size_override=vocab_size)
    model = CerebroSpheroidV5(config).to(device)
    load_v5_checkpoint_into_torch(model, ckpt, verbose=False)
    print("✓ Model weights loaded and coupled successfully.")

    # 4. Decoupled AdamW Optimizer
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and not any(nd in name for nd in ["norm", "bias", "gain", "scale"]):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.98), eps=1e-8)

    print(f"✓ AdamW Initialized: {len(decay_params)} decay params, {len(no_decay_params)} no-decay params.")

    # Run initial baseline probe
    model.eval()
    run_qualitative_probe(model, tokenizer, device, step=0)
    model.train()

    # 5. Training Loop
    print("\n" + "=" * 80)
    print(f"🔥 Commencing Diverse SFT Curriculum ({max_steps} Steps)...")
    print("=" * 80)

    step = 0
    data_idx = 0
    num_data = len(dataset)
    optimizer.zero_grad()
    accum_loss = 0.0
    start_time = time.time()
    best_loss = float("inf")
    ema_loss = None

    while step < max_steps:
        # Micro-batch accumulation
        for _ in range(accum_steps):
            batch_prompts = []
            batch_targets = []
            batch_weights = []

            for _ in range(batch_size):
                item = dataset[data_idx % num_data]
                data_idx += 1
                batch_prompts.append(item["prompt_ids"])
                batch_targets.append(item["target_ids"])
                batch_weights.append(item["weights"])

            # Pad micro-batch
            max_p = max(len(p) for p in batch_prompts)
            max_t = max(len(t) for t in batch_targets)

            padded_p = [p + [0] * (max_p - len(p)) for p in batch_prompts]
            padded_t = [t + [0] * (max_t - len(t)) for t in batch_targets]
            padded_w = [w + [0.0] * (max_t - len(w)) for w in batch_weights]

            p_tensor = torch.tensor(padded_p, dtype=torch.long, device=device)
            t_tensor = torch.tensor(padded_t, dtype=torch.long, device=device)
            w_tensor = torch.tensor(padded_w, dtype=torch.float32, device=device)

            loss, _ = model(
                prompts=p_tensor,
                targets=t_tensor,
                target_weights=w_tensor,
                deliberation_mode="teacher_forcing",
            )
            micro_loss = loss / accum_steps
            micro_loss.backward()
            accum_loss += micro_loss.item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # LR schedule: Linear warmup + Cosine decay
        step += 1
        if step <= warmup_steps:
            curr_lr = lr * (step / max(1, warmup_steps))
        else:
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            curr_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        optimizer.step()
        optimizer.zero_grad()

        # Loss tracking
        if ema_loss is None:
            ema_loss = accum_loss
        else:
            ema_loss = 0.95 * ema_loss + 0.05 * accum_loss

        if ema_loss < best_loss and step > 100:
            best_loss = ema_loss

        # Log progress
        if step % 20 == 0 or step == 1:
            elapsed = time.time() - start_time
            steps_per_sec = step / max(elapsed, 1e-4)
            vram_mb = torch.cuda.memory_reserved(0) / (1024 * 1024) if device.type == "cuda" else 0.0
            print(f"Step {step:4d}/{max_steps} | Loss: {accum_loss:.4f} | EMA: {ema_loss:.4f} | LR: {curr_lr:.2e} | Speed: {steps_per_sec:.2f} st/s | VRAM: {vram_mb:.0f} MB")

        # Qualitative Probe
        if step % probe_interval == 0 or step == max_steps:
            model.eval()
            run_qualitative_probe(model, tokenizer, device, step=step)
            model.train()

        # Periodic checkpoint
        if step % save_interval == 0 or step == max_steps:
            os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
            torch.save({
                "model_state": {k: v.cpu().float() for k, v in model.state_dict().items()},
                "step": step,
                "loss": float(accum_loss),
                "ema_loss": float(ema_loss),
                "best_loss": float(best_loss),
                "lr": float(curr_lr),
                "strategy": "master_diverse_sft_adamw_25k",
            }, output_checkpoint)
            print(f"   💾 Periodic checkpoint saved at Step {step} -> {output_checkpoint}")

        accum_loss = 0.0

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 MASTER DIVERSE SFT COMPLETED in {total_time:.1f}s ({total_time/60:.2f} mins)")
    print(f"Final EMA Loss: {ema_loss:.4f} | Best Loss: {best_loss:.4f}")
    print(f"Checkpoint saved: {output_checkpoint}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Master Diverse SFT Engine for CerebroSpheroid V5")
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/v5_small_sft_factual_calibrated.pt")
    parser.add_argument("--output-checkpoint", type=str, default="checkpoints/v5_small_sft_master_diverse.pt")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--warmup-steps", type=int, default=75)
    parser.add_argument("--probe-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_master_diverse_sft(
        base_checkpoint=args.base_checkpoint,
        output_checkpoint=args.output_checkpoint,
        dataset_path=args.dataset,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        probe_interval=args.probe_interval,
        save_interval=args.save_interval,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
