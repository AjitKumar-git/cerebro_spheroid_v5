#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Local Step-by-Step Reasoning SFT Engine (P=0.50)
======================================================================
Executes local Supervised Fine-Tuning on NVIDIA GPU for the 102M foundation model:
  - Base checkpoint: Step 920.5k foundation weights or existing SFT checkpoint
  - Deliberation Policy: Scheduled Deliberation with Sampling P = 0.50
    (50% internal thought feedback via motor efference / 50% teacher-forced ground truth)
  - Modality: Pure Native Text Output with Natural Step-by-Step Reasoning Scratchpad:
    * Math/Logic/Science: Plain-text derivation before final answer (No <think> or <act> tags)
    * Factual/Conversational: Direct, fluent answers (Doty persona)
  - Dataset: 30,000 verified samples (train: 27.5k, val: 2.5k)
  - Effective Batch Size: Micro 4 × Gradient Accumulation 8 = 32
  - Target Steps: 1,500 steps with continuous best-checkpoint tracking & seamless resume
"""

import os
import sys

# Prevent CUDA memory fragmentation and enable TF32
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import json
import math
import argparse
import random
import re
from typing import Dict, Any, List, Tuple
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.corpus import (
    build_tokenizer,
    format_stimulus_prompt,
    encode_no_special,
)

DEFAULT_BASE_CKPT = os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_latest_step920k.pt")
DEFAULT_OUTPUT_CKPT = os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_reasoning_sft_p05.pt")
DEFAULT_TRAIN_DATA = os.path.join(WORKSPACE_ROOT, "data", "cerebro_v5_custom_dataset", "train.parquet")
DEFAULT_VAL_DATA = os.path.join(WORKSPACE_ROOT, "data", "cerebro_v5_custom_dataset", "validation.parquet")
DEFAULT_TOKENIZER_DIR = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")


def load_tokenized_dataset(
    parquet_path: str,
    tokenizer: Any,
    vocab_size: int,
    max_samples: int = 30000,
    max_prompt_len: int = 160,
    max_target_len: int = 160,
) -> List[Dict[str, Any]]:
    print(f"📦 Loading & tokenizing step-by-step reasoning dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)

    entries = []
    eos_id = getattr(tokenizer, "eos_token_id", 2) or 2
    reasoning_count = 0
    direct_count = 0

    for idx, r in df.iterrows():
        prompt = str(r["prompt"]).strip()
        answer = str(r["answer"]).strip()
        if not prompt or not answer:
            continue

        thinking = str(r["thinking"]).strip() if r.get("thinking") is not None and not pd.isna(r["thinking"]) else ""
        # Clean GSM8K calculator tags like <<3*3=9>> into clean natural text
        thinking = re.sub(r'<<.*?>>', '', thinking).strip()

        # Step-by-step reasoning target formulation:
        # If thinking is available and non-trivial, prepend it to provide compute tokens before the answer.
        # Otherwise, output direct answer.
        if thinking and thinking.lower() != "none" and len(thinking) > 5:
            full_target = f"{thinking}\n\n{answer}"
            reasoning_count += 1
        else:
            full_target = answer
            direct_count += 1

        p_str = format_stimulus_prompt(prompt)
        p_ids = [t % vocab_size for t in encode_no_special(tokenizer, p_str)][:max_prompt_len]
        a_ids = [t % vocab_size for t in encode_no_special(tokenizer, full_target)][:max_target_len]

        # Append EOS to target
        if not a_ids or a_ids[-1] != eos_id:
            a_ids.append(eos_id)

        if len(p_ids) < 2 or len(a_ids) < 2:
            continue

        entries.append({
            "prompt_ids": p_ids,
            "target_ids": a_ids,
            "category": r.get("category", "general"),
            "has_reasoning": (thinking and len(thinking) > 5),
        })

    print(f"   ✓ Prepared {len(entries):,} samples ({reasoning_count:,} step-by-step reasoning, {direct_count:,} direct).")
    return entries


def evaluate_validation_loss(
    model: nn.Module,
    val_entries: List[Dict[str, Any]],
    device: torch.device,
    sampling_p: float = 0.5,
    num_batches: int = 25,
    batch_size: int = 4,
) -> float:
    model.eval()
    losses = []
    val_sample = random.sample(val_entries, min(len(val_entries), num_batches * batch_size))

    with torch.no_grad():
        for i in range(0, len(val_sample), batch_size):
            batch = val_sample[i:i + batch_size]
            if not batch:
                continue
            max_p = max(len(e["prompt_ids"]) for e in batch)
            max_a = max(len(e["target_ids"]) for e in batch)

            p_t = torch.zeros((len(batch), max_p), dtype=torch.long, device=device)
            t_t = torch.full((len(batch), max_a), -100, dtype=torch.long, device=device)

            for b_idx, e in enumerate(batch):
                p_t[b_idx, :len(e["prompt_ids"])] = torch.tensor(e["prompt_ids"], dtype=torch.long, device=device)
                t_t[b_idx, :len(e["target_ids"])] = torch.tensor(e["target_ids"], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(
                    prompts=p_t,
                    targets=t_t,
                    sampling_p=sampling_p,
                    deliberation_mode="scheduled",
                )
            losses.append(loss.item())

    model.train()
    return sum(losses) / max(1, len(losses))


def run_local_sft(
    base_checkpoint: str = DEFAULT_BASE_CKPT,
    output_checkpoint: str = DEFAULT_OUTPUT_CKPT,
    train_path: str = DEFAULT_TRAIN_DATA,
    val_path: str = DEFAULT_VAL_DATA,
    max_steps: int = 1500,
    batch_size: int = 4,
    accum_steps: int = 8,
    lr: float = 3e-5,
    min_lr: float = 3e-6,
    sampling_p: float = 0.5,
    probe_interval: int = 100,
    warmup_steps: int = 50,
    resume: bool = False,
):
    # Auto-resume detection
    load_ckpt_path = base_checkpoint
    if resume and os.path.exists(output_checkpoint):
        load_ckpt_path = output_checkpoint
        print(f"🔄 Auto-resuming from existing output checkpoint: {output_checkpoint}")

    print("=" * 80)
    print("🧠 CEREBROSPHEROID V5: LOCAL STEP-BY-STEP REASONING SFT ENGINE")
    print("=" * 80)
    print(f"  Load Checkpoint            : {load_ckpt_path}")
    print(f"  Target SFT Checkpoint      : {output_checkpoint}")
    print(f"  Total Steps Target         : {max_steps}")
    print(f"  Deliberation Sampling P    : {sampling_p:.2f} (50% internal thought / 50% teacher)")
    print(f"  Batch Configuration        : Micro {batch_size} × Accum {accum_steps} = Effective {batch_size * accum_steps}")
    print(f"  Learning Rate Schedule     : {lr:.2e} -> {min_lr:.2e} (Warmup {warmup_steps} steps)")
    print("=" * 80 + "\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Tokenizer
    tokenizer = build_tokenizer(DEFAULT_TOKENIZER_DIR)
    vocab_size = getattr(tokenizer, "vocab_size", 32000)

    # 2. Load Model
    print(f"\n🚀 Loading weights from: {load_ckpt_path}...")
    ckpt = torch.load(load_ckpt_path, map_location="cpu", weights_only=False)
    cfg = infer_v5_config_from_state_dict(ckpt)
    model = CerebroSpheroidV5(config=cfg).to(device)
    load_v5_checkpoint_into_torch(model, ckpt)
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters).")

    # Determine start step
    start_step = 0
    best_val_loss = float("inf")
    ema_loss = None

    if "step" in ckpt and (resume or load_ckpt_path != DEFAULT_BASE_CKPT):
        start_step = int(ckpt["step"])
        best_val_loss = float(ckpt.get("val_loss", float("inf")))
        ema_loss = float(ckpt.get("ema_loss", 1.0)) if ckpt.get("ema_loss") is not None else None
        print(f"   ⏩ Resuming from Step {start_step} (Prior Best Val Loss: {best_val_loss:.4f})")

    # 3. Load Datasets
    train_entries = load_tokenized_dataset(train_path, tokenizer, vocab_size, max_prompt_len=160, max_target_len=160)
    val_entries = load_tokenized_dataset(val_path, tokenizer, vocab_size, max_prompt_len=160, max_target_len=160)

    # 4. Optimizer & Param Groups
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and not any(nd in name for nd in ["norm", "bias", "phase_embed"]):
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95), eps=1e-8)

    # Cosine learning rate scheduler with warmup
    def get_lr(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return lr * (step_idx + 1) / max(1, warmup_steps)
        progress = (step_idx - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    # Benchmark Probe Prompts across reasoning, math, facts, and persona
    test_prompts = [
        ("What is 15 + 27?", "42 (with derivation)"),
        ("If Alice has 3 apples and Bob gives her 2 more, how many apples does Alice have?", "5 apples"),
        ("A train departs at 2:00 PM and arrives 3 hours later. What time is it?", "5:00 PM"),
        ("What is the capital of France?", "Paris"),
        ("Hello! Who are you?", "Doty"),
        ("What is the chemical symbol for Gold?", "Au"),
        ("Complete the analogy: author is to book as artist is to ____.", "painting"),
    ]

    best_checkpoint_path = output_checkpoint.replace(".pt", "_best.pt")
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)

    print(f"\n🚀 Training ({start_step + 1} -> {max_steps} steps)...\n")
    start_time = time.time()
    step = start_step
    accum_count = 0
    optimizer.zero_grad()

    model.train()

    while step < max_steps:
        # Sample micro-batch
        batch = random.sample(train_entries, batch_size)
        max_p = max(len(e["prompt_ids"]) for e in batch)
        max_a = max(len(e["target_ids"]) for e in batch)

        p_t = torch.zeros((batch_size, max_p), dtype=torch.long, device=device)
        t_t = torch.full((batch_size, max_a), -100, dtype=torch.long, device=device)

        for b_idx, e in enumerate(batch):
            p_t[b_idx, :len(e["prompt_ids"])] = torch.tensor(e["prompt_ids"], dtype=torch.long, device=device)
            t_t[b_idx, :len(e["target_ids"])] = torch.tensor(e["target_ids"], dtype=torch.long, device=device)

        # Forward pass with Scheduled Deliberation at sampling_p=0.5
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(
                prompts=p_t,
                targets=t_t,
                sampling_p=sampling_p,
                deliberation_mode="scheduled",
            )
            loss_scaled = loss / accum_steps

        loss_scaled.backward()
        accum_count += 1

        if accum_count % accum_steps == 0:
            step += 1
            cur_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            ema_loss = loss_val if ema_loss is None else (0.92 * ema_loss + 0.08 * loss_val)

            if step % 10 == 0 or step == start_step + 1 or step == max_steps:
                elapsed = time.time() - start_time
                steps_done = step - start_step
                steps_per_sec = steps_done / max(1e-3, elapsed)
                vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                print(f"[Step {step:4d}/{max_steps}] Loss: {loss_val:.4f} | EMA: {ema_loss:.4f} | LR: {cur_lr:.2e} | Speed: {steps_per_sec:.2f} st/s | VRAM: {vram_gb:.2f} GB")

            # Periodic Evaluation & Qualitative Probes
            if step % probe_interval == 0 or step == max_steps:
                val_loss = evaluate_validation_loss(model, val_entries, device, sampling_p=sampling_p)
                print(f"\n📊 [Validation Step {step}] Held-out Val Loss: {val_loss:.4f} (Best: {min(best_val_loss, val_loss):.4f})")

                save_payload = {
                    "model_state": {k: v.cpu().float() for k, v in model.state_dict().items()},
                    "step": step,
                    "val_loss": float(val_loss),
                    "ema_loss": float(ema_loss),
                    "sampling_p": sampling_p,
                    "lr": cur_lr,
                    "config": cfg,
                }

                # Always save latest checkpoint
                torch.save(save_payload, output_checkpoint)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(save_payload, best_checkpoint_path)
                    print(f"   🏆 New best checkpoint saved to: {best_checkpoint_path} (Val Loss: {val_loss:.4f})")

                # Qualitative generation probe with sufficient tokens to observe step-by-step reasoning
                model.eval()
                print("   --- 🔍 Live Generation Probes (Step-by-Step Scratchpad) ---")
                with torch.no_grad():
                    for p_probe, expected in test_prompts:
                        res = model.generate_text(
                            p_probe,
                            tokenizer=tokenizer,
                            max_new_tokens=64,
                            temperature=0.2,
                            device=device,
                        )
                        res_clean = res.strip().replace("\n", " ⏎ ")
                        print(f"   • \"{p_probe}\"\n     -> \"{res_clean[:90]}\" [Expected: {expected}]")
                print("   ------------------------------------------------------------\n")
                model.train()

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✅ REASONING SFT TRAINING COMPLETE in {total_time / 60:.2f} minutes!")
    print(f"   • Best Val Loss        : {best_val_loss:.4f}")
    print(f"   • Best Checkpoint Path : {best_checkpoint_path}")
    print(f"   • Latest Checkpoint    : {output_checkpoint}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Step-by-Step Reasoning SFT for CerebroSpheroid V5 at 0.5 P")
    parser.add_argument("--base-checkpoint", type=str, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--output-checkpoint", type=str, default=DEFAULT_OUTPUT_CKPT)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--sampling-p", type=float, default=0.5)
    parser.add_argument("--probe-interval", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Resume from output_checkpoint if available")
    args = parser.parse_args()

    run_local_sft(
        base_checkpoint=args.base_checkpoint,
        output_checkpoint=args.output_checkpoint,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        sampling_p=args.sampling_p,
        probe_interval=args.probe_interval,
        warmup_steps=args.warmup_steps,
        resume=args.resume,
    )
