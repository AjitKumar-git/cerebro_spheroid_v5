#!/usr/bin/env python3
"""
scripts/salvage_v5_small_pretrain.py

Salvages V5-Small by Continuing Pretraining with Expanded Elastic Prompt Windows:
  1. Solves the 32-token prompt horizon bottleneck:
     Transitions prompt context from 32 tokens to 256 - 512 tokens.
  2. Uses the strictly filtered, audited online curriculum:
     (FineWeb-Edu >= 3.0, Cosmopedia v2 Digested, FineMath, Indented Python).
  3. Contiguous document chunking:
     Follow-on chunks carry preceding text into the prompt window (zero amnesic slicing).
  4. Preserves modern V5 architecture:
     (Bidirectional sensory self-attention, RoPE, 16 concentric shells, causal KV cache).
  5. Low-learning-rate gentle expansion:
     Expands sensory receptive field while preserving learned token representations.
"""

import os
import sys
import time
import math
import argparse
from typing import Dict, Any, List, Tuple

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5, infer_v5_config_from_state_dict
from src.elastic_window import (
    build_online_dataset_iterator,
    ElasticBucketBatcher,
    GLOBAL_TELEMETRY,
)
from src.corpus import format_stimulus_prompt, parse_model_output, clean_motor_response


# Expanded Receptive Field Buckets (Replacing the old 32-token trap)
WIDE_ELASTIC_BUCKETS = [
    ("standard_context", 256, 512),
    ("extended_context", 384, 512),
    ("balanced_deep",    512, 512),
]

PROBE_PROMPTS = [
    # Short Prompts (< 30 tokens)
    ("What is the capital of France?", "Paris"),
    ("Complete the analogy: hot is to cold as day is to", "night"),
    ("What is 15 + 27?", "42"),
    # Realistic Extended Horizon Prompts (50 - 150 tokens)
    (
        "Study the following passage and answer the question:\n"
        "The Amazon rainforest is the world's largest tropical rainforest. "
        "It spans across nine nations, but the vast majority of the forest, "
        "approximately 60 percent, is contained within the borders of Brazil.\n\n"
        "Question: Which country contains the largest portion of the Amazon rainforest?",
        "Brazil",
    ),
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May? Show step-by-step calculation.",
        "72",
    ),
    (
        "Write a Python function `is_palindrome(s: str) -> bool` that returns True if a given string is a palindrome.",
        "def is_palindrome",
    ),
]


def salvage_v5_small(
    base_checkpoint: str = "checkpoints/v5_small_agentic_tool_aligned.pt",
    output_checkpoint: str = "checkpoints/v5_small_salvaged_pretrain.pt",
    max_steps: int = 500,
    batch_size: int = 2,
    accum_steps: int = 4,
    lr: float = 2.0e-5,
    min_lr: float = 3.0e-6,
    warmup_steps: int = 30,
    probe_interval: int = 100,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 80)
    print("🧠 CEREBRO-SPHEROID V5-SMALL PRETRAINING SALVAGE ENGINE")
    print(f"Base Checkpoint:   {base_checkpoint}")
    print(f"Output Checkpoint: {output_checkpoint}")
    print(f"Device:            {device_str}")
    print(f"Max Steps:         {max_steps} | Micro-Batch: {batch_size} | Accumulation: {accum_steps}")
    print(f"Effective Batch:   {batch_size * accum_steps} sequences/step")
    print(f"Learning Rate:     {lr:.2e} -> {min_lr:.2e} (Warmup: {warmup_steps} steps)")
    print(f"Elastic Buckets:   {WIDE_ELASTIC_BUCKETS}")
    print("=" * 80)

    device = torch.device(device_str)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # 1. Load Tokenizer
    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    print(f"   ✓ Loaded Tokenizer from {tok_dir} (Vocab: {len(tokenizer)})")

    # 2. Load Model & Checkpoint
    raw = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    cfg_d = raw.get("config")
    if isinstance(cfg_d, ModelConfigV5):
        cfg = cfg_d
    elif isinstance(cfg_d, dict):
        cfg = ModelConfigV5(**cfg_d)
    else:
        cfg = infer_v5_config_from_state_dict(raw, default_preset="v5_small")

    model = CerebroSpheroidV5(cfg).to(dtype).to(device)
    sd = raw.get("model_state", raw.get("model", raw))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"   ✓ Loaded base weights into V5 model (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

    # 3. Setup Dataset Streamer with Wide Prompt Elastic Buckets
    print("\n📦 Connecting to Audited Autonomous Curriculum Stream...")
    data_iter = build_online_dataset_iterator(
        dataset_source="autonomous",
        rank=0,
        world_size=1,
        total_steps=1_500_000,
    )

    batcher = ElasticBucketBatcher(
        tokenizer=tokenizer,
        data_iterator=data_iter,
        batch_size=batch_size,
        total_window=1024,
        buckets=WIDE_ELASTIC_BUCKETS,
        framework="torch",
    )

    # 4. Optimizer: Decoupled AdamW
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name.lower() or "bias" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95), eps=1e-8)

    def get_lr(s: int) -> float:
        if s <= warmup_steps:
            return max(min_lr, lr * float(s) / float(max(1, warmup_steps)))
        prog = min(1.0, (s - warmup_steps) / float(max(1, max_steps - warmup_steps)))
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * prog))

    # 5. Training Loop
    print("\n🚀 Beginning Pretraining Expansion on Wide Prompt Context...")
    start_time = time.time()
    best_loss = 999.0
    ema_loss = None
    accum_count = 0
    optimizer.zero_grad()

    model.train()
    for step in range(1, max_steps + 1):
        curr_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        # Pull next elastic batch
        b_idx, b_name, prompts, targets = batcher.next_batch()
        prompts = prompts.to(device)
        targets = targets.to(device)

        with torch.amp.autocast("cuda", dtype=dtype):
            loss, _ = model(
                prompts=prompts,
                targets=targets,
                deliberation_mode="teacher_forcing",
            )
            loss_scaled = loss / accum_steps

        loss_scaled.backward()
        accum_count += 1

        if accum_count >= accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

        loss_val = loss.item()
        if ema_loss is None:
            ema_loss = loss_val
        else:
            ema_loss = 0.95 * ema_loss + 0.05 * loss_val

        if loss_val < best_loss:
            best_loss = loss_val

        if step % 10 == 0 or step == max_steps:
            elapsed = time.time() - start_time
            rate = step / max(1e-3, elapsed)
            vram = torch.cuda.memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0
            print(
                f"   [Step {step:4d}/{max_steps}] Bucket: '{b_name}' | Loss: {loss_val:.4f} (EMA: {ema_loss:.4f}) "
                f"| Best: {best_loss:.4f} | LR: {curr_lr:.2e} | Speed: {rate:.1f} st/s | VRAM: {vram:.0f}MB",
                flush=True,
            )

        # Periodic Qualitative Evaluation
        if step % probe_interval == 0 or step == max_steps:
            model.eval()
            print(f"\n--- 🔍 Pretraining Generalization Probe (Step {step}) ---", flush=True)
            with torch.no_grad():
                for p_text, expected in PROBE_PROMPTS:
                    inp = format_stimulus_prompt(p_text)
                    ids = tokenizer.encode(inp, return_tensors="pt").to(device)
                    out = model.generate(
                        ids,
                        max_new_tokens=96,
                        temperature=0.2,
                        top_k=20,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    gen_ids = out[0]
                    gen_text = tokenizer.decode(gen_ids).strip().replace("\n", " ")
                    parsed = parse_model_output(gen_text)
                    ans_str = parsed.get("answer") if parsed.get("answer") else gen_text
                    tool_str = f" | Tool: <act> {parsed['tool_calls'][0]} </act>" if parsed.get("tool_calls") else ""
                    print(f"   Prompt: \"{p_text}\"", flush=True)
                    print(f"   Target: [{expected}] -> Generated: \"{ans_str}\"{tool_str}", flush=True)
            print("--------------------------------------------------------------------------\n", flush=True)
            model.train()

        # Periodic Checkpoint Save
        if step % 250 == 0 or step == max_steps:
            os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
            torch.save({
                "model_state": {k: v.cpu().float() for k, v in model.state_dict().items()},
                "config": model.config,
                "step": step,
                "loss": float(loss_val),
                "best_loss": float(best_loss),
                "ema_loss": float(ema_loss),
                "strategy": "wide_prompt_salvage_pretrain",
            }, output_checkpoint)
            print(f"   💾 Checkpoint saved at Step {step} -> {output_checkpoint}\n", flush=True)

    print(f"\n🎉 Salvage pretraining complete! Final checkpoint saved to: {output_checkpoint}")
    GLOBAL_TELEMETRY.log_report(force=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Salvage V5-Small by Continuing Pretraining with Wide Prompt Horizons.")
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/v5_small_agentic_tool_aligned.pt")
    parser.add_argument("--output-checkpoint", type=str, default="checkpoints/v5_small_salvaged_pretrain.pt")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--min-lr", type=float, default=3.0e-6)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--probe-interval", type=int, default=100)
    args = parser.parse_args()

    salvage_v5_small(
        base_checkpoint=args.base_checkpoint,
        output_checkpoint=args.output_checkpoint,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        probe_interval=args.probe_interval,
    )
