#!/usr/bin/env python3
"""
CerebroSpheroid V5: Graded Curriculum Pre-Training & Agentic Fine-Tuning Pipeline
==================================================================================

Full 1,000,000-Step Developmental Curriculum:
  • Phase 1 (0 -> 50k steps)      : Basic Syntax, Grammar & Low-Entropy Structure
  • Phase 2 (50k -> 500k steps)   : Knowledge, Reasoning & Calculation (GSM8k, OpenHermes)
  • Phase 3 (500k -> 900k steps)  : Modern Agentic Workflows & Multi-Turn Tool-Use
  • Phase 4 (900k -> 1,000k steps): Model Consolidation, Replay & Energy Annealing

Storage-Optimized:
  • Zero upfront token disk dumps.
  • Streams memory-mapped row batches from Parquets on-the-fly.
  • Adafactor optimizer with 15 MB state tracking.
"""

import argparse
import math
import os
import sys
import time
from typing import Optional

# Set PyTorch memory optimization flag before torch initialization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import Adafactor
from transformers import AutoTokenizer

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5
from src.agentic_harness import ActionType, ToolActionHead
from src.graded_curriculum import (
    CurriculumPhase,
    CurriculumScheduleConfig,
    GradedCurriculumStreamer,
    PHASE_NAMES,
    compute_curriculum_lr,
    get_curriculum_phase,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CerebroSpheroid V5 Graded Curriculum Training")
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--test-steps", type=int, default=None, help="If set, stops after test-steps (for local dry run)")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--max-lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between log lines")
    parser.add_argument("--save-interval", type=int, default=1000, help="Steps between checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/graded_curriculum", help="Checkpoint directory")
    parser.add_argument("--offline", action="store_true", help="Force offline mode using local datasets & cache")
    return parser.parse_args()


def load_local_tokenizer(offline: bool = True):
    """Loads tokenizer from local HuggingFace cache or fallback directory."""
    candidate_paths = [
        "/home/ajit/.cache/hf_home/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549",
        "huggyllama/llama-7b",
    ]
    for p in candidate_paths:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                p,
                local_files_only=offline or os.path.exists(p),
                use_fast=False,
            )
            print(f"[Tokenizer] Successfully loaded tokenizer from: {p}")
            return tokenizer
        except Exception:
            continue

    print("[Tokenizer] HuggingFace cache not accessible directly. Using standard fallback tokenizer.")
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set offline flags if requested
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== CerebroSpheroid V5 Graded Curriculum Engine ===")
    print(f"Device: {device} | Total Target Steps: {args.total_steps}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")

    # 1. Initialize Tokenizer & Model
    tokenizer = load_local_tokenizer(offline=args.offline)

    config = ModelConfigV5(
        vocab_size=32000,
        surface_dim=768,
        radial_dim=512,
        num_shells=32,
        num_shortcut_tracts=96,
        top_k_tracts=24,
        num_heads=8,
        max_seq_len=args.seq_len,
        thinking_cycles=4,
        use_sparse_mot=True,
        use_energy_regularization=True,
    )

    model = CerebroSpheroidV5(config).to(device)
    action_head = ToolActionHead(surface_dim=config.surface_dim, num_actions=len(ActionType)).to(device)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in action_head.parameters())
    print(f"Total Parameters (Model + ActionHead): {total_params:,} (~{total_params/1e6:.1f}M)")

    # 2. Setup Memory-Saving Curriculum Streamer
    schedule_config = CurriculumScheduleConfig(
        total_steps=args.total_steps,
        phase_1_end=50_000,
        phase_2_end=500_000,
        phase_3_end=900_000,
        warmup_steps=2_000,
        max_lr=args.max_lr,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    streamer = GradedCurriculumStreamer(
        tokenizer=tokenizer,
        config=schedule_config,
        search_dirs=[
            "data",
            "../reasoning_agent/data",
            "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/data",
            "/home/ajit/Documents/Startup/Active_Development/reasoning_agent/data",
        ],
    )

    # 3. Setup Adafactor Optimizer (VRAM footprint ~15MB)
    all_params = list(model.parameters()) + list(action_head.parameters())
    optimizer = Adafactor(
        all_params,
        lr=schedule_config.max_lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    action_loss_fn = nn.CrossEntropyLoss()

    print("\n--- Starting Graded Pre-Training Pipeline ---")
    start_time = time.time()
    accum_loss = 0.0
    accum_act_loss = 0.0
    tokens_processed = 0
    step = 0
    max_steps = args.test_steps or args.total_steps

    current_phase = get_curriculum_phase(0, schedule_config)
    print(f"Entering {PHASE_NAMES[current_phase]} (Step 0 -> {schedule_config.phase_1_end})")

    model.train()
    action_head.train()

    while step < max_steps:
        # Check for phase transitions
        new_phase = get_curriculum_phase(step, schedule_config)
        if new_phase != current_phase:
            print(f"\n=======================================================")
            print(f"★ PHASE TRANSITION at Step {step}: Entering {PHASE_NAMES[new_phase]}")
            print(f"=======================================================\n")
            current_phase = new_phase
            # Save checkpoint at phase boundary
            boundary_ckpt = os.path.join(args.checkpoint_dir, f"v5_phase_{int(new_phase)}_boundary_{step}.pt")
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "action_head_state": action_head.state_dict(),
                "config": config,
            }, boundary_ckpt)
            print(f"[Checkpoint Saved] Phase boundary snapshot: {boundary_ckpt}")

        # Update dynamic learning rate
        lr = compute_curriculum_lr(step, schedule_config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()

        # Gradient accumulation loop
        for micro in range(args.grad_accum):
            input_ids, targets, action_labels, batch_phase = streamer.get_batch(
                step=step,
                device=device,
                batch_size=args.batch_size,
            )

            # Mixed-precision forward pass (bfloat16) to conserve VRAM
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                # Forward pass through Spheroid V5
                ce_loss, logits = model(input_ids, targets=targets)

                # In Phase 3 & 4: Train discrete tool action classification directly from Shell 31
                act_loss = torch.tensor(0.0, device=device)
                if batch_phase in (CurriculumPhase.PHASE_3_WORKFLOWS, CurriculumPhase.PHASE_4_CONSOLIDATION):
                    cortical_mantle = model.current_spheroid_state[:, -1, :]  # Shell 31
                    action_logits, _ = action_head(cortical_mantle)
                    act_loss = action_loss_fn(action_logits, action_labels)

                # Combined loss: Language Modeling + Action Classification
                total_batch_loss = (ce_loss + 0.1 * act_loss) / args.grad_accum

            total_batch_loss.backward()

            accum_loss += ce_loss.item() / args.grad_accum
            accum_act_loss += act_loss.item() / args.grad_accum
            tokens_processed += input_ids.numel()

        # Clip gradients & optimizer step
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        step += 1

        # Periodic Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = tokens_processed / max(1.0, elapsed)
            vram_mb = torch.cuda.memory_reserved(device) / (1024**2) if torch.cuda.is_available() else 0.0

            avg_loss = accum_loss / max(1, args.log_interval)
            avg_act_loss = accum_act_loss / max(1, args.log_interval)
            act_str = f" | ActLoss: {avg_act_loss:.4f}" if current_phase >= CurriculumPhase.PHASE_3_WORKFLOWS else ""
            print(
                f"[Step {step:6d}/{max_steps}] Phase: {int(current_phase)} ({PHASE_NAMES[current_phase][:16]}...) | "
                f"Loss: {avg_loss:.4f}{act_str} | LR: {lr:.2e} | "
                f"Tok/s: {tok_per_sec:6.1f} | VRAM: {vram_mb:5.0f}MB"
            )
            accum_loss = 0.0
            accum_act_loss = 0.0

        # Periodic Checkpointing
        if step % args.save_interval == 0 and step > 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"v5_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "action_head_state": action_head.state_dict(),
                "config": config,
            }, ckpt_path)
            print(f"--> Saved checkpoint: {ckpt_path}")

    # Final Checkpoint
    final_ckpt = os.path.join(args.checkpoint_dir, "v5_graded_final.pt")
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "action_head_state": action_head.state_dict(),
        "config": config,
    }, final_ckpt)
    print(f"\n✓ Graded training completed {step} steps! Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
