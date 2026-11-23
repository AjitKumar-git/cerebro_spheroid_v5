#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Closed-Loop Agentic SFT & Alignment Trainer
==================================================================
Trains CerebroSpheroid V5 checkpoint on 9,500 paired multi-turn closed-loop trajectories:
  - Stage 1: User Prompt -> <think> ... </think> <tool_call> ... </tool_call>
  - Stage 2: User Prompt + Tool Call + <tool_response> -> <think> ... </think> Grounded Answer
  - Conversational: User Prompt -> <think> ... </think> Direct Prose

Features:
  - Base Checkpoint : checkpoints/v5_small_latest.pt (Step 202,500)
  - Dataset         : data/consolidated/02_agentic_tools/closed_loop_sft_pairs.jsonl
  - Training Mode   : AMP BF16 / FP16 on local GPU
  - Schedule        : Cosine LR decay with linear warmup
  - Gradient Accum  : Micro-batch 4 x 4 accumulation steps = Effective Batch 16
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    PRESETS_V5,
    load_v5_checkpoint_into_torch,
)
from src.jax.checkpoint import adapt_jax_to_torch_state_dict
from src.tools.unified_tool_dispatcher import UnifiedToolDispatcher


def load_agentic_records(
    dataset_path: str = "data/consolidated/02_agentic_tools/closed_loop_sft_pairs.jsonl",
    max_samples: int = 15000,
) -> List[Dict[str, str]]:
    """Loads consolidated closed-loop agentic training pairs."""
    full_path = os.path.join(WORKSPACE_ROOT, dataset_path)
    records = []
    print(f"📖 Ingesting verified closed-loop dataset: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                inp = row.get("input", "")
                out = row.get("output", "")
                if inp and out:
                    records.append({"input": inp, "output": out, "type": row.get("type", "unknown")})
                if len(records) >= max_samples:
                    break
            except Exception:
                continue
    print(f"✓ Loaded {len(records):,} clean closed-loop agentic training records.")
    return records


def setup_model(
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[CerebroSpheroidV5, AutoTokenizer, ModelConfigV5, torch.device]:
    """Loads CerebroSpheroidV5 with latest adapted weights."""
    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    if not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(WORKSPACE_ROOT, checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model", ckpt))

    cfg_dict = PRESETS_V5["v5_small"].copy()
    cfg_dict["sensory_position_steps"] = 4096
    cfg_dict["max_phase_steps"] = 4096
    cfg_dict["max_seq_len"] = 4096
    cfg_dict["use_rope"] = True
    cfg_dict["top_k_resonance"] = 0
    cfg_dict["use_deliberation_probe"] = False
    cfg_dict["use_adaptive_thinking"] = False
    cfg = ModelConfigV5(**cfg_dict)

    if any(k.startswith(("sensory.", "core.", "motor.")) for k in state.keys()):
        state = adapt_jax_to_torch_state_dict(state, cfg)

    model = CerebroSpheroidV5(cfg)
    matched, total = load_v5_checkpoint_into_torch(model, state, verbose=False)
    print(f"✓ Model loaded with {matched}/{total} parameters matched onto {dev}.")
    model.to(dev)
    return model, tokenizer, cfg, dev


def train_closed_loop_sft(
    checkpoint_path: str = "checkpoints/v5_small_latest.pt",
    dataset_path: str = "data/consolidated/02_agentic_tools/closed_loop_sft_pairs.jsonl",
    output_dir: str = "checkpoints/tool_aligned",
    num_steps: int = 1000,
    micro_batch_size: int = 4,
    grad_accum_steps: int = 4,
    max_lr: float = 2.5e-5,
    min_lr: float = 2.0e-6,
    warmup_steps: int = 50,
    max_input_len: int = 256,
    max_output_len: int = 128,
    device: str = "cuda",
    save_interval: int = 250,
) -> None:
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(WORKSPACE_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer, cfg, dev = setup_model(checkpoint_path, device=device)
    dataset = load_agentic_records(dataset_path=dataset_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01, betas=(0.9, 0.95))
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    eos_id = getattr(tokenizer, "eos_token_id", 2) or 2

    print(f"\n{'='*80}")
    print(f"🚀 CEREBRO-SPHEROID V5: CLOSED-LOOP AGENTIC SFT")
    print(f"Base Model    : {checkpoint_path}")
    print(f"Dataset       : {dataset_path} ({len(dataset):,} samples)")
    print(f"Total Steps   : {num_steps} | Micro-Batch: {micro_batch_size} | Accum: {grad_accum_steps} (Eff: {micro_batch_size * grad_accum_steps})")
    print(f"Learning Rate : {max_lr} -> {min_lr} (Warmup: {warmup_steps} steps)")
    print(f"Output Path   : {output_dir}")
    print(f"{'='*80}\n")

    model.train()
    running_loss = []
    optimizer.zero_grad()

    t_start = time.time()
    for step in range(1, num_steps + 1):
        step_loss = 0.0

        # Cosine LR Schedule with Linear Warmup
        if step < warmup_steps:
            current_lr = min_lr + (max_lr - min_lr) * (step / float(warmup_steps))
        else:
            progress = (step - warmup_steps) / float(max(1, num_steps - warmup_steps))
            current_lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Micro-batch accumulation
        for micro_step in range(grad_accum_steps):
            batch_samples = [random.choice(dataset) for _ in range(micro_batch_size)]

            p_ids_list = []
            t_ids_list = []
            for s in batch_samples:
                p_toks = tokenizer.encode(s["input"], add_special_tokens=False)[:max_input_len]
                t_toks = tokenizer.encode(s["output"], add_special_tokens=False)[:max_output_len]
                if not t_toks or t_toks[-1] != eos_id:
                    t_toks.append(eos_id)
                p_ids_list.append(p_toks)
                t_ids_list.append(t_toks)

            max_p = max(len(p) for p in p_ids_list)
            max_t = max(len(t) for t in t_ids_list)

            inp_tensor = torch.full((micro_batch_size, max_p), pad_id, dtype=torch.long, device=dev)
            tgt_tensor = torch.full((micro_batch_size, max_t), -100, dtype=torch.long, device=dev)

            for i in range(micro_batch_size):
                inp_tensor[i, : len(p_ids_list[i])] = torch.tensor(p_ids_list[i], dtype=torch.long, device=dev)
                tgt_tensor[i, : len(t_ids_list[i])] = torch.tensor(t_ids_list[i], dtype=torch.long, device=dev)

            # Mixed precision forward pass
            use_amp = dev.type == "cuda"
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                loss, _ = model(prompts=inp_tensor, targets=tgt_tensor, use_anchor_loss=False)
                scaled_loss = loss / float(grad_accum_steps)

            scaled_loss.backward()
            step_loss += scaled_loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_loss.append(step_loss)

        if step % 50 == 0:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if step == 1 or step % 25 == 0:
            avg_loss = sum(running_loss[-25:]) / len(running_loss[-25:])
            elapsed = time.time() - t_start
            ips = step / max(1e-4, elapsed)
            remaining_s = (num_steps - step) / ips
            print(f"Step {step:04d}/{num_steps:04d} | Loss: {step_loss:.4f} (Avg: {avg_loss:.4f}) | LR: {current_lr:.2e} | {ips:.1f} step/s | ETA: {remaining_s/60:.1f}m")

        if step % save_interval == 0 or step == num_steps:
            save_path = os.path.join(output_dir, f"v5_small_tool_aligned_step_{step}.pt")
            bf16_state = {
                k: v.bfloat16() if v.is_floating_point() else v
                for k, v in model.state_dict().items()
            }
            torch.save({
                "step": step,
                "model_state": bf16_state,
                "loss": step_loss,
            }, save_path)
            print(f"💾 Checkpoint saved: {save_path}\n")

    print("\n✅ Closed-Loop Agentic SFT Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_small_latest.pt")
    parser.add_argument("--dataset", type=str, default="data/consolidated/02_agentic_tools/agentic_tools_sensory_grounded.jsonl")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    train_closed_loop_sft(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        num_steps=args.steps,
        micro_batch_size=args.micro_batch,
        grad_accum_steps=args.accum,
        max_lr=args.lr,
        warmup_steps=args.warmup,
        device=args.device,
        save_interval=args.save_interval,
    )
