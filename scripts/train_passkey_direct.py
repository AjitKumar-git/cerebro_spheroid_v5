#!/usr/bin/env python3
"""
🧪 Empirical Passkey & Retrieval Optimization Test
=================================================
Validates whether CerebroSpheroid V5 can learn needle-in-a-haystack retrieval
at 128 and 256 token lengths when given balanced data representation,
proper <stim>...</stim> delimiters, and Answer-Anchor Loss Weighting.
"""

import os
import sys
import json
import math
import time
import random
import argparse

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.spheroid_v5 import CerebroSpheroidV5, infer_v5_config_from_state_dict
from src.corpus import format_stimulus_prompt, encode_no_special
from scripts.eval_passkey_retrieval import build_passkey_prompt, evaluate_passkey_matrix
from scripts.generate_retrieval_curriculum_data import build_passkey_sample, build_multihop_sample, build_lost_in_middle_sample


def build_focused_retrieval_dataset(
    tokenizer: Any,
    vocab_size: int,
    num_samples: int = 1500,
    digits_pool: List[int] = [1, 2, 3, 5],
) -> list:
    entries = []
    rng = random.Random(42)
    lengths = [128, 256]
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]

    # 1. 60% Passkey Needle Retrieval with Progressive Digit Lengths
    for _ in range(int(num_samples * 0.60)):
        L = rng.choice(lengths)
        d = rng.choice(depths)
        digits = rng.choice(digits_pool)
        if digits == 1:
            passkey_val = str(rng.randint(1, 9))
        elif digits == 2:
            passkey_val = str(rng.randint(10, 99))
        elif digits == 3:
            passkey_val = str(rng.randint(100, 999))
        elif digits == 4:
            passkey_val = str(rng.randint(1000, 9999))
        else:
            passkey_val = str(rng.randint(10000, 99999))

        prompt_str, _ = build_passkey_prompt(L, d, passkey_val, tokenizer)
        p_str = format_stimulus_prompt(prompt_str)
        if rng.random() < 0.6:
            a_str = f" {passkey_val}."
        else:
            a_str = f"<think> Scanning document for secret passkey. Located: {passkey_val}. </think>\n{passkey_val}."
        p_ids = [t % vocab_size for t in encode_no_special(tokenizer, p_str)][:256]
        a_ids = [t % vocab_size for t in encode_no_special(tokenizer, a_str)][:96]
        if len(p_ids) > 1 and len(a_ids) > 1:
            weights = [5.0 if any(c.isdigit() and c in passkey_val for c in tokenizer.decode([tid])) else 2.0 for tid in a_ids]
            entries.append({
                "prompt_ids": p_ids,
                "target_ids": a_ids,
                "weights": weights,
                "type": f"passkey_{digits}d",
            })

    # 2. 20% Multi-hop & Lost-in-the-Middle
    for _ in range(int(num_samples * 0.20)):
        L = rng.choice(lengths)
        d = rng.choice(depths)
        builder = rng.choice([build_multihop_sample, build_lost_in_middle_sample])
        sample = builder(L, d, tokenizer, rng)
        p_str = format_stimulus_prompt(sample["prompt"])
        a_str = sample["response"].strip()
        p_ids = [t % vocab_size for t in encode_no_special(tokenizer, p_str)][:256]
        a_ids = [t % vocab_size for t in encode_no_special(tokenizer, a_str)][:128]
        if len(p_ids) > 1 and len(a_ids) > 1:
            weights = [4.0 if ("</think>" in a_str or "passkey" in a_str) else 2.5] * len(a_ids)
            entries.append({
                "prompt_ids": p_ids,
                "target_ids": a_ids,
                "weights": weights,
                "type": "reasoning_retrieval",
            })

    # 3. 20% Golden Reasoning & Direct Facts (Anchoring)
    from scripts.run_refined_sft_pytorch import build_direct_slot_qa_corpus
    direct_pairs = build_direct_slot_qa_corpus()
    for q, a in direct_pairs[:int(num_samples * 0.20)]:
        p_str = format_stimulus_prompt(q)
        p_ids = [t % vocab_size for t in encode_no_special(tokenizer, p_str)][:256]
        a_ids = [t % vocab_size for t in encode_no_special(tokenizer, a.strip())][:64]
        if len(p_ids) > 1 and len(a_ids) > 1:
            entries.append({
                "prompt_ids": p_ids,
                "target_ids": a_ids,
                "weights": [3.5] * len(a_ids),
                "type": "direct_fact",
            })

    rng.shuffle(entries)
    print(f"📦 Built focused retrieval dataset: {len(entries)} samples across digits {digits_pool}")
    return entries


def train_passkey_direct(
    base_checkpoint: str = "checkpoints/v5_passkey_focused_test.pt",
    output_checkpoint: str = "checkpoints/v5_passkey_focused_test.pt",
    max_steps: int = 300,
    batch_size: int = 4,
    accum_steps: int = 2,
    lr: float = 6e-5,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    digits_pool: List[int] = [5],
):
    device = torch.device(device_str)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    raw = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    cfg = infer_v5_config_from_state_dict(raw)
    cfg.use_rope = True
    model = CerebroSpheroidV5(cfg)
    from src.spheroid_v5 import load_v5_checkpoint_into_torch
    load_v5_checkpoint_into_torch(model, raw, verbose=False)
    model = model.to(dtype).to(device)

    tokenizer = AutoTokenizer.from_pretrained("data/tokenizer")
    dataset = build_focused_retrieval_dataset(tokenizer, cfg.vocab_size, num_samples=1500, digits_pool=digits_pool)

    # Parameter groups with accelerated cross-attention alignment
    attn_names = ("q_proj", "out_proj", "k_proj", "v_proj")
    attn_params = [p for n, p in model.named_parameters() if p.requires_grad and any(an in n for an in attn_names)]
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2 and "norm" not in n and not any(an in n for an in attn_names)]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and (p.ndim < 2 or "norm" in n) and not any(an in n for an in attn_names)]
    optimizer = torch.optim.AdamW([
        {"params": attn_params, "lr": lr * 2.0, "weight_decay": 0.01},
        {"params": decay_params, "lr": lr, "weight_decay": 0.01},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.95), eps=1e-8)

    print(f"🚀 Training {max_steps} steps of focused retrieval optimization on {device}...")
    model.train()
    optimizer.zero_grad()
    t0 = time.time()

    for step in range(1, max_steps + 1):
        # Sample micro-batch
        batch_entries = [dataset[i] for i in torch.randint(0, len(dataset), (batch_size,)).tolist()]
        max_p = max(len(e["prompt_ids"]) for e in batch_entries)
        max_a = max(len(e["target_ids"]) for e in batch_entries)

        p_tensor = torch.zeros((batch_size, max_p), dtype=torch.long, device=device)
        t_tensor = torch.full((batch_size, max_a), -100, dtype=torch.long, device=device)
        w_tensor = torch.zeros((batch_size, max_a), dtype=torch.float, device=device)

        for b_idx, e in enumerate(batch_entries):
            p_ids = e["prompt_ids"]
            a_ids = e["target_ids"]
            p_tensor[b_idx, :len(p_ids)] = torch.tensor(p_ids, dtype=torch.long, device=device)
            t_tensor[b_idx, :len(a_ids)] = torch.tensor(a_ids, dtype=torch.long, device=device)
            w_tensor[b_idx, :len(a_ids)] = torch.tensor(e["weights"], dtype=torch.float, device=device)

        with torch.amp.autocast("cuda", dtype=dtype):
            loss, _ = model(
                prompts=p_tensor,
                targets=t_tensor,
                target_weights=w_tensor,
                deliberation_mode="teacher_forcing",
            )
            loss_scaled = loss / accum_steps

        loss_scaled.backward()

        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if step % 25 == 0 or step == max_steps:
            elapsed = time.time() - t0
            print(f"   [Step {step:3d}/{max_steps}] Loss: {loss.item():.4f} | {step/max(1e-3, elapsed):.1f} st/s", flush=True)

        # Evaluate passkey retrieval periodically
        if step % 100 == 0 or step == max_steps:
            model.eval()
            print(f"\n🔍 Mid-Training Passkey Evaluation at Step {step} (1-digit):")
            eval_res_1d = evaluate_passkey_matrix(
                model=model,
                tokenizer=tokenizer,
                lengths=[128, 256],
                depths=[0.1, 0.5, 0.9],
                trials_per_cell=2,
                device=device,
                digits=1,
            )
            print(f"\n🔍 Mid-Training Passkey Evaluation at Step {step} (5-digit):")
            eval_res_5d = evaluate_passkey_matrix(
                model=model,
                tokenizer=tokenizer,
                lengths=[128, 256],
                depths=[0.1, 0.5, 0.9],
                trials_per_cell=2,
                device=device,
                digits=5,
            )
            model.train()

    # Save Checkpoint
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    torch.save({
        "model_state": {k: v.cpu().float() for k, v in model.state_dict().items()},
        "config": model.config,
        "step": max_steps,
    }, output_checkpoint)
    print(f"\n💾 Saved focused passkey checkpoint -> {output_checkpoint}")


if __name__ == "__main__":
    train_passkey_direct(max_steps=250, digits_pool=[5])
