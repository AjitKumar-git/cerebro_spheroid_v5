"""
Empirical Benchmark: Progressive Co-Adaptation on OpenHermes-2.5 & Cosmopedia-v2
================================================================================

Tests the two-stage progressive unfreezing recipe on diverse frontier datasets:
1. OpenHermes-2.5: Complex multi-step reasoning, logic puzzles, coding dialogues.
2. Cosmopedia-v2: Dense textbook and educational scientific explanations.

Workflow:
- Baseline: Evaluates initial zero-shot loss on OpenHermes & Cosmopedia.
- Stage 1: Motor Head Alignment (Steps 1-20 | Core FROZEN | lr=2e-3).
- Stage 2: Full End-to-End Co-Adaptation (Steps 21-50 | FULL BRAIN UNFROZEN | Core lr=3e-4, Motor lr=1e-3).
- Qualitative Test: Generates actual responses to OpenHermes & Cosmopedia prompts before vs after!
"""

import os
import sys
import time
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

os.environ["HF_HOME"] = "/home/ajit/.cache/hf_home"

V4_ROOT = "/home/ajit/Documents/Startup/Active_Development/reasoning_agent"
V5_ROOT = "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5"
if V4_ROOT not in sys.path:
    sys.path.insert(0, V4_ROOT)

from src.spheroid_v4 import build_spheroid_v4, CerebroSpheroidV4
from src.corpus import build_tokenizer


def load_diverse_dataset_batches(tokenizer, count_per_source=25, batch_size=4, max_len=64):
    """Loads a 50/50 mixture of OpenHermes-2.5 and Cosmopedia-v2 prompt-response pairs."""
    pairs = []

    # 1. OpenHermes-2.5 (from local parquet)
    oh_path = os.path.join(V4_ROOT, "data/openhermes/openhermes_2_5.parquet")
    if os.path.exists(oh_path):
        df_oh = pd.read_parquet(oh_path)
        for _, row in df_oh.head(count_per_source * 2).iterrows():
            convs = row.get("conversations", [])
            q, a = "", ""
            for turn in convs:
                sender = str(turn.get("from", "")).lower()
                val = str(turn.get("value", "")).strip()
                if "human" in sender and not q:
                    q = val
                elif ("gpt" in sender or "assistant" in sender) and not a:
                    a = val
            if len(q) > 15 and len(a) > 20:
                pairs.append(("OpenHermes", q, a))
            if len(pairs) >= count_per_source:
                break

    # 2. Cosmopedia-v2 (via streaming)
    from datasets import load_dataset
    try:
        cosmo = iter(load_dataset("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", split="train", streaming=True))
        cosmo_collected = 0
        while cosmo_collected < count_per_source:
            r = next(cosmo)
            p = str(r.get("prompt", "")).strip()
            t = str(r.get("text", "")).strip()
            if len(p) < 10:
                p = "Explain the educational context of the following topic:"
            if len(t) > 30:
                pairs.append(("Cosmopedia", p, t))
                cosmo_collected += 1
    except Exception as e:
        print(f"Warning loading Cosmopedia: {e}")

    # Build tokenized batches
    batches = []
    num_batches = len(pairs) // batch_size
    for b in range(num_batches):
        chunk = pairs[b * batch_size : (b + 1) * batch_size]
        sources = [c[0] for c in chunk]
        q_list = [[20436] + tokenizer.encode(c[1], add_special_tokens=False)[:48] for c in chunk]
        a_list = [tokenizer.encode(c[2], add_special_tokens=False)[:max_len] + [2] for c in chunk]

        max_q = max(len(q) for q in q_list)
        max_a = max(len(a) for a in a_list)

        p_tensor = torch.tensor([q + [0] * (max_q - len(q)) for q in q_list], dtype=torch.long)
        t_tensor = torch.tensor([a + [-100] * (max_a - len(a)) for a in a_list], dtype=torch.long)
        batches.append((sources, p_tensor, t_tensor, chunk))

    return batches


def reset_motor_head(model: CerebroSpheroidV4, seed: int = 1337):
    torch.manual_seed(seed)
    for m in model.motor_dnn.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Parameter):
            nn.init.zeros_(m)


def generate_qualitative_sample(model: CerebroSpheroidV4, tokenizer: Any, prompt: str, device: torch.device) -> str:
    """Generates autoregressive response with discrete token auditory feedback."""
    model.eval()
    with torch.no_grad():
        ans = model.generate_text(
            prompt,
            tokenizer=tokenizer,
            thinking_cycles=3,
            max_new_tokens=48,
            temperature=0.6,
            repetition_penalty=1.35,
            device=device,
            strip_bio_prefix=True,
        )
    return ans


def main():
    print("=" * 80)
    print("🔬 OpenHermes-2.5 & Cosmopedia-v2 Co-Adaptation Benchmark")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tok = build_tokenizer("llama")
    print("\n📚 Loading diverse dataset batches (50% OpenHermes-2.5 + 50% Cosmopedia-v2)...")
    batches = load_diverse_dataset_batches(tok, count_per_source=30, batch_size=4, max_len=48)
    print(f"✅ Prepared {len(batches)} batches across OpenHermes and Cosmopedia.")

    ckpt_path = os.path.join(V4_ROOT, "checkpoints", "cerebro_spheroid_v4_step149200_best.pt")

    # Load 10B checkpoint
    model = build_spheroid_v4("v4_small", vocab_size=len(tok)).to(device)
    model.load_compatible_checkpoint(ckpt_path, device=device)

    # 3 Qualitative Test Prompts
    test_prompts = [
        ("Reasoning", "If you have 5 apples and give 2 to a friend, how many apples do you have left?"),
        ("Cosmopedia", "Explain why the sky appears blue during the day:"),
        ("Logic", "Every day a tree drops 7 leaves. How many leaves does it drop in February of a non-leap year?"),
    ]

    print("\n" + "=" * 80)
    print("📝 PRE-ADAPTATION GENERATION BASELINE (10B Base Model):")
    print("=" * 80)
    for category, prompt in test_prompts:
        out = generate_qualitative_sample(model, tok, prompt, device)
        print(f"[{category}] Prompt: {prompt}")
        print(f"  Response: {out[:120]}...\n")

    # Reset motor head to test fresh alignment
    reset_motor_head(model, seed=1337)

    # -------------------------------------------------------------
    # STAGE 1: Motor Head Alignment (Steps 1-20 | Core FROZEN)
    # -------------------------------------------------------------
    print("-" * 80)
    print("📍 STAGE 1: Motor Head Alignment (Steps 1-20 | Core FROZEN | lr=2e-3)")
    print("-" * 80)
    model.volumetric_core.requires_grad_(False)
    model.sensory_dnn.requires_grad_(False)
    model.motor_dnn.requires_grad_(True)

    opt_stage1 = torch.optim.AdamW(model.motor_dnn.parameters(), lr=2e-3)
    losses_s1 = []

    model.train()
    t_s1 = time.time()
    for s in range(20):
        srcs, p_tensor, t_tensor, _ = batches[s % len(batches)]
        p_tensor, t_tensor = p_tensor.to(device), t_tensor.to(device)

        opt_stage1.zero_grad()
        logits = model(p_tensor, targets=t_tensor, response_steps=t_tensor.shape[1], sampling_p=1.0)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), t_tensor.view(-1), ignore_index=-100)
        loss.backward()
        opt_stage1.step()
        losses_s1.append(loss.item())

        if (s + 1) % 5 == 0 or s == 0:
            src_label = "/".join(set(srcs))
            print(f"  Step {s + 1:2d}/20 | Loss: {loss.item():.4f} | Source: {src_label:<20} | VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    el_s1 = time.time() - t_s1
    print(f"✅ Stage 1 Complete in {el_s1:.2f}s (Avg {el_s1 / 20 * 1000:.1f} ms/step). Loss: {losses_s1[-1]:.4f}")

    # -------------------------------------------------------------
    # STAGE 2: Full Brain Unfreeze (Steps 21-50 | Core & Sensory UNFROZEN)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("📍 STAGE 2: Full End-to-End Co-Adaptation (Steps 21-50 | ALL UNFROZEN | Core lr=3e-4, Motor lr=1e-3)")
    print("-" * 80)
    model.volumetric_core.requires_grad_(True)
    model.sensory_dnn.requires_grad_(True)
    model.motor_dnn.requires_grad_(True)

    param_groups = [
        {"params": model.volumetric_core.parameters(), "lr": 3e-4},
        {"params": model.sensory_dnn.parameters(), "lr": 3e-4},
        {"params": model.motor_dnn.parameters(), "lr": 1e-3},
    ]
    opt_stage2 = torch.optim.AdamW(param_groups)
    losses_s2 = []

    t_s2 = time.time()
    for s in range(30):
        step_num = s + 21
        srcs, p_tensor, t_tensor, _ = batches[s % len(batches)]
        p_tensor, t_tensor = p_tensor.to(device), t_tensor.to(device)

        opt_stage2.zero_grad()
        logits = model(p_tensor, targets=t_tensor, response_steps=t_tensor.shape[1], sampling_p=1.0)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), t_tensor.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_stage2.step()
        losses_s2.append(loss.item())

        if (s + 1) % 5 == 0 or s == 0:
            src_label = "/".join(set(srcs))
            print(f"  Step {step_num:2d}/50 | Loss: {loss.item():.4f} | Source: {src_label:<20} | VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    el_s2 = time.time() - t_s2
    print(f"✅ Stage 2 Complete in {el_s2:.2f}s (Avg {el_s2 / 30 * 1000:.1f} ms/step). Loss: {losses_s2[-1]:.4f}")

    # -------------------------------------------------------------
    # POST-ADAPTATION GENERATION TEST
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📝 POST-CO-ADAPTATION GENERATION TEST (OpenHermes + Cosmopedia Co-Adapted):")
    print("=" * 80)
    for category, prompt in test_prompts:
        out = generate_qualitative_sample(model, tok, prompt, device)
        print(f"[{category}] Prompt: {prompt}")
        print(f"  Response: {out[:140]}...\n")

    print("=" * 80)
    print(f"🎯 SUMMARY:")
    print(f"  • Stage 1 Start Loss : {losses_s1[0]:.4f} (Random Motor Head)")
    print(f"  • Stage 1 End Loss   : {losses_s1[-1]:.4f} (Motor Head Aligned)")
    print(f"  • Stage 2 End Loss   : {losses_s2[-1]:.4f} (Full Brain Co-Adapted)")
    print(f"  • Peak VRAM          : {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
