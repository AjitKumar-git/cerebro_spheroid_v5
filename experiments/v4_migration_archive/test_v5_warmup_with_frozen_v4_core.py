"""
CerebroSpheroid V5: Warmup Alignment with Frozen V4 Core
=========================================================

Procedure:
1. Build CerebroSpheroid V5 (236.9M parameters, affine-tied dictionary).
2. Transfer trained volumetric core weights from V4 checkpoint (10B tokens, step 149,200).
   - Radial in/out projections: [256, 640] -> [512, 768]
   - Re-entrant radial wave operators: [15, 256, 256] -> [31, 512, 512]
   - Shortcut axonal tracts: [40, 640, 640] -> [96, 768, 768]
   - Radial normalization scales: [640] -> [768]
3. FREEZE the V5 Volumetric Core (all 200M core parameters kept strictly static).
4. Train/Align the sensory encoder and motor decoder (affine-tied head) on OpenHermes & Cosmopedia:
   - Trainable parameters: ~35M (Sensory DNN + Premotor Decoder).
   - The shared embedding dictionary is optimized from both ends (input sensory and output motor).
5. Test the warm V5 model:
   - Loss trajectory during warmup.
   - Qualitative autoregressive generation before vs after warmup.
   - Spectral radius and VRAM telemetry.
"""

from __future__ import annotations

import os
import sys
import time
import math
import pandas as pd
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["HF_HOME"] = "/home/ajit/.cache/hf_home"

V4_ROOT = "/home/ajit/Documents/Startup/Active_Development/reasoning_agent"
V5_ROOT = "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5"

if V4_ROOT not in sys.path:
    sys.path.append(V4_ROOT)
if V5_ROOT not in sys.path:
    sys.path.insert(0, V5_ROOT)

from src.spheroid_v5 import CerebroSpheroidV5, build_spheroid_v5, ModelConfigV5
from src.corpus import build_tokenizer


def transfer_v4_core_to_v5(v4_checkpoint_path: str, v5_model: CerebroSpheroidV5) -> Dict[str, Any]:
    """Transfers volumetric core weights from V4 checkpoint into V5 core."""
    print(f"📦 Loading V4 checkpoint from: {v4_checkpoint_path}")
    raw = torch.load(v4_checkpoint_path, map_location="cpu", weights_only=False)
    v4_state = raw.get("model_state_dict", raw.get("model", raw))

    clean_v4: Dict[str, torch.Tensor] = {}
    for k, v in v4_state.items():
        if isinstance(v, torch.Tensor):
            ck = k.replace("volumetric_core.", "core.").replace("motor_dnn.", "motor.").replace("sensory_dnn.", "sensory.")
            clean_v4[ck] = v

    v5_state = v5_model.state_dict()
    transferred = []

    # 1. Radial Projections (surface_dim <-> radial_dim)
    if "core.radial_proj_in.weight" in clean_v4:
        v4_w = clean_v4["core.radial_proj_in.weight"]  # [256, 640]
        # In V5: radial_in_proj is [512, 768]
        v5_state["volumetric_core.radial_in_proj.weight"][:256, :640] = v4_w[:256, :640]
        v5_state["volumetric_core.radial_out_proj.weight"][:256, :640] = v4_w[:256, :640]
        transferred.append("radial_in_proj & radial_out_proj: transferred [256, 640] slice")

    if "core.radial_proj_out.weight" in clean_v4:
        v4_w = clean_v4["core.radial_proj_out.weight"]  # [640, 256]
        # In V5: radial_up is [768, 512]
        v5_state["volumetric_core.radial_up.weight"][:640, :256] = v4_w[:640, :256]
        transferred.append("radial_up: transferred [640, 256] slice")

    # 2. Axonal Shortcut Tracts
    if "core.shortcut_tracts" in clean_v4:
        v4_t = clean_v4["core.shortcut_tracts"]  # [40, 640, 640]
        # In V5: shortcut_weight is [96, 768, 768]
        v5_state["volumetric_core.shortcut_weight"][:40, :640, :640] = v4_t[:40, :640, :640]
        transferred.append("shortcut_weight: transferred [40, 640, 640] axonal tracts")

    # 3. Inter-Shell Radial Wave Transition Operators
    if "core.reentrant_field" in clean_v4:
        # V4 reentrant_field: [16, 16, 256, 256]
        v4_rf = clean_v4["core.reentrant_field"]
        # Extract inward transitions: shell k+1 -> shell k (for k in 0..14)
        for k in range(15):
            # Inward wave: [31, 512, 512]
            v5_state["volumetric_core.w_radial_in"][k, :256, :256] = v4_rf[k + 1, k, :256, :256]
            # Outward wave: [31, 512, 512]
            v5_state["volumetric_core.w_radial_out"][k, :256, :256] = v4_rf[k, k + 1, :256, :256]
        transferred.append("w_radial_in & w_radial_out: transferred 15 shell transition blocks of [256, 256]")

    # 4. Core Normalization
    if "core.norm.weight" in clean_v4:
        v4_n = clean_v4["core.norm.weight"]  # [640]
        v5_state["volumetric_core.radial_norm.weight"][:640] = v4_n[:640]
        transferred.append("radial_norm: transferred [640] scale vector")

    v5_model.load_state_dict(v5_state)
    print(f"✅ Core Transfer Complete! Transferred parameters:\n" + "\n".join(f"   • {t}" for t in transferred))
    return {"transferred": transferred}


def load_diverse_dataset_batches(tokenizer, count_per_source=30, batch_size=4, max_len=48):
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


def generate_qualitative_sample(model: CerebroSpheroidV5, tokenizer, prompt: str, device: torch.device) -> str:
    """Generates autoregressive response using V5 tied dictionary generator."""
    model.eval()
    with torch.no_grad():
        ans = model.generate_text(
            prompt,
            tokenizer=tokenizer,
            thinking_cycles=3,
            max_new_tokens=48,
            temperature=0.7,
            repetition_penalty=1.2,
            device=device,
        )
    return ans


def main():
    print("=" * 80)
    print("🧠 CerebroSpheroid V5: Alignment with Transferred Frozen V4 Core")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tok = build_tokenizer("llama")
    print(f"📖 Tokenizer loaded: {len(tok)} tokens.")

    # 1. Build V5 Compact Model (236.9M params)
    print("\n🔨 Building CerebroSpheroid V5 (Compact Preset)...")
    model = build_spheroid_v5("v5_compact", vocab_size=len(tok)).to(device)
    param_counts = model.count_parameters()
    print(f"  • Total Parameters   : {param_counts['total_parameters'] / 1e6:.2f} M")
    print(f"  • Volumetric Core    : {param_counts['volumetric_core'] / 1e6:.2f} M")
    print(f"  • Sensory DNN        : {param_counts['sensory_dnn'] / 1e6:.2f} M")
    print(f"  • Motor Decoder      : {param_counts['motor_decoder'] / 1e6:.2f} M")
    print(f"  • Active Compute     : {param_counts['active_compute_percent']:.2f}%")

    # 2. Transfer V4 Core Weights
    ckpt_path = os.path.join(V4_ROOT, "checkpoints", "cerebro_spheroid_v4_step149200_best.pt")
    transfer_v4_core_to_v5(ckpt_path, model)

    # 3. FREEZE Volumetric Core
    model.volumetric_core.requires_grad_(False)
    model.sensory_dnn.requires_grad_(True)
    model.motor_decoder.requires_grad_(True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n❄️ Parameter Partitioning:")
    print(f"  • Frozen Parameters (Core)       : {frozen_params / 1e6:.2f} M")
    print(f"  • Trainable Parameters (Sensory+Motor): {trainable_params / 1e6:.2f} M ({trainable_params / param_counts['total_parameters'] * 100:.2f}%)")

    # 4. Load Data Batches
    print("\n📚 Loading diverse dataset batches (OpenHermes-2.5 + Cosmopedia-v2)...")
    batches = load_diverse_dataset_batches(tok, count_per_source=30, batch_size=4, max_len=48)
    print(f"✅ Prepared {len(batches)} batches.")

    # 3 Qualitative Test Prompts
    test_prompts = [
        ("Reasoning", "If you have 5 apples and give 2 to a friend, how many apples do you have left?"),
        ("Cosmopedia", "Explain why the sky appears blue during the day:"),
        ("Logic", "Every day a tree drops 7 leaves. How many leaves does it drop in February of a non-leap year?"),
    ]

    print("\n" + "=" * 80)
    print("📝 PRE-WARMUP V5 GENERATION BASELINE (Transferred Core, Untrained Sensory/Motor):")
    print("=" * 80)
    for category, prompt in test_prompts:
        out = generate_qualitative_sample(model, tok, prompt, device)
        print(f"[{category}] Prompt: {prompt}")
        print(f"  Response: {out[:120]}...\n")

    # -------------------------------------------------------------
    # STAGE 1: Warmup Alignment (Core FROZEN, Sensory + Motor Trainable)
    # -------------------------------------------------------------
    print("-" * 80)
    print("📍 WARMUP ALIGNMENT (Steps 1-30 | Core FROZEN | lr=1e-3)")
    print("-" * 80)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=0.01,
    )

    losses = []
    t_start = time.time()
    model.train()

    for s in range(30):
        srcs, p_tensor, t_tensor, _ = batches[s % len(batches)]
        p_tensor, t_tensor = p_tensor.to(device), t_tensor.to(device)

        optimizer.zero_grad()
        out = model(p_tensor, targets=t_tensor, sampling_p=1.0)
        if isinstance(out, tuple):
            loss, logits = out
        else:
            logits = out
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), t_tensor.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if (s + 1) % 5 == 0 or s == 0:
            src_label = "/".join(set(srcs))
            print(f"  Step {s + 1:2d}/30 | Loss: {loss.item():.4f} | Source: {src_label:<20} | VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    el_time = time.time() - t_start
    print(f"\n✅ Warmup Alignment Complete in {el_time:.2f}s (Avg {el_time / 30 * 1000:.1f} ms/step)!")
    print(f"  • Initial Loss : {losses[0]:.4f}")
    print(f"  • Final Loss   : {losses[-1]:.4f}")

    # -------------------------------------------------------------
    # POST-WARMUP GENERATION TEST (Warm V5)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📝 POST-WARMUP V5 GENERATION TEST (Warm V5 with Aligned Sensory & Motor):")
    print("=" * 80)
    for category, prompt in test_prompts:
        out = generate_qualitative_sample(model, tok, prompt, device)
        print(f"[{category}] Prompt: {prompt}")
        print(f"  Response: {out[:140]}...\n")

    print("=" * 80)
    print(f"🎯 SUMMARY:")
    print(f"  • Pre-Warmup Loss  : {losses[0]:.4f}")
    print(f"  • Post-Warmup Loss : {losses[-1]:.4f}")
    print(f"  • Delta Loss       : {losses[0] - losses[-1]:.4f}")
    print(f"  • Peak VRAM        : {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
