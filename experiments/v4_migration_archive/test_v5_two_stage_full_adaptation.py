"""
CerebroSpheroid V5: Two-Stage Progressive Co-Adaptation Benchmark
==================================================================

Stage 1: Warmup Alignment (Steps 1-20 | Core FROZEN | lr=1e-3)
- Trains 48.96M outer parameters (Sensory DNN + Motor Decoder + Affine-Tied Head).
- Bridges fresh vocabulary embeddings into transferred V4 core coordinate space.

Stage 2: Full Brain Co-Adaptation (Steps 21-50 | FULL BRAIN UNFROZEN)
- Core lr = 1e-4, Sensory lr = 3e-4, Motor lr = 5e-4.
- All 236.9M parameters co-adapt end-to-end on OpenHermes-2.5 & Cosmopedia-v2.

Evaluates:
- Loss trajectory across both stages.
- Hardware telemetry (Peak VRAM, step times).
- Qualitative text generation at Baseline -> Post-Stage 1 -> Post-Stage 2.
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers.optimization import Adafactor

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
        v5_state["volumetric_core.radial_in_proj.weight"][:256, :640] = v4_w[:256, :640]
        v5_state["volumetric_core.radial_out_proj.weight"][:256, :640] = v4_w[:256, :640]
        transferred.append("radial_in_proj & radial_out_proj: transferred [256, 640] slice")

    if "core.radial_proj_out.weight" in clean_v4:
        v4_w = clean_v4["core.radial_proj_out.weight"]  # [640, 256]
        v5_state["volumetric_core.radial_up.weight"][:640, :256] = v4_w[:640, :256]
        transferred.append("radial_up: transferred [640, 256] slice")

    # 2. Axonal Shortcut Tracts
    if "core.shortcut_tracts" in clean_v4:
        v4_t = clean_v4["core.shortcut_tracts"]  # [40, 640, 640]
        v5_state["volumetric_core.shortcut_weight"][:40, :640, :640] = v4_t[:40, :640, :640]
        transferred.append("shortcut_weight: transferred [40, 640, 640] axonal tracts")

    # 3. Inter-Shell Radial Wave Transition Operators
    if "core.reentrant_field" in clean_v4:
        v4_rf = clean_v4["core.reentrant_field"]
        for k in range(15):
            v5_state["volumetric_core.w_radial_in"][k, :256, :256] = v4_rf[k + 1, k, :256, :256]
            v5_state["volumetric_core.w_radial_out"][k, :256, :256] = v4_rf[k, k + 1, :256, :256]
        transferred.append("w_radial_in & w_radial_out: transferred 15 shell transition blocks of [256, 256]")

    # 4. Core Normalization
    if "core.norm.weight" in clean_v4:
        v4_n = clean_v4["core.norm.weight"]  # [640]
        v5_state["volumetric_core.radial_norm.weight"][:640] = v4_n[:640]
        transferred.append("radial_norm: transferred [640] scale vector")

    v5_model.load_state_dict(v5_state)
    print(f"✅ Core Transfer Complete! Transferred {len(transferred)} architectural sub-matrices.")
    return {"transferred": transferred}


def load_diverse_dataset_batches(tokenizer, count_per_source=30, batch_size=2, max_len=40):
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

    batches = []
    num_batches = len(pairs) // batch_size
    for b in range(num_batches):
        chunk = pairs[b * batch_size : (b + 1) * batch_size]
        sources = [c[0] for c in chunk]
        q_list = [[20436] + tokenizer.encode(c[1], add_special_tokens=False)[:36] for c in chunk]
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
    print("🧠 CerebroSpheroid V5: Two-Stage Progressive Co-Adaptation")
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

    # 2. Transfer V4 Core Weights
    ckpt_path = os.path.join(V4_ROOT, "checkpoints", "cerebro_spheroid_v4_step149200_best.pt")
    transfer_v4_core_to_v5(ckpt_path, model)

    # 3. Load Data Batches (batch_size=2 for low VRAM headroom)
    print("\n📚 Loading diverse dataset batches (OpenHermes-2.5 + Cosmopedia-v2)...")
    batches = load_diverse_dataset_batches(tok, count_per_source=30, batch_size=2, max_len=40)
    print(f"✅ Prepared {len(batches)} batches.")

    test_prompts = [
        ("Reasoning", "If you have 5 apples and give 2 to a friend, how many apples do you have left?"),
        ("Cosmopedia", "Explain why the sky appears blue during the day:"),
        ("Logic", "Every day a tree drops 7 leaves. How many leaves does it drop in February of a non-leap year?"),
    ]

    # =============================================================
    # STAGE 1: Warmup Alignment (Steps 1-20 | Core FROZEN)
    # =============================================================
    print("\n" + "-" * 80)
    print("📍 STAGE 1: Warmup Alignment (Steps 1-20 | Core FROZEN | lr=1e-3)")
    print("-" * 80)
    model.volumetric_core.requires_grad_(False)
    model.sensory_dnn.requires_grad_(True)
    model.motor_decoder.requires_grad_(True)

    opt_stage1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=0.01,
    )

    losses_s1 = []
    t_s1 = time.time()
    model.train()

    for s in range(20):
        srcs, p_tensor, t_tensor, _ = batches[s % len(batches)]
        p_tensor, t_tensor = p_tensor.to(device), t_tensor.to(device)

        opt_stage1.zero_grad()
        out = model(p_tensor, targets=t_tensor, sampling_p=1.0)
        loss = out[0] if isinstance(out, tuple) else out
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_stage1.step()
        losses_s1.append(loss.item())

        if (s + 1) % 5 == 0 or s == 0:
            src_label = "/".join(set(srcs))
            print(f"  Step {s + 1:2d}/20 | Loss: {loss.item():.4f} | Source: {src_label:<20} | VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    el_s1 = time.time() - t_s1
    print(f"✅ Stage 1 Complete in {el_s1:.2f}s. Loss: {losses_s1[0]:.4f} -> {losses_s1[-1]:.4f}")

    print("\n" + "=" * 80)
    print("📝 POST-STAGE-1 GENERATION SAMPLE (Warm Outer Heads, Frozen Core):")
    print("=" * 80)
    out1 = generate_qualitative_sample(model, tok, test_prompts[0][1], device)
    print(f"Prompt: {test_prompts[0][1]}")
    print(f"Response: {out1[:120]}...\n")

    # =============================================================
    # STAGE 2: Full Brain Co-Adaptation (Steps 21-50 | ALL UNFROZEN)
    # =============================================================
    print("-" * 80)
    print("📍 STAGE 2: Full End-to-End Co-Adaptation (Steps 21-50 | ALL UNFROZEN)")
    print("    • Core lr = 1e-4 | Sensory lr = 3e-4 | Motor lr = 5e-4")
    print("-" * 80)

    # Unfreeze entire brain
    model.volumetric_core.requires_grad_(True)
    model.sensory_dnn.requires_grad_(True)
    model.motor_decoder.requires_grad_(True)

    param_groups = [
        {"params": model.volumetric_core.parameters(), "lr": 1e-4, "weight_decay": 0.01},
        {"params": model.sensory_dnn.parameters(), "lr": 3e-4, "weight_decay": 0.01},
        {"params": model.motor_decoder.parameters(), "lr": 5e-4, "weight_decay": 0.01},
    ]
    opt_stage2 = Adafactor(param_groups, scale_parameter=False, relative_step=False, warmup_init=False)

    del opt_stage1
    torch.cuda.empty_cache()
    losses_s2 = []
    t_s2 = time.time()
    model.train()

    for s in range(30):
        step_num = s + 21
        srcs, p_tensor, t_tensor, _ = batches[(s + 20) % len(batches)]
        p_tensor, t_tensor = p_tensor.to(device), t_tensor.to(device)

        opt_stage2.zero_grad()
        out = model(p_tensor, targets=t_tensor, sampling_p=1.0)
        loss = out[0] if isinstance(out, tuple) else out
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_stage2.step()
        losses_s2.append(loss.item())

        if (s + 1) % 5 == 0 or s == 0:
            src_label = "/".join(set(srcs))
            print(f"  Step {step_num:2d}/50 | Loss: {loss.item():.4f} | Source: {src_label:<20} | VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    el_s2 = time.time() - t_s2
    print(f"✅ Stage 2 Complete in {el_s2:.2f}s (Avg {el_s2 / 30 * 1000:.1f} ms/step). Loss: {losses_s2[0]:.4f} -> {losses_s2[-1]:.4f}")

    # =============================================================
    # POST-STAGE-2 BENCHMARK GENERATION TEST
    # =============================================================
    print("\n" + "=" * 80)
    print("📝 POST-STAGE-2 BENCHMARK GENERATIONS (Full Brain Co-Adapted V5):")
    print("=" * 80)
    for category, prompt in test_prompts:
        out = generate_qualitative_sample(model, tok, prompt, device)
        print(f"[{category}] Prompt: {prompt}")
        print(f"  Response: {out[:140]}...\n")

    print("=" * 80)
    print("🎯 FINAL SUMMARY:")
    print(f"  • Stage 1 (Frozen Core) Start Loss : {losses_s1[0]:.4f}")
    print(f"  • Stage 1 (Frozen Core) End Loss   : {losses_s1[-1]:.4f}")
    print(f"  • Stage 2 (Full Brain) Start Loss  : {losses_s2[0]:.4f}")
    print(f"  • Stage 2 (Full Brain) End Loss    : {losses_s2[-1]:.4f}")
    print(f"  • Total Loss Reduction             : {losses_s1[0] - losses_s2[-1]:.4f}")
    print(f"  • Peak VRAM Allocated              : {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
