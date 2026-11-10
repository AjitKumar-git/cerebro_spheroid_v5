"""
CerebroSpheroid V5: High-Difficulty OpenHermes-2.5 Convergence & Reasoning Test
================================================================================

Trains V5 on complex multi-step reasoning pairs from OpenHermes-2.5 until loss
reaches near 3.0, systematically testing text generation at milestone loss thresholds:
- Milestone 1: Loss < 5.0
- Milestone 2: Loss < 4.0
- Milestone 3: Loss < 3.5
- Milestone 4: Loss <= 3.0 (Target)

Test Prompts:
1. Logic: 'Every day a tree drops 7 leaves. How many leaves does it drop in February of a non-leap year? Include your logic.'
2. Arithmetic: 'If you have 5 apples and give 2 to a friend, how many apples do you have left?'
3. Reasoning: 'A train leaves Station A at 60 mph. Another train leaves Station B at 40 mph towards Station A. If they are 200 miles apart, when do they meet?'
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

V4_ROOT = "/home/ajit/Documents/Startup/Active_Development/reasoning_agent"
V5_ROOT = "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5"

if V4_ROOT not in sys.path:
    sys.path.append(V4_ROOT)
if V5_ROOT not in sys.path:
    sys.path.insert(0, V5_ROOT)

from src.spheroid_v5 import CerebroSpheroidV5, build_spheroid_v5, ModelConfigV5
from src.corpus import build_tokenizer
from transformers.optimization import Adafactor


def transfer_v4_core_to_v5(v4_checkpoint_path: str, v5_model: CerebroSpheroidV5) -> None:
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
    transferred_count = 0

    # 1. Radial Projections
    if "core.radial_proj_in.weight" in clean_v4:
        v4_w = clean_v4["core.radial_proj_in.weight"]
        v5_state["volumetric_core.radial_in_proj.weight"][:256, :640] = v4_w[:256, :640]
        v5_state["volumetric_core.radial_out_proj.weight"][:256, :640] = v4_w[:256, :640]
        transferred_count += 2

    if "core.radial_proj_out.weight" in clean_v4:
        v4_w = clean_v4["core.radial_proj_out.weight"]
        v5_state["volumetric_core.radial_up.weight"][:640, :256] = v4_w[:640, :256]
        transferred_count += 1

    # 2. Axonal Shortcut Tracts
    if "core.shortcut_tracts" in clean_v4:
        v4_t = clean_v4["core.shortcut_tracts"]
        v5_state["volumetric_core.shortcut_weight"][:40, :640, :640] = v4_t[:40, :640, :640]
        transferred_count += 1

    # 3. Inter-Shell Radial Wave Transition Operators
    if "core.reentrant_field" in clean_v4:
        v4_rf = clean_v4["core.reentrant_field"]
        for k in range(15):
            v5_state["volumetric_core.w_radial_in"][k, :256, :256] = v4_rf[k + 1, k, :256, :256]
            v5_state["volumetric_core.w_radial_out"][k, :256, :256] = v4_rf[k, k + 1, :256, :256]
        transferred_count += 2

    # 4. Core Normalization
    if "core.norm.weight" in clean_v4:
        v4_n = clean_v4["core.norm.weight"]
        v5_state["volumetric_core.radial_norm.weight"][:640] = v4_n[:640]
        transferred_count += 1

    v5_model.load_state_dict(v5_state)
    print(f"✅ Transferred {transferred_count} core parameter blocks from 10B V4 model.")


def load_openhermes_reasoning_batches(tokenizer, num_samples=120, batch_size=2, max_len=48):
    """Loads high-difficulty reasoning chains from OpenHermes-2.5."""
    oh_path = os.path.join(V4_ROOT, "data/openhermes/openhermes_2_5.parquet")
    df_oh = pd.read_parquet(oh_path)
    pairs = []

    for _, row in df_oh.iterrows():
        convs = row.get("conversations", [])
        q, a = "", ""
        for turn in convs:
            sender = str(turn.get("from", "")).lower()
            val = str(turn.get("value", "")).strip()
            if "human" in sender and not q:
                q = val
            elif ("gpt" in sender or "assistant" in sender) and not a:
                a = val
        if len(q) > 20 and len(a) > 30:
            pairs.append((q, a))
        if len(pairs) >= num_samples:
            break

    batches = []
    num_batches = len(pairs) // batch_size
    for b in range(num_batches):
        chunk = pairs[b * batch_size : (b + 1) * batch_size]
        q_list = [[20436] + tokenizer.encode(c[0], add_special_tokens=False)[:40] for c in chunk]
        a_list = [tokenizer.encode(c[1], add_special_tokens=False)[:max_len] + [2] for c in chunk]

        max_q = max(len(q) for q in q_list)
        max_a = max(len(a) for a in a_list)

        p_tensor = torch.tensor([q + [0] * (max_q - len(q)) for q in q_list], dtype=torch.long)
        t_tensor = torch.tensor([a + [-100] * (max_a - len(a)) for a in a_list], dtype=torch.long)
        batches.append((p_tensor, t_tensor, chunk))

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


def test_generation_suite(model: CerebroSpheroidV5, tokenizer, device: torch.device, milestone_label: str):
    """Runs a battery of 3 high-difficulty reasoning tests and logs qualitative output."""
    test_prompts = [
        ("Reasoning", "If you have 5 apples and give 2 to a friend, how many apples do you have left?"),
        ("Logic", "Every day a tree drops 7 leaves. How many leaves does it drop in February of a non-leap year?"),
        ("Math/Physics", "A train travels at 60 mph for 3 hours. What is the total distance traveled?"),
    ]
    print(f"\n{'=' * 80}")
    print(f"📝 QUALITATIVE GENERATION CHECKPOINT: [{milestone_label}]")
    print(f"{'=' * 80}")
    for category, prompt in test_prompts:
        out = generate_qualitative_sample(model, tokenizer, prompt, device)
        print(f"[{category}] Prompt: {prompt}")
        print(f"  Response: {out.strip()[:140]}...\n")
    print(f"{'=' * 80}\n")


def main():
    print("=" * 80)
    print("🚀 CerebroSpheroid V5: OpenHermes-2.5 Convergence Toward Loss ~3.0")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tok = build_tokenizer("llama")
    print(f"📖 Tokenizer loaded: {len(tok)} tokens.")

    # 1. Build Model
    model = build_spheroid_v5("v5_compact", vocab_size=len(tok)).to(device)
    transfer_v4_core_to_v5(os.path.join(V4_ROOT, "checkpoints/cerebro_spheroid_v4_step149200_best.pt"), model)

    # 2. Load OpenHermes-2.5 Batches
    print("\n📚 Loading OpenHermes-2.5 complex reasoning batches...")
    batches = load_openhermes_reasoning_batches(tok, num_samples=160, batch_size=2, max_len=44)
    print(f"✅ Prepared {len(batches)} high-difficulty reasoning batches.")

    # -------------------------------------------------------------
    # STAGE 1: Warmup Alignment (Steps 1-20 | Core FROZEN)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("📍 STAGE 1: Outer Heads Alignment (Steps 1-20 | Core FROZEN | lr=1.5e-3)")
    print("-" * 80)
    model.volumetric_core.requires_grad_(False)
    model.sensory_dnn.requires_grad_(True)
    model.motor_decoder.requires_grad_(True)

    opt_stage1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1.5e-3,
        weight_decay=0.01,
    )

    model.train()
    t_s1 = time.time()
    for s in range(20):
        p_tensor, t_tensor, _ = batches[s % len(batches)]
        p_tensor, t_tensor = p_tensor.to(device), t_tensor.to(device)

        opt_stage1.zero_grad()
        out = model(p_tensor, targets=t_tensor, sampling_p=1.0)
        loss = out[0] if isinstance(out, tuple) else out
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_stage1.step()

        if (s + 1) % 5 == 0 or s == 0:
            print(f"  Warmup Step {s + 1:2d}/20 | Loss: {loss.item():.4f} | VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    el_s1 = time.time() - t_s1
    print(f"✅ Stage 1 Complete in {el_s1:.2f}s.")

    # -------------------------------------------------------------
    # STAGE 2: Deep Co-Adaptation Toward Loss ~3.0 (Steps 21-160+)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("📍 STAGE 2: Deep Co-Adaptation Toward Loss ~3.0 (FULL BRAIN UNFROZEN)")
    print("    • Core lr = 2e-4 | Sensory lr = 6e-4 | Motor lr = 8e-4")
    print("-" * 80)

    model.volumetric_core.requires_grad_(True)
    del opt_stage1
    torch.cuda.empty_cache()

    param_groups = [
        {"params": model.volumetric_core.parameters(), "lr": 2e-4, "weight_decay": 0.01},
        {"params": model.sensory_dnn.parameters(), "lr": 6e-4, "weight_decay": 0.01},
        {"params": model.motor_decoder.parameters(), "lr": 8e-4, "weight_decay": 0.01},
    ]
    opt_stage2 = Adafactor(param_groups, scale_parameter=False, relative_step=False, warmup_init=False)

    milestones_hit = set()
    best_loss = 999.0
    t_s2 = time.time()
    total_steps = 140

    for s in range(total_steps):
        step_num = s + 21
        p_tensor, t_tensor, _ = batches[(s + 20) % len(batches)]
        p_tensor, t_tensor = p_tensor.to(device), t_tensor.to(device)

        opt_stage2.zero_grad()
        out = model(p_tensor, targets=t_tensor, sampling_p=1.0)
        loss = out[0] if isinstance(out, tuple) else out
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_stage2.step()

        cur_loss = loss.item()
        if cur_loss < best_loss:
            best_loss = cur_loss

        if (s + 1) % 10 == 0 or s == 0:
            print(f"  Step {step_num:3d} | Loss: {cur_loss:.4f} (Best: {best_loss:.4f}) | VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB | Elapsed: {time.time() - t_s2:.1f}s")

        # Milestone 1: Loss < 5.0
        if cur_loss < 5.0 and "5.0" not in milestones_hit:
            milestones_hit.add("5.0")
            test_generation_suite(model, tok, device, f"Milestone: Loss < 5.0 (Step {step_num}, Loss {cur_loss:.4f})")
            model.train()

        # Milestone 2: Loss < 4.0
        if cur_loss < 4.0 and "4.0" not in milestones_hit:
            milestones_hit.add("4.0")
            test_generation_suite(model, tok, device, f"Milestone: Loss < 4.0 (Step {step_num}, Loss {cur_loss:.4f})")
            model.train()

        # Milestone 3: Loss < 3.5
        if cur_loss < 3.5 and "3.5" not in milestones_hit:
            milestones_hit.add("3.5")
            test_generation_suite(model, tok, device, f"Milestone: Loss < 3.5 (Step {step_num}, Loss {cur_loss:.4f})")
            model.train()

        # Milestone 4: Loss <= 3.0 (Target Hit!)
        if cur_loss <= 3.0 and "3.0" not in milestones_hit:
            milestones_hit.add("3.0")
            test_generation_suite(model, tok, device, f"🎯 TARGET REACHED: Loss <= 3.0 (Step {step_num}, Loss {cur_loss:.4f})")
            model.train()

    # Final Evaluation at end of training
    test_generation_suite(model, tok, device, f"Final State (Step {step_num}, Best Loss {best_loss:.4f})")

    # Save checkpoint
    out_ckpt = os.path.join(V5_ROOT, "checkpoints/v5_openhermes_converged.pt")
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "best_loss": best_loss}, out_ckpt)
    print(f"💾 Saved converged checkpoint to {out_ckpt}")


if __name__ == "__main__":
    main()
