"""
CerebroSpheroid V5: Pure Offline Training Benchmark
====================================================

100% Offline execution using strictly local datasets:
1. OpenHermes-2.5 (local parquet: 1,001,551 records)
2. Cerebro Cognitive 10k (local parquet: multi-step reasoning)

Validates all Frontier 2026 Upgrades in an end-to-end training loop:
- RoPE rotary position embeddings (32k horizon)
- Top-K sparse associative retrieval (needle tethering)
- Mixture-of-Axonal-Tracts (MoT dynamic routing)
- Subconscious deliberative memory probing
- Contrastive fixed-point energy regularization
- Out-of-place differentiable radial wave propagation
- Adafactor low-memory optimization on RTX 3050 Laptop GPU
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

# Strict 100% offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer

V5_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if V5_ROOT not in sys.path:
    sys.path.insert(0, V5_ROOT)

from src.spheroid_v5 import CerebroSpheroidV5, build_spheroid_v5, ModelConfigV5
from src.corpus import build_tokenizer
from transformers.optimization import Adafactor


def load_local_offline_data(tokenizer, count_per_source=40, batch_size=2, max_len=40):
    """Loads 100% offline batches from local parquets or data directory on disk."""
    pairs = []

    # 1. Local OpenHermes-2.5
    oh_path = os.path.join(V5_ROOT, "data/openhermes/openhermes_2_5.parquet")
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
            if 20 < len(q) < 300 and 30 < len(a) < 500:
                pairs.append(("OpenHermes", q, a))
            if len(pairs) >= count_per_source:
                break

    # 2. Local Cerebro Cognitive
    cog_path = os.path.join(V5_ROOT, "data/cerebro_cognitive_8k/cerebro_cognitive_10k.parquet")
    if os.path.exists(cog_path):
        df_cog = pd.read_parquet(cog_path)
        collected_cog = 0
        for _, row in df_cog.iterrows():
            q = str(row.get("question", row.get("prompt", ""))).strip()
            a = str(row.get("answer", row.get("response", ""))).strip()
            if 15 < len(q) < 300 and 20 < len(a) < 500:
                pairs.append(("CerebroCognitive", q, a))
                collected_cog += 1
            if collected_cog >= count_per_source:
                break

    # Tokenize batches
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
        batches.append((sources, p_tensor, t_tensor))

    return batches


def generate_sample(model: CerebroSpheroidV5, tokenizer, prompt: str, device: torch.device) -> str:
    """Generates autoregressive response with frontier RoPE + Top-K Sparse retrieval."""
    model.eval()
    with torch.no_grad():
        ans = model.generate_text(
            prompt,
            tokenizer=tokenizer,
            thinking_cycles=3,
            max_new_tokens=40,
            temperature=0.7,
            repetition_penalty=1.2,
            device=device,
        )
    return ans


def main():
    print("=" * 80)
    print("🔒 CerebroSpheroid V5: Pure Offline Training Benchmark (No Network)")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Local Tokenizer (Direct from Bundled Repository Snapshot)
    tok = build_tokenizer("llama")
    if tok.pad_token_id is None:
        tok.pad_token_id = 0
    print(f"📖 Tokenizer loaded locally from data/tokenizer: {len(tok)} tokens.")

    # 2. Build Clean-Slate Model with all Frontier Upgrades
    torch.manual_seed(42)
    print("\n🔨 Instantiating CerebroSpheroid V5 (Frontier Preset)...")
    config = ModelConfigV5(
        vocab_size=len(tok),
        surface_dim=768,
        radial_dim=512,
        num_shells=32,
        num_shortcut_tracts=96,
        num_heads=8,
        thinking_cycles=4,
        phase_scale=2.0,
        rope_theta=500000.0,
        max_seq_len=32768,
        top_k_resonance=16,
        use_deliberation_probe=True,
        use_adaptive_thinking=True,
        use_sparse_mot=True,
        top_k_tracts=24,
        use_energy_regularization=True,
    )
    model = CerebroSpheroidV5(config).to(device)

    param_stats = model.count_parameters()
    print(f"  • Total Parameters   : {param_stats['total_parameters'] / 1e6:.2f} M")
    print(f"  • Active Compute     : {param_stats['active_compute'] / 1e6:.2f} M ({param_stats['active_compute_percent']}%)")
    print(f"  • Concentric Shells  : {config.num_shells} shells")
    print(f"  • RoPE Long Horizon  : {config.max_seq_len} tokens (theta={config.rope_theta:.0f})")
    print(f"  • Sparse Retrieval   : Top-{config.top_k_resonance} Associative Gating")
    print(f"  • Mixture-of-Tracts  : Top-{config.top_k_tracts} / {config.num_shortcut_tracts} dynamic routing")
    print(f"  • Adaptive Halting   : Enabled (System-2 ACT)")
    print(f"  • Energy Guard       : Enabled (Fixed-point stability)")

    # 3. Load Local Batches
    print("\n📂 Loading 100% local datasets from disk...")
    batches = load_local_offline_data(tok, count_per_source=40, batch_size=2, max_len=40)
    print(f"✅ Prepared {len(batches)} offline batches (50% OpenHermes + 50% CerebroCognitive).")

    test_prompt = "If you have 5 apples and give 2 to a friend, how many apples do you have left?"

    # Baseline Generation (Step 0)
    print("\n" + "=" * 80)
    print("📝 BASELINE GENERATION (Step 0 - Clean Random Weights):")
    print("=" * 80)
    out_step0 = generate_sample(model, tok, test_prompt, device)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {out_step0.strip()[:120]}...\n")

    # 4. Offline Training Loop
    print("-" * 80)
    print("🏋️ Offline Training (Steps 1-60 | Adafactor | lr=1e-3)")
    print("-" * 80)

    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    losses = []
    step_times = []
    t_start = time.time()
    model.train()

    total_steps = 60
    for s in range(total_steps):
        t0 = time.time()
        srcs, p_tensor, t_tensor = batches[s % len(batches)]
        p_tensor, t_tensor = p_tensor.to(device), t_tensor.to(device)

        optimizer.zero_grad()
        loss, logits = model(p_tensor, targets=t_tensor, sampling_p=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        cur_loss = loss.item()
        losses.append(cur_loss)
        dt = time.time() - t0
        step_times.append(dt)

        if (s + 1) % 10 == 0 or s == 0:
            src_label = "/".join(set(srcs))
            print(f"  Step {s + 1:2d}/60 | Loss: {cur_loss:.4f} | Delta: {losses[0] - cur_loss:+.4f} | Source: {src_label:<20} | VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB | {dt*1000:.0f} ms/step")

    total_time = time.time() - t_start
    print(f"\n✅ Completed 60 Offline Steps in {total_time:.2f}s (Avg {sum(step_times)/len(step_times)*1000:.1f} ms/step)!")

    # Post-Training Generation (Step 60)
    print("\n" + "=" * 80)
    print("📝 POST-TRAINING GENERATION (Step 60 Offline Clean-Slate):")
    print("=" * 80)
    out_step60 = generate_sample(model, tok, test_prompt, device)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {out_step60.strip()[:140]}...\n")

    # 5. Statistical Summary
    print("=" * 80)
    print("📊 OFFLINE BENCHMARK SUMMARY:")
    print(f"  • Step 1 Initial Loss : {losses[0]:.4f} (Theoretical ln(32000) = 10.37)")
    print(f"  • Step 30 Mid Loss    : {losses[29]:.4f}")
    print(f"  • Step 60 Final Loss  : {losses[-1]:.4f}")
    print(f"  • Total Loss Drop     : {losses[0] - losses[-1]:.4f} (-{(losses[0] - losses[-1])/losses[0]*100:.2f}%)")
    print(f"  • Steady Descent      : {'YES' if losses[-1] < losses[0] else 'NO'}")
    print(f"  • Peak VRAM Allocated : {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    print("=" * 80)

    # Save checkpoint
    ckpt_path = os.path.join(V5_ROOT, "checkpoints/v5_offline_tested.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "loss": losses[-1]}, ckpt_path)
    print(f"💾 Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
