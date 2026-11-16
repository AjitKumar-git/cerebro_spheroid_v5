#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Progressive Cortical Expansion Engine
============================================================
Transfers pre-trained core and header weights from the 100M foundation model
(v5_small, 922.5k steps) into an expanded 32-shell Compact model (166.3M params),
freezing the established foundation and training only the outer cortical mantle!

Biological Inspiration:
  In mammalian brain evolution, the neocortex expands concentrically outward
  over an intact, functional subcortical/limbic core. The core and sensory-motor
  pathways remain structurally stable while the outer cortical laminae specialize
  in high-order deliberation, abstraction, and recursive reasoning.

Architecture Mapping:
  • Frozen Subcortical Core & Headers (101.96M params | 61.3%):
    - 32,000 Vocabulary Embedding Manifold (sensory.embedding.weight)
    - Contextual Multi-Head Self-Attention & Sensory SwiGLU
    - Concentric Core Shells 0 to 15 (intra-shell SwiGLU & RMSNorms)
    - Internal Radial Wave Propagation (transitions 0 to 14)
    - Primary Axonal Shortcut Tracts (tracts 0 to 39)
    - Motor Decoder, Motor SwiGLU, Plan Gates, & Affine Resonance Projections
  • Trainable Outer Cortical Mantle (64.37M params | 38.7%):
    - Outer Concentric Laminae Shells 16 to 31 (intra-shell SwiGLU & RMSNorms)
    - Outer Radial Wave Convection (transitions 15 to 30)
    - Extended Small-World Axonal Shortcut Tracts (tracts 40 to 95)
    - Outer Shell Scoring & Conductance Weights
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import numpy as np
import torch
from typing import Dict, Tuple, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
from src.jax.v5_model import ModelConfigV5, init_params_v5, training_rollout


def build_compact_32shell_config(
    surface_dim: int = 640,
    radial_dim: int = 256,
    num_shells: int = 32,
    num_shortcut_tracts: int = 96,
    thinking_cycles: int = 4,
) -> ModelConfigV5:
    """Creates a 32-shell compact configuration with 1:1 dimension parity for lossless core transfer."""
    return ModelConfigV5(
        vocab_size=32000,
        surface_dim=surface_dim,
        radial_dim=radial_dim,
        num_shells=num_shells,
        num_shortcut_tracts=num_shortcut_tracts,
        num_heads=8,
        top_k_tracts=24,
        thinking_cycles=thinking_cycles,
        phase_scale=2.0,
        use_tied_embeddings=True,
        use_swiglu=True,
    )


def transfer_and_freeze_cortical_expansion(
    compact_params: Dict[str, jnp.ndarray],
    src_pt_state: Dict[str, Any],
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], Dict[str, int]]:
    """
    Transfers pre-trained weights from 16-shell source state dict into 32-shell target parameters,
    and constructs binary gradient masks (0.0 = frozen, 1.0 = trainable).
    """
    transferred = {}
    frozen_masks = {}
    stats = {
        "total_params": 0,
        "frozen_params": 0,
        "trainable_params": 0,
        "exact_matches": 0,
        "expanded_tensors": 0,
        "new_tensors": 0,
    }

    for k, v in compact_params.items():
        v_np = np.array(v)
        stats["total_params"] += v_np.size

        if k in src_pt_state:
            s_tensor = src_pt_state[k]
            s_np = s_tensor.float().cpu().numpy()

            if v.shape == s_np.shape:
                # 1:1 Direct match (Sensory embedding, motor decoder, core projections)
                v_np = s_np.copy()
                mask = np.zeros_like(v_np)
                stats["frozen_params"] += v_np.size
                stats["exact_matches"] += 1
            elif len(v.shape) == len(s_np.shape) and all(v.shape[i] >= s_np.shape[i] for i in range(len(v.shape))):
                # Concentric shell expansion (16 -> 32 shells, 40 -> 96 tracts)
                mask = np.ones_like(v_np)
                if len(v.shape) == 3:
                    v_np[:s_np.shape[0], :s_np.shape[1], :s_np.shape[2]] = s_np
                    mask[:s_np.shape[0], :s_np.shape[1], :s_np.shape[2]] = 0.0
                elif len(v.shape) == 2:
                    v_np[:s_np.shape[0], :s_np.shape[1]] = s_np
                    mask[:s_np.shape[0], :s_np.shape[1]] = 0.0
                elif len(v.shape) == 1:
                    v_np[:s_np.shape[0]] = s_np
                    mask[:s_np.shape[0]] = 0.0

                stats["frozen_params"] += s_np.size
                stats["trainable_params"] += (v_np.size - s_np.size)
                stats["expanded_tensors"] += 1
            else:
                mask = np.ones_like(v_np)
                stats["trainable_params"] += v_np.size
                stats["new_tensors"] += 1
        else:
            mask = np.ones_like(v_np)
            stats["trainable_params"] += v_np.size
            stats["new_tensors"] += 1

        transferred[k] = jnp.array(v_np)
        frozen_masks[k] = jnp.array(mask)

    return transferred, frozen_masks, stats


def verify_cortical_expansion(ckpt_path: str = "checkpoints/v5_small_best.pt") -> bool:
    """Verifies that weights transfer cleanly, gradients are zeroed for frozen parts, and outer shells train."""
    print("=" * 70)
    print("🧠 CEREBROSPHEROID V5: PROGRESSIVE CORTICAL EXPANSION VERIFICATION")
    print("=" * 70)

    # 1. Config & init
    cfg = build_compact_32shell_config()
    key = jax.random.PRNGKey(42)
    compact_params, compact_buffers = init_params_v5(key, cfg)

    # 2. Load Source Checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    
    torch_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_state = torch_ckpt.get("model_state", torch_ckpt)
    step = torch_ckpt.get("step", "unknown")
    print(f"📦 Loaded pre-trained weights from {os.path.basename(ckpt_path)} (Step: {step})")

    # 3. Transfer and freeze
    params, masks, stats = transfer_and_freeze_cortical_expansion(compact_params, src_state)
    print(f"\n📊 Manifold Topology:")
    print(f"   • Total Compact Parameters  : {stats['total_params']:,} ({stats['total_params']/1e6:.2f}M)")
    print(f"   • Frozen Core & Headers      : {stats['frozen_params']:,} ({stats['frozen_params']/stats['total_params']*100:.1f}%)")
    print(f"   • Trainable Outer Laminae    : {stats['trainable_params']:,} ({stats['trainable_params']/stats['total_params']*100:.1f}%)")

    # 4. Forward & Gradient Verification
    print("\n⚡ Running forward rollout and gradient isolation check...")
    prompts = jnp.ones((2, 32), dtype=jnp.int32) * 42
    targets = jnp.ones((2, 64), dtype=jnp.int32) * 84

    def loss_fn(p):
        return training_rollout(
            params=p,
            prompts=prompts,
            target_len=64,
            buffers=compact_buffers,
            config=cfg,
            pad_id=0,
            targets=targets,
            remat=False,
            pure_teacher_forcing=True,
        )

    loss_before, raw_grads = jax.value_and_grad(loss_fn)(params)
    masked_grads = {k: raw_grads[k] * masks[k] for k in raw_grads}

    # Inspect critical gradients
    g_embed = float(jnp.max(jnp.abs(masked_grads["sensory.embedding.weight"])))
    g_core0 = float(jnp.max(jnp.abs(masked_grads["core.intra_swiglu.w_gate"][0])))
    g_outer20 = float(jnp.max(jnp.abs(masked_grads["core.intra_swiglu.w_gate"][20])))

    print(f"   • Frozen Sensory Embedding Grad : {g_embed:.6e} (Strictly 0.0: {'PASS' if g_embed == 0 else 'FAIL'})")
    print(f"   • Frozen Core Shell 0 Grad      : {g_core0:.6e} (Strictly 0.0: {'PASS' if g_core0 == 0 else 'FAIL'})")
    print(f"   • Active Outer Shell 20 Grad     : {g_outer20:.6e} (Active > 0.0: {'PASS' if g_outer20 > 0 else 'FAIL'})")

    assert g_embed == 0.0, "Sensory embedding gradient leaked!"
    assert g_core0 == 0.0, "Core shell gradient leaked!"
    assert g_outer20 > 0.0, "Outer shell gradient was zero!"

    # 5. Single SGD update on outer shells
    lr = 1e-3
    updated_params = {k: params[k] - lr * masked_grads[k] for k in params}
    loss_after = float(loss_fn(updated_params))
    delta = float(loss_before) - loss_after
    print(f"\n📉 Outer Shell Optimization Step:")
    print(f"   • Initial Loss : {float(loss_before):.4f}")
    print(f"   • Loss After   : {loss_after:.4f} (Drop: -{delta:.4f})")
    assert loss_after < float(loss_before), "Loss did not descend!"

    print("\n" + "=" * 70)
    print("✅ CORTICAL SHELL EXPANSION FULLY VERIFIED WITH 100% SUCCESS!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and run Progressive Cortical Expansion")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_small_best.pt", help="Source 100M checkpoint")
    args = parser.parse_args()

    verify_cortical_expansion(args.checkpoint)
