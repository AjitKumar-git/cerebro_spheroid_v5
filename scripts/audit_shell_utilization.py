#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Dynamic Shell Utilization & Information Flow Auditor

Audits empirical information flow across concentric laminae (Shells 0..K-1):
1. Gradient Utilization: G_k = ||dL/d theta_k||_2 normalized by param count and weight norm.
2. Activation Utilization: RMS magnitude, variance, and cross-shell cosine similarity S_ij.
3. Functional Ablation: Delta L_k = L_ablated,k - L_baseline (Zero & Mean ablation).
4. State Evolution: D_k = E[||s_k^(t+1) - s_k^(t)||_2] across autoregressive generation.
5. Radial Information Propagation: Influence Matrix I_ij = ||Delta s_j|| / ||delta s_i||.
"""

import os
import sys
import math
import argparse
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import jax
import jax.numpy as jnp
from jax import lax

from src.spheroid_v5 import infer_v5_config_from_state_dict
from src.jax.v5_model import ModelConfigV5, training_rollout, deliberate_step, deliberate_cycles_jax, encode_sensory
from src.jax.checkpoint import load_torch_pt_to_jax


def run_audit(checkpoint_path: str, preset: str = "v5_small", seq_len: int = 128):
    print("=" * 80)
    print("🧠 CEREBRO-SPHEROID V5: DYNAMIC SHELL UTILIZATION AUDIT")
    print(f"📁 Target Checkpoint: {checkpoint_path}")
    print("=" * 80)

    # 1. Load Tokenizer & Checkpoint Meta
    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Loaded Tokenizer from {tok_dir} (Vocab size: {len(tokenizer)})")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    cfg = infer_v5_config_from_state_dict(state, default_preset=preset)
    cfg.use_sensory_self_attn = True
    if preset == "v5_small":
        cfg.use_rope = False
        cfg.top_k_resonance = 0

    K = cfg.num_shells
    D = cfg.surface_dim
    print(f"✓ Configuration: {preset} | Shells (K): {K} | Surface Dim (D): {D}")

    # Load JAX Params & Buffers
    params, buffers, meta = load_torch_pt_to_jax(checkpoint_path, cfg, target_dtype=jnp.float32)
    step = meta.get("step", ckpt.get("step", "unknown"))
    loss_val = meta.get("loss", ckpt.get("loss", "unknown"))
    print(f"✓ JAX Engine Initialized | Step: {step} | Checkpoint Train Loss: {loss_val}")

    # 2. Prepare Fixed Validation Batch
    eval_texts = [
        "The fundamental equations of general relativity describe how spacetime curves in the presence of mass-energy.",
        "To implement a distributed key-value store with raft consensus, one must establish leader election and log replication.",
        "In molecular biology, ribosomal RNA facilitates peptide bond formation during protein synthesis in the cytoplasm.",
        "Deep learning architectures with residual connections preserve gradient propagation across hundreds of layers.",
    ]
    tok_res = tokenizer(eval_texts, padding="max_length", max_length=seq_len, truncation=True, return_tensors="np")
    input_ids = jnp.asarray(tok_res["input_ids"], dtype=jnp.int32)
    B, L = input_ids.shape
    prompts = input_ids[:, :L // 2]
    targets = input_ids[:, L // 2:]
    target_len = targets.shape[1]

    # --------------------------------------------------------------------------
    # MEASUREMENT A: GRADIENT UTILIZATION (G_k = ||dL/d theta_k||_2)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔬 MEASUREMENT A: GRADIENT UTILIZATION PER SHELL (G_k = ||dL/d theta_k||_2)")
    print("=" * 80)

    def loss_fn(p):
        return training_rollout(
            p, prompts, target_len=target_len, buffers=buffers, config=cfg, targets=targets
        )

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    print(f"🎯 Baseline Validation Loss: {float(loss_val):.4f}")

    grad_w_down = grads["core.intra_swiglu.w_down"]  # [K, 2D, D]
    grad_w_gate = grads["core.intra_swiglu.w_gate"]  # [K, D, 2D]
    grad_w_val  = grads["core.intra_swiglu.w_val"]   # [K, D, 2D]
    w_down = params["core.intra_swiglu.w_down"]
    w_gate = params["core.intra_swiglu.w_gate"]

    print(f"\n{'Shell k':<8} | {'||Grad w_down||':<15} | {'Rel Grad/Weight':<15} | {'||Grad w_gate||':<15} | {'Grad shell_gain':<15}")
    print("-" * 75)
    for k in range(K):
        g_d_norm = float(jnp.linalg.norm(grad_w_down[k]))
        w_d_norm = float(jnp.linalg.norm(w_down[k]))
        rel_d = g_d_norm / max(w_d_norm, 1e-6)
        g_g_norm = float(jnp.linalg.norm(grad_w_gate[k]))
        g_gain = float(grads["core.shell_gains"][k]) if "core.shell_gains" in grads else 0.0
        print(f"Shell {k:<3} | {g_d_norm:<15.4e} | {rel_d:<15.4e} | {g_g_norm:<15.4e} | {g_gain:<15.4e}")

    # --------------------------------------------------------------------------
    # MEASUREMENT B & D: ACTIVATION PROFILE & STATE EVOLUTION (D_k) ACROSS TIME
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔬 MEASUREMENT B & D: ACTIVATION RMS, DYNAMICS, AND STATE EVOLUTION D_k")
    print("=" * 80)

    # Initial state after prompt deliberation
    sensory_flux, sensory_kv = encode_sensory(params, prompts, config=cfg, pos_offset=0)
    prompt_flux = jnp.mean(sensory_flux, axis=1)
    radial_profile = (jnp.arange(1, K + 1, dtype=jnp.float32) / float(K))[None, :, None]
    init_state = jnp.broadcast_to(prompt_flux[:, None, :], (B, K, D)) * radial_profile
    delib_state, _ = deliberate_cycles_jax(params, init_state, buffers, config=cfg, is_training=False, sensory_kv=sensory_kv)

    # Track state step by step through emit_step
    from src.jax.v5_model import emit_step
    causal_keys, causal_values = sensory_kv
    cur_state = delib_state
    cur_plan = jnp.zeros((B, D), dtype=jnp.float32)
    cur_trace = None

    state_history = [cur_state]  # list of [B, K, D]
    for t in range(target_len):
        prev_tok = targets[:, t - 1:t] if t > 0 else jnp.full((B, 1), -100, dtype=jnp.int32)
        _, cur_state, cur_trace, cur_plan = emit_step(
            params,
            cur_state,
            (causal_keys, causal_values),
            cur_plan,
            cur_trace,
            t,
            buffers,
            config=cfg,
            query_position=prompts.shape[1] + t,
        )
        state_history.append(cur_state)

    state_trajectory = jnp.stack(state_history, axis=1)  # [B, T+1, K, D]

    # Compute step-to-step state displacement D_k = E[||s_k^(t+1) - s_k^(t)||_2]
    step_diffs = jnp.linalg.norm(state_trajectory[:, 1:, :, :] - state_trajectory[:, :-1, :, :], axis=-1)  # [B, T, K]
    mean_displacement = jnp.mean(step_diffs, axis=(0, 1))  # [K]
    total_trajectory_drift = jnp.mean(jnp.linalg.norm(state_trajectory[:, -1, :, :] - state_trajectory[:, 0, :, :], axis=-1), axis=0)  # [K]
    mean_rms = jnp.mean(jnp.sqrt(jnp.mean(jnp.square(state_trajectory), axis=-1)), axis=(0, 1))  # [K]

    print(f"\n{'Shell k':<8} | {'Mean RMS':<12} | {'Step Diff D_k':<15} | {'Total Drift (0->T)':<18} | {'Dynamic Status'}")
    print("-" * 75)
    for k in range(K):
        d_k = float(mean_displacement[k])
        drift_k = float(total_trajectory_drift[k])
        rms_k = float(mean_rms[k])
        status = "Active Recurrent Core" if d_k > 0.05 else ("Slow Drift" if d_k > 1e-4 else "FROZEN STATIC")
        print(f"Shell {k:<3} | {rms_k:<12.4f} | {d_k:<15.4e} | {drift_k:<18.4e} | {status}")

    # Cross-Shell Cosine Similarity at T_final
    final_s = state_trajectory[:, -1, :, :]  # [B, K, D]
    normed_s = final_s / (jnp.linalg.norm(final_s, axis=-1, keepdims=True) + 1e-6)
    sim_matrix = jnp.mean(jnp.einsum("bmd,bnd->bmn", normed_s, normed_s), axis=0)  # [K, K]

    print(f"\nCosine Similarity with Shell 0:")
    cos_with_0 = [f"Shell {k}: {float(sim_matrix[0, k]):.3f}" for k in range(K)]
    print(", ".join(cos_with_0[:8]))
    print(", ".join(cos_with_0[8:]))

    # --------------------------------------------------------------------------
    # MEASUREMENT C: FUNCTIONAL ABLATION (Delta L_k = L_ablated,k - L_baseline)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔬 MEASUREMENT C: FUNCTIONAL SHELL ABLATIONS (Delta L_k = L_ablated - L_baseline)")
    print("=" * 80)

    # Ablation by modifying emit_step state during rollout
    def eval_ablated_loss(ablate_shells: Optional[List[int]] = None, ablate_shell_pool: bool = False):
        def _ablated_rollout(p):
            # Forward rollout with ablated shell states
            B_loc, L_loc = prompts.shape
            s_flux, s_kv = encode_sensory(p, prompts, config=cfg, pos_offset=0)
            p_flux = jnp.mean(s_flux, axis=1)
            r_prof = (jnp.arange(1, K + 1, dtype=jnp.float32) / float(K))[None, :, None]
            s_init = jnp.broadcast_to(p_flux[:, None, :], (B_loc, K, D)) * r_prof
            s_delib, _ = deliberate_cycles_jax(p, s_init, buffers, config=cfg, is_training=False, sensory_kv=s_kv)

            # Apply ablation mask to initial deliberation state
            if ablate_shells is not None:
                mask = np.ones((1, K, 1), dtype=np.float32)
                for sh in ablate_shells:
                    mask[:, sh, :] = 0.0
                s_delib = s_delib * jnp.asarray(mask)

            # Run rollout
            c_keys, c_vals = s_kv
            c_s = s_delib
            c_p = jnp.zeros((B_loc, D), dtype=jnp.float32)
            c_t = None
            total_nll = 0.0

            for t_idx in range(target_len):
                if ablate_shells is not None:
                    mask = np.ones((1, K, 1), dtype=np.float32)
                    for sh in ablate_shells:
                        mask[:, sh, :] = 0.0
                    c_s = c_s * jnp.asarray(mask)

                logits, c_s, c_t, c_p = emit_step(
                    p,
                    c_s,
                    (c_keys, c_vals),
                    c_p,
                    c_t,
                    t_idx,
                    buffers,
                    config=cfg,
                    query_position=L_loc + t_idx,
                )
                tgt = targets[:, t_idx]
                log_p = jax.nn.log_softmax(logits, axis=-1)
                nll = -jnp.take_along_axis(log_p, tgt[:, None], axis=-1).squeeze(-1)
                total_nll += jnp.mean(nll)
            return total_nll / target_len

        return float(_ablated_rollout(params))

    base_l = float(loss_val)
    print(f"Baseline Loss: {base_l:.4f}")

    print(f"\n{'Ablation Experiment':<35} | {'Ablated Loss':<15} | {'Delta Loss (Delta L)':<15} | {'Functional Impact'}")
    print("-" * 80)

    # 1. Ablate individual shells
    for k_ab in [0, 1, 2, 7, 14, 15]:
        l_ab = eval_ablated_loss(ablate_shells=[k_ab])
        delta = l_ab - base_l
        impact = "CRITICAL PATH" if delta > 0.5 else ("Significant" if delta > 0.05 else "Marginal / Negligible")
        print(f"Zero-Ablate Shell {k_ab:<18} | {l_ab:<15.4f} | {delta:<+15.4f} | {impact}")

    # 2. Ablate all non-core shells (1..15) -> Core Only
    l_core_only = eval_ablated_loss(ablate_shells=list(range(1, K)))
    delta_core_only = l_core_only - base_l
    print(f"Core-Only (Ablate Shells 1..15)     | {l_core_only:<15.4f} | {delta_core_only:<+15.4f} | {'Core-Dominance Test'}")

    # 3. Ablate core (Shell 0) alone -> Outer-Only
    l_outer_only = eval_ablated_loss(ablate_shells=[0])
    delta_outer_only = l_outer_only - base_l
    print(f"Outer-Only (Ablate Shell 0)         | {l_outer_only:<15.4f} | {delta_outer_only:<+15.4f} | {'Outer-Mantle Sufficiency'}")

    # --------------------------------------------------------------------------
    # MEASUREMENT E: RADIAL INFLUENCE MATRIX (I_ij = ||Delta s_j|| / ||delta s_i||)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔬 MEASUREMENT E: RADIAL INFLUENCE MATRIX (I_ij = ||Delta s_j^(+1)|| / ||delta s_i^(0)||)")
    print("=" * 80)

    def measure_radial_influence(eps: float = 1e-2):
        # Base step
        s_base = delib_state[:1]  # [1, K, D]
        next_base = deliberate_step(params, s_base, buffers, config=cfg)

        influence_mat = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            pert = np.zeros((1, K, D), dtype=np.float32)
            pert[0, i, :] = eps
            s_pert = s_base + jnp.asarray(pert)
            next_pert = deliberate_step(params, s_pert, buffers, config=cfg)
            delta_out = np.array(jnp.linalg.norm(next_pert - next_base, axis=-1)[0])  # [K]
            influence_mat[i, :] = delta_out / eps
        return influence_mat

    inf_mat = measure_radial_influence()

    print("\nRadial Influence Matrix I_ij (Rows: Perturbed Shell i, Cols: Observed Shell j):")
    header = "      " + " ".join([f"S{j:<4}" for j in range(K)])
    print(header)
    for i in range(K):
        row_str = f"S{i:<4} " + " ".join([f"{inf_mat[i, j]:4.2f}" for j in range(K)])
        print(row_str)

    print("\n" + "=" * 80)
    print("✅ AUDIT COMPLETE: EMPIRICAL FINDINGS READY FOR STRATEGIC SYNTHESIS")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit CerebroSpheroid V5 Dynamic Shell Utilization")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_small_latest.pt", help="Path to checkpoint")
    parser.add_argument("--preset", type=str, default="v5_small", help="Model preset (v5_small)")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length for audit")
    args = parser.parse_args()

    run_audit(args.checkpoint, preset=args.preset, seq_len=args.seq_len)
