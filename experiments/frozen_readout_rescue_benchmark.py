"""
Frozen Readout Rescue Benchmark for CerebroSpheroid V5.
Strictly non-destructive, read-only with respect to foundation checkpoint and backbone parameters.

Stages:
Stage 1: Validation & Verification of Audit Baseline (multi-sample validation set, exact loss aggregation).
Stage 2: Frozen Readout Sweeps:
  - Experiment A: Pool Gain Sweep (beta in {0.0, 0.02, 0.05, 0.1, 0.2412, 0.5, 1.0, 2.0})
  - Experiment B: Shell Weighting Distributions (Current, Uniform, Core-excluded uniform, Interior-only 1-14, Surface-excluded)
  - Experiment C: Direct-Pathway Factorial Scale Sweep (Core gamma_0 in {0.0, 0.5, 1.0, 1.5, 2.0} x Pool beta in {0.0, 0.5, 1.0, 2.0})
Stage 3: Pool-Only Readout Calibration (Freeze backbone, optimize only motor.shell_score, motor.shell_scale, motor.shell_proj.weight on calibration set, evaluate on strictly held-out validation set).
Stage 4: Representation Similarity & Effective Rank (Linear CKA, Cosine Similarity Matrix, SVD Effective Rank of shell states).
"""

import os
import sys
import math
import json
from typing import Dict, Any, Tuple, List

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import torch

from src.jax.v5_model import (
    ModelConfigV5,
    generate_small_world_indices,
    rms_norm,
    swiglu,
)
from src.jax.spheroid_optimizer import init_spheroid_opt_state, _apply_spheroid_grads

CHECKPOINT_PATH = "checkpoints/v5_small_latest.pt"

def load_foundation_params() -> Tuple[Dict[str, jnp.ndarray], ModelConfigV5, Tuple[jnp.ndarray, jnp.ndarray]]:
    print(f"Loading foundation checkpoint: {CHECKPOINT_PATH} (read-only)...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    ms = ckpt["model_state"]
    cfg = ModelConfigV5(
        surface_dim=640,
        num_shells=16,
        radial_dim=256,
        num_shortcut_tracts=40,
        vocab_size=32000,
        num_heads=8,
        max_seq_len=4096,
        thinking_cycles=1,
    )
    params = {k: jnp.array(v.float().numpy()) for k, v in ms.items() if hasattr(v, "numpy")}
    src_idx, dst_idx = generate_small_world_indices(cfg.num_shells, cfg.num_shortcut_tracts)
    buffers = (src_idx, dst_idx)
    return params, cfg, buffers

def generate_evaluation_dataset(cfg: ModelConfigV5, num_batches: int = 8, batch_size: int = 4, seq_len: int = 32, seed: int = 42):
    """Generates calibration and held-out evaluation batches."""
    key = jax.random.PRNGKey(seed)
    batches = []
    head_dim = cfg.surface_dim // cfg.num_heads
    
    for _ in range(num_batches):
        k1, k2, k3, k4, key = jax.random.split(key, 5)
        sensory_k = jax.random.normal(k1, (batch_size, cfg.num_heads, seq_len, head_dim))
        sensory_v = jax.random.normal(k2, (batch_size, cfg.num_heads, seq_len, head_dim))
        sensory_kv = (sensory_k, sensory_v)
        state = jax.random.normal(k3, (batch_size, cfg.num_shells, cfg.surface_dim))
        plan = jnp.zeros((batch_size, cfg.surface_dim), dtype=jnp.float32)
        targets = jax.random.randint(k4, (batch_size,), 0, cfg.vocab_size)
        batches.append((state, sensory_kv, plan, targets))
    return batches

def forward_readout(params: Dict[str, jnp.ndarray], state: jnp.ndarray, sensory_kv: Tuple[jnp.ndarray, jnp.ndarray], plan: jnp.ndarray, cfg: ModelConfigV5, overrides: Dict[str, Any] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes logits and fused_in with flexible parametric overrides:
    - pool_scale: float override for motor.shell_scale
    - core_scale: float multiplier on core pathway (default 1.0)
    - surface_scale: float multiplier on surface pathway (default 1.0)
    - shell_weights: shape [16] explicit weights for pooling
    - pool_proj_weight: override for motor.shell_proj.weight
    - state_override: override full state
    """
    st = state if (overrides is None or "state_override" not in overrides) else overrides["state_override"]
    B, K, D = st.shape
    surface = st[:, -1, :]
    core = st[:, 0, :]
    num_heads = cfg.num_heads
    head_dim = D // num_heads

    # Step idx 0
    phase = params["motor.phase_embed.weight"][0]
    p_norm = jnp.maximum(jnp.linalg.norm(phase, axis=-1, keepdims=True), 1e-6)
    p_scale_val = jnp.clip(params.get("motor.phase_scale", 2.0), 0.05, 5.0)
    p_scaled = phase * (p_scale_val / p_norm)

    surf_mult = 1.0 if (overrides is None or "surface_scale" not in overrides) else overrides["surface_scale"]
    motor_surface = surf_mult * surface + p_scaled

    # Resonance
    keys, values = sensory_kv
    q = jnp.matmul(motor_surface, params["motor.q_proj.weight"].T).reshape(B, 1, num_heads, head_dim)
    q = jnp.swapaxes(q, 1, 2)
    scores = jnp.matmul(q, jnp.swapaxes(keys, -1, -2)) / math.sqrt(head_dim)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    res_heads = jnp.matmul(attn_weights, values)
    res_heads = jnp.swapaxes(res_heads, 1, 2).reshape(B, D)
    resonance = rms_norm(jnp.matmul(res_heads, params["motor.out_proj.weight"].T), params["motor.resonance_norm.weight"])

    # Multi-shell pool
    if overrides is not None and "shell_weights" in overrides:
        shell_weights = overrides["shell_weights"]
    else:
        shell_weights = jax.nn.softmax(params["motor.shell_score"], axis=0) # [16]

    pool_proj = params["motor.shell_proj.weight"] if (overrides is None or "pool_proj_weight" not in overrides) else overrides["pool_proj_weight"]
    shell_proj = jnp.matmul(st, pool_proj.T)

    pool_beta = params["motor.shell_scale"] if (overrides is None or "pool_scale" not in overrides) else overrides["pool_scale"]
    shell_pool = jnp.sum(shell_weights[None, :, None] * shell_proj, axis=1) * pool_beta

    # Fused inputs
    ig = jnp.clip(params.get("motor.input_gate", jnp.ones(5, dtype=jnp.float32)), -5.0, 5.0)
    core_mult = 1.0 if (overrides is None or "core_scale" not in overrides) else overrides["core_scale"]
    
    p0 = ig[0] * motor_surface
    p1 = ig[1] * (core_mult * core)
    p2 = ig[2] * resonance
    p3 = ig[3] * shell_pool
    p4 = ig[4] * plan

    fused_in = jnp.concatenate([p0, p1, p2, p3, p4], axis=-1)
    fused = rms_norm(fused_in, params["motor.fused_norm.weight"])
    h = rms_norm(
        swiglu(
            fused,
            params["motor.swiglu.fc_gate.weight"],
            params["motor.swiglu.fc_val.weight"],
            params["motor.swiglu.fc_down.weight"],
            use_swiglu=True,
        ),
        params["motor.norm_mid.weight"],
    )

    temp = jnp.clip(params["motor.temperature_scale"], 0.1, 5.0)
    affine_scale = params["motor.affine_scale"][None, :]
    h_scaled = (h * affine_scale) / temp
    w_out = params.get("motor.head.weight", params["sensory.embedding.weight"])
    logits = jnp.matmul(h_scaled, w_out.T)
    return logits, fused_in

def evaluate_nll(params: Dict[str, jnp.ndarray], batches: List[Tuple], cfg: ModelConfigV5, overrides: Dict[str, Any] = None) -> float:
    total_loss = 0.0
    total_tokens = 0
    for state, sensory_kv, plan, targets in batches:
        logits, _ = forward_readout(params, state, sensory_kv, plan, cfg, overrides)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(log_probs, targets[:, None], axis=-1)
        total_loss += float(jnp.sum(nll))
        total_tokens += targets.shape[0]
    return total_loss / max(1, total_tokens)

# ----------------------------------------------------
# Stage 4 Metrics: Representation Similarity & Rank
# ----------------------------------------------------
def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Computes Linear Centered Kernel Alignment (CKA) between two representation matrices [N, D]."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    dot_X = X.dot(X.T)
    dot_Y = Y.dot(Y.T)
    tr_XY = np.sum(dot_X * dot_Y)
    tr_XX = np.sum(dot_X * dot_X)
    tr_YY = np.sum(dot_Y * dot_Y)
    denom = np.sqrt(tr_XX * tr_YY) + 1e-10
    return float(tr_XY / denom)

def compute_effective_rank(X: np.ndarray) -> float:
    """Computes Roy & Vetterli (2007) entropy-based effective rank from singular values."""
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 1e-7]
    p = s / np.sum(s)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))

def run_benchmark():
    params, cfg, buffers = load_foundation_params()
    
    # Generate calibration and held-out validation batches
    print("Generating calibration and held-out validation batches...")
    calib_batches = generate_evaluation_dataset(cfg, num_batches=6, batch_size=4, seq_len=32, seed=101)
    val_batches = generate_evaluation_dataset(cfg, num_batches=10, batch_size=4, seq_len=32, seed=2026)

    # ----------------------------------------------------
    # Stage 1: Validate Audit Baseline on Multi-Sample Held-Out Set
    # ----------------------------------------------------
    base_nll = evaluate_nll(params, val_batches, cfg)
    print(f"\n[Stage 1] Multi-Sample Held-Out Base NLL: {base_nll:.6f}")
    
    # Confirm baseline ablations on held-out set
    # Ablate Core
    nll_core_zero = evaluate_nll(params, val_batches, cfg, {"core_scale": 0.0})
    # Ablate Interior
    # Construct zero interior state
    val_interior_zero = [(b[0].at[:, 1:15, :].set(0.0), b[1], b[2], b[3]) for b in val_batches]
    nll_int_zero = evaluate_nll(params, val_interior_zero, cfg)
    # Ablate Pool entirely (beta = 0.0)
    nll_pool_zero = evaluate_nll(params, val_batches, cfg, {"pool_scale": 0.0})

    stage1_results = {
        "base_nll": base_nll,
        "delta_core_zero": nll_core_zero - base_nll,
        "delta_interior_zero": nll_int_zero - base_nll,
        "delta_pool_zero": nll_pool_zero - base_nll,
    }
    print(f"  Delta Core Zero:     {stage1_results['delta_core_zero']:+.6f}")
    print(f"  Delta Interior Zero: {stage1_results['delta_interior_zero']:+.6f}")
    print(f"  Delta Pool Zero:     {stage1_results['delta_pool_zero']:+.6f}")

    # ----------------------------------------------------
    # Stage 2: Frozen Readout Sweeps
    # ----------------------------------------------------
    print("\n[Stage 2] Running Frozen Readout Sweeps...")
    
    # Exp A: Pool Gain Sweep (beta)
    betas = [0.0, 0.02, 0.05, 0.1, 0.2412, 0.5, 1.0, 2.0]
    pool_sweep = {}
    for b in betas:
        loss_b = evaluate_nll(params, val_batches, cfg, {"pool_scale": b})
        pool_sweep[str(b)] = {"nll": loss_b, "delta": loss_b - base_nll}
        print(f"  Pool Gain beta={b:6.4f}: NLL={loss_b:.6f} (Delta={loss_b - base_nll:+.6f})")

    # Exp B: Shell Weighting Distribution
    weight_schemes = {}
    # 1. Current learned
    current_w = jax.nn.softmax(params["motor.shell_score"], axis=0)
    weight_schemes["Current Learned"] = current_w
    # 2. Uniform (all 16 shells)
    weight_schemes["Uniform (All 16)"] = jnp.ones(16, dtype=jnp.float32) / 16.0
    # 3. Core-excluded uniform (1-15)
    w_core_ex = jnp.zeros(16, dtype=jnp.float32).at[1:].set(1.0 / 15.0)
    weight_schemes["Core-Excluded Uniform (1-15)"] = w_core_ex
    # 4. Interior-only uniform (1-14)
    w_int_only = jnp.zeros(16, dtype=jnp.float32).at[1:15].set(1.0 / 14.0)
    weight_schemes["Interior-Only Uniform (1-14)"] = w_int_only
    # 5. Surface-excluded uniform (0-14)
    w_surf_ex = jnp.zeros(16, dtype=jnp.float32).at[:15].set(1.0 / 15.0)
    weight_schemes["Surface-Excluded Uniform (0-14)"] = w_surf_ex

    weighting_results = {}
    for name, w_dist in weight_schemes.items():
        loss_w = evaluate_nll(params, val_batches, cfg, {"shell_weights": w_dist})
        weighting_results[name] = {"nll": loss_w, "delta": loss_w - base_nll}
        print(f"  Weighting '{name}': NLL={loss_w:.6f} (Delta={loss_w - base_nll:+.6f})")

    # Exp C: Direct-Pathway Factorial Scale Sweep (Core gamma_0 x Pool beta)
    core_gammas = [0.0, 0.5, 1.0, 1.5, 2.0]
    pool_betas_c = [0.0, 0.1, 0.2412, 0.5, 1.0]
    factorial_results = {}
    print("  Factorial Sweep (Core gamma_0 x Pool beta):")
    for g in core_gammas:
        factorial_results[str(g)] = {}
        for b in pool_betas_c:
            loss_gb = evaluate_nll(params, val_batches, cfg, {"core_scale": g, "pool_scale": b})
            factorial_results[str(g)][str(b)] = {"nll": loss_gb, "delta": loss_gb - base_nll}
            print(f"    gamma_0={g:.1f}, beta={b:.4f}: NLL={loss_gb:.6f} (Delta={loss_gb - base_nll:+.6f})")

    # ----------------------------------------------------
    # Stage 3: Pool-Only Readout Calibration
    # ----------------------------------------------------
    print("\n[Stage 3] Running Pool-Only Readout Calibration (Backbone Frozen)...")
    # Calibrate only: motor.shell_score [16], motor.shell_scale [], motor.shell_proj.weight [640, 640]
    calib_params = {
        "motor.shell_score": jnp.array(params["motor.shell_score"]),
        "motor.shell_scale": jnp.array(params["motor.shell_scale"]),
        "motor.shell_proj.weight": jnp.array(params["motor.shell_proj.weight"]),
    }
    
    # 50 steps of Adam on calib_params using calibration dataset
    lr = 1e-3
    m_buf = {k: jnp.zeros_like(v) for k, v in calib_params.items()}
    v_buf = {k: jnp.zeros_like(v) for k, v in calib_params.items()}
    
    def calib_loss_fn(p_train, batch):
        st, skv, pl, tgt = batch
        # Merge trained parameters with frozen params
        merged_p = dict(params)
        merged_p.update(p_train)
        logits, _ = forward_readout(merged_p, st, skv, pl, cfg)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(log_probs, tgt[:, None], axis=-1)
        return jnp.mean(nll)

    print("  Calibrating 50 steps on calibration set...")
    cur_p = calib_params
    for step in range(50):
        batch = calib_batches[step % len(calib_batches)]
        loss_val, grads = jax.value_and_grad(calib_loss_fn)(cur_p, batch)
        # Adam update
        new_p = {}
        for k in cur_p:
            g = grads[k]
            m_buf[k] = 0.9 * m_buf[k] + 0.1 * g
            v_buf[k] = 0.99 * v_buf[k] + 0.01 * (g ** 2)
            m_hat = m_buf[k] / (1.0 - (0.9 ** (step + 1)))
            v_hat = v_buf[k] / (1.0 - (0.99 ** (step + 1)))
            new_p[k] = cur_p[k] - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        cur_p = new_p

    # Evaluate calibrated readout on held-out validation set
    merged_calib_params = dict(params)
    merged_calib_params.update(cur_p)
    calib_heldout_nll = evaluate_nll(merged_calib_params, val_batches, cfg)
    
    # Also evaluate core-only baseline on held-out set
    core_only_nll = evaluate_nll(params, val_batches, cfg, {"pool_scale": 0.0, "surface_scale": 0.0})
    
    stage3_results = {
        "baseline_nll": base_nll,
        "calibrated_pool_nll": calib_heldout_nll,
        "calibrated_delta": calib_heldout_nll - base_nll,
        "core_only_nll": core_only_nll,
        "calibrated_shell_scale": float(cur_p["motor.shell_scale"]),
        "calibrated_shell_weights": [float(x) for x in jax.nn.softmax(cur_p["motor.shell_score"], axis=0)],
    }
    print(f"  Calibrated Held-Out NLL: {calib_heldout_nll:.6f} (Delta={calib_heldout_nll - base_nll:+.6f})")
    print(f"  Core-Only Held-Out NLL:  {core_only_nll:.6f} (Delta={core_only_nll - base_nll:+.6f})")
    print(f"  Calibrated Shell Scale:  {stage3_results['calibrated_shell_scale']:.4f}")

    # ----------------------------------------------------
    # Stage 4: Representation Similarity & Effective Rank
    # ----------------------------------------------------
    print("\n[Stage 4] Computing Representation Similarities & Effective Ranks...")
    # Collect flattened shell state activations across validation batches
    # Shape per shell: [total_samples, D]
    all_states = [b[0] for b in val_batches] # list of [B, K, D]
    cat_states = np.array(jnp.concatenate(all_states, axis=0)) # [N, K, D]
    N, K, D = cat_states.shape

    cka_matrix = np.zeros((K, K))
    cosine_matrix = np.zeros((K, K))
    effective_ranks = []

    for i in range(K):
        X_i = cat_states[:, i, :]
        effective_ranks.append(compute_effective_rank(X_i))
        norm_i = X_i / (np.linalg.norm(X_i, axis=-1, keepdims=True) + 1e-8)
        for j in range(i, K):
            X_j = cat_states[:, j, :]
            norm_j = X_j / (np.linalg.norm(X_j, axis=-1, keepdims=True) + 1e-8)
            cka_val = compute_linear_cka(X_i, X_j)
            cos_val = float(np.mean(np.sum(norm_i * norm_j, axis=-1)))
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val
            cosine_matrix[i, j] = cos_val
            cosine_matrix[j, i] = cos_val

    stage4_results = {
        "effective_ranks": effective_ranks,
        "mean_interior_rank": float(np.mean(effective_ranks[1:15])),
        "core_rank": effective_ranks[0],
        "surface_rank": effective_ranks[15],
        "cka_matrix": cka_matrix.tolist(),
        "cosine_matrix": cosine_matrix.tolist(),
    }
    print(f"  Core Effective Rank (S0):         {effective_ranks[0]:.2f} / {min(N, D)}")
    print(f"  Mean Interior Effective Rank:     {stage4_results['mean_interior_rank']:.2f}")
    print(f"  Surface Effective Rank (S15):     {effective_ranks[15]:.2f}")
    print(f"  Mean Interior CKA (Shells 1-14):  {float(np.mean(cka_matrix[1:15, 1:15])):.4f}")

    # Final compilation
    benchmark_report = {
        "stage1_baseline": stage1_results,
        "stage2_sweeps": {
            "pool_gain_sweep": pool_sweep,
            "weighting_schemes": weighting_results,
            "factorial_sweep": factorial_results,
        },
        "stage3_calibration": stage3_results,
        "stage4_representations": stage4_results,
    }

    out_file = "experiments/frozen_readout_rescue_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\nBenchmark completed successfully. Full data persisted to: {out_file}")

if __name__ == "__main__":
    run_benchmark()
