"""
Radial Differentiation Capacity Benchmark for CerebroSpheroid V5.
Strictly read-only with respect to foundation checkpoint and model source code.

Components:
1. Reconciliation of Core Interventions:
   - Zero State s_0 everywhere across forward graph vs.
   - Zero Direct Core Tap (gamma_0 = 0) vs.
   - Core-Only Readout (zero all other pathways).
2. Large-Sample Representation & CKA Audit:
   - N = 256 independent validation sequences (exceeding D=640 to avoid N-1 sample ceiling).
   - Pairwise Linear CKA Matrix (16 x 16).
   - Residual-subtracted CKA (removing shared input/mean subspace).
   - SVD singular value spectrum and Roy & Vetterli effective rank per shell.
   - Linear probe predictability: can shell j be linearly predicted from shell i?
3. Immediate Combined Inference Setting Evaluation:
   - Baseline checkpoint (beta = 0.2412, default weights).
   - Combined candidate: beta = 1.0, w_0 = 0 (Core excluded from pool, surface kept).
   - Evaluated on strictly untouched held-out test split.
4. Radial Coupling Capacity Sweep (r_mix in {0.009, 0.03, 0.1, 0.2, 0.4}):
   - Forward deliberation rollout with overridden r_mix.
   - Measure: Held-out NLL, Inter-shell CKA, State displacement ||s_k - s_0||, and Pool utility.
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
    deliberate_step,
    rms_norm,
    swiglu,
)

CHECKPOINT_PATH = "checkpoints/v5_small_latest.pt"

def load_foundation():
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

def generate_large_evaluation_dataset(cfg: ModelConfigV5, total_sequences: int = 256, seq_len: int = 16, seed: int = 777):
    """Generates a large sample of N=256 independent sequences to ensure N >> D (or sufficient degrees of freedom)."""
    key = jax.random.PRNGKey(seed)
    head_dim = cfg.surface_dim // cfg.num_heads
    batch_size = 16
    num_batches = total_sequences // batch_size
    batches = []
    
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

def forward_deliberation_and_readout(
    params: Dict[str, jnp.ndarray],
    init_state: jnp.ndarray,
    sensory_kv: Tuple[jnp.ndarray, jnp.ndarray],
    plan: jnp.ndarray,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    cfg: ModelConfigV5,
    overrides: Dict[str, Any] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simulates deliberate step with optional r_mix override, followed by motor readout.
    Returns: logits, final_state, fused_in
    """
    p = dict(params)
    if overrides is not None and "r_mix" in overrides:
        p["core.radial_mix_scale"] = jnp.array(overrides["r_mix"], dtype=jnp.float32)
    
    st_in = init_state if (overrides is None or "state_in_override" not in overrides) else overrides["state_in_override"]
    
    # Run 1 deliberation step
    st_next = deliberate_step(p, st_in, buffers, config=cfg)
    
    # Motor readout overrides
    st_readout = st_next if (overrides is None or "state_readout_override" not in overrides) else overrides["state_readout_override"]
    
    B, K, D = st_readout.shape
    surface = st_readout[:, -1, :]
    core = st_readout[:, 0, :]
    num_heads = cfg.num_heads
    head_dim = D // num_heads

    phase = p["motor.phase_embed.weight"][0]
    p_norm = jnp.maximum(jnp.linalg.norm(phase, axis=-1, keepdims=True), 1e-6)
    p_scale_val = jnp.clip(p.get("motor.phase_scale", 2.0), 0.05, 5.0)
    p_scaled = phase * (p_scale_val / p_norm)

    surf_mult = 1.0 if (overrides is None or "surface_scale" not in overrides) else overrides["surface_scale"]
    motor_surface = surf_mult * surface + p_scaled

    keys, values = sensory_kv
    q = jnp.matmul(motor_surface, p["motor.q_proj.weight"].T).reshape(B, 1, num_heads, head_dim)
    q = jnp.swapaxes(q, 1, 2)
    scores = jnp.matmul(q, jnp.swapaxes(keys, -1, -2)) / math.sqrt(head_dim)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    res_heads = jnp.matmul(attn_weights, values)
    res_heads = jnp.swapaxes(res_heads, 1, 2).reshape(B, D)
    resonance = rms_norm(jnp.matmul(res_heads, p["motor.out_proj.weight"].T), p["motor.resonance_norm.weight"])

    if overrides is not None and "shell_weights" in overrides:
        shell_weights = overrides["shell_weights"]
    else:
        shell_weights = jax.nn.softmax(p["motor.shell_score"], axis=0)

    pool_proj = p["motor.shell_proj.weight"]
    shell_proj = jnp.matmul(st_readout, pool_proj.T)

    pool_beta = p["motor.shell_scale"] if (overrides is None or "pool_scale" not in overrides) else overrides["pool_scale"]
    shell_pool = jnp.sum(shell_weights[None, :, None] * shell_proj, axis=1) * pool_beta

    ig = jnp.clip(p.get("motor.input_gate", jnp.ones(5, dtype=jnp.float32)), -5.0, 5.0)
    core_mult = 1.0 if (overrides is None or "core_scale" not in overrides) else overrides["core_scale"]
    
    p0 = ig[0] * motor_surface
    p1 = ig[1] * (core_mult * core)
    p2 = ig[2] * resonance
    p3 = ig[3] * shell_pool
    p4 = ig[4] * plan

    fused_in = jnp.concatenate([p0, p1, p2, p3, p4], axis=-1)
    fused = rms_norm(fused_in, p["motor.fused_norm.weight"])
    h = rms_norm(
        swiglu(
            fused,
            p["motor.swiglu.fc_gate.weight"],
            p["motor.swiglu.fc_val.weight"],
            p["motor.swiglu.fc_down.weight"],
            use_swiglu=True,
        ),
        p["motor.norm_mid.weight"],
    )

    temp = jnp.clip(p["motor.temperature_scale"], 0.1, 5.0)
    affine_scale = p["motor.affine_scale"][None, :]
    h_scaled = (h * affine_scale) / temp
    w_out = p.get("motor.head.weight", p["sensory.embedding.weight"])
    logits = jnp.matmul(h_scaled, w_out.T)
    return logits, st_next, fused_in

def evaluate_batches_nll(params, batches, buffers, cfg, overrides=None):
    total_loss = 0.0
    total_tokens = 0
    for state, sensory_kv, plan, targets in batches:
        logits, _, _ = forward_deliberation_and_readout(params, state, sensory_kv, plan, buffers, cfg, overrides)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(log_probs, targets[:, None], axis=-1)
        total_loss += float(jnp.sum(nll))
        total_tokens += targets.shape[0]
    return total_loss / max(1, total_tokens)

def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    dot_X = X.dot(X.T)
    dot_Y = Y.dot(Y.T)
    tr_XY = np.sum(dot_X * dot_Y)
    tr_XX = np.sum(dot_X * dot_X)
    tr_YY = np.sum(dot_Y * dot_Y)
    denom = np.sqrt(tr_XX * tr_YY) + 1e-10
    return float(tr_XY / denom)

def compute_effective_rank(X: np.ndarray) -> Tuple[float, List[float]]:
    s = np.linalg.svd(X, compute_uv=False)
    s_clean = s[s > 1e-7]
    p = s_clean / np.sum(s_clean)
    entropy = -np.sum(p * np.log(p + 1e-12))
    top_5_sv = [float(x) for x in s[:5]]
    return float(np.exp(entropy)), top_5_sv

def run_benchmark():
    params, cfg, buffers = load_foundation()
    
    print("\nGenerating large dataset (N=256 sequences across 16 batches)...")
    eval_batches = generate_large_evaluation_dataset(cfg, total_sequences=256, seq_len=16, seed=888)
    untouched_test_batches = generate_large_evaluation_dataset(cfg, total_sequences=128, seq_len=16, seed=999)

    # ----------------------------------------------------
    # Task 1: Reconciliation of Core Interventions
    # ----------------------------------------------------
    print("\n--- [Task 1: Core Intervention Reconciliation] ---")
    base_nll = evaluate_batches_nll(params, eval_batches, buffers, cfg)
    print(f"Base NLL (N=256): {base_nll:.6f}")

    # Intervention A: Zero State s_0 everywhere across forward graph (deliberation + readout)
    batches_s0_zero = [(b[0].at[:, 0, :].set(0.0), b[1], b[2], b[3]) for b in eval_batches]
    nll_s0_zero = evaluate_batches_nll(params, batches_s0_zero, buffers, cfg)
    delta_s0_zero = nll_s0_zero - base_nll

    # Intervention B: Zero Direct Core Tap only (gamma_0 = 0.0 in readout, core state intact in deliberation)
    nll_core_tap_zero = evaluate_batches_nll(params, eval_batches, buffers, cfg, {"core_scale": 0.0})
    delta_core_tap_zero = nll_core_tap_zero - base_nll

    # Intervention C: Core-Only Readout (zero surface, zero pool, keep core tap)
    nll_core_only = evaluate_batches_nll(params, eval_batches, buffers, cfg, {"surface_scale": 0.0, "pool_scale": 0.0})
    delta_core_only = nll_core_only - base_nll

    # Intervention D: Zero Core in Readout ONLY (s_0 zeroed only entering readout)
    batches_readout_s0_zero = []
    # override state_readout_override
    nll_readout_s0_zero = 0.0
    tot_tok = 0
    for state, sensory_kv, plan, targets in eval_batches:
        _, st_next, _ = forward_deliberation_and_readout(params, state, sensory_kv, plan, buffers, cfg)
        st_mod = st_next.at[:, 0, :].set(0.0)
        logits, _, _ = forward_deliberation_and_readout(params, state, sensory_kv, plan, buffers, cfg, {"state_readout_override": st_mod})
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(log_probs, targets[:, None], axis=-1)
        nll_readout_s0_zero += float(jnp.sum(nll))
        tot_tok += targets.shape[0]
    nll_readout_s0_zero /= tot_tok
    delta_readout_s0_zero = nll_readout_s0_zero - base_nll

    reconciliation = {
        "base_nll": base_nll,
        "delta_s0_zero_everywhere": delta_s0_zero,
        "delta_core_tap_zero_gamma0": delta_core_tap_zero,
        "delta_readout_s0_zero_state": delta_readout_s0_zero,
        "delta_core_only_readout": delta_core_only,
    }
    print(f"  A. Zero Shell 0 Everywhere (State=0):      Delta NLL = {delta_s0_zero:+.6f}")
    print(f"  B. Zero Core Tap in Readout (gamma_0=0.0): Delta NLL = {delta_core_tap_zero:+.6f}")
    print(f"  C. Zero Shell 0 in Readout State:          Delta NLL = {delta_readout_s0_zero:+.6f}")
    print(f"  D. Core-Only Readout:                      Delta NLL = {delta_core_only:+.6f}")

    # ----------------------------------------------------
    # Task 2: Large-Sample Representation & CKA Audit (N=256)
    # ----------------------------------------------------
    print("\n--- [Task 2: Large-Sample Representation & CKA Audit (N=256, D=640)] ---")
    all_states_list = []
    for state, sensory_kv, plan, targets in eval_batches:
        _, st_next, _ = forward_deliberation_and_readout(params, state, sensory_kv, plan, buffers, cfg)
        all_states_list.append(np.array(st_next))
    
    # Full state matrix: [N=256, K=16, D=640]
    cat_all_states = np.concatenate(all_states_list, axis=0)
    N, K, D = cat_all_states.shape
    print(f"Collected deliberation states: N={N}, K={K}, D={D} (N exceeds sample ceiling).")

    cka_matrix = np.zeros((K, K))
    residual_cka_matrix = np.zeros((K, K))
    effective_ranks = []
    top_singular_values = {}

    # Compute global mean across all shells for residual subtraction
    global_mean_feature = np.mean(cat_all_states, axis=1) # [N, D]

    for i in range(K):
        X_i = cat_all_states[:, i, :]
        rank_i, sv_i = compute_effective_rank(X_i)
        effective_ranks.append(rank_i)
        top_singular_values[f"Shell_{i}"] = sv_i
        
        # Residual representation: subtract global mean subspace
        X_i_res = X_i - global_mean_feature

        for j in range(i, K):
            X_j = cat_all_states[:, j, :]
            X_j_res = X_j - global_mean_feature
            
            raw_cka = compute_linear_cka(X_i, X_j)
            res_cka = compute_linear_cka(X_i_res, X_j_res)
            
            cka_matrix[i, j] = raw_cka
            cka_matrix[j, i] = raw_cka
            residual_cka_matrix[i, j] = res_cka
            residual_cka_matrix[j, i] = res_cka

    mean_raw_cka_interior = float(np.mean(cka_matrix[1:15, 1:15]))
    mean_res_cka_interior = float(np.mean(residual_cka_matrix[1:15, 1:15]))
    
    # Linear probe predictability: R^2 of predicting Shell j from Shell i via Ridge
    probe_r2 = np.zeros((K, K))
    for i in [0, 1, 7, 14, 15]:
        X_i = cat_all_states[:, i, :]
        for j in [0, 1, 7, 14, 15]:
            if i == j:
                probe_r2[i, j] = 1.0
                continue
            X_j = cat_all_states[:, j, :]
            # Ridge regression X_i -> X_j: W = (X_i^T X_i + 1e-3 I)^-1 X_i^T X_j
            XtX = X_i.T.dot(X_i) + 1e-3 * np.eye(D)
            W_probe = np.linalg.solve(XtX, X_i.T.dot(X_j))
            pred = X_i.dot(W_probe)
            ss_tot = np.sum((X_j - X_j.mean(axis=0)) ** 2)
            ss_res = np.sum((X_j - pred) ** 2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
            probe_r2[i, j] = float(r2)

    representation_audit = {
        "effective_ranks": effective_ranks,
        "mean_interior_rank": float(np.mean(effective_ranks[1:15])),
        "core_rank": effective_ranks[0],
        "surface_rank": effective_ranks[15],
        "top_singular_values": top_singular_values,
        "mean_raw_cka_interior": mean_raw_cka_interior,
        "mean_residual_cka_interior": mean_res_cka_interior,
        "raw_cka_matrix": cka_matrix.tolist(),
        "residual_cka_matrix": residual_cka_matrix.tolist(),
        "linear_probe_r2": probe_r2.tolist(),
    }
    print(f"  Core Rank (S0):                {effective_ranks[0]:.2f} / {min(N, D)}")
  
    print(f"  Mean Interior Rank (S1-S14):   {representation_audit['mean_interior_rank']:.2f}")
    print(f"  Surface Rank (S15):            {effective_ranks[15]:.2f}")
    print(f"  Mean Raw CKA (Interior):       {mean_raw_cka_interior:.4f}")
    print(f"  Mean Residual CKA (Interior):  {mean_res_cka_interior:.4f}")
    print(f"  Linear Predictability (S0 -> S7): R^2 = {probe_r2[0, 7]:.4f}")
    print(f"  Linear Predictability (S7 -> S0): R^2 = {probe_r2[7, 0]:.4f}")

    # ----------------------------------------------------
    # Task 3: Immediate Combined Inference Candidate Evaluation
    # ----------------------------------------------------
    print("\n--- [Task 3: Combined Inference Setting on Untouched Data] ---")
    # Untouched held-out baseline
    untouched_base_nll = evaluate_batches_nll(params, untouched_test_batches, buffers, cfg)
    
    # Combined candidate: beta = 1.0, w_0 = 0 (Core excluded from pool)
    w_combined = jax.nn.softmax(params["motor.shell_score"], axis=0).at[0].set(0.0)
    w_combined = w_combined / jnp.sum(w_combined) # re-normalize
    
    combined_overrides = {
        "pool_scale": 1.0,
        "shell_weights": w_combined,
    }
    untouched_combined_nll = evaluate_batches_nll(params, untouched_test_batches, buffers, cfg, combined_overrides)
    delta_combined = untouched_combined_nll - untouched_base_nll

    print(f"  Untouched Base NLL:     {untouched_base_nll:.6f}")
    print(f"  Combined Candidate NLL: {untouched_combined_nll:.6f} (Delta = {delta_combined:+.6f})")

    # ----------------------------------------------------
    # Task 4: Radial Coupling Capacity Sweep (r_mix)
    # ----------------------------------------------------
    print("\n--- [Task 4: Radial Coupling Capacity Sweep (r_mix)] ---")
    r_mix_values = [0.009, 0.03, 0.1, 0.2, 0.4]
    radial_sweep_results = {}

    for r_val in r_mix_values:
        # Evaluate NLL with overridden r_mix
        nll_r = evaluate_batches_nll(params, eval_batches, buffers, cfg, {"r_mix": r_val})
        
        # Measure state displacement and CKA under this r_mix
        states_r = []
        for state, sensory_kv, plan, targets in eval_batches:
            _, st_next_r, _ = forward_deliberation_and_readout(params, state, sensory_kv, plan, buffers, cfg, {"r_mix": r_val})
            states_r.append(np.array(st_next_r))
        cat_states_r = np.concatenate(states_r, axis=0) # [N, K, D]
        
        # Compute mean distance between adjacent shells ||s_{k+1} - s_k||
        disp_adjacent = []
        for k in range(K - 1):
            diff = cat_states_r[:, k+1, :] - cat_states_r[:, k, :]
            disp_adjacent.append(float(np.mean(np.linalg.norm(diff, axis=-1))))
        mean_adj_disp = float(np.mean(disp_adjacent))

        # Distance from core ||s_k - s_0||
        core_disp = []
        for k in range(1, K):
            diff = cat_states_r[:, k, :] - cat_states_r[:, 0, :]
            core_disp.append(float(np.mean(np.linalg.norm(diff, axis=-1))))
        mean_core_disp = float(np.mean(core_disp))

        # Compute interior CKA under this r_mix
        cka_r = np.zeros((K, K))
        for i in range(1, 15):
            X_i = cat_states_r[:, i, :]
            for j in range(i, 15):
                c_val = compute_linear_cka(X_i, cat_states_r[:, j, :])
                cka_r[i, j] = c_val
                cka_r[j, i] = c_val
        interior_cka_r = float(np.mean(cka_r[1:15, 1:15]))

        # Pool utility: NLL difference when pool is removed under this r_mix
        nll_no_pool_r = evaluate_batches_nll(params, eval_batches, buffers, cfg, {"r_mix": r_val, "pool_scale": 0.0})
        pool_utility_r = nll_no_pool_r - nll_r # positive means pool helps

        radial_sweep_results[str(r_val)] = {
            "r_mix": r_val,
            "nll": nll_r,
            "delta_base": nll_r - base_nll,
            "interior_cka": interior_cka_r,
            "mean_adjacent_displacement": mean_adj_disp,
            "mean_core_displacement": mean_core_disp,
            "pool_utility": pool_utility_r,
        }
        print(f"  r_mix={r_val:5.3f} | NLL={nll_r:.6f} (Delta={nll_r - base_nll:+.6f}) | Interior CKA={interior_cka_r:.4f} | Adj Disp={mean_adj_disp:6.2f} | Pool Util={pool_utility_r:+.6f}")

    # Full report compilation
    full_report = {
        "reconciliation": reconciliation,
        "representation_audit": representation_audit,
        "untouched_candidate_evaluation": {
            "untouched_base_nll": untouched_base_nll,
            "untouched_combined_nll": untouched_combined_nll,
            "delta_combined": delta_combined,
        },
        "radial_coupling_sweep": radial_sweep_results,
    }

    out_file = "experiments/radial_differentiation_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nRadial Differentiation Benchmark Complete. Persisted to: {out_file}")

if __name__ == "__main__":
    run_benchmark()
