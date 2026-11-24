"""
Audit of Shell Utility, Credit Assignment, and Optimizer Dynamics for CerebroSpheroid V5.
Strictly read-only with respect to foundation checkpoints and model/optimizer source code.

Methodological components:
1. State & Comprehensive Parameter Sensitivity (w_down, w_gate, w_val, intra_norms, w_radial).
2. Tri-Condition Ablation Utility (Zero, Mean, Pool-only) + Grouped Ablations (Core, Interior, Surface).
3. Scale-Relative Directional Perturbation Analysis (alpha in {1e-3, 1e-2, 1e-1}) along Random, Gradient, and Orthogonal directions.
4. Simulated Optimizer Update Effectiveness (E_k, cosine alignment, task vs decay decomposition, actual 1-step delta L).
5. Readout Pathway Attribution & Interaction (Core, Surface, Sensory, Pool, Plan input energy and pairwise interaction).
6. Shell Substitution Matrix (s_k <- s_j for adjacent/distant shells).
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
    deliberate_step,
    emit_step,
    generate_small_world_indices,
    rms_norm,
    swiglu,
)
from src.jax.spheroid_optimizer import (
    _apply_spheroid_grads,
    init_spheroid_opt_state,
    is_manifold_operator,
    zeropower_via_newtonschulz5_jax,
)

CHECKPOINT_PATH = "checkpoints/v5_small_latest.pt"

def load_checkpoint_and_model():
    print(f"Loading checkpoint: {CHECKPOINT_PATH} (read-only)...")
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
    # Convert all torch tensors to float32 jax arrays
    params = {k: jnp.array(v.float().numpy()) for k, v in ms.items() if hasattr(v, "numpy")}
    src_idx, dst_idx = generate_small_world_indices(cfg.num_shells, cfg.num_shortcut_tracts)
    buffers = (src_idx, dst_idx)
    return params, cfg, buffers

def get_representative_batch(cfg: ModelConfigV5, batch_size: int = 2, seq_len: int = 16):
    key = jax.random.PRNGKey(1337)
    k1, k2, k3 = jax.random.split(key, 3)
    # Synthetic representative sensory KV and states for reproducible local evaluation
    # Shape of sensory KV: [B, H, S, head_dim]
    head_dim = cfg.surface_dim // cfg.num_heads
    sensory_k = jax.random.normal(k1, (batch_size, cfg.num_heads, seq_len, head_dim))
    sensory_v = jax.random.normal(k2, (batch_size, cfg.num_heads, seq_len, head_dim))
    sensory_kv = (sensory_k, sensory_v)
    
    # State shape: [B, K, D]
    state = jax.random.normal(k3, (batch_size, cfg.num_shells, cfg.surface_dim))
    plan = jnp.zeros((batch_size, cfg.surface_dim), dtype=jnp.float32)
    # Target token IDs for cross-entropy loss evaluation
    target_tokens = jax.random.randint(k1, (batch_size,), 0, cfg.vocab_size)
    return state, sensory_kv, plan, target_tokens

def compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, custom_overrides=None):
    """
    Computes cross-entropy loss given state and optional custom overrides.
    custom_overrides can contain:
      - 'pool_mask': shape [16], binary or continuous mask for multi-shell pool
      - 'pathway_mask': shape [5], mask for [surface, core, sensory, pool, plan]
      - 'state_override': alternative state tensor
    """
    st = state if (custom_overrides is None or "state_override" not in custom_overrides) else custom_overrides["state_override"]
    B, K, D = st.shape
    surface = st[:, -1, :]
    core = st[:, 0, :]
    num_heads = cfg.num_heads
    head_dim = D // num_heads

    # Step idx 0
    p_idx = 0
    phase = params["motor.phase_embed.weight"][p_idx]
    p_norm = jnp.maximum(jnp.linalg.norm(phase, axis=-1, keepdims=True), 1e-6)
    p_scale_val = jnp.clip(params.get("motor.phase_scale", 2.0), 0.05, 5.0)
    p_scaled = phase * (p_scale_val / p_norm)
    motor_surface = surface + p_scaled

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
    shell_weights = jax.nn.softmax(params["motor.shell_score"], axis=0) # [16]
    if custom_overrides is not None and "pool_mask" in custom_overrides:
        shell_weights = shell_weights * custom_overrides["pool_mask"]
    
    shell_proj = jnp.matmul(st, params["motor.shell_proj.weight"].T)
    shell_pool = jnp.sum(shell_weights[None, :, None] * shell_proj, axis=1) * params["motor.shell_scale"]

    # Fused in
    ig = jnp.clip(params.get("motor.input_gate", jnp.ones(5, dtype=jnp.float32)), -5.0, 5.0)
    path_mask = jnp.ones(5, dtype=jnp.float32) if (custom_overrides is None or "pathway_mask" not in custom_overrides) else custom_overrides["pathway_mask"]
    
    p0 = path_mask[0] * ig[0] * motor_surface
    p1 = path_mask[1] * ig[1] * core
    p2 = path_mask[2] * ig[2] * resonance
    p3 = path_mask[3] * ig[3] * shell_pool
    p4 = path_mask[4] * ig[4] * plan

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
    h_scaled = (h * active_affine_scale(affine_scale)) / temp
    w_out = params.get("motor.head.weight", params["sensory.embedding.weight"])
    logits = jnp.matmul(h_scaled, w_out.T)

    # Cross-entropy
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, target_tokens[:, None], axis=-1)
    return jnp.mean(nll), (logits, fused_in, (p0, p1, p2, p3, p4))

def active_affine_scale(aff):
    return aff

def run_audit():
    params, cfg, buffers = load_checkpoint_and_model()
    state, sensory_kv, plan, target_tokens = get_representative_batch(cfg)

    # Baseline loss
    base_loss, (base_logits, base_fused, base_pathways) = compute_loss(
        params, state, sensory_kv, plan, buffers, cfg, target_tokens
    )
    base_loss_val = float(base_loss)
    print(f"\n==================================================")
    print(f"BASE VALIDATION NLL LOSS: {base_loss_val:.6f}")
    print(f"==================================================\n")

    # ----------------------------------------------------
    # METRIC 1: State Sensitivity & Comprehensive Parameter Sensitivity
    # ----------------------------------------------------
    print("Computing Metric 1: State & Comprehensive Parameter Sensitivity...")
    
    def loss_wrt_state(s):
        l, _ = compute_loss(params, s, sensory_kv, plan, buffers, cfg, target_tokens)
        return l
    
    grad_state = jax.grad(loss_wrt_state)(state) # [B, K, D]
    state_norms = [float(jnp.linalg.norm(grad_state[:, k, :])) for k in range(cfg.num_shells)]
    median_state_norm = float(np.median(state_norms))

    def loss_wrt_params(p):
        l, _ = compute_loss(p, state, sensory_kv, plan, buffers, cfg, target_tokens)
        return l

    grad_params = jax.grad(loss_wrt_params)(params)

    param_groups = {
        "w_down": "core.intra_swiglu.w_down",
        "w_gate": "core.intra_swiglu.w_gate",
        "w_val": "core.intra_swiglu.w_val",
        "intra_norms": "core.intra_norms.weight",
    }
    
    w_radial_in_grad = grad_params.get("core.w_radial_in", None) # [15, 256, 256]
    w_radial_out_grad = grad_params.get("core.w_radial_out", None)

    shell_param_grads = {}
    for name, key in param_groups.items():
        g = grad_params[key] # [16, ...]
        shell_param_grads[name] = [float(jnp.linalg.norm(g[k])) for k in range(cfg.num_shells)]

    # ----------------------------------------------------
    # METRIC 2: Tri-Condition Ablation Utility
    # ----------------------------------------------------
    print("Computing Metric 2: Tri-Condition Ablations...")
    # Condition A: Zero ablation s_k = 0
    delta_zero = []
    for k in range(cfg.num_shells):
        s_mod = state.at[:, k, :].set(0.0)
        l, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_mod})
        delta_zero.append(float(l) - base_loss_val)

    # Condition B: Mean ablation s_k = mean(s_k across batch/feature)
    mean_state_k = jnp.mean(state, axis=(0, 2), keepdims=True) # [1, 16, 1]
    delta_mean = []
    for k in range(cfg.num_shells):
        s_mod = state.at[:, k, :].set(float(mean_state_k[0, k, 0]))
        l, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_mod})
        delta_mean.append(float(l) - base_loss_val)

    # Condition C: Pool contribution removal (remove shell k from softmax pool)
    delta_pool = []
    for k in range(cfg.num_shells):
        pool_mask = jnp.ones(cfg.num_shells).at[k].set(0.0)
        l, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"pool_mask": pool_mask})
        delta_pool.append(float(l) - base_loss_val)

    # Grouped Ablations: Core, Interior (Shells 1-14), Surface (Shell 15)
    s_core_zero = state.at[:, 0, :].set(0.0)
    l_core_zero, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_core_zero})
    delta_grp_core = float(l_core_zero) - base_loss_val

    s_int_zero = state.at[:, 1:15, :].set(0.0)
    l_int_zero, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_int_zero})
    delta_grp_interior = float(l_int_zero) - base_loss_val

    s_surf_zero = state.at[:, 15, :].set(0.0)
    l_surf_zero, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_surf_zero})
    delta_grp_surface = float(l_surf_zero) - base_loss_val

    # Grouped interior subgroups: lower (1-4), mid (5-10), upper (11-14)
    s_sub1 = state.at[:, 1:5, :].set(0.0)
    l_sub1, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_sub1})
    delta_int_lower = float(l_sub1) - base_loss_val

    s_sub2 = state.at[:, 5:11, :].set(0.0)
    l_sub2, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_sub2})
    delta_int_mid = float(l_sub2) - base_loss_val

    s_sub3 = state.at[:, 11:15, :].set(0.0)
    l_sub3, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_sub3})
    delta_int_upper = float(l_sub3) - base_loss_val

    # ----------------------------------------------------
    # METRIC 3: Scale-Relative Directional Perturbation Analysis
    # ----------------------------------------------------
    print("Computing Metric 3: Directional Perturbations (Random, Gradient, Orthogonal)...")
    alphas = [1e-3, 1e-2, 1e-1]
    perturb_results = {k: {} for k in range(cfg.num_shells)}
    
    key_p = jax.random.PRNGKey(2026)
    for k in range(cfg.num_shells):
        s_k = state[:, k, :]
        rms_k = float(jnp.sqrt(jnp.mean(jnp.square(s_k))))
        g_k = grad_state[:, k, :]
        g_norm = jnp.linalg.norm(g_k)
        v_grad = g_k / (g_norm + 1e-7)

        key_p, k_rand = jax.random.split(key_p)
        v_rand_raw = jax.random.normal(k_rand, s_k.shape)
        v_rand = v_rand_raw / (jnp.linalg.norm(v_rand_raw) + 1e-7)

        # Orthogonal projection: v_ortho = v_rand - (v_rand . v_grad) * v_grad
        proj = jnp.sum(v_rand * v_grad)
        v_ortho_raw = v_rand - proj * v_grad
        v_ortho = v_ortho_raw / (jnp.linalg.norm(v_ortho_raw) + 1e-7)

        for alpha in alphas:
            eps = alpha * rms_k
            
            # Directional evaluations: L(s_k + eps*v)
            # 1. Gradient direction
            s_plus_g = state.at[:, k, :].add(eps * v_grad)
            s_minus_g = state.at[:, k, :].add(-eps * v_grad)
            l_plus_g, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_plus_g})
            l_minus_g, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_minus_g})
            deriv_g = (float(l_plus_g) - base_loss_val) / eps
            curv_g = (float(l_plus_g) - 2.0 * base_loss_val + float(l_minus_g)) / (eps ** 2)

            # 2. Random direction
            s_plus_r = state.at[:, k, :].add(eps * v_rand)
            s_minus_r = state.at[:, k, :].add(-eps * v_rand)
            l_plus_r, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_plus_r})
            l_minus_r, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_minus_r})
            deriv_r = (float(l_plus_r) - base_loss_val) / eps
            curv_r = (float(l_plus_r) - 2.0 * base_loss_val + float(l_minus_r)) / (eps ** 2)

            # 3. Orthogonal direction
            s_plus_o = state.at[:, k, :].add(eps * v_ortho)
            s_minus_o = state.at[:, k, :].add(-eps * v_ortho)
            l_plus_o, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_plus_o})
            l_minus_o, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_minus_o})
            deriv_o = (float(l_plus_o) - base_loss_val) / eps
            curv_o = (float(l_plus_o) - 2.0 * base_loss_val + float(l_minus_o)) / (eps ** 2)

            perturb_results[k][alpha] = {
                "deriv_grad": deriv_g, "curv_grad": curv_g,
                "deriv_rand": deriv_r, "curv_rand": curv_r,
                "deriv_ortho": deriv_o, "curv_ortho": curv_o,
            }

    # ----------------------------------------------------
    # METRIC 4: Optimizer Update Effectiveness (Fresh-State Simulation)
    # ----------------------------------------------------
    print("Computing Metric 4: Simulated Optimizer Update Effectiveness...")
    opt_state = init_spheroid_opt_state(params)
    base_lr = 1e-4
    weight_decay = 0.01

    # Simulate 1-step update
    updated_params, new_opt_s = _apply_spheroid_grads(
        params, opt_state, grad_params, base_lr=base_lr, weight_decay=weight_decay
    )
    
    # Also evaluate pure decay step (zero gradients)
    zero_grads = {k: jnp.zeros_like(v) for k, v in grad_params.items()}
    decay_params, _ = _apply_spheroid_grads(
        params, opt_state, zero_grads, base_lr=base_lr, weight_decay=weight_decay
    )

    actual_loss_after_step, _ = compute_loss(
        updated_params, state, sensory_kv, plan, buffers, cfg, target_tokens
    )
    delta_L_actual = float(actual_loss_after_step) - base_loss_val

    # Per-shell analysis on intra_swiglu.w_down
    w_down_param = params["core.intra_swiglu.w_down"] # [16, 1280, 640]
    w_down_grad = grad_params["core.intra_swiglu.w_down"]
    w_down_upd = updated_params["core.intra_swiglu.w_down"]
    w_down_decay = decay_params["core.intra_swiglu.w_down"]

    optimizer_metrics = []
    for k in range(cfg.num_shells):
        theta_k = w_down_param[k]
        g_k = w_down_grad[k]
        theta_next_k = w_down_upd[k]
        theta_decay_k = w_down_decay[k]

        delta_theta_k = theta_next_k - theta_k
        delta_decay_k = theta_decay_k - theta_k
        delta_task_k = delta_theta_k - delta_decay_k

        norm_g = float(jnp.linalg.norm(g_k))
        norm_delta = float(jnp.linalg.norm(delta_theta_k))
        norm_theta = float(jnp.linalg.norm(theta_k))
        norm_task = float(jnp.linalg.norm(delta_task_k))
        norm_decay = float(jnp.linalg.norm(delta_decay_k))

        # E_k = -g^T Delta theta / (||Delta theta|| + eps)
        inner_prod = float(jnp.sum(-g_k * delta_theta_k))
        E_k = inner_prod / (norm_delta + 1e-7)
        cos_align = inner_prod / ((norm_g * norm_delta) + 1e-7)
        rel_update = norm_delta / (norm_theta + 1e-7)

        optimizer_metrics.append({
            "shell": k,
            "norm_g": norm_g,
            "norm_delta": norm_delta,
            "rel_update": rel_update,
            "E_k": E_k,
            "cos_align": cos_align,
            "norm_task": norm_task,
            "norm_decay": norm_decay,
            "task_to_decay_ratio": norm_task / (norm_decay + 1e-7),
        })

    # ----------------------------------------------------
    # METRIC 5: Readout Pathway Attribution & Interaction
    # ----------------------------------------------------
    print("Computing Metric 5: Readout Pathway Attribution & Interaction...")
    p0, p1, p2, p3, p4 = base_pathways
    path_names = ["Surface (p0)", "Core (p1)", "Sensory (p2)", "Shell Pool (p3)", "Plan (p4)"]
    path_energies = [
        float(jnp.mean(jnp.sum(jnp.square(p), axis=-1))) for p in [p0, p1, p2, p3, p4]
    ]

    # Individual pathway ablations Delta L_i
    delta_pathways = []
    for i in range(5):
        mask = jnp.ones(5).at[i].set(0.0)
        l, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"pathway_mask": mask})
        delta_pathways.append(float(l) - base_loss_val)

    # Pairwise interaction I_ij = Delta L_ij - Delta L_i - Delta L_j
    interaction_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(i + 1, 5):
            mask = jnp.ones(5).at[i].set(0.0).at[j].set(0.0)
            l_ij, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"pathway_mask": mask})
            d_ij = float(l_ij) - base_loss_val
            I_ij = d_ij - delta_pathways[i] - delta_pathways[j]
            interaction_matrix[i, j] = I_ij
            interaction_matrix[j, i] = I_ij

    # ----------------------------------------------------
    # METRIC 6: Shell Substitution Matrix
    # ----------------------------------------------------
    print("Computing Metric 6: Shell Substitution Matrix...")
    # s_k <- s_j for subset of representative shells to measure redundancy
    test_shells = [0, 1, 3, 7, 10, 13, 14, 15]
    substitution_matrix = np.zeros((len(test_shells), len(test_shells)))
    for idx_k, k in enumerate(test_shells):
        for idx_j, j in enumerate(test_shells):
            if k == j:
                substitution_matrix[idx_k, idx_j] = 0.0
                continue
            s_sub = state.at[:, k, :].set(state[:, j, :])
            l_sub, _ = compute_loss(params, state, sensory_kv, plan, buffers, cfg, target_tokens, {"state_override": s_sub})
            substitution_matrix[idx_k, idx_j] = float(l_sub) - base_loss_val

    # Assemble report structure
    report = {
        "base_loss": base_loss_val,
        "delta_L_actual_1step": delta_L_actual,
        "grouped_ablations": {
            "core": delta_grp_core,
            "interior_all_1_to_14": delta_grp_interior,
            "surface": delta_grp_surface,
            "interior_lower_1_to_4": delta_int_lower,
            "interior_mid_5_to_10": delta_int_mid,
            "interior_upper_11_to_14": delta_int_upper,
        },
        "pathway_attribution": {
            "names": path_names,
            "energies": path_energies,
            "individual_ablations": delta_pathways,
            "interaction_matrix": interaction_matrix.tolist(),
        },
        "shell_metrics": [],
        "substitution_test": {
            "shells": test_shells,
            "matrix": substitution_matrix.tolist(),
        }
    }

    for k in range(cfg.num_shells):
        report["shell_metrics"].append({
            "shell": k,
            "state_grad_norm": state_norms[k],
            "state_grad_rel": state_norms[k] / (median_state_norm + 1e-7),
            "w_down_grad_norm": shell_param_grads["w_down"][k],
            "w_gate_grad_norm": shell_param_grads["w_gate"][k],
            "w_val_grad_norm": shell_param_grads["w_val"][k],
            "intra_norms_grad_norm": shell_param_grads["intra_norms"][k],
            "ablation_zero": delta_zero[k],
            "ablation_mean": delta_mean[k],
            "ablation_pool": delta_pool[k],
            "perturbations": perturb_results[k],
            "optimizer": optimizer_metrics[k],
        })

    out_file = "experiments/shell_utility_audit_results.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nAudit complete. Full JSON persisted to: {out_file}")

if __name__ == "__main__":
    run_audit()
