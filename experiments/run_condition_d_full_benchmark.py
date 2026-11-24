"""
Condition D Full Benchmark: Controlled Radial Adaptation across 4 Arms x 3 Seeds.
Strictly read-only with respect to foundation checkpoint. Cloned parameters only.

Experimental Arms:
- Arm A (Control): Frozen Baseline (r_mix = 0.009, no training)
- Arm B (Coupling Baseline Control): Adapted at original coupling (r_mix = 0.009)
- Arm C (Low Coupling Adaptation): Adapted at r_mix = 0.10
- Arm D (Moderate Coupling Adaptation): Adapted at r_mix = 0.20

Seeds: [42, 137, 2026]
Steps: 250 steps per seed
Learning Rate: 3e-5 (verified stable, non-exploding, non-vanishing in pilot)
Evaluation:
- Held-out NLL at Step 0, 100, 250
- Interior CKA (Shells 1-14)
- Interior Ablation Utility (delta NLL when S1-14 = 0)
- Adjacent Shell Displacement
- Seed-level tables and aggregate statistics
"""

import os
import sys
import math
import json
import hashlib
from typing import Dict, Any, Tuple, List
from functools import partial

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np
import torch

from src.jax.v5_model import (
    ModelConfigV5,
    generate_small_world_indices,
    deliberate_step,
    rms_norm,
    swiglu,
)
from src.jax.spheroid_optimizer import (
    zeropower_via_newtonschulz5_jax,
    clamp_spectral_norm_jax,
)

CHECKPOINT_PATH = "checkpoints/v5_small_latest.pt"

TARGET_RADIAL_KEYS = [
    "core.w_radial_in",
    "core.w_radial_out",
    "core.radial_in_proj.weight",
    "core.radial_out_proj.weight",
    "core.radial_up.weight",
    "core.radial_up_out.weight",
    "core.radial_norm.weight",
]

def load_foundation():
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
    rad_params = {}
    frozen_params = {}
    for k, v in ms.items():
        if hasattr(v, "numpy"):
            arr = jnp.array(v.float().numpy())
            if k in TARGET_RADIAL_KEYS:
                rad_params[k] = arr
            else:
                frozen_params[k] = arr

    src_idx, dst_idx = generate_small_world_indices(cfg.num_shells, cfg.num_shortcut_tracts)
    buffers = (src_idx, dst_idx)
    return rad_params, frozen_params, cfg, buffers

def get_frozen_checksum(frozen_params: Dict[str, jnp.ndarray]) -> str:
    hasher = hashlib.sha256()
    for k in sorted(frozen_params.keys()):
        hasher.update(np.array(frozen_params[k]).tobytes())
    return hasher.hexdigest()

def generate_stream(cfg: ModelConfigV5, num_batches: int = 250, batch_size: int = 1, seq_len: int = 16, seed: int = 42):
    key = jax.random.PRNGKey(seed)
    head_dim = cfg.surface_dim // cfg.num_heads
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

def build_engine(cfg: ModelConfigV5, buffers: Tuple[jnp.ndarray, jnp.ndarray]):
    def forward_state_and_loss(rad_p, frozen_p, batch, r_mix_fixed, state_override=None):
        state, sensory_kv, plan, targets = batch
        st_in = state if state_override is None else state_override

        p_full = dict(frozen_p)
        p_full.update(rad_p)
        p_full["core.radial_mix_scale"] = jnp.array(r_mix_fixed, dtype=jnp.float32)

        st_next = deliberate_step(p_full, st_in, buffers, config=cfg)
        B, K, D = st_next.shape
        surface = st_next[:, -1, :]
        core = st_next[:, 0, :]
        num_heads = cfg.num_heads
        head_dim = D // num_heads

        phase = p_full["motor.phase_embed.weight"][0]
        p_norm = jnp.maximum(jnp.linalg.norm(phase, axis=-1, keepdims=True), 1e-6)
        p_scale_val = jnp.clip(p_full.get("motor.phase_scale", 2.0), 0.05, 5.0)
        p_scaled = phase * (p_scale_val / p_norm)
        motor_surface = surface + p_scaled

        keys, values = sensory_kv
        q = jnp.matmul(motor_surface, p_full["motor.q_proj.weight"].T).reshape(B, 1, num_heads, head_dim)
        q = jnp.swapaxes(q, 1, 2)
        scores = jnp.matmul(q, jnp.swapaxes(keys, -1, -2)) / math.sqrt(head_dim)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        res_heads = jnp.matmul(attn_weights, values)
        res_heads = jnp.swapaxes(res_heads, 1, 2).reshape(B, D)
        resonance = rms_norm(jnp.matmul(res_heads, p_full["motor.out_proj.weight"].T), p_full["motor.resonance_norm.weight"])

        shell_weights = jax.nn.softmax(p_full["motor.shell_score"], axis=0)
        pool_proj = p_full["motor.shell_proj.weight"]
        shell_proj = jnp.matmul(st_next, pool_proj.T)
        pool_beta = p_full["motor.shell_scale"]
        shell_pool = jnp.sum(shell_weights[None, :, None] * shell_proj, axis=1) * pool_beta

        ig = jnp.clip(p_full.get("motor.input_gate", jnp.ones(5, dtype=jnp.float32)), -5.0, 5.0)
        p0 = ig[0] * motor_surface
        p1 = ig[1] * core
        p2 = ig[2] * resonance
        p3 = ig[3] * shell_pool
        p4 = ig[4] * plan

        fused_in = jnp.concatenate([p0, p1, p2, p3, p4], axis=-1)
        fused = rms_norm(fused_in, p_full["motor.fused_norm.weight"])
        h = rms_norm(
            swiglu(
                fused,
                p_full["motor.swiglu.fc_gate.weight"],
                p_full["motor.swiglu.fc_val.weight"],
                p_full["motor.swiglu.fc_down.weight"],
                use_swiglu=True,
            ),
            p_full["motor.norm_mid.weight"],
        )

        temp = jnp.clip(p_full["motor.temperature_scale"], 0.1, 5.0)
        affine_scale = p_full["motor.affine_scale"][None, :]
        h_scaled = (h * affine_scale) / temp
        w_out = p_full.get("motor.head.weight", p_full["sensory.embedding.weight"])
        logits = jnp.matmul(h_scaled, w_out.T)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(log_probs, targets[:, None], axis=-1)
        return jnp.mean(nll), st_next

    @jax.jit
    def step_fn(rad_p, frozen_p, opt_m, opt_v_power, batch, r_mix, lr, wd):
        def loss_wrapper(p):
            l, _ = forward_state_and_loss(p, frozen_p, batch, r_mix)
            return l

        loss_val, grads = jax.value_and_grad(loss_wrapper)(rad_p)
        
        g_sq = sum(jnp.sum(jnp.square(g)) for g in grads.values())
        g_norm = jnp.sqrt(g_sq + 1e-7)
        clip_scale = jnp.minimum(1.0, 1.0 / (g_norm + 1e-6))
        
        new_rad_p = {}
        new_opt_m = {}
        new_opt_v_power = dict(opt_v_power)
        
        for k, p in rad_p.items():
            g = grads[k] * clip_scale
            m_prev = opt_m[k]
            m_new = 0.95 * m_prev + 0.05 * g
            new_opt_m[k] = m_new

            p_decayed = p * (1.0 - lr * wd)

            if p.ndim >= 2 and "norm" not in k.lower():
                O = zeropower_via_newtonschulz5_jax(m_new, steps=3)
                rows = p.shape[-2]
                cols = p.shape[-1]
                aspect_scale = math.sqrt(max(1.0, float(rows) / float(cols)))
                p_upd = p_decayed - (lr * aspect_scale) * O
                
                if "w_radial" in k:
                    p_upd, v_next = clamp_spectral_norm_jax(p_upd, opt_v_power[k], max_spec_norm=1.0)
                    new_opt_v_power[k] = v_next
            else:
                p_upd = p_decayed - lr * m_new

            new_rad_p[k] = p_upd

        return new_rad_p, new_opt_m, new_opt_v_power, loss_val, g_norm

    return forward_state_and_loss, step_fn

def evaluate_metrics(rad_p, frozen_p, val_batches, r_mix, forward_fn):
    losses = []
    states = []
    int_abl_losses = []

    for b in val_batches:
        l, st = forward_fn(rad_p, frozen_p, b, r_mix)
        losses.append(float(l))
        states.append(np.array(st))

        # Interior ablation (S1-S14 = 0 entering deliberate)
        st_int_zero = b[0].at[:, 1:15, :].set(0.0)
        l_abl, _ = forward_fn(rad_p, frozen_p, b, r_mix, state_override=st_int_zero)
        int_abl_losses.append(float(l_abl))

    mean_nll = float(np.mean(losses))
    mean_abl_nll = float(np.mean(int_abl_losses))
    # Interior utility = L(ablated) - L(full). Positive means interior is beneficial.
    interior_utility = mean_abl_nll - mean_nll

    cat_states = np.concatenate(states, axis=0) # [N, K, D]
    N, K, D = cat_states.shape

    # Interior CKA (Shells 1-14)
    cka_mat = np.zeros((14, 14))
    for i in range(14):
        for j in range(i, 14):
            c = compute_linear_cka(cat_states[:, i+1, :], cat_states[:, j+1, :])
            cka_mat[i, j] = c
            cka_mat[j, i] = c
    mean_interior_cka = float(np.mean(cka_mat))

    # Adjacent displacement
    adj_disp = [float(np.mean(np.linalg.norm(cat_states[:, k+1, :] - cat_states[:, k, :], axis=-1))) for k in range(K-1)]
    mean_adj_disp = float(np.mean(adj_disp))

    return {
        "nll": mean_nll,
        "interior_utility": interior_utility,
        "interior_cka": mean_interior_cka,
        "adj_displacement": mean_adj_disp,
    }

def run_benchmark():
    rad_base, frozen_base, cfg, buffers = load_foundation()
    forward_fn, step_fn = build_engine(cfg, buffers)
    init_checksum = get_frozen_checksum(frozen_base)
    print(f"Verified initial frozen parameter SHA-256 checksum: {init_checksum[:16]}...")

    # Fixed strictly held-out validation set (16 batches = 16 sequences)
    val_batches = generate_stream(cfg, num_batches=16, batch_size=1, seed=9999)

    # Arm configurations
    arms = [
        {"name": "Arm_A_Frozen_Baseline", "r_mix": 0.009, "train": False},
        {"name": "Arm_B_Coupling_Control_0.009", "r_mix": 0.009, "train": True},
        {"name": "Arm_C_Low_Coupling_0.10", "r_mix": 0.10, "train": True},
        {"name": "Arm_D_Moderate_Coupling_0.20", "r_mix": 0.20, "train": True},
    ]

    seeds = [42, 137, 2026]
    lr = 3e-5
    steps = 250
    eval_intervals = [0, 100, 250]

    all_results = {}

    for arm in arms:
        arm_name = arm["name"]
        r_mix = arm["r_mix"]
        should_train = arm["train"]
        print(f"\n==================================================")
        print(f"RUNNING: {arm_name} (r_mix={r_mix}, Trainable={should_train})")
        print(f"==================================================")

        all_results[arm_name] = {"r_mix": r_mix, "trainable": should_train, "seeds": {}}

        if not should_train:
            # Baseline is deterministic and evaluated once
            metrics_0 = evaluate_metrics(rad_base, frozen_base, val_batches, r_mix, forward_fn)
            print(f"  Step 0 Baseline: NLL={metrics_0['nll']:.4f} | CKA={metrics_0['interior_cka']:.4f} | Util={metrics_0['interior_utility']:+.4f} | Disp={metrics_0['adj_displacement']:.2f}")
            all_results[arm_name]["baseline_metrics"] = metrics_0
            continue

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            train_stream = generate_stream(cfg, num_batches=steps, batch_size=1, seed=seed)
            
            # Reset parameters exactly from foundation base
            cur_rad_p = {k: jnp.array(v) for k, v in rad_base.items()}
            opt_m = {k: jnp.zeros_like(v) for k, v in rad_base.items()}
            opt_v_power = {}
            for k in ["core.w_radial_in", "core.w_radial_out"]:
                shape = rad_base[k].shape
                v_init = jax.random.normal(jax.random.PRNGKey(seed), (shape[0], shape[-1], 1))
                v_init = v_init / jnp.linalg.norm(v_init, axis=1, keepdims=True)
                opt_v_power[k] = v_init

            seed_trajectory = {}

            # Initial evaluation at Step 0
            m_0 = evaluate_metrics(cur_rad_p, frozen_base, val_batches, r_mix, forward_fn)
            seed_trajectory["step_0"] = m_0
            print(f"  Step   0: NLL={m_0['nll']:.4f} | CKA={m_0['interior_cka']:.4f} | Util={m_0['interior_utility']:+.4f}")

            for step in range(1, steps + 1):
                batch = train_stream[step - 1]
                cur_rad_p, opt_m, opt_v_power, _, _ = step_fn(
                    cur_rad_p, frozen_base, opt_m, opt_v_power, batch, r_mix, lr, 0.01
                )

                if step in [100, 250]:
                    m_step = evaluate_metrics(cur_rad_p, frozen_base, val_batches, r_mix, forward_fn)
                    seed_trajectory[f"step_{step}"] = m_step
                    print(f"  Step {step:3d}: NLL={m_step['nll']:.4f} | CKA={m_step['interior_cka']:.4f} | Util={m_step['interior_utility']:+.4f} | Disp={m_step['adj_displacement']:.2f}")

            # Verify checksum at end of seed
            post_checksum = get_frozen_checksum(frozen_base)
            assert post_checksum == init_checksum, f"Frozen parameter leakage detected in seed {seed}!"

            all_results[arm_name]["seeds"][str(seed)] = seed_trajectory

    out_file = "experiments/condition_d_full_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCondition D Benchmark Complete! Full data persisted to: {out_file}")

if __name__ == "__main__":
    run_benchmark()
