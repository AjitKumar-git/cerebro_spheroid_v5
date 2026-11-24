"""
Phase 2 Benchmark: Radial Subsystem + Pool Readout Co-Adaptation vs Readout-Only Control.
Strictly read-only with respect to foundation checkpoint. Cloned parameters only.

Experimental Arms:
- Arm A (Frozen Baseline): r_mix = 0.009, no training.
- Arm B (Readout-Only Control): r_mix = 0.009, Trainable: pool readout only.
- Arm C (Low Coupling Co-Adaptation): r_mix = 0.10, Trainable: radial + pool readout.
- Arm D (Moderate Coupling Co-Adaptation): r_mix = 0.20, Trainable: radial + pool readout.

Standardized Evaluation:
- N = 256 held-out validation samples (identical to the large-sample baseline, NLL = 19.2003, CKA = 0.7353).
- Untouched test set (N = 128) evaluated strictly once at the end.
- Metrics at Steps 0, 50, 100, 250, 500:
  * Held-out NLL
  * Radial and Pool update norms & gradient norms
  * Shell score softmax distribution
  * Interior CKA (Shells 1-14)
  * Adjacent shell displacement
  * Interior, Pool, Core, and Surface ablation utilities.
"""

import os
import sys
import math
import json
import hashlib
from functools import partial
from typing import Dict, Any, Tuple, List

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

RADIAL_KEYS = [
    "core.w_radial_in",
    "core.w_radial_out",
    "core.radial_in_proj.weight",
    "core.radial_out_proj.weight",
    "core.radial_up.weight",
    "core.radial_up_out.weight",
    "core.radial_norm.weight",
]

POOL_KEYS = [
    "motor.shell_score",
    "motor.shell_scale",
    "motor.shell_proj.weight",
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
    pool_params = {}
    frozen_params = {}

    for k, v in ms.items():
        if hasattr(v, "numpy"):
            arr = jnp.array(v.float().numpy())
            if k in RADIAL_KEYS:
                rad_params[k] = arr
            elif k in POOL_KEYS:
                pool_params[k] = arr
            else:
                frozen_params[k] = arr

    src_idx, dst_idx = generate_small_world_indices(cfg.num_shells, cfg.num_shortcut_tracts)
    buffers = (src_idx, dst_idx)
    return rad_params, pool_params, frozen_params, cfg, buffers

def get_frozen_checksum(frozen_params: Dict[str, jnp.ndarray]) -> str:
    hasher = hashlib.sha256()
    for k in sorted(frozen_params.keys()):
        hasher.update(np.array(frozen_params[k]).tobytes())
    return hasher.hexdigest()

def generate_stream(cfg: ModelConfigV5, total_sequences: int = 500, batch_size: int = 1, seq_len: int = 16, seed: int = 42):
    key = jax.random.PRNGKey(seed)
    head_dim = cfg.surface_dim // cfg.num_heads
    batches = []
    for _ in range(total_sequences):
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

def build_phase2_engine(cfg: ModelConfigV5, buffers: Tuple[jnp.ndarray, jnp.ndarray]):
    def forward_state_and_loss(rad_p, pool_p, frozen_p, batch, r_mix_fixed, overrides=None):
        state, sensory_kv, plan, targets = batch
        st_in = state if (overrides is None or "state_in" not in overrides) else overrides["state_in"]

        p_full = dict(frozen_p)
        p_full.update(rad_p)
        p_full.update(pool_p)
        p_full["core.radial_mix_scale"] = jnp.array(r_mix_fixed, dtype=jnp.float32)

        st_next = deliberate_step(p_full, st_in, buffers, config=cfg)
        st_readout = st_next if (overrides is None or "state_readout" not in overrides) else overrides["state_readout"]

        B, K, D = st_readout.shape
        surface = st_readout[:, -1, :]
        core = st_readout[:, 0, :]
        num_heads = cfg.num_heads
        head_dim = D // num_heads

        phase = p_full["motor.phase_embed.weight"][0]
        p_norm = jnp.maximum(jnp.linalg.norm(phase, axis=-1, keepdims=True), 1e-6)
        p_scale_val = jnp.clip(p_full.get("motor.phase_scale", 2.0), 0.05, 5.0)
        p_scaled = phase * (p_scale_val / p_norm)
        
        surf_mult = 1.0 if (overrides is None or "surf_mult" not in overrides) else overrides["surf_mult"]
        motor_surface = surf_mult * surface + p_scaled

        keys, values = sensory_kv
        q = jnp.matmul(motor_surface, p_full["motor.q_proj.weight"].T).reshape(B, 1, num_heads, head_dim)
        q = jnp.swapaxes(q, 1, 2)
        scores = jnp.matmul(q, jnp.swapaxes(keys, -1, -2)) / math.sqrt(head_dim)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        res_heads = jnp.matmul(attn_weights, values)
        res_heads = jnp.swapaxes(res_heads, 1, 2).reshape(B, D)
        resonance = rms_norm(jnp.matmul(res_heads, p_full["motor.out_proj.weight"].T), p_full["motor.resonance_norm.weight"])

        shell_weights = jax.nn.softmax(pool_p["motor.shell_score"], axis=0)
        pool_proj = pool_p["motor.shell_proj.weight"]
        shell_proj = jnp.matmul(st_readout, pool_proj.T)
        pool_beta = pool_p["motor.shell_scale"]
        
        pool_mult = 1.0 if (overrides is None or "pool_mult" not in overrides) else overrides["pool_mult"]
        shell_pool = pool_mult * jnp.sum(shell_weights[None, :, None] * shell_proj, axis=1) * pool_beta

        core_mult = 1.0 if (overrides is None or "core_mult" not in overrides) else overrides["core_mult"]
        ig = jnp.clip(p_full.get("motor.input_gate", jnp.ones(5, dtype=jnp.float32)), -5.0, 5.0)
        p0 = ig[0] * motor_surface
        p1 = ig[1] * (core_mult * core)
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

    @partial(jax.jit, static_argnums=(9, 10))
    def step_fn(rad_p, pool_p, frozen_p, opt_m, opt_v_power, batch, r_mix, lr, wd, train_radial, train_pool):
        def loss_wrapper(r_params, p_params):
            l, _ = forward_state_and_loss(r_params, p_params, frozen_p, batch, r_mix)
            return l

        loss_val, (rad_grads, pool_grads) = jax.value_and_grad(loss_wrapper, argnums=(0, 1))(rad_p, pool_p)

        # Compute gradient norms
        rad_g_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in rad_grads.values()) + 1e-7)
        pool_g_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in pool_grads.values()) + 1e-7)

        # Combined clipping
        tot_g_sq = (rad_g_norm ** 2 if train_radial else 0.0) + (pool_g_norm ** 2 if train_pool else 0.0)
        tot_g_norm = jnp.sqrt(tot_g_sq + 1e-7)
        clip_scale = jnp.minimum(1.0, 1.0 / (tot_g_norm + 1e-6))

        new_rad_p = dict(rad_p)
        new_pool_p = dict(pool_p)
        new_opt_m = dict(opt_m)
        new_opt_v_power = dict(opt_v_power)
        
        rad_upd_norm = jnp.array(0.0)
        pool_upd_norm = jnp.array(0.0)

        # Update Radial if train_radial
        if train_radial:
            rad_diff_sq = 0.0
            for k, p in rad_p.items():
                g = rad_grads[k] * clip_scale
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

                rad_diff_sq = rad_diff_sq + jnp.sum(jnp.square(p_upd - p))
                new_rad_p[k] = p_upd
            rad_upd_norm = jnp.sqrt(rad_diff_sq)

        # Update Pool if train_pool
        if train_pool:
            pool_diff_sq = 0.0
            for k, p in pool_p.items():
                g = pool_grads[k] * clip_scale
                m_prev = opt_m[k]
                m_new = 0.95 * m_prev + 0.05 * g
                new_opt_m[k] = m_new

                p_decayed = p * (1.0 - lr * wd)
                if p.ndim >= 2:
                    O = zeropower_via_newtonschulz5_jax(m_new, steps=3)
                    rows = p.shape[-2]
                    cols = p.shape[-1]
                    aspect_scale = math.sqrt(max(1.0, float(rows) / float(cols)))
                    p_upd = p_decayed - (lr * aspect_scale) * O
                else:
                    p_upd = p_decayed - lr * m_new

                pool_diff_sq = pool_diff_sq + jnp.sum(jnp.square(p_upd - p))
                new_pool_p[k] = p_upd
            pool_upd_norm = jnp.sqrt(pool_diff_sq)

        return new_rad_p, new_pool_p, new_opt_m, new_opt_v_power, loss_val, rad_g_norm, pool_g_norm, rad_upd_norm, pool_upd_norm

    return forward_state_and_loss, step_fn

def evaluate_full_metrics(rad_p, pool_p, frozen_p, val_batches, r_mix, forward_fn):
    losses = []
    states = []
    int_abl_losses = []
    pool_abl_losses = []
    core_abl_losses = []
    surf_abl_losses = []

    for b in val_batches:
        l, st = forward_fn(rad_p, pool_p, frozen_p, b, r_mix)
        losses.append(float(l))
        states.append(np.array(st))

        # Interior ablation (S1-14 = 0 in input state)
        st_int_zero = b[0].at[:, 1:15, :].set(0.0)
        l_int, _ = forward_fn(rad_p, pool_p, frozen_p, b, r_mix, {"state_in": st_int_zero})
        int_abl_losses.append(float(l_int))

        # Pool ablation (pool_mult = 0.0)
        l_pool, _ = forward_fn(rad_p, pool_p, frozen_p, b, r_mix, {"pool_mult": 0.0})
        pool_abl_losses.append(float(l_pool))

        # Core ablation (core_mult = 0.0)
        l_core, _ = forward_fn(rad_p, pool_p, frozen_p, b, r_mix, {"core_mult": 0.0})
        core_abl_losses.append(float(l_core))

        # Surface ablation (surf_mult = 0.0)
        l_surf, _ = forward_fn(rad_p, pool_p, frozen_p, b, r_mix, {"surf_mult": 0.0})
        surf_abl_losses.append(float(l_surf))

    mean_nll = float(np.mean(losses))
    int_util = float(np.mean(int_abl_losses)) - mean_nll
    pool_util = float(np.mean(pool_abl_losses)) - mean_nll
    core_util = float(np.mean(core_abl_losses)) - mean_nll
    surf_util = float(np.mean(surf_abl_losses)) - mean_nll

    cat_states = np.concatenate(states, axis=0) # [N=256, K=16, D=640]
    N, K, D = cat_states.shape

    cka_mat = np.zeros((14, 14))
    for i in range(14):
        for j in range(i, 14):
            c = compute_linear_cka(cat_states[:, i+1, :], cat_states[:, j+1, :])
            cka_mat[i, j] = c
            cka_mat[j, i] = c
    mean_interior_cka = float(np.mean(cka_mat))

    adj_disp = [float(np.mean(np.linalg.norm(cat_states[:, k+1, :] - cat_states[:, k, :], axis=-1))) for k in range(K-1)]
    mean_adj_disp = float(np.mean(adj_disp))

    # Shell weights softmax
    shell_w = [float(x) for x in jax.nn.softmax(pool_p["motor.shell_score"], axis=0)]

    return {
        "nll": mean_nll,
        "interior_utility": int_util,
        "pool_utility": pool_util,
        "core_utility": core_util,
        "surface_utility": surf_util,
        "interior_cka": mean_interior_cka,
        "adj_displacement": mean_adj_disp,
        "shell_weights": shell_w,
        "shell_scale": float(pool_p["motor.shell_scale"]),
    }

def run_phase2():
    rad_base, pool_base, frozen_base, cfg, buffers = load_foundation()
    forward_fn, step_fn = build_phase2_engine(cfg, buffers)
    init_checksum = get_frozen_checksum(frozen_base)
    print(f"Verified initial frozen parameter SHA-256 checksum: {init_checksum[:16]}...")

    # Standardized Large-Sample Validation Set (N=256) and Untouched Test Set (N=128)
    print("Generating standardized validation set (N=256) and untouched test set (N=128)...")
    val_batches = generate_stream(cfg, total_sequences=256, batch_size=1, seed=888)
    untouched_test_batches = generate_stream(cfg, total_sequences=128, batch_size=1, seed=999)

    arms = [
        {"name": "Arm_A_Frozen_Baseline", "r_mix": 0.009, "train_radial": False, "train_pool": False},
        {"name": "Arm_B_Readout_Only_Control", "r_mix": 0.009, "train_radial": False, "train_pool": True},
        {"name": "Arm_C_Low_Coupling_CoAdapt", "r_mix": 0.10, "train_radial": True, "train_pool": True},
        {"name": "Arm_D_Mod_Coupling_CoAdapt", "r_mix": 0.20, "train_radial": True, "train_pool": True},
    ]

    seeds = [42, 137, 2026]
    lr = 3e-5
    steps = 500
    eval_intervals = [0, 50, 100, 250, 500]

    all_results = {}

    for arm in arms:
        arm_name = arm["name"]
        r_mix = arm["r_mix"]
        tr_rad = arm["train_radial"]
        tr_pool = arm["train_pool"]
        print(f"\n==================================================")
        print(f"RUNNING: {arm_name} (r_mix={r_mix}, Train Radial={tr_rad}, Train Pool={tr_pool})")
        print(f"==================================================")

        all_results[arm_name] = {"r_mix": r_mix, "train_radial": tr_rad, "train_pool": tr_pool, "seeds": {}}

        if not tr_rad and not tr_pool:
            # Baseline evaluation
            metrics_0 = evaluate_full_metrics(rad_base, pool_base, frozen_base, val_batches, r_mix, forward_fn)
            test_nll_base = evaluate_full_metrics(rad_base, pool_base, frozen_base, untouched_test_batches, r_mix, forward_fn)["nll"]
            print(f"  Step 0 Baseline (N=256): Val NLL={metrics_0['nll']:.4f} | CKA={metrics_0['interior_cka']:.4f} | Int Util={metrics_0['interior_utility']:+.4f} | Untouched Test NLL={test_nll_base:.4f}")
            all_results[arm_name]["baseline_metrics"] = metrics_0
            all_results[arm_name]["untouched_test_nll"] = test_nll_base
            continue

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            train_stream = generate_stream(cfg, total_sequences=steps, batch_size=1, seed=seed)

            # Reset parameters exactly from foundation base
            cur_rad_p = {k: jnp.array(v) for k, v in rad_base.items()}
            cur_pool_p = {k: jnp.array(v) for k, v in pool_base.items()}
            
            # Setup optimizer state
            opt_m = {}
            for k, v in rad_base.items(): opt_m[k] = jnp.zeros_like(v)
            for k, v in pool_base.items(): opt_m[k] = jnp.zeros_like(v)
            
            opt_v_power = {}
            for k in ["core.w_radial_in", "core.w_radial_out"]:
                shape = rad_base[k].shape
                v_init = jax.random.normal(jax.random.PRNGKey(seed), (shape[0], shape[-1], 1))
                v_init = v_init / jnp.linalg.norm(v_init, axis=1, keepdims=True)
                opt_v_power[k] = v_init

            seed_trajectory = {}

            # Step 0 eval
            m_0 = evaluate_full_metrics(cur_rad_p, cur_pool_p, frozen_base, val_batches, r_mix, forward_fn)
            seed_trajectory["step_0"] = m_0
            print(f"  Step   0: Val NLL={m_0['nll']:.4f} | CKA={m_0['interior_cka']:.4f} | Int Util={m_0['interior_utility']:+.4f} | Pool Util={m_0['pool_utility']:+.4f}")

            for step in range(1, steps + 1):
                batch = train_stream[step - 1]
                cur_rad_p, cur_pool_p, opt_m, opt_v_power, _, r_gn, p_gn, r_un, p_un = step_fn(
                    cur_rad_p, cur_pool_p, frozen_base, opt_m, opt_v_power, batch, r_mix, lr, 0.01, tr_rad, tr_pool
                )

                if step in eval_intervals:
                    m_step = evaluate_full_metrics(cur_rad_p, cur_pool_p, frozen_base, val_batches, r_mix, forward_fn)
                    m_step["rad_grad_norm"] = float(r_gn)
                    m_step["pool_grad_norm"] = float(p_gn)
                    m_step["rad_upd_norm"] = float(r_un)
                    m_step["pool_upd_norm"] = float(p_un)
                    seed_trajectory[f"step_{step}"] = m_step
                    print(f"  Step {step:3d}: Val NLL={m_step['nll']:.4f} | CKA={m_step['interior_cka']:.4f} | Int Util={m_step['interior_utility']:+.4f} | Pool Util={m_step['pool_utility']:+.4f} | Upd(R,P)=({float(r_un):.4f},{float(p_un):.4f})")

            # Final evaluation on untouched test set
            test_m = evaluate_full_metrics(cur_rad_p, cur_pool_p, frozen_base, untouched_test_batches, r_mix, forward_fn)
            seed_trajectory["untouched_test_nll"] = test_m["nll"]
            print(f"  Final Untouched Test NLL: {test_m['nll']:.4f}")

            post_checksum = get_frozen_checksum(frozen_base)
            assert post_checksum == init_checksum, f"Frozen parameter leakage detected in seed {seed}!"

            all_results[arm_name]["seeds"][str(seed)] = seed_trajectory

    out_file = "experiments/phase2_coadaptation_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nPhase 2 Benchmark Complete! Results persisted to: {out_file}")

if __name__ == "__main__":
    run_phase2()
