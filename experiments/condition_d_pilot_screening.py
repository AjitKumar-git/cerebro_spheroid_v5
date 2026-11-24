"""
Condition D Pre-Flight Pilot: Stability & Gradient-Flow Screening across Learning Rates.
Memory-Optimized Implementation with pure closure JIT over config.
"""

import os
import sys
import math
import json
import hashlib
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

def generate_stream(cfg: ModelConfigV5, num_batches: int = 50, batch_size: int = 1, seq_len: int = 16, seed: int = 42):
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

def build_train_step(cfg: ModelConfigV5, buffers: Tuple[jnp.ndarray, jnp.ndarray]):
    def loss_fn_radial(rad_p, frozen_p, batch, r_mix_fixed):
        state, sensory_kv, plan, targets = batch
        p_full = dict(frozen_p)
        p_full.update(rad_p)
        p_full["core.radial_mix_scale"] = jnp.array(r_mix_fixed, dtype=jnp.float32)

        st_next = deliberate_step(p_full, state, buffers, config=cfg)
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
        return jnp.mean(nll)

    @jax.jit
    def step_fn(rad_p, frozen_p, opt_m, opt_v_power, batch, r_mix, lr, wd):
        loss_val, grads = jax.value_and_grad(loss_fn_radial)(rad_p, frozen_p, batch, r_mix)
        
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

    return loss_fn_radial, step_fn

def run_pilot():
    rad_base, frozen_base, cfg, buffers = load_foundation()
    loss_fn_radial, step_fn = build_train_step(cfg, buffers)
    init_checksum = get_frozen_checksum(frozen_base)
    print(f"Verified frozen parameter SHA-256 checksum: {init_checksum[:16]}...")

    r_mix_test = 0.10
    train_batches = generate_stream(cfg, num_batches=50, batch_size=1, seed=123)
    val_batches = generate_stream(cfg, num_batches=10, batch_size=1, seed=456)

    def eval_fn(rad_p):
        tot = sum(float(loss_fn_radial(rad_p, frozen_base, b, r_mix_test)) for b in val_batches)
        return tot / len(val_batches)

    val_base_nll = eval_fn(rad_base)
    print(f"Base NLL at r_mix=0.10 before training: {val_base_nll:.6f}\n")

    lrs = [1e-5, 3e-5, 1e-4]
    results = {}

    for lr in lrs:
        print(f"--- Screening Pilot LR: {lr} ---")
        cur_rad_p = {k: jnp.array(v) for k, v in rad_base.items()}
        opt_m = {k: jnp.zeros_like(v) for k, v in rad_base.items()}
        opt_v_power = {}
        for k in ["core.w_radial_in", "core.w_radial_out"]:
            shape = rad_base[k].shape
            v_init = jax.random.normal(jax.random.PRNGKey(42), (shape[0], shape[-1], 1))
            v_init = v_init / jnp.linalg.norm(v_init, axis=1, keepdims=True)
            opt_v_power[k] = v_init

        train_losses = []
        grad_norms = []

        for step in range(50):
            batch = train_batches[step]
            cur_rad_p, opt_m, opt_v_power, l_val, g_n = step_fn(
                cur_rad_p, frozen_base, opt_m, opt_v_power, batch, r_mix_test, lr, 0.01
            )
            train_losses.append(float(l_val))
            grad_norms.append(float(g_n))

        post_val_nll = eval_fn(cur_rad_p)
        post_checksum = get_frozen_checksum(frozen_base)
        checksum_verified = (post_checksum == init_checksum)

        w_in_np = np.array(cur_rad_p["core.w_radial_in"])
        sigmas = [float(np.linalg.svd(w_in_np[i], compute_uv=False)[0]) for i in range(15)]
        max_sigma = max(sigmas)

        print(f"  Step 0 Loss: {train_losses[0]:.4f} -> Step 50 Loss: {train_losses[-1]:.4f}")
        print(f"  Mean Radial Grad Norm: {np.mean(grad_norms):.4f}")
        print(f"  Val NLL: {val_base_nll:.4f} -> {post_val_nll:.4f} (Delta: {post_val_nll - val_base_nll:+.4f})")
        print(f"  Max Radial Spec Norm:  {max_sigma:.4f} (Clamp guard <= 1.0)")
        print(f"  Frozen Checksum Match: {checksum_verified}")

        results[str(lr)] = {
            "lr": lr,
            "train_loss_start": train_losses[0],
            "train_loss_end": train_losses[-1],
            "val_loss_start": val_base_nll,
            "val_loss_end": post_val_nll,
            "val_delta": post_val_nll - val_base_nll,
            "mean_grad_norm": float(np.mean(grad_norms)),
            "max_spectral_norm": max_sigma,
            "frozen_checksum_match": checksum_verified,
        }

    out_file = "experiments/condition_d_pilot_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPilot screening complete. Data saved to: {out_file}")

if __name__ == "__main__":
    run_pilot()
