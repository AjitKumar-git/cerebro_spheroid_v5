#!/usr/bin/env python3
"""
Comprehensive Read-Only Diagnostic Audit of CerebroSpheroid V5.
Strictly read-only with respect to foundation checkpoints and source code.

Phases Implemented:
  Phase 0: Establish exact system state, parameter shapes, scalars, and baseline config.
  Phase 1: Trace every path from loss to every shell (forward activations, energy, bypasses).
  Phase 2: Diagnose shell-by-shell gradient distribution, SNR, and cross-batch/cross-shell cosine similarities.
  Phase 3: Audit temporal credit assignment in autoregressive emission recurrence.
  Phase 4: Measure radial communication, transmission matrix T_{i->j}, and wave coupling behavior.
  Phase 5: Audit SpheroidOptimizer transfer function, Polar Newton-Schulz normalization, and update fidelity.
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
from jax import lax
import numpy as np
import torch
from transformers import AutoTokenizer

from src.jax.v5_model import (
    ModelConfigV5,
    deliberate_step,
    deliberate_cycles_jax,
    emit_step,
    generate_small_world_indices,
    rms_norm,
    swiglu,
    training_rollout,
)
from src.jax.spheroid_optimizer import (
    _apply_spheroid_grads,
    init_spheroid_opt_state,
    is_manifold_operator,
    zeropower_via_newtonschulz5_jax,
    clamp_spectral_norm_jax,
)

CHECKPOINT_PATH = "checkpoints/v5_small_latest.pt"
TOKENIZER_DIR = "data/tokenizer"
DATA_PATH = "data/consolidated/01_doty_identity/doty_identity_spheroid.jsonl"
OUTPUT_JSON = "experiments/comprehensive_read_only_audit_results.json"


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_baseline():
    print(f"Loading checkpoint {CHECKPOINT_PATH}...")
    sha256_hash = compute_sha256(CHECKPOINT_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    step = ckpt.get("step", -1)
    loss = float(ckpt.get("loss", -1.0))
    eval_loss = float(ckpt.get("eval_loss", -1.0))
    preset = ckpt.get("preset", "v5_small")

    ms = ckpt["model_state"]
    params = {k: jnp.array(v.float().numpy()) for k, v in ms.items() if hasattr(v, "numpy")}

    cfg = ModelConfigV5(
        surface_dim=640,
        radial_dim=256,
        num_shells=16,
        num_shortcut_tracts=40,
        num_heads=8,
        top_k_tracts=16,
        thinking_cycles=1,
        vocab_size=32000,
    )
    src_idx, dst_idx = generate_small_world_indices(cfg.num_shells, cfg.num_shortcut_tracts)
    buffers = (src_idx, dst_idx)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    return params, cfg, buffers, tokenizer, {
        "checkpoint_path": CHECKPOINT_PATH,
        "sha256": sha256_hash,
        "step": step,
        "training_loss": loss,
        "eval_loss": eval_loss,
        "preset": preset,
    }


def load_dataset_samples(tokenizer, num_samples: int = 8, prompt_len: int = 32, target_len: int = 16):
    prompts_list = []
    targets_list = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            inp = item.get("formatted_input", item.get("prompt", ""))
            out = item.get("formatted_output", item.get("answer", ""))

            p_enc = tokenizer.encode(inp, add_special_tokens=True)
            t_enc = tokenizer.encode(out, add_special_tokens=False)

            if len(p_enc) >= prompt_len and len(t_enc) >= target_len:
                p_tokens = p_enc[:prompt_len]
                t_tokens = t_enc[:target_len]
                prompts_list.append(p_tokens)
                targets_list.append(t_tokens)
            if len(prompts_list) >= num_samples:
                break

    # If dataset has fewer samples with exact length, pad
    while len(prompts_list) < num_samples:
        p_tokens = [1] + [100 + i for i in range(prompt_len - 1)]
        t_tokens = [200 + i for i in range(target_len)]
        prompts_list.append(p_tokens)
        targets_list.append(t_tokens)

    return jnp.array(prompts_list, dtype=jnp.int32), jnp.array(targets_list, dtype=jnp.int32)


# ==============================================================================
# PHASE 0: Baseline Specification
# ==============================================================================
def run_phase_0(params, meta, cfg):
    print("\n" + "=" * 50)
    print("PHASE 0: Verified Baseline Specification")
    print("=" * 50)

    r_mix_raw = float(params.get("core.radial_mix_scale", 0.0))
    s_mix_raw = float(params.get("core.shortcut_mix_scale", 0.0))
    r_mix_effective = float(np.clip(r_mix_raw, 0.01, 1.0))
    s_mix_effective = float(np.clip(s_mix_raw, 0.005, 1.0))

    shell_gains = [float(x) for x in params["core.shell_gains"]]
    input_gate = [float(x) for x in params["motor.input_gate"]]
    shell_score = [float(x) for x in params["motor.shell_score"]]
    shell_scale = float(params["motor.shell_scale"])
    temp_scale = float(params["motor.temperature_scale"])
    phase_scale = float(params["motor.phase_scale"])

    param_counts = {
        "total": sum(p.size for p in params.values()),
        "core": sum(p.size for k, p in params.items() if k.startswith("core.")),
        "motor": sum(p.size for k, p in params.items() if k.startswith("motor.")),
        "sensory": sum(p.size for k, p in params.items() if k.startswith("sensory.")),
    }

    spec = {
        "metadata": meta,
        "architecture": {
            "num_shells": cfg.num_shells,
            "surface_dim": cfg.surface_dim,
            "radial_dim": cfg.radial_dim,
            "num_shortcut_tracts": cfg.num_shortcut_tracts,
            "num_heads": cfg.num_heads,
            "vocab_size": cfg.vocab_size,
        },
        "coupling_parameters": {
            "radial_mix_scale_raw": r_mix_raw,
            "radial_mix_scale_effective": r_mix_effective,
            "is_radial_clipped": r_mix_raw < 0.01,
            "shortcut_mix_scale_raw": s_mix_raw,
            "shortcut_mix_scale_effective": s_mix_effective,
            "is_shortcut_clipped": s_mix_raw < 0.005,
            "shell_gains": shell_gains,
        },
        "readout_parameters": {
            "input_gate": input_gate,
            "shell_scale": shell_scale,
            "shell_score": shell_score,
            "temperature_scale": temp_scale,
            "phase_scale": phase_scale,
        },
        "parameter_counts": param_counts,
    }

    print(f"  Checkpoint SHA256: {meta['sha256'][:16]}...")
    print(f"  Training Step: {meta['step']} | Loss: {meta['training_loss']:.4f}")
    print(f"  Total Parameters: {param_counts['total']:,}")
    print(f"  r_mix raw={r_mix_raw:.6f} -> effective in graph={r_mix_effective:.4f} (clipped: {spec['coupling_parameters']['is_radial_clipped']})")
    print(f"  s_mix raw={s_mix_raw:.6f} -> effective in graph={s_mix_effective:.4f} (clipped: {spec['coupling_parameters']['is_shortcut_clipped']})")
    print(f"  Motor input gate: {[round(x, 4) for x in input_gate]}")
    print(f"  Motor shell scale: {shell_scale:.4f}")

    return spec


# ==============================================================================
# PHASE 1: Loss & Activation Pathway Tracing
# ==============================================================================
def run_phase_1(params, cfg, buffers, prompts, targets):
    print("\n" + "=" * 50)
    print("PHASE 1: Trace Loss Pathways & Activation Energy")
    print("=" * 50)

    # Compute sensory forward
    B = prompts.shape[0]
    total_prompt_L = prompts.shape[1]
    prompt_valid = (prompts != 0)[:, None, None, :]
    from src.jax.v5_model import encode_sensory
    sensory_flux, sensory_kv = encode_sensory(params, prompts, config=cfg, pos_offset=0, attn_mask=prompt_valid)

    # Prompt flux
    mask_2d = (prompts != 0).astype(jnp.float32)
    pos_idx = jnp.arange(total_prompt_L, dtype=jnp.float32)[None, :]
    last_valid_pos = jnp.max(jnp.where(mask_2d > 0.0, pos_idx, 0.0), axis=1, keepdims=True)
    recency_weights = jnp.exp(0.05 * (pos_idx - last_valid_pos)) * mask_2d
    weight_sum = jnp.maximum(jnp.sum(recency_weights, axis=1, keepdims=True), 1.0)
    prompt_flux = jnp.sum(sensory_flux.astype(jnp.float32) * recency_weights[:, :, None], axis=1) / weight_sum

    # Initial concentric state
    radial_profile = (jnp.arange(1, cfg.num_shells + 1, dtype=jnp.float32) / float(cfg.num_shells))[None, :, None]
    state_0 = prompt_flux[:, None, :] * radial_profile

    # Deliberate 1 cycle
    state_delib = deliberate_step(params, state_0, buffers, config=cfg)

    # Single emission step components
    surface = state_delib[:, -1, :]
    core = state_delib[:, 0, :]
    phase = params["motor.phase_embed.weight"][0]
    p_norm = jnp.maximum(jnp.linalg.norm(phase, axis=-1, keepdims=True), 1e-6)
    p_scaled = phase * (float(params.get("motor.phase_scale", 2.0)) / p_norm)
    motor_surface = surface + p_scaled

    # Resonance
    num_heads = cfg.num_heads
    head_dim = cfg.surface_dim // num_heads
    keys, values = sensory_kv
    q = jnp.matmul(motor_surface, params["motor.q_proj.weight"].T).reshape(B, 1, num_heads, head_dim)
    q = jnp.swapaxes(q, 1, 2)
    scores = jnp.matmul(q, jnp.swapaxes(keys, -1, -2)) / math.sqrt(head_dim)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    res_heads = jnp.matmul(attn_weights, values)
    res_heads = jnp.swapaxes(res_heads, 1, 2).reshape(B, cfg.surface_dim)
    resonance = rms_norm(jnp.matmul(res_heads, params["motor.out_proj.weight"].T), params["motor.resonance_norm.weight"])

    # Multi-shell pool
    shell_weights = jax.nn.softmax(params["motor.shell_score"], axis=0)
    shell_proj = jnp.matmul(state_delib, params["motor.shell_proj.weight"].T)
    shell_components = shell_weights[None, :, None] * shell_proj * params["motor.shell_scale"]
    shell_pool = jnp.sum(shell_components, axis=1)

    plan = jnp.zeros((B, cfg.surface_dim), dtype=jnp.float32)

    ig = jnp.clip(params.get("motor.input_gate", jnp.ones(5, dtype=jnp.float32)), -5.0, 5.0)

    act_surface = ig[0] * motor_surface
    act_core = ig[1] * core
    act_resonance = ig[2] * resonance
    act_pool = ig[3] * shell_pool
    act_plan = ig[4] * plan

    # Energy calculations
    e_surface = float(jnp.mean(jnp.sum(jnp.square(act_surface), axis=-1)))
    e_core = float(jnp.mean(jnp.sum(jnp.square(act_core), axis=-1)))
    e_resonance = float(jnp.mean(jnp.sum(jnp.square(act_resonance), axis=-1)))
    e_pool = float(jnp.mean(jnp.sum(jnp.square(act_pool), axis=-1)))
    e_plan = float(jnp.mean(jnp.sum(jnp.square(act_plan), axis=-1)))
    e_total = e_surface + e_core + e_resonance + e_pool + e_plan

    per_shell_pool_energies = [
        float(jnp.mean(jnp.sum(jnp.square(ig[3] * shell_components[:, k, :]), axis=-1)))
        for k in range(cfg.num_shells)
    ]

    pathway_table = [
        {"name": "1. Sensory Cross-Resonance", "input_source": "Sensory KV + Surface", "energy": e_resonance, "energy_pct": 100 * e_resonance / e_total, "bypasses_s1_14": True, "active_training": True, "active_inference": True},
        {"name": "2. Motor Surface Direct Tap", "input_source": "Shell 15", "energy": e_surface, "energy_pct": 100 * e_surface / e_total, "bypasses_s1_14": True, "active_training": True, "active_inference": True},
        {"name": "3. Core Direct Tap", "input_source": "Shell 0", "energy": e_core, "energy_pct": 100 * e_core / e_total, "bypasses_s1_14": True, "active_training": True, "active_inference": True},
        {"name": "4. Multi-Shell Pool", "input_source": "Shells 0-15 (Softmax)", "energy": e_pool, "energy_pct": 100 * e_pool / e_total, "bypasses_s1_14": False, "active_training": True, "active_inference": True},
        {"name": "5. Executive Plan", "input_source": "Premotor recurrent trace", "energy": e_plan, "energy_pct": 100 * e_plan / e_total, "bypasses_s1_14": True, "active_training": True, "active_inference": True},
    ]

    direct_bypass_energy_pct = 100.0 * (e_resonance + e_surface + e_core + e_plan) / e_total

    print("  Input Energy Breakdown into Deep Motor SwiGLU:")
    for p in pathway_table:
        print(f"    {p['name']:30s}: Energy = {p['energy']:10.4f} ({p['energy_pct']:6.2f}%) | Bypasses S1-14: {p['bypasses_s1_14']}")
    print(f"  -> Total Direct Bypass Energy (Non-Interior): {direct_bypass_energy_pct:.2f}%")
    print(f"  -> Total Multi-Shell Pool Energy: {100 * e_pool / e_total:.2f}%")

    return {
        "pathway_table": pathway_table,
        "direct_bypass_energy_pct": direct_bypass_energy_pct,
        "pool_energy_pct": 100.0 * e_pool / e_total,
        "per_shell_pool_energies": per_shell_pool_energies,
    }


# ==============================================================================
# PHASE 2: Shell Gradient Distribution & SNR Audit
# ==============================================================================
def run_phase_2(params, cfg, buffers, prompts, targets):
    print("\n" + "=" * 50)
    print("PHASE 2: Diagnose Shell Gradient Distribution, SNR & Alignment")
    print("=" * 50)

    num_batches = prompts.shape[0]
    batch_grads_w_down = []
    batch_grads_w_gate = []
    batch_grads_w_val = []
    batch_grads_norms = []
    batch_grads_radial_in = []
    batch_grads_radial_out = []
    batch_state_grads = []

    def single_batch_loss(p, pr, tg):
        return training_rollout(
            p,
            pr[None, :],
            target_len=tg.shape[0],
            buffers=buffers,
            config=cfg,
            targets=tg[None, :],
            pure_teacher_forcing=True,
            deliberation_mode="scheduled",
        )

    grad_fn = jax.value_and_grad(single_batch_loss)

    for b in range(num_batches):
        loss_val, grads = grad_fn(params, prompts[b], targets[b])

        # Parameter gradients
        batch_grads_w_down.append(np.array(grads["core.intra_swiglu.w_down"]))  # [16, 1280, 640]
        batch_grads_w_gate.append(np.array(grads["core.intra_swiglu.w_gate"]))  # [16, 640, 1280]
        batch_grads_w_val.append(np.array(grads["core.intra_swiglu.w_val"]))    # [16, 640, 1280]
        batch_grads_norms.append(np.array(grads["core.intra_norms.weight"]))    # [16, 640]
        batch_grads_radial_in.append(np.array(grads["core.w_radial_in"]))       # [15, 256, 256]
        batch_grads_radial_out.append(np.array(grads["core.w_radial_out"]))     # [15, 256, 256]

        # State sensitivity dL/ds_k
        def state_loss_fn(st):
            # Compute loss with perturbed deliberated state
            # Simplified emission loss directly w.r.t deliberated state
            surface = st[:, -1, :]
            core = st[:, 0, :]
            shell_weights = jax.nn.softmax(params["motor.shell_score"], axis=0)
            shell_proj = jnp.matmul(st, params["motor.shell_proj.weight"].T)
            shell_pool = jnp.sum(shell_weights[None, :, None] * shell_proj, axis=1) * params["motor.shell_scale"]

            ig = jnp.clip(params.get("motor.input_gate", jnp.ones(5)), -5.0, 5.0)
            fused_in = jnp.concatenate([
                ig[0] * surface,
                ig[1] * core,
                jnp.zeros_like(core),
                ig[3] * shell_pool,
                jnp.zeros_like(core),
            ], axis=-1)
            fused = rms_norm(fused_in, params["motor.fused_norm.weight"])
            h = rms_norm(
                swiglu(fused, params["motor.swiglu.fc_gate.weight"], params["motor.swiglu.fc_val.weight"], params["motor.swiglu.fc_down.weight"]),
                params["motor.norm_mid.weight"]
            )
            w_out = params.get("motor.head.weight", params["sensory.embedding.weight"])
            logits = jnp.matmul(h, w_out.T)
            tgt = targets[b, 0]
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            return -log_probs[0, tgt]

        # Initial concentric state
        p_flux = jnp.ones((1, cfg.surface_dim), dtype=jnp.float32)
        rad_prof = (jnp.arange(1, cfg.num_shells + 1, dtype=jnp.float32) / float(cfg.num_shells))[None, :, None]
        st0 = p_flux[:, None, :] * rad_prof
        st_delib = deliberate_step(params, st0, buffers, config=cfg)
        d_state = jax.grad(state_loss_fn)(st_delib)
        batch_state_grads.append(np.array(d_state[0]))  # [16, 640]

    # Convert to numpy arrays: shape [N, 16, ...]
    arr_w_down = np.stack(batch_grads_w_down, axis=0)
    arr_w_gate = np.stack(batch_grads_w_gate, axis=0)
    arr_w_val = np.stack(batch_grads_w_val, axis=0)
    arr_norms = np.stack(batch_grads_norms, axis=0)
    arr_state = np.stack(batch_state_grads, axis=0)

    # Shell metrics
    shell_metrics = []
    med_g = np.median([np.linalg.norm(np.mean(arr_w_down[:, k], axis=0)) for k in range(cfg.num_shells)])

    for k in range(cfg.num_shells):
        g_w_down = arr_w_down[:, k]  # [N, 1280, 640]
        mean_g = np.mean(g_w_down, axis=0)
        norm_mean_g = float(np.linalg.norm(mean_g))

        # Variance across batches
        var_g = np.mean([np.linalg.norm(g_w_down[b] - mean_g)**2 for b in range(num_batches)])
        std_g = math.sqrt(var_g)

        # Signal to Noise Ratio
        snr = float(norm_mean_g / (std_g + 1e-7))

        # Batch-to-batch cosine similarity
        cos_sims = []
        for i in range(num_batches):
            for j in range(i + 1, num_batches):
                ni = np.linalg.norm(g_w_down[i])
                nj = np.linalg.norm(g_w_down[j])
                if ni > 1e-8 and nj > 1e-8:
                    cos_sims.append(np.sum(g_w_down[i] * g_w_down[j]) / (ni * nj))
        mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0

        # State sensitivity
        g_state = arr_state[:, k]
        state_norm = float(np.mean([np.linalg.norm(g_state[b]) for b in range(num_batches)]))

        # Gate, val, norm gradient norms
        g_gate_norm = float(np.linalg.norm(np.mean(arr_w_gate[:, k], axis=0)))
        g_val_norm = float(np.linalg.norm(np.mean(arr_w_val[:, k], axis=0)))
        g_norm_norm = float(np.linalg.norm(np.mean(arr_norms[:, k], axis=0)))

        shell_metrics.append({
            "shell": k,
            "w_down_norm": norm_mean_g,
            "w_gate_norm": g_gate_norm,
            "w_val_norm": g_val_norm,
            "intra_norm_grad": g_norm_norm,
            "relative_gradient_norm": float(norm_mean_g / (med_g + 1e-8)),
            "state_sensitivity": state_norm,
            "gradient_snr": snr,
            "cross_batch_cosine_sim": mean_cos,
        })

    # Cross-shell cosine similarity matrix on w_down
    cross_shell_sim = np.zeros((cfg.num_shells, cfg.num_shells))
    mean_grads_per_shell = [np.mean(arr_w_down[:, k], axis=0) for k in range(cfg.num_shells)]
    for i in range(cfg.num_shells):
        for j in range(cfg.num_shells):
            ni = np.linalg.norm(mean_grads_per_shell[i])
            nj = np.linalg.norm(mean_grads_per_shell[j])
            if ni > 1e-8 and nj > 1e-8:
                cross_shell_sim[i, j] = np.sum(mean_grads_per_shell[i] * mean_grads_per_shell[j]) / (ni * nj)

    print("  Shell-by-Shell Gradient Norms, SNR & Cross-Batch Consistency:")
    print("  Shell |  w_down   |  State G  | Rel Norm |   SNR   | Batch CosSim")
    print("  " + "-" * 60)
    for m in shell_metrics:
        print(f"   S{m['shell']:02d}  | {m['w_down_norm']:9.5f} | {m['state_sensitivity']:9.5f} | {m['relative_gradient_norm']:8.3f} | {m['gradient_snr']:7.4f} | {m['cross_batch_cosine_sim']:8.4f}")

    return {
        "shell_metrics": shell_metrics,
        "cross_shell_sim_matrix": cross_shell_sim.tolist(),
    }


# ==============================================================================
# PHASE 3: Temporal Credit Assignment Audit in Emission
# ==============================================================================
def run_phase_3(params, cfg, buffers, prompts, targets):
    print("\n" + "=" * 50)
    print("PHASE 3: Audit Temporal Credit Assignment in Emission")
    print("=" * 50)

    # 1. Forward state evolution check
    B = 1
    total_prompt_L = prompts.shape[1]
    prompt_valid = (prompts[:1] != 0)[:, None, None, :]
    from src.jax.v5_model import encode_sensory
    sensory_flux, sensory_kv = encode_sensory(params, prompts[:1], config=cfg, pos_offset=0, attn_mask=prompt_valid)

    p_flux = sensory_flux[:, -1, :]
    rad_prof = (jnp.arange(1, cfg.num_shells + 1, dtype=jnp.float32) / float(cfg.num_shells))[None, :, None]
    state_0 = p_flux[:, None, :] * rad_prof
    state_init = deliberate_step(params, state_0, buffers, config=cfg)

    # Execute T=8 emission steps and record state at each step
    T = 8
    states_over_time = [state_init]
    cur_state = state_init
    cur_plan = jnp.zeros((B, cfg.surface_dim), dtype=jnp.float32)
    cur_trace = None

    for t in range(T):
        _, next_state, next_trace, next_plan = emit_step(
            params,
            cur_state,
            sensory_kv,
            cur_plan,
            cur_trace,
            step_idx=t,
            buffers=buffers,
            config=cfg,
        )
        states_over_time.append(next_state)
        cur_state = next_state
        cur_plan = next_plan
        cur_trace = next_trace

    # Check state displacement: || s_t[k] - s_0[k] ||
    displacement_matrix = np.zeros((T + 1, cfg.num_shells))
    for t in range(T + 1):
        for k in range(cfg.num_shells):
            diff = np.array(states_over_time[t][0, k, :]) - np.array(state_init[0, k, :])
            displacement_matrix[t, k] = float(np.linalg.norm(diff))

    # 2. Backward gradient propagation: dL_t / ds_init[k] for t in {0, 1, 3, 7}
    def loss_at_step_t(st_init, step_target_t):
        # Unroll emission up to step_target_t
        st = st_init
        pl = jnp.zeros((B, cfg.surface_dim), dtype=jnp.float32)
        tr = None
        for s in range(step_target_t):
            _, st, tr, pl = emit_step(params, st, sensory_kv, pl, tr, step_idx=s, buffers=buffers, config=cfg)
        logits, _, _, _ = emit_step(params, st, sensory_kv, pl, tr, step_idx=step_target_t, buffers=buffers, config=cfg)
        tgt = targets[0, step_target_t]
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -log_probs[0, tgt]

    temporal_grad_matrix = {}
    for test_t in [0, 1, 3, 7]:
        d_init = jax.grad(lambda s: loss_at_step_t(s, test_t))(state_init)
        temporal_grad_matrix[f"step_{test_t}"] = [
            float(np.linalg.norm(d_init[0, k, :])) for k in range(cfg.num_shells)
        ]

    print("  Forward State Displacement ||s_t[k] - s_0[k]||:")
    print("  Time t |    S00    |    S01    |    S07    |    S14    |    S15   ")
    print("  " + "-" * 55)
    for t in [0, 1, 2, 4, 8]:
        d = displacement_matrix[t]
        print(f"   t={t:02d}  | {d[0]:9.4f} | {d[1]:9.4f} | {d[7]:9.4f} | {d[14]:9.4f} | {d[15]:9.4f}")

    print("\n  Backward Gradient Sensitivity dL_t / ds_init[k]:")
    print("  Target t |    S00     |    S01     |    S07     |    S14     |    S15    ")
    print("  " + "-" * 60)
    for t in [0, 1, 3, 7]:
        g = temporal_grad_matrix[f"step_{t}"]
        print(f"   t={t:02d}    | {g[0]:10.5f} | {g[1]:10.5f} | {g[7]:10.5f} | {g[14]:10.5f} | {g[15]:10.5f}")

    is_s1_static = bool(np.all(displacement_matrix[1:, 1:] == 0.0))
    print(f"\n  -> Are shells S1-S15 completely static during emission? {is_s1_static}")

    return {
        "is_s1_15_static_in_forward": is_s1_static,
        "displacement_matrix": displacement_matrix.tolist(),
        "temporal_grad_matrix": temporal_grad_matrix,
    }


# ==============================================================================
# PHASE 4: Radial Communication & Transmission Matrix
# ==============================================================================
def run_phase_4(params, cfg, buffers):
    print("\n" + "=" * 50)
    print("PHASE 4: Diagnose Radial Communication & Transmission Matrix")
    print("=" * 50)

    # Reference state
    key = jax.random.PRNGKey(42)
    st_ref = jax.random.normal(key, (1, cfg.num_shells, cfg.surface_dim))

    def compute_transmission(r_mix_val: float):
        # Override radial mix scale
        p_copy = dict(params)
        p_copy["core.radial_mix_scale"] = jnp.array(r_mix_val, dtype=jnp.float32)

        # Full Jacobian T_{i->j} = || d(s_next[j]) / d(s_in[i]) ||_F
        def single_delib(s_in):
            return deliberate_step(p_copy, s_in, buffers, config=cfg)

        # Compute row-by-row directional derivatives via vector-Jacobian products
        T_mat = np.zeros((cfg.num_shells, cfg.num_shells))
        # Probe using orthogonal basis vectors
        num_probes = 16
        probe_keys = jax.random.split(key, num_probes)
        for i in range(cfg.num_shells):
            for p_idx in range(num_probes):
                v_probe = jax.random.normal(probe_keys[p_idx], (1, cfg.surface_dim))
                v_probe = v_probe / jnp.linalg.norm(v_probe)
                
                # Perturb shell i
                delta = 1e-3
                s_plus = st_ref.at[:, i, :].add(delta * v_probe)
                s_minus = st_ref.at[:, i, :].add(-delta * v_probe)
                out_plus = single_delib(s_plus)
                out_minus = single_delib(s_minus)
                
                diff = (out_plus - out_minus) / (2.0 * delta)  # [1, 16, 640]
                for j in range(cfg.num_shells):
                    T_mat[i, j] += float(np.linalg.norm(diff[0, j, :])) / num_probes

        return T_mat

    # Evaluate across r_mix sweep
    sweep_r_mix = [0.008972, 0.05, 0.10, 0.20, 0.40]
    sweep_results = {}
    for r in sweep_r_mix:
        t_mat = compute_transmission(r)
        self_c = float(np.mean([t_mat[k, k] for k in range(cfg.num_shells)]))
        out_c = float(np.mean([t_mat[k, k + 1] for k in range(cfg.num_shells - 1)]))
        in_c = float(np.mean([t_mat[k + 1, k] for k in range(cfg.num_shells - 1)]))
        sweep_results[str(r)] = {
            "T_matrix": t_mat.tolist(),
            "mean_self_coupling": self_c,
            "mean_outward_coupling": out_c,
            "mean_inward_coupling": in_c,
            "outward_to_self_ratio": out_c / self_c,
        }
        print(f"  r_mix = {r:8.6f} | Self: {self_c:7.4f} | Outward: {out_c:7.4f} | Inward: {in_c:7.4f} | Coupling/Self Ratio: {out_c/self_c:7.4f}")

    return sweep_results


# ==============================================================================
# PHASE 5: SpheroidOptimizer Update Audit & Polar Normalization
# ==============================================================================
def run_phase_5(params, cfg, buffers, prompts, targets):
    print("\n" + "=" * 50)
    print("PHASE 5: Audit SpheroidOptimizer Transfer Function & Polar Mechanics")
    print("=" * 50)

    # 1. Compute loss & grads on batch 0
    def single_batch_loss(p):
        return training_rollout(
            p,
            prompts[:1],
            target_len=targets.shape[1],
            buffers=buffers,
            config=cfg,
            targets=targets[:1],
            pure_teacher_forcing=True,
            deliberation_mode="scheduled",
        )

    loss_val, grads = jax.value_and_grad(single_batch_loss)(params)

    # Global gradient norm & clipping
    grad_norm = float(np.sqrt(sum(np.sum(np.square(np.array(g))) for g in grads.values() if hasattr(g, "dtype"))))
    clip_scale = float(np.minimum(1.0, 1.0 / (grad_norm + 1e-6)))

    # Optimizer state mock
    opt_s = init_spheroid_opt_state(params, key=jax.random.PRNGKey(42))

    # Apply SpheroidOptimizer step
    base_lr = 2e-3
    weight_decay = 0.01

    new_params, new_opt_s = _apply_spheroid_grads(
        params,
        opt_s,
        grads,
        base_lr=base_lr,
        weight_decay=weight_decay,
        scmer_lambda=0.0,
    )

    # Analyze update tensors for w_down: shape [16, 1280, 640]
    p_orig = np.array(params["core.intra_swiglu.w_down"])
    p_new = np.array(new_params["core.intra_swiglu.w_down"])
    delta_w_down = p_new - p_orig
    raw_g_w_down = np.array(grads["core.intra_swiglu.w_down"])

    opt_audit = []
    aspect_scale = math.sqrt(max(1.0, 1280.0 / 640.0))  # sqrt(2) ~= 1.4142

    for k in range(cfg.num_shells):
        g_k = raw_g_w_down[k]
        u_k = delta_w_down[k]
        p_k = p_orig[k]

        g_norm = float(np.linalg.norm(g_k))
        u_norm = float(np.linalg.norm(u_k))
        p_norm = float(np.linalg.norm(p_k))

        rel_update = float(u_norm / (p_norm + 1e-8))

        # Cosine similarity between raw gradient and update
        cos_align = float(-np.sum(g_k * u_k) / (g_norm * u_norm + 1e-8))

        opt_audit.append({
            "shell": k,
            "raw_grad_norm": g_norm,
            "clipped_grad_norm": g_norm * clip_scale,
            "update_norm": u_norm,
            "param_norm": p_norm,
            "relative_update": rel_update,
            "gradient_update_alignment": cos_align,
        })

    print(f"  Global Gradient Norm: {grad_norm:.5f} | Clip Scale: {clip_scale:.4f}")
    print("  Shell | Raw Grad Norm | Update Norm U_k | Rel Update R_k | Grad-Update Alignment")
    print("  " + "-" * 75)
    for m in opt_audit:
        print(f"   S{m['shell']:02d}  |   {m['raw_grad_norm']:11.5f} |   {m['update_norm']:11.5f}   |   {m['relative_update']:10.5f}   |       {m['gradient_update_alignment']:8.4f}")

    u0 = opt_audit[0]["update_norm"]
    u7 = opt_audit[7]["update_norm"]
    g0 = opt_audit[0]["raw_grad_norm"]
    g7 = opt_audit[7]["raw_grad_norm"]

    print(f"\n  -> Disparity Analysis (S0 vs S7):")
    print(f"     Raw Gradient Ratio (S0 / S7): {g0 / (g7 + 1e-8):.2f}x")
    print(f"     Optimizer Update Norm Ratio (S0 / S7): {u0 / (u7 + 1e-8):.4f}x (Normalized by Newton-Schulz!)")
    print(f"     S0 Grad-Update Alignment: {opt_audit[0]['gradient_update_alignment']:.4f}")
    print(f"     S7 Grad-Update Alignment: {opt_audit[7]['gradient_update_alignment']:.4f}")

    return {
        "global_grad_norm": grad_norm,
        "clip_scale": clip_scale,
        "opt_audit": opt_audit,
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("=" * 60)
    print("STARTING COMPREHENSIVE READ-ONLY CAUSAL AUDIT")
    print("=" * 60)

    params, cfg, buffers, tokenizer, meta = load_baseline()
    prompts, targets = load_dataset_samples(tokenizer, num_samples=8, prompt_len=32, target_len=16)

    phase_0_results = run_phase_0(params, meta, cfg)
    phase_1_results = run_phase_1(params, cfg, buffers, prompts, targets)
    phase_2_results = run_phase_2(params, cfg, buffers, prompts, targets)
    phase_3_results = run_phase_3(params, cfg, buffers, prompts, targets)
    phase_4_results = run_phase_4(params, cfg, buffers)
    phase_5_results = run_phase_5(params, cfg, buffers, prompts, targets)

    full_results = {
        "phase_0_baseline_specification": phase_0_results,
        "phase_1_pathway_tracing": phase_1_results,
        "phase_2_gradient_distribution": phase_2_results,
        "phase_3_temporal_credit_assignment": phase_3_results,
        "phase_4_radial_communication": phase_4_results,
        "phase_5_optimizer_audit": phase_5_results,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(full_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"COMPREHENSIVE AUDIT COMPLETE. Results saved to {OUTPUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
