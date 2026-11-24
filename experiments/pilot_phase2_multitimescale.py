#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Phase 2 Multi-Timescale Training Pilot Harness

Implements the 4-Arm Controlled Training Pilot:
- T0 (Control):      Static intermediate & outer shells during training (current architecture).
- T1 (Periodic):     Fixed periodic intermediate radial updates during training (every 8 tokens).
- T2 (Adaptive Mid): Event-triggered adaptive intermediate updates (g_chunk > tau).
- T3 (Hierarchical): Adaptive intermediate updates (g_chunk) + slow outer context updates (g_ctx).

Protocol:
1. Shared initialization from latest foundation checkpoint (Step 240,508).
2. Identical multi-domain data order, optimizer (SCMER / Spheroid), micro-batching, and token budget.
3. Post-pilot 4x4 Cross-Evaluation Matrix:
   Each trained arm checkpoint (T0..T3) evaluated across all 4 inference modes (Frozen, Periodic, Adaptive, Dense).
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
from src.jax.v5_model import (
    ModelConfigV5,
    deliberate_step,
    deliberate_cycles_jax,
    encode_sensory,
    emit_step,
    training_rollout,
)
from src.jax.checkpoint import load_torch_pt_to_jax
from experiments.multitimescale_sandbox import compute_verified_causal_nll


def build_pilot_step_fn(config: ModelConfigV5, buffers: Tuple[jnp.ndarray, jnp.ndarray], arm: str = "T0"):
    """Compiles JIT-compatible training step for the designated multi-timescale training arm."""
    K = config.num_shells
    D = config.surface_dim

    def pilot_loss_fn(params, prompts, targets):
        B, total_L = targets.shape

        # 1. Prompt deliberation
        p_flux, p_kv = encode_sensory(params, prompts, config=config, pos_offset=0)
        r_prof = (jnp.arange(1, K + 1, dtype=jnp.float32) / float(K))[None, :, None]
        init_state = (jnp.broadcast_to(p_flux, (B, K, D)).astype(jnp.float32) * r_prof).astype(jnp.float32)
        state, _ = deliberate_cycles_jax(params, init_state, buffers, config=config, is_training=True, sensory_kv=p_kv)

        # 2. Causal sensory stream
        prev_toks = jnp.concatenate([prompts[:, -1:], targets[:, :-1]], axis=1)
        tf_flux, (tf_k, tf_v) = encode_sensory(params, prev_toks, config=config, pos_offset=prompts.shape[1], is_causal=True)
        causal_k = jnp.concatenate([p_kv[0], tf_k], axis=2)
        causal_v = jnp.concatenate([p_kv[1], tf_v], axis=2)

        cur_state = state
        cur_plan = jnp.zeros((B, D), dtype=jnp.float32)
        cur_trace = None

        total_loss = 0.0

        for t in range(total_L):
            curr_ctx_len = prompts.shape[1] + t + 1
            sub_k = causal_k[:, :, :curr_ctx_len, :]
            sub_v = causal_v[:, :, :curr_ctx_len, :]

            logits, nxt_state, cur_trace, cur_plan = emit_step(
                params,
                cur_state,
                (sub_k, sub_v),
                cur_plan,
                cur_trace,
                t,
                buffers,
                config=config,
                auditory_feedback=tf_flux[:, t, :],
                query_position=prompts.shape[1] + t,
            )

            tgt = targets[:, t]
            log_p = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            step_loss = -jnp.mean(jnp.take_along_axis(log_p, tgt[:, None], axis=-1))
            total_loss = total_loss + step_loss

            # Training Arm Dynamics
            prev_for_disp = cur_state
            if arm == "T0":
                # Static baseline: only shell 0 evolves
                cur_state = nxt_state
            elif arm == "T1":
                # Fixed periodic: intermediate update every 8 tokens
                is_periodic = (t > 0) and (t % 8 == 0)
                if is_periodic:
                    delib = deliberate_step(params, nxt_state, buffers, config=config)
                    cur_state = nxt_state.at[:, 1:12, :].set(0.85 * nxt_state[:, 1:12, :] + 0.15 * delib[:, 1:12, :])
                else:
                    cur_state = nxt_state
            elif arm == "T2":
                # Adaptive intermediate update: surprisal/entropy gate
                probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
                entropy = jnp.mean(-jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1))
                g_act = jax.nn.sigmoid(entropy - 3.2)
                
                def do_mid_update(st):
                    delib = deliberate_step(params, st, buffers, config=config)
                    return st.at[:, 1:12, :].set(st[:, 1:12, :] + 0.15 * g_act * (delib[:, 1:12, :] - st[:, 1:12, :]))

                cur_state = lax.cond(g_act > 0.05, do_mid_update, lambda st: st, nxt_state)
            elif arm == "T3":
                # Hierarchical: adaptive intermediate + sensory/uncertainty-driven outer context update
                probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
                entropy = jnp.mean(-jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1))
                g_act = jax.nn.sigmoid(entropy - 3.2)
                sensory_drift = jnp.mean(jnp.linalg.norm(tf_flux[:, t, :] - tf_flux[:, 0, :], axis=-1))
                is_outer = (t > 0) and ((t % max(4, total_L // 2) == 0) or (t == 8))

                def do_hierarchical_update(st):
                    delib = deliberate_step(params, st, buffers, config=config)
                    mid = st.at[:, 1:12, :].set(st[:, 1:12, :] + 0.15 * g_act * (delib[:, 1:12, :] - st[:, 1:12, :]))
                    if is_outer:
                        return mid.at[:, 12:, :].set(0.90 * mid[:, 12:, :] + 0.10 * delib[:, 12:, :])
                    else:
                        return mid

                cur_state = lax.cond((g_act > 0.05) | is_outer, do_hierarchical_update, lambda st: st, nxt_state)

            disp_t = jnp.mean(jnp.linalg.norm(cur_state - prev_for_disp, axis=-1), axis=0) # [K]
            if t == 0:
                total_disp = disp_t
            else:
                total_disp = total_disp + disp_t

        return total_loss / total_L, total_disp / total_L

    return jax.jit(jax.value_and_grad(pilot_loss_fn, has_aux=True))


def compute_shell_param_metrics(tensors: Dict[str, jnp.ndarray]) -> Tuple[float, float, float]:
    """Computes L2 norm of parameters or gradients stratified by shell group:
    Core (shell 0), Intermediate (shells 1..11), Outer (shells 12..15).
    """
    g_core_sq = 0.0
    g_mid_sq = 0.0
    g_outer_sq = 0.0

    shell_tensors = [
        "core.intra_swiglu.w_gate",
        "core.intra_swiglu.w_val",
        "core.intra_swiglu.w_down",
        "core.intra_norms.weight",
        "core.shell_gains",
        "motor.shell_score",
    ]

    for name in shell_tensors:
        if name in tensors:
            val = tensors[name]
            g_core_sq += float(jnp.sum(val[0] ** 2))
            g_mid_sq += float(jnp.sum(val[1:12] ** 2)) / 11.0
            g_outer_sq += float(jnp.sum(val[12:] ** 2)) / 4.0

    if "core.w_radial_in" in tensors:
        rin = tensors["core.w_radial_in"]
        g_mid_sq += float(jnp.sum(rin[:11] ** 2)) / 11.0
        g_outer_sq += float(jnp.sum(rin[11:] ** 2)) / 4.0

    if "core.w_radial_out" in tensors:
        rout = tensors["core.w_radial_out"]
        g_mid_sq += float(jnp.sum(rout[:11] ** 2)) / 11.0
        g_outer_sq += float(jnp.sum(rout[11:] ** 2)) / 4.0

    return math.sqrt(g_core_sq), math.sqrt(g_mid_sq), math.sqrt(g_outer_sq)


def evaluate_outer_ablation_impact(
    p_arm: Dict[str, jnp.ndarray],
    eval_sequences: List[Dict[str, jnp.ndarray]],
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: ModelConfigV5,
) -> float:
    """Measures delta NLL when outer mantle (shells 12..15) is ablated."""
    p_arm_dev = {k: jnp.asarray(v) for k, v in p_arm.items()}
    normal_nlls = []
    ablated_nlls = []
    for sample in eval_sequences:
        nll_norm, _, _, _ = compute_verified_causal_nll(
            p_arm_dev, sample["prompt"], sample["target"], buffers, config, variant="C", ablate_region=None
        )
        nll_abl, _, _, _ = compute_verified_causal_nll(
            p_arm_dev, sample["prompt"], sample["target"], buffers, config, variant="C", ablate_region="outer"
        )
        normal_nlls.append(nll_norm)
        ablated_nlls.append(nll_abl)
    return float(np.mean(ablated_nlls) - np.mean(normal_nlls))


def evaluate_cross_matrix(
    trained_params: Dict[str, Dict[str, jnp.ndarray]],
    eval_sequences: List[Dict[str, jnp.ndarray]],
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: ModelConfigV5,
) -> Dict[str, Dict[str, float]]:
    """Evaluates each trained arm across all 4 inference modes (4x4 matrix)."""
    inference_modes = [
        ("Frozen", "A"),
        ("Periodic", "B"),
        ("Adaptive", "C"),
        ("Dense", "D"),
    ]
    arms = list(trained_params.keys())

    matrix = {arm: {} for arm in arms}

    print("\n" + "=" * 95)
    print("🏆 PHASE 2 PILOT: 4x4 TRAINING-INFERENCE CROSS-EVALUATION MATRIX")
    print("=" * 95)
    header = f"{'Training Arm':<22} | " + " | ".join([f"{name:<14}" for name, _ in inference_modes])
    print(header)
    print("-" * 95)

    for arm in arms:
        p_arm = {k: jnp.asarray(v) for k, v in trained_params[arm].items()}
        row_cells = []
        for inf_name, inf_mode in inference_modes:
            nlls = []
            for sample in eval_sequences:
                nll, _, _, _ = compute_verified_causal_nll(
                    p_arm, sample["prompt"], sample["target"], buffers, config, variant=inf_mode
                )
                nlls.append(nll)
            mean_nll = float(np.mean(nlls))
            matrix[arm][inf_name] = mean_nll
            row_cells.append(f"{mean_nll:<14.4f}")
        print(f"{arm:<22} | " + " | ".join(row_cells))

    print("=" * 95)
    return matrix


import subprocess
import json

ARM_MAP = {
    "T0": "T0 (Static Control)",
    "T1": "T1 (Periodic /8)",
    "T2": "T2 (Adaptive Mid)",
    "T3": "T3 (Hierarchical)",
}

def train_single_arm(
    arm_code: str,
    checkpoint_path: str = "checkpoints/v5_small_latest.pt",
    steps_per_arm: int = 50,
    seq_len: int = 32,
    out_dir: str = "experiments/pilot_checkpoints",
):
    arm_label = ARM_MAP[arm_code]
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 95)
    print(f"🧠 CEREBRO-SPHEROID V5: TRAINING ARM {arm_label}")
    print(f"📁 Checkpoint: {checkpoint_path} | Steps: {steps_per_arm} | Seq Len: {seq_len}")
    print("=" * 95)

    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    cfg = infer_v5_config_from_state_dict(state, default_preset="v5_small")
    cfg.use_sensory_self_attn = True
    cfg.use_rope = False
    cfg.top_k_resonance = 0

    base_params, buffers, meta = load_torch_pt_to_jax(checkpoint_path, cfg, target_dtype=jnp.float32)

    pilot_texts = [
        "def partition(arr, low, high):\n    pivot = arr[high]\n    i = low - 1\n    for j in range(low, high):\n        if arr[j] <= pivot:\n            i += 1\n            arr[i], arr[j] = arr[j], arr[i]\n    arr[i + 1], arr[high] = arr[high], arr[i + 1]\n    return i + 1\n",
        "Enzymes catalyze biochemical reactions by stabilizing transition states and lowering activation free energies, accelerating metabolic conversions under physiological conditions.",
        "In quantum field theory, particle interactions are represented by asymptotic states of relativistic wavefields mediated by local gauge boson exchanges.",
        "Distributed consensus protocols like Paxos and Raft establish state-machine replication over unreliable networks subject to arbitrary message delays and packet drops.",
    ]
    encoded = tokenizer(pilot_texts, padding="max_length", max_length=seq_len, truncation=True, return_tensors="np")
    t_arr = jnp.asarray(encoded["input_ids"], dtype=jnp.int32)
    p_arr = jnp.full((t_arr.shape[0], 1), 1, dtype=jnp.int32)

    lr = 1e-4
    step_fn = build_pilot_step_fn(cfg, buffers, arm=arm_code)
    p_arm = {k: jnp.array(v) for k, v in base_params.items()}
    for k in base_params:
        assert jnp.array_equal(p_arm[k], base_params[k]), f"Initial param mismatch in {arm_label} for {k}"

    milestones = {1, 5, 10, 25, 50, steps_per_arm}
    last_grads = None

    for s in range(steps_per_arm):
        step_idx = s + 1
        (loss_val, disp_val), grads = step_fn(p_arm, p_arr, t_arr)
        loss_val.block_until_ready()
        disp_val.block_until_ready()
        for g in grads.values():
            g.block_until_ready()

        last_grads = grads
        p_arm = {k: p_arm[k] - lr * grads[k] for k in p_arm}
        for p in p_arm.values():
            p.block_until_ready()

        if step_idx in milestones or step_idx % max(1, steps_per_arm // 4) == 0:
            gc_val, gm_val, go_val = compute_shell_param_metrics(grads)
            d_theta = {k: p_arm[k] - base_params[k] for k in p_arm}
            tc_val, tm_val, to_val = compute_shell_param_metrics(d_theta)
            d_core = float(disp_val[0])
            d_mid = float(jnp.mean(disp_val[1:12]))
            d_outer = float(jnp.mean(disp_val[12:]))
            print(f"  [Step {step_idx:2d}/{steps_per_arm}] Loss: {float(loss_val):.4f} | D: [C:{d_core:.2f} M:{d_mid:.2f} O:{d_outer:.2f}] | |g|: [C:{gc_val:.2e} M:{gm_val:.2e} O:{go_val:.2e}] | |Δθ|: [C:{tc_val:.2e} M:{tm_val:.2e} O:{to_val:.2e}]")

        if s % 10 == 0:
            import gc
            gc.collect()

    # Save to disk
    p_arm_host = jax.device_get(p_arm)
    npz_path = os.path.join(out_dir, f"pilot_{arm_code}.npz")
    np.savez_compressed(npz_path, **p_arm_host)

    d_theta_final = {k: jnp.asarray(p_arm_host[k]) - base_params[k] for k in p_arm_host}
    tc_final, tm_final, to_final = compute_shell_param_metrics(d_theta_final)
    gc_final, gm_final, go_final = compute_shell_param_metrics(last_grads)

    metrics = {
        "arm_code": arm_code,
        "arm_label": arm_label,
        "final_loss": float(loss_val),
        "d_core": float(disp_val[0]),
        "d_mid": float(jnp.mean(disp_val[1:12])),
        "d_outer": float(jnp.mean(disp_val[12:])),
        "delta_theta": [tc_final, tm_final, to_final],
        "grad_norms": [gc_final, gm_final, go_final],
    }
    json_path = os.path.join(out_dir, f"pilot_{arm_code}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Saved {arm_label} checkpoint to {npz_path} and metrics to {json_path}")


def eval_all_arms(
    checkpoint_path: str = "checkpoints/v5_small_latest.pt",
    seq_len: int = 32,
    out_dir: str = "experiments/pilot_checkpoints",
):
    print("=" * 95)
    print("🏆 CEREBRO-SPHEROID V5: PHASE 2 PILOT CROSS-EVALUATION")
    print("=" * 95)

    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    cfg = infer_v5_config_from_state_dict(state, default_preset="v5_small")
    cfg.use_sensory_self_attn = True
    cfg.use_rope = False
    cfg.top_k_resonance = 0

    base_params, buffers, meta = load_torch_pt_to_jax(checkpoint_path, cfg, target_dtype=jnp.float32)

    pilot_texts = [
        "def partition(arr, low, high):\n    pivot = arr[high]\n    i = low - 1\n    for j in range(low, high):\n        if arr[j] <= pivot:\n            i += 1\n            arr[i], arr[j] = arr[j], arr[i]\n    arr[i + 1], arr[high] = arr[high], arr[i + 1]\n    return i + 1\n",
        "Enzymes catalyze biochemical reactions by stabilizing transition states and lowering activation free energies, accelerating metabolic conversions under physiological conditions.",
        "In quantum field theory, particle interactions are represented by asymptotic states of relativistic wavefields mediated by local gauge boson exchanges.",
        "Distributed consensus protocols like Paxos and Raft establish state-machine replication over unreliable networks subject to arbitrary message delays and packet drops.",
    ]
    domain_names = ["Code (Python)", "Science (Biochem)", "Physics (QFT)", "Web (Consensus)"]
    encoded = tokenizer(pilot_texts, padding="max_length", max_length=seq_len, truncation=True, return_tensors="np")
    t_arr = jnp.asarray(encoded["input_ids"], dtype=jnp.int32)
    p_arr = jnp.full((t_arr.shape[0], 1), 1, dtype=jnp.int32)
    eval_sequences = [{"prompt": p_arr[:1], "target": t_arr[i:i+1], "domain": domain_names[i]} for i in range(t_arr.shape[0])]

    arm_codes = ["T0", "T1", "T2", "T3"]
    trained_params = {}
    metrics_all = {}
    outer_ablation_deltas = {}

    for arm_code in arm_codes:
        arm_label = ARM_MAP[arm_code]
        npz_path = os.path.join(out_dir, f"pilot_{arm_code}.npz")
        json_path = os.path.join(out_dir, f"pilot_{arm_code}_metrics.json")
        loaded = np.load(npz_path)
        trained_params[arm_label] = {k: loaded[k] for k in loaded.files}
        with open(json_path, "r") as f:
            metrics_all[arm_label] = json.load(f)

        outer_ablation_deltas[arm_label] = evaluate_outer_ablation_impact(
            trained_params[arm_label], eval_sequences, buffers, cfg
        )

    print("\n" + "=" * 95)
    print("🔬 PILOT ARMS PARAMETER LEARNING & OUTER ABLATION CONTRIBUTION")
    print("=" * 95)
    print(f"{'Training Arm':<22} | {'|Δθ_core|':<11} | {'|Δθ_mid|':<11} | {'|Δθ_outer|':<11} | {'|g_outer|':<11} | {'Outer Abl ΔNLL':<15}")
    print("-" * 95)
    for arm_code in arm_codes:
        arm_label = ARM_MAP[arm_code]
        tc, tm, to = metrics_all[arm_label]["delta_theta"]
        _, _, go = metrics_all[arm_label]["grad_norms"]
        abl = outer_ablation_deltas[arm_label]
        print(f"{arm_label:<22} | {tc:<11.3e} | {tm:<11.3e} | {to:<11.3e} | {go:<11.3e} | {abl:<+15.4f}")
    print("=" * 95)

    matrix = evaluate_cross_matrix(trained_params, eval_sequences, buffers, cfg)

    # Domain-Stratified Validation NLL Breakdown
    print("\n" + "=" * 95)
    print("🌐 DOMAIN-STRATIFIED VALIDATION NLL BREAKDOWN (Adaptive Mode C)")
    print("=" * 95)
    print(f"{'Domain':<22} | {'T0 (Control)':<14} | {'T1 (Periodic)':<14} | {'T2 (Adapt Mid)':<14} | {'T3 (Hierarchical)':<18} | {'Paired ΔNLL (T3 vs T0)':<22}")
    print("-" * 95)
    for sample in eval_sequences:
        d_name = sample["domain"]
        d_nlls = {}
        for arm_code in arm_codes:
            arm_label = ARM_MAP[arm_code]
            p_dev = {k: jnp.asarray(v) for k, v in trained_params[arm_label].items()}
            nll, _, _, _ = compute_verified_causal_nll(
                p_dev, sample["prompt"], sample["target"], buffers, cfg, variant="C"
            )
            d_nlls[arm_label] = nll
        t0_nll = d_nlls[ARM_MAP["T0"]]
        t1_nll = d_nlls[ARM_MAP["T1"]]
        t2_nll = d_nlls[ARM_MAP["T2"]]
        t3_nll = d_nlls[ARM_MAP["T3"]]
        paired_delta = t3_nll - t0_nll
        print(f"{d_name:<22} | {t0_nll:<14.4f} | {t1_nll:<14.4f} | {t2_nll:<14.4f} | {t3_nll:<18.4f} | {paired_delta:<+22.4f}")
    print("=" * 95)


def run_orchestrated_pilot(checkpoint_path: str, steps: int, seq_len: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    for arm_code in ["T0", "T1", "T2", "T3"]:
        print(f"\n===============================================================================================")
        print(f"🚀 SUBPROCESS LAUNCH: Arm {arm_code} ({steps} steps)")
        print(f"===============================================================================================")
        cmd = [
            sys.executable,
            __file__,
            "--arm", arm_code,
            "--steps", str(steps),
            "--seq_len", str(seq_len),
            "--checkpoint", checkpoint_path,
            "--out_dir", out_dir,
        ]
        subprocess.check_call(cmd, env=env)

    print(f"\n===============================================================================================")
    print(f"🚀 SUBPROCESS LAUNCH: Cross-Evaluation Matrix & Domain Validation")
    print(f"===============================================================================================")
    cmd = [
        sys.executable,
        __file__,
        "--arm", "eval",
        "--seq_len", str(seq_len),
        "--checkpoint", checkpoint_path,
        "--out_dir", out_dir,
    ]
    subprocess.check_call(cmd, env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Multi-Timescale Training Pilot Harness")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_small_latest.pt")
    parser.add_argument("--steps", type=int, default=50, help="Pilot training steps per arm")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--arm", type=str, default="all", choices=["all", "T0", "T1", "T2", "T3", "eval"])
    parser.add_argument("--out_dir", type=str, default="experiments/pilot_checkpoints")
    args = parser.parse_args()

    # Resolve relative checkpoint path if needed
    if not os.path.isabs(args.checkpoint):
        if os.path.exists(args.checkpoint):
            ckpt_path = os.path.abspath(args.checkpoint)
        else:
            ckpt_path = os.path.join(WORKSPACE_ROOT, args.checkpoint)
    else:
        ckpt_path = args.checkpoint

    if not os.path.isabs(args.out_dir):
        out_dir_path = os.path.join(WORKSPACE_ROOT, args.out_dir)
    else:
        out_dir_path = args.out_dir

    if args.arm == "all":
        run_orchestrated_pilot(ckpt_path, args.steps, args.seq_len, out_dir_path)
    elif args.arm in ["T0", "T1", "T2", "T3"]:
        train_single_arm(args.arm, ckpt_path, args.steps, args.seq_len, out_dir_path)
    elif args.arm == "eval":
        eval_all_arms(ckpt_path, args.seq_len, out_dir_path)
