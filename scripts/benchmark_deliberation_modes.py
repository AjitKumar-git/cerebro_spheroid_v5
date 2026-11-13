#!/usr/bin/env python3
"""
CerebroSpheroid V5: Empirical Benchmark of Deliberation Modes
============================================================
Benchmarks and compares:
1. Pure Teacher Forcing (Fast systolic baseline)
2. Standard Scheduled Sampling (sampling_p = 0.5)
3. Token-Level Interleaved Anchoring (Even-Odd Ping-Pong: even=1.0, odd=0.0)
4. Dual-Pass Twin Resonance (Teacher-forcing pass + Deliberative pass + Alignment)
5. Predictive Coding (Active inference: prev_thought + kappa*(tf_aud - prev_thought))
6. Step-Alternating (Round-robin: step 2k=TF, step 2k+1=Deliberation)

Evaluates on identical tokens:
- Initial vs Final Loss convergence
- Training Throughput (tok/s) and latency (ms/step)
- Autonomous Generation Loss (sampling_p = 0.0, exposure bias resistance)
- Standard Teacher-Forced Held-Out Loss (sampling_p = 1.0)
"""

import math
import os
import sys
import time
from typing import Dict, Any, List

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from src.spheroid_optimizer import SpheroidOptimizer


def generate_synthetic_corpus(num_batches: int, batch_size: int, prompt_len: int, target_len: int, vocab_size: int, seed: int = 42):
    """Generates synthetic reasoning sequences with repetitive patterns and logic transitions."""
    torch.manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        prompts = torch.randint(4, vocab_size - 1, (batch_size, prompt_len))
        # Build structured targets with causal transitions: next token = (prev + 3) % vocab
        targets = torch.zeros(batch_size, target_len, dtype=torch.long)
        targets[:, 0] = (prompts[:, -1] + 1) % vocab_size
        for t in range(1, target_len):
            targets[:, t] = (targets[:, t - 1] * 7 + 13) % (vocab_size - 4) + 4
        # Feedback is 1-shifted targets with BOS
        pad_col = torch.full((batch_size, 1), -100, dtype=torch.long)
        feedback = torch.cat([pad_col, targets[:, :-1]], dim=1)
        batches.append((prompts, targets, feedback))
    return batches


def run_benchmark(steps_per_mode: int = 30, batch_size: int = 4, prompt_len: int = 16, seq_len: int = 32):
    print("=" * 85)
    print("🧠 CerebroSpheroid V5: Alternating Deliberation vs Teacher Forcing Benchmark")
    print(f"   • Steps per mode : {steps_per_mode}")
    print(f"   • Batch Size     : {batch_size}")
    print(f"   • Prompt Length  : {prompt_len}")
    print(f"   • Sequence Length: {seq_len}")
    print("=" * 85)

    vocab_size = 1000
    cfg = dict(PRESETS_V5["v5_small"])
    cfg["vocab_size"] = vocab_size
    cfg["num_shells"] = 4
    cfg["surface_dim"] = 256
    cfg["radial_dim"] = 128
    cfg["num_shortcut_tracts"] = 16
    cfg["thinking_cycles"] = 2
    cfg["use_deliberation_probe"] = False
    cfg["use_adaptive_thinking"] = False
    cfg["num_mtp_heads"] = 1
    base_config = ModelConfigV5(**cfg)

    # Generate identical training and evaluation corpora
    train_batches = generate_synthetic_corpus(steps_per_mode + 5, batch_size, prompt_len, seq_len, vocab_size, seed=123)
    val_batches = generate_synthetic_corpus(8, batch_size, prompt_len, seq_len, vocab_size, seed=999)

    # Save initial weights for fair reset across all modes
    ref_model = CerebroSpheroidV5(base_config)
    init_state_dict = {k: v.clone() for k, v in ref_model.state_dict().items()}

    modes_to_test = [
        {"name": "1. Pure Teacher Forcing", "mode": "teacher_forcing", "sampling_p": 1.0, "is_step_alt": False},
        {"name": "2. Standard Scheduled Sampling", "mode": "scheduled", "sampling_p": 0.5, "is_step_alt": False},
        {"name": "3. Token-Level Interleaved", "mode": "interleaved", "sampling_p": 0.0, "is_step_alt": False},
        {"name": "4. Dual-Pass Twin Resonance", "mode": "dual_pass", "sampling_p": 0.5, "is_step_alt": False},
        {"name": "5. Predictive Coding (Kalman)", "mode": "predictive_coding", "sampling_p": 0.5, "is_step_alt": False},
        {"name": "6. Step-Alternating (TF / Delib)", "mode": "step_alt", "sampling_p": 0.5, "is_step_alt": True},
    ]

    results = []

    for cfg_mode in modes_to_test:
        mode_name = cfg_mode["name"]
        delib_mode = cfg_mode["mode"]
        sampling_p = cfg_mode["sampling_p"]
        is_step_alt = cfg_mode["is_step_alt"]

        print(f"\n▶ Evaluating: {mode_name}...")

        # Fresh model with identical initial weights
        model = CerebroSpheroidV5(base_config)
        model.load_state_dict(init_state_dict)
        model.train()

        optimizer = SpheroidOptimizer(model.parameters(), lr=1e-3, weight_decay=0.01)

        step_times = []
        losses = []

        total_tokens_trained = 0
        t0 = time.time()

        for step in range(steps_per_mode):
            p, y, fb = train_batches[step]
            step_tokens = p.numel() + y.numel()

            t_step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            if is_step_alt:
                # Even step = Teacher Forcing, Odd step = Deliberation
                cur_mode = "teacher_forcing" if (step % 2 == 0) else "scheduled"
                cur_p = 1.0 if (step % 2 == 0) else sampling_p
                out = model(p, targets=y, feedback_tokens=fb, sampling_p=cur_p, deliberation_mode=cur_mode)
            else:
                out = model(p, targets=y, feedback_tokens=fb, sampling_p=sampling_p, deliberation_mode=delib_mode)

            loss = out[0] if isinstance(out, tuple) else out
            loss.backward()
            optimizer.step()

            t_step_end = time.perf_counter()
            step_times.append(t_step_end - t_step_start)
            losses.append(loss.item())
            total_tokens_trained += step_tokens

        train_time = time.time() - t0
        avg_step_ms = (sum(step_times) / len(step_times)) * 1000.0
        tok_per_sec = total_tokens_trained / max(train_time, 1e-4)

        # Held-out evaluation
        model.eval()
        with torch.no_grad():
            # 1. Autonomous Deliberation Evaluation (sampling_p = 0.0: model must generate from its own thoughts)
            auto_losses = []
            for vp, vy, vfb in val_batches:
                out_auto = model(vp, targets=vy, feedback_tokens=vfb, sampling_p=0.0, deliberation_mode="scheduled")
                l_auto = out_auto[0] if isinstance(out_auto, tuple) else out_auto
                auto_losses.append(l_auto.item())
            eval_auto_loss = sum(auto_losses) / len(auto_losses)

            # 2. Teacher-Forced Evaluation (sampling_p = 1.0: standard ground truth cross-entropy)
            tf_losses = []
            for vp, vy, vfb in val_batches:
                out_tf = model(vp, targets=vy, feedback_tokens=vfb, sampling_p=1.0, deliberation_mode="teacher_forcing")
                l_tf = out_tf[0] if isinstance(out_tf, tuple) else out_tf
                tf_losses.append(l_tf.item())
            eval_tf_loss = sum(tf_losses) / len(tf_losses)

        res = {
            "name": mode_name,
            "init_loss": losses[0],
            "final_loss": losses[-1],
            "loss_drop": losses[0] - losses[-1],
            "step_ms": avg_step_ms,
            "tok_per_sec": tok_per_sec,
            "eval_auto_loss": eval_auto_loss,
            "eval_tf_loss": eval_tf_loss,
        }
        results.append(res)
        print(f"   ✓ Train Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (drop: {losses[0]-losses[-1]:+.4f})")
        print(f"   ✓ Throughput: {tok_per_sec:,.0f} tok/s ({avg_step_ms:.1f} ms/step)")
        print(f"   ✓ Eval Auto Loss (sampling_p=0.0): {eval_auto_loss:.4f} | Eval TF Loss: {eval_tf_loss:.4f}")

    print("\n" + "=" * 95)
    print("📊 COMPARATIVE BENCHMARK SUMMARY")
    print("=" * 95)
    print(f"{'Deliberation Mode':<32} | {'Train Loss':<14} | {'Eval Auto':<10} | {'Eval TF':<10} | {'Throughput':<12}")
    print("-" * 95)
    for r in results:
        train_str = f"{r['init_loss']:.2f}->{r['final_loss']:.2f}"
        print(f"{r['name']:<32} | {train_str:<14} | {r['eval_auto_loss']:<10.4f} | {r['eval_tf_loss']:<10.4f} | {r['tok_per_sec']:>7.0f} tok/s")
    print("=" * 95)
    return results


if __name__ == "__main__":
    run_benchmark()
