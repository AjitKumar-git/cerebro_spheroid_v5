#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CerebroSpheroid V5: Architectural Efficiency & Pareto Scaling Experiment
==============================================================================

Systematic empirical sweep across:
  - K (Concentric Shells): 16, 24, 32, 40, 48
  - D (Surface Dimension): 512, 640, 768, 1024
  - Dr (Radial Dimension / Ratio Dr/D): 0.25, 0.40, 0.50, 0.67
  - H (Attention Heads): 4, 8, 16
  - T_delib (Thinking Cycles): 1, 2, 4, 6

Evaluates the Pareto frontier between:
  1. Space Efficiency: Parameter count, active compute density, working state memory (KB).
  2. Thinking Capability: Reasoning convergence (delta loss on multi-step logic),
     deliberation gain (PPL reduction from T=1 to T=target), core-to-surface signal retention.
"""

import contextlib
import gc
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import src.spheroid_v5
from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, _motor_recur_block
from src.spheroid_optimizer import SpheroidOptimizer

# Use eager block during sweep to avoid repeated Triton JIT re-compilations on dynamic shapes
src.spheroid_v5._get_compiled_motor_recur_block = lambda: _motor_recur_block


@dataclass
class ArchitectureResult:
    name: str
    K: int
    D: int
    Dr: int
    heads: int
    tracts: int
    thinking_cycles: int
    total_params: int
    active_params: int
    active_ratio: float
    state_kb: float
    ms_per_step: float
    init_loss: float
    final_loss: float
    delta_loss: float
    delib_gain: float  # PPL(T=1) / PPL(T=target)
    core_transmission: float  # ||grad(s_0)|| / ||grad(s_{K-1})||
    efficiency_index: float


def generate_reasoning_benchmark(batch_size: int = 1, prompt_len: int = 16, target_len: int = 16, vocab_size: int = 1024, seed: int = 42, device: str = "cpu"):
    """
    Generates a synthetic algorithmic reasoning task:
    Requires tracking internal accumulator states and multi-step modular logic.
    """
    torch.manual_seed(seed)
    # Sensory prompts
    prompts = torch.randint(4, vocab_size - 1, (batch_size, prompt_len), device=device)
    # Targets depend causally on multi-hop state transitions from prompt
    targets = torch.zeros(batch_size, target_len, dtype=torch.long, device=device)
    targets[:, 0] = (prompts[:, -1] * 3 + 7) % (vocab_size - 4) + 4
    for t in range(1, target_len):
        targets[:, t] = (targets[:, t - 1] * 5 + prompts[:, t % prompt_len] * 2 + 11) % (vocab_size - 4) + 4
    return prompts, targets


def run_single_config(
    name: str,
    K: int,
    D: int,
    Dr: int,
    heads: int,
    tracts: int,
    thinking_cycles: int,
    device: str = "cpu",
    train_steps: int = 10,
    vocab_size: int = 2048,
) -> ArchitectureResult:
    cfg = ModelConfigV5(
        vocab_size=vocab_size,
        surface_dim=D,
        radial_dim=Dr,
        num_shells=K,
        num_shortcut_tracts=tracts,
        num_heads=heads,
        thinking_cycles=thinking_cycles,
        phase_scale=2.0,
        use_tied_embeddings=True,
        use_adaptive_thinking=False,
    )

    model = CerebroSpheroidV5(cfg).to(device)
    optimizer = SpheroidOptimizer(model.parameters(), lr=1e-3)

    total_params = sum(p.numel() for p in model.parameters())
    # Vocabulary lookup table size: V * D
    vocab_table_params = vocab_size * D
    active_params = total_params - vocab_table_params
    active_ratio = (active_params / max(1, total_params)) * 100.0

    # Spheroid persistent state footprint: K * D * 2 bytes (bfloat16)
    state_kb = (K * D * 2) / 1024.0

    # Benchmark inputs
    prompts, targets = generate_reasoning_benchmark(batch_size=1, prompt_len=16, target_len=16, vocab_size=vocab_size, device=device)

    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else contextlib.nullcontext()

    # Initial loss
    with torch.no_grad():
        with autocast_ctx:
            init_loss, _ = model(prompts, targets)
            init_loss_val = float(init_loss.item())

    # Training loop to measure convergence on reasoning
    t0 = time.time()
    losses = []
    core_transmission = 0.5
    for step in range(train_steps):
        optimizer.zero_grad()
        with autocast_ctx:
            loss, _ = model(prompts, targets)
        loss.backward()

        if step == train_steps - 1:
            # Measure gradient energy transmission from surface to core
            if hasattr(model, "core") and hasattr(model.core, "shell_gains") and model.core.shell_gains.grad is not None:
                gains_grad = model.core.shell_gains.grad
                grad_core = float(gains_grad[0].abs().item())
                grad_surf = float(gains_grad[-1].abs().item() + 1e-8)
                core_transmission = min(2.0, grad_core / grad_surf)

        optimizer.step()
        losses.append(float(loss.item()))

    dt = time.time() - t0
    ms_per_step = (dt / max(1, train_steps)) * 1000.0
    final_loss_val = float(losses[-1])
    delta_loss = max(0.0, init_loss_val - final_loss_val)

    # Measure Deliberation Gain: compare T=1 vs target T_delib
    with torch.no_grad():
        model.config.thinking_cycles = 1
        with autocast_ctx:
            loss_t1, _ = model(prompts, targets)
        ppl_t1 = math.exp(min(float(loss_t1.item()), 20.0))

        model.config.thinking_cycles = thinking_cycles
        with autocast_ctx:
            loss_target, _ = model(prompts, targets)
        ppl_target = math.exp(min(float(loss_target.item()), 20.0))

        delib_gain = max(1.0, ppl_t1 / max(1e-4, ppl_target))

    # Spheroid Efficiency Index (SEI):
    # Rewards reasoning loss reduction and deliberation gain
    # Penalizes parameter volume, state size, and step latency
    norm_latency = max(10.0, ms_per_step)
    log_params = math.log10(max(1e5, total_params))
    efficiency_index = (delta_loss * delib_gain * (active_ratio / 100.0) * 1000.0) / (
        log_params * math.sqrt(max(1.0, state_kb)) * (norm_latency / 100.0)
    )

    # Clean up device memory immediately
    del model, optimizer, prompts, targets
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return ArchitectureResult(
        name=name,
        K=K,
        D=D,
        Dr=Dr,
        heads=heads,
        tracts=tracts,
        thinking_cycles=thinking_cycles,
        total_params=total_params,
        active_params=active_params,
        active_ratio=active_ratio,
        state_kb=state_kb,
        ms_per_step=ms_per_step,
        init_loss=init_loss_val,
        final_loss=final_loss_val,
        delta_loss=delta_loss,
        delib_gain=delib_gain,
        core_transmission=core_transmission,
        efficiency_index=efficiency_index,
    )


def run_architectural_optimization_suite():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 105)
    print("⚡ CEREBROSPHEROID V5: ARCHITECTURAL EFFICIENCY & PARETO SCALING EXPERIMENT")
    print(f"Device: {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print("=" * 105)

    configs = [
        # --- Group 1: Shell Depth (K Sweep at D=640, Dr=256, H=8, T=2) ---
        ("K16_Baseline", 16, 640, 256, 8, 40, 2),
        ("K24_Intermediate", 24, 640, 256, 8, 64, 2),
        ("K32_Compact", 32, 640, 256, 8, 96, 2),
        ("K40_Extended", 40, 640, 256, 8, 120, 2),
        ("K48_Base", 48, 640, 256, 8, 144, 2),

        # --- Group 2: Radial-to-Surface Ratio (Dr / D at K=32, D=640, H=8, T=2) ---
        ("Dr160_Narrow (0.25x)", 32, 640, 160, 8, 96, 2),
        ("Dr256_Standard (0.40x)", 32, 640, 256, 8, 96, 2),
        ("Dr320_Balanced (0.50x)", 32, 640, 320, 8, 96, 2),
        ("Dr426_Wide (0.67x)", 32, 640, 426, 8, 96, 2),

        # --- Group 3: Attention Heads (H Sweep at K=32, D=640, Dr=256, T=2) ---
        ("H4_Coarse", 32, 640, 256, 4, 96, 2),
        ("H8_Standard", 32, 640, 256, 8, 96, 2),
        ("H16_Fine", 32, 640, 256, 16, 96, 2),

        # --- Group 4: Thinking Cycles (T Sweep at K=32, D=640, Dr=256, H=8) ---
        ("T1_Direct", 32, 640, 256, 8, 96, 1),
        ("T2_DualWave", 32, 640, 256, 8, 96, 2),
        ("T4_DeepDelib", 32, 640, 256, 8, 96, 4),
        ("T6_ExtendedDelib", 32, 640, 256, 8, 96, 6),

        # --- Group 5: Surface Width (D Sweep at K=32, Dr=0.4D, H=8, T=2) ---
        ("D512_Light", 32, 512, 204, 8, 96, 2),
        ("D768_Wide", 32, 768, 307, 8, 96, 2),
    ]

    results: List[ArchitectureResult] = []

    for idx, (name, K, D, Dr, heads, tracts, T) in enumerate(configs, 1):
        print(f"[{idx:2d}/{len(configs):2d}] Testing {name:<24} | K={K:2d}, D={D:4d}, Dr={Dr:3d}, H={heads:2d}, T={T}...", end="", flush=True)
        try:
            res = run_single_config(
                name=name,
                K=K,
                D=D,
                Dr=Dr,
                heads=heads,
                tracts=tracts,
                thinking_cycles=T,
                device=device,
                train_steps=10,
            )
            results.append(res)
            print(f" ✓ Done! ΔLoss: {res.delta_loss:.4f} | DelibGain: {res.delib_gain:.2f}x | Latency: {res.ms_per_step:5.1f}ms | SEI: {res.efficiency_index:.2f}")
        except Exception as e:
            print(f" ⚠️ Failed: {e}")

    # Rank results by Spheroid Efficiency Index (SEI)
    results.sort(key=lambda r: r.efficiency_index, reverse=True)

    print("\n" + "=" * 125)
    print("🏆 FINAL ARCHITECTURAL EFFICIENCY & PARETO RANKING (Ranked by Spheroid Efficiency Index)")
    print("=" * 125)
    print(f"{'Rank':<4} {'Configuration':<24} {'K':<3} {'D':<4} {'Dr':<4} {'H':<2} {'T':<2} {'Params':<8} {'State':<7} {'Active%':<8} {'ms/step':<8} {'ΔLoss':<7} {'DelibGain':<9} {'SEI':<7}")
    print("-" * 125)

    for rank, r in enumerate(results, 1):
        p_str = f"{r.total_params/1e6:.1f}M"
        st_str = f"{r.state_kb:.1f}KB"
        act_str = f"{r.active_ratio:.1f}%"
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"{rank:2d}."))
        print(
            f"{medal:<4} {r.name:<24} {r.K:2d}  {r.D:4d} {r.Dr:4d} {r.heads:2d} {r.thinking_cycles:2d} "
            f"{p_str:<8} {st_str:<7} {act_str:<8} {r.ms_per_step:6.1f}ms {r.delta_loss:6.4f} {r.delib_gain:8.2f}x  {r.efficiency_index:6.2f}"
        )

    print("=" * 125)


if __name__ == "__main__":
    run_architectural_optimization_suite()
