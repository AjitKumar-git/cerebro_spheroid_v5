"""
CerebroSpheroid V5: Pure JAX Auto-Tuning & Environment Benchmark Engine
========================================================================

Automatically tests the execution environment (Cloud TPU v5e-1, GPU, or CPU)
across candidate hyperparameters (batch sizes, remat/activation checkpointing)
to discover and configure the MAXIMUM tokens/second (tok/s) throughput.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from .v5_model import ModelConfigV5, PRESETS_V5, init_params_v5
from .spheroid_optimizer import init_spheroid_opt_state, build_train_step_spheroid


def benchmark_config(
    preset: str = "v5_small",
    batch_size: int = 32,
    remat: bool = True,
    seq_len: int = 256,
    num_steps: int = 5,
    dtype: str = "auto",
) -> Dict[str, Any]:
    """
    Benchmarks a single training configuration and returns throughput metrics.
    Gracefully catches OOM and compilation errors.
    """
    devices = jax.devices()
    is_tpu = any("TPU" in d.device_kind.upper() for d in devices)

    if dtype == "auto":
        target_dtype = jnp.bfloat16 if is_tpu else jnp.float32
    elif dtype == "bf16":
        target_dtype = jnp.bfloat16
    else:
        target_dtype = jnp.float32

    cfg_dict = PRESETS_V5.get(preset, PRESETS_V5["v5_small"])
    config = ModelConfigV5(**cfg_dict)

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)

    try:
        # 1. Initialize parameters
        params, buffers = init_params_v5(init_key, config)
        if target_dtype != jnp.float32:
            params = {k: v.astype(target_dtype) if jnp.issubdtype(v.dtype, jnp.floating) else v for k, v in params.items()}

        opt_state = init_spheroid_opt_state(params)
        train_step_fn = build_train_step_spheroid(
            config=config,
            buffers=buffers,
            pad_id=0,
            weight_decay=1e-4,
            remat=remat,
        )

        # 2. Measure Ahead-Of-Time XLA Compilation
        dummy_prompts = jnp.zeros((batch_size, 32), dtype=jnp.int32)
        dummy_targets = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

        t0_compile = time.time()
        params, opt_state, init_loss = train_step_fn(params, opt_state, dummy_prompts, dummy_targets, 5e-4)
        jax.block_until_ready(init_loss)
        compile_time = time.time() - t0_compile

        # 3. Micro-benchmark step execution
        tokens_per_step = batch_size * seq_len
        step_times = []

        for step in range(num_steps):
            key, step_key = jax.random.split(key)
            prompt_b = jax.random.randint(step_key, (batch_size, 32), 0, config.vocab_size)
            target_b = jax.random.randint(step_key, (batch_size, seq_len), 0, config.vocab_size)

            t0_step = time.time()
            params, opt_state, loss = train_step_fn(params, opt_state, prompt_b, target_b, 5e-4)
            jax.block_until_ready(loss)
            step_times.append(time.time() - t0_step)

        avg_step_sec = sum(step_times) / len(step_times)
        tok_per_sec = tokens_per_step / max(avg_step_sec, 1e-6)

        return {
            "preset": preset,
            "batch_size": batch_size,
            "remat": remat,
            "seq_len": seq_len,
            "compile_time": compile_time,
            "step_ms": avg_step_sec * 1000.0,
            "tok_per_sec": tok_per_sec,
            "initial_loss": float(init_loss),
            "status": "PASS",
        }

    except Exception as e:
        err_msg = str(e)
        if "RESOURCE_EXHAUSTED" in err_msg or "HbmOom" in err_msg:
            status_desc = "OOM (HBM Exceeded)"
        else:
            status_desc = f"FAIL ({type(e).__name__}: {err_msg[:40]})"

        return {
            "preset": preset,
            "batch_size": batch_size,
            "remat": remat,
            "seq_len": seq_len,
            "compile_time": None,
            "step_ms": None,
            "tok_per_sec": 0.0,
            "initial_loss": None,
            "status": status_desc,
        }


def auto_tune_environment(
    preset: str = "v5_small",
    candidates: Optional[List[Dict[str, Any]]] = None,
    seq_len: int = 256,
    dtype: str = "auto",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Explores candidate execution configurations and identifies the setting that
    maximizes token throughput on the active accelerator.
    """
    devices = jax.devices()
    is_tpu = any("TPU" in d.device_kind.upper() for d in devices)
    dev_kind = devices[0].device_kind if devices else "CPU"

    if verbose:
        print("=" * 80)
        print("🚀 CEREBROSPHEROID V5: AUTOMATIC THROUGHPUT BENCHMARK & AUTO-TUNER")
        print("=" * 80)
        print(f"Hardware Platform  : {len(devices)} x {dev_kind} ({'TPU MXU BF16' if is_tpu else 'Standard'})")
        print(f"Target Preset      : {preset} (SeqLen: {seq_len})")
        print(f"Benchmark Strategy : Micro-benchmark candidate batch sizes & activation remat")
        print("-" * 80)

    # Tailor candidates based on hardware
    if candidates is None:
        if is_tpu:
            # TPU v5e-1 has 16GB HBM with fast 128x128 systolic matrix units
            if preset in ("v5_small", "tpu_v5e_1"):
                candidates = [
                    {"batch_size": 64, "remat": True},
                    {"batch_size": 128, "remat": True},
                ]
            elif preset == "v5_compact":
                candidates = [
                    {"batch_size": 8, "remat": True},
                    {"batch_size": 16, "remat": True},
                    {"batch_size": 32, "remat": True},
                ]
            elif preset == "v5_test":
                candidates = [
                    {"batch_size": 4, "remat": True},
                    {"batch_size": 8, "remat": True},
                ]
            else:  # v5_base
                candidates = [
                    {"batch_size": 4, "remat": True},
                    {"batch_size": 8, "remat": True},
                ]
        else:
            # Lightweight candidates for GPU/CPU fallback testing
            candidates = [
                {"batch_size": 2, "remat": True},
                {"batch_size": 4, "remat": True},
            ]

    results = []
    best_config = None
    max_tok_s = -1.0

    for idx, c in enumerate(candidates, 1):
        b = c["batch_size"]
        r = c.get("remat", True)
        if verbose:
            print(f"[{idx}/{len(candidates)}] Testing: Batch {b:2d} | Remat: {str(r):5s} ...", end="", flush=True)

        res = benchmark_config(
            preset=preset,
            batch_size=b,
            remat=r,
            seq_len=seq_len,
            num_steps=3 if is_tpu else 2,
            dtype=dtype,
        )
        results.append(res)

        if res["status"] == "PASS":
            tok_s = res["tok_per_sec"]
            if verbose:
                print(f"  ✓ {tok_s:8,.0f} tok/s | Latency: {res['step_ms']:6.1f} ms | Compile: {res['compile_time']:5.1f}s")
            if tok_s > max_tok_s:
                max_tok_s = tok_s
                best_config = res
        else:
            if verbose:
                print(f"  ❌ {res['status']}")

    if verbose:
        print("\n" + "=" * 80)
        print("📊 AUTO-TUNING BENCHMARK RESULTS")
        print("=" * 80)
        print(f"{'Preset':<10} | {'Batch':<6} | {'Remat':<7} | {'Compile (s)':<12} | {'Step (ms)':<10} | {'Throughput':<15} | {'Status':<10}")
        print("-" * 80)
        for r in results:
            c_str = f"{r['compile_time']:.1f}s" if r['compile_time'] is not None else "-"
            s_str = f"{r['step_ms']:.1f}ms" if r['step_ms'] is not None else "-"
            t_str = f"{r['tok_per_sec']:,.0f} tok/s" if r['tok_per_sec'] > 0 else "-"
            winner_mark = " 🏆 (WINNER)" if (best_config and r == best_config) else ""
            print(f"{r['preset']:<10} | {r['batch_size']:<6} | {str(r['remat']):<7} | {c_str:<12} | {s_str:<10} | {t_str:<15} | {r['status'] + winner_mark:<10}")
        print("=" * 80)

    if best_config is None:
        # Fallback to safe default if all failed
        best_config = {
            "preset": preset,
            "batch_size": 16 if is_tpu else 2,
            "remat": True,
            "seq_len": seq_len,
            "tok_per_sec": 0.0,
            "status": "DEFAULT_FALLBACK",
        }
        if verbose:
            print("⚠️ All benchmark candidates failed. Falling back to safe conservative defaults.")
    else:
        if verbose:
            print(f"🏆 OPTIMAL CONFIGURATION SELECTED FOR MAXIMUM TOK/S:")
            print(f"   • Batch Size  : {best_config['batch_size']}")
            print(f"   • Remat Scan  : {best_config['remat']}")
            print(f"   • Peak Speed  : {best_config['tok_per_sec']:,.0f} tokens/second")
            print("=" * 80 + "\n")

    return best_config


# Backward-compatible alias
run_auto_tuner = auto_tune_environment
