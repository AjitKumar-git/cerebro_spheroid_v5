"""
CerebroSpheroid V5 Architecture Adapter for Auto-Tuning.
=======================================================

Encapsulates all CerebroSpheroid V5 specific model creation, parameter
initialization, AOT XLA step compilation, and execution profiling.
Optimized with template caching, fast array generation, and zero-overhead execution.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, Optional, Tuple

from ..hardware import get_hardware_info, resolve_jax_compute_dtype
from ..scoring import calculate_loss_reduction_velocity
from ..memory import estimate_v5_peak_hbm

# Module-level template cache: avoids re-initializing identical random weight matrices
_V5_PARAM_TEMPLATE_CACHE: Dict[Tuple[str, str], Tuple[Any, Any, Any]] = {}
_V5_TRAIN_STEP_CACHE: Dict[Tuple[str, bool], Any] = {}


def clear_v5_adapter_cache() -> None:
    """Clears cached parameter templates and step functions."""
    _V5_PARAM_TEMPLATE_CACHE.clear()
    _V5_TRAIN_STEP_CACHE.clear()


def benchmark_v5_model(
    preset: str = "v5_small",
    batch_size: int = 32,
    remat: bool = True,
    seq_len: int = 512,
    num_steps: int = 2,
    dtype: str = "auto",
    prompt_len: int = 256,
) -> Dict[str, Any]:
    """
    Benchmarks a single training configuration and returns step latency, steps/s,
    token throughput, loss reduction velocity, and memory metrics. Gracefully catches OOM.
    """
    import jax
    import jax.numpy as jnp

    # Lazy import to prevent coupling the root library with v5 code
    from src.jax.v5_model import ModelConfigV5, PRESETS_V5, init_params_v5
    from src.jax.spheroid_optimizer import init_spheroid_opt_state, build_train_step_spheroid

    hw = get_hardware_info()
    is_tpu = hw["is_tpu"]
    is_gpu = hw["is_gpu"]
    total_mem_bytes = hw["total_memory_bytes"]

    target_dtype, _ = resolve_jax_compute_dtype(dtype)
    dtype_str = str(target_dtype).replace("jax.numpy.", "")

    cfg_dict = PRESETS_V5.get(preset, PRESETS_V5.get("v5_small", {}))
    config = ModelConfigV5(**cfg_dict)

    params = opt_state = train_step_fn = None
    init_loss_val: Optional[float] = None
    final_loss_val: Optional[float] = None
    dummy_prompts = dummy_targets = prompt_b = target_b = None

    try:
        # 1. Parameter & Step Function Retrieval / Initialization (with warm template caching)
        cache_key = (preset, dtype_str)
        if cache_key in _V5_PARAM_TEMPLATE_CACHE:
            params, buffers, opt_state = _V5_PARAM_TEMPLATE_CACHE[cache_key]
        else:
            key = jax.random.PRNGKey(42)
            key, init_key = jax.random.split(key)
            params, buffers = init_params_v5(init_key, config)
            if target_dtype != jnp.float32:
                params = {
                    k: v.astype(target_dtype) if jnp.issubdtype(v.dtype, jnp.floating) else v
                    for k, v in params.items()
                }
            opt_state = init_spheroid_opt_state(params)
            _V5_PARAM_TEMPLATE_CACHE[cache_key] = (params, buffers, opt_state)

        step_fn_key = (preset, remat)
        if step_fn_key in _V5_TRAIN_STEP_CACHE:
            train_step_fn = _V5_TRAIN_STEP_CACHE[step_fn_key]
        else:
            train_step_fn = build_train_step_spheroid(
                config=config,
                buffers=buffers,
                pad_id=0,
                weight_decay=1e-4,
                remat=remat,
            )
            _V5_TRAIN_STEP_CACHE[step_fn_key] = train_step_fn

        # 2. Accurate Theoretical HBM Memory Calculation based on tensor shapes:
        # Peak HBM ≈ Static Weights + Opt State + (Batch × SeqLen × Shells × BytesPerTok)
        theo_mem = estimate_v5_peak_hbm(
            config=config,
            params=params,
            batch_size=batch_size,
            prompt_len=prompt_len,
            seq_len=seq_len,
            remat=remat,
            dtype=target_dtype,
            total_memory_bytes=total_mem_bytes,
        )

        mem_label = "HBM" if is_tpu else ("VRAM" if is_gpu else "RAM")

        # Instant OOM Prediction: if mathematically guaranteed to exceed memory, prune in 0ms!
        if theo_mem.predicted_oom and total_mem_bytes and total_mem_bytes > 0:
            return {
                "preset": preset,
                "batch_size": batch_size,
                "remat": remat,
                "seq_len": seq_len,
                "prompt_len": prompt_len,
                "dtype": dtype_str,
                "compile_time": 0.0,
                "step_ms": None,
                "steps_per_sec": 0.0,
                "tok_per_sec": 0.0,
                "memory_gb": theo_mem.peak_hbm_gb,
                "memory_utilization_pct": theo_mem.hbm_utilization_pct,
                "memory_type": mem_label,
                "vram_gb": theo_mem.peak_hbm_gb,
                "vram_utilization_pct": theo_mem.hbm_utilization_pct,
                "initial_loss": None,
                "final_loss": None,
                "delta_loss": 0.0,
                "loss_reduction_per_sec": 0.0,
                "status": f"OOM (Theoretical Peak {theo_mem.peak_hbm_gb:.1f}GB exceeds {total_mem_bytes / (1024**3):.1f}GB {mem_label})",
            }

        # 3. Ahead-Of-Time XLA Compilation & Warmup
        dummy_prompts = jnp.zeros((batch_size, prompt_len), dtype=jnp.int32)
        dummy_targets = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

        t0_compile = time.time()
        params, opt_state, init_loss = train_step_fn(params, opt_state, dummy_prompts, dummy_targets, 5e-4)
        jax.block_until_ready(init_loss)
        compile_time = time.time() - t0_compile
        init_loss_val = float(init_loss)

        # 3. High-Speed Timed Benchmark Steps (zero host-allocation latency)
        key = jax.random.PRNGKey(101)
        key, step_key = jax.random.split(key)
        prompt_b = jax.random.randint(step_key, (batch_size, prompt_len), 0, config.vocab_size)
        target_b = jax.random.randint(step_key, (batch_size, seq_len), 0, config.vocab_size)
        jax.block_until_ready(target_b)

        tokens_per_step = batch_size * (prompt_len + seq_len)
        step_times = []
        last_loss = init_loss
        steps_to_run = max(1, int(num_steps))

        for _ in range(steps_to_run):
            t0_step = time.time()
            params, opt_state, loss = train_step_fn(params, opt_state, prompt_b, target_b, 5e-4)
            jax.block_until_ready(loss)
            step_times.append(time.time() - t0_step)
            last_loss = loss

        final_loss_val = float(last_loss)
        avg_step_sec = sum(step_times) / max(len(step_times), 1)
        steps_per_sec = 1.0 / max(avg_step_sec, 1e-6)
        tok_per_sec = tokens_per_step * steps_per_sec

        delta_loss = max(0.0, (init_loss_val - final_loss_val) if init_loss_val is not None else 0.0)
        loss_red_sec = calculate_loss_reduction_velocity(
            initial_loss=init_loss_val,
            final_loss=final_loss_val,
            num_steps=steps_to_run,
            steps_per_sec=steps_per_sec,
            batch_size=batch_size,
            tok_per_sec=tok_per_sec,
        )

        # 4. Memory Profiling: Use theoretical peak HBM based on tensor shapes
        # Peak HBM ≈ Static Weights + Opt State + (Batch × SeqLen × Shells × BytesPerTok)
        vram_gb = theo_mem.peak_hbm_gb
        vram_utilization_pct = theo_mem.hbm_utilization_pct

        # If live device memory_stats are available and higher, record peak
        dev = jax.devices()[0] if jax.devices() else None
        if dev and hasattr(dev, "memory_stats"):
            try:
                stats = dev.memory_stats()
                if stats:
                    live_bytes = stats.get("bytes_in_use") or stats.get("peak_bytes_in_use") or 0
                    if live_bytes > theo_mem.peak_hbm_bytes:
                        vram_gb = live_bytes / (1024**3)
                        limit = stats.get("bytes_limit", total_mem_bytes)
                        if limit and limit > 0:
                            vram_utilization_pct = min(100.0, (live_bytes / limit) * 100.0)
            except Exception:
                pass

        mem_label = "HBM" if is_tpu else ("VRAM" if is_gpu else "RAM")
        return {
            "preset": preset,
            "batch_size": batch_size,
            "remat": remat,
            "seq_len": seq_len,
            "prompt_len": prompt_len,
            "dtype": dtype_str,
            "compile_time": compile_time,
            "step_ms": avg_step_sec * 1000.0,
            "steps_per_sec": steps_per_sec,
            "tok_per_sec": tok_per_sec,
            "memory_gb": vram_gb,
            "memory_utilization_pct": vram_utilization_pct,
            "memory_type": mem_label,
            "vram_gb": vram_gb,
            "vram_utilization_pct": vram_utilization_pct,
            "initial_loss": init_loss_val,
            "final_loss": final_loss_val,
            "delta_loss": delta_loss,
            "loss_reduction_per_sec": loss_red_sec,
            "status": "PASS",
        }

    except Exception as e:
        err_msg = str(e)
        if "RESOURCE_EXHAUSTED" in err_msg or "HbmOom" in err_msg or "out of memory" in err_msg.lower():
            status_desc = "OOM (Memory Exceeded)"
            try:
                import jax
                jax.clear_caches()
                clear_v5_adapter_cache()
            except Exception:
                pass
        else:
            status_desc = f"FAIL ({type(e).__name__}: {err_msg[:40]})"

        mem_label = "HBM" if is_tpu else ("VRAM" if is_gpu else "RAM")
        return {
            "preset": preset,
            "batch_size": batch_size,
            "remat": remat,
            "seq_len": seq_len,
            "prompt_len": prompt_len,
            "dtype": str(dtype),
            "compile_time": None,
            "step_ms": None,
            "steps_per_sec": 0.0,
            "tok_per_sec": 0.0,
            "memory_gb": 0.0,
            "memory_utilization_pct": 0.0,
            "memory_type": mem_label,
            "vram_gb": 0.0,
            "vram_utilization_pct": 0.0,
            "initial_loss": None,
            "final_loss": None,
            "delta_loss": 0.0,
            "loss_reduction_per_sec": 0.0,
            "status": status_desc,
        }
    finally:
        dummy_prompts = None
        dummy_targets = None
        prompt_b = None
        target_b = None
        gc.collect()
