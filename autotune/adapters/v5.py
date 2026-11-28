"""
CerebroSpheroid V5 Architecture Adapter for Auto-Tuning.
=======================================================

Encapsulates all CerebroSpheroid V5 specific model creation, parameter
initialization, AOT XLA step compilation, and execution profiling.
Optimized with template caching, fast array generation, and zero-overhead execution.
"""

from __future__ import annotations

import gc
import math
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


def benchmark_v5_pytorch_model(
    preset: str = "v5_small",
    batch_size: int = 32,
    remat: bool = True,
    seq_len: int = 512,
    num_steps: int = 2,
    dtype: str = "auto",
    prompt_len: int = 256,
    model_config: Optional[Any] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    worst_case_seq_len: Optional[int] = None,
    num_devices: int = 1,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Benchmarks PyTorch CerebroSpheroidV5 on CUDA/CPU and returns AutoTuner metrics."""
    import torch
    from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5

    hw = get_hardware_info()
    is_gpu = hw["is_gpu"]
    total_mem_bytes = hw["total_memory_bytes"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    pref = (dtype or "auto").lower()
    if pref in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    elif pref in ("fp16", "float16", "half"):
        amp_dtype = torch.float16
    elif pref in ("fp32", "float32"):
        amp_dtype = torch.float32
    else:
        amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.get_device_capability(0)[0] >= 8) else (torch.float16 if use_cuda else torch.float32)

    if model_config is not None:
        config = model_config
    else:
        cfg_dict = dict(PRESETS_V5.get(preset, PRESETS_V5.get("v5_small", {})))
        if config_overrides:
            cfg_dict.update(config_overrides)
        config = ModelConfigV5(**cfg_dict)

    model = optimizer = prompts = targets = None
    try:
        if use_cuda:
            torch.cuda.set_device(0)
            torch.cuda.reset_peak_memory_stats(0)
        model = CerebroSpheroidV5(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        eff_seq = max(seq_len, worst_case_seq_len or 0)
        prompts = torch.randint(1, min(1000, config.vocab_size), (batch_size, prompt_len), device=device)
        targets = torch.randint(1, min(1000, config.vocab_size), (batch_size, eff_seq), device=device)

        t0 = time.perf_counter()
        init_loss = final_loss = None
        for s_idx in range(max(1, num_steps)):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_cuda and amp_dtype != torch.float32, dtype=amp_dtype):
                loss, _ = model(prompts, targets=targets, sampling_p=1.0, deliberation_mode="teacher_forcing")
            loss.backward()
            optimizer.step()
            if use_cuda:
                torch.cuda.synchronize(0)
            lv = float(loss.detach().item())
            if s_idx == 0:
                init_loss = lv
            final_loss = lv
            if s_idx == 0:
                t_warmup = time.perf_counter()

        compile_t = (t_warmup - t0) if num_steps > 1 else 0.0
        elapsed = max(1e-6, (time.perf_counter() - (t_warmup if num_steps > 1 else t0)))
        timed_steps = max(1, num_steps - 1) if num_steps > 1 else 1
        steps_per_sec = timed_steps / elapsed
        tok_per_step = batch_size * (prompt_len + eff_seq) * max(1, num_devices)
        tok_per_sec = steps_per_sec * tok_per_step
        peak_bytes = float(torch.cuda.max_memory_allocated(device)) if use_cuda else 0.0
        mem_gb = peak_bytes / (1024.0 ** 3)
        mem_pct = (peak_bytes / max(1.0, float(total_mem_bytes))) * 100.0 if total_mem_bytes else 0.0
        vel = calculate_loss_reduction_velocity(
            initial_loss=init_loss,
            final_loss=final_loss,
            num_steps=timed_steps,
            steps_per_sec=steps_per_sec,
            batch_size=batch_size,
            tok_per_sec=tok_per_sec,
        )
        return {
            "preset": preset,
            "batch_size": batch_size,
            "remat": remat,
            "seq_len": seq_len,
            "prompt_len": prompt_len,
            "dtype": str(amp_dtype).replace("torch.", ""),
            "compile_time": compile_t,
            "step_ms": (elapsed / timed_steps) * 1000.0,
            "steps_per_sec": steps_per_sec,
            "tok_per_sec": tok_per_sec,
            "memory_gb": mem_gb,
            "memory_utilization_pct": mem_pct,
            "memory_type": "VRAM" if is_gpu else "RAM",
            "vram_gb": mem_gb,
            "vram_utilization_pct": mem_pct,
            "initial_loss": init_loss,
            "final_loss": final_loss,
            "delta_loss": (init_loss - final_loss) if (init_loss is not None and final_loss is not None) else 0.0,
            "loss_reduction_per_sec": vel,
            "status": "OK",
        }
    except Exception as e:
        err_msg = str(e)
        status_desc = "OOM (Memory Exceeded)" if "out of memory" in err_msg.lower() else f"FAIL ({type(e).__name__}: {err_msg[:40]})"
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
            "memory_type": "VRAM" if is_gpu else "RAM",
            "vram_gb": 0.0,
            "vram_utilization_pct": 0.0,
            "initial_loss": None,
            "final_loss": None,
            "delta_loss": 0.0,
            "loss_reduction_per_sec": 0.0,
            "status": status_desc,
        }
    finally:
        del model, optimizer, prompts, targets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def benchmark_v5_model(
    preset: str = "v5_small",
    batch_size: int = 32,
    remat: bool = True,
    seq_len: int = 512,
    num_steps: int = 2,
    dtype: str = "auto",
    prompt_len: int = 256,
    model_config: Optional[Any] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    worst_case_seq_len: Optional[int] = None,
    num_devices: int = 1,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Benchmarks a single training configuration and returns step latency, steps/s,
    token throughput, loss reduction velocity, and memory metrics. Gracefully catches OOM.
    Enforces the worst-case envelope across distributed devices and elastic bucket sizes.
    """
    if str(kwargs.get("framework", "jax")).lower() in ("pytorch", "torch"):
        return benchmark_v5_pytorch_model(
            preset=preset,
            batch_size=batch_size,
            remat=remat,
            seq_len=seq_len,
            num_steps=num_steps,
            dtype=dtype,
            prompt_len=prompt_len,
            model_config=model_config,
            config_overrides=config_overrides,
            worst_case_seq_len=worst_case_seq_len,
            num_devices=num_devices,
            **kwargs,
        )
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

    if model_config is not None:
        config = model_config
    else:
        cfg_dict = dict(PRESETS_V5.get(preset, PRESETS_V5.get("v5_small", {})))
        if config_overrides:
            cfg_dict.update(config_overrides)
        config = ModelConfigV5(**cfg_dict)

    params = opt_state = train_step_fn = None
    init_loss_val: Optional[float] = None
    final_loss_val: Optional[float] = None
    dummy_prompts = dummy_targets = prompt_b = target_b = None

    try:
        # 1. Parameter & Step Function Retrieval / Initialization (with warm template caching)
        cfg_sig = (
            tuple(sorted((k, str(v)) for k, v in config.__dict__.items()))
            if hasattr(config, "__dict__")
            else str(config)
        )
        cache_key = (preset, dtype_str, cfg_sig)
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

        step_fn_key = (preset, remat, dtype_str, cfg_sig)
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
            num_devices=num_devices,
            worst_case_seq_len=worst_case_seq_len,
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

        # 3. Ahead-Of-Time XLA Compilation & Warmup. Use the same fixed
        # synthetic batch for baseline and timed steps so loss velocity is comparable.
        key = jax.random.PRNGKey(101)
        key, prompt_key, target_key = jax.random.split(key, 3)
        dummy_prompts = jax.random.randint(prompt_key, (batch_size, prompt_len), 0, config.vocab_size)
        dummy_targets = jax.random.randint(target_key, (batch_size, seq_len), 0, config.vocab_size)

        t0_compile = time.time()
        params, opt_state, init_loss = train_step_fn(params, opt_state, dummy_prompts, dummy_targets, 5e-4)
        jax.block_until_ready(init_loss)
        compile_time = time.time() - t0_compile
        init_loss_val = float(init_loss)

        # 3. High-Speed Timed Benchmark Steps (zero host-allocation latency)
        prompt_b, target_b = dummy_prompts, dummy_targets
        jax.block_until_ready(target_b)

        # Training throughput is measured in target/loss-bearing tokens.
        tokens_per_step = batch_size * seq_len
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
        if not math.isfinite(init_loss_val) or not math.isfinite(final_loss_val):
            raise FloatingPointError(
                f"Non-finite benchmark loss (initial={init_loss_val}, final={final_loss_val})"
            )
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
            "benchmark_steps": steps_to_run,
            "tokens_per_step": tokens_per_step,
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
