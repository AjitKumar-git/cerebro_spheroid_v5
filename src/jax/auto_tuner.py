"""
CerebroSpheroid V5: Pure JAX Auto-Tuning & Environment Benchmark Engine
========================================================================

Bridge module providing backward-compatible access to the standalone
`autotune` library while binding CerebroSpheroid V5 architectural adapters.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import autotune
from autotune import (
    AutoTuner,
    TuneObjective,
    HardwareConfigCache,
    LOCAL_CONFIG_PATH,
    _BF16_GPU_MARKERS,
    extract_architecture_signature,
    score_benchmark_result as _score_benchmark_result,
)
from autotune.hardware import (
    get_hardware_info as _autotune_get_hw,
    resolve_jax_compute_dtype as _autotune_resolve_dtype,
)
from autotune.search import generate_vram_maximizing_candidates as _autotune_gen_candidates
from autotune.cache import (
    load_local_hardware_config as _autotune_load_cache,
    save_local_hardware_config as _autotune_save_cache,
)
from autotune.adapters.v5 import benchmark_v5_model


# Re-export module-level bindings for direct monkeypatching and unittest mocking
def get_hardware_info() -> Dict[str, Any]:
    """Profiles active accelerator devices and determines memory capacity."""
    return _autotune_get_hw()


def generate_vram_maximizing_candidates(
    preset: str = "v5_small",
    total_gb: float = 16.0,
    seq_len: int = 256,
    max_batch_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generates candidate search ladder matching available memory."""
    return _autotune_gen_candidates(
        preset=preset,
        total_gb=total_gb,
        seq_len=seq_len,
        max_batch_size=max_batch_size,
    )


def load_local_hardware_config(preset: str, device_signature: str) -> Optional[Dict[str, Any]]:
    """Loads cached configuration from local filesystem if present."""
    return _autotune_load_cache(preset, device_signature)


def save_local_hardware_config(preset: str, device_signature: str, config: Dict[str, Any]) -> None:
    """Saves verified configuration to local filesystem."""
    _autotune_save_cache(preset, device_signature, config)


def resolve_jax_compute_dtype(dtype: str = "auto") -> Tuple[Any, str]:
    """Resolves optimal compute precision matching TPU MXU vs Ampere+ vs T4/CPU hardware."""
    # Use module-level get_hardware_info in case it is patched in tests
    hw = get_hardware_info()
    is_tpu = hw["is_tpu"]
    is_gpu = hw["is_gpu"]
    dev_kind = hw.get("device_kind", "Accelerator")
    dev_lower = dev_kind.lower()

    try:
        import jax.numpy as jnp
        bf16_t = jnp.bfloat16
        fp16_t = jnp.float16
        fp32_t = jnp.float32
    except ImportError:
        bf16_t = "bfloat16"
        fp16_t = "float16"
        fp32_t = "float32"

    if dtype == "auto":
        if is_tpu:
            return bf16_t, f"bfloat16 (Native {dev_kind} MXU)"
        if is_gpu and any(tok in dev_lower for tok in _BF16_GPU_MARKERS):
            return bf16_t, f"bfloat16 (Native {dev_kind} Tensor Core)"
        return fp32_t, f"float32 ({dev_kind if is_gpu else 'CPU'})"
    if dtype in ("bf16", "bfloat16"):
        return bf16_t, f"bfloat16 ({dev_kind})"
    if dtype in ("fp16", "float16"):
        return fp16_t, f"float16 ({dev_kind})"
    return fp32_t, f"float32 ({dev_kind})"


def benchmark_config(
    preset: str = "v5_small",
    batch_size: int = 32,
    remat: bool = True,
    seq_len: int = 512,
    num_steps: int = 3,
    dtype: str = "auto",
    prompt_len: int = 256,
) -> Dict[str, Any]:
    """Benchmarks a single training configuration using CerebroSpheroid V5 adapter."""
    return benchmark_v5_model(
        preset=preset,
        batch_size=batch_size,
        remat=remat,
        seq_len=seq_len,
        num_steps=num_steps,
        dtype=dtype,
        prompt_len=prompt_len,
    )


def auto_tune_environment(
    preset: str = "v5_small",
    candidates: Optional[List[Dict[str, Any]]] = None,
    seq_len: int = 256,
    dtype: str = "auto",
    use_cached_config: bool = True,
    force_tune: bool = False,
    upload_winner: bool = True,
    cloud_url: Optional[str] = None,
    cloud_token: Optional[str] = None,
    verbose: bool = True,
    tune_objective: str = "balanced",
    prompt_len: int = 256,
    max_batch_size: Optional[int] = None,
    fast: bool = True,
    num_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Explores candidate execution configurations and identifies the setting that
    optimizes optimizer steps/sec, token throughput, or loss reduction/sec without OOM.
    """
    # Create cache manager delegating to current module's load/save (supports monkeypatching)
    cache_mgr = HardwareConfigCache()
    cache_mgr.load = load_local_hardware_config
    cache_mgr.save = save_local_hardware_config

    tuner = AutoTuner(
        objective=tune_objective,
        cache_manager=cache_mgr,
        benchmark_fn=benchmark_config,
        hardware_info=get_hardware_info(),
        get_hardware_info_fn=get_hardware_info,
        use_cached_config=use_cached_config,
        upload_winner=upload_winner,
        cloud_url=cloud_url,
        cloud_token=cloud_token,
        verbose=verbose,
        fast=fast,
    )
    return tuner.tune(
        preset=preset,
        candidates=candidates,
        seq_len=seq_len,
        dtype=dtype,
        use_cached_config=use_cached_config,
        force_tune=force_tune,
        tune_objective=tune_objective,
        prompt_len=prompt_len,
        max_batch_size=max_batch_size,
        benchmark_fn=benchmark_config,
        fast=fast,
        num_steps=num_steps,
    )


# Backward-compatible alias
run_auto_tuner = auto_tune_environment

__all__ = [
    "AutoTuner",
    "get_hardware_info",
    "generate_vram_maximizing_candidates",
    "load_local_hardware_config",
    "save_local_hardware_config",
    "resolve_jax_compute_dtype",
    "benchmark_config",
    "auto_tune_environment",
    "run_auto_tuner",
    "_score_benchmark_result",
    "LOCAL_CONFIG_PATH",
    "_BF16_GPU_MARKERS",
]
