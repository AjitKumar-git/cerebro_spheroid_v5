"""
Autotune: High-Performance Accelerator & Architecture Auto-Tuning Engine
========================================================================

A modular, architecture-agnostic library for discovering optimal training
hyperparameters (batch size, activation recomputation/remat, precision) across
accelerators (Cloud TPUs, NVIDIA/AMD GPUs, and CPUs).

Key Capabilities:
-----------------
1. Dynamic Hardware Memory Profiling:
   Accurately detects TPU (v2-v6e), GPU (Ampere/Hopper/Blackwell/Ada/Turing),
   and VRAM/HBM memory capacity tiers (4GB, 8GB, 16GB, 24GB, 32GB, 80GB, 95GB+).

2. Architecture Detection & Intelligent Zero-Overhead Caching:
   Fingerprints model architectures. If an architecture is already verified on the
   current hardware, optimal parameters are returned in 0ms. Micro-benchmarks
   only trigger when a new architecture or unknown accelerator is detected.

3. Multi-Objective Pareto Optimization:
   - tokens/sec (throughput maximization)
   - steps/s (update frequency & latency minimization)
   - loss reduction/sec (loss descent velocity, dLoss/dt)
   - balanced (Pareto-optimal efficiency)

4. Safe OOM Mitigation & Branch Pruning:
   Gracefully traps accelerator OOM (HbmOom, RESOURCE_EXHAUSTED, CUDA OOM) and
   intelligently prunes impossible branches without aborting execution.
"""

from __future__ import annotations

from .objectives import TuneObjective
from .hardware import (
    get_hardware_info,
    resolve_jax_compute_dtype,
    HardwareInfo,
    _BF16_GPU_MARKERS,
)
from .architecture import (
    ArchitectureDetector,
    ArchitectureInfo,
    extract_architecture_signature,
)
from .scoring import (
    score_benchmark_result,
    calculate_loss_reduction_velocity,
)
from .search import (
    generate_vram_maximizing_candidates,
    CandidateLadder,
)
from .cache import (
    HardwareConfigCache,
    load_local_hardware_config,
    save_local_hardware_config,
    LOCAL_CONFIG_PATH,
)
from .benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
)
from .engine import (
    AutoTuner,
    auto_tune_environment,
    run_auto_tuner,
)

__all__ = [
    "AutoTuner",
    "TuneObjective",
    "HardwareInfo",
    "get_hardware_info",
    "resolve_jax_compute_dtype",
    "_BF16_GPU_MARKERS",
    "ArchitectureDetector",
    "ArchitectureInfo",
    "extract_architecture_signature",
    "score_benchmark_result",
    "calculate_loss_reduction_velocity",
    "generate_vram_maximizing_candidates",
    "CandidateLadder",
    "HardwareConfigCache",
    "load_local_hardware_config",
    "save_local_hardware_config",
    "LOCAL_CONFIG_PATH",
    "BenchmarkResult",
    "BenchmarkRunner",
    "auto_tune_environment",
    "run_auto_tuner",
]

__version__ = "1.0.0"
