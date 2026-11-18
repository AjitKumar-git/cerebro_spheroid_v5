"""
Cerebro AutoTune: Standalone Accelerator & Architecture Auto-Tuning Engine.
"""

from autotune import (
    AutoTuner,
    TuneObjective,
    HardwareInfo,
    get_hardware_info,
    resolve_jax_compute_dtype,
    ArchitectureDetector,
    ArchitectureInfo,
    extract_architecture_signature,
    score_benchmark_result,
    calculate_loss_reduction_velocity,
    generate_vram_maximizing_candidates,
    CandidateLadder,
    HardwareConfigCache,
    load_local_hardware_config,
    save_local_hardware_config,
    LOCAL_CONFIG_PATH,
    BenchmarkResult,
    BenchmarkRunner,
    auto_tune_environment,
    run_auto_tuner,
)

__all__ = [
    "AutoTuner",
    "TuneObjective",
    "HardwareInfo",
    "get_hardware_info",
    "resolve_jax_compute_dtype",
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
