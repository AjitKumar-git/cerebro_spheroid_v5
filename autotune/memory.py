"""
Accurate Theoretical HBM & VRAM Analytical Memory Estimator.
============================================================

Calculates theoretical peak accelerator memory (HBM on TPU / VRAM on GPU)
based on tensor shapes, parameter counts, optimizer state buffers, and
activation memory footprints:

    Peak HBM ≈ Static Weights + Opt State + (Batch × SeqLen × Shells × BytesPerTok)

Provides instant, zero-delay theoretical memory profiling and pre-execution
OOM prediction without relying on delayed post-step `dev.memory_stats()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union


@dataclass
class MemoryBreakdown:
    static_weights_bytes: int
    static_weights_gb: float
    opt_state_bytes: int
    opt_state_gb: float
    activation_bytes: int
    activation_gb: float
    peak_hbm_bytes: int
    peak_hbm_gb: float
    hbm_utilization_pct: float
    predicted_oom: bool
    collective_bytes: int = 0
    collective_gb: float = 0.0
    headroom_gb: float = 0.0
    is_greedy_safe: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "static_weights_bytes": self.static_weights_bytes,
            "static_weights_gb": self.static_weights_gb,
            "opt_state_bytes": self.opt_state_bytes,
            "opt_state_gb": self.opt_state_gb,
            "activation_bytes": self.activation_bytes,
            "activation_gb": self.activation_gb,
            "collective_bytes": self.collective_bytes,
            "collective_gb": self.collective_gb,
            "peak_hbm_bytes": self.peak_hbm_bytes,
            "peak_hbm_gb": self.peak_hbm_gb,
            "headroom_gb": self.headroom_gb,
            "hbm_utilization_pct": self.hbm_utilization_pct,
            "predicted_oom": self.predicted_oom,
            "is_greedy_safe": self.is_greedy_safe,
        }


def estimate_peak_hbm(
    param_bytes: int,
    batch_size: int,
    seq_len: int,
    prompt_len: int = 0,
    num_shells: int = 12,
    surface_dim: int = 768,
    remat: bool = False,
    bytes_per_tok: int = 2,
    opt_multiplier: float = 4.5,
    total_memory_bytes: Optional[int] = None,
    safety_margin: float = 0.90,
    vocab_size: int = 0,
    num_devices: int = 1,
) -> MemoryBreakdown:
    """
    Computes exact theoretical peak accelerator memory based on tensor shapes:

    1. Static Weights:
       W_bytes = sum(size * itemsize)

    2. Optimizer State:
       O_bytes = W_bytes * opt_multiplier (typically 4.5x for Spheroid in BF16:
       stores FP32 master weights (2x), FP32 first moments (2x), plus second moments (0.5x))

    3. Activation Footprint:
       - remat=False: all shell intermediate activations held simultaneously:
         A_bytes = Batch × (PromptLen + SeqLen) × SurfaceDim × BytesPerTok × (NumShells × 2)
       - remat=True: only shell boundary checkpoints stored; intermediate layers recomputed:
         A_bytes = Batch × (PromptLen + SeqLen) × SurfaceDim × BytesPerTok × 4

    4. Distributed Collective & Replica Buffers:
       - num_devices > 1: reserves cross-device pmean / all-reduce buffer capacity
    """
    # 1. Static weights and optimizer state (calibrated for FP32 master weights + moments in Spheroid)
    static_bytes = int(param_bytes)
    effective_opt_multiplier = float(opt_multiplier)
    if bytes_per_tok <= 2 and effective_opt_multiplier < 4.0:
        effective_opt_multiplier = 4.5
    opt_bytes = int(static_bytes * effective_opt_multiplier)

    # 2. Total sequence tokens per batch element
    total_tokens = max(1, prompt_len + seq_len)

    # 3. Activation multiplier:
    if remat:
        act_factor = 4
    else:
        act_factor = max(4, num_shells * 2)

    # 4. Activation bytes & materialized logits cross-entropy projection
    logits_bytes = int(batch_size * seq_len * vocab_size * bytes_per_tok) if vocab_size > 0 else 0
    activation_bytes = int(
        batch_size * total_tokens * surface_dim * bytes_per_tok * act_factor + logits_bytes
    )

    # 5. Distributed collective communication headroom (cross-replica pmean / all-reduce buffers)
    collective_bytes = int(static_bytes) if num_devices > 1 else 0

    peak_bytes = static_bytes + opt_bytes + activation_bytes + collective_bytes
    peak_gb = peak_bytes / (1024**3)

    util_pct = 0.0
    predicted_oom = False
    headroom_gb = 0.0
    is_greedy_safe = False
    if total_memory_bytes and total_memory_bytes > 0:
        util_pct = min(100.0, (peak_bytes / total_memory_bytes) * 100.0)
        predicted_oom = peak_bytes > (total_memory_bytes * safety_margin)
        headroom_bytes = max(0, total_memory_bytes - peak_bytes)
        headroom_gb = headroom_bytes / (1024**3)
        # Optimal greedy saturation envelope: 60% to 90% utilization
        is_greedy_safe = (util_pct >= 60.0) and (not predicted_oom)

    return MemoryBreakdown(
        static_weights_bytes=static_bytes,
        static_weights_gb=static_bytes / (1024**3),
        opt_state_bytes=opt_bytes,
        opt_state_gb=opt_bytes / (1024**3),
        activation_bytes=activation_bytes,
        activation_gb=activation_bytes / (1024**3),
        collective_bytes=collective_bytes,
        collective_gb=collective_bytes / (1024**3),
        peak_hbm_bytes=peak_bytes,
        peak_hbm_gb=peak_gb,
        headroom_gb=headroom_gb,
        hbm_utilization_pct=util_pct,
        predicted_oom=predicted_oom,
        is_greedy_safe=is_greedy_safe,
    )


def estimate_v5_peak_hbm(
    config: Any,
    params: Optional[Dict[str, Any]] = None,
    batch_size: int = 32,
    prompt_len: int = 256,
    seq_len: int = 256,
    remat: bool = False,
    dtype: str = "bfloat16",
    total_memory_bytes: Optional[int] = None,
    num_devices: int = 1,
    worst_case_seq_len: Optional[int] = None,
) -> MemoryBreakdown:
    """Convenience wrapper tailored for CerebroSpheroid V5 configurations."""
    # Determine bytes per token / element
    dt_str = str(dtype).lower()
    bytes_per_tok = 2 if ("16" in dt_str or "bf16" in dt_str or "fp16" in dt_str) else 4

    # Extract model structural hyperparameters
    num_shells = getattr(config, "num_shells", 12)
    surface_dim = getattr(config, "surface_dim", getattr(config, "hidden_dim", 768))
    vocab = getattr(config, "vocab_size", 32000)

    # Calculate parameter bytes
    if params:
        param_bytes = sum(p.size * p.itemsize for p in params.values())
    else:
        est_params = (vocab * surface_dim) + (num_shells * 4 * surface_dim * surface_dim) + (surface_dim * vocab)
        param_bytes = est_params * bytes_per_tok

    target_seq = seq_len
    if worst_case_seq_len is not None and worst_case_seq_len > seq_len:
        target_seq = worst_case_seq_len

    return estimate_peak_hbm(
        param_bytes=param_bytes,
        batch_size=batch_size,
        seq_len=target_seq,
        prompt_len=prompt_len,
        num_shells=num_shells,
        surface_dim=surface_dim,
        remat=remat,
        bytes_per_tok=bytes_per_tok,
        opt_multiplier=2.0,
        total_memory_bytes=total_memory_bytes,
        vocab_size=vocab,
        num_devices=num_devices,
    )
