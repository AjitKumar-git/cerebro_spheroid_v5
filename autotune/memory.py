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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "static_weights_bytes": self.static_weights_bytes,
            "static_weights_gb": self.static_weights_gb,
            "opt_state_bytes": self.opt_state_bytes,
            "opt_state_gb": self.opt_state_gb,
            "activation_bytes": self.activation_bytes,
            "activation_gb": self.activation_gb,
            "peak_hbm_bytes": self.peak_hbm_bytes,
            "peak_hbm_gb": self.peak_hbm_gb,
            "hbm_utilization_pct": self.hbm_utilization_pct,
            "predicted_oom": self.predicted_oom,
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
    opt_multiplier: float = 2.0,
    total_memory_bytes: Optional[int] = None,
    safety_margin: float = 0.95,
) -> MemoryBreakdown:
    """
    Computes exact theoretical peak accelerator memory based on tensor shapes:

    1. Static Weights:
       W_bytes = sum(size * itemsize)

    2. Optimizer State:
       O_bytes = W_bytes * opt_multiplier (typically 2x for AdamW/Spheroid first & second moments)

    3. Activation Footprint:
       - remat=False: all shell intermediate activations held simultaneously:
         A_bytes = Batch × (PromptLen + SeqLen) × SurfaceDim × BytesPerTok × (NumShells × 2)
       - remat=True: only shell boundary checkpoints stored; intermediate layers recomputed:
         A_bytes = Batch × (PromptLen + SeqLen) × SurfaceDim × BytesPerTok × 4
    """
    # 1. Static weights and optimizer state
    static_bytes = int(param_bytes)
    opt_bytes = int(static_bytes * opt_multiplier)

    # 2. Total sequence tokens per batch element
    total_tokens = max(1, prompt_len + seq_len)

    # 3. Activation multiplier:
    # If remat=True, activations inside shells are recomputed during backward pass,
    # so peak activation memory is bounded by boundary checkpoints (~4x surface slice).
    # If remat=False, activations across all shells are preserved in memory (~2 * num_shells).
    if remat:
        act_factor = 4
    else:
        act_factor = max(4, num_shells * 2)

    activation_bytes = int(
        batch_size * total_tokens * surface_dim * bytes_per_tok * act_factor
    )

    peak_bytes = static_bytes + opt_bytes + activation_bytes
    peak_gb = peak_bytes / (1024**3)

    util_pct = 0.0
    predicted_oom = False
    if total_memory_bytes and total_memory_bytes > 0:
        util_pct = min(100.0, (peak_bytes / total_memory_bytes) * 100.0)
        predicted_oom = peak_bytes > (total_memory_bytes * safety_margin)

    return MemoryBreakdown(
        static_weights_bytes=static_bytes,
        static_weights_gb=static_bytes / (1024**3),
        opt_state_bytes=opt_bytes,
        opt_state_gb=opt_bytes / (1024**3),
        activation_bytes=activation_bytes,
        activation_gb=activation_bytes / (1024**3),
        peak_hbm_bytes=peak_bytes,
        peak_hbm_gb=peak_gb,
        hbm_utilization_pct=util_pct,
        predicted_oom=predicted_oom,
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
) -> MemoryBreakdown:
    """Convenience wrapper tailored for CerebroSpheroid V5 configurations."""
    # Determine bytes per token / element
    dt_str = str(dtype).lower()
    bytes_per_tok = 2 if ("16" in dt_str or "bf16" in dt_str or "fp16" in dt_str) else 4

    # Extract model structural hyperparameters
    num_shells = getattr(config, "num_shells", 12)
    surface_dim = getattr(config, "surface_dim", getattr(config, "hidden_dim", 768))

    # Calculate parameter bytes
    if params:
        param_bytes = sum(p.size * p.itemsize for p in params.values())
    else:
        # Theoretical parameter estimation from config if params not instantiated
        vocab = getattr(config, "vocab_size", 32000)
        # Embedding: vocab * surface_dim
        # Shells: num_shells * (~4 * surface_dim^2)
        # Heads / Motor: ~2 * surface_dim * vocab
        est_params = (vocab * surface_dim) + (num_shells * 4 * surface_dim * surface_dim) + (surface_dim * vocab)
        param_bytes = est_params * bytes_per_tok

    return estimate_peak_hbm(
        param_bytes=param_bytes,
        batch_size=batch_size,
        seq_len=seq_len,
        prompt_len=prompt_len,
        num_shells=num_shells,
        surface_dim=surface_dim,
        remat=remat,
        bytes_per_tok=bytes_per_tok,
        opt_multiplier=2.0,
        total_memory_bytes=total_memory_bytes,
    )
