"""
Hardware Profiling & Accelerator Specification Engine.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

_BF16_GPU_MARKERS = (
    "a100", "h100", "h200", "b100", "b200", "l4", "l40",
    "a10", "a30", "a40", "a6000", "rtx 30", "rtx 40", "rtx 50",
)


@dataclass
class HardwareInfo:
    device_kind: str
    platform: str
    is_tpu: bool
    is_gpu: bool
    memory_type: str
    total_memory_bytes: int
    total_gb: float
    hardware_signature: str
    device_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_kind": self.device_kind,
            "platform": self.platform,
            "is_tpu": self.is_tpu,
            "is_gpu": self.is_gpu,
            "memory_type": self.memory_type,
            "total_memory_bytes": self.total_memory_bytes,
            "total_gb": self.total_gb,
            "hardware_signature": self.hardware_signature,
            "device_count": self.device_count,
        }


def get_hardware_info() -> Dict[str, Any]:
    """
    Profiles active accelerator devices and accurately determines memory capacity
    and architectural characteristics across TPU, GPU, and CPU.
    """
    devices = []
    dev = None
    dev_kind = "cpu"
    platform = "cpu"

    # 1. Try JAX device detection
    try:
        import jax
        devices = jax.devices()
        dev = devices[0] if devices else None
        if dev:
            dev_kind = getattr(dev, "device_kind", "cpu")
            platform = getattr(dev, "platform", "cpu").lower()
    except Exception:
        pass

    # 2. Try PyTorch if JAX is unavailable or detected CPU
    if platform == "cpu" or not devices:
        try:
            import torch
            if torch.cuda.is_available():
                platform = "gpu"
                dev_kind = torch.cuda.get_device_name(0)
                devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        except Exception:
            pass

    is_tpu = "tpu" in platform or "tpu" in dev_kind.lower()
    is_gpu = "gpu" in platform or "cuda" in platform or "rocm" in platform or "gpu" in dev_kind.lower() or "nvidia" in dev_kind.lower()

    total_memory_bytes: Optional[int] = None

    # Method 1: JAX device memory_stats API
    if dev and hasattr(dev, "memory_stats"):
        try:
            stats = dev.memory_stats()
            if stats and "bytes_limit" in stats and stats["bytes_limit"] > 0:
                total_memory_bytes = stats["bytes_limit"]
        except Exception:
            pass

    # Method 2: TPU known architectural specifications
    if is_tpu and not total_memory_bytes:
        k = dev_kind.lower()
        if "v5p" in k:
            total_memory_bytes = 95 * (1024**3)
        elif "v6e" in k or "v6" in k or "trillium" in k:
            total_memory_bytes = 32 * (1024**3)
        elif "v4" in k:
            total_memory_bytes = 32 * (1024**3)
        elif "v5e" in k or "v5 lite" in k or "v5" in k or "v3" in k:
            total_memory_bytes = 16 * (1024**3)
        elif "v2" in k:
            total_memory_bytes = 8 * (1024**3)
        else:
            total_memory_bytes = 16 * (1024**3)

    # Method 3: PyTorch CUDA probe for GPUs
    if is_gpu and not total_memory_bytes:
        try:
            import torch
            if torch.cuda.is_available():
                total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        except Exception:
            pass

    # Fallback conservative defaults
    if not total_memory_bytes:
        if is_tpu:
            total_memory_bytes = 16 * (1024**3)
        elif is_gpu:
            k = dev_kind.lower()
            if "h200" in k or "b200" in k or "b100" in k:
                total_memory_bytes = 141 * (1024**3)
            elif "h100" in k or "a100" in k:
                total_memory_bytes = 80 * (1024**3)
            elif "l40" in k or "a6000" in k or "rtx 6000" in k:
                total_memory_bytes = 48 * (1024**3)
            elif "l4" in k or "rtx 4090" in k or "rtx 3090" in k or "a5000" in k:
                total_memory_bytes = 24 * (1024**3)
            elif "t4" in k or "v100" in k:
                total_memory_bytes = 16 * (1024**3)
            elif "3070" in k or "4070" in k or "2080" in k:
                total_memory_bytes = 8 * (1024**3)
            else:
                total_memory_bytes = 16 * (1024**3)
        else:
            total_memory_bytes = 8 * (1024**3)

    total_gb = total_memory_bytes / (1024**3)
    clean_kind = dev_kind.replace(" ", "_").replace("/", "_").replace("-", "_")
    sig = f"{clean_kind}_{int(round(total_gb))}GB"
    mem_type = "HBM" if is_tpu else ("VRAM" if is_gpu else "RAM")

    return {
        "device_kind": dev_kind,
        "platform": platform,
        "is_tpu": is_tpu,
        "is_gpu": is_gpu,
        "memory_type": mem_type,
        "total_memory_bytes": total_memory_bytes,
        "total_gb": total_gb,
        "hardware_signature": sig,
        "device_count": max(len(devices), 1),
    }


def resolve_jax_compute_dtype(dtype: str = "auto") -> Tuple[Any, str]:
    """
    Resolves optimal compute precision matching TPU MXU vs Ampere/Hopper Tensor Cores vs T4/CPU.
    """
    try:
        import jax.numpy as jnp
        bfloat16_t = jnp.bfloat16
        float16_t = jnp.float16
        float32_t = jnp.float32
    except ImportError:
        # Fallback if JAX is not installed
        bfloat16_t = "bfloat16"
        float16_t = "float16"
        float32_t = "float32"

    hw = get_hardware_info()
    is_tpu = hw["is_tpu"]
    is_gpu = hw["is_gpu"]
    dev_kind = hw.get("device_kind", "Accelerator")
    dev_lower = dev_kind.lower()

    if dtype == "auto":
        if is_tpu:
            return bfloat16_t, f"bfloat16 (Native {dev_kind} MXU)"
        if is_gpu and any(tok in dev_lower for tok in _BF16_GPU_MARKERS):
            return bfloat16_t, f"bfloat16 (Native {dev_kind} Tensor Core)"
        return float32_t, f"float32 ({dev_kind if is_gpu else 'CPU'})"

    if dtype in ("bf16", "bfloat16"):
        return bfloat16_t, f"bfloat16 ({dev_kind})"
    if dtype in ("fp16", "float16"):
        return float16_t, f"float16 ({dev_kind})"
    return float32_t, f"float32 ({dev_kind})"
