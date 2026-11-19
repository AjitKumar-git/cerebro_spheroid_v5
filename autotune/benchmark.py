"""
Benchmark Execution & Accelerator Profiling Runner.
===================================================

Safely executes micro-benchmarks with AOT compilation isolation,
accurate timing, live memory tracking, and graceful OOM trapping.
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .hardware import get_hardware_info, resolve_jax_compute_dtype
from .scoring import calculate_loss_reduction_velocity


@dataclass
class BenchmarkResult:
    preset: str
    batch_size: int
    remat: bool
    seq_len: int
    prompt_len: int
    dtype: str
    compile_time: Optional[float]
    step_ms: Optional[float]
    steps_per_sec: float
    tok_per_sec: float
    memory_gb: float
    memory_utilization_pct: float
    memory_type: str
    vram_gb: float
    vram_utilization_pct: float
    initial_loss: Optional[float]
    final_loss: Optional[float] = None
    delta_loss: float = 0.0
    loss_reduction_per_sec: float = 0.0
    status: str = "PASS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "batch_size": self.batch_size,
            "remat": self.remat,
            "seq_len": self.seq_len,
            "prompt_len": self.prompt_len,
            "dtype": self.dtype,
            "compile_time": self.compile_time,
            "step_ms": self.step_ms,
            "steps_per_sec": self.steps_per_sec,
            "tok_per_sec": self.tok_per_sec,
            "memory_gb": self.memory_gb,
            "memory_utilization_pct": self.memory_utilization_pct,
            "memory_type": self.memory_type,
            "vram_gb": self.vram_gb,
            "vram_utilization_pct": self.vram_utilization_pct,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "delta_loss": self.delta_loss,
            "loss_reduction_per_sec": self.loss_reduction_per_sec,
            "status": self.status,
        }


class BenchmarkRunner:
    """Orchestrates benchmark runs using a pluggable execution adapter."""

    def __init__(self, step_executor: Optional[Callable[..., Dict[str, Any]]] = None):
        self.step_executor = step_executor

    def run(
        self,
        preset: str = "v5_small",
        batch_size: int = 32,
        remat: bool = True,
        seq_len: int = 512,
        num_steps: int = 3,
        dtype: str = "auto",
        prompt_len: int = 256,
        **kwargs,
    ) -> Dict[str, Any]:
        if self.step_executor is not None:
            import inspect
            sig = inspect.signature(self.step_executor)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            call_kwargs = {
                "preset": preset,
                "batch_size": batch_size,
                "remat": remat,
                "seq_len": seq_len,
                "num_steps": num_steps,
                "dtype": dtype,
                "prompt_len": prompt_len,
            }
            if has_var_kw:
                call_kwargs.update(kwargs)
            else:
                for k, v in kwargs.items():
                    if k in sig.parameters:
                        call_kwargs[k] = v
            return self.step_executor(**call_kwargs)

        # Default fallback to the registered adapter
        from .adapters.v5 import benchmark_v5_model
        return benchmark_v5_model(
            preset=preset,
            batch_size=batch_size,
            remat=remat,
            seq_len=seq_len,
            num_steps=num_steps,
            dtype=dtype,
            prompt_len=prompt_len,
            **kwargs,
        )
