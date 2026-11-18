"""
Generic Framework Step Adapter.
===============================

Allows arbitrary user-defined models, loss functions, and optimizers
to be auto-tuned using standard callable signatures.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Callable, Dict, Optional

from .base import BaseBenchmarkAdapter
from ..hardware import get_hardware_info
from ..scoring import calculate_loss_reduction_velocity


class GenericStepAdapter(BaseBenchmarkAdapter):
    """
    Adapter accepting a user-supplied step executor function.

    The step executor should accept:
        (batch_size: int, remat: bool, seq_len: int, num_steps: int)
    and return a dict or tuple with:
        {"step_ms": float, "tok_per_sec": float, "initial_loss": float, "final_loss": float, "status": "PASS"|"OOM"}
    """

    def __init__(self, step_fn: Callable[..., Dict[str, Any]]):
        self.step_fn = step_fn

    def benchmark(
        self,
        preset: str,
        batch_size: int,
        remat: bool,
        seq_len: int,
        num_steps: int = 3,
        dtype: str = "auto",
        prompt_len: int = 256,
        **kwargs,
    ) -> Dict[str, Any]:
        hw = get_hardware_info()
        mem_label = hw["memory_type"]

        try:
            t0 = time.time()
            res = self.step_fn(
                preset=preset,
                batch_size=batch_size,
                remat=remat,
                seq_len=seq_len,
                num_steps=num_steps,
                dtype=dtype,
                prompt_len=prompt_len,
                **kwargs,
            )
            step_ms = res.get("step_ms", 100.0)
            steps_per_sec = res.get("steps_per_sec", 1000.0 / max(step_ms, 1e-6))
            tok_s = res.get("tok_per_sec", batch_size * (prompt_len + seq_len) * steps_per_sec)

            init_loss = res.get("initial_loss")
            fin_loss = res.get("final_loss")
            delta_loss = max(0.0, (init_loss - fin_loss) if (init_loss is not None and fin_loss is not None) else 0.0)
            loss_red_sec = calculate_loss_reduction_velocity(
                initial_loss=init_loss,
                final_loss=fin_loss,
                num_steps=num_steps,
                steps_per_sec=steps_per_sec,
                batch_size=batch_size,
                tok_per_sec=tok_s,
            )

            return {
                "preset": preset,
                "batch_size": batch_size,
                "remat": remat,
                "seq_len": seq_len,
                "prompt_len": prompt_len,
                "dtype": str(dtype),
                "compile_time": res.get("compile_time", 1.0),
                "step_ms": step_ms,
                "steps_per_sec": steps_per_sec,
                "tok_per_sec": tok_s,
                "memory_gb": res.get("memory_gb", 0.0),
                "memory_utilization_pct": res.get("memory_utilization_pct", 0.0),
                "memory_type": mem_label,
                "vram_gb": res.get("memory_gb", 0.0),
                "vram_utilization_pct": res.get("memory_utilization_pct", 0.0),
                "initial_loss": init_loss,
                "final_loss": fin_loss,
                "delta_loss": delta_loss,
                "loss_reduction_per_sec": loss_red_sec,
                "status": res.get("status", "PASS"),
            }
        except Exception as e:
            err_msg = str(e)
            is_oom = "OOM" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "out of memory" in err_msg.lower()
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
                "status": "OOM (Memory Exceeded)" if is_oom else f"FAIL ({type(e).__name__}: {err_msg[:40]})",
            }
        finally:
            gc.collect()
