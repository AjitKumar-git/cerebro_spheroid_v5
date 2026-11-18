"""
Base Benchmark Adapter Interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseBenchmarkAdapter(ABC):
    """Abstract interface for model/framework benchmark execution."""

    @abstractmethod
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
        """Runs the micro-benchmark and returns structured benchmark metrics."""
        pass
