"""
Optimization Objectives for Accelerator Auto-Tuning.
"""

from __future__ import annotations

from enum import Enum
from typing import Union


class TuneObjective(str, Enum):
    """Supported optimization objectives."""
    TOKENS_PER_SEC = "tok_per_sec"
    STEPS_PER_SEC = "steps_per_sec"
    LOSS_REDUCTION_PER_SEC = "loss_reduction_per_sec"
    BALANCED = "balanced"

    @classmethod
    def from_str(cls, value: Union[str, TuneObjective, None]) -> TuneObjective:
        """Normalizes user strings and aliases to a valid TuneObjective enum."""
        if isinstance(value, TuneObjective):
            return value
        if not value:
            return cls.BALANCED

        v = str(value).strip().lower().replace("-", "_").replace(" ", "_")

        # Tokens / sec aliases
        if v in ("tok_per_sec", "tokens_per_sec", "tok_per_s", "tok/s", "tokens/sec", "tokens/s", "throughput"):
            return cls.TOKENS_PER_SEC

        # Steps / sec aliases
        if v in ("steps_per_sec", "step_rate", "steps/s", "step/s", "steps/sec", "latency"):
            return cls.STEPS_PER_SEC

        # Loss reduction / sec aliases
        if v in (
            "loss_reduction_per_sec",
            "loss_reduction_sec",
            "loss_reduction/sec",
            "loss_reduction/s",
            "loss_reduction_s",
            "dloss_dt",
            "loss_reduction",
            "loss_speed",
            "loss_per_sec",
        ):
            return cls.LOSS_REDUCTION_PER_SEC

        # Balanced / Pareto efficiency
        if v in ("balanced", "pareto", "default"):
            return cls.BALANCED

        return cls.BALANCED

    @property
    def label(self) -> str:
        if self == TuneObjective.TOKENS_PER_SEC:
            return "tokens/sec"
        elif self == TuneObjective.STEPS_PER_SEC:
            return "steps/s"
        elif self == TuneObjective.LOSS_REDUCTION_PER_SEC:
            return "loss reduction/sec"
        return "balanced (Pareto tok/s * steps/s^0.25)"
