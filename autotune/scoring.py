"""
Multi-Objective Pareto Scoring Engine.
=====================================

Evaluates benchmarked configurations against target training criteria:
- tokens/sec (throughput maximization)
- steps/s (parameter update frequency)
- loss reduction/sec (rate of convergence per second, dLoss/dt)
- balanced (Pareto efficiency)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional
from .objectives import TuneObjective


def calculate_loss_reduction_velocity(
    initial_loss: Optional[float],
    final_loss: Optional[float],
    num_steps: int,
    steps_per_sec: float,
    batch_size: int = 32,
    tok_per_sec: float = 0.0,
) -> float:
    """
    Computes empirical and estimated rate of loss descent per wall-clock second:
        dLoss/dt = (delta_loss / step) * (steps / second)

    When loss delta is observed during micro-benchmarking steps, uses the measured
    velocity. If loss change is unmeasurable (e.g. synthetic data with 0 lr or frozen weights),
    falls back to the theoretical scaling law of loss descent velocity with critical batch size.
    """
    if steps_per_sec <= 0.0:
        return 0.0

    measured_velocity = 0.0
    if initial_loss is not None and final_loss is not None:
        delta_loss = max(0.0, float(initial_loss) - float(final_loss))
        effective_steps = max(1, num_steps - 1) if num_steps > 1 else 1
        delta_per_step = delta_loss / effective_steps
        measured_velocity = delta_per_step * steps_per_sec

    if measured_velocity > 1e-6:
        return measured_velocity

    # Theoretical convergence velocity under noise-scale law:
    # Convergence rate per second is proportional to tok_per_sec / (1 + B / B_crit)
    # where B_crit is the critical batch size (typical ~128 to 256 for small/medium models)
    b_crit = 128.0
    efficiency_factor = 1.0 / (1.0 + (batch_size / b_crit))
    theoretical_velocity = (tok_per_sec * 1e-4) * efficiency_factor * (steps_per_sec ** 0.25)
    return max(0.0, theoretical_velocity)


def score_benchmark_result(
    res: Dict[str, Any],
    tune_objective: str = "balanced",
    greedy: bool = True,
) -> float:
    """
    Scores a passing benchmark result based on the requested optimization objective.
    When `greedy=True`, applies a hardware saturation bonus for safely utilizing
    60% to 92% of accelerator HBM/VRAM and prioritizing remat=False when memory permits.

    Objectives:
    - "tok_per_sec" / "tokens/sec": Raw throughput maximization.
    - "steps_per_sec" / "steps/s": Highest optimizer updates per second (lowest latency).
    - "loss_reduction_per_sec" / "loss reduction/sec": Convergence velocity (dLoss/dt).
    - "balanced": Pareto efficiency balancing token throughput and optimizer updates.
    """
    tok_s = max(0.0, float(res.get("tok_per_sec", 0.0)))
    steps_s = max(0.0, float(res.get("steps_per_sec", 0.0)))
    if steps_s <= 0.0 and res.get("step_ms"):
        steps_s = 1000.0 / max(float(res["step_ms"]), 1e-6)

    if tok_s <= 0.0 or steps_s <= 0.0:
        return 0.0

    obj = TuneObjective.from_str(tune_objective)

    if obj == TuneObjective.TOKENS_PER_SEC:
        base_score = tok_s
    elif obj == TuneObjective.STEPS_PER_SEC:
        base_score = (tok_s ** 0.25) * steps_s
    elif obj == TuneObjective.LOSS_REDUCTION_PER_SEC:
        init_loss = res.get("initial_loss")
        fin_loss = res.get("final_loss")
        num_steps = res.get("num_steps", 4)
        b = res.get("batch_size", 32)
        base_score = calculate_loss_reduction_velocity(
            initial_loss=init_loss,
            final_loss=fin_loss,
            num_steps=num_steps,
            steps_per_sec=steps_s,
            batch_size=b,
            tok_per_sec=tok_s,
        )
    else:
        # "balanced": Pareto efficiency balancing token throughput and optimizer updates/sec
        base_score = tok_s * (steps_s ** 0.25)

    if not greedy:
        return base_score

    # Greedy accelerator saturation bonus:
    # 1. Incentivize utilizing 65% to 92% of hardware capacity
    mem_pct = float(res.get("memory_utilization_pct", res.get("vram_utilization_pct", 0.0)))
    sat_mult = 1.0
    if 65.0 <= mem_pct <= 92.0:
        sat_mult += 0.08 * ((mem_pct - 65.0) / 27.0)
    elif mem_pct > 95.0:
        sat_mult *= 0.85  # Penalize precarious boundary allocations exceeding 95%

    # 2. Prefer remat=False when memory permits (avoids ~30% forward recomputation overhead)
    remat_bonus = 1.05 if not res.get("remat", False) else 1.0

    return base_score * sat_mult * remat_bonus


# Backward-compatible alias
_score_benchmark_result = score_benchmark_result
