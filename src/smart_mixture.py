# -*- coding: utf-8 -*-
"""
🧠 CerebroSpheroid V5: Smart Dynamic Dataset Mixture Engine
===========================================================
Implements DoReMi & Generalization-Regularized Dynamic Mixture Optimization:
  • Multi-Domain Loss & Velocity Tracking : Tracks held-out validation loss and
    gradient velocity (-ΔLoss / ΔTokens) per domain (broad_web, code, math, cosmopedia, edu_academic).
  • Overfit & Memorization Detector       : Identifies domains whose validation loss
    begins to regress or exceed best recorded loss, penalizing their allocation to prevent memorization.
  • Saturation / Diminishing Returns      : Flags domains where loss velocity drops near zero (|v| < ε),
    diverting precious TPU/GPU compute to high-yield reasoning domains.
  • Exponentiated Gradient (Hedge)        : Multiplicative weight updates with Dirichlet prior
    smoothing (toward developmental baseline) and hard safety clamps [α_min, α_max].
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DomainMetrics:
    name: str
    target_share: float
    current_val_loss: float
    best_val_loss: float
    loss_delta: float
    raw_velocity: float
    smoothed_velocity: float
    total_tokens: int
    delta_tokens: int
    overfit_streak: int
    saturation_streak: int
    is_overfitting: bool
    is_saturated: bool
    overfit_penalty: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def map_domain_label_to_tag(name: str) -> str:
    """Normalizes an arbitrary domain display name or label to its canonical curriculum tag."""
    lower = str(name).lower()
    if "edu" in lower or "academic" in lower:
        return "edu_academic"
    if "cosmo" in lower:
        return "cosmopedia"
    if "math" in lower:
        return "math"
    if "code" in lower or "stack" in lower or "python" in lower:
        return "code"
    if "web" in lower:
        return "broad_web"
    if "retr" in lower:
        return "retrieval"
    return lower.strip().replace(" ", "_")


class SmartMixtureController:
    """
    Dynamic mixture controller for continuous pretraining and reasoning curricula.
    Balances compute allocation dynamically across domains to maximize overall loss reduction
    without overfitting.
    """

    DEFAULT_MIN_SHARES: Dict[str, float] = {
        "broad_web": 0.10,
        "code": 0.05,
        "math": 0.05,
        "cosmopedia": 0.05,
        "edu_academic": 0.05,
        "retrieval": 0.02,
    }

    DEFAULT_MAX_SHARES: Dict[str, float] = {
        "broad_web": 0.75,
        "code": 0.35,
        "math": 0.45,
        "cosmopedia": 0.45,
        "edu_academic": 0.40,
        "retrieval": 0.20,
    }

    def __init__(
        self,
        initial_shares: Dict[str, float],
        min_shares: Optional[Dict[str, float]] = None,
        max_shares: Optional[Dict[str, float]] = None,
        eta: float = 0.08,                     # Exponentiated gradient step size
        prior_weight: float = 0.20,            # Dirichlet prior blend factor (preserves curriculum intent)
        overfit_threshold: float = 0.03,       # Loss rise above best_loss before overfit penalty triggers
        overfit_decay: float = 0.60,           # Penalty multiplier applied per overfit streak count
        saturation_velocity: float = 1e-9,     # Velocity threshold below which domain is considered saturated
        velocity_ema_beta: float = 0.70,       # Exponential moving average smoothing for velocity
        history_window: int = 10,
    ):
        self.eta = float(eta)
        self.prior_weight = max(0.0, min(1.0, float(prior_weight)))
        self.overfit_threshold = float(overfit_threshold)
        self.overfit_decay = float(overfit_decay)
        self.saturation_velocity = float(saturation_velocity)
        self.velocity_ema_beta = float(velocity_ema_beta)
        self.history_window = int(history_window)

        # Baseline prior (Dirichlet anchor)
        norm_init = self._normalize_shares(initial_shares)
        self.prior_shares: Dict[str, float] = copy.deepcopy(norm_init)
        self.current_shares: Dict[str, float] = copy.deepcopy(norm_init)

        # Configure domain boundary clamps
        self.min_shares: Dict[str, float] = {}
        self.max_shares: Dict[str, float] = {}
        for d in self.current_shares.keys():
            self.min_shares[d] = float(min_shares.get(d, self.DEFAULT_MIN_SHARES.get(d, 0.02))) if min_shares else self.DEFAULT_MIN_SHARES.get(d, 0.02)
            self.max_shares[d] = float(max_shares.get(d, self.DEFAULT_MAX_SHARES.get(d, 0.80))) if max_shares else self.DEFAULT_MAX_SHARES.get(d, 0.80)
            if self.min_shares[d] > self.max_shares[d]:
                self.min_shares[d] = self.max_shares[d] * 0.5

        # Domain histories
        self.val_loss_history: Dict[str, List[float]] = {d: [] for d in self.current_shares}
        self.step_history: Dict[str, List[int]] = {d: [] for d in self.current_shares}
        self.token_history: Dict[str, List[int]] = {d: [] for d in self.current_shares}
        self.best_val_loss: Dict[str, float] = {d: float("inf") for d in self.current_shares}
        self.smoothed_velocity: Dict[str, float] = {d: 0.0 for d in self.current_shares}
        self.overfit_streak: Dict[str, int] = {d: 0 for d in self.current_shares}
        self.saturation_streak: Dict[str, int] = {d: 0 for d in self.current_shares}

        # Ensure initial shares satisfy bounded simplex
        self.current_shares = self.project_to_bounded_simplex(self.current_shares)

    @staticmethod
    def _normalize_shares(shares: Dict[str, float]) -> Dict[str, float]:
        tot = sum(max(0.0, float(v)) for v in shares.values())
        if tot <= 0.0:
            n = len(shares)
            return {k: 1.0 / n for k in shares}
        return {k: max(0.0, float(v)) / tot for k, v in shares.items()}

    def project_to_bounded_simplex(self, shares: Dict[str, float]) -> Dict[str, float]:
        """
        Projects an arbitrary positive vector onto the probability simplex sum(x)=1.0
        subject to box constraints min_shares[d] <= x[d] <= max_shares[d].
        Uses iterative clip-and-rescale convergence.
        """
        keys = list(shares.keys())
        mins = {k: self.min_shares.get(k, 0.02) for k in keys}
        maxs = {k: self.max_shares.get(k, 0.80) for k in keys}

        # Check feasibility
        sum_min = sum(mins.values())
        sum_max = sum(maxs.values())
        if sum_min > 1.0:
            # Scaled down mins
            mins = {k: v / sum_min for k, v in mins.items()}
        if sum_max < 1.0:
            # Scaled up maxs
            maxs = {k: v / sum_max for k, v in maxs.items()}

        x = {k: max(mins[k], min(maxs[k], float(shares.get(k, 1.0 / len(keys))))) for k in keys}

        # Iterative redistribution
        for _ in range(50):
            cur_sum = sum(x.values())
            deficit = 1.0 - cur_sum
            if abs(deficit) < 1e-6:
                break

            if deficit > 0:
                # Need to increase: find components not at max
                free_keys = [k for k in keys if x[k] < maxs[k] - 1e-7]
                if not free_keys:
                    break
                free_weight = sum(x[k] for k in free_keys)
                if free_weight <= 0:
                    delta = deficit / len(free_keys)
                    for k in free_keys:
                        x[k] = min(maxs[k], x[k] + delta)
                else:
                    for k in free_keys:
                        added = deficit * (x[k] / free_weight)
                        x[k] = min(maxs[k], x[k] + added)
            else:
                # Need to decrease: find components not at min
                surplus = -deficit
                free_keys = [k for k in keys if x[k] > mins[k] + 1e-7]
                if not free_keys:
                    break
                free_weight = sum(x[k] for k in free_keys)
                if free_weight <= 0:
                    delta = surplus / len(free_keys)
                    for k in free_keys:
                        x[k] = max(mins[k], x[k] - delta)
                else:
                    for k in free_keys:
                        subbed = surplus * (x[k] / free_weight)
                        x[k] = max(mins[k], x[k] - subbed)

        final_sum = sum(x.values())
        return {k: round(v / final_sum, 6) for k, v in x.items()}

    def step(
        self,
        step: int,
        domain_val_losses: Dict[str, float],
        domain_tokens: Optional[Dict[str, int]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, DomainMetrics]]:
        """
        Updates dynamic mixture weights based on new validation losses and token counts.

        Returns:
          new_shares: Dict[str, float] mapped to each active domain.
          metrics: Dict[str, DomainMetrics] with comprehensive per-domain diagnostics.
        """
        metrics: Dict[str, DomainMetrics] = {}
        utilities: Dict[str, float] = {}

        # Register any new domains dynamically
        for d in domain_val_losses:
            if d not in self.current_shares:
                self.current_shares[d] = self.DEFAULT_MIN_SHARES.get(d, 0.05)
                self.prior_shares[d] = self.DEFAULT_MIN_SHARES.get(d, 0.05)
                self.min_shares[d] = self.DEFAULT_MIN_SHARES.get(d, 0.02)
                self.max_shares[d] = self.DEFAULT_MAX_SHARES.get(d, 0.80)
                self.val_loss_history[d] = []
                self.step_history[d] = []
                self.token_history[d] = []
                self.best_val_loss[d] = float("inf")
                self.smoothed_velocity[d] = 0.0
                self.overfit_streak[d] = 0
                self.saturation_streak[d] = 0

        # 1. Compute velocity, overfit penalty, and utility per domain
        for d, loss in domain_val_losses.items():
            if not math.isfinite(loss):
                continue

            loss = float(loss)
            tok_count = int(domain_tokens.get(d, 0)) if domain_tokens else 0
            prev_loss = self.val_loss_history[d][-1] if self.val_loss_history[d] else loss
            prev_toks = self.token_history[d][-1] if self.token_history[d] else tok_count

            delta_loss = loss - prev_loss
            delta_toks = max(1000, tok_count - prev_toks)

            # Update best recorded validation loss
            if loss < self.best_val_loss[d]:
                self.best_val_loss[d] = loss

            # Detect overfitting: loss is higher than best recorded by more than overfit_threshold
            overfit_margin = loss - self.best_val_loss[d]
            is_overfitting = (overfit_margin > self.overfit_threshold) or (delta_loss > 0.05)
            if is_overfitting:
                self.overfit_streak[d] += 1
            else:
                self.overfit_streak[d] = max(0, self.overfit_streak[d] - 1)

            # Compute learning velocity: -ΔLoss / ΔTokens (scaled to human order of magnitude)
            # Positive velocity = learning rapidly (loss dropping).
            # Negative velocity = regressing.
            raw_velocity = -delta_loss / float(delta_toks) * 1e8
            prev_smoothed = self.smoothed_velocity[d]
            smoothed_v = (
                self.velocity_ema_beta * prev_smoothed + (1.0 - self.velocity_ema_beta) * raw_velocity
                if self.val_loss_history[d]
                else raw_velocity
            )
            self.smoothed_velocity[d] = smoothed_v

            # Detect saturation: smoothed velocity is near zero
            is_saturated = (abs(smoothed_v) < self.saturation_velocity * 1e8) and (len(self.val_loss_history[d]) >= 3)
            if is_saturated:
                self.saturation_streak[d] += 1
            else:
                self.saturation_streak[d] = max(0, self.saturation_streak[d] - 1)

            # Calculate overfit penalty factor (decay exponentially with streak)
            penalty_factor = (self.overfit_decay ** self.overfit_streak[d]) if self.overfit_streak[d] > 0 else 1.0

            # Utility formulation:
            # Reward high velocity, penalize overfitting and saturation
            utility = smoothed_v
            if self.overfit_streak[d] > 0:
                utility -= (2.5 * self.overfit_streak[d] + overfit_margin * 5.0)
            if is_saturated:
                utility -= 0.50

            utilities[d] = float(utility)

            # Append to history
            self.val_loss_history[d].append(loss)
            self.step_history[d].append(step)
            self.token_history[d].append(tok_count)
            if len(self.val_loss_history[d]) > self.history_window:
                self.val_loss_history[d].pop(0)
                self.step_history[d].pop(0)
                self.token_history[d].pop(0)

            metrics[d] = DomainMetrics(
                name=d,
                target_share=self.current_shares.get(d, 0.0),
                current_val_loss=loss,
                best_val_loss=self.best_val_loss[d],
                loss_delta=delta_loss,
                raw_velocity=raw_velocity,
                smoothed_velocity=smoothed_v,
                total_tokens=tok_count,
                delta_tokens=delta_toks,
                overfit_streak=self.overfit_streak[d],
                saturation_streak=self.saturation_streak[d],
                is_overfitting=is_overfitting,
                is_saturated=is_saturated,
                overfit_penalty=penalty_factor,
            )

        if not utilities:
            return self.current_shares, metrics

        # 2. Exponentiated Gradient Update
        # Center utilities around mean to prevent numerical overflow/underflow
        u_vals = list(utilities.values())
        mean_u = sum(u_vals) / len(u_vals)
        std_u = math.sqrt(sum((x - mean_u) ** 2 for x in u_vals) / max(1, len(u_vals))) + 1e-4

        cand_shares: Dict[str, float] = {}
        for d in self.current_shares:
            u_norm = (utilities.get(d, 0.0) - mean_u) / std_u
            # Exponentiated step
            mult = math.exp(self.eta * max(-3.0, min(3.0, u_norm)))
            # If domain has severe overfit streak, apply direct dampening
            if d in metrics and metrics[d].overfit_streak > 0:
                mult *= metrics[d].overfit_penalty
            cand_shares[d] = self.current_shares[d] * mult

        # 3. Blend with Dirichlet prior baseline
        norm_cand = self._normalize_shares(cand_shares)
        blended_shares: Dict[str, float] = {}
        for d in self.current_shares:
            prior_s = self.prior_shares.get(d, 1.0 / len(self.current_shares))
            blended_shares[d] = (1.0 - self.prior_weight) * norm_cand[d] + self.prior_weight * prior_s

        # 4. Project onto Bounded Simplex [min_shares, max_shares]
        self.current_shares = self.project_to_bounded_simplex(blended_shares)

        # Update target_share in returned metrics
        for d, m in metrics.items():
            m.target_share = self.current_shares.get(d, m.target_share)

        return self.current_shares, metrics

    def format_status_report(self, metrics: Dict[str, DomainMetrics]) -> str:
        """Returns a formatted terminal status table of current mixture and domain states."""
        lines = [
            "⚖️ [Smart Mixture Engine] Dynamic Multi-Domain Loss & Velocity Breakdown:",
            "   ┌────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────────┐",
            "   │ Domain         │ Share %  │ Val Loss │ Best L   │ Velocity │ Overfit  │ Saturation   │",
            "   ├────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤",
        ]
        for d, m in sorted(metrics.items(), key=lambda x: -x[1].target_share):
            overfit_str = f"⚠️ STREAK {m.overfit_streak}" if m.overfit_streak > 0 else "✓ Clean"
            sat_str = f"⏸️ Saturated" if m.is_saturated else "⚡ Active"
            lines.append(
                f"   │ {d:<14} │ {m.target_share*100:>7.1f}% │ {m.current_val_loss:>8.4f} │ {m.best_val_loss:>8.4f} │ "
                f"{m.smoothed_velocity:>+8.2f} │ {overfit_str:<8} │ {sat_str:<12} │"
            )
        lines.append("   └────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────────┘")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes controller state for checkpointing and telemetry sync."""
        return {
            "current_shares": self.current_shares,
            "prior_shares": self.prior_shares,
            "min_shares": self.min_shares,
            "max_shares": self.max_shares,
            "best_val_loss": self.best_val_loss,
            "smoothed_velocity": self.smoothed_velocity,
            "overfit_streak": self.overfit_streak,
            "saturation_streak": self.saturation_streak,
            "val_loss_history": self.val_loss_history,
            "step_history": self.step_history,
            "token_history": self.token_history,
            "eta": self.eta,
            "prior_weight": self.prior_weight,
            "overfit_threshold": self.overfit_threshold,
        }

    def load_dict(self, state: Dict[str, Any]) -> None:
        """Restores controller state from serialized dictionary."""
        if not isinstance(state, dict):
            return
        if "current_shares" in state:
            self.current_shares = {k: float(v) for k, v in state["current_shares"].items()}
        if "prior_shares" in state:
            self.prior_shares = {k: float(v) for k, v in state["prior_shares"].items()}
        if "best_val_loss" in state:
            self.best_val_loss = {k: float(v) for k, v in state["best_val_loss"].items()}
        if "smoothed_velocity" in state:
            self.smoothed_velocity = {k: float(v) for k, v in state["smoothed_velocity"].items()}
        if "overfit_streak" in state:
            self.overfit_streak = {k: int(v) for k, v in state["overfit_streak"].items()}
        if "saturation_streak" in state:
            self.saturation_streak = {k: int(v) for k, v in state["saturation_streak"].items()}
        if "val_loss_history" in state:
            self.val_loss_history = {k: list(v) for k, v in state["val_loss_history"].items()}
        if "step_history" in state:
            self.step_history = {k: list(v) for k, v in state["step_history"].items()}
        if "token_history" in state:
            self.token_history = {k: list(v) for k, v in state["token_history"].items()}
        self.current_shares = self.project_to_bounded_simplex(self.current_shares)
