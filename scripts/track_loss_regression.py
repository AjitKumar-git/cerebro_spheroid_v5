#!/usr/bin/env python3
"""
CerebroSpheroid V5: Live Loss Trajectory & Regression Monitoring Engine
=====================================================================
Tracks training progress, computes objective held-out validation loss,
detects sudden loss spikes or persistent regressions, and renders live
trajectory dashboards for Google Colab, Kaggle, and terminal sessions.

Features:
  • Continuous Cloud Hub or Local Checkpoint Monitoring.
  • Deterministic Held-Out Validation Evaluation (true generalization vs batch noise).
  • Exponential Moving Average (EMA) and 5-Point Slope Analysis.
  • Anomaly & Regression Detection (spikes > threshold or consecutive climbs).
  • Dual Visualization: In-place Colab Matplotlib dashboard + ASCII terminal sparkline.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time

# Ensure repo root is on sys.path regardless of execution CWD
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# Optional Matplotlib for Colab / Interactive Environments
try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class RegressionStatus(str, Enum):
    SMOOTH_DESCENT = "SMOOTH_DESCENT"
    PLATEAU = "PLATEAU"
    REGRESSION_SPIKE = "REGRESSION_SPIKE"
    PERSISTENT_REGRESSION = "PERSISTENT_REGRESSION"
    INITIALIZING = "INITIALIZING"


@dataclass
class LossRecord:
    step: int
    loss: float
    eval_loss: Optional[float] = None
    ema_loss: Optional[float] = None
    delta: Optional[float] = None
    status: RegressionStatus = RegressionStatus.INITIALIZING
    timestamp: Optional[str] = None
    source: str = "cloud"
    md5: Optional[str] = None
    size_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, RegressionStatus) else str(self.status)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LossRecord":
        payload = dict(data)
        if "status" in payload:
            try:
                payload["status"] = RegressionStatus(payload["status"])
            except ValueError:
                payload["status"] = RegressionStatus.INITIALIZING
        return cls(**payload)


@dataclass
class RegressionAnalysis:
    status: RegressionStatus
    is_regressing: bool
    slope: float
    last_delta: float
    consecutive_regressions: int
    projected_steps_to_target: Dict[float, Optional[int]] = field(default_factory=dict)
    message: str = ""


class LossTrajectoryAnalyzer:
    """Analyzes loss progressions to detect anomalies, regressions, or healthy descent."""

    def __init__(
        self,
        regression_threshold: float = 0.15,
        plateau_threshold: float = 0.005,
        ema_alpha: float = 0.2,
    ):
        self.regression_threshold = regression_threshold
        self.plateau_threshold = plateau_threshold
        self.ema_alpha = ema_alpha

    def analyze(self, records: List[LossRecord]) -> RegressionAnalysis:
        if not records:
            return RegressionAnalysis(
                status=RegressionStatus.INITIALIZING,
                is_regressing=False,
                slope=0.0,
                last_delta=0.0,
                consecutive_regressions=0,
                message="No records available yet.",
            )

        if len(records) == 1:
            return RegressionAnalysis(
                status=RegressionStatus.INITIALIZING,
                is_regressing=False,
                slope=0.0,
                last_delta=0.0,
                consecutive_regressions=0,
                message=f"Initial checkpoint at Step {records[0].step} (Loss: {records[0].loss:.4f}).",
            )

        # Work on primary metric: prioritize eval_loss if consistently present, else loss
        has_eval = any(r.eval_loss is not None for r in records[-5:])
        vals = [
            (r.eval_loss if (has_eval and r.eval_loss is not None) else r.loss)
            for r in records
        ]
        steps = [r.step for r in records]

        last_delta = vals[-1] - vals[-2]
        records[-1].delta = last_delta

        # Update EMA loss
        ema = records[0].loss
        for r in records:
            metric = r.eval_loss if (has_eval and r.eval_loss is not None) else r.loss
            ema = self.ema_alpha * metric + (1.0 - self.ema_alpha) * ema
            r.ema_loss = ema

        # Check consecutive rises
        consecutive_regressions = 0
        for i in range(len(vals) - 1, 0, -1):
            if vals[i] > vals[i - 1]:
                consecutive_regressions += 1
            else:
                break

        # Compute rolling slope (last min(5, N) points)
        k = min(5, len(records))
        sub_steps = steps[-k:]
        sub_vals = vals[-k:]
        step_span = sub_steps[-1] - sub_steps[0]
        if step_span > 0:
            slope = (sub_vals[-1] - sub_vals[0]) / float(step_span)
        else:
            slope = 0.0

        # Classify status
        is_regressing = False
        if consecutive_regressions >= 3:
            status = RegressionStatus.PERSISTENT_REGRESSION
            is_regressing = True
            msg = f"🚨 PERSISTENT REGRESSION: Loss has increased over {consecutive_regressions} consecutive checkpoints! (Total climb: {vals[-1] - vals[-consecutive_regressions-1]:+.4f})"
        elif last_delta > self.regression_threshold:
            status = RegressionStatus.REGRESSION_SPIKE
            is_regressing = True
            msg = f"⚠️ SUDDEN REGRESSION SPIKE: Loss jumped by {last_delta:+.4f} (Threshold: +{self.regression_threshold:.2f})!"
        elif abs(last_delta) <= self.plateau_threshold and abs(slope * 1000.0) <= self.plateau_threshold:
            status = RegressionStatus.PLATEAU
            msg = f"🟡 LOSS PLATEAU: Progress is currently flat (Δ={last_delta:+.4f}, slope={slope*1000:+.4f}/1k steps)."
        elif slope < 0.0 or last_delta < 0.0:
            status = RegressionStatus.SMOOTH_DESCENT
            msg = f"🟢 SMOOTH DESCENT: Training progressing well (Δ={last_delta:+.4f}, slope={slope*1000:+.4f}/1k steps)."
        else:
            status = RegressionStatus.SMOOTH_DESCENT
            msg = f"🟢 NORMAL VARIANCE: Micro-fluctuation within acceptable limits (Δ={last_delta:+.4f})."

        records[-1].status = status

        # Forecast horizons for milestones (3.0, 2.5, 2.0)
        projections: Dict[float, Optional[int]] = {}
        for target in (3.0, 2.5, 2.0):
            if vals[-1] <= target:
                projections[target] = records[-1].step
            elif slope < -1e-6:
                needed = vals[-1] - target
                add_steps = int(needed / abs(slope))
                projections[target] = records[-1].step + add_steps
            else:
                projections[target] = None

        return RegressionAnalysis(
            status=status,
            is_regressing=is_regressing,
            slope=slope,
            last_delta=last_delta,
            consecutive_regressions=consecutive_regressions,
            projected_steps_to_target=projections,
            message=msg,
        )


def render_ascii_sparkline(values: List[float], width: int = 24) -> str:
    """Renders a unicode mini-sparkline of numerical values."""
    if not values:
        return ""
    if len(values) > width:
        # Downsample or take the latest window
        sub = values[-width:]
    else:
        sub = values
    vmin, vmax = min(sub), max(sub)
    if math.isclose(vmin, vmax, abs_tol=1e-6):
        return "─" * len(sub)
    ticks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    res = []
    for v in sub:
        idx = int(round((v - vmin) / (vmax - vmin) * (len(ticks) - 1)))
        idx = max(0, min(len(ticks) - 1, idx))
        res.append(ticks[idx])
    return "".join(res)


def evaluate_checkpoint_loss(
    model: torch.nn.Module,
    eval_batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int = 0,
    device: Optional[torch.device] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Runs a deterministic held-out evaluation pass over pre-batched validation tensors."""
    dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for prompts, targets, feedback in eval_batches:
            p = prompts.to(dev)
            t = targets.to(dev)
            fb = feedback.to(dev)
            amp_dtype = torch.float16 if (dev.type == "cuda") else torch.bfloat16
            with torch.amp.autocast(dev.type, enabled=use_amp, dtype=amp_dtype):
                out = model(p, targets=t, feedback_tokens=fb, pad_id=pad_id)
                loss = out[0] if isinstance(out, tuple) else out
                logits = out[1] if (isinstance(out, tuple) and len(out) > 1) else None

            total_loss += float(loss.item())
            if logits is not None:
                preds = torch.argmax(logits, dim=-1)
                active = (t != -100) & (t != pad_id)
                correct_tokens += int(((preds == t) & active).sum().item())
                total_tokens += int(active.sum().item())

    avg_loss = total_loss / max(1, len(eval_batches))
    ppl = math.exp(min(avg_loss, 20.0))
    tok_acc = (correct_tokens / max(1, total_tokens)) * 100.0 if total_tokens > 0 else 0.0

    return {
        "val_loss": avg_loss,
        "perplexity": ppl,
        "token_accuracy": tok_acc,
    }


def render_colab_figure(
    records: List[LossRecord],
    analysis: RegressionAnalysis,
    save_path: Optional[str] = None,
    show_in_notebook: bool = True,
):
    """Generates an informative, dual-panel trajectory plot optimized for Google Colab."""
    if not MATPLOTLIB_AVAILABLE or not records:
        return

    # In Colab/Jupyter, clear prior output for a live in-place updating dashboard
    try:
        from IPython.display import clear_output, display
        in_ipython = True
    except Exception:
        in_ipython = False

    steps = [r.step for r in records]
    losses = [r.loss for r in records]
    emas = [r.ema_loss if r.ema_loss is not None else r.loss for r in records]
    eval_steps = [r.step for r in records if r.eval_loss is not None]
    eval_losses = [r.eval_loss for r in records if r.eval_loss is not None]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    # Panel 1: Loss Trajectory
    ax1.plot(steps, losses, "o--", color="#6c757d", alpha=0.5, label="Reported Step Loss", markersize=4)
    ax1.plot(steps, emas, "-", color="#0d6efd", linewidth=2.5, label="Smoothed EMA Loss")
    if eval_losses:
        ax1.plot(eval_steps, eval_losses, "D-", color="#198754", linewidth=2.0, markersize=6, label="Held-Out Validation Loss")

    # Horizon milestones
    ax1.axhline(3.0, color="#ffc107", linestyle=":", linewidth=1.5, label="Target Milestone (3.0)")
    ax1.axhline(2.5, color="#fd7e14", linestyle=":", linewidth=1.5, label="Consolidation Milestone (2.5)")

    # Highlight regression spikes in red
    for i in range(1, len(records)):
        if records[i].status in (RegressionStatus.REGRESSION_SPIKE, RegressionStatus.PERSISTENT_REGRESSION):
            ax1.axvspan(records[i - 1].step, records[i].step, color="#dc3545", alpha=0.2)

    status_color = "#198754" if not analysis.is_regressing else "#dc3545"
    ax1.set_title(
        f"CerebroSpheroid V5 Training Trajectory | Status: {analysis.status.value}\n{analysis.message}",
        fontsize=12,
        fontweight="bold",
        color=status_color,
    )
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Step-by-Step Delta (dL)
    deltas = [r.delta if r.delta is not None else 0.0 for r in records]
    bar_colors = ["#dc3545" if d > 0 else "#198754" for d in deltas]
    ax2.bar(steps, deltas, width=max(10, (steps[-1] - steps[0]) // max(1, len(steps) * 2)), color=bar_colors, alpha=0.85)
    ax2.axhline(0.0, color="#000000", linewidth=1.0)
    ax2.set_xlabel("Global Training Step", fontsize=11)
    ax2.set_ylabel("Δ Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150)

    if in_ipython and show_in_notebook:
        clear_output(wait=True)
        display(fig)
        plt.close(fig)
    else:
        plt.close(fig)


def build_deterministic_eval_batches(
    tokenizer: Any,
    preset: str = "v5_small",
    num_samples: int = 16,
    prompt_len: int = 32,
    seq_len: int = 256,
    seed: int = 4242,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Builds a fixed, reproducible validation probe from local corpus or gold reasoning pairs."""
    from src.graded_curriculum import GradedCurriculumStreamer, CurriculumScheduleConfig

    sched = CurriculumScheduleConfig(prompt_len=prompt_len, seq_len=seq_len, batch_size=min(8, num_samples))
    streamer = GradedCurriculumStreamer(
        tokenizer=tokenizer,
        config=sched,
        seed=seed,
        mix="auto",
        search_dirs=["data"],
        offline=True,
        rank=0,
        world_size=1,
    )
    batches = []
    needed_batches = max(1, num_samples // sched.batch_size)
    for _ in range(needed_batches):
        p, t, fb, _, _, _, _ = streamer.get_elastic_batch(
            step=0,
            device=torch.device("cpu"),
            batch_size=sched.batch_size,
            total_window=prompt_len + seq_len,
            tail_probability=0.0,
            buckets=[("static", prompt_len, seq_len)],
        )
        batches.append((p, t, fb))
    streamer.close()
    return batches


def run_loss_tracker(
    model: str = "v5_small",
    watch: bool = False,
    interval_sec: int = 60,
    eval_model: bool = False,
    regression_threshold: float = 0.15,
    history_file: Optional[str] = None,
    save_plot: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    cloud_url: Optional[str] = None,
    cloud_token: Optional[str] = None,
    save_to_cloud: bool = True,
):
    """
    Main entry point for live loss tracking and regression audit.
    Can be imported and called directly inside Google Colab cells or run from CLI.
    Persists and synchronizes loss trajectory over time to local disk and bioflo.in website data.
    """
    from src.cloud_sync import (
        check_remote_manifest,
        pull_checkpoint,
        push_remote_loss_history,
        fetch_remote_loss_history,
    )
    from src.corpus import build_tokenizer
    from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5, load_v5_checkpoint_into_torch

    hist_path = history_file or f"loss_history_{model}.json"
    records: List[LossRecord] = []

    # 1. Load existing history if available locally
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = [LossRecord.from_dict(d) for d in data]
                print(f"📂 Loaded {len(records)} historical tracking entries from local {hist_path}")
        except Exception as e:
            print(f"⚠️ Could not read local history file {hist_path}: {e}")

    # 2. Fetch remote loss history from bioflo.in website data and merge
    if save_to_cloud:
        try:
            remote_records = fetch_remote_loss_history(model=model, url=cloud_url, token=cloud_token)
            if remote_records:
                by_step: Dict[int, LossRecord] = {}
                for d in remote_records:
                    r = LossRecord.from_dict(d)
                    by_step[r.step] = r
                for r in records:
                    if r.step in by_step:
                        if r.eval_loss is not None and by_step[r.step].eval_loss is None:
                            by_step[r.step].eval_loss = r.eval_loss
                        if r.ema_loss is not None:
                            by_step[r.step].ema_loss = r.ema_loss
                    else:
                        by_step[r.step] = r
                records = sorted(by_step.values(), key=lambda x: x.step)
                print(f"🌐 [cloud_sync] Synchronized {len(records)} loss records from bioflo.in website data!")
                try:
                    with open(hist_path, "w", encoding="utf-8") as f:
                        json.dump([r.to_dict() for r in records], f, indent=2)
                except Exception:
                    pass
            elif records:
                # Seed website data with existing local records
                push_remote_loss_history(model=model, records=records, url=cloud_url, token=cloud_token)
        except Exception as sync_init_err:
            print(f"ℹ️ [cloud_sync] Notice on initial website history fetch: {sync_init_err}")

    analyzer = LossTrajectoryAnalyzer(regression_threshold=regression_threshold)
    tokenizer = None
    eval_batches = None
    cached_model = None
    last_processed_md5 = records[-1].md5 if records else None

    if eval_model:
        try:
            tokenizer = build_tokenizer("llama")
            eval_batches = build_deterministic_eval_batches(tokenizer=tokenizer, preset=model)
            print(f"✓ Initialized {len(eval_batches)} deterministic held-out evaluation batches")
        except Exception as e:
            print(f"ℹ️ Could not initialize held-out eval batches ({e}). Falling back to manifest loss tracking.")
            eval_model = False

    print("=" * 80)
    print(f"📊 CEREBROSPHEROID V5 LOSS REGRESSION MONITOR [{model.upper()}]")
    print("=" * 80)
    print(f"Mode                : {'Continuous Watch (Polling every ' + str(interval_sec) + 's)' if watch else 'Single Audit Pass'}")
    print(f"Regression Trigger  : Spike > +{regression_threshold:.2f} or 3 consecutive climbs")
    print(f"Held-Out Evaluation : {'Active (Deterministic Probe)' if eval_model else 'Disabled (Manifest only)'}")
    print(f"Website Data Sync   : {'Active (https://bioflo.in/api/cerebro/data/)' if save_to_cloud else 'Disabled'}")
    print(f"History File        : {hist_path}")
    print("-" * 80)

    try:
        while True:
            # 1. Fetch remote cloud manifest or inspect local directory
            manifest = check_remote_manifest(model=model, url=cloud_url)
            model_info = manifest.get("models", {}).get(model, {})
            latest_meta = model_info.get("latest_manifest") or manifest.get("manifest", {}).get(f"{model}_latest")

            if latest_meta is None:
                # Fallback to root latest if available
                latest_meta = manifest.get("manifest", {}).get("latest")

            if latest_meta:
                step = int(latest_meta.get("step", 0))
                reported_loss = float(latest_meta.get("loss", 0.0))
                updated_at = latest_meta.get("updated_at", "")
                md5_val = latest_meta.get("md5", "")
                size_bytes = latest_meta.get("size_bytes", 0)

                # Check if this is a newly arrived checkpoint
                is_new_step = (not records or records[-1].step != step or records[-1].md5 != md5_val)

                if is_new_step:
                    computed_eval_loss = None

                    # If evaluation requested, download checkpoint and evaluate held-out loss
                    if eval_model and eval_batches and tokenizer:
                        scratch_ckpt = f"/tmp/{model}_eval_probe.pt"
                        pulled = pull_checkpoint(scratch_ckpt, target_type="latest", model=model, url=cloud_url)
                        if pulled and os.path.exists(scratch_ckpt):
                            try:
                                if cached_model is None:
                                    preset_cfg = PRESETS_V5[model].copy()
                                    preset_cfg["vocab_size"] = tokenizer.vocab_size
                                    cached_model = CerebroSpheroidV5(ModelConfigV5(**preset_cfg))
                                ckpt_dict = torch.load(scratch_ckpt, map_location="cpu", weights_only=False)
                                load_v5_checkpoint_into_torch(cached_model, ckpt_dict, verbose=False)
                                eval_metrics = evaluate_checkpoint_loss(
                                    model=cached_model,
                                    eval_batches=eval_batches,
                                    pad_id=getattr(tokenizer, "pad_token_id", 0) or 0,
                                )
                                computed_eval_loss = eval_metrics["val_loss"]
                            except Exception as ev_err:
                                print(f"⚠️ Eval error on new checkpoint: {ev_err}")

                    rec = LossRecord(
                        step=step,
                        loss=reported_loss,
                        eval_loss=computed_eval_loss,
                        timestamp=updated_at,
                        source=latest_meta.get("source", "cloud"),
                        md5=md5_val,
                        size_bytes=size_bytes,
                    )
                    records.append(rec)

                    # Update analysis so status, ema_loss, delta are computed before persisting
                    analysis = analyzer.analyze(records)

                    # Save updated history locally
                    try:
                        with open(hist_path, "w", encoding="utf-8") as f:
                            json.dump([r.to_dict() for r in records], f, indent=2)
                    except Exception as wr_err:
                        print(f"⚠️ Could not write history: {wr_err}")

                    # Mirror loss trajectory over time to bioflo.in website data
                    if save_to_cloud:
                        ok = push_remote_loss_history(model=model, records=records, url=cloud_url, token=cloud_token)
                        if ok:
                            print(f"🌐 [cloud_sync] Mirrored loss trajectory to bioflo.in website data (Step {rec.step:,}, Loss {rec.loss:.4f})")

            # 2. Run Trajectory Analysis
            analysis = analyzer.analyze(records)

            # 3. Print Live Progress & Sparklines
            if records:
                latest = records[-1]
                spark_vals = [r.loss for r in records]
                sparkline = render_ascii_sparkline(spark_vals, width=20)
                eval_str = f" | EvalLoss: {latest.eval_loss:.4f}" if latest.eval_loss is not None else ""
                ema_str = f" | EMA: {latest.ema_loss:.4f}" if latest.ema_loss is not None else ""
                delta_str = f" | Δ: {latest.delta:+.4f}" if latest.delta is not None else ""

                print(
                    f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Step {latest.step:6d} | "
                    f"Loss: {latest.loss:.4f}{ema_str}{eval_str}{delta_str} | "
                    f"Trend: [{sparkline}] | {analysis.status.value}"
                )
                if analysis.is_regressing:
                    print(f"  >>> {analysis.message}")

            # 4. Render visual dashboard if in Colab/notebook or save_plot requested
            if MATPLOTLIB_AVAILABLE and records:
                render_colab_figure(
                    records=records,
                    analysis=analysis,
                    save_path=save_plot or f"loss_trajectory_{model}.png",
                    show_in_notebook=watch,
                )

            if not watch:
                break

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user.")

    return records, analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Live Loss Trajectory & Regression Monitor for CerebroSpheroid V5")
    parser.add_argument("--model", type=str, default="v5_small", choices=["v5_small", "v5_compact", "v5_base"], help="Model preset architecture tier")
    parser.add_argument("--watch", action="store_true", help="Continuously poll and monitor for new checkpoints")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds (default: 60)")
    parser.add_argument("--eval", dest="eval_model", action="store_true", help="Download checkpoints and evaluate held-out validation loss")
    parser.add_argument("--threshold", type=float, default=0.15, help="Regression jump threshold (default: 0.15)")
    parser.add_argument("--history-file", type=str, default=None, help="Path to JSON file storing tracking records")
    parser.add_argument("--save-plot", type=str, default=None, help="Save matplotlib trajectory plot image to this path")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Optional local checkpoint directory to monitor instead of cloud")
    parser.add_argument("--cloud-url", type=str, default=None, help="Custom cloud sync endpoint URL")
    parser.add_argument("--cloud-token", type=str, default=None, help="Custom authorization token for bioflo.in")
    parser.add_argument("--cloud-sync", "--cloud_sync", dest="save_to_cloud", action="store_true", default=True, help="Save and sync loss over time to bioflo.in website data (default: True)")
    parser.add_argument("--no-cloud-sync", "--no_cloud_sync", dest="save_to_cloud", action="store_false", help="Disable saving loss to website data")
    return parser.parse_args()


def main():
    args = parse_args()
    run_loss_tracker(
        model=args.model,
        watch=args.watch,
        interval_sec=args.interval,
        eval_model=args.eval_model,
        regression_threshold=args.threshold,
        history_file=args.history_file,
        save_plot=args.save_plot,
        checkpoint_dir=args.checkpoint_dir,
        cloud_url=args.cloud_url,
        cloud_token=args.cloud_token,
        save_to_cloud=args.save_to_cloud,
    )


if __name__ == "__main__":
    main()
