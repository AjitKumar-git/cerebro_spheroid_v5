#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 CerebroSpheroid V5: Continuous Self-Healing Training Supervisor
=================================================================
Production-grade 24/7 continuous supervisor for CerebroSpheroid V5 training:
  • Autonomous Resilience   : Automatically restarts on transient network drops, HuggingFace timeouts,
                              CUDA/TPU memory resets, or platform preemption notices.
  • Seamless Resumption     : Restores the latest verified checkpoint from bioflo.in or local storage
                              and continues token digestion without manual intervention.
  • Stage-Aware Transitions : Manages curriculum progression:
                                - Stage 1 (0 -> 15.7B tokens): Broad Web Pretrain
                                - Stage 2 (15.7B -> 50B tokens): Deep Reasoning (FineMath, Cosmopedia v2, Code)
                                - Guarded Learning Rate: Protects Stage 1 foundations from catastrophic forgetting.
  • Heartbeat & Telemetry   : Reports live step, loss, tokens, and restart counts to bioflo.in and ntfy.
  • Graceful Preemption     : Intercepts SIGINT/SIGTERM, safely finishes in-flight cloud uploads, and exits cleanly.

Usage:
  python3 training/continuous_trainer.py --preset v5_small --target-tokens 50000000000
  python3 training/continuous_trainer.py --preset auto --cloud-sync
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "cs5-tpu-ajitsheoran")
DEFAULT_CLOUD_URL = os.environ.get("BIOFLO_SYNC_URL", "https://bioflo.in/api/cerebro/sync.php")
DEFAULT_SYNC_TOKEN = os.environ.get("BIOFLO_SYNC_TOKEN", "CerebroSpheroid_Sync_Secure_2026")


def publish_telemetry(event: str, **kwargs) -> None:
    """Dispatches asynchronous telemetry event to ntfy.sh and console."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    msg = {"event": event, "timestamp": ts, **kwargs}
    print(f"📡 [supervisor] {event}: {json.dumps(kwargs)}", flush=True)

    if not NTFY_TOPIC:
        return
    try:
        payload = {
            "topic": NTFY_TOPIC,
            "title": f"Cerebro Continuous Trainer: {event}",
            "message": json.dumps(msg),
            "tags": ["cerebro", "continuous-training", event],
        }
        req = urllib.request.Request(
            "https://ntfy.sh",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=4)
    except Exception:
        pass


def fetch_remote_status(model_preset: str, cloud_url: str = DEFAULT_CLOUD_URL, token: str = DEFAULT_SYNC_TOKEN) -> Dict[str, Any]:
    """Queries bioflo.in for current checkpoint, step, loss, and cumulative tokens."""
    req_url = f"{cloud_url}?action=status&model={model_preset}"
    req = urllib.request.Request(
        req_url,
        headers={"Authorization": f"Bearer {token}", "User-Agent": "ContinuousSupervisor/5.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except Exception as e:
        print(f"⚠️ [supervisor] Remote status query notice: {e}", flush=True)
        return {}


def cleanup_system_resources() -> None:
    """Flushes garbage collection and releases local hardware memory buffers."""
    gc.collect()
    try:
        import jax
        jax.clear_caches()
    except Exception:
        pass


def build_trainer_command(args: argparse.Namespace, extra_args: List[str]) -> List[str]:
    """Constructs the exact Python command line to invoke train_v5_tpu_jax.py."""
    python_bin = sys.executable
    train_script = os.path.join(REPO_ROOT, "training", "train_v5_tpu_jax.py")

    cmd = [
        python_bin,
        train_script,
        "--preset", args.preset,
        "--effective-batch-size", str(args.effective_batch_size),
        "--target-tokens", str(args.target_tokens),
        "--checkpoint-dir", args.checkpoint_dir,
        "--scratch-dir", args.scratch_dir or args.checkpoint_dir,
        "--save-interval", str(args.save_interval),
        "--eval_interval", str(args.eval_interval),
        "--log-every", str(args.log_every),
    ]

    if args.cloud_sync:
        cmd.extend(["--cloud-sync", "--cloud_url", args.cloud_url])
        if args.cloud_token:
            cmd.extend(["--cloud_token", args.cloud_token])

    if args.reasoning_transition_tokens:
        cmd.extend(["--reasoning-transition-tokens", str(args.reasoning_transition_tokens)])
    if args.final_campaign_tokens:
        cmd.extend(["--final-campaign-tokens", str(args.final_campaign_tokens)])
    if args.reasoning_max_lr:
        cmd.extend(["--reasoning-max-lr", str(args.reasoning_max_lr)])
    if args.reasoning_min_lr:
        cmd.extend(["--reasoning-min-lr", str(args.reasoning_min_lr)])

    if args.deliberation_mode:
        cmd.extend(["--deliberation_mode", args.deliberation_mode])

    if getattr(args, "smart_mixture", True):
        cmd.append("--smart-mixture")
    if getattr(args, "smart_mixture_eta", None) is not None:
        cmd.extend(["--smart-mixture-eta", str(args.smart_mixture_eta)])
    if getattr(args, "smart_mixture_prior_weight", None) is not None:
        cmd.extend(["--smart-mixture-prior-weight", str(args.smart_mixture_prior_weight)])
    if getattr(args, "smart_mixture_overfit_thresh", None) is not None:
        cmd.extend(["--smart-mixture-overfit-thresh", str(args.smart_mixture_overfit_thresh)])

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def run_continuous_training_supervisor(args: argparse.Namespace, extra_args: List[str]) -> int:
    """
    Main supervisor loop: launches the trainer process, monitors its health,
    catches crashes or preemption, and self-heals by continuously resuming.
    """
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.scratch_dir:
        os.makedirs(args.scratch_dir, exist_ok=True)

    print("=" * 84)
    print("⚡ CEREBROSPHEROID V5: CONTINUOUS SELF-HEALING TRAINING SUPERVISOR")
    print("=" * 84)
    print(f"  Preset Target          : {args.preset}")
    print(f"  Target Token Budget    : {args.target_tokens / 1e9:.2f} Billion Tokens")
    print(f"  Transition Milestone   : {args.reasoning_transition_tokens / 1e9:.2f}B (Stage 1 -> Stage 2)")
    print(f"  Stage 2 LR Guard       : {args.reasoning_max_lr:g} -> {args.reasoning_min_lr:g}")
    print(f"  Effective Batch Size   : {args.effective_batch_size}")
    print(f"  Checkpoints Directory  : {args.checkpoint_dir}")
    print(f"  Cloud Synchronization  : {args.cloud_sync} ({args.cloud_url})")
    print(f"  Max Restarts Limit     : {'Infinite' if args.max_restarts is None else args.max_restarts}")
    print("=" * 84 + "\n", flush=True)

    restart_count = 0
    start_wall_time = time.time()
    active_process: Optional[subprocess.Popen] = None
    shutdown_requested = False

    def handle_shutdown(signum, frame):
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        print(f"\n🛑 [supervisor] Received {sig_name}. Initiating graceful shutdown...", flush=True)
        shutdown_requested = True
        if active_process is not None and active_process.poll() is None:
            publish_telemetry("supervisor_preemption", signal=sig_name)
            active_process.send_signal(signum)
            try:
                active_process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print("⚠️ [supervisor] Child process timed out on graceful shutdown. Killing...", flush=True)
                active_process.kill()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    publish_telemetry("supervisor_started", preset=args.preset, target_tokens=args.target_tokens)

    while not shutdown_requested:
        # Check wall-clock time limit (e.g., Kaggle 8.5h / 9h)
        if args.max_hours is not None and args.max_hours > 0:
            elapsed_hours = (time.time() - start_wall_time) / 3600.0
            if elapsed_hours >= args.max_hours:
                print(f"⏱️ [supervisor] Wall-clock limit of {args.max_hours:.2f} hours reached. Stopping cleanly.", flush=True)
                publish_telemetry("supervisor_wall_clock_reached", elapsed_hours=elapsed_hours)
                break

        # Check max restarts limit
        if args.max_restarts is not None and restart_count >= args.max_restarts:
            print(f"🛑 [supervisor] Maximum restarts limit ({args.max_restarts}) reached. Exiting.", flush=True)
            publish_telemetry("supervisor_max_restarts_reached", restarts=restart_count)
            return 1

        # Check remote status before launch to display current state
        if args.cloud_sync:
            status_data = fetch_remote_status(args.preset, args.cloud_url, args.cloud_token or DEFAULT_SYNC_TOKEN)
            models_info = status_data.get("models", {}).get(args.preset, {})
            latest_m = models_info.get("latest_manifest", {})
            if latest_m:
                print(f"🔍 [supervisor] Remote state verified: Step {latest_m.get('step', '?')} (Loss: {latest_m.get('loss', '?')})", flush=True)

        cmd = build_trainer_command(args, extra_args)
        cmd_str = " ".join(cmd)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Enable transparent hugepages optimization if available
        env["JAX_PLATFORMS"] = env.get("JAX_PLATFORMS", "tpu,cuda,cpu")

        print(f"\n🚀 [supervisor] Launching training instance (Attempt #{restart_count + 1})...", flush=True)
        publish_telemetry("instance_launching", attempt=restart_count + 1)

        launch_time = time.time()
        try:
            active_process = subprocess.Popen(cmd, env=env, cwd=REPO_ROOT)
            returncode = active_process.wait()
            active_process = None
        except Exception as e:
            print(f"❌ [supervisor] Failed to execute trainer process: {e}", flush=True)
            returncode = -1

        instance_runtime = time.time() - launch_time

        if shutdown_requested:
            print("🏁 [supervisor] Shutdown complete. Goodbye.", flush=True)
            return 0

        # Successful completion (target budget reached)
        if returncode == 0:
            print(f"\n🎉 [supervisor] Trainer process completed successfully with exit code 0!", flush=True)
            publish_telemetry("supervisor_completed_cleanly", total_restarts=restart_count, runtime_sec=time.time() - start_wall_time)
            return 0

        # Process failed / crashed / OOM / preemption
        restart_count += 1
        print(f"\n⚠️ [supervisor] Trainer exited unexpectedly with returncode {returncode} (Ran for {instance_runtime:.1f}s).", flush=True)
        publish_telemetry("instance_crashed", returncode=returncode, runtime_sec=instance_runtime, attempt=restart_count)

        # Cleanup hardware resources before restarting
        cleanup_system_resources()

        # Exponential backoff to avoid hammering remote services if network is down
        backoff_sec = min(args.max_backoff_sec, args.initial_backoff_sec * (1.5 ** min(restart_count - 1, 6)))
        # If the process ran successfully for more than 15 minutes before crashing, reset backoff
        if instance_runtime > 900.0:
            backoff_sec = args.initial_backoff_sec

        print(f"🔄 [supervisor] Self-healing restart #{restart_count} in {backoff_sec:.1f}s...", flush=True)
        time.sleep(backoff_sec)

    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="CerebroSpheroid V5 Continuous Self-Healing Trainer")
    parser.add_argument("--preset", type=str, default="v5_small", help="Architecture preset (v5_small, v5_compact, v5_base)")
    parser.add_argument("--effective-batch-size", "--effective_batch_size", dest="effective_batch_size", type=int, default=64, help="Global effective batch size")
    parser.add_argument("--target-tokens", "--target_tokens", dest="target_tokens", type=int, default=50_000_000_000, help="Target total tokens for campaign (default: 50B)")
    parser.add_argument("--reasoning-transition-tokens", "--reasoning_transition_tokens", dest="reasoning_transition_tokens", type=int, default=15_700_000_000, help="Token milestone for Stage 1 -> Stage 2 transition")
    parser.add_argument("--final-campaign-tokens", "--final_campaign_tokens", dest="final_campaign_tokens", type=int, default=50_000_000_000, help="Final campaign token target")
    parser.add_argument("--reasoning-max-lr", "--reasoning_max_lr", dest="reasoning_max_lr", type=float, default=2e-4, help="Guarded max learning rate for Stage 2 (default: 2e-4)")
    parser.add_argument("--reasoning-min-lr", "--reasoning_min_lr", dest="reasoning_min_lr", type=float, default=3e-5, help="Annealed min learning rate for Stage 2 (default: 3e-5)")
    parser.add_argument("--deliberation_mode", type=str, default="teacher_forcing", help="Deliberation mode (teacher_forcing, scheduled)")
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", type=str, default="/content/checkpoints" if os.path.exists("/content") else os.path.join(REPO_ROOT, "checkpoints"), help="Checkpoint directory")
    parser.add_argument("--scratch-dir", "--scratch_dir", dest="scratch_dir", type=str, default=None, help="Scratch directory for local staging")
    parser.add_argument("--save-interval", "--save_interval", dest="save_interval", type=int, default=250, help="Steps between local checkpoint saves and cloud sync")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Steps between validation evaluations")
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=10, help="Steps between loss logging")
    parser.add_argument("--cloud-sync", "--cloud_sync", dest="cloud_sync", action=argparse.BooleanOptionalAction, default=True, help="Synchronize checkpoints and telemetry to bioflo.in")
    parser.add_argument("--cloud_url", type=str, default=DEFAULT_CLOUD_URL, help="Cloud sync endpoint")
    parser.add_argument("--cloud_token", type=str, default=DEFAULT_SYNC_TOKEN, help="Cloud sync authentication token")
    parser.add_argument("--max_hours", type=float, default=None, help="Optional maximum wall-clock runtime in hours")
    parser.add_argument("--max_restarts", type=int, default=None, help="Maximum restart attempts (default: None for infinite self-healing)")
    parser.add_argument("--initial_backoff_sec", type=float, default=5.0, help="Initial retry backoff delay in seconds")
    parser.add_argument("--max_backoff_sec", type=float, default=60.0, help="Maximum retry backoff delay in seconds")
    parser.add_argument("--smart-mixture", "--smart_mixture", dest="smart_mixture", action=argparse.BooleanOptionalAction, default=True, help="Enable dynamic DoReMi & generalization-regularized dataset mixture controller (default: True)")
    parser.add_argument("--smart-mixture-eta", "--smart_mixture_eta", dest="smart_mixture_eta", type=float, default=0.08, help="Exponentiated gradient step size for smart mixture")
    parser.add_argument("--smart-mixture-prior-weight", "--smart_mixture_prior_weight", dest="smart_mixture_prior_weight", type=float, default=0.20, help="Dirichlet prior smoothing weight")
    parser.add_argument("--smart-mixture-overfit-thresh", "--smart_mixture_overfit_thresh", dest="smart_mixture_overfit_thresh", type=float, default=0.04, help="Validation loss threshold above best loss triggering overfit penalty")

    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == "__main__":
    args, unknown = parse_args()
    sys.exit(run_continuous_training_supervisor(args, unknown))
