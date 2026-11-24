#!/usr/bin/env python3
"""
scripts/train_v5_attenuated_recovery.py
======================================

CerebroSpheroid V5: Attenuated Deliberation Recovery & Telemetry Isolation Engine
----------------------------------------------------------------------------------
Safely adapts CerebroSpheroid V5 from the t-1 bypass shortcut to pure, autonomous
deliberation without polluting the production training curve on bioflo.in:

1. Telemetry Isolation (Zero Production Pollution):
   - All intermediate step logs, loss trajectories, and checkpoints are synced
     exclusively to an isolated temporary bioflo.in namespace: `v5_small_recovery`.
   - The official production graph (`v5_small`) on bioflo.in remains completely
     pristine at Step 240,508 (Loss 2.8595).

2. Attenuated Recovery Dynamics:
   - Gentle LR Warmup (300 steps ramp-up) avoids initial gradient shock.
   - Vectorized depth-decayed efference update dynamically evolves all 16 shells
     (S_0 to S_15) during emission.
   - Polar Newton-Schulz and SCMER/negative covariance preserve manifold contractility.

3. Automated Graduation Gate:
   - Evaluates held-out validation loss across FineWeb and The Stack.
   - Monitors when unassisted evaluation loss descends to <= 2.85 (target threshold).
   - Upon consistent convergence (>= 2 consecutive cycles):
     * Promotes the calibrated weights to `checkpoints/v5_small_latest.pt`.
     * Asynchronously uploads `latest.pt` and `best.pt` to the official `v5_small`
       slot on bioflo.in.
     * Transmits a graduation milestone record to bioflo.in.
"""

import os
import sys
import time
import math
import shutil
import argparse
from typing import Optional, List, Dict, Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.train_v5_tpu_jax import (
    main as tpu_train_main,
    parse_args as tpu_parse_args,
    run_training,
)
from src.cloud_sync import (
    push_checkpoint_async,
    push_remote_loss_history,
    push_remote_token_tracker,
    wait_for_in_flight_uploads,
    get_sync_token,
    detect_source_environment,
)


def parse_recovery_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CerebroSpheroid V5: Attenuated Recovery Training with Isolated Telemetry"
    )
    # Model & Checkpoint Targeting
    parser.add_argument(
        "--init_from_checkpoint",
        "--init-from-checkpoint",
        type=str,
        default="checkpoints/v5_small_latest.pt",
        help="Base checkpoint to load (default: checkpoints/v5_small_latest.pt)",
    )
    parser.add_argument(
        "--temp_model_tier",
        "--temp-model-tier",
        type=str,
        default="v5_small_recovery",
        help="Temporary bioflo.in model namespace to isolate loss graph (default: v5_small_recovery)",
    )
    parser.add_argument(
        "--main_model_tier",
        "--main-model-tier",
        type=str,
        default="v5_small",
        help="Official production bioflo.in model namespace to update only upon graduation (default: v5_small)",
    )
    parser.add_argument(
        "--target_loss",
        "--target-loss",
        type=float,
        default=2.85,
        help="Target held-out evaluation loss required to graduate back to main model (default: 2.85)",
    )
    parser.add_argument(
        "--consistent_evals",
        "--consistent-evals",
        type=int,
        default=2,
        help="Number of consecutive evaluation cycles <= target_loss required for graduation (default: 2)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="v5_small",
        choices=["v5_small", "v5_compact", "v5_base", "v5_test"],
        help="Model preset architecture (default: v5_small)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        "--checkpoint-dir",
        type=str,
        default="checkpoints/recovery",
        help="Isolated local checkpoint directory for recovery session (default: checkpoints/recovery)",
    )
    parser.add_argument(
        "--scratch_dir",
        "--scratch-dir",
        type=str,
        default="scratch/recovery",
        help="Scratch directory for temporary buffers (default: scratch/recovery)",
    )
    parser.add_argument(
        "--total_steps",
        "--total-steps",
        type=int,
        default=280_000,
        help="Maximum absolute step budget for recovery session (default: 280,000 steps)",
    )
    parser.add_argument(
        "--eval_interval",
        "--eval-interval",
        type=int,
        default=200,
        help="Interval of training steps between validation evaluations (default: 200)",
    )
    parser.add_argument(
        "--save_interval",
        "--save-interval",
        type=int,
        default=500,
        help="Interval of training steps between checkpoint saves (default: 500)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.5e-4,
        help="Conservative peak learning rate for deliberative recalibration (default: 1.5e-4)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=3e-5,
        help="Annealed learning rate floor (default: 3e-5)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=300,
        help="Steps for gentle linear warmup from 0 to peak lr (default: 300)",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=None,
        help="Per-device batch size (defaults to hardware optimal)",
    )
    parser.add_argument(
        "--dataset_source",
        "--dataset-source",
        type=str,
        default="local",
        choices=["local", "huggingface", "online"],
        help="Training dataset source (default: local)",
    )
    parser.add_argument(
        "--cloud_sync",
        "--cloud-sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live synchronization with bioflo.in (default: True)",
    )
    parser.add_argument(
        "--auto_promote",
        "--auto-promote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically promote to main v5_small on bioflo.in once loss <= target_loss (default: True)",
    )
    return parser.parse_args()


def promote_to_production(
    grad_checkpoint_path: str,
    main_model_tier: str,
    target_loss: float,
    eval_loss: float,
    step: int,
    cloud_sync: bool,
) -> bool:
    """Promotes graduated recovery checkpoint to production locally and on bioflo.in."""
    print("\n" + "=" * 80)
    print("🏆 [GRADUATION PROTOCOL INITIATED]")
    print(f"Target Loss Met: {eval_loss:.4f} <= {target_loss:.4f} at Step {step:,}!")
    print(f"Promoting recovered weights to official production tier [{main_model_tier}]...")
    print("=" * 80)

    # 1. Update local canonical latest checkpoint
    canonical_latest = os.path.join("checkpoints", f"{main_model_tier}_latest.pt")
    canonical_grad = os.path.join("checkpoints", f"{main_model_tier}_graduated_step{step}.pt")
    try:
        shutil.copy2(grad_checkpoint_path, canonical_latest)
        shutil.copy2(grad_checkpoint_path, canonical_grad)
        print(f"✅ Local production checkpoint updated: {canonical_latest}")
        print(f"✅ Recovery archive created: {canonical_grad}")
    except Exception as e:
        print(f"⚠️ Error updating local production files: {e}")

    # 2. Upload to official bioflo.in tier without prior graph pollution
    if cloud_sync:
        print(f"\n🌐 Mirroring graduated checkpoint to official bioflo.in tier [{main_model_tier}]...")
        push_checkpoint_async(
            file_path=canonical_latest,
            step=step,
            loss=eval_loss,
            target_type="latest",
            model=main_model_tier,
        )
        push_checkpoint_async(
            file_path=canonical_latest,
            step=step,
            loss=eval_loss,
            target_type="best",
            model=main_model_tier,
        )

        # Transmit clean graduation milestone record
        grad_record = {
            "step": step,
            "loss": eval_loss,
            "eval_loss": eval_loss,
            "status": "RECOVERY_GRADUATION_COMPLETE",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": detect_source_environment(),
            "note": f"Successfully adapted from t-1 bypass to pure deliberative sequence modeling. Unassisted loss: {eval_loss:.4f} <= {target_loss:.4f}",
        }
        push_remote_loss_history(
            model=main_model_tier,
            records=[grad_record],
        )
        wait_for_in_flight_uploads(timeout=60.0)
        print(f"✅ Production graduation telemetry registered on bioflo.in [{main_model_tier}]!")

    print("\n" + "=" * 80)
    print("🎉 GRADUATION COMPLETE: Production training graph remains pristine and ready.")
    print("=" * 80 + "\n")
    return True


def main():
    args = parse_recovery_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.scratch_dir, exist_ok=True)

    print("=" * 80)
    print("⚡ CEREBROSPHEROID V5: ATTENUATED DELIBERATION RECOVERY ENGINE")
    print("=" * 80)
    print(f"Base Checkpoint       : {args.init_from_checkpoint}")
    print(f"Temporary Cloud Tier  : {args.temp_model_tier} (Isolated bioflo.in namespace)")
    print(f"Official Target Tier  : {args.main_model_tier} (Protected from graph pollution)")
    print(f"Graduation Threshold  : Loss <= {args.target_loss:.4f} ({args.consistent_evals} consecutive cycles)")
    print(f"Learning Rate Profile : Warmup {args.warmup_steps} steps | Peak: {args.lr} | Floor: {args.min_lr}")
    print(f"Local Checkpoint Dir  : {args.checkpoint_dir}")
    print("=" * 80 + "\n")

    # Build argument array forwarding to pure JAX train engine
    forward_args = [
        "--preset", args.preset,
        "--init_from_checkpoint", args.init_from_checkpoint,
        "--sync_model", args.temp_model_tier,
        "--checkpoint-dir", args.checkpoint_dir,
        "--scratch_dir", args.scratch_dir,
        "--optimizer", "spheroid",
        "--dataset_source", args.dataset_source,
        "--total-steps", str(args.total_steps),
        "--eval_interval", str(args.eval_interval),
        "--save-interval", str(args.save_interval),
        "--target-eval-loss", str(args.target_loss),
        "--consistent-eval-threshold", str(args.consistent_evals),
        "--lr", str(args.lr),
        "--min_lr", str(args.min_lr),
        "--warmup_steps", str(args.warmup_steps),
        "--resume_warmup_steps", str(args.warmup_steps),
        "--reset_best",
    ]

    if not args.cloud_sync:
        forward_args.append("--no-cloud-sync")
    if args.batch_size is not None:
        forward_args.extend(["--batch_size", str(args.batch_size)])

    # Parse via train_v5_tpu_jax argument schema
    tpu_args = tpu_parse_args(forward_args)

    print(f"🛡️  [Telemetry Shield] Bioflo.in sync namespace set strictly to [{args.temp_model_tier}].")
    print(f"    Official [{args.main_model_tier}] curve will remain untouched until convergence is achieved.\n")

    # Execute training loop with isolated telemetry
    try:
        tpu_train_main(tpu_args)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")

    # Check for graduation checkpoint in recovery directory
    temp_best_path = os.path.join(args.checkpoint_dir, f"{args.preset}_best.pt")
    temp_latest_path = os.path.join(args.checkpoint_dir, f"{args.preset}_latest.pt")
    candidate_path = temp_best_path if os.path.exists(temp_best_path) else temp_latest_path

    if os.path.exists(candidate_path) and args.auto_promote:
        import torch
        ckpt = torch.load(candidate_path, map_location="cpu", weights_only=False)
        achieved_loss = float(ckpt.get("eval_loss", ckpt.get("loss", float("inf"))))
        step = int(ckpt.get("step", 0))

        if achieved_loss <= args.target_loss:
            promote_to_production(
                grad_checkpoint_path=candidate_path,
                main_model_tier=args.main_model_tier,
                target_loss=args.target_loss,
                eval_loss=achieved_loss,
                step=step,
                cloud_sync=args.cloud_sync,
            )
        else:
            print(f"\nℹ️ Recovery session concluded. Achieved Loss: {achieved_loss:.4f} (Target <= {args.target_loss:.4f}).")
            print(f"   Checkpoint preserved in {candidate_path}. Production tier [{args.main_model_tier}] unchanged.")


if __name__ == "__main__":
    main()
