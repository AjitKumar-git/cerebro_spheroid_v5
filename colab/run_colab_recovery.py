#!/usr/bin/env python3
"""
colab/run_colab_recovery.py
==========================

CerebroSpheroid V5: Colab TPU/GPU Attenuated Deliberation Recovery Runner
------------------------------------------------------------------------
Designed specifically for Google Colab (TPU v2-8, v3-8, v4-8, v5e, or GPU A100/V100/T4)
to recalibrate CerebroSpheroid V5 from the t-1 bypass to pure autonomous deliberation.

Key Guarantees:
1. Telemetry Shield (Zero Graph Pollution):
   All training telemetry, intermediate loss curves, and checkpoints are synced
   exclusively to the isolated temporary namespace: `v5_small_recovery` on bioflo.in.
   The official `v5_small` production graph remains locked and pristine at Step 240,508.

2. Preemption & Asset Auto-Recovery:
   If `checkpoints/v5_small_latest.pt` is not present in the local Colab filesystem,
   it automatically pulls the latest champion checkpoint directly from bioflo.in.

3. Attenuated Dynamics:
   - 300-step learning rate warmup (0 -> 1.5e-4) to prevent initial shock.
   - Active depth-decayed efference update across all 16 shells.
   - Spheroid optimizer with Polar Newton-Schulz and SCMER negative covariance.

4. Automated Production Graduation Gate:
   Once unassisted held-out evaluation loss reaches <= 2.85 across consecutive cycles:
   - Updates `checkpoints/v5_small_latest.pt`.
   - Mirrors latest and best weights to the official `v5_small` production slot on bioflo.in.
   - Transmits graduation milestone record.

Usage in Google Colab Notebook:
  !python3 colab/run_colab_recovery.py
"""

import os
import sys
import time
import json
import shutil
import urllib.request
import argparse

# 1. Workspace Configuration
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# 2. Hardware Preemption Guards (Ensure JAX grabs TPU before Torch)
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TORCH_XLA_DISABLE_DISTRIBUTED", "1")
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/tmp/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import jax
import jax.numpy as jnp

from src.cloud_sync import (
    DEFAULT_URL,
    get_sync_token,
    check_remote_manifest,
    pull_checkpoint,
    push_checkpoint_async,
    push_remote_loss_history,
    wait_for_in_flight_uploads,
    detect_source_environment,
)
from training.train_v5_tpu_jax import (
    main as tpu_main,
    parse_args as tpu_parse_args,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch CerebroSpheroid V5 Attenuated Recovery on Google Colab"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="auto",
        choices=["auto", "v5_small", "v5_compact", "v5_base", "tpu_v5e_1", "tpu_v6e_1"],
        help="Model preset architecture (default: auto -> auto-detects hardware)",
    )
    parser.add_argument(
        "--init-from-checkpoint",
        "--init_from_checkpoint",
        dest="init_from_checkpoint",
        type=str,
        default="checkpoints/v5_small_latest.pt",
        help="Path to initial foundation checkpoint (default: checkpoints/v5_small_latest.pt)",
    )
    parser.add_argument(
        "--temp-model-tier",
        "--temp_model_tier",
        dest="temp_model_tier",
        type=str,
        default="v5_small_recovery",
        help="Temporary bioflo.in namespace for isolated telemetry (default: v5_small_recovery)",
    )
    parser.add_argument(
        "--main-model-tier",
        "--main_model_tier",
        dest="main_model_tier",
        type=str,
        default="v5_small",
        help="Production bioflo.in namespace updated only upon graduation (default: v5_small)",
    )
    parser.add_argument(
        "--target-loss",
        "--target_loss",
        dest="target_loss",
        type=float,
        default=2.85,
        help="Target held-out evaluation loss required for graduation (default: 2.85)",
    )
    parser.add_argument(
        "--consistent-evals",
        "--consistent_evals",
        dest="consistent_evals",
        type=int,
        default=2,
        help="Consecutive evaluation cycles <= target_loss required to confirm graduation (default: 2)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        dest="checkpoint_dir",
        type=str,
        default="/content/checkpoints/recovery" if os.path.exists("/content") else "checkpoints/recovery",
        help="Local checkpoint directory",
    )
    parser.add_argument(
        "--scratch-dir",
        "--scratch_dir",
        dest="scratch_dir",
        type=str,
        default="/content/scratch/recovery" if os.path.exists("/content") else "scratch/recovery",
        help="Local scratch directory",
    )
    parser.add_argument(
        "--dataset-source",
        "--dataset_source",
        dest="dataset_source",
        type=str,
        default="pretrain",
        help="Dataset source for recovery: 'pretrain' (SmolLM2 online streaming) or 'local' (default: pretrain)",
    )
    parser.add_argument(
        "--total-steps",
        "--total_steps",
        dest="total_steps",
        type=int,
        default=275_000,
        help="Target total step boundary for recovery session (default: 275,000)",
    )
    parser.add_argument(
        "--eval-interval",
        "--eval_interval",
        dest="eval_interval",
        type=int,
        default=200,
        help="Steps between validation evaluations (default: 200)",
    )
    parser.add_argument(
        "--save-interval",
        "--save_interval",
        dest="save_interval",
        type=int,
        default=500,
        help="Steps between local and cloud checkpoint saves (default: 500)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.5e-4,
        help="Peak learning rate for deliberative recalibration (default: 1.5e-4)",
    )
    parser.add_argument(
        "--min-lr",
        "--min_lr",
        dest="min_lr",
        type=float,
        default=3e-5,
        help="Annealed learning rate floor (default: 3e-5)",
    )
    parser.add_argument(
        "--warmup-steps",
        "--warmup_steps",
        dest="warmup_steps",
        type=int,
        default=300,
        help="Steps for gentle linear warmup (default: 300)",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=None,
        help="Per-device batch size (defaults to hardware optimal)",
    )
    parser.add_argument(
        "--cloud-sync",
        dest="cloud_sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live synchronization with bioflo.in (default: True)",
    )
    return parser.parse_args()


def ensure_foundation_checkpoint(ckpt_path: str, model_tier: str, force_pull: bool = False) -> str:
    """Ensures base checkpoint exists locally, pulling from bioflo.in if missing or force_pull is True."""
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    token = get_sync_token()

    if force_pull:
        print(f"\n🌐 [cloud_sync] Explicit pull requested: synchronizing champion [{model_tier}] from bioflo.in...")
        try:
            pulled = pull_checkpoint(
                local_dest=ckpt_path,
                target_type="latest",
                model=model_tier,
                token=token,
            )
            if pulled and os.path.exists(ckpt_path):
                sz_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                print(f"✅ Successfully pulled {model_tier} from bioflo.in: {ckpt_path} ({sz_mb:.1f} MB)")
                return ckpt_path
        except Exception as e:
            print(f"⚠️ Cloud pull attempt notice: {e}")

    if os.path.exists(ckpt_path):
        sz_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        print(f"✅ Found local base checkpoint: {ckpt_path} ({sz_mb:.1f} MB)")
        return ckpt_path

    # Check fallback location in checkpoints/
    fallback = os.path.join("checkpoints", f"{model_tier}_latest.pt")
    if os.path.exists(fallback):
        sz_mb = os.path.getsize(fallback) / (1024 * 1024)
        print(f"✅ Found local checkpoint in standard directory: {fallback} ({sz_mb:.1f} MB)")
        return fallback

    print(f"\n📥 Base checkpoint not found locally at {ckpt_path}.")
    print(f"🌐 Querying bioflo.in to pull remote champion [{model_tier}]...")
    try:
        pulled = pull_checkpoint(
            local_dest=ckpt_path,
            target_type="latest",
            model=model_tier,
            token=token,
        )
        if pulled and os.path.exists(ckpt_path):
            print(f"✅ Successfully pulled {ckpt_path} from bioflo.in cloud!")
            return ckpt_path
    except Exception as e:
        print(f"⚠️ Cloud pull attempt notice: {e}")

    # If still not found, check if destination directory has any .pt
    candidates = [f for f in os.listdir(os.path.dirname(os.path.abspath(ckpt_path))) if f.endswith(".pt")]
    if candidates:
        selected = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), candidates[0])
        print(f"ℹ️ Using discovered local checkpoint: {selected}")
        return selected

    raise FileNotFoundError(
        f"Unable to locate base checkpoint {ckpt_path} locally or on bioflo.in. "
        f"Please verify network access or place v5_small_latest.pt in checkpoints/."
    )


def promote_graduated_checkpoint(
    recovery_ckpt: str,
    main_tier: str,
    target_loss: float,
    achieved_loss: float,
    step: int,
    cloud_sync: bool,
):
    """Promotes graduated recovery model to official production tier locally and on bioflo.in."""
    print("\n" + "=" * 80)
    print("🏆 [COLAB GRADUATION PROTOCOL INITIATED]")
    print(f"  Target Achieved : Loss {achieved_loss:.4f} <= {target_loss:.4f} at Step {step:,}!")
    print(f"  Promoting to Production Tier [{main_tier}] on bioflo.in...")
    print("=" * 80)

    # 1. Update local production checkpoints
    prod_dir = "checkpoints"
    os.makedirs(prod_dir, exist_ok=True)
    canonical_latest = os.path.join(prod_dir, f"{main_tier}_latest.pt")
    canonical_grad = os.path.join(prod_dir, f"{main_tier}_recovered_step{step}.pt")

    try:
        shutil.copy2(recovery_ckpt, canonical_latest)
        shutil.copy2(recovery_ckpt, canonical_grad)
        print(f"✅ Local production checkpoint updated: {canonical_latest}")
        print(f"✅ Recovery archive preserved: {canonical_grad}")
    except Exception as e:
        print(f"⚠️ Local copy notice: {e}")

    # 2. Upload to official bioflo.in tier without prior graph pollution
    if cloud_sync:
        print(f"🌐 Mirroring graduated checkpoint to official bioflo.in tier [{main_tier}]...")
        push_checkpoint_async(
            file_path=canonical_latest,
            step=step,
            loss=achieved_loss,
            target_type="latest",
            model=main_tier,
        )
        push_checkpoint_async(
            file_path=canonical_latest,
            step=step,
            loss=achieved_loss,
            target_type="best",
            model=main_tier,
        )

        grad_record = {
            "step": step,
            "loss": achieved_loss,
            "eval_loss": achieved_loss,
            "status": "RECOVERY_GRADUATION_COMPLETE",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": detect_source_environment(),
            "note": f"Colab session successfully adapted model from t-1 bypass to pure deliberative sequence modeling. Unassisted loss: {achieved_loss:.4f} <= {target_loss:.4f}",
        }
        push_remote_loss_history(model=main_tier, records=[grad_record])
        wait_for_in_flight_uploads(timeout=60.0)
        print(f"✅ Official production telemetry updated on bioflo.in [{main_tier}]!")

    print("=" * 80)
    print("🎉 GRADUATION COMPLETE: Production training graph remains pristine and ready.")
    print("=" * 80 + "\n")


def run_recovery(
    preset: str = "auto",
    init_from_checkpoint: str = "checkpoints/v5_small_latest.pt",
    temp_model_tier: str = "v5_small_recovery",
    main_model_tier: str = "v5_small",
    target_loss: float = 2.85,
    consistent_evals: int = 2,
    checkpoint_dir: str = None,
    scratch_dir: str = None,
    dataset_source: str = "pretrain",
    total_steps: int = 285_000,
    eval_interval: int = 200,
    save_interval: int = 500,
    lr: float = 1.5e-4,
    min_lr: float = 3e-5,
    warmup_steps: int = 300,
    effective_batch_size: int = 64,
    batch_size: int = None,
    cloud_sync: bool = True,
    pull_remote: bool = False,
):
    """
    Executes the attenuated recovery process directly in-process.
    Can be invoked from interactive Colab notebooks or the CLI main runner.
    """
    if checkpoint_dir is None:
        checkpoint_dir = "/content/checkpoints/recovery" if os.path.exists("/content") else "checkpoints/recovery"
    if scratch_dir is None:
        scratch_dir = "/content/scratch/recovery" if os.path.exists("/content") else "scratch/recovery"

    # Hardware Detection
    if preset == "auto":
        try:
            devs = jax.devices()
            dev_kind = devs[0].device_kind.lower() if devs else ""
            if "v6" in dev_kind or "trillium" in dev_kind:
                preset = "tpu_v6e_1"
            elif "v5" in dev_kind:
                preset = "tpu_v5e_1"
            else:
                preset = "v5_small"
        except Exception:
            preset = "v5_small"
        print(f"⚡ Hardware auto-detected preset: {preset}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    print("=" * 80)
    print("⚡ CEREBROSPHEROID V5: COLAB ATTENUATED RECOVERY RUNNER")
    print("=" * 80)
    print(f"  Execution Node        : {detect_source_environment().upper()} Runtime")
    print(f"  Architecture Preset   : {preset}")
    print(f"  Temporary Cloud Tier  : {temp_model_tier} (Isolated bioflo.in namespace)")
    print(f"  Official Target Tier  : {main_model_tier} (Protected from graph pollution)")
    print(f"  Dataset Source        : {dataset_source}")
    print(f"  Target Graduation Loss: <= {target_loss:.4f} ({consistent_evals} cycles)")
    print(f"  Learning Rate Profile : Warmup {warmup_steps} -> Peak {lr} -> Floor {min_lr}")
    print(f"  Checkpoint Directory  : {checkpoint_dir}")
    print("=" * 80 + "\n")

    # Ensure Foundation Checkpoint Exists
    base_ckpt = ensure_foundation_checkpoint(init_from_checkpoint, main_model_tier, force_pull=pull_remote)

    # Check remote bioflo.in status
    try:
        manifest = check_remote_manifest(main_model_tier)
        print(f"🌐 [bioflo.in] Remote production tier status for [{main_model_tier}]: Connected.")
    except Exception as e:
        print(f"⚠️ [bioflo.in] Manifest check notice: {e}")

    # Build argument array for high-throughput JAX training engine
    forward_args = [
        "--preset", preset,
        "--init_from_checkpoint", base_ckpt,
        "--sync_model", temp_model_tier,
        "--checkpoint-dir", checkpoint_dir,
        "--scratch_dir", scratch_dir,
        "--optimizer", "spheroid",
        "--dataset_source", dataset_source,
        "--total-steps", str(total_steps),
        "--eval_interval", str(eval_interval),
        "--save-interval", str(save_interval),
        "--target-eval-loss", str(target_loss),
        "--consistent-eval-threshold", str(consistent_evals),
        "--lr", str(lr),
        "--min_lr", str(min_lr),
        "--warmup_steps", str(warmup_steps),
        "--resume_warmup_steps", str(warmup_steps),
        "--reset_best",
    ]

    if not cloud_sync:
        forward_args.append("--no-cloud-sync")
    if batch_size is not None:
        forward_args.extend(["--batch_size", str(batch_size)])

    tpu_args = tpu_parse_args(forward_args)

    print(f"\n🛡️  [Telemetry Shield Active] All progress synced strictly to [{temp_model_tier}].")
    print(f"    Official [{main_model_tier}] curve on bioflo.in remains completely protected.\n")

    # Run training loop
    try:
        tpu_main(tpu_args)
    except KeyboardInterrupt:
        print("\n⚠️ Colab training interrupted by user.")

    # Post-training graduation check
    best_path = os.path.join(checkpoint_dir, f"{preset}_best.pt")
    latest_path = os.path.join(checkpoint_dir, f"{preset}_latest.pt")
    candidate_path = best_path if os.path.exists(best_path) else latest_path

    if os.path.exists(candidate_path):
        import torch
        ckpt = torch.load(candidate_path, map_location="cpu", weights_only=False)
        achieved_loss = float(ckpt.get("eval_loss", ckpt.get("loss", float("inf"))))
        step = int(ckpt.get("step", 0))

        if achieved_loss <= target_loss:
            promote_graduated_checkpoint(
                recovery_ckpt=candidate_path,
                main_tier=main_model_tier,
                target_loss=target_loss,
                achieved_loss=achieved_loss,
                step=step,
                cloud_sync=cloud_sync,
            )
        else:
            print(f"\nℹ️ Session complete. Best Loss: {achieved_loss:.4f} (Graduation Target: <= {target_loss:.4f}).")
            print(f"   Model preserved in {candidate_path}. Telemetry securely recorded in [{temp_model_tier}].")


def main():
    args = parse_args()
    run_recovery(
        preset=args.preset,
        init_from_checkpoint=args.init_from_checkpoint,
        temp_model_tier=args.temp_model_tier,
        main_model_tier=args.main_model_tier,
        target_loss=args.target_loss,
        consistent_evals=args.consistent_evals,
        checkpoint_dir=args.checkpoint_dir,
        scratch_dir=args.scratch_dir,
        dataset_source=args.dataset_source,
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        cloud_sync=args.cloud_sync,
        pull_remote=getattr(args, "pull_remote", False),
    )


if __name__ == "__main__":
    main()
