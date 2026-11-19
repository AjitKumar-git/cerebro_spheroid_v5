#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Supervised Fine-Tuning (SFT) & Instruction Alignment Launcher
=====================================================================================
One-click CLI launcher for SFT alignment of CerebroSpheroid V5 models:
  • Automatically initializes from pre-trained champion checkpoint (Step 900k on bioflo.in).
  • Instruction-following alignment using curated SmolTalk, GSM8K-CoT, Self-OSS, and Magpie.
  • Invariant science anchor replay (10% FineWeb-Edu) guarantees zero knowledge forgetting.
  • Hardware Auto-Tuning: Adapts TPU v5e/v6e optimal batch sizes (Batch 128 @ 390k tok/s).
  • Live conversational probing every 500 steps across 5 canonical task categories.
  • Asynchronous live synchronization with bioflo.in under {preset}_sft with full token telemetry.

Usage:
  # Launch SFT on Google Colab or Cloud TPU/GPU:
  python3 scripts/run_sft_training.py --preset tpu_v5e_1 --effective-batch-size 256 --total-steps 50000 --lr 3e-5
  python3 scripts/run_sft_training.py --preset auto --total-steps 50000
"""

import argparse
import json
import math
import os
import sys
import urllib.request

# Ensure repo root is in python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch CerebroSpheroid V5 Supervised Fine-Tuning (SFT)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="auto",
        choices=["auto", "v5_small", "v5_compact", "v5_base", "tpu_v5e_1", "tpu_v6e_1"],
        help="Architecture preset (default: auto -> auto-detects tpu_v6e_1 / tpu_v5e_1 / v5_small)",
    )
    parser.add_argument(
        "--sft-model-name",
        "--sft_model_name",
        dest="sft_model_name",
        type=str,
        default=None,
        help="Cloud sync model name (default: {preset}_sft)",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        "--pretrained_checkpoint",
        dest="pretrained_checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained base model checkpoint (auto-discovered if None)",
    )
    parser.add_argument(
        "--resume-sft",
        "--resume_sft",
        dest="resume_sft",
        action="store_true",
        default=False,
        help="Resume an existing SFT run from {preset}_sft_latest.pt",
    )
    parser.add_argument(
        "--total-steps",
        "--steps",
        dest="total_steps",
        type=int,
        default=50_000,
        help="Total SFT steps (default: 50,000)",
    )
    parser.add_argument(
        "--target-tokens",
        "--target_tokens",
        dest="target_tokens",
        type=int,
        default=None,
        help="Optional target token budget for SFT campaign",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=None,
        help="Micro-batch size per device (auto-selected via hardware config if None)",
    )
    parser.add_argument(
        "--effective-batch-size",
        "--effective_batch_size",
        dest="effective_batch_size",
        type=int,
        default=256,
        help="Target effective batch size (default: 256 for TPU, 128 for GPU)",
    )
    parser.add_argument(
        "--max-batch-size",
        "--max_batch_size",
        dest="max_batch_size",
        type=int,
        default=128,
        help="Max micro-batch size cap for AutoTuner (default: 128)",
    )
    parser.add_argument(
        "--prompt-len",
        "--prompt_len",
        dest="prompt_len",
        type=int,
        default=1024,
        help="Sensory context prompt length (default: 1024 tokens)",
    )
    parser.add_argument(
        "--seq-len",
        "--seq_len",
        dest="seq_len",
        type=int,
        default=1024,
        help="Motor action target generation length (default: 1024 tokens)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3.0e-5,
        help="Peak SFT learning rate (default: 3.0e-5)",
    )
    parser.add_argument(
        "--min-lr",
        "--min_lr",
        dest="min_lr",
        type=float,
        default=3.0e-6,
        help="SFT floor learning rate (default: 3.0e-6)",
    )
    parser.add_argument(
        "--warmup-steps",
        "--warmup_steps",
        dest="warmup_steps",
        type=int,
        default=500,
        help="Linear warmup steps (default: 500)",
    )
    parser.add_argument(
        "--save-interval",
        "--save_interval",
        dest="save_interval",
        type=int,
        default=250,
        help="Checkpoint save interval (default: 250)",
    )
    parser.add_argument(
        "--eval-interval",
        "--eval_interval",
        dest="eval_interval",
        type=int,
        default=100,
        help="Canonical evaluation interval (default: 100)",
    )
    parser.add_argument(
        "--probe-interval",
        "--probe_interval",
        dest="probe_interval",
        type=int,
        default=500,
        help="Live conversational generation probe interval (default: 500)",
    )
    parser.add_argument(
        "--target-eval-loss",
        "--target_eval_loss",
        dest="target_eval_loss",
        type=float,
        default=1.40,
        help="Target held-out evaluation loss for convergence (default: 1.40)",
    )
    parser.add_argument(
        "--consistent-eval-threshold",
        "--consistent_eval_threshold",
        dest="consistent_eval_threshold",
        type=int,
        default=3,
        help="Consecutive evaluations <= target-eval-loss required to confirm convergence (default: 3)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        dest="checkpoint_dir",
        type=str,
        default="/content/checkpoints" if os.path.exists("/content") else os.path.join(REPO_ROOT, "checkpoints"),
        help="Local checkpoint directory",
    )
    parser.add_argument(
        "--sampling-p",
        "--sampling_p",
        dest="sampling_p",
        type=float,
        default=1.0,
        help="Teacher sampling probability (default: 1.0 for teacher-forcing, 0.5 for deliberative SFT)",
    )
    parser.add_argument(
        "--deliberation-mode",
        "--deliberation_mode",
        dest="deliberation_mode",
        type=str,
        default="teacher_forcing",
        choices=["teacher_forcing", "scheduled", "pitfall_recovery", "internal"],
        help="Deliberation mode (default: teacher_forcing)",
    )
    parser.add_argument(
        "--cloud-sync",
        dest="cloud_sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live bi-directional sync with bioflo.in (default: True)",
    )
    parser.add_argument(
        "--auto-tune",
        "--auto_tune",
        dest="auto_tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-tune or adopt verified hardware config (default: True)",
    )
    parser.add_argument(
        "--force-tune",
        "--force_tune",
        dest="force_tune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force re-benchmarking ladder (default: False)",
    )
    return parser.parse_args()


def check_remote_status(preset: str, sft_name: str):
    base_norm = "v5_small" if preset in ("tpu_v5e_1", "tpu_v6e_1", "tpu", "v5_tpu") else preset
    try:
        url = f"https://bioflo.in/api/cerebro/sync.php?action=status&model={base_norm}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {os.environ.get('BIOFLO_SYNC_TOKEN', os.environ.get('CEREBRO_SYNC_TOKEN', ''))}",
                "User-Agent": "CerebroCloudSync/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models_block = data.get("models", {})
            m_info = models_block.get(base_norm, {})
            l_info = m_info.get("latest_manifest", {})
            b_info = m_info.get("best_manifest", {})
            print(f"🌐 [bioflo.in] Remote Pre-Trained Foundation for {base_norm}:")
            if l_info and "step" in l_info:
                print(f"   ▶ Latest Step : {int(l_info.get('step', 0)):,} (Loss: {float(l_info.get('loss', 0)):.4f})")
            if b_info and "step" in b_info:
                print(f"   ★ Best Step   : {int(b_info.get('step', 0)):,} (Loss: {float(b_info.get('loss', 0)):.4f})")
    except Exception as e:
        print(f"⚠️ [bioflo.in] Pre-trained status check notice: {e}")


def main():
    args = parse_args()

    # Hardware auto-detection
    if args.preset == "auto":
        try:
            import jax
            devs = jax.devices()
            dev_name = devs[0].device_kind.lower() if devs else ""
            if "v6" in dev_name or "trillium" in dev_name:
                args.preset = "tpu_v6e_1"
            elif "v5" in dev_name:
                args.preset = "tpu_v5e_1"
            else:
                args.preset = "v5_small"
        except Exception:
            args.preset = "v5_small"
        print(f"⚡ Hardware auto-detected preset: {args.preset}")

    base_preset = "v5_small" if args.preset in ("tpu_v5e_1", "tpu_v6e_1") else args.preset
    sft_model_name = args.sft_model_name or f"{base_preset}_sft"

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/tmp/hf_cache"
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    print("=" * 84)
    print("🧠 CEREBROSPHEROID V5: CONVERSATIONAL & REASONING SFT LAUNCHER")
    print("=" * 84)
    print(f"  Architecture Preset : {args.preset} (Base: {base_preset})")
    print(f"  SFT Model Registry  : {sft_model_name}")
    print(f"  Total SFT Steps     : {args.total_steps:,}")
    print(f"  Context Budget      : Prompt={args.prompt_len} tok | Target={args.seq_len} tok")
    print(f"  SFT Learning Rate   : {args.lr:g} -> {args.min_lr:g} (cosine cooldown)")
    print(f"  Target Effective BS : {args.effective_batch_size}")
    print(f"  Loss Masking Mode   : Strict instruction masking (sensory stimuli masked with -100)")
    print(f"  Data Blend          : 30% SmolTalk + 25% Magpie/Tools + 20% GSM8K + 15% Code + 10% Edu")
    print(f"  Live Generation     : Every {args.probe_interval} steps")
    print(f"  Cloud Sync          : {args.cloud_sync} (bioflo.in)")
    print(f"  Checkpoints Dir     : {args.checkpoint_dir}")
    print("=" * 84)

    # Check remote bioflo.in status
    check_remote_status(args.preset, sft_model_name)

    # Verify offline LLaMA-32k tokenizer
    tok_path = os.path.join(REPO_ROOT, "data", "tokenizer", "tokenizer.json")
    if os.path.exists(tok_path):
        print(f"🔤 Offline LLaMA-32k tokenizer verified: {tok_path}")
    else:
        print("⚠️ Warning: data/tokenizer/tokenizer.json not found; git clone may be partial.")

    # Import and launch SFT training engine
    from training.train_v5_sft import run_sft_training

    run_sft_training(
        preset=args.preset,
        sft_model_name=sft_model_name,
        pretrained_checkpoint=args.pretrained_checkpoint,
        resume_sft=args.resume_sft,
        total_steps=args.total_steps,
        target_tokens=args.target_tokens,
        batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        max_batch_size=args.max_batch_size,
        prompt_len=args.prompt_len,
        seq_len=args.seq_len,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        sampling_p=args.sampling_p,
        deliberation_mode=args.deliberation_mode,
        auto_tune=args.auto_tune,
        force_tune=args.force_tune,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        probe_interval=args.probe_interval,
        target_eval_loss=args.target_eval_loss,
        consistent_eval_threshold=args.consistent_eval_threshold,
        cloud_sync=args.cloud_sync,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
