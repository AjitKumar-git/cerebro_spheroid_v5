#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Colab TPU/GPU Phase 3 Reasoning & Math-CoT Launcher
===========================================================================
Curriculum Phase : Phase 3: Reasoning & Math-CoT (Steps 500,000 -> 900,000)
Data Stream      : reasoning_anneal (GSM8K Math-CoT + OpenHermes-2.5 Logic/Code)
Context Budget   : 768 tokens (Prompt=256, Target=512)
Learning Rate    : Cosine decay 1.5e-4 -> 3.0e-5 with smooth resume warmup
Resumption Target: Latest checkpoint on bioflo.in (Step 583k+)
Target Loss      : Transitioning 1.89 -> 1.5 - 1.6

Usage on Google Colab:
  !python3 scripts/run_colab_reasoning.py --preset tpu_v5e_1
  !python3 scripts/run_colab_reasoning.py --preset tpu_v6e_1
  !python3 scripts/run_colab_reasoning.py --preset v5_small
"""

import argparse
import json
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
        description="Launch CerebroSpheroid V5 Phase 3 Reasoning & Math-CoT Training"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="auto",
        choices=["auto", "v5_small", "v5_compact", "v5_base", "tpu_v5e_1", "tpu_v6e_1"],
        help="Model preset architecture (default: auto -> auto-detects tpu_v6e_1 / tpu_v5e_1)",
    )
    parser.add_argument(
        "--dataset-source",
        "--dataset_source",
        dest="dataset_source",
        type=str,
        default="reasoning_anneal",
        choices=["reasoning_anneal", "reasoning", "dense_edu", "consolidation"],
        help="Dataset stream (default: reasoning_anneal for GSM8K + OpenHermes-2.5)",
    )
    parser.add_argument(
        "--total-steps",
        "--steps",
        dest="total_steps",
        type=int,
        default=900_000,
        help="Target boundary steps for Phase 3 (default: 900,000)",
    )
    parser.add_argument(
        "--prompt-len",
        "--prompt_len",
        dest="prompt_len",
        type=int,
        default=256,
        help="Context prompt length (default: 256 tokens for reasoning premises)",
    )
    parser.add_argument(
        "--seq-len",
        "--seq_len",
        dest="seq_len",
        type=int,
        default=512,
        help="Target sequence length (default: 512 tokens for CoT derivation)",
    )
    parser.add_argument(
        "--total-window",
        "--total_window",
        dest="total_window",
        type=int,
        default=768,
        help="Total context budget (default: 768 tokens, 256 prompt + 512 target)",
    )
    parser.add_argument(
        "--elastic-window",
        "--elastic_window",
        dest="elastic_window",
        action="store_true",
        default=False,
        help="Enable dynamic elastic sliding context buckets",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.5e-4,
        help="Phase 3 peak learning rate (default: 1.5e-4)",
    )
    parser.add_argument(
        "--min-lr",
        "--min_lr",
        dest="min_lr",
        type=float,
        default=3.0e-5,
        help="Phase 3 annealed learning rate floor (default: 3.0e-5)",
    )
    parser.add_argument(
        "--lr-schedule",
        "--lr_schedule",
        dest="lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "warm_restart"],
        help="Learning rate schedule (default: cosine)",
    )
    parser.add_argument(
        "--resume-warmup-steps",
        "--resume_warmup_steps",
        dest="resume_warmup_steps",
        type=int,
        default=200,
        help="Steps to smoothly warm up optimizer momentum on resume (default: 200)",
    )
    parser.add_argument(
        "--deliberation-mode",
        "--deliberation_mode",
        dest="deliberation_mode",
        type=str,
        default="teacher_forcing",
        choices=["teacher_forcing", "scheduled", "pitfall_recovery", "internal"],
        help="Deliberation feedback mode (default: teacher_forcing for ~200k tok/s)",
    )
    parser.add_argument(
        "--min-sampling-p",
        "--min_sampling_p",
        dest="min_sampling_p",
        type=float,
        default=1.00,
        help="Floor for sampling_p (1.00 for pure teacher forcing, 0.50 for scheduled)",
    )
    parser.add_argument(
        "--schedule-steps",
        "--schedule_steps",
        dest="schedule_steps",
        type=int,
        default=0,
        help="Steps to anneal sampling_p (0 = immediate)",
    )
    parser.add_argument(
        "--pitfall-prob",
        "--pitfall_prob",
        dest="pitfall_prob",
        type=float,
        default=0.15,
        help="Counterfactual exploration probability (default: 0.15)",
    )
    parser.add_argument(
        "--from-scratch",
        dest="from_scratch",
        action="store_true",
        default=False,
        help="Start fresh from Step 0 (default: False, resumes from checkpoint)",
    )
    parser.add_argument(
        "--resume-target",
        "--resume_target",
        dest="resume_target",
        type=str,
        default="latest",
        choices=["latest", "best"],
        help="Checkpoint resumption target ('latest' or 'best', default: latest)",
    )
    parser.add_argument(
        "--auto-tune",
        dest="auto_tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run micro-benchmark batch ladder auto-tuner (default: True)",
    )
    parser.add_argument(
        "--force-tune",
        dest="force_tune",
        action="store_true",
        default=False,
        help="Bypass cached auto-tune configuration and re-benchmark",
    )
    parser.add_argument(
        "--tune-objective",
        "--tune_objective",
        dest="tune_objective",
        type=str,
        default="balanced",
        choices=["balanced", "steps_per_sec", "tok_per_sec"],
        help="Auto-tune optimization goal (default: balanced)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        dest="checkpoint_dir",
        type=str,
        default="/content/checkpoints" if os.path.exists("/content") else os.path.join(REPO_ROOT, "checkpoints"),
        help="Local scratch directory for checkpoints",
    )
    parser.add_argument(
        "--cloud-sync",
        dest="cloud_sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bi-directional synchronization with bioflo.in (default: True)",
    )
    parser.add_argument(
        "--save-interval",
        "--save_interval",
        dest="save_interval",
        type=int,
        default=250,
        help="Checkpoint save and sync step interval (default: 250)",
    )
    parser.add_argument(
        "--loss-window-governor",
        "--loss_window_governor",
        dest="loss_window_governor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically clamp loss within floor and ceiling (default: True)",
    )
    parser.add_argument(
        "--loss-ceiling",
        "--loss_ceiling",
        dest="loss_ceiling",
        type=float,
        default=3.25,
        help="Loss ceiling governor threshold (default: 3.25)",
    )
    parser.add_argument(
        "--loss-floor",
        "--loss_floor",
        dest="loss_floor",
        type=float,
        default=2.80,
        help="Loss floor governor threshold (default: 2.80)",
    )
    return parser.parse_args()


def check_remote_status(preset: str):
    norm = "v5_small" if preset in ("tpu_v5e_1", "tpu_v6e_1", "tpu", "v5_tpu") else preset
    try:
        url = f"https://bioflo.in/api/cerebro/sync.php?action=status&model={norm}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": "Bearer CerebroSpheroid_Sync_Secure_2026",
                "User-Agent": "CerebroCloudSync/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            manifest = data.get("manifest", {})
            models_block = data.get("models", {})
            m_info = models_block.get(norm, {})

            cand_latest = [
                e
                for e in [manifest.get(f"{norm}_latest"), m_info.get("latest_manifest"), manifest.get("latest")]
                if isinstance(e, dict) and "step" in e
            ]
            latest = max(cand_latest, key=lambda x: int(x.get("step", 0))) if cand_latest else {}

            cand_best = [
                e
                for e in [manifest.get(f"{norm}_best"), m_info.get("best_manifest"), manifest.get("best")]
                if isinstance(e, dict) and "step" in e
            ]
            best = min(cand_best, key=lambda x: float(x.get("loss", float("inf")))) if cand_best else {}

            l_step = latest.get("step", "None")
            l_loss = latest.get("loss", "None")
            b_step = best.get("step", "None")
            b_loss = best.get("loss", "None")
            print(f"🌐 [bioflo.in] Remote Status for {preset} ({norm}):")
            if l_step != "None":
                print(f"   ▶ Latest Step : {int(l_step):,} (Loss: {float(l_loss):.4f})")
            if b_step != "None":
                print(f"   ★ Best Step   : {int(b_step):,} (Loss: {float(b_loss):.4f})")
    except Exception as e:
        print(f"⚠️ [bioflo.in] Status check notice: {e}")


def main():
    args = parse_args()

    # Hardware auto-detection when preset is 'auto'
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
            args.preset = "tpu_v6e_1"
        print(f"⚡ Hardware auto-detected preset: {args.preset}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Ensure total_window matches prompt + seq_len
    if args.total_window != args.prompt_len + args.seq_len:
        args.total_window = args.prompt_len + args.seq_len

    # Ensure HF_HOME is writable
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/tmp/hf_cache"
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    print("=" * 80)
    print("🧠 CEREBROSPHEROID V5: PHASE 3 REASONING & MATH-COT RUNNER")
    print("=" * 80)
    print(f"  Curriculum Phase  : Phase 3 (Steps 500,000 -> {args.total_steps:,})")
    print(f"  Architecture      : {args.preset} (101.6M params, 16 shells, 640 surface dim)")
    print(f"  Dataset Stream    : {args.dataset_source} (GSM8K Math-CoT + OpenHermes-2.5)")
    print(f"  Context Window    : Prompt={args.prompt_len} tok | Target={args.seq_len} tok (Total={args.total_window} tok)")
    print(f"  Learning Rate     : {args.lr:g} -> {args.min_lr:g} ({args.lr_schedule} decay)")
    print(f"  Resume Warmup     : {args.resume_warmup_steps} steps (momentum shock prevention)")
    print(f"  Resumption Target : {args.resume_target} (from_scratch={args.from_scratch})")
    print(f"  Deliberation Mode : {args.deliberation_mode} (min_sampling_p={args.min_sampling_p})")
    print(f"  Loss Governor     : [{args.loss_floor:.2f}, {args.loss_ceiling:.2f}] active={args.loss_window_governor}")
    print(f"  Checkpoint Dir    : {args.checkpoint_dir}")
    print("=" * 80)

    # Check remote bioflo.in status
    check_remote_status(args.preset)

    # Verify offline tokenizer
    tok_path = os.path.join(REPO_ROOT, "data", "tokenizer", "tokenizer.json")
    if os.path.exists(tok_path):
        print(f"🔤 Offline LLaMA-32k tokenizer verified: {tok_path}")
    else:
        print("⚠️ Warning: data/tokenizer/tokenizer.json not found; git clone may be partial.")

    # Import and launch training
    from training.train_v5_tpu_jax import run_training

    run_training(
        preset=args.preset,
        dataset_source=args.dataset_source,
        lr_schedule=args.lr_schedule,
        optimizer="spheroid",
        auto_tune=args.auto_tune,
        force_tune=args.force_tune,
        tune_objective=args.tune_objective,
        deliberation_mode=args.deliberation_mode,
        min_sampling_p=args.min_sampling_p,
        schedule_steps=args.schedule_steps,
        pitfall_prob=args.pitfall_prob,
        from_scratch=args.from_scratch,
        resume_target=args.resume_target,
        loss_window_governor=args.loss_window_governor,
        loss_ceiling=args.loss_ceiling,
        loss_floor=args.loss_floor,
        dtype="auto",
        total_steps=args.total_steps,
        prompt_len=args.prompt_len,
        seq_len=args.seq_len,
        total_window=args.total_window,
        elastic_window=args.elastic_window,
        lr=args.lr,
        min_lr=args.min_lr,
        resume_warmup_steps=args.resume_warmup_steps,
        save_interval=args.save_interval,
        cloud_sync=args.cloud_sync,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
