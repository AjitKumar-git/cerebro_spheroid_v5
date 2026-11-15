#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Autonomous Deep Pre-Training & Reasoning Engine
=======================================================================
Fully autonomous, resilient training harness for CerebroSpheroid V5 models:
  • Presets: v5_base (512M params, 48 shells, 1024 surface, 160 tracts)
             v5_small (101.6M params, 16 shells, 640 surface, 64 tracts)
  • Autonomous Curriculum: 4-Phase Developmental Evolution
      - Phase 1 (0% -> 15%): Foundation & Factual Science Imprinting
      - Phase 2 (15% -> 50%): Commonsense Scenarios & Indented Python Code
      - Phase 3 (50% -> 85%): Frontier 1:1:1 Tri-Pillar Reasoning (Science + Commonsense + Math/Code)
      - Phase 4 (85% -> 100%): Annealing Basin & Sharp Boundary Consolidation
  • Fixed Canonical 4-Domain Validation Split (Science, Commonsense, Math-CoT, Code)
  • Bi-directional cloud synchronization with bioflo.in (Zero-overhead async worker)
  • Self-healing divergence watchdog and momentum shock absorber on resume

Usage:
  python3 scripts/run_autonomous_training.py --preset v5_base --total-steps 1500000
  python3 scripts/run_autonomous_training.py --preset v5_small --total-steps 900000
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import urllib.request

# Ensure repo root is in python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CerebroSpheroid V5 Autonomous Training Engine"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="auto",
        choices=["auto", "v5_base", "v5_small", "v5_compact", "tpu_v5e_1", "tpu_v6e_1"],
        help="Architecture preset (default: auto -> auto-detects hardware and picks optimal preset)",
    )
    parser.add_argument(
        "--dataset-source",
        "--dataset_source",
        dest="dataset_source",
        type=str,
        default="autonomous",
        choices=["autonomous", "v5_base", "reasoning_anneal", "tri_pillar", "dense_edu", "consolidation", "conversation_sft", "local"],
        help="Curriculum stream: 'autonomous' for 4-phase step-aware developmental curriculum (default: autonomous)",
    )
    parser.add_argument(
        "--total-steps",
        "--steps",
        dest="total_steps",
        type=int,
        default=None,
        help="Total training steps (default: 1,500,000 for v5_base, 900,000 for v5_small)",
    )
    parser.add_argument(
        "--prompt-len",
        "--prompt_len",
        dest="prompt_len",
        type=int,
        default=256,
        help="Context prompt length (default: 256 tokens)",
    )
    parser.add_argument(
        "--seq-len",
        "--seq_len",
        dest="seq_len",
        type=int,
        default=512,
        help="Target sequence generation length (default: 512 tokens)",
    )
    parser.add_argument(
        "--elastic-window",
        "--elastic_window",
        dest="elastic_window",
        action="store_true",
        default=False,
        help="Enable dynamic sliding context buckets",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Peak learning rate (default: 2.0e-4 for v5_base, 1.5e-4 for v5_small)",
    )
    parser.add_argument(
        "--min-lr",
        "--min_lr",
        dest="min_lr",
        type=float,
        default=2.5e-5,
        help="Annealed learning rate floor (default: 2.5e-5)",
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
        default=250,
        help="Steps to smoothly warm up optimizer momentum on resume (default: 250)",
    )
    parser.add_argument(
        "--deliberation-mode",
        "--deliberation_mode",
        dest="deliberation_mode",
        type=str,
        default="teacher_forcing",
        choices=["teacher_forcing", "scheduled", "pitfall_recovery", "internal"],
        help="Deliberation feedback mode (default: teacher_forcing for maximum throughput)",
    )
    parser.add_argument(
        "--from-scratch",
        dest="from_scratch",
        action="store_true",
        default=False,
        help="Start clean from Step 0 (default: False, auto-resumes latest checkpoint)",
    )
    parser.add_argument(
        "--resume-target",
        "--resume_target",
        dest="resume_target",
        type=str,
        default="latest",
        choices=["latest", "best"],
        help="Resumption target ('latest' or 'best', default: latest)",
    )
    parser.add_argument(
        "--auto-tune",
        dest="auto_tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-tune batch ladder for maximum HBM throughput (default: True)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        dest="checkpoint_dir",
        type=str,
        default="/content/checkpoints" if os.path.exists("/content") else os.path.join(REPO_ROOT, "checkpoints"),
        help="Local checkpoint storage directory",
    )
    parser.add_argument(
        "--cloud-sync",
        dest="cloud_sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bi-directional live synchronization with bioflo.in (default: True)",
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
        "--eval-interval",
        "--eval_interval",
        dest="eval_interval",
        type=int,
        default=100,
        help="Canonical multi-domain evaluation step interval (default: 100)",
    )
    parser.add_argument(
        "--loss-window-governor",
        "--loss_window_governor",
        dest="loss_window_governor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Loss governor to clamp climb (default: True)",
    )
    parser.add_argument(
        "--loss-ceiling",
        "--loss_ceiling",
        dest="loss_ceiling",
        type=float,
        default=3.35,
        help="Upper loss threshold for governor intervention (default: 3.35)",
    )
    parser.add_argument(
        "--loss-floor",
        "--loss_floor",
        dest="loss_floor",
        type=float,
        default=2.70,
        help="Lower loss threshold for deliberation (default: 2.70)",
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
                print(f"   ▶ Remote Latest : Step {int(l_step):,} (Loss: {float(l_loss):.4f})")
            if b_step != "None":
                print(f"   ★ Remote Best   : Step {int(b_step):,} (Loss: {float(b_loss):.4f})")
    except Exception as e:
        print(f"⚠️ [bioflo.in] Status check notice: {e}")


def main():
    args = parse_args()

    # Hardware & Architecture Auto-Detection
    if args.preset == "auto":
        try:
            import jax
            devs = jax.devices()
            dev_name = devs[0].device_kind.lower() if devs else ""
            if "v6" in dev_name or "trillium" in dev_name:
                args.preset = "v5_base"
            elif "v5" in dev_name:
                args.preset = "tpu_v5e_1"
            else:
                args.preset = "v5_small"
        except Exception:
            args.preset = "v5_small"
        print(f"⚡ Auto-selected preset: {args.preset}")

    # Set preset-specific defaults
    is_base = (args.preset == "v5_base")
    if args.total_steps is None:
        args.total_steps = 1_500_000 if is_base else 900_000
    if args.lr is None:
        args.lr = 2.0e-4 if is_base else 1.5e-4

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Ensure writable HuggingFace cache
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/tmp/hf_cache"
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    total_window = args.prompt_len + args.seq_len

    print("=" * 84)
    print("🧠 CEREBROSPHEROID V5: AUTONOMOUS TRAINING & REASONING ENGINE")
    print("=" * 84)
    print(f"  Architecture Preset   : {args.preset} ({'512M params | 48 shells | 160 tracts' if is_base else '101.6M params | 16 shells | 64 tracts'})")
    print(f"  Target Steps Boundary : {args.total_steps:,} Steps")
    print(f"  Autonomous Curriculum : {args.dataset_source} (4-Phase Dynamic Staging)")
    print(f"  Context Budget        : Prompt={args.prompt_len} tok | Target={args.seq_len} tok (Total={total_window} tok)")
    print(f"  Learning Rate Decay   : {args.lr:g} -> {args.min_lr:g} ({args.lr_schedule})")
    print(f"  Momentum Warmup       : {args.resume_warmup_steps} steps on resumption (shock prevention)")
    print(f"  Deliberation Mode     : {args.deliberation_mode}")
    print(f"  Loss Window Governor  : [{args.loss_floor:.2f}, {args.loss_ceiling:.2f}] active={args.loss_window_governor}")
    print(f"  Canonical Validation  : Every {args.eval_interval} steps across 4 gold domains")
    print(f"  Checkpoints Directory : {args.checkpoint_dir}")
    print("=" * 84)

    # Check remote bioflo.in status
    check_remote_status(args.preset)

    # Verify offline LLaMA-32k tokenizer
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
        tune_objective="balanced",
        deliberation_mode=args.deliberation_mode,
        min_sampling_p=1.0,
        schedule_steps=0,
        from_scratch=args.from_scratch,
        resume_target=args.resume_target,
        loss_window_governor=args.loss_window_governor,
        loss_ceiling=args.loss_ceiling,
        loss_floor=args.loss_floor,
        dtype="auto",
        total_steps=args.total_steps,
        prompt_len=args.prompt_len,
        seq_len=args.seq_len,
        total_window=total_window,
        elastic_window=args.elastic_window,
        lr=args.lr,
        min_lr=args.min_lr,
        resume_warmup_steps=args.resume_warmup_steps,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        cloud_sync=args.cloud_sync,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
