#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Supervised Fine-Tuning (SFT) & Conversational Alignment Engine
=====================================================================================
Specialized instruction-tuning and conversational alignment engine for CerebroSpheroid V5:
  • Base Checkpoint Loading : Automatically initializes from the 900k pre-trained champion weights.
  • Instruction Loss Masking: Strict -100 target padding on sensory stimuli (<stim>...</stim>);
                              loss is backpropagated solely on motor articulations (<act>...</act>).
  • Hardware Auto-Tuning    : Full TPU v5e/v6e MXU hardware optimization, auto-adopting verified batch
                              ladder parameters (Batch 128 @ 390k tok/s) and effective batch scaling.
  • Gradient Accumulation   : Multi-micro-step gradient accumulation for target effective batch size (256/128).
  • Curated SFT Data Blend   :
      - 30% SmolTalk Everyday Conversations & OpenHermes-100k (Instruction following & STEM)
      - 25% Smol-Magpie-Ultra & Apigen-80k (Multi-turn deductive reasoning & agentic tool calls)
      - 20% GSM8K & MetaMathQA (Step-by-step mathematical reasoning with #### anchors)
      - 15% SmolTalk Self-OSS-Instruct (Indented Python algorithms & software implementation)
      - 10% FineWeb-Edu 4.0 Replay (Invariant science anchor preventing catastrophic forgetting)
  • SFT Optimizer Dynamics   :
      - Spheroid Optimizer with low learning rate: 3.0e-5 -> 3.0e-6 (cosine cooldown)
      - Conservative weight decay: 0.01 (preserves foundational manifold topology)
      - 500-step smooth linear warmup with Loss Window Governor
  • Continuous Live Probing  : Generates and logs actual assistant responses to test prompts every 500 steps.
  • Bi-directional Cloud Sync: Asynchronously syncs SFT latest and champion checkpoints and live token
                              telemetry (breakdown by dataset) to bioflo.in.

Usage:
  python3 training/train_v5_sft.py --preset tpu_v5e_1 --effective-batch-size 256 --total-steps 50000 --lr 3e-5
  python3 training/train_v5_sft.py --preset auto --total-steps 50000
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
import signal
import sys
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import torch

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.corpus import build_tokenizer, format_stimulus_prompt, format_motor_response, encode_no_special
from src.spheroid_v5 import PRESETS_V5
from src.cloud_sync import (
    push_checkpoint,
    pull_checkpoint,
    push_remote_loss_history,
    fetch_remote_loss_history,
    check_remote_manifest,
    detect_source_environment,
    push_remote_token_tracker,
)
from src.jax.v5_model import (
    ModelConfigV5,
    init_params_v5,
    training_rollout,
    emit_step,
    encode_sensory,
    rms_norm,
)
from src.jax.checkpoint import adapt_and_load_checkpoint_weights, discover_latest_checkpoint, AsyncCheckpointManager
from src.jax.auto_tuner import resolve_jax_compute_dtype, run_auto_tuner
from src.jax.data_streamer import LinguisticDataStreamer
from src.jax.spheroid_optimizer import init_spheroid_opt_state, build_spheroid_train_engine
from src.elastic_window import (
    build_online_dataset_iterator,
    build_canonical_eval_split,
    get_default_elastic_buckets,
    ElasticBucketBatcher,
    PrefetchedBatcher,
)


def parse_sft_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CerebroSpheroid V5 Supervised Fine-Tuning (SFT) & Conversational Alignment Engine"
    )
    parser.add_argument("--preset", type=str, default="auto", choices=["auto", "v5_small", "v5_compact", "v5_base", "tpu_v5e_1", "tpu_v6e_1"], help="Architecture preset")
    parser.add_argument("--sft_model_name", "--sft-model-name", dest="sft_model_name", type=str, default=None, help="Cloud sync and checkpoint model name (default: {preset}_sft)")
    parser.add_argument("--pretrained_checkpoint", "--pretrained-checkpoint", dest="pretrained_checkpoint", type=str, default=None, help="Path to pre-trained base checkpoint (auto-discovered if None)")
    parser.add_argument("--resume_sft", "--resume-sft", dest="resume_sft", action="store_true", default=False, help="Resume an in-progress SFT run from {preset}_sft_latest.pt")
    parser.add_argument("--fresh_sft", "--fresh-sft", dest="fresh_sft", action="store_true", default=False, help="Force fresh SFT initialization from foundation pre-trained base model, ignoring any existing SFT checkpoints")
    parser.add_argument("--total_steps", "--total-steps", dest="total_steps", type=int, default=50_000, help="Total SFT steps (default: 50,000)")
    parser.add_argument("--target_tokens", "--target-tokens", dest="target_tokens", type=int, default=None, help="Optional token ingestion budget for SFT")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=None, help="Micro-batch size per device (auto-selected via hardware config if None)")
    parser.add_argument("--effective_batch_size", "--effective-batch-size", dest="effective_batch_size", type=int, default=256, help="Target global effective batch size (default: 256 for TPU, 128 for GPU; automatically computes grad_accum)")
    parser.add_argument("--max_batch_size", "--max-batch-size", dest="max_batch_size", type=int, default=128, help="Cap on auto-tuned micro-batch size (default: 128)")
    parser.add_argument("--grad_accum", "--grad-accum", dest="grad_accum", type=int, default=1, help="Gradient accumulation micro-steps per optimizer update")
    parser.add_argument("--prompt_len", "--prompt-len", dest="prompt_len", type=int, default=512, help="Sensory context prompt length (default: 512 tokens)")
    parser.add_argument("--seq_len", "--seq-len", dest="seq_len", type=int, default=1024, help="Motor action target length (default: 1024 tokens)")
    parser.add_argument("--lr", type=float, default=3.0e-5, help="Peak SFT learning rate (default: 3.0e-5)")
    parser.add_argument("--min_lr", "--min-lr", dest="min_lr", type=float, default=3.0e-6, help="SFT floor learning rate (default: 3.0e-6)")
    parser.add_argument("--warmup_steps", "--warmup-steps", dest="warmup_steps", type=int, default=500, help="Linear warmup steps (default: 500)")
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=0.01, help="SFT weight decay (default: 0.01)")
    parser.add_argument("--repetition_penalty", "--repetition-penalty", dest="repetition_penalty", type=float, default=1.20, help="Motor action repetition penalty (default: 1.20)")
    parser.add_argument("--save_interval", "--save-interval", dest="save_interval", type=int, default=250, help="Checkpoint save step interval (default: 250)")
    parser.add_argument("--eval_interval", "--eval-interval", dest="eval_interval", type=int, default=100, help="Validation step interval (default: 100)")
    parser.add_argument("--probe_interval", "--probe-interval", dest="probe_interval", type=int, default=500, help="Live generation probe step interval (default: 500)")
    parser.add_argument("--checkpoint_dir", "--checkpoint-dir", dest="checkpoint_dir", type=str, default="/content/checkpoints" if os.path.exists("/content") else "checkpoints/sft", help="Checkpoint storage directory")
    parser.add_argument("--scratch_dir", dest="scratch_dir", type=str, default="/tmp/cerebro_v5_sft", help="Fast local disk for staging")
    parser.add_argument("--cloud_sync", "--cloud-sync", dest="cloud_sync", action=argparse.BooleanOptionalAction, default=True, help="Live bi-directional sync with bioflo.in")
    parser.add_argument("--cloud_url", dest="cloud_url", type=str, default="https://bioflo.in/api/cerebro/sync.php", help="bioflo.in sync endpoint")
    parser.add_argument("--cloud_token", dest="cloud_token", type=str, default="CerebroSpheroid_Sync_Secure_2026", help="Sync token")
    parser.add_argument("--sampling_p", "--sampling-p", dest="sampling_p", type=float, default=1.0, help="Sampling probability for deliberation (1.0 for teacher forcing, 0.5 for scheduled deliberation)")
    parser.add_argument("--deliberation_mode", dest="deliberation_mode", type=str, default="teacher_forcing", choices=["teacher_forcing", "scheduled", "pitfall_recovery", "internal"], help="Deliberation mode")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Matrix compute dtype")
    parser.add_argument("--remat", type=str, default="false", choices=["true", "false", "auto"], help="Activation rematerialization (default: false for peak throughput)")
    parser.add_argument("--auto_tune", "--auto-tune", dest="auto_tune", action=argparse.BooleanOptionalAction, default=True, help="Verify hardware config with AutoTuner or reuse cached optimal config")
    parser.add_argument("--force_tune", "--force-tune", dest="force_tune", action=argparse.BooleanOptionalAction, default=False, help="Re-benchmark batch ladder even if cached config exists")
    parser.add_argument("--tune_objective", "--tune-objective", dest="tune_objective", type=str, default="tok_per_sec", choices=["tok_per_sec", "balanced", "steps_per_sec"], help="AutoTuner objective")
    parser.add_argument("--target_eval_loss", "--target-eval-loss", dest="target_eval_loss", type=float, default=0.60, help="Target held-out evaluation loss for convergence (default: 0.60)")
    parser.add_argument("--consistent_eval_threshold", "--consistent-eval-threshold", dest="consistent_eval_threshold", type=int, default=3, help="Consecutive evaluations required <= target-eval-loss to confirm convergence (default: 3)")
    parser.add_argument("--dataset_source", "--data-source", dest="dataset_source", type=str, default="chat_thinking", help="SFT streaming dataset mixture ('chat_thinking' for database-augmented chat, CoT, math, SQL, code)")
    parser.add_argument("--loss_window_governor", "--loss-window-governor", dest="loss_window_governor", action=argparse.BooleanOptionalAction, default=True, help="Automatically damp LR when loss climbs")
    parser.add_argument("--loss_ceiling", "--loss-ceiling", dest="loss_ceiling", type=float, default=2.50, help="Loss ceiling threshold for governor")
    parser.add_argument("--loss_floor", "--loss-floor", dest="loss_floor", type=float, default=0.40, help="Loss floor threshold for governor")

    return parser.parse_args(args_list)


# Interactive test prompts for live generation probing during SFT
SFT_PROBE_PROMPTS = [
    "hi",
    "what is earth?",
    "python script to sort a list",
    "What is 15 + 27?",
    "what is a pixel?",
    "what is the capital of Germany?",
    "Write a clean Python function that checks if a given string is a palindrome:",
    "select all users from database where age > 21",
    "If Tom has 15 books and gives 4 to Alice and 3 to Bob, how many books does Tom have left?",
]


def run_live_probe(
    params: Dict[str, jnp.ndarray],
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: ModelConfigV5,
    tokenizer: Any,
    prompt_len: int,
    gen_len: int = 128,
):
    """Executes live greedy autoregressive rollout on test prompts to inspect conversational alignment."""
    print("\n" + "-" * 76)
    print("🔬 [SFT Live Conversational & Reasoning Probe]")
    print("-" * 76)

    for idx, p_text in enumerate(SFT_PROBE_PROMPTS, 1):
        formatted_prompt = format_stimulus_prompt(p_text)
        if hasattr(tokenizer, "encode"):
            raw_ids = encode_no_special(tokenizer, formatted_prompt, max_length=prompt_len)
        else:
            raw_ids = [1] * min(len(formatted_prompt.split()), prompt_len)

        p_padded = raw_ids[:prompt_len] if len(raw_ids) >= prompt_len else raw_ids + [0] * (prompt_len - len(raw_ids))
        p_arr = jnp.asarray([p_padded], dtype=jnp.int32)

        try:
            # Autoregressive generation rollout (targets=None, feedback_tokens=None)
            logits_out = training_rollout(
                params=params,
                prompts=p_arr,
                target_len=gen_len,
                buffers=buffers,
                config=config,
                feedback_tokens=None,
                targets=None,
                remat=False,
                repetition_penalty=1.20,
            )
            gen_ids = np.asarray(jnp.argmax(logits_out[0], axis=-1))

            eos_id = getattr(tokenizer, "eos_token_id", 2) or 2
            valid_ids = []
            for t_id in gen_ids:
                if int(t_id) == eos_id:
                    break
                valid_ids.append(int(t_id))

            if hasattr(tokenizer, "decode"):
                raw_text = tokenizer.decode(valid_ids).strip()
            else:
                raw_text = f"[Tokens: {len(valid_ids)}]"

            # Clean output display using 3-tier cognitive parser
            from src.corpus import parse_model_output
            parsed = parse_model_output(raw_text)

            print(f"▶ Q{idx}: {p_text}")
            if parsed.get("thinking"):
                print(f"  💭 Thought: {parsed['thinking']}")
            if parsed.get("tool_calls"):
                print(f"  ⚡ Action : {parsed['tool_calls']}")
            ans_display = parsed.get("answer") or parsed.get("raw") or "..."
            print(f"  🧠 Answer : {ans_display}\n")
        except Exception as e:
            print(f"  [Probe notice for Q{idx}: {e}]")

    print("-" * 76 + "\n")


def main(args=None):
    if args is None:
        args = parse_sft_args()

    # Graceful shutdown handling
    shutdown_requested = False

    def sig_handler(sig, frame):
        nonlocal shutdown_requested
        print("\n🛑 [SFT Shutdown] Received termination signal. Flushing checkpoint safely...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    devices = jax.devices()
    num_devices = len(devices)
    dev_kind = devices[0].device_kind.lower() if devices else ""
    backend = jax.default_backend().lower()
    is_tpu = "tpu" in dev_kind or "tpu" in backend

    # Hardware & Preset Resolution
    if args.preset == "auto":
        if "v6" in dev_kind or "trillium" in dev_kind:
            args.preset = "tpu_v6e_1"
        elif "v5" in dev_kind or is_tpu:
            args.preset = "tpu_v5e_1"
        else:
            args.preset = "v5_small"

    base_preset = "v5_small" if args.preset in ("tpu_v5e_1", "tpu_v6e_1") else args.preset
    sft_model_name = args.sft_model_name or f"{base_preset}_sft"

    # Hardware Auto-Tuning & Micro-Batch Optimization
    remat_choice = True if str(args.remat).lower() == "true" else False
    max_bs = getattr(args, "max_batch_size", None) or 128

    if getattr(args, "auto_tune", True):
        best_cfg = run_auto_tuner(
            preset=args.preset,
            seq_len=args.seq_len,
            dtype=args.dtype,
            use_cached_config=not getattr(args, "force_tune", False),
            force_tune=getattr(args, "force_tune", False),
            upload_winner=True,
            cloud_url=args.cloud_url,
            cloud_token=args.cloud_token,
            verbose=True,
            tune_objective=getattr(args, "tune_objective", "tok_per_sec"),
            prompt_len=args.prompt_len,
            max_batch_size=max_bs,
        )
        args.batch_size = best_cfg["batch_size"]
        remat_choice = best_cfg["remat"]
        print(f"\n⚡ Applying Tuned Hyperparameters for SFT: batch_size={args.batch_size}, remat={remat_choice}")
        if "tok_per_sec" in best_cfg and best_cfg["tok_per_sec"] > 0:
            m_type = best_cfg.get("memory_type", "HBM" if is_tpu else "VRAM")
            v_gb = best_cfg.get("memory_gb", best_cfg.get("vram_gb", 0))
            v_pct = best_cfg.get("memory_utilization_pct", best_cfg.get("vram_utilization_pct", 0))
            sps = best_cfg.get("steps_per_sec", 0.0)
            sps_str = f"{sps:.2f} steps/s | " if sps and sps > 0 else ""
            print(f"⚡ Target Benchmark Speed: {sps_str}{best_cfg['tok_per_sec']:,.0f} tok/s ({m_type}: {v_gb:.1f} GB, {v_pct:.1f}% utilization)")
    elif args.batch_size is None:
        if is_tpu:
            args.batch_size = 128
        else:
            args.batch_size = 16
    elif max_bs is not None:
        args.batch_size = min(int(args.batch_size), int(max_bs))

    # Effective Batch Size & Gradient Accumulation
    eff_target = getattr(args, "effective_batch_size", None)
    if eff_target is not None and eff_target > 0:
        if args.batch_size > eff_target:
            args.batch_size = eff_target
        args.grad_accum = max(1, eff_target // max(1, args.batch_size))
        eff_bs = args.batch_size * args.grad_accum
        print(f"⚡ [SFT Batch Scaling] Target Effective Batch Size: {eff_bs} (Micro Batch: {args.batch_size} × Grad Accum: {args.grad_accum})")
    else:
        args.grad_accum = getattr(args, "grad_accum", 1)
        eff_bs = args.batch_size * args.grad_accum

    eff_batch_size = eff_bs
    tokens_per_opt_step = eff_batch_size * (args.prompt_len + args.seq_len)
    if getattr(args, "target_tokens", None) is not None and args.target_tokens > 0:
        calculated_steps = int(math.ceil(float(args.target_tokens) / max(1.0, float(tokens_per_opt_step))))
        args.total_steps = calculated_steps
        print(f"🎯 [Target Token Budget] {args.target_tokens/1e9:.2f} Billion Tokens -> Scheduled for {args.total_steps:,} Steps (at {tokens_per_opt_step:,} tok/step)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.scratch_dir, exist_ok=True)

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/tmp/hf_cache"
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    if args.sampling_p < 1.0 - 1e-4 and args.deliberation_mode == "teacher_forcing":
        args.deliberation_mode = "scheduled"

    print("=" * 84)
    print("🧠 CEREBROSPHEROID V5: CONVERSATIONAL & REASONING SFT ENGINE")
    print("=" * 84)
    print(f"  Architecture Preset : {args.preset} (Base: {base_preset})")
    print(f"  SFT Model Registry  : {sft_model_name}")
    print(f"  Total SFT Steps     : {args.total_steps:,}")
    print(f"  Batch Configuration : Micro Batch {args.batch_size} × Grad Accum {args.grad_accum} = Effective Batch {eff_batch_size}")
    print(f"  Context Budget      : Prompt={args.prompt_len} tok | Target={args.seq_len} tok")
    print(f"  Deliberation Policy : Mode={args.deliberation_mode} | Sampling P={args.sampling_p:.2f}")
    print(f"  Learning Rate Decay : {args.lr:g} -> {args.min_lr:g} ({args.warmup_steps} step warmup)")
    print(f"  Convergence Target  : Held-out Loss <= {args.target_eval_loss:.4f} (Consecutive threshold: {args.consistent_eval_threshold})")
    print(f"  Weight Decay        : {args.weight_decay:g}")
    print(f"  Cloud Sync          : {args.cloud_sync} (Endpoint: {args.cloud_url})")
    print(f"  Checkpoints Dir     : {args.checkpoint_dir}")
    print("=" * 84)

    # 1. Model Configuration & Memory Buffers
    config_dict = PRESETS_V5.get(base_preset, PRESETS_V5["v5_small"]).copy()
    config = ModelConfigV5(**config_dict)
    target_dtype, dtype_str = resolve_jax_compute_dtype(args.dtype)

    init_rng = jax.random.PRNGKey(42)
    params, buffers = init_params_v5(init_rng, config=config)
    params = {k: v.astype(target_dtype) for k, v in params.items()}

    # 2. Base Checkpoint Loading or SFT Resumption
    start_step = 0
    best_eval_loss = float("inf")
    loaded_source = None

    # Candidate paths for resuming an in-progress SFT run
    sft_candidates = [
        args.pretrained_checkpoint if (args.pretrained_checkpoint and "sft" in os.path.basename(args.pretrained_checkpoint)) else None,
        os.path.join(args.checkpoint_dir, f"{sft_model_name}_latest.pt"),
        os.path.join(args.checkpoint_dir, f"{sft_model_name}_best.pt"),
        os.path.join(args.checkpoint_dir, f"{base_preset}_sft_latest.pt"),
        os.path.join(args.checkpoint_dir, f"{base_preset}_sft_best.pt"),
        os.path.join(args.checkpoint_dir, "v5_small_sft_latest.pt"),
        os.path.join(args.checkpoint_dir, "v5_small_sft_best.pt"),
        os.path.join(REPO_ROOT, "checkpoints", f"{sft_model_name}_latest.pt"),
        os.path.join(REPO_ROOT, "checkpoints", f"{sft_model_name}_best.pt"),
        os.path.join(REPO_ROOT, "checkpoints", f"{base_preset}_sft_latest.pt"),
        os.path.join(REPO_ROOT, "checkpoints", f"{base_preset}_sft_best.pt"),
        os.path.join(REPO_ROOT, "checkpoints", "v5_small_sft_master_diverse.pt"),
        os.path.join(REPO_ROOT, "checkpoints", "v5_small_sft_best.pt"),
    ]
    sft_resume_path = next((p for p in sft_candidates if p and os.path.exists(p)), None)

    if getattr(args, "fresh_sft", False):
        args.resume_sft = False
        sft_resume_path = None
        print("🌱 [Option A] Fresh SFT requested: discarding any existing SFT checkpoints and loading pre-trained foundation weights.")

    # If resume_sft is requested but not found locally, attempt remote pull from bioflo.in
    if args.resume_sft and not sft_resume_path and args.cloud_sync:
        print(f"\n📥 SFT checkpoint not found locally. Checking remote manifest for [{sft_model_name}] on bioflo.in...")
        try:
            manifest = check_remote_manifest(url=args.cloud_url, token=args.cloud_token)
            models_dict = manifest.get("models", {})
            target_remote_sft = None
            if sft_model_name in models_dict:
                target_remote_sft = sft_model_name
            elif f"{base_preset}_sft" in models_dict:
                target_remote_sft = f"{base_preset}_sft"
            elif "v5_small_sft" in models_dict:
                target_remote_sft = "v5_small_sft"

            if target_remote_sft:
                dest_file = os.path.join(args.checkpoint_dir, f"{sft_model_name}_latest.pt")
                print(f"🔄 [cloud_sync] Pulling remote SFT checkpoint [{target_remote_sft}] to {dest_file}...")
                success = pull_checkpoint(
                    local_dest=dest_file,
                    target_type="latest",
                    model=target_remote_sft,
                    url=args.cloud_url,
                    token=args.cloud_token,
                )
                if success and os.path.exists(dest_file):
                    sft_resume_path = dest_file
        except Exception as e:
            print(f"⚠️ Remote SFT pull notice: {e}")

    if args.resume_sft and sft_resume_path:
        print(f"\n🔄 [SFT Resume] Resuming in-progress SFT from: {sft_resume_path}")
        ckpt = torch.load(sft_resume_path, map_location="cpu", weights_only=False)
        start_step = int(ckpt.get("step", 0))
        best_eval_loss = float(ckpt.get("eval_loss", float("inf")))
        restored = adapt_and_load_checkpoint_weights(ckpt, params, target_dtype, config)
        print(f"✓ Resumed SFT Step {start_step:,} (Eval Loss: {best_eval_loss:.4f}, Restored: {restored}/{len(params)} arrays)")
        loaded_source = sft_resume_path
    else:
        # Discover or download base pre-trained foundation model
        cand_paths = [
            args.pretrained_checkpoint,
            os.path.join(args.checkpoint_dir, f"{args.preset}_latest.pt"),
            os.path.join(args.checkpoint_dir, f"{args.preset}_best.pt"),
            os.path.join(args.checkpoint_dir, f"{base_preset}_latest_step900k.pt"),
            os.path.join(args.checkpoint_dir, f"{base_preset}_latest.pt"),
            os.path.join(args.checkpoint_dir, f"{base_preset}_best.pt"),
            os.path.join(REPO_ROOT, "checkpoints", f"{base_preset}_latest_step900k.pt"),
            os.path.join(REPO_ROOT, "checkpoints", f"{base_preset}_latest.pt"),
            os.path.join(REPO_ROOT, "checkpoints", f"{base_preset}_best.pt"),
        ]
        base_ckpt_path = next((p for p in cand_paths if p and os.path.exists(p)), None)

        if not base_ckpt_path and args.cloud_sync:
            print(f"\n📥 Base pre-trained checkpoint not found locally. Inspecting remote manifest on bioflo.in...")
            try:
                manifest = check_remote_manifest(url=args.cloud_url, token=args.cloud_token)
                models_dict = manifest.get("models", {})
                target_remote_model = base_preset
                if args.preset in models_dict:
                    target_remote_model = args.preset
                elif "v5_small" in models_dict:
                    target_remote_model = "v5_small"

                dest_file = os.path.join(args.checkpoint_dir, f"{args.preset}_latest.pt")
                print(f"🔄 [cloud_sync] Downloading [{target_remote_model}] latest.pt from {args.cloud_url}...")
                success = pull_checkpoint(
                    local_dest=dest_file,
                    target_type="latest",
                    model=target_remote_model,
                    url=args.cloud_url,
                    token=args.cloud_token,
                )
                if success and os.path.exists(dest_file):
                    base_ckpt_path = dest_file
            except Exception as e:
                print(f"⚠️ Remote sync notice: {e}")

        if base_ckpt_path and os.path.exists(base_ckpt_path):
            print(f"\n🚀 Loading base pre-trained foundation weights from: {os.path.basename(base_ckpt_path)}...")
            ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
            pretrain_step = int(ckpt.get("step", 0))
            pretrain_loss = float(ckpt.get("loss", ckpt.get("eval_loss", 0.0)))
            restored = adapt_and_load_checkpoint_weights(ckpt, params, target_dtype, config)
            print(f"✅ Foundation Model Loaded: Step {pretrain_step:,} (Loss: {pretrain_loss:.4f}) | Adapted {restored}/{len(params)} arrays.")
            print("🌱 Initializing SFT Step 0 with calibrated pre-trained weights!")
            start_step = 0
            best_eval_loss = float("inf")
            loaded_source = base_ckpt_path
        else:
            print("⚠️ Warning: No pre-trained checkpoint found. Training SFT from initialized parameters.")

    # 3. Tokenizer Verification
    tokenizer = build_tokenizer()
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    # 4. Asynchronous Checkpoint Manager
    ckpt_mgr = AsyncCheckpointManager(
        local_dir=args.scratch_dir,
        target_dir=args.checkpoint_dir,
        cloud_sync=args.cloud_sync,
        cloud_url=args.cloud_url,
        cloud_token=args.cloud_token,
        min_cloud_upload_interval_sec=120.0,
    )

    # 5. Spheroid SFT Train Step Engine
    compute_grad_step, apply_opt_step = build_spheroid_train_engine(
        config,
        buffers,
        pad_id=pad_id,
        weight_decay=args.weight_decay,
        repetition_penalty=args.repetition_penalty,
        remat=remat_choice,
        deliberation_mode=args.deliberation_mode,
    )

    @jax.jit
    def sft_single_step_fn(p, opt_s, q_batch, r_batch, cur_lr, feedback_tokens=None, sampling_p=1.0):
        loss_val, grads = compute_grad_step(
            p, q_batch, r_batch, feedback_tokens=feedback_tokens, sampling_p=sampling_p, deliberation_mode=args.deliberation_mode
        )
        new_p, new_opt_s = apply_opt_step(p, opt_s, grads, cur_lr)
        return new_p, new_opt_s, loss_val

    # Initialize fresh Spheroid optimizer state
    opt_state = init_spheroid_opt_state(params)

    # 6. SFT Multi-Corpus Data Ingestion & Prefetching
    print(f"\n📦 Initializing Curated Reasoning & Conversational SFT Streamer (Source: {args.dataset_source})...")
    sft_data_iter = build_online_dataset_iterator(dataset_source=args.dataset_source, total_steps=args.total_steps)
    first_row = next(sft_data_iter)
    import itertools
    sft_chain_iter = itertools.chain([first_row], sft_data_iter)

    buckets = [("sft_static", args.prompt_len, args.seq_len)]
    batcher = ElasticBucketBatcher(
        tokenizer, sft_chain_iter, batch_size=args.batch_size, total_window=args.prompt_len + args.seq_len, buckets=buckets, framework="numpy", return_feedback=True
    )
    prefetched_sft = PrefetchedBatcher(batcher, queue_size=16)

    def _unpack_batch(batch_tuple):
        if len(batch_tuple) == 5:
            return batch_tuple[2], batch_tuple[3], batch_tuple[4]
        if len(batch_tuple) == 4:
            return batch_tuple[2], batch_tuple[3], None
        if len(batch_tuple) == 3:
            return batch_tuple[0], batch_tuple[1], batch_tuple[2]
        return batch_tuple[0], batch_tuple[1], None

    # 7. Canonical SFT Evaluation Split
    eval_batches, eval_domains = [], []
    try:
        eval_batches, eval_domains = build_canonical_eval_split(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            target_len=args.seq_len,
            target_pad_id=-100,
            pad_id=pad_id,
        )
        print(f"✅ Canonical SFT Evaluation Split: {len(eval_batches)} domains prepared.")
    except Exception as e:
        print(f"⚠️ Canonical validation setup notice: {e}")

    @jax.jit
    def eval_step_loss(p, q_b, r_b, fb_b=None):
        return training_rollout(
            params=p,
            prompts=q_b,
            target_len=r_b.shape[1],
            buffers=buffers,
            config=config,
            feedback_tokens=fb_b,
            feedback_is_shifted=True,
            pad_id=pad_id,
            targets=r_b,
            remat=False,
            pure_teacher_forcing=True,
        )

    # 8. Learning Rate Schedule for SFT
    def get_sft_lr(step_idx: int) -> float:
        if step_idx < args.warmup_steps:
            return float(args.lr * (float(step_idx) / max(1.0, float(args.warmup_steps))))
        progress = float(step_idx - args.warmup_steps) / max(1.0, float(args.total_steps - args.warmup_steps))
        ratio = min(1.0, max(0.0, progress))
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return float(args.min_lr + (args.lr - args.min_lr) * coeff)

    # 9. Main SFT Training Loop
    print("\n" + "=" * 84)
    print(f"🚀 Starting SFT Training ({start_step:,} -> {args.total_steps:,} Steps)")
    print(f"   • Peak SFT LR      : {args.lr:g} (cooldown floor: {args.min_lr:g})")
    print(f"   • Effective Batch  : {eff_batch_size} (Micro: {args.batch_size} × Grad Accum: {args.grad_accum})")
    print(f"   • Prompt / Target  : {args.prompt_len} tok / {args.seq_len} tok (Prompt tokens masked: -100)")
    print(f"   • Live Probing     : Every {args.probe_interval} steps")
    print(f"   • Cloud Registry   : {sft_model_name}")
    print("=" * 84 + "\n")

    step = start_step
    t_start = time.time()
    t_last_log = time.time()
    step_window_tokens = 0
    step_window_count = 0
    cumulative_tokens = 0
    last_eval_loss = float("inf")
    ema_loss: Optional[float] = None
    consecutive_target_evals = 0
    converged_target_met = False

    try:
        while step < args.total_steps and not shutdown_requested:
            prev_step = step
            step += 1

            cur_lr = get_sft_lr(step)

            # SFT Loss Window Governor Dynamic Damping
            if getattr(args, "loss_window_governor", True) and ema_loss is not None:
                if ema_loss > args.loss_ceiling:
                    alpha_gov = min(1.0, (ema_loss - args.loss_ceiling) / 0.5)
                elif ema_loss < args.loss_floor:
                    alpha_gov = -min(0.5, (args.loss_floor - ema_loss) / 0.5)
                else:
                    alpha_gov = 0.0
            else:
                alpha_gov = 0.0

            if alpha_gov > 0.0:
                cur_lr = cur_lr * (1.0 - 0.4 * alpha_gov)

            # SFT Optimization Step with Multi-Step Gradient Accumulation
            if args.grad_accum > 1:
                acc_grads = None
                loss_acc = 0.0
                step_tokens = 0
                inv_accum = 1.0 / float(args.grad_accum)
                for _ in range(args.grad_accum):
                    b_tup = prefetched_sft.get_batch()
                    prompt_b, target_b, feedback_b = _unpack_batch(b_tup)
                    g_loss, micro_grads = compute_grad_step(
                        params, prompt_b, target_b, feedback_tokens=feedback_b,
                        sampling_p=args.sampling_p, deliberation_mode=args.deliberation_mode
                    )
                    acc_grads = (
                        micro_grads
                        if acc_grads is None
                        else jax.tree_util.tree_map(lambda a, b: a + b, acc_grads, micro_grads)
                    )
                    loss_acc = loss_acc + g_loss
                    step_tokens += prompt_b.shape[0] * (prompt_b.shape[1] + target_b.shape[1])
                acc_grads = jax.tree_util.tree_map(lambda a: a * inv_accum, acc_grads)
                loss_val = float(loss_acc * inv_accum)
                params, opt_state = apply_opt_step(params, opt_state, acc_grads, cur_lr)
            else:
                b_tup = prefetched_sft.get_batch()
                prompt_b, target_b, feedback_b = _unpack_batch(b_tup)
                params, opt_state, loss = sft_single_step_fn(
                    params,
                    opt_state,
                    prompt_b,
                    target_b,
                    cur_lr,
                    feedback_tokens=feedback_b,
                    sampling_p=args.sampling_p,
                )
                loss_val = float(loss)
                step_tokens = prompt_b.shape[0] * (prompt_b.shape[1] + target_b.shape[1])

            # Divergence Watchdog
            if math.isfinite(loss_val) and loss_val <= 4.5:
                ema_loss = loss_val if ema_loss is None else (0.95 * ema_loss + 0.05 * loss_val)
            else:
                print(f"\n⚠️ [SFT Divergence Watchdog] Non-finite or elevated loss ({loss_val}) at Step {step:,}! Stabilizing optimizer momentum...")
                ema_loss = last_eval_loss if math.isfinite(last_eval_loss) else 2.50
                cur_lr = cur_lr * 0.75

            step_window_tokens += step_tokens
            step_window_count += 1
            cumulative_tokens += step_tokens

            # Console Progress
            if (prev_step == start_step) or (step % 10 == 0):
                elapsed_log = time.time() - t_last_log
                steps_per_sec = step_window_count / max(elapsed_log, 1e-6)
                tok_per_sec = step_window_tokens / max(elapsed_log, 1e-6)
                gov_str = f" | Gov: {alpha_gov:.2f}" if alpha_gov > 0.01 else ""
                print(
                    f"[SFT Step {step:5d}/{args.total_steps}] "
                    f"Loss: {loss_val:6.4f} | "
                    f"EMA: {ema_loss:6.4f} | "
                    f"Steps/s: {steps_per_sec:5.2f} | "
                    f"Tok/s: {tok_per_sec:7,.0f} | "
                    f"LR: {cur_lr:.2e}{gov_str} | "
                    f"Elapsed: {time.time() - t_start:5.1f}s"
                )
                t_last_log = time.time()
                step_window_tokens = 0
                step_window_count = 0

            # Canonical SFT Evaluation Split
            crossed_eval = bool(eval_batches) and (step % args.eval_interval == 0)
            if crossed_eval:
                val_losses = [float(eval_step_loss(params, vp, vt, vfb)) for vp, vt, vfb in eval_batches]
                eval_loss = float(np.mean(val_losses)) if val_losses else float("inf")
                last_eval_loss = eval_loss
                eval_ppl = math.exp(min(eval_loss, 20.0))
                is_champion = eval_loss < best_eval_loss

                champ_tag = "🏆 (NEW SFT CHAMPION)" if is_champion else ""
                print(f"📊 [SFT Eval @ Step {step:,}] Composite Loss: {eval_loss:.4f} | PPL: {eval_ppl:.2f} {champ_tag}")
                if len(val_losses) == len(eval_domains):
                    for d_name, d_val in zip(eval_domains, val_losses):
                        d_ppl = math.exp(min(d_val, 20.0))
                        print(f"   ├─ {d_name:<26}: Loss {d_val:.4f} (PPL: {d_ppl:.2f})")

                # Convergence detection
                if eval_loss <= args.target_eval_loss:
                    consecutive_target_evals += 1
                    print(f"🎯 [SFT Convergence] Step {step:,} eval loss {eval_loss:.4f} <= target {args.target_eval_loss:.4f} ({consecutive_target_evals}/{args.consistent_eval_threshold})")
                    if consecutive_target_evals >= args.consistent_eval_threshold:
                        converged_target_met = True
                        print(f"🏆 [SFT Converged] Reached target held-out loss {args.target_eval_loss:.4f} consistently across {consecutive_target_evals} evaluations!")
                else:
                    consecutive_target_evals = 0

                if is_champion or converged_target_met:
                    if is_champion:
                        best_eval_loss = eval_loss
                    best_fname = f"{sft_model_name}_best.pt"
                    meta_best = {
                        "step": step,
                        "eval_loss": eval_loss,
                        "loss": float(loss_val),
                        "ema_loss": ema_loss if ema_loss is not None else float(loss_val),
                        "ppl": eval_ppl,
                        "preset": base_preset,
                        "model": sft_model_name,
                        "sync_model": sft_model_name,
                        "cumulative_tokens": cumulative_tokens,
                        "effective_batch_size": eff_batch_size,
                        "tokens_by_dataset": batcher.get_tokens_by_dataset() if hasattr(batcher, "get_tokens_by_dataset") else {},
                        "converged": bool(converged_target_met),
                        "timestamp": time.time(),
                    }
                    ckpt_mgr.dispatch(params, buffers, best_fname, meta_dict=meta_best, block_if_busy=True)

                    if converged_target_met:
                        conv_fname = f"{sft_model_name}_converged.pt"
                        ckpt_mgr.dispatch(params, buffers, conv_fname, meta_dict=meta_best, block_if_busy=True)

                    if args.cloud_sync:
                        try:
                            champ_rec = {
                                "step": step,
                                "loss": float(loss_val),
                                "eval_loss": eval_loss,
                                "status": "CONVERGED_SFT_CHAMPION" if converged_target_met else "NEW_SFT_CHAMPION",
                                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "source": detect_source_environment(),
                            }
                            push_remote_loss_history(
                                model=sft_model_name,
                                records=[champ_rec],
                                url=args.cloud_url,
                                token=args.cloud_token,
                            )
                        except Exception:
                            pass

            # Live Conversational & Reasoning Generation Probe
            if step % args.probe_interval == 0:
                run_live_probe(
                    params=params,
                    buffers=buffers,
                    config=config,
                    tokenizer=tokenizer,
                    prompt_len=args.prompt_len,
                    gen_len=128,
                )

            # Checkpoint Saving & Remote Token Telemetry
            if step % args.save_interval == 0:
                latest_fname = f"{sft_model_name}_latest.pt"
                meta_latest = {
                    "step": step,
                    "loss": float(loss_val),
                    "ema_loss": ema_loss if ema_loss is not None else float(loss_val),
                    "eval_loss": last_eval_loss if math.isfinite(last_eval_loss) else float(loss_val),
                    "best_eval_loss": best_eval_loss,
                    "preset": base_preset,
                    "model": sft_model_name,
                    "sync_model": sft_model_name,
                    "cumulative_tokens": cumulative_tokens,
                    "effective_batch_size": eff_batch_size,
                    "tokens_by_dataset": batcher.get_tokens_by_dataset() if hasattr(batcher, "get_tokens_by_dataset") else {},
                    "converged": bool(converged_target_met),
                    "timestamp": time.time(),
                }
                ckpt_mgr.dispatch(params, buffers, latest_fname, meta_dict=meta_latest)

                if args.cloud_sync:
                    try:
                        latest_rec = {
                            "step": step,
                            "loss": float(loss_val),
                            "eval_loss": last_eval_loss if math.isfinite(last_eval_loss) else None,
                            "status": "SFT_ACTIVE",
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "source": detect_source_environment(),
                        }
                        push_remote_loss_history(
                            model=sft_model_name,
                            records=[latest_rec],
                            url=args.cloud_url,
                            token=args.cloud_token,
                        )

                        # Push live token ingestion metrics & dataset breakdown
                        ds_stats = batcher.get_tokens_by_dataset() if hasattr(batcher, "get_tokens_by_dataset") else {}
                        push_remote_token_tracker(
                            model=sft_model_name,
                            cumulative_tokens=cumulative_tokens,
                            target_tokens=getattr(args, "target_tokens", None) or (args.total_steps * eff_batch_size * (args.prompt_len + args.seq_len)),
                            tokens_by_dataset=ds_stats,
                            effective_batch_size=eff_batch_size,
                            step=step,
                            loss=float(loss_val),
                            tok_per_sec=float(tok_per_sec) if "tok_per_sec" in locals() else 0.0,
                            url=args.cloud_url,
                            token=args.cloud_token,
                        )
                    except Exception:
                        pass

    except Exception as e:
        print(f"\n⚠️ Encountered SFT exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n💾 Flushing final SFT checkpoint state for Step {step:,}...")
        final_fname = f"{sft_model_name}_latest.pt"
        final_meta = {
            "step": step,
            "loss": float(loss_val) if "loss_val" in locals() else 0.0,
            "ema_loss": ema_loss if ema_loss is not None else 0.0,
            "eval_loss": best_eval_loss,
            "preset": base_preset,
            "model": sft_model_name,
            "sync_model": sft_model_name,
            "cumulative_tokens": cumulative_tokens if "cumulative_tokens" in locals() else 0,
            "effective_batch_size": eff_batch_size if "eff_batch_size" in locals() else 256,
            "tokens_by_dataset": batcher.get_tokens_by_dataset() if "batcher" in locals() and hasattr(batcher, "get_tokens_by_dataset") else {},
            "timestamp": time.time(),
        }
        ckpt_mgr.flush(params, buffers, final_fname, meta_dict=final_meta)
        ckpt_mgr.shutdown(wait=True)
        if prefetched_sft is not None:
            prefetched_sft.shutdown()

        if args.cloud_sync and "batcher" in locals() and hasattr(batcher, "get_tokens_by_dataset"):
            try:
                push_remote_token_tracker(
                    model=sft_model_name,
                    cumulative_tokens=cumulative_tokens if "cumulative_tokens" in locals() else 0,
                    target_tokens=getattr(args, "target_tokens", None) or (args.total_steps * (eff_batch_size if "eff_batch_size" in locals() else 256) * (args.prompt_len + args.seq_len)),
                    tokens_by_dataset=batcher.get_tokens_by_dataset(),
                    effective_batch_size=eff_batch_size if "eff_batch_size" in locals() else 256,
                    step=step,
                    loss=float(loss_val) if "loss_val" in locals() else 0.0,
                    tok_per_sec=float(tok_per_sec) if "tok_per_sec" in locals() else 0.0,
                    url=args.cloud_url,
                    token=args.cloud_token,
                )
            except Exception:
                pass

        print(f"🎉 SFT session successfully completed at Step {step:,}!")


def run_sft_training(**kwargs):
    """Programmatic entrypoint for notebooks and scripts."""
    args = parse_sft_args([])
    for k, v in kwargs.items():
        if hasattr(args, k):
            setattr(args, k, v)
        elif k == "checkpoint_dir":
            setattr(args, "checkpoint_dir", v)
        elif k == "max_steps":
            setattr(args, "total_steps", v)
        elif k == "batch_size":
            setattr(args, "batch_size", v)
    return main(args)


if __name__ == "__main__":
    main()
