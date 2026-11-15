#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Supervised Fine-Tuning (SFT) & Conversational Alignment Engine
=====================================================================================
Specialized instruction-tuning and conversational alignment engine for CerebroSpheroid V5:
  • Base Checkpoint Loading : Automatically initializes from the 900k pre-trained champion weights.
  • Instruction Loss Masking: Strict -100 target padding on sensory stimuli (<stim>...</stim>);
                              loss is backpropagated solely on motor articulations (<act>...</act>).
  • Curated SFT Data Blend   :
      - 30% SmolTalk Everyday Conversations & OpenHermes-100k (Instruction following & STEM)
      - 25% Smol-Magpie-Ultra & Apigen-80k (Multi-turn deductive reasoning & agentic tool calls)
      - 20% GSM8K & MetaMathQA (Step-by-step mathematical reasoning with #### anchors)
      - 15% SmolTalk Self-OSS-Instruct (Indented Python algorithms & software implementation)
      - 10% FineWeb-Edu 4.0 Replay (Invariant science anchor preventing catastrophic forgetting)
  • SFT Optimizer Dynamics   :
      - Spheroid Optimizer with low learning rate: 3.0e-5 -> 3.0e-6 (cosine cooldown)
      - Conservative weight decay: 0.01 (preserves foundational manifold topology)
      - 500-step smooth linear warmup
  • Continuous Live Probing  : Generates and logs actual assistant responses to test prompts every 500 steps.
  • Bi-directional Cloud Sync: Asynchronously syncs SFT latest and champion checkpoints to bioflo.in.

Usage:
  python3 training/train_v5_sft.py --preset v5_small --total-steps 50000 --lr 3e-5
  python3 training/train_v5_sft.py --preset v5_base --total-steps 50000 --lr 2.5e-5
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
from src.jax.auto_tuner import resolve_jax_compute_dtype
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
        description="CerebroSpheroid V5 Supervised Fine-Tuning (SFT) Engine"
    )
    parser.add_argument("--preset", type=str, default="auto", choices=["auto", "v5_small", "v5_compact", "v5_base", "tpu_v5e_1", "tpu_v6e_1"], help="Architecture preset")
    parser.add_argument("--sft_model_name", "--sft-model-name", dest="sft_model_name", type=str, default=None, help="Cloud sync and checkpoint model name (default: {preset}_sft)")
    parser.add_argument("--pretrained_checkpoint", "--pretrained-checkpoint", dest="pretrained_checkpoint", type=str, default=None, help="Path to pre-trained base checkpoint (auto-discovered if None)")
    parser.add_argument("--resume_sft", "--resume-sft", dest="resume_sft", action="store_true", default=False, help="Resume an in-progress SFT run from {preset}_sft_latest.pt")
    parser.add_argument("--total_steps", "--total-steps", dest="total_steps", type=int, default=50_000, help="Total SFT steps (default: 50,000)")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=None, help="Batch size per device (auto-selected if None)")
    parser.add_argument("--prompt_len", "--prompt-len", dest="prompt_len", type=int, default=256, help="Sensory context prompt length (default: 256 tokens)")
    parser.add_argument("--seq_len", "--seq-len", dest="seq_len", type=int, default=512, help="Motor action target length (default: 512 tokens)")
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
    parser.add_argument("--remat", type=str, default="true", choices=["true", "false", "auto"], help="Activation rematerialization")

    return parser.parse_args(args_list)


# Interactive test prompts for live generation probing during SFT
SFT_PROBE_PROMPTS = [
    "Explain how a neural network learns using backpropagation in simple, clear terms:",
    "Solve step-by-step: If Tom has 15 books and gives 4 to Alice and 3 to Bob, how many books does Tom have left?",
    "Write a clean Python function that checks if a given string is a palindrome:",
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
    print("\n" + "-" * 72)
    print("🔬 [SFT Live Conversational & Reasoning Probe]")
    print("-" * 72)

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
                answer = tokenizer.decode(valid_ids).strip()
            else:
                answer = f"[Tokens: {len(valid_ids)}]"

            # Clean output display
            if answer.startswith("<act>"):
                answer = answer[5:].strip()
            if answer.endswith("</act>"):
                answer = answer[:-6].strip()

            print(f"▶ Q{idx}: {p_text}")
            print(f"  A{idx}: {answer}\n")
        except Exception as e:
            print(f"  [Probe notice for Q{idx}: {e}]")

    print("-" * 72 + "\n")


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
    is_tpu = "tpu" in dev_kind

    # Hardware & Preset Resolution
    if args.preset == "auto":
        if "v6" in dev_kind or "trillium" in dev_kind:
            args.preset = "tpu_v6e_1"
        elif "v5" in dev_kind:
            args.preset = "tpu_v5e_1"
        else:
            args.preset = "v5_small"

    base_preset = "v5_small" if args.preset in ("tpu_v5e_1", "tpu_v6e_1") else args.preset
    sft_model_name = args.sft_model_name or f"{base_preset}_sft"

    if args.batch_size is None:
        # Optimal default SFT batch sizes
        if is_tpu:
            args.batch_size = 32 if "v6" in dev_kind else 16
        else:
            args.batch_size = 8

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
    print(f"  Batch Size          : {args.batch_size} per device ({num_devices} active devices)")
    print(f"  Context Budget      : Prompt={args.prompt_len} tok | Target={args.seq_len} tok")
    print(f"  Deliberation Policy : Mode={args.deliberation_mode} | Sampling P={args.sampling_p:.2f}")
    print(f"  Learning Rate Decay : {args.lr:g} -> {args.min_lr:g} ({args.warmup_steps} step warmup)")
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
    sft_latest_path = os.path.join(args.checkpoint_dir, f"{sft_model_name}_latest.pt")
    start_step = 0
    best_eval_loss = float("inf")

    loaded_source = None
    if args.resume_sft and os.path.exists(sft_latest_path):
        print(f"\n🔄 [SFT Resume] Resuming in-progress SFT from {sft_latest_path}...")
        ckpt = torch.load(sft_latest_path, map_location="cpu", weights_only=False)
        start_step = int(ckpt.get("step", 0))
        best_eval_loss = float(ckpt.get("eval_loss", float("inf")))
        restored = adapt_and_load_checkpoint_weights(ckpt, params, target_dtype, config)
        print(f"✓ Resumed SFT Step {start_step:,} (Eval Loss: {best_eval_loss:.4f}, Restored: {restored}/{len(params)} arrays)")
        loaded_source = sft_latest_path
    else:
        # Discover or download base pre-trained model
        cand_paths = [
            args.pretrained_checkpoint,
            os.path.join(args.checkpoint_dir, f"{base_preset}_latest.pt"),
            os.path.join(args.checkpoint_dir, f"{base_preset}_best.pt"),
            os.path.join(REPO_ROOT, "checkpoints", f"{base_preset}_latest.pt"),
            os.path.join(REPO_ROOT, "checkpoints", f"{base_preset}_best.pt"),
        ]
        base_ckpt_path = next((p for p in cand_paths if p and os.path.exists(p)), None)

        if not base_ckpt_path and args.cloud_sync:
            print(f"\n📥 Base pre-trained checkpoint not found locally. Pulling latest {base_preset} from bioflo.in...")
            dest_file = os.path.join(args.checkpoint_dir, f"{base_preset}_latest.pt")
            success = pull_checkpoint(
                local_dest=dest_file,
                target_type="latest",
                model=base_preset,
                url=args.cloud_url,
                token=args.cloud_token,
            )
            if success and os.path.exists(dest_file):
                base_ckpt_path = dest_file

        if base_ckpt_path and os.path.exists(base_ckpt_path):
            print(f"\n🚀 Loading base pre-trained weights from {os.path.basename(base_ckpt_path)}...")
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
    remat_choice = (args.remat == "true") or (args.remat == "auto" and is_tpu)
    compute_grad_step, apply_opt_step = build_spheroid_train_engine(
        config,
        buffers,
        pad_id=pad_id,
        weight_decay=args.weight_decay,
        repetition_penalty=args.repetition_penalty,
        remat=remat_choice,
        deliberation_mode=args.deliberation_mode,
    )

    def sft_train_step_fn(p, opt_s, q_batch, r_batch, cur_lr, feedback_tokens=None):
        loss_val, grads = compute_grad_step(
            p, q_batch, r_batch, feedback_tokens=feedback_tokens, sampling_p=args.sampling_p, deliberation_mode=args.deliberation_mode
        )
        new_p, new_opt_s = apply_opt_step(p, opt_s, grads, cur_lr)
        return new_p, new_opt_s, loss_val

    # Initialize fresh Spheroid optimizer state
    opt_state = init_spheroid_opt_state(params)

    # 6. SFT Multi-Corpus Data Ingestion & Prefetching
    print(f"\n📦 Initializing Curated Conversational & Reasoning SFT Streamer...")
    sft_data_iter = build_online_dataset_iterator(dataset_source="sft", total_steps=args.total_steps)
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
    print(f"   • Prompt / Target  : {args.prompt_len} tok / {args.seq_len} tok (Prompt tokens masked: -100)")
    print(f"   • Live Probing     : Every {args.probe_interval} steps")
    print(f"   • Cloud Registry   : {sft_model_name}")
    print("=" * 84 + "\n")

    step = start_step
    t_start = time.time()
    t_last_log = time.time()
    step_window_tokens = 0
    step_window_count = 0
    last_eval_loss = float("inf")
    ema_loss: Optional[float] = None

    try:
        while step < args.total_steps and not shutdown_requested:
            prev_step = step
            step += 1

            cur_lr = get_sft_lr(step)

            # Get pre-tokenized batch
            b_tup = prefetched_sft.get_batch()
            prompt_b, target_b, feedback_b = _unpack_batch(b_tup)

            # SFT Optimization Step
            params, opt_state, loss = sft_train_step_fn(
                params,
                opt_state,
                prompt_b,
                target_b,
                cur_lr,
                feedback_tokens=feedback_b,
            )

            loss_val = float(loss)
            if math.isfinite(loss_val):
                ema_loss = loss_val if ema_loss is None else (0.95 * ema_loss + 0.05 * loss_val)

            step_tokens = prompt_b.shape[0] * (prompt_b.shape[1] + target_b.shape[1])
            step_window_tokens += step_tokens
            step_window_count += 1

            # Console Progress
            if (prev_step == start_step) or (step % 10 == 0):
                elapsed_log = time.time() - t_last_log
                steps_per_sec = step_window_count / max(elapsed_log, 1e-6)
                tok_per_sec = step_window_tokens / max(elapsed_log, 1e-6)
                print(
                    f"[SFT Step {step:5d}/{args.total_steps}] "
                    f"Loss: {loss_val:6.4f} | "
                    f"EMA: {ema_loss:6.4f} | "
                    f"Steps/s: {steps_per_sec:5.2f} | "
                    f"Tok/s: {tok_per_sec:7,.0f} | "
                    f"LR: {cur_lr:.2e} | "
                    f"Elapsed: {time.time() - t_start:5.1f}s"
                )
                t_last_log = time.time()
                step_window_tokens = 0
                step_window_count = 0

            # Canonical SFT Evaluation
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

                if is_champion:
                    best_eval_loss = eval_loss
                    best_fname = f"{sft_model_name}_best.pt"
                    meta_best = {
                        "step": step,
                        "eval_loss": eval_loss,
                        "loss": float(loss),
                        "ema_loss": ema_loss if ema_loss is not None else float(loss),
                        "ppl": eval_ppl,
                        "preset": base_preset,
                        "model": sft_model_name,
                        "sync_model": sft_model_name,
                        "timestamp": time.time(),
                    }
                    ckpt_mgr.dispatch(params, buffers, best_fname, meta_dict=meta_best, block_if_busy=True)

                    if args.cloud_sync:
                        try:
                            champ_rec = {
                                "step": step,
                                "loss": float(loss),
                                "eval_loss": eval_loss,
                                "status": "NEW_SFT_CHAMPION",
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

            # Live Conversational Probe
            if step % args.probe_interval == 0:
                run_live_probe(
                    params=params,
                    buffers=buffers,
                    config=config,
                    tokenizer=tokenizer,
                    prompt_len=args.prompt_len,
                    gen_len=128,
                )

            # Checkpoint Saving
            if step % args.save_interval == 0:
                latest_fname = f"{sft_model_name}_latest.pt"
                meta_latest = {
                    "step": step,
                    "loss": float(loss),
                    "ema_loss": ema_loss if ema_loss is not None else float(loss),
                    "eval_loss": last_eval_loss if math.isfinite(last_eval_loss) else float(loss),
                    "best_eval_loss": best_eval_loss,
                    "preset": base_preset,
                    "model": sft_model_name,
                    "sync_model": sft_model_name,
                    "timestamp": time.time(),
                }
                ckpt_mgr.dispatch(params, buffers, latest_fname, meta_dict=meta_latest)

                if args.cloud_sync:
                    try:
                        latest_rec = {
                            "step": step,
                            "loss": float(loss),
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
            "loss": float(loss) if "loss" in locals() else 0.0,
            "ema_loss": ema_loss if ema_loss is not None else 0.0,
            "eval_loss": best_eval_loss,
            "preset": base_preset,
            "model": sft_model_name,
            "sync_model": sft_model_name,
            "timestamp": time.time(),
        }
        ckpt_mgr.flush(params, buffers, final_fname, meta_dict=final_meta)
        ckpt_mgr.shutdown(wait=True)
        if prefetched_sft is not None:
            prefetched_sft.shutdown()
        print(f"🎉 SFT session successfully completed at Step {step:,}!")


def run_sft_training(**kwargs):
    """Programmatic entrypoint for notebooks and scripts."""
    args = parse_sft_args([])
    for k, v in kwargs.items():
        if hasattr(args, k):
            setattr(args, k, v)
        elif k == "checkpoint_dir":
            setattr(args, "checkpoint_dir", v)
    return main(args)


if __name__ == "__main__":
    main()
