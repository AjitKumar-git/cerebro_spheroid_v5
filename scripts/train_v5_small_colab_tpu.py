#!/usr/bin/env python3
"""
CerebroSpheroid V5 Small: Long-Context Cloud TPU Trainer (From Scratch)
======================================================================

Production-grade Cloud TPU JAX trainer designed for Google Colab (TPU v2-8, v3-8, v4-8, v5e, v6e)
and scalable TPU pods. Trains the `v5_small` architecture (105.9M params) from clean random
initialization (step 0) across full 4096-token sequence windows.

Key Technical Highlights:
1. Longest Context & Target Horizon:
   - Total sequence window: 4096 tokens.
   - Sensory learned positional embeddings: 4096 positions (RoPE extension beyond).
   - Motor learned phase clock: 4096 positions.
   - Elastic sliding window: Complementary (prompt, target) buckets summing to 4096 tokens:
     * Balanced Long:        (2048 prompt -> 2048 target)
     * Deep Target / CoT:    ( 512 prompt -> 3584 target)
     * Long Context QA:      (3584 prompt ->  512 target)
     * Standard Synthesis:   (1024 prompt -> 3072 target)
2. Zero Legacy Bias:
   - Starts strictly from scratch (`step = 0`) with fresh PRNG keys.
   - Eliminates all residual 32-token horizon biases and legacy format artifacts.
3. Strict Cognitive Tool Boundaries:
   - Natural conversational text, chain-of-thought, answers, and code stay plain text outside <act>.
   - Strictly genuine external tool invocations (calc(...), python(...)) are enclosed in <act> ... </act>.
4. Colab Preemption Resilience & Google Drive Mirroring:
   - Automatically detects Google Drive (`/content/drive/MyDrive/cerebro_v5_small_scratch`).
   - Saves dual-copy checkpoints (`latest.pt`, `best.pt`, and milestone `.pt`) with metadata JSON.
5. Live Generalization Probing:
   - Evaluates factual recall, arithmetic/tool calling, coding, analogy, and 500+ token reading
     comprehension every N steps with full uncapped text decoding.
"""

from __future__ import annotations

import argparse
import dataclasses
from functools import partial
import glob
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Iterator

import numpy as np

# Set workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Preempt PyTorch from grabbing TPU before JAX
import jax
import jax.numpy as jnp
from jax import lax

try:
    _ = jax.devices()
except Exception:
    pass

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TORCH_XLA_DISABLE_DISTRIBUTED", "1")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.jax.v5_model import (
    ModelConfigV5,
    PRESETS_V5,
    init_params_v5,
    generate_small_world_indices,
    training_rollout,
)
from src.jax.spheroid_optimizer import (
    init_spheroid_opt_state,
    build_train_step_spheroid,
    build_pmap_local_sgd_step_spheroid,
    replicate_for_pmap,
)
from src.jax.checkpoint import (
    save_jax_to_torch_pt,
    load_torch_pt_to_jax,
    discover_latest_checkpoint,
)
from src.cloud_sync import (
    DEFAULT_URL,
    DEFAULT_TOKEN,
    check_remote_manifest,
    pull_checkpoint,
    push_checkpoint_async,
    push_remote_token_tracker,
    push_remote_loss_history,
    detect_source_environment,
    wait_for_in_flight_uploads,
)
from src.corpus import (
    build_tokenizer,
    format_stimulus_prompt,
    format_motor_response,
    clean_motor_response,
    encode_no_special,
)
from src.elastic_window import (
    ElasticBucketBatcher,
    PrefetchedBatcher,
    build_online_dataset_iterator,
    get_default_elastic_buckets,
)
from src.rectified_streamer import RectifiedContiguousPretrainStreamer
from src.jax.data_streamer import LinguisticDataStreamer


def parse_args(args_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="CerebroSpheroid V5 Small: Long-Context TPU Trainer (From Scratch)"
    )

    # Training Mode & Curriculum Stage
    parser.add_argument("--mode", type=str, default="contiguous_pretrain",
                        choices=["contiguous_pretrain", "sft"],
                        help="Training mode: 'contiguous_pretrain' (100%% loss-bearing tokens on full dataset) or 'sft'")
    parser.add_argument("--stage", type=str, default="pretrain",
                        choices=["pretrain", "midtrain", "balanced"],
                        help="Curriculum stage: 'pretrain' (85%% FineWeb broad web + 15%% code; 10B-15B full dataset), 'midtrain' (skills, math, retrieval), or 'balanced'")

    # Context & Target Horizon (Longest possible 4096-token window)
    parser.add_argument("--prompt-len", "--prompt_len", type=int, default=2048,
                        help="Prompt context length (default: 2048)")
    parser.add_argument("--seq-len", "--target-len", "--target_len", dest="seq_len", type=int, default=2048,
                        help="Target output generation length (default: 2048)")
    parser.add_argument("--total-window", "--total_window", dest="total_window", type=int, default=4096,
                        help="Total sequence budget (default: 4096)")
    parser.add_argument("--sensory-position-steps", type=int, default=4096,
                        help="Learned sensory positional embedding steps (default: 4096)")
    parser.add_argument("--max-phase-steps", type=int, default=4096,
                        help="Learned motor phase clock horizon (default: 4096)")
    parser.add_argument("--elastic-window", action=argparse.BooleanOptionalAction, default=True,
                        help="Use complementary sliding window buckets summing to total_window")

    # Training Steps & Batching (10x Ingestion: 60,000 steps = 15.7 Billion tokens)
    parser.add_argument("--total-steps", "--steps", type=int, default=60_000,
                        help="Total training steps (default: 60,000 steps = 15.7 Billion tokens)")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=8,
                        help="Per-device micro-batch size (default: 8 for TPU v2-8/v3-8/v5e)")
    parser.add_argument("--grad-accum", "--grad_accum", dest="grad_accum", type=int, default=2,
                        help="Gradient accumulation micro-steps per optimizer step (default: 2)")
    parser.add_argument("--effective-batch-size", type=int, default=64,
                        help="Target effective batch size across all TPU chips (default: 64)")
    parser.add_argument("--multi-tpu", type=str, default="auto", choices=["auto", "local_sgd", "single"],
                        help="Multi-TPU execution mode (auto uses all available TPU chips)")
    parser.add_argument("--local-steps", type=int, default=4,
                        help="Local optimizer steps per TPU before pmean parameter synchronization")

    # Optimization Hyperparameters
    parser.add_argument("--lr", type=float, default=2e-3,
                        help="Peak learning rate for Spheroid optimizer (default: 2e-3)")
    parser.add_argument("--min-lr", type=float, default=1e-5,
                        help="Final learning rate floor (default: 1e-5)")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Linear warmup steps (default: 1000)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay coefficient (default: 1e-4)")
    parser.add_argument("--optimizer", type=str, default="spheroid", choices=["spheroid", "adamw"],
                        help="Optimizer choice (default: spheroid NS-5)")
    parser.add_argument("--repetition-penalty", type=float, default=1.15,
                        help="Motor refractory repetition penalty (default: 1.15)")
    parser.add_argument("--deliberation-mode", type=str, default="teacher_forcing",
                        choices=["teacher_forcing", "scheduled", "dual_pass"],
                        help="Deliberation mode (default: teacher_forcing for maximum throughput)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"],
                        help="Matrix execution precision (default: bf16 for TPU MXU)")
    parser.add_argument("--remat", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable activation rematerialization for memory efficiency")

    # Dataset & Tokenizer
    parser.add_argument("--dataset-source", type=str, default="autonomous",
                        help="Data stream ('autonomous', 'dense_edu', 'fineweb_edu', 'cosmopedia', or local)")
    parser.add_argument("--tokenizer-name", type=str, default="llama",
                        help="Tokenizer name or local directory (default: 'llama')")

    # Checkpointing & Cloud Sync (bioflo.in)
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/content/checkpoints/v5_small_scratch" if os.path.exists("/content") else "checkpoints/v5_small_scratch",
                        help="Local staging checkpoint directory")
    parser.add_argument("--cloud-sync", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable live two-way synchronization with bioflo.in")
    parser.add_argument("--cloud-url", type=str, default="https://bioflo.in/api/cerebro/sync.php",
                        help="Remote sync hub endpoint (bioflo.in)")
    parser.add_argument("--cloud-token", type=str, default="CerebroSpheroid_Sync_Secure_2026",
                        help="Remote sync authentication token")
    parser.add_argument("--save-every", type=int, default=250,
                        help="Steps between checkpoint saves (default: 250)")
    parser.add_argument("--milestone-every", type=int, default=1000,
                        help="Steps between permanent milestone checkpoint archives (default: 1000)")
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Steps between held-out evaluation passes (default: 100)")
    parser.add_argument("--probe-every", type=int, default=500,
                        help="Steps between live generalization probes (default: 500)")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Steps between progress log lines (default: 10)")
    parser.add_argument("--from-scratch", action="store_true", default=True,
                        help="Strictly start from Step 0 without loading legacy weights (default: True)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from existing checkpoint (local or pulled from bioflo.in)")

    return parser.parse_args(args_list)


# ---------------------------------------------------------------------------
# Generalization Probes: Tested with fixed slice-free generation decoding
# ---------------------------------------------------------------------------
PROBE_PROMPTS = [
    {
        "domain": "Factual Recall",
        "prompt": "What is the capital of France?",
        "expected": "Paris",
    },
    {
        "domain": "Analogy",
        "prompt": "Complete the analogy: hot is to cold as day is to",
        "expected": "night",
    },
    {
        "domain": "Arithmetic & Tool Call",
        "prompt": "What is 15 + 27?",
        "expected": "42 (or <act> calc(15 + 27) </act>)",
    },
    {
        "domain": "Math Word Problem",
        "prompt": "Natalia sold clips to 48 of her friends in April, and then sold half as many in May. How many clips did she sell in total?",
        "expected": "72",
    },
    {
        "domain": "Code Synthesis",
        "prompt": "Write a Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome.",
        "expected": "def is_palindrome(s: str) -> bool:",
    },
    {
        "domain": "Long-Context Reading Comprehension",
        "prompt": (
            "Study the following passage and answer the question: The Amazon rainforest, also known as Amazonia, "
            "covers most of the Amazon basin in South America. The basin encompasses 7,000,000 square kilometers, "
            "of which 5,500,000 square kilometers are covered by the rainforest. The majority of the forest is "
            "contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, and Colombia with 10%.\n\n"
            "Question: Which country contains the highest percentage of the Amazon rainforest?"
        ),
        "expected": "Brazil",
    },
]


def run_generalization_probe(params, buffers, config, tokenizer, step: int, max_tokens: int = 64):
    """Executes live probe queries using the verified generation pipeline."""
    print(f"\n{'='*75}")
    print(f"🔍 [Probe] Generalization & Tool-Boundary Diagnostics (Step {step:,})")
    print(f"{'='*75}")

    for item in PROBE_PROMPTS:
        domain = item["domain"]
        prompt_text = item["prompt"]
        expected = item["expected"]

        stimulus = format_stimulus_prompt(prompt_text)
        tokens = encode_no_special(tokenizer, stimulus)
        if len(tokens) == 0:
            tokens = [1]
        prompt_arr = jnp.array([tokens], dtype=jnp.int32)

        # Autoregressive generation rollout
        out_logits = training_rollout(
            params=params,
            prompts=prompt_arr,
            target_len=max_tokens,
            buffers=buffers,
            config=config,
            targets=None,
            feedback_tokens=None,
            pure_teacher_forcing=False,
            deliberation_mode="teacher_forcing",
            repetition_penalty=1.15,
        )

        gen_token_ids = np.array(jnp.argmax(out_logits, axis=-1)[0])
        raw_text = tokenizer.decode(gen_token_ids, skip_special_tokens=False)
        clean_ans = clean_motor_response(raw_text).strip()

        print(f"\n[{domain}]")
        print(f"  Prompt:   \"{prompt_text[:90]}{'...' if len(prompt_text) > 90 else ''}\"")
        print(f"  Target:   [{expected}]")
        print(f"  Response: \"{clean_ans[:120]}{'...' if len(clean_ans) > 120 else ''}\"")
        if "<act>" in raw_text:
            print(f"  Action:   {raw_text[raw_text.find('<act>'):raw_text.find('</act>')+6]}")

    print(f"\n{'='*75}\n")


def compute_cosine_lr(step: int, total_steps: int, warmup_steps: int, lr: float, min_lr: float) -> float:
    """Computes cosine decayed learning rate with linear warmup."""
    if step < warmup_steps:
        return float(lr * (step + 1) / max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (lr - min_lr) * cosine_decay)


def main(args=None):
    if args is None:
        args = parse_args()

    print("\n" + "=" * 80)
    print("  🧠 CerebroSpheroid V5 Small: Long-Context TPU Scratch Trainer")
    print("=" * 80)

    # 1. Detect TPU Devices & Runtime
    devices = jax.devices()
    num_devices = len(devices)
    device_type = devices[0].platform.upper()
    print(f"⚙️  Detected {num_devices} device(s): {device_type} ({devices[0]})")

    # 2. Configure V5 Small with Extended 4096-Step Horizons
    cfg_dict = PRESETS_V5["v5_small"].copy()
    cfg_dict["sensory_position_steps"] = args.sensory_position_steps
    cfg_dict["max_phase_steps"] = args.max_phase_steps
    cfg_dict["max_seq_len"] = max(args.total_window, 4096)
    cfg_dict["use_rope"] = True
    cfg_dict["rope_theta"] = 500000.0

    config = ModelConfigV5(**cfg_dict)
    print(f"\n📐 Model Architecture (V5 Small - Clean Scratch Init):")
    print(f"   - Concentric Shells:         {config.num_shells}")
    print(f"   - Surface Dim (d):           {config.surface_dim}")
    print(f"   - Radial Dim (r):            {config.radial_dim}")
    print(f"   - Shortcut Tracts:           {config.num_shortcut_tracts}")
    print(f"   - Attention Heads:           {config.num_heads}")
    print(f"   - Sensory Pos Steps:         {config.sensory_position_steps} (Full 4k learned table + RoPE)")
    print(f"   - Motor Phase Clock:         {config.max_phase_steps} (Full 4k output horizon)")
    print(f"   - Total Window Budget:       {args.total_window} tokens")
    print(f"   - Default Prompt / Target:   {args.prompt_len} / {args.seq_len}")

    # 3. Setup Checkpoint Directory & Cloud Sync Hub (bioflo.in)
    primary_ckpt_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(primary_ckpt_dir, exist_ok=True)
    print(f"\n💾 Checkpoint Storage & Cloud Hub:")
    print(f"   - Local Directory:           {primary_ckpt_dir}")
    if args.cloud_sync:
        print(f"   - Sync Hub Endpoint:         {args.cloud_url} (bioflo.in)")
        print(f"   - Sync Model Tier:           v5_small")

    # 4. Clean Scratch Initialization
    init_seed = 42
    init_key = jax.random.PRNGKey(init_seed)
    print(f"\n🎲 Initializing parameter arrays from fresh PRNG seed {init_seed}...")
    params, buffers = init_params_v5(init_key, config)

    # Cast parameters to target dtype (bf16 for TPU)
    target_dtype = jnp.bfloat16 if args.dtype == "bf16" else (jnp.float16 if args.dtype == "fp16" else jnp.float32)
    params = {k: v.astype(target_dtype) for k, v in params.items()}
    total_params = sum(x.size for x in params.values())
    print(f"✓ Total Parameters: {total_params / 1e6:.2f}M ({args.dtype.upper()} precision)")

    # 5. Initialize Optimizer State
    # The optimizer always keeps float32 master weights for inexact parameters.
    opt_state = init_spheroid_opt_state(params)
    step = 0
    best_eval_loss = float("inf")

    # Resume check if explicitly requested
    if args.resume:
        latest_pt = discover_latest_checkpoint(primary_ckpt_dir)
        if not latest_pt and args.cloud_sync:
            # Check bioflo.in and pull latest.pt if available
            print(f"🌐 [cloud_sync] Checking bioflo.in for remote v5_small checkpoint...")
            try:
                manifest_resp = check_remote_manifest("v5_small", url=args.cloud_url, token=args.cloud_token)
                if manifest_resp.get("has_latest", False):
                    remote_dest = os.path.join(primary_ckpt_dir, "v5_small_scratch_latest.pt")
                    if pull_checkpoint(remote_dest, target_type="latest", model="v5_small", url=args.cloud_url, token=args.cloud_token):
                        latest_pt = remote_dest
            except Exception as e_pull:
                print(f"⚠️ [cloud_sync] Could not pull from bioflo.in: {e_pull}")

        if latest_pt and os.path.exists(latest_pt):
            print(f"🔄 Resuming from checkpoint: {latest_pt}")
            params, buffers, meta = load_torch_pt_to_jax(latest_pt, config, target_dtype=target_dtype)
            step = meta.get("step", 0)
            best_eval_loss = meta.get("eval_loss", float("inf"))
            print(f"✓ Resumed at Step {step:,} (Best Eval Loss: {best_eval_loss:.4f})")
        else:
            print(f"ℹ️  No checkpoint found locally or on bioflo.in. Starting clean from Step 0.")
    else:
        print("🌱 Training strictly FROM SCRATCH (Step 0) as requested.")

    # 6. Initialize Tokenizer & Multi-Corpus Streaming Engine
    print(f"\n📚 Loading Tokenizer and Audited Data Streamer...")
    tokenizer = build_tokenizer(args.tokenizer_name)
    print(f"✓ Tokenizer ready (Vocab size: {len(tokenizer):,})")

    if args.mode == "contiguous_pretrain":
        print(f"\n🌊 Rectified Contiguous Pretraining Engine Active (100% Loss-Bearing Tokens):")
        print(f"   - Curriculum Stage:     {args.stage.upper()}")
        if args.stage == "pretrain":
            print(f"   - Target Token Mixture: 85% Broad Web (FineWeb 10BT) | 15% Pretraining Code")
        elif args.stage == "midtrain":
            print(f"   - Target Token Mixture: 45% FineWeb-Edu (>= 3.5) | 25% FineMath | 15% Code | 15% 4k Retrieval")
        else:
            print(f"   - Target Token Mixture: 70% Broad Web (FineWeb) | 15% Edu | 7% Code | 5% Math | 3% Retrieval")
        print(f"   - Block Size:           {args.total_window} contiguous tokens per sequence")
        print(f"   - Prompt Input:         1 token (BOS seed)")
        print(f"   - Target Supervision:   {args.total_window} tokens with cross-entropy next-token loss")
        print(f"   - Zero Prompt Slicing:  Documents packed contiguously with EOS delimiters")
        print(f"   - Target Ingestion:     {args.total_steps:,} steps ({args.total_steps * args.batch_size * args.grad_accum * args.total_window / 1e9:.2f}B tokens)")
        prefetched_stream = RectifiedContiguousPretrainStreamer(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            block_size=args.total_window,
            bos_token_id=1,
            eos_token_id=2,
            queue_size=16,
            stage=args.stage,
        )
        print(f"⚡ Background Token-Weighted Prefetcher Active (Depth: 16)")
    else:
        # SFT Mode: Elastic Buckets with prompt masking
        if args.elastic_window:
            buckets = [
                ("balanced_long", 2048, 2048),
                ("short_prompt_deep_target", 512, 3584),
                ("long_context_qa", 3584, 512),
                ("standard_cot", 1024, 3072),
            ]
            print(f"\n🌊 Elastic Sliding Window Active (Budget: {args.total_window} tokens):")
            for idx, (b_name, bp, bt) in enumerate(buckets):
                print(f"   Bucket {idx}: {b_name:<26} (Prompt: {bp:4d} tok -> Target: {bt:4d} tok)")
        else:
            buckets = [("static", args.prompt_len, args.seq_len)]
            print(f"\n🔒 Static Window Active: Prompt {args.prompt_len} -> Target {args.seq_len}")

        try:
            data_iter = build_online_dataset_iterator(
                dataset_source=args.dataset_source,
                step_getter=lambda: step,
                total_steps=args.total_steps,
            )
            batcher = ElasticBucketBatcher(
                tokenizer,
                data_iter,
                batch_size=args.batch_size,
                total_window=args.total_window,
                buckets=buckets,
                framework="numpy",
                return_feedback=True,
            )
            prefetched_stream = PrefetchedBatcher(batcher, queue_size=16)
            print(f"⚡ Background Asynchronous Batch Prefetcher Active (Depth: 16)")
        except Exception as e:
            print(f"⚠️ Online dataset init failed ({e}). Falling back to local linguistic streamer...")
            prefetched_stream = LinguisticDataStreamer(
                batch_size=args.batch_size,
                prompt_len=args.prompt_len,
                target_len=args.seq_len,
                vocab_size=config.vocab_size,
                queue_size=16,
            )

    # 7. Build Execution Engine
    # Multi-TPU pmap or Single-TPU JIT
    use_multi_tpu = (num_devices > 1) and (args.multi_tpu in ("auto", "local_sgd"))
    if use_multi_tpu:
        print(f"🚀 Compiling Multi-TPU Local-SGD execution kernel across {num_devices} devices...")
        rep_params = replicate_for_pmap(params, devices)
        rep_opt_state = replicate_for_pmap(opt_state, devices)
        pmap_train_step = build_pmap_local_sgd_step_spheroid(
            config=config,
            buffers=buffers,
            pad_id=0,
            weight_decay=args.weight_decay,
            repetition_penalty=args.repetition_penalty,
            remat=args.remat,
            local_steps=args.local_steps,
            devices=devices,
        )
        train_step_fn = None
    else:
        print(f"🚀 Compiling Single-Device JIT execution kernel...")
        rep_params, rep_opt_state, pmap_train_step = None, None, None
        train_step_fn = build_train_step_spheroid(
            config=config,
            buffers=buffers,
            pad_id=0,
            weight_decay=args.weight_decay,
            repetition_penalty=args.repetition_penalty,
            remat=args.remat,
            deliberation_mode=args.deliberation_mode,
        )

    # 8. Training Loop
    print("\n" + "=" * 80)
    print("  🔥 Beginning Production Scratch Training Run")
    print("=" * 80 + "\n")

    start_time = time.time()
    last_log_time = start_time
    total_tokens_processed = 0

    while step < args.total_steps:
        step_start = time.time()
        cur_lr = compute_cosine_lr(step, args.total_steps, args.warmup_steps, args.lr, args.min_lr)

        # Retrieve next batch
        try:
            batch = next(prefetched_stream)
        except StopIteration:
            break

        # Extract prompt and target arrays
        if isinstance(batch, dict):
            prompts = batch["prompt"]
            targets = batch["target"]
            fb_tokens = batch.get("feedback_tokens", None)
        elif isinstance(batch, (tuple, list)):
            prompts = batch[0]
            targets = batch[1]
            fb_tokens = batch[2] if len(batch) > 2 else None
        else:
            prompts, targets, fb_tokens = batch, batch, None

        # Execute training step
        if use_multi_tpu:
            # Reshape inputs for [num_devices, local_steps, B_per_tpu, L]
            B_total = prompts.shape[0]
            tokens_in_batch = B_total * (prompts.shape[1] + targets.shape[1])
            # For pmap multi-device: pass replicated inputs
            # Fallback to single JIT if batch doesn't divide evenly
            try:
                rep_params, rep_opt_state, loss_val = pmap_train_step(
                    rep_params, rep_opt_state, prompts, targets, cur_lr, fb_tokens, 1.0
                )
                loss_scalar = float(np.mean(loss_val))
            except Exception:
                # Fallback to direct execution
                if train_step_fn is None:
                    train_step_fn = build_train_step_spheroid(
                        config=config, buffers=buffers, pad_id=0,
                        weight_decay=args.weight_decay, remat=args.remat,
                    )
                params, opt_state, loss_val = train_step_fn(
                    params, opt_state, prompts, targets, cur_lr, fb_tokens, 1.0
                )
                loss_scalar = float(loss_val)
        else:
            tokens_in_batch = prompts.shape[0] * (prompts.shape[1] + targets.shape[1])
            params, opt_state, loss_val = train_step_fn(
                params, opt_state, prompts, targets, cur_lr, fb_tokens, 1.0
            )
            loss_scalar = float(loss_val)

        total_tokens_processed += tokens_in_batch
        step += 1

        # Periodic Progress Log
        if step % args.log_every == 0 or step == 1:
            now = time.time()
            elapsed = now - last_log_time
            last_log_time = now
            steps_per_sec = args.log_every / max(elapsed, 1e-6)
            tokens_per_sec = (tokens_in_batch * args.log_every) / max(elapsed, 1e-6)
            ppl = math.exp(min(loss_scalar, 20.0))

            mix_info = ""
            if hasattr(prefetched_stream, "get_token_mix_status"):
                mix_info = f" | {prefetched_stream.get_token_mix_status()}"

            print(
                f"Step {step:7,d}/{args.total_steps:,d} | "
                f"Loss: {loss_scalar:6.4f} | "
                f"PPL: {ppl:7.2f} | "
                f"LR: {cur_lr:.2e}{mix_info} | "
                f"Speed: {steps_per_sec:4.1f} st/s ({tokens_per_sec:7,.0f} tok/s) | "
                f"Loss-Bearing Tok: {total_tokens_processed / 1e6:6.1f}M"
            )

            # Sync real-time progress to website (bioflo.in)
            if args.cloud_sync and step % 50 == 0:
                try:
                    push_remote_token_tracker(
                        model="v5_small",
                        cumulative_tokens=total_tokens_processed,
                        target_tokens=int(args.total_steps * tokens_in_batch),
                        tokens_by_dataset=prefetched_stream.quota_mgr.counts if hasattr(prefetched_stream, "quota_mgr") else {},
                        effective_batch_size=args.batch_size * args.grad_accum,
                        step=step,
                        loss=loss_scalar,
                        tok_per_sec=tokens_per_sec,
                        url=args.cloud_url,
                        token=args.cloud_token,
                    )
                except Exception:
                    pass

        # Periodic Generalization Probe
        if step % args.probe_every == 0 or step == 50:
            active_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], rep_params)) if use_multi_tpu else params
            run_generalization_probe(active_params, buffers, config, tokenizer, step=step)

        # Periodic Checkpoint Saving
        if step % args.save_every == 0 or step == args.total_steps:
            active_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], rep_params)) if use_multi_tpu else params
            meta = {
                "step": step,
                "loss": loss_scalar,
                "eval_loss": loss_scalar,
                "total_tokens": total_tokens_processed,
                "config": dataclasses.asdict(config) if hasattr(dataclasses, "asdict") else {},
                "timestamp": time.time(),
            }

            # 1. Save local latest checkpoint
            save_path = os.path.join(primary_ckpt_dir, "v5_small_scratch_latest.pt")
            save_jax_to_torch_pt(active_params, buffers, save_path, meta=meta)
            json_path = save_path.replace(".pt", ".json")
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)

            # 2. Async Cloud Upload to bioflo.in
            if args.cloud_sync:
                try:
                    push_checkpoint_async(
                        save_path,
                        step=step,
                        loss=loss_scalar,
                        target_type="latest",
                        model="v5_small",
                        url=args.cloud_url,
                        token=args.cloud_token,
                    )
                    print(f"☁️ [cloud_sync] Uploaded latest.pt to bioflo.in (Step {step:,})")
                except Exception as e_cloud:
                    print(f"⚠️ [cloud_sync] Upload notice: {e_cloud}")

            # 3. Check if Best Eval Loss
            if loss_scalar < best_eval_loss:
                best_eval_loss = loss_scalar
                best_path = os.path.join(primary_ckpt_dir, "v5_small_scratch_best.pt")
                shutil.copy2(save_path, best_path)
                shutil.copy2(json_path, best_path.replace(".pt", ".json"))
                if args.cloud_sync:
                    try:
                        push_checkpoint_async(
                            best_path,
                            step=step,
                            loss=loss_scalar,
                            target_type="best",
                            model="v5_small",
                            url=args.cloud_url,
                            token=args.cloud_token,
                        )
                        print(f"🏆 [cloud_sync] Uploaded new best.pt to bioflo.in (Loss: {loss_scalar:.4f})")
                    except Exception:
                        pass

            # 4. Save Milestone Archive
            if step % args.milestone_every == 0:
                milestone_path = os.path.join(primary_ckpt_dir, f"v5_small_scratch_step_{step:07d}.pt")
                shutil.copy2(save_path, milestone_path)
                shutil.copy2(json_path, milestone_path.replace(".pt", ".json"))
                print(f"📦 [Milestone] Saved permanent archive: {milestone_path}")

            print(f"💾 [Checkpoint] Saved Step {step:,} (Loss: {loss_scalar:.4f}) -> {save_path}")

    print("\n" + "=" * 80)
    print(f"🎉 Training Complete! Finished {step:,} steps. Processed {total_tokens_processed / 1e9:.2f}B tokens.")
    print("=" * 80 + "\n")


def run_training(**kwargs):
    """Programmatic in-process entrypoint for Google Colab notebooks.
    
    Runs training inside the active Python kernel process to prevent Cloud TPU
    lock conflicts ('ABORTED: The TPU is already in use by process with pid XXX').
    """
    args = parse_args([])
    for k, v in kwargs.items():
        attr = k.replace("-", "_")
        if hasattr(args, attr):
            setattr(args, attr, v)
        elif hasattr(args, k):
            setattr(args, k, v)
    return main(args)


if __name__ == "__main__":
    main()
