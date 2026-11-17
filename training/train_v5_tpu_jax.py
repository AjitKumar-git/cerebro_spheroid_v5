#!/usr/bin/env python3
"""
CerebroSpheroid V5: Pure JAX Cloud TPU High-Throughput Production Trainer
========================================================================

Ported and enhanced from the battle-tested CerebroSpheroid V4 production trainer:
- Pure JAX V5 Architecture: 3D SwiGLU tensor contractions, vectorized axonal shortcuts, affine-tied emission
- Native TPU BF16 MXU Systolic Acceleration (~50,000 to 120,000 tokens/sec on TPU v5e-1)
- Multi-Corpus Cognitive Streaming: Cosmopedia-v2, FineWeb-Edu, SmolTalk, OpenHermes-2.5, GSM8K, and offline local corpus
- Elastic Sliding Context Window Engine (ElasticBucketBatcher) with complementary prompt/target buckets
- Background Asynchronous Batch Prefetcher (PrefetchedBatcher) eliminating I/O starvation
- Continuous Scheduled Sampling Curriculum (Cosine annealing of sampling_p down to min_sampling_p)
- Fixed Clean Held-Out Validation Split (True held-out loss and perplexity evaluation)
- Zero-Stall Background Asynchronous Checkpoint Manager (AsyncCheckpointManager, 0ms TPU pause)
- SpheroidOptimizer (Quintic Newton-Schulz NS-5 polar updates + Contractive Spectral Norm Guard)
- Autonomous Checkpoint Discovery, Bi-directional Cloud Synchronization (bioflo.in), and 2D->3D tensor adaptation
- Graceful Signal Handling (SIGINT, SIGTERM, max_hours wall-clock budget, checkpoints/PAUSE)
"""

import argparse
from functools import partial
import glob
import json
import math
import os
import random
import re
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Iterator

import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Kaggle / Python 3.12 compatibility shim: Newer pyarrow (>=19) deprecated and removed pa.PyExtensionType,
# which causes older HuggingFace datasets (<3.0.0) to raise AttributeError
try:
    import pyarrow as pa
    if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
        pa.PyExtensionType = pa.ExtensionType
except Exception:
    pass

# Claim TPU for JAX first before importing PyTorch so torch/torch_xla cannot steal TPU devices
import jax
import jax.numpy as jnp
from jax import lax

try:
    _ = jax.devices()
except Exception:
    pass

os.environ.setdefault("PJRT_DEVICE", "CPU")
os.environ.setdefault("TORCH_XLA_DISABLE_DISTRIBUTED", "1")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import optax
    HAS_OPTAX = True
except ImportError:
    optax = None
    HAS_OPTAX = False

from src.jax.v5_model import (
    ModelConfigV5 as JAXConfigV5,
    PRESETS_V5,
    init_params_v5,
    training_rollout,
)
from src.jax.spheroid_optimizer import (
    init_spheroid_opt_state,
    build_spheroid_train_engine,
    build_train_step_spheroid,
    build_pmap_local_sgd_step_spheroid,
    replicate_for_pmap,
)
from src.jax.checkpoint import (
    save_jax_to_torch_pt,
    load_torch_pt_to_jax,
    adapt_and_load_checkpoint_weights,
    discover_latest_checkpoint,
    AsyncCheckpointManager,
)
from src.cloud_sync import (
    check_remote_manifest,
    select_highest_step_slot,
    pull_checkpoint,
    push_checkpoint_async,
    push_remote_loss_history,
    push_remote_token_tracker,
    fetch_remote_token_tracker,
    detect_source_environment,
    wait_for_in_flight_uploads,
)
from src.corpus import build_tokenizer
from src.elastic_window import (
    ElasticBucketBatcher,
    PrefetchedBatcher,
    build_online_dataset_iterator,
    get_default_elastic_buckets,
)
from src.jax.data_streamer import LinguisticDataStreamer
from src.jax.auto_tuner import run_auto_tuner, resolve_jax_compute_dtype


def parse_args(args_list=None):
    if args_list is None:
        args_list = sys.argv[1:]
    parser = argparse.ArgumentParser(description="CerebroSpheroid V5 Cloud TPU JAX Trainer")

    # Target steps & duration
    parser.add_argument("--total-steps", "--steps", "--max-steps", dest="total_steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--max_hours", type=float, default=None, help="Maximum training wall-clock time in hours before graceful shutdown")
    parser.add_argument("--cloud_save_hours", type=float, default=None, help="Hours between asynchronous cloud checkpoint uploads to bioflo.in")

    # Batching & Dimensions
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128, help="Per-device batch size (default: 128 for TPU v5e-1)")
    parser.add_argument("--max-batch-size", "--max_batch_size", dest="max_batch_size", type=int, default=None, help="Optional cap on auto-tuned per-TPU batch size (e.g., 64)")
    parser.add_argument("--grad-accum", "--grad_accum", dest="grad_accum", type=int, default=1, help="Gradient accumulation micro-steps per optimizer update")
    parser.add_argument("--multi-tpu", "--multi_tpu", dest="multi_tpu", type=str, default="auto", choices=["auto", "local_sgd", "single"], help="Multi-TPU execution mode ('auto'/'local_sgd' uses all TPUs with 1 step per TPU)")
    parser.add_argument("--local-steps", "--local_steps", dest="local_steps", type=int, default=4, help="Sequential local optimizer steps per TPU before pmean parameter merge (default: 4)")
    parser.add_argument("--seq-len", "--target_len", dest="seq_len", type=int, default=512, help="Target response sequence length (default: 512 for deep reasoning)")
    parser.add_argument("--target-seq-len", "--target_seq_len", dest="target_seq_len", type=int, default=None, help="Target sequence length to transition toward (e.g. 512 or 1024)")
    parser.add_argument("--seq-transition-steps", "--seq_transition_steps", dest="seq_transition_steps", type=int, default=0, help="Steps over which to transition sequence length (0 = immediate transition to target_seq_len)")
    parser.add_argument("--prompt_len", type=int, default=256, help="Prompt context length (used when --elastic_window is disabled, default: 256)")
    parser.add_argument("--total_window", type=int, default=768, help="Total sequence budget for elastic sliding window (default: 768 tokens, 256 prompt + 512 target)")
    parser.add_argument("--elastic_window", action="store_true", default=False, help="Enable elastic sliding context window with complementary buckets")

    # Learning Rate & Optimization
    parser.add_argument("--lr", type=float, default=2e-3, help="Peak Spheroid learning rate (default: 2e-3)")
    parser.add_argument("--min_lr", type=float, default=1e-4, help="Final annealed learning rate floor (default: 1e-4)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps (default: 500)")
    parser.add_argument("--resume_warmup_steps", "--resume-warmup-steps", dest="resume_warmup_steps", type=int, default=100, help="Smooth warmup steps upon checkpoint resumption (default: 100)")
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "warm_restart", "session", "anneal"], help="Learning rate schedule")
    parser.add_argument("--restart_steps", type=int, default=25000, help="Cycle period for warm_restart schedule")
    parser.add_argument("--optimizer", type=str, default="spheroid", choices=["spheroid", "adamw"], help="Optimizer choice (default: spheroid)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay multiplier")
    parser.add_argument("--repetition_penalty", type=float, default=1.25, help="Two-tier refractory repetition penalty multiplier")

    # Model & Precision
    parser.add_argument("--preset", type=str, default="auto", choices=["auto", "v5_small", "v5_compact_32shells", "v5_compact", "v5_base", "v5_test", "tpu_v5e_1", "tpu_v6e_1", "tpu_v6e"], help="Model preset (default: auto -> auto-detects tpu_v6e_1 / tpu_v5e_1)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Matrix execution precision ('auto' selects bf16 on TPU)")
    parser.add_argument("--remat", type=str, default="true", choices=["true", "false", "auto"], help="Activation remat scan to minimize HBM footprint (default: true)")
    parser.add_argument("--auto-tune", action="store_true", default=False, help="Micro-benchmark candidate batch sizes at startup to maximize steps/s, tok/s, and HBM/VRAM utilization")
    parser.add_argument("--force-tune", "--force_tune", dest="force_tune", action="store_true", default=False, help="Force re-running auto-tuner benchmarks even if cached config exists on bioflo.in")
    parser.add_argument(
        "--tune-objective",
        "--tune_objective",
        dest="tune_objective",
        type=str,
        default="balanced",
        choices=["balanced", "steps_per_sec", "tok_per_sec"],
        help="Auto-tuner selection objective: 'balanced' (tok/s * sqrt(steps/s)), 'steps_per_sec', or 'tok_per_sec'",
    )

    # Scheduled Sampling & Deliberation Curriculum
    parser.add_argument("--deliberation_mode", type=str, default="teacher_forcing", choices=["teacher_forcing", "scheduled", "pitfall_recovery", "internal", "interleaved", "dual_pass", "predictive_coding"], help="Deliberation feedback mode: 'teacher_forcing' uses pure ground truth (peak 200k tok/s); 'scheduled' anneals sampling_p smoothly; 'interleaved' alternates deliberation and teacher forcing; 'dual_pass' runs twin resonance passes; 'predictive_coding' trains predictive state innovation; 'pitfall_recovery' trains model to recover from erroneous thoughts")
    parser.add_argument("--schedule_steps", type=int, default=0, help="Steps to anneal from teacher-forcing down to min_sampling_p (default: 0)")
    parser.add_argument("--min_sampling_p", type=float, default=1.0, help="Minimum teacher sampling floor (default: 1.0 for teacher-forcing)")
    parser.add_argument("--pitfall_prob", type=float, default=0.15, help="Probability of exploring counterfactual/pitfall thoughts during deliberation (default: 0.15)")
    parser.add_argument("--loss_window_governor", action=argparse.BooleanOptionalAction, default=True, help="Automatically clamp deliberation and float sampling_p back to 1.0 if loss climbs above window ceiling")
    parser.add_argument("--loss_ceiling", type=float, default=3.25, help="Upper loss boundary: if smoothed loss exceeds this, restore full (100%%) teacher forcing to prevent loss climb (default: 3.25)")
    parser.add_argument("--loss_floor", type=float, default=2.80, help="Lower loss boundary: if smoothed loss is at/below this, permit full deliberation at min_sampling_p (default: 2.80)")
    parser.add_argument("--loss_lr_damping", action=argparse.BooleanOptionalAction, default=True, help="Damp learning rate when loss is climbing towards ceiling to prevent optimizer overshoot")

    # Target Workload & Cognitive Steering Hyperparameters
    parser.add_argument("--num_mtp_heads", "--num-mtp-heads", dest="num_mtp_heads", type=int, default=1, choices=[1, 2, 3, 4], help="Multi-Token Prediction heads (1: standard single-token emission; 2 or 3: speculative decoding & future-token predictive training)")
    parser.add_argument("--persona_id", "--persona-id", dest="persona_id", type=int, default=None, choices=[0, 1, 2, 3], help="Cognitive Persona Steering basis (None: unbiased representation; 0: Reasoner, 1: Explorer, 2: Concise, 3: Verifier)")
    parser.add_argument("--use_adaptive_thinking", "--use-adaptive-thinking", dest="use_adaptive_thinking", action=argparse.BooleanOptionalAction, default=True, help="Toggle dynamic subconscious deliberation halting gate (True: adaptive thinking up to max cycles; False: fast fixed cycles)")
    parser.add_argument("--no_adaptive_thinking", "--no-adaptive-thinking", dest="use_adaptive_thinking", action="store_false", help="Disable adaptive thinking (use fast fixed cycles)")
    parser.add_argument("--thinking_cycles", "--thinking-cycles", dest="thinking_cycles", type=int, default=4, help="Subconscious deliberation iterations per token (default: 4)")
    parser.add_argument("--max_thinking_cycles", "--max-thinking-cycles", dest="max_thinking_cycles", type=int, default=16, help="Maximum thinking cycles when adaptive thinking is active (default: 16)")

    # Data Source & Tokenizer
    parser.add_argument("--dataset_source", "--dataset-source", dest="dataset_source", type=str, default="autonomous",
                        help="Data stream: 'autonomous', custom Hugging Face repo (e.g. 'AjitKumar-Bio/cosmopedia-v2-digested'), 'reasoning_anneal', 'dense_edu', 'conversation_sft', 'consolidation'")
    parser.add_argument("--tokenizer_name", type=str, default="llama", help="Tokenizer model name or path (default: 'llama' -> data/tokenizer)")

    # Checkpoint & Cloud Sync
    parser.add_argument("--checkpoint-dir", "--ckpt-dir", dest="checkpoint_dir", type=str, default="/content/checkpoints" if os.path.exists("/content") else "checkpoints/tpu_jax", help="Checkpoint directory")
    parser.add_argument("--scratch_dir", type=str, default="/tmp/cerebro_v5_ckpts", help="Fast local disk for staging")
    parser.add_argument("--save-interval", "--save-every", dest="save_interval", type=int, default=250, help="Steps between local checkpoint saves (default: 250)")
    parser.add_argument("--periodic_every", type=int, default=1000, help="Steps between milestone archive saves (default: 1000)")
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=10, help="Steps between console progress lines (default: 10)")
    parser.add_argument("--eval_interval", type=int, default=100, help="Steps between held-out validation evaluations (default: 100)")
    parser.add_argument("--cloud-sync", "--cloud_sync", dest="cloud_sync", action="store_true", default=True, help="Enable live sync with bioflo.in")
    parser.add_argument("--no-cloud-sync", "--no_cloud_sync", dest="cloud_sync", action="store_false", help="Disable live sync with bioflo.in")
    parser.add_argument("--cloud-sync-interval", type=int, default=100, help="Steps between cloud sync uploads")
    parser.add_argument("--cloud_url", type=str, default="https://bioflo.in/api/cerebro/sync.php", help="bioflo.in sync endpoint")
    parser.add_argument("--cloud_token", type=str, default="CerebroSpheroid_Sync_Secure_2026", help="Secret sync token")
    parser.add_argument("--resume_target", type=str, default="latest", choices=["latest", "best"], help="Target checkpoint to pull from cloud ('latest' or 'best')")
    parser.add_argument("--reset_best", action="store_true", default=False, help="Reset best validation loss tracker")
    parser.add_argument("--from-scratch", "--from_scratch", dest="from_scratch", action="store_true", default=False, help="Start clean from Step 0 without loading any checkpoint")
    parser.add_argument("--effective-batch-size", "--effective_batch_size", dest="effective_batch_size", type=int, default=256, help="Target global effective batch size (default: 256 for TPU v5e-1, supports 128 / 126; automatically computes grad_accum)")
    parser.add_argument("--target-tokens", "--target_tokens", dest="target_tokens", type=int, default=200_000_000_000, help="Target token ingestion budget for campaign (default: 200,000,000,000 tokens)")
    parser.add_argument("--init_from_checkpoint", "--init-from-checkpoint", dest="init_from_checkpoint", type=str, default=None, help="Base foundation checkpoint to load weights from (e.g. step900k)")
    parser.add_argument("--reset_step_counter", "--reset-step-counter", dest="reset_step_counter", action="store_true", default=False, help="Reset step counter to 0 after loading base weights to start new campaign from scratch")
    parser.add_argument("--target-eval-loss", "--target_eval_loss", dest="target_eval_loss", type=float, default=1.60, help="Target held-out evaluation loss for convergence early stopping (default: 1.60)")
    parser.add_argument("--consistent-eval-threshold", "--consistent_eval_threshold", dest="consistent_eval_threshold", type=int, default=3, help="Consecutive evaluations required <= target-eval-loss to confirm convergence (default: 3)")
    parser.add_argument("--freeze_core", "--freeze-core", dest="freeze_core", action=argparse.BooleanOptionalAction, default=False, help="Enable cyclical cortical expansion pretraining (freeze core for 900 steps, anneal for 100 steps every 1000 steps)")
    parser.add_argument("--cycle_length", "--cycle-length", dest="cycle_length", type=int, default=1000, help="Cycle length in steps for freeze/anneal consolidation (default: 1000)")
    parser.add_argument("--anneal_steps", "--anneal-steps", dest="anneal_steps", type=int, default=100, help="Steps within each cycle spent annealing/unfrozen (default: 100)")
    parser.add_argument("--teacher_digest", "--teacher-digest", dest="teacher_digest", type=str, default="none", choices=["online", "heuristic", "none"], help="Enable Teacher Digestion & Dynamic Next-Token Loss Weighting (default: none)")
    parser.add_argument("--teacher_model", "--teacher-model", dest="teacher_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="HuggingFace teacher model for digestion")
    parser.add_argument("--teacher_scale", "--teacher-scale", dest="teacher_scale", type=float, default=1.5, help="Scaling factor alpha for dynamic next-token weights")
    parser.add_argument("--encephalization_warmup_steps", "--encephalization-warmup-steps", dest="encephalization_warmup_steps", type=int, default=1000, help="Steps to smoothly transition motor surface readout from core to outer mantle")

    return parser.parse_args(args_list)


def build_data_stream(
    dataset_source: str,
    tokenizer,
    batch_size: int,
    prompt_len: int,
    target_len: int,
    total_window: int,
    use_elastic_window: bool,
    vocab_size: int = 32000,
    prefetch_queue_size: int = 16,
    step_getter: Optional[Any] = None,
    total_steps: int = 1_500_000,
):
    """Initializes multi-corpus streaming engine with background prefetching and offline fallback."""
    # Check if HuggingFace datasets library is available for online streaming
    HAS_DATASETS = False
    try:
        from datasets import load_dataset
        HAS_DATASETS = True
    except ImportError:
        pass

    if dataset_source in ("local", "pretrain", "offline") or not HAS_DATASETS:
        if dataset_source not in ("local", "pretrain", "offline") and not HAS_DATASETS:
            print(f"⚠️ [DataStreamer] 'datasets' library not found. Falling back to high-throughput local corpus.")
        print(f"📦 Initializing background linguistic streaming queue (Local Corpus in data/)...")
        streamer = LinguisticDataStreamer(
            batch_size=batch_size,
            prompt_len=prompt_len,
            target_len=target_len,
            vocab_size=vocab_size,
            queue_size=max(16, int(prefetch_queue_size)),
        )
        return streamer, None

    # Multi-corpus online streaming
    print(f"\n📦 Initializing Online Streaming Dataset: [{dataset_source.upper()}]...")
    data_iter = None
    try:
        import itertools

        raw_iter = build_online_dataset_iterator(
            dataset_source=dataset_source,
            step_getter=step_getter,
            total_steps=total_steps,
        )
        first_row = next(raw_iter)
        data_iter = itertools.chain([first_row], raw_iter)

        # Build Elastic Context Batcher
        if use_elastic_window:
            buckets = get_default_elastic_buckets(total_window)
            print(f"🌊 Elastic Sliding Context Window Active (Budget: {total_window} tokens | Global Batch: {batch_size}):")
            for idx, (b_name, bp, bt) in enumerate(buckets):
                print(f"   Bucket {idx}: {b_name:<18} (Prompt: {bp:4d} tok -> Target: {bt:4d} tok)")
            batcher = ElasticBucketBatcher(
                tokenizer, data_iter, batch_size=batch_size, total_window=total_window, buckets=buckets, framework="numpy", return_feedback=True
            )
        else:
            buckets = [("static", prompt_len, target_len)]
            batcher = ElasticBucketBatcher(
                tokenizer, data_iter, batch_size=batch_size, total_window=prompt_len + target_len, buckets=buckets, framework="numpy", return_feedback=True
            )

        q_depth = max(16, int(prefetch_queue_size))
        prefetched = PrefetchedBatcher(batcher, queue_size=q_depth)
        print(f"⚡ Background Asynchronous Batch Prefetcher Active (Queue Depth: {q_depth} batches)")
        return None, prefetched

    except Exception as e:
        print(f"⚠️ [DataStreamer] Online dataset initialization failed: {e}. Falling back to local corpus.")
        streamer = LinguisticDataStreamer(
            batch_size=batch_size,
            prompt_len=prompt_len,
            target_len=target_len,
            vocab_size=vocab_size,
            queue_size=max(16, int(prefetch_queue_size)),
        )
        return streamer, None


def main(args=None):
    if args is None:
        args = parse_args()

    # Graceful shutdown handler
    shutdown_requested = False

    def sig_handler(sig, frame):
        nonlocal shutdown_requested
        print(f"\n⚠️ Caught signal {sig}. Requesting graceful shutdown...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Sequence Length Transition & Context Budget Progression
    if getattr(args, "target_seq_len", None) is not None and args.target_seq_len > args.seq_len:
        print(f"\n🔄 [Sequence Transition] Advancing target sequence length from {args.seq_len} -> {args.target_seq_len} tokens for deep reasoning!")
        args.seq_len = args.target_seq_len

    if args.total_window in (288, 768) and (args.seq_len != 512 or args.prompt_len != 256):
        args.total_window = args.prompt_len + args.seq_len

    # 1. Hardware Platform & Devices
    devices = jax.devices()
    backend = jax.default_backend().upper()
    dev_kind = devices[0].device_kind.lower() if devices else ""
    num_devices = len(devices)

    # Resolve 'auto' preset based on active hardware
    if getattr(args, "preset", "tpu_v5e_1") == "auto":
        dev_name = devices[0].device_kind.lower() if devices else ""
        if "v6" in dev_name or "trillium" in dev_name:
            args.preset = "tpu_v6e_1"
        elif "v5" in dev_name:
            args.preset = "tpu_v5e_1"
        else:
            args.preset = "v5_small"

    target_dtype, dtype_str = resolve_jax_compute_dtype(args.dtype)

    print("=" * 80)
    print("⚡ CEREBROSPHEROID V5: PURE JAX HIGH-THROUGHPUT ENGINE")
    print("=" * 80)
    print(f"JAX Version     : {jax.__version__}")
    print(f"JAX Backend     : {backend}")
    print(f"Active Devices  : {num_devices} x {devices[0].device_kind}")
    print(f"Precision       : {dtype_str}")
    print(f"Preset Target   : {args.preset}")
    print(f"Destination Dir : {args.checkpoint_dir}")
    print("=" * 80)

    # 2. Auto-Tuning Micro-Benchmark
    remat_choice = True if args.remat.lower() == "true" else False
    max_bs = getattr(args, "max_batch_size", None)
    if args.auto_tune:
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
            tune_objective=getattr(args, "tune_objective", "balanced"),
            prompt_len=args.prompt_len,
            max_batch_size=max_bs,
        )
        args.batch_size = best_cfg["batch_size"]
        remat_choice = best_cfg["remat"]
        print(f"\n⚡ Applying Tuned Hyperparameters: batch_size={args.batch_size}, remat={remat_choice}")
        if "tok_per_sec" in best_cfg and best_cfg["tok_per_sec"] > 0:
            m_type = best_cfg.get("memory_type", "HBM" if ("tpu" in dev_kind or "tpu" in backend.lower()) else "VRAM")
            v_gb = best_cfg.get("memory_gb", best_cfg.get("vram_gb", 0))
            v_pct = best_cfg.get("memory_utilization_pct", best_cfg.get("vram_utilization_pct", 0))
            sps = best_cfg.get("steps_per_sec", 0.0)
            sps_str = f"{sps:.2f} steps/s | " if sps and sps > 0 else ""
            print(f"⚡ Target Benchmark Speed: {sps_str}{best_cfg['tok_per_sec']:,.0f} tok/s ({m_type}: {v_gb:.1f} GB, {v_pct:.1f}% utilization)")
    elif max_bs is not None:
        args.batch_size = min(int(args.batch_size), int(max_bs))

    # Effective Batch Size & Gradient Accumulation
    if getattr(args, "effective_batch_size", None) is not None and args.effective_batch_size > 0:
        eff_target = args.effective_batch_size
        if eff_target == 126:
            if args.batch_size > 126:
                args.batch_size = 126
                args.grad_accum = 1
            elif 126 % args.batch_size == 0:
                args.grad_accum = 126 // args.batch_size
            else:
                args.batch_size = 126
                args.grad_accum = 1
        else:
            if args.batch_size > eff_target:
                args.batch_size = eff_target
            args.grad_accum = max(1, eff_target // max(1, args.batch_size))
        eff_bs = args.batch_size * args.grad_accum
        print(f"⚡ [Batch Scaling] Target Effective Batch Size: {eff_bs} (Micro Batch: {args.batch_size} × Grad Accum: {args.grad_accum})")

    eff_batch_size = args.batch_size * getattr(args, "grad_accum", 1)
    tokens_per_opt_step = eff_batch_size * (args.prompt_len + args.seq_len)
    if getattr(args, "target_tokens", None) is not None and args.target_tokens > 0:
        calculated_steps = int(math.ceil(float(args.target_tokens) / max(1.0, float(tokens_per_opt_step))))
        args.total_steps = calculated_steps
        print(f"🎯 [Target Token Budget] {args.target_tokens/1e9:.2f} Billion Tokens -> Scheduled for {args.total_steps:,} Steps (at {tokens_per_opt_step:,} tok/step)")

    # 3. Model Configuration & Parameters
    config_preset = "v5_small" if args.preset in ("tpu_v5e_1", "tpu_v6e_1", "tpu_v6e") else args.preset
    if config_preset not in PRESETS_V5:
        config_preset = "v5_small"
    cfg_kwargs = PRESETS_V5[config_preset].copy()
    if getattr(args, "num_mtp_heads", None) is not None:
        cfg_kwargs["num_mtp_heads"] = args.num_mtp_heads
    if getattr(args, "use_adaptive_thinking", None) is not None:
        cfg_kwargs["use_adaptive_thinking"] = args.use_adaptive_thinking
    if getattr(args, "thinking_cycles", None) is not None:
        cfg_kwargs["thinking_cycles"] = args.thinking_cycles
    if getattr(args, "max_thinking_cycles", None) is not None:
        cfg_kwargs["max_thinking_cycles"] = args.max_thinking_cycles
    config = JAXConfigV5(**cfg_kwargs)
    params, buffers = init_params_v5(jax.random.PRNGKey(42), config)

    # Convert parameter arrays to target precision
    params = {k: v.astype(target_dtype) for k, v in params.items()}

    total_params = sum(p.size for p in params.values())
    active_compute = sum(p.size for k, p in params.items() if "embedding" not in k)
    print(f"\nModel Config    : {args.preset} | Shells: {config.num_shells} | Surface Dim: {config.surface_dim} | Radial Dim: {config.radial_dim}")
    print(f"Total Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    print(f"Active Compute  : {active_compute:,} (~{active_compute/1e6:.1f}M) ({active_compute/total_params*100:.2f}%)")
    print(f"Optimizer       : {args.optimizer.upper()} ({'Polar Newton-Schulz NS-5 + Spectral Clamp' if args.optimizer == 'spheroid' else 'Optax AdamW'})")

    # 4. Tokenizer
    tokenizer = build_tokenizer(args.tokenizer_name)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    # 5. Checkpoint Manager & Resumption
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.scratch_dir, exist_ok=True)

    ckpt_mgr = AsyncCheckpointManager(
        local_dir=args.scratch_dir,
        target_dir=args.checkpoint_dir,
        cloud_sync=args.cloud_sync,
        cloud_url=args.cloud_url,
        cloud_token=args.cloud_token,
        min_cloud_upload_interval_sec=120.0,
    )

    start_step = 0
    best_eval_loss = float("inf")
    last_eval_loss = float("inf")
    initial_ema_loss: Optional[float] = None
    sync_model_name = "v5_small" if args.preset in ("tpu_v5e_1", "tpu_v6e_1", "tpu_v6e") else args.preset
    local_latest = os.path.join(args.checkpoint_dir, f"{args.preset}_latest.pt")
    fallback_latest = os.path.join(args.checkpoint_dir, f"{sync_model_name}_latest.pt")

    # Checkpoint Restoration or Clean Scratch Start
    if args.from_scratch:
        print("\n🌱 [from_scratch=True] Initializing clean Step 0 training with random orthogonal weights (checkpoint restoration bypassed)!")
        start_step = 0
        best_eval_loss = float("inf")
        last_eval_loss = float("inf")
        # Clean local target checkpoints if present to prevent accidental mixing
        for old_f in [local_latest, fallback_latest, os.path.join(args.checkpoint_dir, f"{args.preset}_best.pt")]:
            if os.path.exists(old_f):
                try:
                    os.remove(old_f)
                except Exception:
                    pass
    else:
        # Cloud sync pull if active (pick whichever slot — named or root — has highest step)
        if args.cloud_sync:
            try:
                print(f"\n[cloud_sync] Checking remote manifest on bioflo.in for {args.preset}...")
                remote_manifest = check_remote_manifest(sync_model_name, url=args.cloud_url, token=args.cloud_token)
                if not remote_manifest and sync_model_name != args.preset:
                    remote_manifest = check_remote_manifest(args.preset, url=args.cloud_url, token=args.cloud_token)
                if remote_manifest:
                    slot_model, remote_step, remote_loss = select_highest_step_slot(
                        remote_manifest, sync_model_name, args.resume_target
                    )
                    # Safety Override: If 'best' was requested but 'latest' is significantly ahead,
                    # prevent accidental regression (e.g. starting back at 586k when 666k is available)
                    if args.resume_target == "best":
                        _, latest_step, _ = select_highest_step_slot(
                            remote_manifest, sync_model_name, "latest"
                        )
                        if latest_step > remote_step + 5000:
                            print(
                                f"🛡️ [SAFETY OVERRIDE] 'best.pt' (Step {remote_step:,}) is >5,000 steps behind 'latest.pt' (Step {latest_step:,})!\n"
                                f"   Automatically switching resumption to 'latest.pt' to prevent step regression."
                            )
                            args.resume_target = "latest"
                            slot_model, remote_step, remote_loss = select_highest_step_slot(
                                remote_manifest, sync_model_name, "latest"
                            )

                    print(
                        f"🔎 [cloud_sync] Selected remote slot model={slot_model or 'root'} "
                        f"(Step {remote_step:,}, Loss {remote_loss:.4f}) for {args.resume_target}.pt"
                    )
                    target_dest = os.path.join(args.checkpoint_dir, f"{args.preset}_{args.resume_target}.pt")
                    pulled_ok = pull_checkpoint(
                        target_dest,
                        target_type=args.resume_target,
                        model=slot_model,
                        url=args.cloud_url,
                        token=args.cloud_token,
                    )
                    if pulled_ok and os.path.exists(target_dest):
                        local_latest = target_dest
                        print(f"✓ Restored latest checkpoint from cloud hub: {target_dest}")
            except Exception as e:
                print(f"⚠️ [cloud_sync] Notice: {e}")

        # Discover and restore latest local checkpoint or explicit foundation checkpoint
        discovered_ckpt = None
        candidate_init = getattr(args, "init_from_checkpoint", None)
        if candidate_init:
            candidate_paths = [
                candidate_init,
                os.path.join(args.checkpoint_dir, candidate_init),
                os.path.join(args.checkpoint_dir, f"{candidate_init}.pt"),
                os.path.join("checkpoints", candidate_init),
                os.path.join("checkpoints", f"{candidate_init}.pt"),
                "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_latest_step900k.pt"
                if ("900" in str(candidate_init)) else None,
                "/home/ajit/Documents/Startup/Active_Development/reasoning_agent/checkpoints/v5_small_latest_step900k.pt"
                if ("900" in str(candidate_init)) else None,
            ]
            for p in candidate_paths:
                if p and os.path.exists(p):
                    discovered_ckpt = p
                    break
            if discovered_ckpt:
                print(f"📦 Loading base model weights from explicit checkpoint: {discovered_ckpt}")
            else:
                print(f"⚠️ Explicit checkpoint '{candidate_init}' not found locally. Searching default checkpoint paths...")

        if not discovered_ckpt:
            if not os.path.exists(local_latest) and os.path.exists(fallback_latest):
                local_latest = fallback_latest
            if os.path.exists(local_latest):
                discovered_ckpt = local_latest
            else:
                found_ckpt, found_step, found_loss = discover_latest_checkpoint(args.checkpoint_dir, preset=args.preset)
                if found_ckpt:
                    discovered_ckpt = found_ckpt

        if discovered_ckpt and os.path.exists(discovered_ckpt):
            try:
                ckpt = torch.load(discovered_ckpt, map_location="cpu", weights_only=False)
                start_step = int(ckpt.get("step", 0))
                best_eval_loss = float(ckpt.get("eval_loss", ckpt.get("loss", float("inf"))))
                last_eval_loss = best_eval_loss
                restored_ema = ckpt.get("ema_loss", ckpt.get("loss", ckpt.get("eval_loss", None)))
                if restored_ema is not None and math.isfinite(float(restored_ema)) and float(restored_ema) > 0.0:
                    initial_ema_loss = float(restored_ema)
                restored = adapt_and_load_checkpoint_weights(ckpt, params, target_dtype, config, buffers=buffers)
                ema_info = f" (EMA: {initial_ema_loss:.4f})" if initial_ema_loss is not None else ""
                print(f"✓ Resumed from {os.path.basename(discovered_ckpt)} at Step {start_step:,} (Loss: {best_eval_loss:.4f}{ema_info})")
                if restored == len(params):
                    print(f"✅ Restored & adapted {restored}/{len(params)} parameter arrays into JAX engine (100% parity)!")
                else:
                    print(f"⚠️ Restored & adapted {restored}/{len(params)} parameter arrays into JAX engine ({len(params) - restored} missing).")
                
                # Campaign Reset: Keep weights, reset schedule & step counter
                if getattr(args, "reset_step_counter", False):
                    start_step = 0
                    best_eval_loss = float("inf")
                    last_eval_loss = float("inf")
                    initial_ema_loss = None
                    print("🌱 [Campaign Reset Active] Step counter reset to 0! Calibrating fresh schedule from adapted base weights.")
                else:
                    if args.total_steps <= start_step:
                        args.total_steps = start_step + calculated_steps
                        print(f"🎯 Resuming from Step {start_step:,} -> Extended campaign schedule to Step {args.total_steps:,} (+{calculated_steps:,} steps)")
            except Exception as e:
                print(f"⚠️ Checkpoint load notice: {e}. Starting fresh.")

    if args.reset_best:
        best_eval_loss = float("inf")
        last_eval_loss = float("inf")
        print("🔄 [Benchmark Recalibration] Reset best_eval_loss tracker baseline.")

    # 6. Spheroid Optimizer & JIT / Multi-TPU Local-SGD Train Step Engine
    grad_accum = max(1, int(getattr(args, "grad_accum", 1)))
    multi_mode = getattr(args, "multi_tpu", "auto")
    use_multi_tpu_local_sgd = (num_devices > 1) and (args.optimizer == "spheroid") and (multi_mode != "single")
    eff_num_devices = num_devices if use_multi_tpu_local_sgd else 1
    eff_local_steps = max(1, int(getattr(args, "local_steps", 4))) if use_multi_tpu_local_sgd else 1
    step_stride = eff_num_devices * eff_local_steps

    if use_multi_tpu_local_sgd and args.elastic_window:
        print("ℹ️ Multi-TPU Local-SGD enforces fixed static shapes for uniform pmap shards; disabling --elastic_window.")
        args.elastic_window = False

    param_masks = None
    if getattr(args, "freeze_core", False):
        from src.cortical_expansion import build_cortical_expansion_masks, count_cortical_expansion_params
        param_masks = build_cortical_expansion_masks(params, num_frozen_shells=16, num_frozen_tracts=40, framework="jax")
        stats = count_cortical_expansion_params(params, param_masks)
        print("\n" + "=" * 84)
        print("🧠 [Progressive Cortical Expansion TPU Mode Enabled]")
        print(f"   • Total Compact Parameters  : {stats['total_params']:,} ({stats['total_params']/1e6:.2f}M)")
        print(f"   • Frozen Core & Headers      : {stats['frozen_params']:,} ({stats['frozen_pct']:.1f}%)")
        print(f"   • Trainable Outer Mantle     : {stats['trainable_params']:,} ({stats['trainable_pct']:.1f}%)")
        print(f"   • Cyclical Anneal Schedule   : Freeze {args.cycle_length - args.anneal_steps} steps ➔ Anneal {args.anneal_steps} steps (every {args.cycle_length} steps)")
        print("=" * 84 + "\n")

    pmap_round_fn = None
    rep_params = None
    rep_opt_state = None
    apply_opt_step_frozen = None
    apply_opt_step_anneal = None
    train_step_fn_frozen = None
    train_step_fn_anneal = None

    if args.optimizer == "spheroid":
        opt_state = init_spheroid_opt_state(params)
        if use_multi_tpu_local_sgd:
            pmap_round_fn = build_pmap_local_sgd_step_spheroid(
                config,
                buffers,
                pad_id=pad_id,
                weight_decay=args.weight_decay,
                repetition_penalty=args.repetition_penalty,
                remat=remat_choice,
                local_steps=eff_local_steps,
                devices=devices,
            )
            rep_params = replicate_for_pmap(params, devices)
            rep_opt_state = replicate_for_pmap(opt_state, devices)
            compute_grad_step, apply_opt_step, train_step_fn = None, None, None
        elif grad_accum <= 1:
            train_step_fn_anneal = build_train_step_spheroid(
                config,
                buffers,
                pad_id=pad_id,
                weight_decay=args.weight_decay,
                repetition_penalty=args.repetition_penalty,
                remat=remat_choice,
                deliberation_mode=args.deliberation_mode,
                param_masks=None,
            )
            train_step_fn_frozen = build_train_step_spheroid(
                config,
                buffers,
                pad_id=pad_id,
                weight_decay=args.weight_decay,
                repetition_penalty=args.repetition_penalty,
                remat=remat_choice,
                deliberation_mode=args.deliberation_mode,
                param_masks=param_masks,
            ) if param_masks is not None else train_step_fn_anneal
            train_step_fn = train_step_fn_anneal
            compute_grad_step, apply_opt_step = None, None
        else:
            compute_grad_step, apply_opt_step_anneal = build_spheroid_train_engine(
                config,
                buffers,
                pad_id=pad_id,
                weight_decay=args.weight_decay,
                repetition_penalty=args.repetition_penalty,
                remat=remat_choice,
                deliberation_mode=args.deliberation_mode,
                param_masks=None,
            )
            _, apply_opt_step_frozen = build_spheroid_train_engine(
                config,
                buffers,
                pad_id=pad_id,
                weight_decay=args.weight_decay,
                repetition_penalty=args.repetition_penalty,
                remat=remat_choice,
                deliberation_mode=args.deliberation_mode,
                param_masks=param_masks,
            ) if param_masks is not None else (None, apply_opt_step_anneal)
            apply_opt_step = apply_opt_step_anneal

            def train_step_fn(p, opt_s, q_batch, r_batch, cur_lr, feedback_tokens=None, sampling_p=1.0, encephalization_alpha=1.0):
                loss_val, grads = compute_grad_step(
                    p, q_batch, r_batch,
                    feedback_tokens=feedback_tokens,
                    sampling_p=sampling_p,
                    deliberation_mode=args.deliberation_mode,
                    encephalization_alpha=encephalization_alpha,
                )
                new_p, new_opt_s = apply_opt_step(p, opt_s, grads, cur_lr)
                return new_p, new_opt_s, loss_val

    else:
        if not HAS_OPTAX:
            raise ImportError("optax is required for AdamW training. Install via: pip install optax")
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=args.lr, b1=0.9, b2=0.95, weight_decay=args.weight_decay),
        )
        opt_state = optimizer.init(params)
        compute_grad_step, apply_opt_step = None, None

        @partial(jax.jit, static_argnames=("pure_tf", "deliberation_mode"))
        def _train_step_adamw_jit(p, opt_s, q_batch, r_batch, cur_lr, feedback_tokens, sampling_p, pure_tf: bool, deliberation_mode: str, encephalization_alpha: float = 1.0):
            def loss_fn(param_dict):
                return training_rollout(
                    params=param_dict,
                    prompts=q_batch,
                    target_len=r_batch.shape[1],
                    buffers=buffers,
                    config=config,
                    feedback_tokens=feedback_tokens,
                    feedback_is_shifted=True,
                    pad_id=pad_id,
                    sampling_p=sampling_p,
                    repetition_penalty=args.repetition_penalty,
                    targets=r_batch,
                    remat=remat_choice,
                    pure_teacher_forcing=pure_tf,
                    deliberation_mode=deliberation_mode,
                    encephalization_alpha=encephalization_alpha,
                )
            loss_val, grads = jax.value_and_grad(loss_fn)(p)
            updates, next_opt_s = optimizer.update(grads, opt_s, p)
            next_p = optax.apply_updates(p, updates)
            return next_p, next_opt_s, loss_val

        def train_step_fn(p, opt_s, q_batch, r_batch, cur_lr, feedback_tokens=None, sampling_p=1.0, encephalization_alpha=1.0):
            pure_tf = (args.deliberation_mode in ("teacher_forcing", "pure_tf")) and (not isinstance(sampling_p, jax.Array)) and (float(sampling_p) >= 1.0)
            return _train_step_adamw_jit(
                p, opt_s, q_batch, r_batch, cur_lr, feedback_tokens, sampling_p, pure_tf, args.deliberation_mode, encephalization_alpha
            )

    # 7. Multi-Stream Ingestion & Prefetching
    current_step_holder = [start_step]
    local_streamer = None
    elastic_batcher = None
    batcher = None
    is_champion = False
    save_latest = False
    tok_per_sec = 0.0

    local_streamer, elastic_batcher = build_data_stream(
        dataset_source=args.dataset_source,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        target_len=args.seq_len,
        total_window=args.total_window,
        use_elastic_window=args.elastic_window,
        vocab_size=config.vocab_size,
        prefetch_queue_size=max(16, eff_num_devices * eff_local_steps * 2),
        step_getter=lambda: current_step_holder[0],
        total_steps=args.total_steps,
    )
    batcher = elastic_batcher

    def _unpack_batch(batch_tuple):
        if len(batch_tuple) == 5:
            return batch_tuple[2], batch_tuple[3], batch_tuple[4]
        if len(batch_tuple) == 4:
            return batch_tuple[2], batch_tuple[3], None
        if len(batch_tuple) == 3:
            return batch_tuple[0], batch_tuple[1], batch_tuple[2]
        return batch_tuple[0], batch_tuple[1], None

    # 8. XLA Ahead-Of-Time Compilation (non-mutating warmup matching real feedback signature)
    print("\n⚡ Compiling XLA computation graph on TPU...")
    t0_compile = time.time()
    if use_multi_tpu_local_sgd:
        dummy_prompts = np.zeros((eff_num_devices, eff_local_steps, args.batch_size, args.prompt_len), dtype=np.int32)
        dummy_targets = np.zeros((eff_num_devices, eff_local_steps, args.batch_size, args.seq_len), dtype=np.int32)
        dummy_feedback = (
            np.full((eff_num_devices, eff_local_steps, args.batch_size, args.seq_len), -100, dtype=np.int32)
            if elastic_batcher is not None
            else None
        )
        _warm_p, _warm_opt, initial_loss = pmap_round_fn(
            rep_params,
            rep_opt_state,
            dummy_prompts,
            dummy_targets,
            args.lr,
            feedback_micro=dummy_feedback,
        )
        jax.block_until_ready(initial_loss)
        init_loss_val = float(initial_loss[0])
        del _warm_p, _warm_opt
    else:
        dummy_prompts = jnp.zeros((args.batch_size, args.prompt_len), dtype=jnp.int32)
        dummy_targets = jnp.zeros((args.batch_size, args.seq_len), dtype=jnp.int32)
        dummy_feedback = (
            jnp.full((args.batch_size, args.seq_len), -100, dtype=jnp.int32)
            if elastic_batcher is not None
            else None
        )
        _warm_p, _warm_opt, initial_loss = train_step_fn(
            params,
            opt_state,
            dummy_prompts,
            dummy_targets,
            args.lr,
            feedback_tokens=dummy_feedback,
        )
        jax.block_until_ready(initial_loss)
        init_loss_val = float(initial_loss)
        del _warm_p, _warm_opt
    if start_step > 0 and math.isfinite(best_eval_loss):
        print(f"✓ XLA Compilation complete in {time.time() - t0_compile:.2f}s! Baseline Restored Loss: {best_eval_loss:.4f} (Tracer init: {init_loss_val:.2f})")
    else:
        print(f"✓ XLA Compilation complete in {time.time() - t0_compile:.2f}s! Initial Loss: {init_loss_val:.4f}")

    # 9. Clean Canonical Multi-Domain Held-Out Validation Split
    eval_batches = []
    eval_domains = []
    try:
        from src.elastic_window import build_canonical_eval_split
        eval_batches, eval_domains = build_canonical_eval_split(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            target_len=args.seq_len,
            target_pad_id=-100,
            pad_id=pad_id,
        )
        print(f"✅ Clean Benchmark Held-Out Validation: Prepared {len(eval_batches)} canonical evaluation domains:")
        for d_name in eval_domains:
            print(f"   ├─ Domain: {d_name}")
    except Exception as e:
        print(f"⚠️ Canonical validation setup notice: {e}. Falling back to active stream.")
        eval_batches = []
        try:
            if elastic_batcher is not None:
                for _ in range(4):
                    b_tup = (
                        elastic_batcher.get_batch()
                        if hasattr(elastic_batcher, "get_batch")
                        else elastic_batcher.next_batch()
                    )
                    eval_batches.append(_unpack_batch(b_tup))
            elif os.path.exists("data/tokenizer"):
                stream_val = LinguisticDataStreamer(
                    batch_size=args.batch_size,
                    prompt_len=args.prompt_len,
                    target_len=args.seq_len,
                    vocab_size=config.vocab_size,
                    queue_size=4,
                    target_pad_id=-100,
                )
                try:
                    for _ in range(4):
                        vp, vt = stream_val.get_batch()
                        eval_batches.append((vp, vt, None))
                finally:
                    if hasattr(stream_val, "stop"):
                        stream_val.stop()
            eval_domains = [f"Domain_{i}" for i in range(len(eval_batches))]
            print(f"✅ Prepared {len(eval_batches)} fallback evaluation batches.")
        except Exception as e_inner:
            print(f"⚠️ Fallback validation setup notice: {e_inner}")

    @jax.jit
    def eval_step_loss(p, q_b, r_b, fb_b=None, enc_alpha: float = 1.0):
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
            encephalization_alpha=enc_alpha,
        )

    # 10. Learning Rate & Scheduled Sampling Schedules
    def get_learning_rate(step_idx: int, total_s: int, peak_lr: float, min_lr: float, warmup_s: int) -> float:
        if step_idx <= warmup_s:
            return float(peak_lr * (float(step_idx) / max(1.0, float(warmup_s))))
        if args.lr_schedule == "warm_restart":
            cycle_step = (step_idx - warmup_s) % max(1, args.restart_steps)
            ratio = float(cycle_step) / float(args.restart_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
            return float(min_lr + (peak_lr - min_lr) * coeff)
        else:  # cosine
            ratio = min(1.0, float(step_idx - warmup_s) / max(1.0, float(total_s - warmup_s)))
            coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
            return float(min_lr + (peak_lr - min_lr) * coeff)

    def get_sampling_p(step_idx: int, sched_steps: int, min_p: float, nominal_pitfall: float = 0.0, current_loss: Optional[float] = None) -> Tuple[float, float, float]:
        if args.deliberation_mode == "teacher_forcing" or min_p >= 1.0 - 1e-4:
            return 1.0, 0.0, 0.0
        if sched_steps <= 0:
            nominal_p = min_p
        elif args.deliberation_mode == "internal":
            nominal_p = min_p
        else:
            # Relative annealing from start_step allows smooth curriculum transition on resumption without gradient shock
            elapsed = max(0, step_idx - start_step)
            ratio = min(1.0, float(elapsed) / float(sched_steps))
            coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
            nominal_p = float(min_p + (1.0 - min_p) * coeff)

        # Loss-Window Governor: automatically keeps loss within [loss_floor, loss_ceiling]
        alpha_gov = 0.0
        if getattr(args, "loss_window_governor", True) and current_loss is not None and math.isfinite(current_loss):
            c_floor = float(getattr(args, "loss_floor", 2.80))
            c_ceil = float(getattr(args, "loss_ceiling", 3.25))
            if c_ceil > c_floor:
                alpha_gov = max(0.0, min(1.0, (float(current_loss) - c_floor) / (c_ceil - c_floor)))

        # When alpha_gov -> 1.0 (loss climbing), sampling_p floats smoothly back to 1.0 (pure teacher forcing)
        # and pitfall perturbation is suppressed to 0.0 to stabilize the descent
        effective_p = nominal_p + alpha_gov * (1.0 - nominal_p)
        effective_pitfall = nominal_pitfall * (1.0 - alpha_gov)
        if effective_p >= 1.0 - 1e-4:
            effective_p = 1.0
        return float(effective_p), float(effective_pitfall), float(alpha_gov)

    # 11. High-Throughput Training Loop
    print("\n" + "=" * 80)
    print(f"🚀 Training {args.total_steps:,} Steps (Resuming Step {start_step:,})")
    print(f"   • Batch Size       : {args.batch_size} per TPU")
    if use_multi_tpu_local_sgd:
        print(f"   • Multi-TPU Engine : Local-SGD ({eff_num_devices} TPUs x {eff_local_steps} local steps = +{step_stride} steps/round)")
    print(f"   • Sequence Length  : {args.seq_len}")
    print(f"   • Deliberation Mode: {args.deliberation_mode} (Sampling P: {args.min_sampling_p})")
    print(f"   • Async Checkpoint : 0ms TPU Pause (Latest: every {args.save_interval} steps)")
    print("=" * 80 + "\n")

    def _next_batch():
        if elastic_batcher is not None:
            batch_tuple = (
                elastic_batcher.get_batch()
                if hasattr(elastic_batcher, "get_batch")
                else elastic_batcher.next_batch()
            )
            return _unpack_batch(batch_tuple)
        prompt_b, target_b = local_streamer.get_batch()
        return prompt_b, target_b, None

    def _next_multi_tpu_round():
        total_shards = eff_num_devices * eff_local_steps
        qs, rs, fbs = [], [], []
        for _ in range(total_shards):
            pb, tb, fb = _next_batch()
            qs.append(np.asarray(pb, dtype=np.int32))
            rs.append(np.asarray(tb, dtype=np.int32))
            if fb is not None:
                fbs.append(np.asarray(fb, dtype=np.int32))
        b_p, l_p = qs[0].shape
        b_t, l_t = rs[0].shape
        q_arr = np.stack(qs, axis=0).reshape(eff_num_devices, eff_local_steps, b_p, l_p)
        r_arr = np.stack(rs, axis=0).reshape(eff_num_devices, eff_local_steps, b_t, l_t)
        fb_arr = (
            np.stack(fbs, axis=0).reshape(eff_num_devices, eff_local_steps, b_t, l_t)
            if len(fbs) == total_shards
            else None
        )
        return q_arr, r_arr, fb_arr

    def _get_host_params():
        if use_multi_tpu_local_sgd:
            return jax.tree_util.tree_map(lambda x: jax.device_get(x[0]), rep_params)
        return params

    step = start_step
    t_start = time.time()
    t_last_log = time.time()
    last_cloud_save_time = time.time()
    step_window_tokens = 0
    step_window_count = 0
    ema_loss: Optional[float] = initial_ema_loss
    consistent_target_count = 0

    # Persistent Cumulative Token Accounting
    loaded_ckpt = ckpt if ("ckpt" in locals() and isinstance(ckpt, dict)) else {}
    if getattr(args, "reset_step_counter", False):
        cumulative_tokens = 0
        initial_lifetime_tokens = int(loaded_ckpt.get("lifetime_tokens", 14_745_600_000))
    else:
        cumulative_tokens = int(loaded_ckpt.get("cumulative_tokens", 0))
        initial_lifetime_tokens = int(loaded_ckpt.get("lifetime_tokens", 0))
    lifetime_tokens = initial_lifetime_tokens + cumulative_tokens

    # Restore per-dataset token attribution if present
    if not getattr(args, "reset_step_counter", False):
        loaded_ds_tokens = loaded_ckpt.get("tokens_by_dataset", {})
        if loaded_ds_tokens and hasattr(batcher, "set_tokens_by_dataset"):
            batcher.set_tokens_by_dataset(loaded_ds_tokens)
            print(f"📦 [Checkpoint Restore] Restored per-dataset token stats: {loaded_ds_tokens}")

    try:
        while step < args.total_steps and not shutdown_requested:
            prev_step = step
            step += step_stride
            current_step_holder[0] = step
            is_champion = False
            save_latest = False

            cur_lr = get_learning_rate(step, args.total_steps, args.lr, args.min_lr, args.warmup_steps)

            # Smooth resumption warmup: prevents cold-optimizer momentum shock on resume
            resume_warm_steps = int(getattr(args, "resume_warmup_steps", 100))
            if start_step > 0 and resume_warm_steps > 0:
                elapsed_resume = step - start_step
                if elapsed_resume < resume_warm_steps:
                    warmup_factor = 0.10 + 0.90 * (float(elapsed_resume) / float(max(1, resume_warm_steps)))
                    cur_lr = cur_lr * warmup_factor

            cur_sampling_p, cur_pitfall_prob, alpha_gov = get_sampling_p(
                step, args.schedule_steps, args.min_sampling_p, args.pitfall_prob, current_loss=ema_loss
            )
            # Resumption protection: smoothly ramp governor during cold resumption to prevent early jerks
            if start_step > 0 and resume_warm_steps > 0 and (step - start_step) < resume_warm_steps:
                warm_frac = float(step - start_step) / float(max(1, resume_warm_steps))
                alpha_gov = alpha_gov * warm_frac

            if getattr(args, "loss_lr_damping", True) and alpha_gov > 0.0:
                cur_lr = cur_lr * (1.0 - 0.4 * alpha_gov)

            # Training Step
            # Progressive Encephalization Warmup Alpha
            enc_warmup = max(1, getattr(args, "encephalization_warmup_steps", 1000))
            if config.num_shells > 16 and (step - start_step) < enc_warmup:
                enc_alpha = float(min(1.0, max(0.0, (step - start_step) / float(enc_warmup))))
            else:
                enc_alpha = 1.0

            # Cyclical Core Freeze & Consolidation Anneal Schedule
            is_anneal = False
            if getattr(args, "freeze_core", False):
                step_in_cycle = step % args.cycle_length
                is_anneal = step_in_cycle >= (args.cycle_length - args.anneal_steps)
                if is_anneal:
                    anneal_prog = (step_in_cycle - (args.cycle_length - args.anneal_steps)) / max(1, args.anneal_steps)
                    anneal_factor = 0.5 * (1.0 + math.cos(math.pi * anneal_prog))
                    cur_lr = 2e-6 + (cur_lr - 2e-6) * anneal_factor

            if use_multi_tpu_local_sgd:
                q_shard, r_shard, fb_shard = _next_multi_tpu_round()
                rep_params, rep_opt_state, loss_rep = pmap_round_fn(
                    rep_params,
                    rep_opt_state,
                    q_shard,
                    r_shard,
                    cur_lr,
                    feedback_micro=fb_shard,
                    sampling_p=cur_sampling_p,
                )
                loss = float(loss_rep[0])
                step_tokens = int(q_shard.shape[0] * q_shard.shape[1] * q_shard.shape[2] * (q_shard.shape[3] + r_shard.shape[3]))
            elif compute_grad_step is not None and grad_accum > 1:
                acc_grads = None
                loss_acc = 0.0
                step_tokens = 0
                inv_accum = 1.0 / float(grad_accum)
                for _ in range(grad_accum):
                    prompt_b, target_b, feedback_b = _next_batch()
                    g_loss, micro_grads = compute_grad_step(
                        params, prompt_b, target_b, feedback_tokens=feedback_b, sampling_p=cur_sampling_p, deliberation_mode=args.deliberation_mode, encephalization_alpha=enc_alpha
                    )
                    acc_grads = (
                        micro_grads
                        if acc_grads is None
                        else jax.tree_util.tree_map(lambda a, b: a + b, acc_grads, micro_grads)
                    )
                    loss_acc = loss_acc + g_loss
                    step_tokens += prompt_b.shape[0] * (prompt_b.shape[1] + target_b.shape[1])
                acc_grads = jax.tree_util.tree_map(lambda a: a * inv_accum, acc_grads)
                loss = loss_acc * inv_accum
                cur_apply_opt = apply_opt_step_anneal if (is_anneal or param_masks is None) else (apply_opt_step_frozen or apply_opt_step_anneal)
                params, opt_state = cur_apply_opt(params, opt_state, acc_grads, cur_lr)
            else:
                prompt_b, target_b, feedback_b = _next_batch()
                cur_train_step = train_step_fn_anneal if (is_anneal or param_masks is None) else (train_step_fn_frozen or train_step_fn_anneal)
                params, opt_state, loss = cur_train_step(
                    params,
                    opt_state,
                    prompt_b,
                    target_b,
                    cur_lr,
                    feedback_tokens=feedback_b,
                    sampling_p=cur_sampling_p,
                    encephalization_alpha=enc_alpha,
                )
                step_tokens = prompt_b.shape[0] * (prompt_b.shape[1] + target_b.shape[1])

            loss_val = float(loss)
            if math.isfinite(loss_val):
                # Smooth 0.96 / 0.04 EMA update to prevent single-batch spikes from oscillating the governor
                ema_loss = loss_val if ema_loss is None else (0.96 * ema_loss + 0.04 * loss_val)
            else:
                print(f"\n⚠️ [Divergence Watchdog] Non-finite loss ({loss_val}) detected at Step {step:,}! Stabilizing optimizer momentum...")
                ema_loss = last_eval_loss if math.isfinite(last_eval_loss) else 3.00
                cur_lr = cur_lr * 0.75

            # Step & token throughput accounting
            step_window_tokens += step_tokens
            step_window_count += step_stride
            cumulative_tokens += step_tokens
            lifetime_tokens = initial_lifetime_tokens + cumulative_tokens

            # Console Progress Logging
            crossed_log = (prev_step == start_step) or ((step // max(1, args.log_every)) > (prev_step // max(1, args.log_every)))
            if crossed_log:
                elapsed_log = time.time() - t_last_log
                steps_per_sec = step_window_count / max(elapsed_log, 1e-6)
                tok_per_sec = step_window_tokens / max(elapsed_log, 1e-6)

                gov_str = f" | Gov: {alpha_gov:.2f}" if getattr(args, "loss_window_governor", True) and alpha_gov > 0.01 else ""

                tgt_tok = getattr(args, "target_tokens", 200_000_000_000)
                if tgt_tok and tgt_tok > 0:
                    pct = (cumulative_tokens / tgt_tok) * 100.0
                    eta_sec = max(0.0, tgt_tok - cumulative_tokens) / max(tok_per_sec, 1e-6)
                    eta_h = eta_sec / 3600.0
                    tok_str = f"Tokens: {cumulative_tokens/1e9:5.3f}B/{tgt_tok/1e9:4.2f}B ({pct:5.2f}%) | ETA: {eta_h:4.1f}h | "
                else:
                    tok_str = f"Tokens: {cumulative_tokens/1e9:5.3f}B | "

                phase_tag = f" [{'ANNEAL' if is_anneal else 'FROZEN'}]" if getattr(args, "freeze_core", False) else ""
                print(
                    f"[Step {step:6d}/{args.total_steps}]{phase_tag} "
                    f"{tok_str}"
                    f"Loss: {loss_val:6.4f} | "
                    f"Steps/s: {steps_per_sec:5.2f} | "
                    f"Tok/s: {tok_per_sec:7,.0f} | "
                    f"LR: {cur_lr:.2e} | "
                    f"Samp_p: {cur_sampling_p:.2f}{gov_str}"
                )
                t_last_log = time.time()
                step_window_tokens = 0
                step_window_count = 0

            # Held-Out Validation Evaluation
            crossed_eval = bool(eval_batches) and ((step // max(1, args.eval_interval)) > (prev_step // max(1, args.eval_interval)))
            if crossed_eval:
                eval_p = jax.device_put(_get_host_params(), devices[0]) if use_multi_tpu_local_sgd else params
                val_losses = [float(eval_step_loss(eval_p, vp, vt, vfb, enc_alpha=enc_alpha)) for vp, vt, vfb in eval_batches]
                eval_loss = float(np.mean(val_losses)) if val_losses else float("inf")
                last_eval_loss = eval_loss
                eval_ppl = math.exp(min(eval_loss, 20.0))
                is_champion = eval_loss < best_eval_loss

                champ_tag = "🏆 (NEW CHAMPION RECORD)" if is_champion else ""
                print(f"📊 [Eval @ Step {step:,}] Composite Held-Out Loss: {eval_loss:.4f} | PPL: {eval_ppl:.2f} {champ_tag}")
                if len(val_losses) == len(eval_domains):
                    for d_name, d_val in zip(eval_domains, val_losses):
                        d_ppl = math.exp(min(d_val, 20.0))
                        print(f"   ├─ {d_name:<26}: Loss {d_val:.4f} (PPL: {d_ppl:.2f})")
                if hasattr(batcher, "get_tokens_by_dataset"):
                    ds_stats = batcher.get_tokens_by_dataset()
                    if ds_stats:
                        parts = [f"{k}: {v/1e9:.3f}B" for k, v in sorted(ds_stats.items())]
                        print(f"   └─ Ingested Tokens Breakdown: " + " | ".join(parts))

                # Convergence Early Stopping Check
                target_eval_loss = getattr(args, "target_eval_loss", None)
                consistent_eval_threshold = max(1, int(getattr(args, "consistent_eval_threshold", 3)))
                converged_target_met = False

                if target_eval_loss is not None and math.isfinite(eval_loss):
                    if eval_loss <= target_eval_loss:
                        consistent_target_count += 1
                        print(
                            f"🎯 [Convergence Target] Held-out loss {eval_loss:.4f} <= {target_eval_loss:.4f} "
                            f"({consistent_target_count}/{consistent_eval_threshold} consecutive checks)"
                        )
                        if consistent_target_count >= consistent_eval_threshold:
                            converged_target_met = True
                            print(
                                f"\n"
                                f"================================================================================\n"
                                f"🏆 [CONVERGENCE ACHIEVED & VERIFIED CONSISTENT]\n"
                                f"Held-out composite loss has reached {eval_loss:.4f} (Target <= {target_eval_loss:.4f})\n"
                                f"Consistently sustained across {consistent_eval_threshold} consecutive evaluation cycles!\n"
                                f"Final Step: {step:,} | Held-Out Loss: {eval_loss:.4f} | PPL: {eval_ppl:.2f}\n"
                                f"================================================================================\n"
                            )
                    else:
                        if consistent_target_count > 0:
                            print(
                                f"ℹ️ [Convergence Notice] Held-out loss {eval_loss:.4f} > target {target_eval_loss:.4f}. "
                                f"Resetting consistent counter ({consistent_target_count} -> 0)."
                            )
                        consistent_target_count = 0

                if is_champion or converged_target_met:
                    best_eval_loss = eval_loss
                    best_fname = f"{args.preset}_best.pt"
                    meta_best = {
                        "step": step,
                        "eval_loss": eval_loss,
                        "loss": float(loss),
                        "ema_loss": ema_loss if ema_loss is not None else float(loss),
                        "ppl": eval_ppl,
                        "preset": args.preset,
                        "sync_model": sync_model_name,
                        "cumulative_tokens": cumulative_tokens,
                        "lifetime_tokens": lifetime_tokens,
                        "target_tokens": getattr(args, "target_tokens", 200_000_000_000),
                        "effective_batch_size": args.batch_size * grad_accum,
                        "tokens_by_dataset": batcher.get_tokens_by_dataset() if hasattr(batcher, "get_tokens_by_dataset") else {},
                        "converged": bool(converged_target_met),
                        "timestamp": time.time(),
                    }
                    ckpt_mgr.dispatch(_get_host_params(), buffers, best_fname, meta_dict=meta_best, block_if_busy=True)
                    if converged_target_met:
                        converged_fname = f"{args.preset}_converged_1.6.pt"
                        ckpt_mgr.dispatch(_get_host_params(), buffers, converged_fname, meta_dict=meta_best, block_if_busy=True)

                    if args.cloud_sync:
                        try:
                            import datetime
                            status_tag = "CONVERGED_1.6" if converged_target_met else "NEW_CHAMPION"
                            champ_rec = {
                                "step": step,
                                "loss": float(loss),
                                "eval_loss": eval_loss,
                                "status": status_tag,
                                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "source": detect_source_environment(),
                            }
                            push_remote_loss_history(
                                model=sync_model_name or args.preset,
                                records=[champ_rec],
                                url=args.cloud_url,
                                token=args.cloud_token,
                            )
                        except Exception:
                            pass

                    if converged_target_met:
                        print("🏁 Gracefully completing training session as convergence target has been achieved!")
                        break

            # Asynchronous Checkpoint Saving
            save_latest = False
            if args.cloud_save_hours is not None:
                if (time.time() - last_cloud_save_time) >= (args.cloud_save_hours * 3600.0):
                    save_latest = True
                    last_cloud_save_time = time.time()

            if (step // max(1, args.save_interval)) > (prev_step // max(1, args.save_interval)):
                save_latest = True

            if save_latest:
                latest_fname = f"{args.preset}_latest.pt"
                meta_latest = {
                    "step": step,
                    "loss": float(loss),
                    "ema_loss": ema_loss if ema_loss is not None else float(loss),
                    "eval_loss": last_eval_loss if math.isfinite(last_eval_loss) else float(loss),
                    "best_eval_loss": best_eval_loss,
                    "preset": args.preset,
                    "sync_model": sync_model_name,
                    "cumulative_tokens": cumulative_tokens,
                    "lifetime_tokens": lifetime_tokens,
                    "target_tokens": getattr(args, "target_tokens", 200_000_000_000),
                    "effective_batch_size": args.batch_size * grad_accum,
                    "tokens_by_dataset": batcher.get_tokens_by_dataset() if hasattr(batcher, "get_tokens_by_dataset") else {},
                    "timestamp": time.time(),
                }
                ckpt_mgr.dispatch(_get_host_params(), buffers, latest_fname, meta_dict=meta_latest)

                if args.cloud_sync:
                    try:
                        import datetime
                        latest_rec = {
                            "step": step,
                            "loss": float(loss),
                            "eval_loss": last_eval_loss if math.isfinite(last_eval_loss) else None,
                            "status": "SMOOTH_DESCENT",
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "source": detect_source_environment(),
                        }
                        push_remote_loss_history(
                            model=sync_model_name or args.preset,
                            records=[latest_rec],
                            url=args.cloud_url,
                            token=args.cloud_token,
                        )
                    except Exception:
                        pass

            # Live Cloud Token Tracker Sync (Periodic & on Milestones)
            crossed_sync = (step % max(1, getattr(args, "cloud_sync_interval", 100)) == 0) or save_latest or is_champion
            if args.cloud_sync and crossed_sync:
                try:
                    ds_stats = batcher.get_tokens_by_dataset() if hasattr(batcher, "get_tokens_by_dataset") else {}
                    push_remote_token_tracker(
                        model=sync_model_name or args.preset,
                        cumulative_tokens=cumulative_tokens,
                        target_tokens=getattr(args, "target_tokens", 200_000_000_000),
                        tokens_by_dataset=ds_stats,
                        effective_batch_size=args.batch_size * grad_accum,
                        step=step,
                        loss=float(loss_val),
                        tok_per_sec=float(tok_per_sec) if 'tok_per_sec' in locals() else 0.0,
                        url=args.cloud_url,
                        token=args.cloud_token,
                    )
                except Exception:
                    pass

            # Periodic Milestone Archive Saving
            if (step // max(1, args.periodic_every)) > (prev_step // max(1, args.periodic_every)):
                periodic_fname = f"{args.preset}_step_{step}.pt"
                ckpt_mgr.dispatch(
                    _get_host_params(),
                    buffers,
                    periodic_fname,
                    meta_dict={
                        "step": step,
                        "loss": float(loss),
                        "ema_loss": ema_loss if ema_loss is not None else float(loss),
                        "eval_loss": last_eval_loss if math.isfinite(last_eval_loss) else float(loss),
                        "sync_model": sync_model_name,
                    },
                )

            # Sentinel file pause check
            if os.path.exists(os.path.join(args.checkpoint_dir, "PAUSE")):
                print("\n🛑 Found checkpoints/PAUSE sentinel file. Gracefully halting...")
                break

            # Wall-clock budget limit check
            if args.max_hours is not None:
                if (time.time() - t_start) >= (args.max_hours * 3600.0):
                    print(f"\n⏰ Reached maximum training time limit of {args.max_hours:.2f} hours.")
                    break

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
    except Exception as e:
        print(f"\n⚠️ Encountered training exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        step_val = step if "step" in locals() else start_step
        print(f"\n💾 Flushing final checkpoint state for Step {step_val:,}...")
        final_fname = f"{args.preset}_latest.pt"
        b_ref = batcher if ("batcher" in locals() and batcher is not None) else (elastic_batcher if ("elastic_batcher" in locals() and elastic_batcher is not None) else None)
        ds_stats = b_ref.get_tokens_by_dataset() if (b_ref is not None and hasattr(b_ref, "get_tokens_by_dataset")) else {}
        final_meta = {
            "step": step_val,
            "loss": float(loss) if "loss" in locals() else 0.0,
            "ema_loss": ema_loss if ("ema_loss" in locals() and ema_loss is not None) else (float(loss) if "loss" in locals() else 0.0),
            "eval_loss": best_eval_loss if "best_eval_loss" in locals() else float("inf"),
            "preset": args.preset,
            "sync_model": sync_model_name if "sync_model_name" in locals() else args.preset,
            "cumulative_tokens": cumulative_tokens if "cumulative_tokens" in locals() else 0,
            "lifetime_tokens": lifetime_tokens if "lifetime_tokens" in locals() else 0,
            "target_tokens": getattr(args, "target_tokens", 200_000_000_000),
            "effective_batch_size": args.batch_size * (grad_accum if "grad_accum" in locals() else 1),
            "tokens_by_dataset": ds_stats,
            "timestamp": time.time(),
        }
        if "ckpt_mgr" in locals() and ckpt_mgr is not None:
            ckpt_mgr.flush(_get_host_params(), buffers, final_fname, meta_dict=final_meta)
            ckpt_mgr.shutdown(wait=True)

        if args.cloud_sync:
            try:
                push_remote_token_tracker(
                    model=(sync_model_name if "sync_model_name" in locals() else None) or args.preset,
                    cumulative_tokens=cumulative_tokens if "cumulative_tokens" in locals() else 0,
                    target_tokens=getattr(args, "target_tokens", 200_000_000_000),
                    tokens_by_dataset=ds_stats,
                    effective_batch_size=args.batch_size * (grad_accum if "grad_accum" in locals() else 1),
                    step=step_val,
                    loss=float(loss) if "loss" in locals() else 0.0,
                    tok_per_sec=float(tok_per_sec) if 'tok_per_sec' in locals() else 0.0,
                    url=args.cloud_url,
                    token=args.cloud_token,
                )
            except Exception:
                pass
        if "local_streamer" in locals() and local_streamer is not None:
            local_streamer.stop()
        if b_ref is not None:
            if hasattr(b_ref, "stop"):
                b_ref.stop()
            elif hasattr(b_ref, "shutdown"):
                b_ref.shutdown()
        print(f"🎉 Training session completed at Step {step:,}!")


def run_training(**kwargs):
    """Programmatic entrypoint for Jupyter / Colab notebooks."""
    args = parse_args([])
    for k, v in kwargs.items():
        if hasattr(args, k):
            setattr(args, k, v)
        elif k == "drive_dir":
            setattr(args, "checkpoint_dir", v)
        elif k == "max_steps":
            setattr(args, "total_steps", v)
        elif k == "remat":
            setattr(args, "remat", str(v).lower())
    return main(args)


if __name__ == "__main__":
    main()
