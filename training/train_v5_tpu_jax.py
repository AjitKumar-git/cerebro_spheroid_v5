#!/usr/bin/env python3
"""
CerebroSpheroid V5: Pure JAX Cloud TPU High-Throughput Production Trainer
========================================================================

High-throughput CerebroSpheroid V5 production trainer:
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
import dataclasses
import os
import random
import re
import shutil
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
    safe_torch_load,
    adapt_and_load_checkpoint_weights,
    discover_latest_checkpoint,
    AsyncCheckpointManager,
    get_current_rss_gb,
)
from src.cloud_sync import (
    get_sync_token,
    check_remote_manifest,
    select_highest_step_slot,
    pull_checkpoint,
    push_checkpoint_async,
    push_remote_loss_history,
    push_remote_loss_history_async,
    push_remote_token_tracker,
    push_remote_token_tracker_async,
    fetch_remote_token_tracker,
    detect_source_environment,
    wait_for_in_flight_uploads,
    wait_for_telemetry_uploads,
)
from src.corpus import build_tokenizer
from src.elastic_window import (
    ElasticBucketBatcher,
    PrefetchedBatcher,
    build_online_dataset_iterator,
    get_default_elastic_buckets,
)
from src.jax.data_streamer import LinguisticDataStreamer
from autotune import (
    run_auto_tuner,
    resolve_jax_compute_dtype,
    is_oom_error,
    TrainingOOMHandler,
)


def apply_from_scratch_arch_upgrades(cfg_kwargs: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Activates V5.1 architectural upgrades (multi-layer sensory, full-mantle readout, bidirectional tracts, sensory-core coupling, parallel plan scan) ONLY on from-scratch runs."""
    explicit_layers = getattr(args, "num_sensory_layers", None)
    if getattr(args, "from_scratch", False):
        cfg_kwargs["num_sensory_layers"] = int(explicit_layers) if explicit_layers is not None else 4
        cfg_kwargs["use_centralized_readout"] = False
        cfg_kwargs["use_shell_tap"] = True
        cfg_kwargs["use_bidirectional_tracts"] = True
        cfg_kwargs["use_sensory_core_coupling"] = True
        cfg_kwargs["use_parallel_plan_scan"] = True
        cfg_kwargs["top_k_resonance"] = 0
        cfg_kwargs["preserve_shell_norm_clamp"] = False
    else:
        if explicit_layers is not None:
            cfg_kwargs["num_sensory_layers"] = int(explicit_layers)
        cfg_kwargs["use_bidirectional_tracts"] = bool(getattr(args, "use_bidirectional_tracts", False))
        cfg_kwargs["use_sequential_cascade"] = True
        cfg_kwargs.setdefault("use_sensory_core_coupling", False)
        cfg_kwargs.setdefault("use_parallel_plan_scan", False)
        cfg_kwargs.setdefault("preserve_shell_norm_clamp", True)
    return cfg_kwargs


def resolve_checkpoint_arch_overrides(ckpt: Dict[str, Any], cfg_kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Preserves V5.1 architecture flags when resuming a checkpoint created from scratch, leaving legacy runs untouched."""
    updated = dict(cfg_kwargs)
    changed = False
    raw_state = ckpt.get("model_state") or ckpt.get("params") or ckpt.get("state_dict") or {}
    saved_cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}

    extra_idx = -1
    if isinstance(raw_state, dict):
        for k in raw_state.keys():
            if k.startswith("sensory_dnn.extra_layers."):
                try:
                    idx = int(k.split(".")[2])
                    if idx > extra_idx:
                        extra_idx = idx
                except (IndexError, ValueError):
                    pass
    detected_layers = (extra_idx + 2) if extra_idx >= 0 else int(ckpt.get("num_sensory_layers", saved_cfg.get("num_sensory_layers", updated.get("num_sensory_layers", 1))))
    if detected_layers != updated.get("num_sensory_layers", 1):
        updated["num_sensory_layers"] = detected_layers
        changed = True

    for flag in ("use_centralized_readout", "use_shell_tap", "use_bidirectional_tracts", "use_sensory_core_coupling", "use_parallel_plan_scan", "preserve_shell_norm_clamp", "use_sequential_cascade"):
        if flag in ckpt:
            val = bool(ckpt[flag])
            if val != updated.get(flag):
                updated[flag] = val
                changed = True
        elif flag in saved_cfg:
            val = bool(saved_cfg[flag])
            if val != updated.get(flag):
                updated[flag] = val
                changed = True

    if "top_k_resonance" in ckpt:
        tk_val = int(ckpt["top_k_resonance"])
        if tk_val != updated.get("top_k_resonance"):
            updated["top_k_resonance"] = tk_val
            changed = True
    elif "top_k_resonance" in saved_cfg:
        tk_val = int(saved_cfg["top_k_resonance"])
        if tk_val != updated.get("top_k_resonance"):
            updated["top_k_resonance"] = tk_val
            changed = True

    return updated, changed


def parse_args(args_list=None):
    if args_list is None:
        args_list = sys.argv[1:]
    parser = argparse.ArgumentParser(description="CerebroSpheroid V5 Cloud TPU JAX Trainer")

    # Target steps & duration
    parser.add_argument("--total-steps", "--total_steps", "--steps", "--max-steps", "--max_steps", dest="total_steps", type=int, default=1_000_000, help="Total training steps")
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
    parser.add_argument("--prompt_len", type=int, default=256, help="Prompt context length (default: 256)")
    parser.add_argument("--total_window", type=int, default=None, help="Total sequence budget for elastic sliding window (default: prompt_len + seq_len)")
    parser.add_argument("--elastic_window", action=argparse.BooleanOptionalAction, default=True, help="Enable complementary context/target buckets")
    parser.add_argument("--context-curriculum", action=argparse.BooleanOptionalAction, default=True, help="Grow context window through 512, 1024, 2048, and 4096 token stages")
    parser.add_argument("--context-stage-steps", type=int, default=100000, help="Steps per context stage (default: 100,000 steps/stage across 512, 1024, 2048, 4096)")
    parser.add_argument("--min-context-window", "--min_context_window", dest="min_context_window", type=int, default=512, help="Minimum context window stage for curriculum (default: 512, set 1024 to start at 1024)")
    parser.add_argument("--use-rope", "--use_rope", dest="use_rope", action=argparse.BooleanOptionalAction, default=True, help="Use rotary Q/K positions for new long-context training runs (default: True)")
    parser.add_argument("--max-phase-steps", "--max_phase_steps", dest="max_phase_steps", type=int, default=4096, help="Learned output phase-clock horizon for new runs (default: 4096)")
    parser.add_argument("--sensory-position-steps", "--sensory_position_steps", dest="sensory_position_steps", type=int, default=4096, help="Learned sensory positional embedding steps (default: 4096)")

    # Learning Rate & Optimization
    parser.add_argument("--lr", type=float, default=2e-3, help="Peak Spheroid learning rate (default: 2e-3)")
    parser.add_argument("--min_lr", type=float, default=1e-4, help="Final annealed learning rate floor (default: 1e-4)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps (default: 500)")
    parser.add_argument("--resume_warmup_steps", "--resume-warmup-steps", dest="resume_warmup_steps", type=int, default=100, help="Smooth warmup steps upon checkpoint resumption (default: 100)")
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "warm_restart", "session", "anneal"], help="Learning rate schedule")
    parser.add_argument("--restart_steps", type=int, default=25000, help="Cycle period for warm_restart schedule")
    parser.add_argument("--optimizer", type=str, default="scmer", choices=["scmer", "spheroid", "adamw"], help="Optimizer choice (default: scmer)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay multiplier")
    parser.add_argument("--repetition_penalty", type=float, default=1.25, help="Two-tier refractory repetition penalty multiplier")
    parser.add_argument("--scmer-lambda", "--scmer_lambda", dest="scmer_lambda", type=float, default=0.05, help="Selective Cosine-Margin Embedding Regularization strength (0 disables it, default: 0.05)")
    parser.add_argument("--scmer-margin", "--scmer_margin", dest="scmer_margin", type=float, default=0.12, help="Cosine margin threshold m above which positive similarity is penalized (default: 0.12)")
    parser.add_argument("--scmer-subsample", "--scmer_subsample", dest="scmer_subsample", type=int, default=512, help="Number of token embedding rows sampled per regularizer event")
    parser.add_argument("--scmer-pair-samples", "--scmer_pair_samples", dest="scmer_pair_samples", type=int, default=8192, help="Number of unique unordered token pairs sampled without replacement")
    parser.add_argument("--scmer-interval", "--scmer_interval", dest="scmer_interval", type=int, default=16, help="Apply SCMER regularization every N optimizer updates (default: 16)")

    # Backward compatibility aliases for SCMER
    parser.add_argument("--neg-cov-lambda", "--neg_cov_lambda", dest="neg_cov_lambda", type=float, default=None, help="Alias for --scmer-lambda")
    parser.add_argument("--neg-cov-mode", "--neg_cov_mode", dest="neg_cov_mode", choices=["asymmetric", "full_decorr"], default="asymmetric", help="Compatibility flag")
    parser.add_argument("--neg-cov-subsample", "--neg_cov_subsample", dest="neg_cov_subsample", type=int, default=None, help="Alias for --scmer-subsample")
    parser.add_argument("--neg-cov-interval", "--neg_cov_interval", dest="neg_cov_interval", type=int, default=None, help="Alias for --scmer-interval")

    # Model & Precision
    parser.add_argument("--preset", type=str, default="auto", choices=["auto", "v5_small", "v5_compact_32shells", "v5_compact", "v5_base", "v5_test", "tpu_v5e_1", "tpu_v6e_1", "tpu_v6e"], help="Model preset (default: auto -> auto-detects tpu_v6e_1 / tpu_v5e_1)")
    parser.add_argument("--vocab-size", "--vocab_size", dest="vocab_size", type=int, default=None, help="Optional model vocabulary size override (default: preset vocab_size)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Matrix execution precision ('auto' selects bf16 on TPU)")
    parser.add_argument("--remat", type=str, default="true", choices=["true", "false", "auto"], help="Activation remat scan to minimize HBM footprint (default: true)")
    parser.add_argument("--auto_tune", "--auto-tune", dest="auto_tune", action=argparse.BooleanOptionalAction, default=True, help="Micro-benchmark candidate batch sizes at startup to maximize steps/s, tok/s, and HBM/VRAM utilization (default: True)")
    parser.add_argument("--force-tune", "--force_tune", dest="force_tune", action="store_true", default=False, help="Force re-running auto-tuner benchmarks even if cached config exists on bioflo.in")
    parser.add_argument(
        "--tune-objective",
        "--tune_objective",
        dest="tune_objective",
        type=str,
        default="balanced",
        choices=["balanced", "steps_per_sec", "steps/s", "tok_per_sec", "tokens/sec", "loss_reduction_per_sec", "loss_reduction_sec", "loss reduction/sec"],
        help="Auto-tuner selection objective: 'balanced', 'steps_per_sec' (steps/s), 'tok_per_sec' (tokens/sec), or 'loss_reduction_per_sec' (loss reduction/sec)",
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
    parser.add_argument("--selective_loss", "--selective-loss", dest="selective_loss", action=argparse.BooleanOptionalAction, default=False, help="Enable noise-capped selective token weighting to focus limited compute on learnable tokens")
    parser.add_argument("--selective_focal_gamma", "--selective-focal-gamma", dest="selective_focal_gamma", type=float, default=0.5, help="Mild focal exponent (1-p)^gamma for downweighting mastered tokens (default: 0.5)")
    parser.add_argument("--selective_noise_cap", "--selective-noise-cap", dest="selective_noise_cap", type=float, default=8.5, help="Per-token CE ceiling above which unlearnable web-noise spikes are attenuated (default: 8.5)")
    parser.add_argument("--z_loss_weight", "--z-loss-weight", dest="z_loss_weight", type=float, default=0.0, help="Logit Z-loss regularizer weight (e.g., 1e-4) to stabilize bf16 logits (default: 0.0)")
    parser.add_argument("--use_bidirectional_tracts", "--use-bidirectional-tracts", dest="use_bidirectional_tracts", action=argparse.BooleanOptionalAction, default=False, help="Enable bidirectional corticofugal small-world shortcut tracts (for new from-scratch runs; default: False)")
    parser.add_argument("--num_sensory_layers", "--num-sensory-layers", dest="num_sensory_layers", type=int, default=None, help="Causal sensory self-attention layers before KV projection (default: 4 on --from_scratch, 1 on resume)")

    # Target Workload & Cognitive Steering Hyperparameters
    parser.add_argument("--num_mtp_heads", "--num-mtp-heads", dest="num_mtp_heads", type=int, default=1, choices=[1, 2, 3, 4], help="Multi-Token Prediction heads (1: standard single-token emission; 2 or 3: speculative decoding & future-token predictive training)")
    parser.add_argument("--persona_id", "--persona-id", dest="persona_id", type=int, default=None, choices=[0, 1, 2, 3], help="Cognitive Persona Steering basis (None: unbiased representation; 0: Reasoner, 1: Explorer, 2: Concise, 3: Verifier)")
    parser.add_argument("--use_adaptive_thinking", "--use-adaptive-thinking", dest="use_adaptive_thinking", action=argparse.BooleanOptionalAction, default=True, help="Toggle dynamic subconscious deliberation halting gate (True: adaptive thinking up to max cycles; False: fast fixed cycles)")
    parser.add_argument("--no_adaptive_thinking", "--no-adaptive-thinking", dest="use_adaptive_thinking", action="store_false", help="Disable adaptive thinking (use fast fixed cycles)")
    parser.add_argument("--thinking_cycles", "--thinking-cycles", dest="thinking_cycles", type=int, default=4, help="Subconscious deliberation iterations per token (default: 4)")
    parser.add_argument("--max_thinking_cycles", "--max-thinking-cycles", dest="max_thinking_cycles", type=int, default=4, help="Maximum thinking cycles when adaptive thinking is active (default: 4 for fast pretraining; raise to 8-16 for Stage 2 reasoning)")

    # Data Source & Tokenizer
    parser.add_argument("--hf-token", "--hf_token", dest="hf_token", type=str, default=None, help="Hugging Face API token for authenticated dataset streaming")
    parser.add_argument("--dataset_source", "--dataset-source", dest="dataset_source", type=str, default="pretrain",
                        help="Data stream: 'pretrain' (SmolLM2 4-stage mixture: FineWeb, Stack-Edu, FineMath, Cosmopedia), 'rectified', 'clean_pretrain', 'local', or a Hugging Face dataset repo")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default=None, help="Optional local data directory override for '--dataset_source local'")
    parser.add_argument("--dataset-config", type=str, default=None, help="Optional Hugging Face dataset config (FineWeb defaults to sample-10BT)")
    parser.add_argument("--dataset-revision", type=str, default=None, help="Optional immutable Hugging Face commit/revision override")
    parser.add_argument("--retrieval-injection", action=argparse.BooleanOptionalAction, default=False, help="Mix curated retrieval examples into online streams (disabled by default for clean pretrain baseline)")
    parser.add_argument("--tokenizer_name", type=str, default="llama", help="Tokenizer model name or path (default: 'llama' -> data/tokenizer)")

    # Checkpoint & Cloud Sync
    parser.add_argument("--checkpoint-dir", "--ckpt-dir", dest="checkpoint_dir", type=str, default="/content/checkpoints" if os.path.exists("/content") else "checkpoints/tpu_jax", help="Checkpoint directory")
    parser.add_argument("--scratch_dir", type=str, default="/tmp/cerebro_v5_ckpts", help="Fast local disk for staging")
    parser.add_argument("--save-interval", "--save-every", dest="save_interval", type=int, default=1000, help="Steps between local checkpoint saves (default: 1000)")
    parser.add_argument("--periodic_every", type=int, default=5000, help="Steps between milestone archive saves (default: 5000)")
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=50, help="Steps between console progress lines (default: 50)")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Steps between held-out validation evaluations (default: 1000)")
    parser.add_argument("--cloud-sync", "--cloud_sync", dest="cloud_sync", action="store_true", default=True, help="Enable live sync with bioflo.in")
    parser.add_argument("--no-cloud-sync", "--no_cloud_sync", dest="cloud_sync", action="store_false", help="Disable live sync with bioflo.in")
    parser.add_argument("--cloud_sync_interval", "--cloud-sync-interval", dest="cloud_sync_interval", type=int, default=1000, help="Steps between cloud sync uploads (default: 1000)")
    parser.add_argument(
        "--cloud_weights_only",
        "--cloud-weights-only",
        dest="cloud_weights_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload lightweight weights-only checkpoints (~229 MB) to cloud for rapid, non-blocking transfer (default: True)",
    )
    parser.add_argument(
        "--save_optimizer_state",
        "--save-optimizer-state",
        dest="save_optimizer_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save Spheroid optimizer moments (m, v, master_p) in local checkpoints (default: True)",
    )
    parser.add_argument("--sync_model", "--sync-model", dest="sync_model", type=str, default=None, help="Explicit bioflo.in model namespace for cloud telemetry and checkpoints (overrides preset, e.g. v5_small_recovery)")
    parser.add_argument("--cloud_url", type=str, default="https://bioflo.in/api/cerebro/sync.php", help="bioflo.in sync endpoint")
    parser.add_argument(
        "--cloud_token",
        type=str,
        default=get_sync_token(),
        help="Secret sync token (defaults to BIOFLO_SYNC_TOKEN environment variable or website vault)",
    )
    parser.add_argument("--run_id", "--run-id", "--run", dest="run_id", type=str, default=None, help="Unique training run identifier linking token ingestion and telemetry to this specific session")
    parser.add_argument("--resume_target", type=str, default="latest", choices=["latest", "best"], help="Target checkpoint to pull from cloud ('latest' or 'best')")
    parser.add_argument("--reset_best", action="store_true", default=False, help="Reset best validation loss tracker")
    parser.add_argument("--from-scratch", "--from_scratch", dest="from_scratch", action="store_true", default=False, help="Start clean from Step 0 without loading any checkpoint")
    parser.add_argument("--effective-batch-size", "--effective_batch_size", dest="effective_batch_size", type=int, default=256, help="Target global effective batch size (default: 256 for TPU v5e-1, supports 128 / 126; automatically computes grad_accum)")
    parser.add_argument("--target-tokens", "--target_tokens", dest="target_tokens", type=int, default=200_000_000_000, help="Target token ingestion budget for campaign (default: 200,000,000,000 tokens)")
    parser.add_argument("--auto-reasoning-transition", "--auto_reasoning_transition", dest="auto_reasoning_transition", action=argparse.BooleanOptionalAction, default=True, help="Automatically transition from pretraining to deep reasoning curriculum after reaching milestone tokens (default: True)")
    parser.add_argument("--reasoning-transition-tokens", "--reasoning_transition_tokens", dest="reasoning_transition_tokens", type=int, default=41_400_000_000, help="Cumulative token milestone to trigger automatic transition to deep reasoning (default: 41,400,000,000 tokens)")
    parser.add_argument("--final-campaign-tokens", "--final_campaign_tokens", dest="final_campaign_tokens", type=int, default=50_000_000_000, help="Extended token budget target for deep reasoning campaign (default: 50,000,000,000 tokens)")
    parser.add_argument("--reasoning-max-lr", "--reasoning_max_lr", dest="reasoning_max_lr", type=float, default=2e-4, help="Conservative peak learning rate during Deep Reasoning Stage 2 to prevent forgetting Stage 1 foundations (default: 2e-4)")
    parser.add_argument("--reasoning-min-lr", "--reasoning_min_lr", dest="reasoning_min_lr", type=float, default=3e-5, help="Annealed learning rate floor during Deep Reasoning Stage 2 (default: 3e-5)")
    parser.add_argument("--init_from_checkpoint", "--init-from-checkpoint", dest="init_from_checkpoint", type=str, default=None, help="Base foundation checkpoint to load weights from (e.g. step900k)")
    parser.add_argument("--reset_step_counter", "--reset-step-counter", dest="reset_step_counter", action="store_true", default=False, help="Reset step counter to 0 after loading base weights to start new campaign from scratch")
    parser.add_argument("--target-eval-loss", "--target_eval_loss", dest="target_eval_loss", type=float, default=1.60, help="Target held-out evaluation loss for convergence early stopping (default: 1.60)")
    parser.add_argument("--consistent-eval-threshold", "--consistent_eval_threshold", dest="consistent_eval_threshold", type=int, default=3, help="Consecutive evaluations required <= target-eval-loss to confirm convergence (default: 3)")
    parser.add_argument("--teacher_digest", "--teacher-digest", dest="teacher_digest", type=str, default="none", choices=["online", "heuristic", "none"], help="Enable Teacher Digestion & Dynamic Next-Token Loss Weighting (default: none)")
    parser.add_argument("--teacher_model", "--teacher-model", dest="teacher_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="HuggingFace teacher model for digestion")
    parser.add_argument("--teacher_scale", "--teacher-scale", dest="teacher_scale", type=float, default=1.5, help="Scaling factor alpha for dynamic next-token weights")
    parser.add_argument("--encephalization_warmup_steps", "--encephalization-warmup-steps", dest="encephalization_warmup_steps", type=int, default=1000, help="Steps to smoothly transition motor surface readout from core to outer mantle")
    parser.add_argument("--smart-mixture", "--smart_mixture", dest="smart_mixture", action=argparse.BooleanOptionalAction, default=True, help="Enable dynamic DoReMi & generalization-regularized dataset mixture controller (default: True)")
    parser.add_argument("--smart-mixture-eta", "--smart_mixture_eta", dest="smart_mixture_eta", type=float, default=0.08, help="Exponentiated gradient step size for smart mixture")
    parser.add_argument("--smart-mixture-prior-weight", "--smart_mixture_prior_weight", dest="smart_mixture_prior_weight", type=float, default=0.20, help="Dirichlet prior smoothing weight")
    parser.add_argument("--smart-mixture-overfit-thresh", "--smart_mixture_overfit_thresh", dest="smart_mixture_overfit_thresh", type=float, default=0.04, help="Validation loss threshold above best loss triggering overfit penalty")

    parsed = parser.parse_args(args_list)
    raw_args = args_list if args_list is not None else sys.argv[1:]
    parsed.local_steps_explicit = any(
        arg == "--local-steps" or arg == "--local_steps" or arg.startswith("--local-steps=") or arg.startswith("--local_steps=")
        for arg in args_list
    )
    parsed.remat_explicit = any(
        arg == "--remat" or arg.startswith("--remat=")
        for arg in args_list
    )
    parsed.batch_size_explicit = any(
        arg == "--batch-size" or arg == "--batch_size" or arg.startswith("--batch-size=") or arg.startswith("--batch_size=")
        for arg in raw_args
    )
    parsed.total_steps_explicit = any(
        arg in ("--total-steps", "--total_steps", "--steps", "--max-steps", "--max_steps")
        or arg.startswith(("--total-steps=", "--total_steps=", "--steps=", "--max-steps=", "--max_steps="))
        for arg in raw_args
    )
    parsed.effective_batch_size_explicit = any(
        arg == "--effective-batch-size" or arg == "--effective_batch_size" or arg.startswith("--effective-batch-size=") or arg.startswith("--effective_batch_size=")
        for arg in raw_args
    )
    if parsed.total_window is None:
        parsed.total_window = parsed.prompt_len + parsed.seq_len
    return parsed


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
    tokens_getter: Optional[Any] = None,
    target_tokens: int = 41_400_000_000,
    prefetch: bool = True,
    dataset_config: Optional[str] = None,
    dataset_revision: Optional[str] = None,
    retrieval_injection: bool = False,
    data_dir: Optional[str] = None,
):
    """Initializes multi-corpus streaming engine with background prefetching and offline fallback."""
    # Check if HuggingFace datasets library is available for online streaming
    HAS_DATASETS = False
    try:
        from datasets import load_dataset
        HAS_DATASETS = True
    except ImportError:
        pass

    # 100% Loss-Bearing Rectified Contiguous Pretraining Streamer
    if dataset_source in ("pretrain", "rectified", "rectified_pretrain", "clean_pretrain", "midtrain", "midtraining", "deep_reasoning"):
        if not HAS_DATASETS:
            raise RuntimeError(
                f"❌ [DataStreamer Fatal] 'datasets' library is required for online pretraining dataset '{dataset_source}', "
                f"but is not installed. Please install it via: pip install datasets.\n"
                f"   Cleanly stopping training without falling back to mock data."
            )
        from src.rectified_streamer import RectifiedContiguousPretrainStreamer
        is_mid = dataset_source in ("midtrain", "midtraining", "deep_reasoning")
        stage_name = "midtrain" if is_mid else "pretrain"
        print(f"\n📦 Initializing 100% Loss-Bearing Rectified Contiguous Streamer [{dataset_source.upper()}]...")
        if is_mid:
            print(f"   • Data Mixture (Static)     : 40% FineWeb-Edu (>=3.5) | 30% FineMath | 15% Cosmopedia v2 | 15% Code [ZERO SFT]")
        else:
            tgt_tok_val = float(target_tokens or 41_400_000_000)
            cur_tok_val = float(tokens_getter()) if tokens_getter is not None else 0.0
            cur_pct = min(100.0, max(0.0, (cur_tok_val / max(1.0, tgt_tok_val)) * 100.0))
            if cur_tok_val < 15_700_000_000:
                phase_str = "Phase 0 (0-15.7B: 90% FineWeb / 10% Code)"
            elif cur_tok_val < 31_400_000_000:
                phase_str = "Phase 1 (15.7B-31.4B: 75% FineWeb-Edu / 20% Code / 5% Math)"
            elif cur_tok_val < 36_400_000_000:
                phase_str = "Phase 2 (31.4B-36.4B: 70% FineWeb-Edu / 20% Code / 10% Math)"
            else:
                phase_str = "Phase 3 (36.4B-41.4B: 58% FineWeb-Edu / 24% Code / 14% Math / 4% Cosmo)"
            print(f"   • Data Mixture (Token-Based): 4-phase schedule ({tgt_tok_val/1e9:.2f}B token budget)")
            print(f"     - Phases: 90/10 FineWeb (15.7B) -> 75/20/5 Edu (15.7B) -> 70/20/10 Edu (5.0B) -> 58/24/14/4 Edu (5.0B)")
            print(f"     - Active: {phase_str} @ {cur_tok_val/1e9:.3f}B/{tgt_tok_val/1e9:.2f}B tokens ({cur_pct:.1f}%)")
        print(f"   • Input Window (Step-Based) : {total_window} tokens (contiguous unbroken next-token loss)")
        print(f"   • Global Batch              : {batch_size} sequences/step")
        streamer = RectifiedContiguousPretrainStreamer(
            tokenizer=tokenizer,
            batch_size=batch_size,
            block_size=total_window,
            stage=stage_name,
            queue_size=max(16, int(prefetch_queue_size)),
            progress_getter=step_getter,
            total_steps=total_steps,
            tokens_getter=tokens_getter,
            target_tokens=int(target_tokens or 41_400_000_000),
        )
        return streamer, None

    if dataset_source in ("local", "offline"):
        print(f"📦 Initializing background linguistic streaming queue (Local Corpus in {data_dir or 'data/'})...")
        streamer = LinguisticDataStreamer(
            batch_size=batch_size,
            prompt_len=prompt_len,
            target_len=target_len,
            vocab_size=vocab_size,
            data_dir=data_dir,
            queue_size=max(16, int(prefetch_queue_size)),
        )
        return streamer, None

    if not HAS_DATASETS:
        raise RuntimeError(
            f"❌ [DataStreamer Fatal] 'datasets' library is required for online streaming dataset '{dataset_source}', "
            f"but is not installed. Please install it via: pip install datasets.\n"
            f"   Cleanly stopping training without falling back to mock data."
        )

    # Multi-corpus online streaming
    print(f"\n📦 Initializing Online Streaming Dataset: [{dataset_source.upper()}]...")
    data_iter = None
    try:
        import itertools

        raw_iter = build_online_dataset_iterator(
            dataset_source=dataset_source,
            step_getter=step_getter,
            total_steps=total_steps,
            inject_retrieval=retrieval_injection,
            dataset_config=dataset_config,
            dataset_revision=dataset_revision,
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
                tokenizer, data_iter, batch_size=batch_size, total_window=total_window, buckets=buckets, framework="numpy", return_feedback=True, target_pad_id=-100
            )
        else:
            buckets = [("static", prompt_len, target_len)]
            batcher = ElasticBucketBatcher(
                tokenizer, data_iter, batch_size=batch_size, total_window=prompt_len + target_len, buckets=buckets, framework="numpy", return_feedback=True, target_pad_id=-100
            )

        q_depth = max(16, int(prefetch_queue_size))
        if not prefetch:
            return None, batcher
        prefetched = PrefetchedBatcher(batcher, queue_size=q_depth)
        print(f"⚡ Background Asynchronous Batch Prefetcher Active (Queue Depth: {q_depth} batches)")
        return None, prefetched

    except Exception as e:
        raise RuntimeError(
            f"❌ [DataStreamer Fatal] Online dataset '{dataset_source}' failed to initialize: {e}.\n"
            f"   Cleanly stopping training without falling back to mock data."
        ) from e


def _iter_curriculum_stage_segments(
    start_step: int,
    eff_batch_size: float,
    base_prompt_len: int,
    base_seq_len: int,
    context_curriculum: bool = False,
    context_stages: Optional[List[int]] = None,
    context_stage_steps: int = 100_000,
):
    """Yield `(steps_in_segment, tok_per_step)` pairs across curriculum stages from `start_step`.

    `steps_in_segment` is `None` for the final open-ended stage.
    """
    eff_bs = float(eff_batch_size)
    if eff_bs <= 0.0:
        return

    if not context_curriculum or not context_stages:
        yield None, max(1.0, eff_bs * float(base_prompt_len + base_seq_len))
        return

    stage_len_steps = max(1, int(context_stage_steps))
    clamped_start = max(0, int(start_step))
    curr_stage_idx = min(len(context_stages) - 1, clamped_start // stage_len_steps)
    elapsed = 0

    for s_idx in range(curr_stage_idx, len(context_stages)):
        s_len = context_stages[s_idx]
        s_tok_per_step = max(1.0, eff_bs * float(base_prompt_len + s_len))
        if s_idx == len(context_stages) - 1:
            yield None, s_tok_per_step
            return
        steps_left_in_stage = stage_len_steps - ((clamped_start + elapsed) % stage_len_steps)
        yield steps_left_in_stage, s_tok_per_step
        elapsed += steps_left_in_stage


def calculate_curriculum_aware_target_steps(
    target_budget: float,
    eff_batch_size: float,
    base_prompt_len: int,
    base_seq_len: int,
    context_curriculum: bool = False,
    context_stages: Optional[List[int]] = None,
    context_stage_steps: int = 100_000,
    start_step: int = 0,
) -> int:
    """Calculates exact steps required to ingest a target token budget,

    accounting for BOS-only prompts in pretraining and expanding context
    windows across curriculum stages.
    """
    if target_budget <= 0.0 or float(eff_batch_size) <= 0.0:
        return 0

    rem = float(target_budget)
    tot_steps = 0

    for steps_cap, s_tok_per_step in _iter_curriculum_stage_segments(
        start_step=start_step,
        eff_batch_size=eff_batch_size,
        base_prompt_len=base_prompt_len,
        base_seq_len=base_seq_len,
        context_curriculum=context_curriculum,
        context_stages=context_stages,
        context_stage_steps=context_stage_steps,
    ):
        if steps_cap is None:
            tot_steps += int(math.ceil(rem / s_tok_per_step))
            break
        stage_cap_tok = float(steps_cap) * s_tok_per_step
        if rem <= stage_cap_tok:
            tot_steps += int(math.ceil(rem / s_tok_per_step))
            break
        tot_steps += steps_cap
        rem -= stage_cap_tok
    return tot_steps


def calculate_curriculum_window_tokens(
    start_step: int,
    end_step: int,
    eff_batch_size: float,
    base_prompt_len: int,
    base_seq_len: int,
    context_curriculum: bool = False,
    context_stages: Optional[List[int]] = None,
    context_stage_steps: int = 100_000,
) -> float:
    """Calculate nominal tokens ingested over `[start_step, end_step)` under the stage schedule."""
    steps_rem = max(0, int(end_step) - int(start_step))
    if steps_rem <= 0 or float(eff_batch_size) <= 0.0:
        return 0.0

    total_tokens = 0.0
    for steps_cap, s_tok_per_step in _iter_curriculum_stage_segments(
        start_step=start_step,
        eff_batch_size=eff_batch_size,
        base_prompt_len=base_prompt_len,
        base_seq_len=base_seq_len,
        context_curriculum=context_curriculum,
        context_stages=context_stages,
        context_stage_steps=context_stage_steps,
    ):
        take = steps_rem if steps_cap is None else min(steps_rem, steps_cap)
        total_tokens += float(take) * s_tok_per_step
        steps_rem -= take
        if steps_rem <= 0:
            break
    return total_tokens


def calibrate_curriculum_total_steps(
    step: int,
    start_step: int,
    cumulative_tokens: int,
    resumed_cum_tokens: int,
    target_token_budget: float,
    eff_batch_size: float,
    base_prompt_len: int,
    base_seq_len: int,
    context_curriculum: bool = False,
    context_stages: Optional[List[int]] = None,
    context_stage_steps: int = 100_000,
    min_empirical_steps: int = 20,
) -> int:
    """Recalibrate campaign `total_steps` at `step` across remaining curriculum stages.

    Scales the nominal effective batch size by `observed_tokens / nominal_tokens` over
    `[start_step, step)` once `min_empirical_steps` have elapsed, avoiding Stage-0 linear
    overestimation across upcoming longer-context curriculum stages.
    """
    toks_left = max(0.0, float(target_token_budget) - float(cumulative_tokens))
    if toks_left <= 0.0:
        return int(step)

    steps_done = max(0, int(step) - int(start_step))
    toks_done = max(0, int(cumulative_tokens) - int(resumed_cum_tokens))
    calibrated_eff_batch = float(eff_batch_size)

    if steps_done >= min_empirical_steps and toks_done > 0:
        nominal_done = calculate_curriculum_window_tokens(
            start_step=start_step,
            end_step=step,
            eff_batch_size=eff_batch_size,
            base_prompt_len=base_prompt_len,
            base_seq_len=base_seq_len,
            context_curriculum=context_curriculum,
            context_stages=context_stages,
            context_stage_steps=context_stage_steps,
        )
        if nominal_done > 0.0:
            calibrated_eff_batch *= float(toks_done) / nominal_done

    rem_steps = calculate_curriculum_aware_target_steps(
        target_budget=toks_left,
        eff_batch_size=calibrated_eff_batch,
        base_prompt_len=base_prompt_len,
        base_seq_len=base_seq_len,
        context_curriculum=context_curriculum,
        context_stages=context_stages,
        context_stage_steps=context_stage_steps,
        start_step=step,
    )
    return int(step) + rem_steps


def main(args=None):
    if args is None:
        args = parse_args()

    # 0. Hugging Face Authentication & Token Resolution
    from src.hf_utils import setup_hf_token
    setup_hf_token(getattr(args, "hf_token", None), verbose=True)

    # Graceful shutdown handler
    shutdown_requested = False

    def sig_handler(sig, frame):
        nonlocal shutdown_requested
        print(f"\n⚠️ Caught signal {sig}. Requesting graceful shutdown...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Sequence Length Transition & Context Budget Progression
    initial_seq_len = args.seq_len
    target_seq_len = getattr(args, "target_seq_len", None)
    seq_trans_steps = getattr(args, "seq_transition_steps", 0)

    if target_seq_len is not None and target_seq_len > args.seq_len:
        if seq_trans_steps <= 0:
            print(f"\n🔄 [Sequence Transition] Advancing target sequence length immediately from {args.seq_len} -> {target_seq_len} tokens for deep reasoning!")
            args.seq_len = target_seq_len
        else:
            print(f"\n🔄 [Sequence Transition Scheduled] Gradual transition planned from {args.seq_len} -> {target_seq_len} tokens over {seq_trans_steps:,} steps.")

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

    # 2. Model Configuration & Parameters
    config_preset = "v5_small" if args.preset in ("tpu_v5e_1", "tpu_v6e_1", "tpu_v6e") else args.preset
    if config_preset not in PRESETS_V5:
        config_preset = "v5_small"
    cfg_kwargs = PRESETS_V5[config_preset].copy()
    cfg_kwargs["use_rope"] = bool(args.use_rope)
    cfg_kwargs["max_phase_steps"] = int(args.max_phase_steps)
    if getattr(args, "sensory_position_steps", None) is not None:
        cfg_kwargs["sensory_position_steps"] = int(args.sensory_position_steps)
    if getattr(args, "num_mtp_heads", None) is not None:
        cfg_kwargs["num_mtp_heads"] = args.num_mtp_heads
    if getattr(args, "use_adaptive_thinking", None) is not None:
        cfg_kwargs["use_adaptive_thinking"] = args.use_adaptive_thinking
    if getattr(args, "thinking_cycles", None) is not None:
        cfg_kwargs["thinking_cycles"] = args.thinking_cycles
    if getattr(args, "max_thinking_cycles", None) is not None:
        cfg_kwargs["max_thinking_cycles"] = args.max_thinking_cycles
    if getattr(args, "vocab_size", None) is not None:
        cfg_kwargs["vocab_size"] = int(args.vocab_size)
    cfg_kwargs["use_selective_loss"] = bool(getattr(args, "selective_loss", False))
    cfg_kwargs["selective_focal_gamma"] = float(getattr(args, "selective_focal_gamma", 0.5))
    cfg_kwargs["selective_noise_cap"] = float(getattr(args, "selective_noise_cap", 8.5))
    cfg_kwargs["z_loss_weight"] = float(getattr(args, "z_loss_weight", 0.0))
    cfg_kwargs = apply_from_scratch_arch_upgrades(cfg_kwargs, args)
    config = JAXConfigV5(**cfg_kwargs)
    params, buffers = init_params_v5(jax.random.PRNGKey(42), config)

    # Convert parameter arrays to target precision
    from src.jax.spheroid_optimizer import is_manifold_operator, VOCAB_KEYS
    params = {
        k: (v.astype(target_dtype) if (is_manifold_operator(k, v) or k in VOCAB_KEYS) else v.astype(jnp.float32))
        for k, v in params.items()
    }

    total_params = sum(p.size for p in params.values())
    active_compute = sum(p.size for k, p in params.items() if "embedding" not in k)
    print(f"\nModel Config    : {args.preset} | Shells: {config.num_shells} | Surface Dim: {config.surface_dim} | Radial Dim: {config.radial_dim} | Thinking Cycles: {config.thinking_cycles}")
    print(f"Total Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    print(f"Active Compute  : {active_compute:,} (~{active_compute/1e6:.1f}M) ({active_compute/total_params*100:.2f}%)")
    print(f"Optimizer       : {args.optimizer.upper()} ({'SCMER (Polar NS-5 + Spectral Clamp + Selective Cosine Margin)' if args.optimizer in ('spheroid', 'scmer') else 'Optax AdamW'})")

    # 3. Tokenizer (validate vocabulary compatibility upfront across all data paths)
    from src.jax.data_streamer import validate_tokenizer_vocab

    tokenizer = build_tokenizer(args.tokenizer_name)
    validate_tokenizer_vocab(tokenizer, config.vocab_size)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    # 5. Checkpoint Manager & Resumption
    args.cloud_token = get_sync_token(args.cloud_token)
    os.environ["BIOFLO_SYNC_TOKEN"] = args.cloud_token
    os.environ["CEREBRO_SYNC_TOKEN"] = args.cloud_token
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.scratch_dir, exist_ok=True)

    ckpt_mgr = AsyncCheckpointManager(
        local_dir=args.scratch_dir,
        target_dir=args.checkpoint_dir,
        cloud_sync=args.cloud_sync,
        cloud_url=args.cloud_url,
        cloud_token=args.cloud_token,
        min_cloud_upload_interval_sec=120.0,
        cloud_weights_only=args.cloud_weights_only,
        save_optimizer_state=args.save_optimizer_state,
    )

    start_step = 0
    best_eval_loss = float("inf")
    last_eval_loss = float("inf")
    initial_ema_loss: Optional[float] = None
    sync_model_name = getattr(args, "sync_model", None) or ("v5_small" if args.preset in ("tpu_v5e_1", "tpu_v6e_1", "tpu_v6e") else args.preset)
    local_latest = os.path.join(args.checkpoint_dir, f"{args.preset}_latest.pt")
    fallback_latest = os.path.join(args.checkpoint_dir, f"{sync_model_name}_latest.pt")

    # 5.1 Run Identification (strictly link token ingestion and telemetry to model and run)
    run_id_file = os.path.join(args.checkpoint_dir, "run_id.txt")
    if getattr(args, "run_id", None):
        run_id = str(args.run_id).strip()
    elif not args.from_scratch and os.path.exists(run_id_file):
        try:
            with open(run_id_file, "r") as rf:
                run_id = rf.read().strip()
        except Exception:
            run_id = f"run_{sync_model_name}_{int(time.time())}"
    else:
        run_id = f"run_{sync_model_name}_{int(time.time())}"

    try:
        with open(run_id_file, "w") as rf:
            rf.write(run_id)
    except Exception:
        pass
    print(f"Active Campaign : Model='{sync_model_name}', Run='{run_id}'")

    # Checkpoint Restoration or Clean Scratch Start
    if args.from_scratch:
        print("\n🌱 [from_scratch=True] Initializing clean Step 0 training with random orthogonal weights (zero foundation copying)!")
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
            old_j = old_f.replace(".pt", ".json")
            if os.path.exists(old_j):
                try:
                    os.remove(old_j)
                except Exception:
                    pass

        # Fix starting model at Step 0
        step0_meta = {
            "step": 0,
            "loss": float("inf"),
            "eval_loss": float("inf"),
            "config": dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config,
            "tokens_by_dataset": {},
            "lifetime_tokens": 0,
            "sync_model": sync_model_name,
            "run_id": run_id,
        }
        step0_fname = f"{args.preset}_step0.pt"
        step0_path = os.path.join(args.checkpoint_dir, step0_fname)
        save_jax_to_torch_pt(params, buffers, local_latest, meta=step0_meta)
        save_jax_to_torch_pt(params, buffers, step0_path, meta=step0_meta)
        print(f"📦 [Starting Model Fixed] Clean base architecture saved at Step 0:\n   ├─ {local_latest}\n   └─ {step0_path}")

        if args.cloud_sync:
            try:
                # 1. Reset remote token tracker online for this model & run
                push_remote_token_tracker(
                    model=sync_model_name,
                    cumulative_tokens=0,
                    target_tokens=getattr(args, "target_tokens", 41_400_000_000),
                    tokens_by_dataset={},
                    effective_batch_size=args.effective_batch_size,
                    step=0,
                    loss=0.0,
                    run_id=run_id,
                    url=args.cloud_url,
                    token=args.cloud_token,
                    force_reset=True,
                )
                print(f"🌐 [cloud_sync] Online token counter reset to 0 for Model='{sync_model_name}', Run='{run_id}'")
            except Exception as e_tok:
                print(f"⚠️ [cloud_sync] Notice on initial token counter reset: {e_tok}")

            try:
                step0_stage = f"{local_latest}.upload.{os.getpid()}.{time.time_ns()}.pt"
                save_jax_to_torch_pt(params, buffers, step0_stage, meta=step0_meta, opt_state=None, bare_bf16=True)
                push_checkpoint_async(
                    step0_stage,
                    step=0,
                    loss=9.9999,
                    target_type="latest",
                    model=sync_model_name,
                    url=args.cloud_url,
                    token=args.cloud_token,
                )
                print(f"☁️ [cloud_sync] Fixed starting model registered on bioflo.in for model='{sync_model_name}' at Step 0.")
            except Exception as e_sync:
                print(f"⚠️ [cloud_sync] Notice on initial step 0 save: {e_sync}")
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
                    if remote_step > 0:
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
                        else:
                            # Autonomous Fallback: If primary target fails or is corrupted, try the alternative slot ('best' vs 'latest')
                            alt_target = "best" if args.resume_target == "latest" else "latest"
                            alt_slot_model, alt_step, alt_loss = select_highest_step_slot(
                                remote_manifest, sync_model_name, alt_target
                            )
                            if alt_step > 0:
                                print(
                                    f"⚠️ [cloud_sync] Primary slot '{args.resume_target}.pt' failed verification. "
                                    f"Falling back to candidate '{alt_target}.pt' (Step {alt_step:,}, Loss {alt_loss:.4f})..."
                                )
                                alt_dest = os.path.join(args.checkpoint_dir, f"{args.preset}_{alt_target}.pt")
                                alt_ok = pull_checkpoint(
                                    alt_dest,
                                    target_type=alt_target,
                                    model=alt_slot_model,
                                    url=args.cloud_url,
                                    token=args.cloud_token,
                                )
                                if alt_ok and os.path.exists(alt_dest):
                                    local_latest = alt_dest
                                    dest_latest = os.path.join(args.checkpoint_dir, f"{args.preset}_latest.pt")
                                    if os.path.abspath(alt_dest) != os.path.abspath(dest_latest):
                                        try:
                                            shutil.copyfile(alt_dest, dest_latest)
                                            local_latest = dest_latest
                                        except Exception:
                                            pass
                                    print(f"✓ Restored alternative candidate checkpoint from cloud hub: {alt_dest}")
                    else:
                        print(f"ℹ️ [cloud_sync] No existing checkpoint on bioflo.in for {args.preset} (Step 0). Ready for clean training.")
            except Exception as e:
                print(f"⚠️ [cloud_sync] Notice: {e}")

        # Check if an existing checkpoint for THIS preset is already present (local or restored from cloud)
        # Priority: Always pick the checkpoint with the highest step number in checkpoint_dir
        discovered_ckpt = None
        existing_preset_step = 0
        found_ckpt, found_step, found_loss = discover_latest_checkpoint(args.checkpoint_dir, preset=args.preset)
        if found_ckpt and found_step > 0:
            discovered_ckpt = found_ckpt
            existing_preset_step = found_step

        target_local = local_latest if os.path.exists(local_latest) else (fallback_latest if os.path.exists(fallback_latest) else None)
        if target_local and os.path.exists(target_local):
            try:
                chk_test = safe_torch_load(target_local, map_location="cpu", weights_only=True)
                step_t = int(chk_test.get("step", 0))
                if step_t > existing_preset_step:
                    existing_preset_step = step_t
                    discovered_ckpt = target_local
            except Exception:
                pass

        if existing_preset_step > 0:
            print(
                f"✓ Found existing training checkpoint for {args.preset} at Step {existing_preset_step:,} ({discovered_ckpt}).\n"
                f"   Resuming directly (foundation core re-copying completely bypassed to preserve trained weights)."
            )
        else:
            # Only if no active preset checkpoint exists, check for explicit foundation checkpoint
            candidate_init = getattr(args, "init_from_checkpoint", None)
            if candidate_init:
                candidate_paths = [
                    candidate_init,
                    os.path.join(args.checkpoint_dir, candidate_init),
                    os.path.join(args.checkpoint_dir, f"{candidate_init}.pt"),
                    os.path.join("checkpoints", candidate_init),
                    os.path.join("checkpoints", f"{candidate_init}.pt"),
                    os.path.join("/home/ajit/Documents/Startup/Active_Development/cerebro_checkpoints_archive", candidate_init),
                    os.path.join("/home/ajit/Documents/Startup/Active_Development/cerebro_checkpoints_archive", f"{candidate_init}.pt"),
                    os.path.join("/home/ajit/Documents/Startup/Active_Development/cerebro_checkpoints_archive/archived_v5_small_runs", candidate_init),
                    os.path.join("/home/ajit/Documents/Startup/Active_Development/cerebro_checkpoints_archive/archived_v5_small_runs", f"{candidate_init}.pt"),
                    "/home/ajit/Documents/Startup/Active_Development/cerebro_checkpoints_archive/archived_v5_small_runs/v5_small_latest_step900k.pt"
                    if ("900" in str(candidate_init)) else None,
                    "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_latest_step900k.pt"
                    if ("900" in str(candidate_init)) else None,
                    "/home/ajit/Documents/Startup/Active_Development/reasoning_agent/checkpoints/v5_small_latest_step900k.pt"
                    if ("900" in str(candidate_init)) else None,
                ]
                for p in candidate_paths:
                    if p and os.path.exists(p):
                        discovered_ckpt = p
                        break

                # If explicit foundation checkpoint not found locally, pull it from bioflo.in
                if not discovered_ckpt and args.cloud_sync:
                    c_str = str(candidate_init).lower()
                    pull_model = "v5_small" if ("v5_small" in c_str or "step920" in c_str or "step900" in c_str or "foundation" in c_str) else None
                    dest_file = candidate_init if candidate_init.endswith(".pt") else f"{candidate_init}.pt"
                    dest_path = os.path.join(args.checkpoint_dir, dest_file)
                    print(f"🔄 [cloud_sync] Explicit foundation checkpoint '{candidate_init}' requested. Pulling model='{pull_model or 'v5_small'}' from bioflo.in...")
                    pulled_ok = pull_checkpoint(
                        dest_path,
                        target_type="latest",
                        model=pull_model or "v5_small",
                        url=args.cloud_url,
                        token=args.cloud_token,
                    )
                    if pulled_ok and os.path.exists(dest_path):
                        discovered_ckpt = dest_path
                        print(f"✓ Restored foundation checkpoint from cloud hub: {dest_path}")

                if discovered_ckpt:
                    print(f"📦 Loading base model weights from explicit checkpoint: {discovered_ckpt}")
                else:
                    print(f"⚠️ Explicit checkpoint '{candidate_init}' not found locally or on cloud hub. Searching default checkpoint paths...")

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
                ckpt = safe_torch_load(discovered_ckpt, map_location="cpu", weights_only=True)
                start_step = int(ckpt.get("step", 0))
                saved_eval = float(ckpt.get("eval_loss", float("inf")))
                saved_train_loss = float(ckpt.get("loss", float("inf")))
                if saved_eval > 7.0 and saved_train_loss < 5.0:
                    best_eval_loss = float("inf")
                    last_eval_loss = float("inf")
                else:
                    best_eval_loss = saved_eval if math.isfinite(saved_eval) else saved_train_loss
                    last_eval_loss = best_eval_loss
                restored_ema = ckpt.get("ema_loss", ckpt.get("loss", ckpt.get("eval_loss", None)))
                if restored_ema is not None and math.isfinite(float(restored_ema)) and float(restored_ema) > 0.0:
                    initial_ema_loss = float(restored_ema)
                is_resuming_active = (existing_preset_step > 0 or not getattr(args, "from_scratch", False))
                if not getattr(args, "from_scratch", False):
                    cfg_kwargs, arch_changed = resolve_checkpoint_arch_overrides(ckpt, cfg_kwargs)
                    if arch_changed:
                        config = JAXConfigV5(**cfg_kwargs)
                        params, buffers = init_params_v5(jax.random.PRNGKey(42), config)
                adapt_res = adapt_and_load_checkpoint_weights(
                    ckpt, params, target_dtype, config, buffers=buffers,
                    apply_net2net=False
                )
                if isinstance(adapt_res, tuple):
                    restored, buffers = adapt_res
                else:
                    restored = adapt_res
                params = {
                    k: (v.astype(target_dtype) if (is_manifold_operator(k, v) or k in VOCAB_KEYS) else v.astype(jnp.float32))
                    for k, v in params.items()
                }
                ema_info = f" (EMA: {initial_ema_loss:.4f})" if initial_ema_loss is not None else ""
                print(f"✓ Resumed from {os.path.basename(discovered_ckpt)} at Step {start_step:,} (Loss: {best_eval_loss:.4f}{ema_info})")
                if restored == len(params):
                    print(f"✅ Restored & adapted {restored}/{len(params)} parameter arrays into JAX engine (100% parity)!")
                else:
                    print(f"⚠️ Restored & adapted {restored}/{len(params)} parameter arrays into JAX engine ({len(params) - restored} missing).")
                
                # Campaign Reset: Keep weights, reset schedule & step counter ONLY when seeding from external foundation model
                if getattr(args, "reset_step_counter", False):
                    if discovered_ckpt == target_local and existing_preset_step > 0:
                        print(f"ℹ️ [Resumption Safeguard] Preserving Step {start_step:,} for active {args.preset} run (reset_step_counter ignored on resumption).")
                    else:
                        start_step = 0
                        best_eval_loss = float("inf")
                        last_eval_loss = float("inf")
                        initial_ema_loss = None
                        print("🌱 [Campaign Reset Active] Step counter reset to 0! Calibrating fresh schedule from adapted base weights.")
                else:
                    pass
            except Exception as e:
                print(f"⚠️ Checkpoint load notice: {e}. Starting fresh.")

    if args.reset_best:
        best_eval_loss = float("inf")
        last_eval_loss = float("inf")
        print("🔄 [Benchmark Recalibration] Reset best_eval_loss tracker baseline.")

    # 5. Context Stage Curriculum & Horizon Sizing
    min_curriculum_window = int(getattr(args, "min_context_window", 512) or 512)
    max_curriculum_window = args.total_window
    if max_curriculum_window is None:
        if getattr(args, "context_curriculum", False):
            max_curriculum_window = 4096  # Curriculum defaults to advancing through the full 4K horizon
        else:
            max_curriculum_window = args.prompt_len + args.seq_len
    if getattr(args, "target_seq_len", None) is not None:
        max_curriculum_window = max(max_curriculum_window, args.prompt_len + args.target_seq_len)

    context_stages = [w for w in [512, 1024, 2048, 4096] if w >= min_curriculum_window and w <= max_curriculum_window]
    if not context_stages:
        context_stages = [max_curriculum_window]

    if getattr(args, "context_curriculum", False):
        context_stage_steps = args.context_stage_steps if (getattr(args, "context_stage_steps", 0) and args.context_stage_steps > 0) else 100_000
        active_stage_idx = min(len(context_stages) - 1, start_step // context_stage_steps)
        args.total_window = context_stages[active_stage_idx]
        print(
            f"🌊 [Window Expansion Schedule (Step-Based)] Windows: {context_stages} tokens | {context_stage_steps:,} steps/window\n"
            f"   Initial Window Stage {active_stage_idx}: Input Window = {args.total_window} tokens"
        )
    else:
        context_stage_steps = max(1, args.total_steps)
        active_stage_idx = 0
        args.total_window = max(min_curriculum_window, args.total_window or min_curriculum_window)

    is_pretrain_source = args.dataset_source in ("pretrain", "rectified", "rectified_pretrain", "clean_pretrain")
    if is_pretrain_source:
        args.seq_len = args.total_window

    # 6. Hardware Optimization & Auto-Tuning
    remat_choice = True if args.remat.lower() == "true" else False
    max_bs = getattr(args, "max_batch_size", None)
    worst_case_seq = max(args.seq_len, args.total_window, getattr(args, "target_seq_len", 0) or 0)
    require_tuned_remat = (
        args.seq_len >= 1024
        or args.total_window >= 1024
        or getattr(args, "context_curriculum", False)
    )

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
            prompt_len=(1 if is_pretrain_source else args.prompt_len),
            max_batch_size=max_bs,
            worst_case_seq_len=worst_case_seq,
            model_config=config,
            num_devices=num_devices,
            require_remat=require_tuned_remat,
        )
        if not getattr(args, "batch_size_explicit", False):
            args.batch_size = best_cfg["batch_size"]
        remat_choice = best_cfg["remat"]
        print(f"\n⚡ Applying Tuned Hyperparameters: batch_size={args.batch_size}, remat={remat_choice}")
        if "tok_per_sec" in best_cfg and best_cfg["tok_per_sec"] > 0:
            m_type = best_cfg.get("memory_type", "HBM" if ("tpu" in dev_kind or "tpu" in backend.lower()) else "VRAM")
            v_gb = best_cfg.get("memory_gb", best_cfg.get("vram_gb", 0))
            v_pct = best_cfg.get("memory_utilization_pct", best_cfg.get("vram_utilization_pct", 0))
            sps = best_cfg.get("steps_per_sec", 0.0)
            sps_str = f"{sps:.2f} steps/s | " if sps and sps > 0 else ""
            print(f"⚡ Advisory Single-Device Benchmark Speed: {sps_str}{best_cfg['tok_per_sec']:,.0f} tok/s ({m_type}: {v_gb:.1f} GB, {v_pct:.1f}% utilization)")
            print(f"   ℹ️ [Notice] Throughput is an advisory single-device micro-benchmark proxy. Production multi-device rate is governed by Multi-TPU Local-SGD execution.")
            if getattr(args, "context_curriculum", False):
                print(f"   ℹ️ [Curriculum Notice] Benchmark measured initial stage ({args.seq_len} tokens). Worst-case memory ceiling was guarded at {worst_case_seq} tokens.")
    elif max_bs is not None:
        args.batch_size = min(int(args.batch_size), int(max_bs))

    # Strict HBM Safety Enforcement for Long-Context Horizons (respects explicit --remat=false)
    if require_tuned_remat and not remat_choice and not getattr(args, "remat_explicit", False):
        print(f"🛡️ [HBM Safety Guard] Sequence window {max(args.seq_len, args.total_window)} >= 1024 requires activation rematerialization. Enabling remat=True.")
        remat_choice = True

    # Scale micro-batch if necessary to keep micro-batch tokens bounded
    # On 16GB accelerators (TPU v5e-1), 65,536 tokens per microbatch fits comfortably in HBM (4.5GB / 28.9%).
    # 131,072 tokens per microbatch at seq >= 1024 causes OOM due to 32k-vocab logit tensors and optimizer states.
    max_micro_tokens = 65536
    if args.batch_size * args.seq_len > max_micro_tokens:
        old_bs = args.batch_size
        capped_bs = max(1, max_micro_tokens // args.seq_len)
        if capped_bs >= 16:
            capped_bs = (capped_bs // 16) * 16
        elif capped_bs >= 8:
            capped_bs = (capped_bs // 8) * 8
        args.batch_size = capped_bs
        print(f"🛡️ [HBM Safety Guard] Scaling micro batch size {old_bs} -> {args.batch_size} for sequence length {args.seq_len} (caps micro tokens at {args.batch_size * args.seq_len:,} <= {max_micro_tokens:,})")

    # 7. Effective Batch Size & Target Token Budget Scheduling
    if getattr(args, "effective_batch_size_explicit", False) and args.effective_batch_size > 0:
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

    # Honest token slot accounting:
    # In pretraining (RectifiedContiguousPretrainStreamer), prompts are 1 token (BOS seed)
    # and targets are 100% loss-bearing chunks of seq_len tokens.
    # In SFT / elastic mode, prompts are prompt_len and targets are seq_len.
    if is_pretrain_source:
        real_prompt_slots = 1
        real_target_slots = args.seq_len
    else:
        real_prompt_slots = args.prompt_len
        real_target_slots = args.seq_len

    tokens_per_opt_step = eff_batch_size * (real_prompt_slots + real_target_slots)

    if not getattr(args, "total_steps_explicit", False) and getattr(args, "target_tokens", None) is not None and args.target_tokens > 0:
        target_budget = float(args.target_tokens)
        resumed_cum_tokens = 0
        if "ckpt" in locals() and isinstance(ckpt, dict) and not getattr(args, "reset_step_counter", False):
            resumed_cum_tokens = int(ckpt.get("cumulative_tokens", 0))

        remaining_budget = max(0.0, target_budget - float(resumed_cum_tokens))

        calculated_steps = calculate_curriculum_aware_target_steps(
            target_budget=remaining_budget,
            eff_batch_size=eff_batch_size,
            base_prompt_len=real_prompt_slots,
            base_seq_len=real_target_slots,
            context_curriculum=getattr(args, "context_curriculum", False),
            context_stages=context_stages if "context_stages" in locals() else None,
            context_stage_steps=getattr(args, "context_stage_steps", 100_000),
            start_step=start_step,
        )

        if start_step > 0:
            args.total_steps = start_step + calculated_steps
            print(
                f"🎯 [Target Token Budget Resumption] Target: {target_budget/1e9:.2f}B Tokens | "
                f"Already Ingested: {resumed_cum_tokens/1e9:.2f}B Tokens | "
                f"Remaining: {remaining_budget/1e9:.2f}B Tokens -> Scheduled for {calculated_steps:,} additional Steps "
                f"(Campaign Ends at Step {args.total_steps:,} at ~{tokens_per_opt_step:,} tok/step)"
            )
        else:
            args.total_steps = calculated_steps
            print(
                f"🎯 [Target Token Budget] {target_budget/1e9:.2f} Billion Tokens -> "
                f"Scheduled for {args.total_steps:,} Steps (at ~{tokens_per_opt_step:,} tok/step)"
            )

    # 6. Spheroid Optimizer & JIT / Multi-TPU Local-SGD Train Step Engine
    grad_accum = max(1, int(getattr(args, "grad_accum", 1)))
    oom_handler = TrainingOOMHandler(
        initial_batch_size=args.batch_size,
        effective_batch_size=eff_batch_size,
        initial_grad_accum=grad_accum,
        remat=remat_choice,
        min_batch_size=1,
        verbose=True,
    )
    multi_mode = getattr(args, "multi_tpu", "auto")
    use_multi_tpu_local_sgd = (num_devices > 1) and (args.optimizer in ("spheroid", "scmer")) and (multi_mode != "single")
    if num_devices > 1 and args.optimizer not in ("spheroid", "scmer"):
        print(f"ℹ️ [Multi-Device Dispatch] Multi-device pmap Local-SGD is specialized for the SCMER / Spheroid optimizer (polar NS-5 + spectral clamp). Running {args.optimizer.upper()} in standard single-accelerator JIT mode.")
    eff_num_devices = num_devices if use_multi_tpu_local_sgd else 1
    base_local_steps = max(1, int(getattr(args, "local_steps", 4)))
    if use_multi_tpu_local_sgd and num_devices >= 8 and not getattr(args, "local_steps_explicit", False):
        base_local_steps = max(base_local_steps, 8)
    eff_local_steps = max(1, base_local_steps * grad_accum) if use_multi_tpu_local_sgd else 1
    step_stride = eff_num_devices * eff_local_steps

    # Multi-TPU rounds select a single elastic bucket and bypass the async
    # prefetch wrapper, which otherwise mixes shapes within a pmap scan.

    param_masks = None

    pmap_round_fn = None
    rep_params = None
    rep_opt_state = None
    apply_opt_step_frozen = None
    apply_opt_step_anneal = None
    train_step_fn_frozen = None
    train_step_fn_anneal = None

    eff_scmer_lambda = args.neg_cov_lambda if args.neg_cov_lambda is not None else args.scmer_lambda
    eff_scmer_subsample = args.neg_cov_subsample if args.neg_cov_subsample is not None else args.scmer_subsample
    eff_scmer_interval = args.neg_cov_interval if args.neg_cov_interval is not None else args.scmer_interval
    eff_scmer_margin = args.scmer_margin
    eff_scmer_pairs = args.scmer_pair_samples

    if args.optimizer in ("spheroid", "scmer"):
        if start_step > 0 and "discovered_ckpt" in locals() and discovered_ckpt and os.path.exists(discovered_ckpt):
            from src.jax.checkpoint import restore_spheroid_opt_state
            opt_state = restore_spheroid_opt_state(discovered_ckpt, params)
        else:
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
                scmer_lambda=eff_scmer_lambda,
                scmer_subsample=eff_scmer_subsample,
                scmer_interval=eff_scmer_interval,
                scmer_margin=eff_scmer_margin,
                scmer_pair_samples=eff_scmer_pairs,
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
                scmer_lambda=eff_scmer_lambda,
                scmer_subsample=eff_scmer_subsample,
                scmer_interval=eff_scmer_interval,
                scmer_margin=eff_scmer_margin,
                scmer_pair_samples=eff_scmer_pairs,
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
                scmer_lambda=eff_scmer_lambda,
                scmer_subsample=eff_scmer_subsample,
                scmer_interval=eff_scmer_interval,
                scmer_margin=eff_scmer_margin,
                scmer_pair_samples=eff_scmer_pairs,
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
                scmer_lambda=eff_scmer_lambda,
                scmer_subsample=eff_scmer_subsample,
                scmer_interval=eff_scmer_interval,
                scmer_margin=eff_scmer_margin,
                scmer_pair_samples=eff_scmer_pairs,
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
                scmer_lambda=eff_scmer_lambda,
                scmer_subsample=eff_scmer_subsample,
                scmer_interval=eff_scmer_interval,
                scmer_margin=eff_scmer_margin,
                scmer_pair_samples=eff_scmer_pairs,
            ) if param_masks is not None else (None, apply_opt_step_anneal)
            apply_opt_step = apply_opt_step_anneal

            def train_step_fn(p, opt_s, q_batch, r_batch, cur_lr, feedback_tokens=None, sampling_p=1.0, encephalization_alpha=1.0, target_weights=None):
                loss_val, grads = compute_grad_step(
                    p, q_batch, r_batch,
                    feedback_tokens=feedback_tokens,
                    sampling_p=sampling_p,
                    deliberation_mode=args.deliberation_mode,
                    encephalization_alpha=encephalization_alpha,
                    target_weights=target_weights,
                )
                new_p, new_opt_s = apply_opt_step(p, opt_s, grads, cur_lr)
                return new_p, new_opt_s, loss_val

    else:
        if not HAS_OPTAX:
            raise ImportError("optax is required for AdamW training. Install via: pip install optax")
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(b1=0.9, b2=0.95),
            optax.add_decayed_weights(weight_decay=args.weight_decay),
        )
        optax_init_state = optimizer.init(params)
        opt_state = {
            "optax": optax_init_state,
            "diag": {
                "grad_norm": jnp.array(0.0, dtype=jnp.float32),
                "was_clipped": jnp.array(0.0, dtype=jnp.float32),
                "was_skipped": jnp.array(0.0, dtype=jnp.float32),
                "u_ratio_manifold": jnp.array(0.0, dtype=jnp.float32),
                "u_ratio_vocab": jnp.array(0.0, dtype=jnp.float32),
                "u_ratio_1d": jnp.array(0.0, dtype=jnp.float32),
                "replica_drift": jnp.array(0.0, dtype=jnp.float32),
            },
        }

        @partial(jax.jit, static_argnames=("pure_tf", "deliberation_mode"))
        def _train_step_adamw_jit(p, opt_s, q_batch, r_batch, cur_lr, feedback_tokens, sampling_p, pure_tf: bool, deliberation_mode: str, encephalization_alpha: float = 1.0, target_weights: Optional[jnp.ndarray] = None):
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
                    target_weights=target_weights,
                )
            loss_val, grads = jax.value_and_grad(loss_fn)(p)
            grad_norm = optax.global_norm(grads)
            is_grad_finite = jnp.isfinite(grad_norm)
            raw_updates, next_optax_s = optimizer.update(grads, opt_s["optax"], p)
            # Dynamically scale by scheduled cur_lr (negative for gradient descent in optax.apply_updates)
            updates = jax.tree_util.tree_map(lambda u: -cur_lr * u, raw_updates)
            next_p = optax.apply_updates(p, updates)

            params_finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(np_)) for np_ in jax.tree_util.tree_leaves(next_p)]))
            is_step_valid = is_grad_finite & params_finite

            # Strict non-finite skip protection: restore previous parameters and optimizer state on NaN/Inf
            safe_p = jax.tree_util.tree_map(lambda np_, op_: jnp.where(is_step_valid, np_, op_), next_p, p)
            safe_optax_s = jax.tree_util.tree_map(lambda ns_, os_: jnp.where(is_step_valid, ns_, os_), next_optax_s, opt_s["optax"])

            sq_diff = sum(jnp.sum(jnp.square(u)) for u in jax.tree_util.tree_leaves(updates))
            sq_norm = sum(jnp.sum(jnp.square(p_i)) for p_i in jax.tree_util.tree_leaves(p))
            u_ratio = jnp.sqrt(sq_diff / (sq_norm + 1e-8))

            adamw_diag = {
                "grad_norm": grad_norm,
                "was_clipped": jnp.where(grad_norm > 1.0, 1.0, 0.0),
                "was_skipped": jnp.where(is_step_valid, 0.0, 1.0),
                "u_ratio_manifold": jnp.where(is_step_valid, u_ratio, 0.0),
                "u_ratio_vocab": jnp.where(is_step_valid, u_ratio, 0.0),
                "u_ratio_1d": jnp.where(is_step_valid, u_ratio, 0.0),
                "replica_drift": jnp.array(0.0, dtype=jnp.float32),
            }
            next_opt_s = {
                "optax": safe_optax_s,
                "diag": adamw_diag,
            }
            return safe_p, next_opt_s, loss_val

        def train_step_fn(p, opt_s, q_batch, r_batch, cur_lr, feedback_tokens=None, sampling_p=1.0, encephalization_alpha=1.0, target_weights=None):
            pure_tf = (args.deliberation_mode in ("teacher_forcing", "pure_tf")) and (not isinstance(sampling_p, jax.Array)) and (float(sampling_p) >= 1.0)
            return _train_step_adamw_jit(
                p, opt_s, q_batch, r_batch, cur_lr, feedback_tokens, sampling_p, pure_tf, args.deliberation_mode, encephalization_alpha, target_weights
            )

        @partial(jax.jit, static_argnames=("pure_tf", "deliberation_mode"))
        def _compute_grad_adamw_jit(p, q_batch, r_batch, feedback_tokens, sampling_p, pure_tf: bool, deliberation_mode: str, encephalization_alpha: float = 1.0, target_weights: Optional[jnp.ndarray] = None):
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
                    target_weights=target_weights,
                )
            return jax.value_and_grad(loss_fn)(p)

        @jax.jit
        def _apply_opt_adamw_jit(p, opt_s, acc_grads, cur_lr):
            grad_norm = optax.global_norm(acc_grads)
            is_grad_finite = jnp.isfinite(grad_norm)
            raw_updates, next_optax_s = optimizer.update(acc_grads, opt_s["optax"], p)
            updates = jax.tree_util.tree_map(lambda u: -cur_lr * u, raw_updates)
            next_p = optax.apply_updates(p, updates)

            params_finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(np_)) for np_ in jax.tree_util.tree_leaves(next_p)]))
            is_step_valid = is_grad_finite & params_finite

            # Strict non-finite skip protection: exactly mirrors Spheroid optimizer
            safe_p = jax.tree_util.tree_map(lambda np_, op_: jnp.where(is_step_valid, np_, op_), next_p, p)
            safe_optax_s = jax.tree_util.tree_map(lambda ns_, os_: jnp.where(is_step_valid, ns_, os_), next_optax_s, opt_s["optax"])

            sq_diff = sum(jnp.sum(jnp.square(u)) for u in jax.tree_util.tree_leaves(updates))
            sq_norm = sum(jnp.sum(jnp.square(p_i)) for p_i in jax.tree_util.tree_leaves(p))
            u_ratio = jnp.sqrt(sq_diff / (sq_norm + 1e-8))

            adamw_diag = {
                "grad_norm": grad_norm,
                "was_clipped": jnp.where(grad_norm > 1.0, 1.0, 0.0),
                "was_skipped": jnp.where(is_step_valid, 0.0, 1.0),
                "u_ratio_manifold": jnp.where(is_step_valid, u_ratio, 0.0),
                "u_ratio_vocab": jnp.where(is_step_valid, u_ratio, 0.0),
                "u_ratio_1d": jnp.where(is_step_valid, u_ratio, 0.0),
                "replica_drift": jnp.array(0.0, dtype=jnp.float32),
            }
            next_opt_s = {
                "optax": safe_optax_s,
                "diag": adamw_diag,
            }
            return safe_p, next_opt_s

        def compute_grad_step(p, q_batch, r_batch, feedback_tokens=None, sampling_p=1.0, deliberation_mode=args.deliberation_mode, encephalization_alpha=1.0, target_weights=None):
            pure_tf = (deliberation_mode in ("teacher_forcing", "pure_tf")) and (not isinstance(sampling_p, jax.Array)) and (float(sampling_p) >= 1.0)
            return _compute_grad_adamw_jit(p, q_batch, r_batch, feedback_tokens, sampling_p, pure_tf, deliberation_mode, encephalization_alpha, target_weights)

        apply_opt_step = _apply_opt_adamw_jit
        apply_opt_step_anneal = _apply_opt_adamw_jit
        apply_opt_step_frozen = _apply_opt_adamw_jit
        train_step_fn_anneal = train_step_fn
        train_step_fn_frozen = train_step_fn

    # 9. Multi-Stream Ingestion & Prefetched Data Stream
    # Restore persistent cumulative token accounting BEFORE initializing data streams & eval splits
    loaded_ckpt = ckpt if ("ckpt" in locals() and isinstance(ckpt, dict)) else {}
    if getattr(args, "reset_step_counter", False):
        cumulative_tokens = 0
        cumulative_prompt_tokens = 0
        cumulative_loss_tokens = 0
        cumulative_padded_tokens = 0
        initial_lifetime_tokens = int(loaded_ckpt.get("lifetime_tokens", 14_745_600_000))
    else:
        cumulative_tokens = int(loaded_ckpt.get("cumulative_tokens", 0))
        cumulative_prompt_tokens = int(loaded_ckpt.get("cumulative_prompt_tokens", 0))
        cumulative_loss_tokens = int(loaded_ckpt.get("cumulative_loss_tokens", 0))
        cumulative_padded_tokens = int(loaded_ckpt.get("cumulative_padded_tokens", 0))
        initial_lifetime_tokens = int(loaded_ckpt.get("lifetime_tokens", 0))

        # Check website remote token tracker to prevent token regression on worker restart within the same run
        if args.cloud_sync:
            try:
                remote_tracker = fetch_remote_token_tracker(sync_model_name or args.preset, run_id=run_id, url=args.cloud_url, token=args.cloud_token)
                if remote_tracker and isinstance(remote_tracker, dict):
                    remote_cum = int(remote_tracker.get("cumulative_tokens", 0))
                    remote_run = remote_tracker.get("run_id")
                    if (not remote_run or remote_run == run_id) and remote_cum > cumulative_tokens:
                        print(f"🌐 [cloud_sync] Restored persistent cumulative tokens from bioflo.in website ({run_id}): {remote_cum:,} tokens")
                        cumulative_tokens = remote_cum
            except Exception:
                pass

    lifetime_tokens = initial_lifetime_tokens + cumulative_tokens
    resumed_cum_tokens = cumulative_tokens
    current_step_holder = [start_step]
    cumulative_tokens_holder = [cumulative_tokens]
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
        tokens_getter=lambda: cumulative_tokens_holder[0],
        target_tokens=int(getattr(args, "target_tokens", 41_400_000_000) or 41_400_000_000),
        prefetch=not use_multi_tpu_local_sgd,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        retrieval_injection=args.retrieval_injection,
        data_dir=getattr(args, "data_dir", None),
    )
    batcher = elastic_batcher

    def _unpack_batch(batch_tuple):
        if len(batch_tuple) == 6:
            return batch_tuple[2], batch_tuple[3], batch_tuple[4], batch_tuple[5]
        if len(batch_tuple) == 5:
            return batch_tuple[2], batch_tuple[3], batch_tuple[4], None
        if len(batch_tuple) == 4:
            if hasattr(batch_tuple[0], "shape") and hasattr(batch_tuple[1], "shape"):
                return batch_tuple[0], batch_tuple[1], batch_tuple[2], batch_tuple[3]
            return batch_tuple[2], batch_tuple[3], None, None
        if len(batch_tuple) == 3:
            return batch_tuple[0], batch_tuple[1], batch_tuple[2], None
        return batch_tuple[0], batch_tuple[1], None, None

    # 10. XLA Ahead-Of-Time Compilation (non-mutating warmup matching real feedback signature)
    print("\n⚡ Compiling XLA computation graph on TPU...")
    t0_compile = time.time()
    dummy_prompt_len = 1 if is_pretrain_source else args.prompt_len
    dummy_target_len = args.total_window if is_pretrain_source else args.seq_len
    dummy_fb_fill = pad_id if is_pretrain_source else -100
    if use_multi_tpu_local_sgd:
        dummy_prompts = np.zeros((eff_num_devices, eff_local_steps, args.batch_size, dummy_prompt_len), dtype=np.int32)
        dummy_targets = np.zeros((eff_num_devices, eff_local_steps, args.batch_size, dummy_target_len), dtype=np.int32)
        dummy_feedback = (
            np.full((eff_num_devices, eff_local_steps, args.batch_size, dummy_target_len), dummy_fb_fill, dtype=np.int32)
            if (elastic_batcher is not None or is_pretrain_source)
            else None
        )
        dummy_weights = (
            np.ones((eff_num_devices, eff_local_steps, args.batch_size, dummy_target_len), dtype=np.float32)
            if is_pretrain_source
            else None
        )
        _warm_p, _warm_opt, initial_loss = pmap_round_fn(
            rep_params,
            rep_opt_state,
            dummy_prompts,
            dummy_targets,
            args.lr,
            feedback_micro=dummy_feedback,
            target_weights_micro=dummy_weights,
        )
        jax.block_until_ready(initial_loss)
        init_loss_val = float(initial_loss[0])
        del _warm_p, _warm_opt
    else:
        while True:
            dummy_prompts = jnp.zeros((args.batch_size, dummy_prompt_len), dtype=jnp.int32)
            dummy_targets = jnp.zeros((args.batch_size, dummy_target_len), dtype=jnp.int32)
            dummy_feedback = (
                jnp.full((args.batch_size, dummy_target_len), dummy_fb_fill, dtype=jnp.int32)
                if (elastic_batcher is not None or is_pretrain_source)
                else None
            )
            try:
                if compute_grad_step is not None and grad_accum > 1:
                    _warm_loss, _warm_grads = compute_grad_step(
                        params,
                        dummy_prompts,
                        dummy_targets,
                        feedback_tokens=dummy_feedback,
                        sampling_p=args.min_sampling_p,
                    )
                    jax.block_until_ready(_warm_loss)
                    init_loss_val = float(_warm_loss)
                    del _warm_loss, _warm_grads
                else:
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
                break
            except Exception as e_warm:
                if not is_oom_error(e_warm) or args.batch_size <= 1:
                    raise
                args.batch_size, grad_accum = oom_handler.handle_oom(e_warm, step=0)
                oom_handler.update_streamer(elastic_batcher or local_streamer)
                if grad_accum > 1 and compute_grad_step is None:
                    compute_grad_step, apply_opt_step_anneal = build_spheroid_train_engine(
                        config, buffers, pad_id=pad_id, weight_decay=args.weight_decay,
                        repetition_penalty=args.repetition_penalty, remat=remat_choice,
                        deliberation_mode=args.deliberation_mode, param_masks=None,
                        scmer_lambda=eff_scmer_lambda, scmer_subsample=eff_scmer_subsample,
                        scmer_interval=eff_scmer_interval, scmer_margin=eff_scmer_margin,
                        scmer_pair_samples=eff_scmer_pairs,
                    )
                    _, apply_opt_step_frozen = build_spheroid_train_engine(
                        config, buffers, pad_id=pad_id, weight_decay=args.weight_decay,
                        repetition_penalty=args.repetition_penalty, remat=remat_choice,
                        deliberation_mode=args.deliberation_mode, param_masks=param_masks,
                        scmer_lambda=eff_scmer_lambda, scmer_subsample=eff_scmer_subsample,
                        scmer_interval=eff_scmer_interval, scmer_margin=eff_scmer_margin,
                        scmer_pair_samples=eff_scmer_pairs,
                    ) if param_masks is not None else (None, apply_opt_step_anneal)
    if start_step > 0 and math.isfinite(best_eval_loss):
        print(f"✓ XLA Compilation complete in {time.time() - t0_compile:.2f}s! Baseline Restored Loss: {best_eval_loss:.4f} (Tracer init: {init_loss_val:.2f})")
    else:
        print(f"✓ XLA Compilation complete in {time.time() - t0_compile:.2f}s! Initial Loss: {init_loss_val:.4f}")

    # 9. Clean Multi-Domain Held-Out Validation Split
    eval_batches = []
    eval_domains = []
    eval_weights = []
    is_pretrain_source = args.dataset_source in ("pretrain", "rectified", "rectified_pretrain", "clean_pretrain")

    def generate_held_out_split(stage_name: str, target_length: int):
        nonlocal eval_batches, eval_domains, eval_weights
        if not is_pretrain_source:
            return
        try:
            from src.elastic_window import build_pretrain_eval_split
            eval_prompt_len = 1
            is_mid = stage_name in ("midtrain", "midtraining") or getattr(args, "dataset_source", "") in ("midtrain", "midtraining")
            is_reasoning = stage_name in ("deep_reasoning", "reasoning", "stage2", "stage_2")
            if is_reasoning:
                weights_req = {"math": 0.35, "cosmopedia": 0.35, "code": 0.20, "edu_academic": 0.10}
                desc = "Stage 2: Deep Reasoning (Math 35%, Cosmopedia 35%, Code 20%, Edu 10%)"
            elif is_mid:
                weights_req = {"edu_academic": 0.40, "math": 0.30, "cosmopedia": 0.15, "code": 0.15}
                desc = "Mid-Training (Edu 40%, Math 30%, Cosmopedia 15%, Code 15%)"
            else:
                if local_streamer is not None and hasattr(local_streamer, "_refresh_curriculum_shares"):
                    try:
                        local_streamer._refresh_curriculum_shares()
                    except Exception:
                        pass
                if local_streamer is not None and hasattr(local_streamer, "shares") and local_streamer.shares:
                    weights_req = dict(local_streamer.shares)
                else:
                    cur_tok = float(cumulative_tokens_holder[0]) if "cumulative_tokens_holder" in locals() and cumulative_tokens_holder else 0.0
                    if cur_tok < 15_700_000_000:
                        weights_req = {"broad_web": 0.90, "code": 0.10}
                    elif cur_tok < 31_400_000_000:
                        weights_req = {"edu_academic": 0.75, "code": 0.20, "math": 0.05}
                    elif cur_tok < 36_400_000_000:
                        weights_req = {"edu_academic": 0.70, "code": 0.20, "math": 0.10}
                    else:
                        weights_req = {"edu_academic": 0.58, "code": 0.24, "math": 0.14, "cosmopedia": 0.04}
                pct_str = ", ".join(f"{d} {w*100:.0f}%" for d, w in weights_req.items())
                desc = f"Pretrain Token-Based Mixture ({pct_str})"
            domains_req = list(weights_req.keys())
            new_batches, new_domains, new_weights = build_pretrain_eval_split(
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                prompt_len=eval_prompt_len,
                target_len=target_length,
                target_pad_id=pad_id,
                pad_id=pad_id,
                return_weights=True,
                domains=domains_req,
                domain_weights=weights_req,
            )
            if new_batches:
                eval_batches = new_batches
                eval_domains = new_domains
                eval_weights = new_weights
                print(f"✅ Held-Out Validation [{desc}]: Prepared {len(eval_batches)} domains (Target Length: {target_length}):")
                for d_name, d_wt in zip(eval_domains, eval_weights):
                    print(f"   ├─ Domain: {d_name} (Weight: {d_wt*100:.0f}%)")
            else:
                print(f"⚠️ Held-Out Validation ({stage_name}): Online streams returned 0 batches. Keeping current validation set.")
        except Exception as e:
            print(f"⚠️ Pretrain validation setup notice ({stage_name}): {e}. Keeping current validation set.")

    if is_pretrain_source:
        # Pretraining Step 1 only tests FineWeb and The Stack Code
        generate_held_out_split("pretrain", args.total_window)
    else:
        try:
            from src.elastic_window import build_canonical_eval_split
            eval_batches, eval_domains = build_canonical_eval_split(
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                prompt_len=(1 if is_pretrain_source else args.prompt_len),
                target_len=args.seq_len,
                target_pad_id=-100,
                pad_id=pad_id,
            )
            eval_weights = [1.0 / len(eval_domains)] * len(eval_domains)
            print(f"✅ Clean Benchmark Held-Out Validation: Prepared {len(eval_batches)} canonical evaluation domains:")
            for d_name in eval_domains:
                print(f"   ├─ Domain: {d_name}")
        except Exception as e:
            print(f"⚠️ Canonical validation setup notice: {e}. Falling back to active stream.")
            eval_batches = []

    if not eval_batches:
        try:
            if elastic_batcher is not None:
                for _ in range(4):
                    b_tup = (
                        elastic_batcher.get_batch()
                        if hasattr(elastic_batcher, "get_batch")
                        else elastic_batcher.next_batch()
                    )
                    eval_batches.append(_unpack_batch(b_tup))
            elif local_streamer is not None:
                for _ in range(4):
                    vp, vt = local_streamer.get_batch()
                    eval_batches.append((vp, vt, None))
            elif os.path.exists("data/tokenizer"):
                stream_val = LinguisticDataStreamer(
                    batch_size=args.batch_size,
                    prompt_len=(1 if is_pretrain_source else args.prompt_len),
                    target_len=args.seq_len,
                    vocab_size=config.vocab_size,
                    data_dir=getattr(args, "data_dir", None),
                    queue_size=4,
                    target_pad_id=pad_id if is_pretrain_source else -100,
                )
                try:
                    for _ in range(4):
                        vp, vt = stream_val.get_batch()
                        eval_batches.append((vp, vt, None))
                finally:
                    if hasattr(stream_val, "stop"):
                        stream_val.stop()
            eval_domains = [f"Domain_{i}" for i in range(len(eval_batches))]
            eval_weights = [1.0 / len(eval_batches)] * len(eval_batches)
            print(f"✅ Prepared {len(eval_batches)} fallback evaluation batches.")
        except Exception as e_inner:
            print(f"⚠️ Fallback validation setup notice: {e_inner}")

    # 9b. Smart Dynamic Multi-Domain Mixture Controller
    smart_mixture_controller = None
    if getattr(args, "smart_mixture", False) and hasattr(batcher, "update_shares"):
        try:
            from src.smart_mixture import SmartMixtureController
            smart_mixture_controller = SmartMixtureController(
                initial_shares=getattr(batcher, "shares", {}),
                eta=getattr(args, "smart_mixture_eta", 0.08),
                prior_weight=getattr(args, "smart_mixture_prior_weight", 0.20),
                overfit_threshold=getattr(args, "smart_mixture_overfit_thresh", 0.04),
            )
            print(f"⚖️ [Smart Mixture Engine] Initialized dynamic multi-domain mixture controller across {len(smart_mixture_controller.current_shares)} domains.")
        except Exception as e_mix:
            print(f"⚠️ [Smart Mixture Engine] Setup notice: {e_mix}")
            smart_mixture_controller = None

    eval_config = dataclasses.replace(config, use_selective_loss=False, z_loss_weight=0.0) if dataclasses.is_dataclass(config) else config

    @jax.jit
    def eval_step_loss(p, q_b, r_b, fb_b=None, enc_alpha: float = 1.0):
        return training_rollout(
            params=p,
            prompts=q_b,
            target_len=r_b.shape[1],
            buffers=buffers,
            config=eval_config,
            feedback_tokens=fb_b,
            feedback_is_shifted=True,
            pad_id=pad_id,
            targets=r_b,
            remat=False,
            pure_teacher_forcing=True,
            deliberation_mode=args.deliberation_mode,
            repetition_penalty=1.0,
            encephalization_alpha=enc_alpha,
        )

    # 10. Learning Rate & Scheduled Sampling Schedules
    def get_learning_rate(
        step_idx: int,
        total_s: int,
        peak_lr: float,
        min_lr: float,
        warmup_s: int,
        cur_tokens: Optional[float] = None,
        target_toks: Optional[float] = None,
    ) -> float:
        if step_idx <= warmup_s:
            return float(peak_lr * (float(step_idx) / max(1.0, float(warmup_s))))
        if args.lr_schedule == "warm_restart":
            # Relative to start_step on resumption so fresh cycles always start at peak_lr rather than arbitrary modulo offset
            ref_step = start_step if start_step > 0 else warmup_s
            cycle_step = max(0, step_idx - ref_step) % max(1, args.restart_steps)
            ratio = float(cycle_step) / float(args.restart_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
            return float(min_lr + (peak_lr - min_lr) * coeff)
        elif args.lr_schedule == "session":
            # Session-relative cosine decay across current resumption window
            elapsed = max(0, step_idx - start_step)
            session_len = max(1, total_s - start_step)
            ratio = min(1.0, float(elapsed) / float(session_len))
            coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
            return float(min_lr + (peak_lr - min_lr) * coeff)
        else:  # cosine (token-progress when token budget is active, step-progress fallback otherwise)
            if cur_tokens is not None and target_toks is not None and float(target_toks) > 0.0:
                ratio = min(1.0, max(0.0, float(cur_tokens) / float(target_toks)))
            else:
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
        if hasattr(local_streamer, "get_batch"):
            try:
                prompt_b, target_b, weights_b = local_streamer.get_batch(return_weights=True)
                return prompt_b, target_b, None, weights_b
            except Exception:
                prompt_b, target_b = local_streamer.get_batch()
                return prompt_b, target_b, None, None
        prompt_b, target_b = local_streamer.get_batch()
        return prompt_b, target_b, None, None

    _round_bucket_idx = 0

    def _build_multi_tpu_round():
        nonlocal _round_bucket_idx
        total_shards = eff_num_devices * eff_local_steps
        bucket_batcher = elastic_batcher
        if bucket_batcher is not None and hasattr(bucket_batcher, "batcher"):
            bucket_batcher = bucket_batcher.batcher
        bucket_idx = None
        if bucket_batcher is not None and hasattr(bucket_batcher, "buckets"):
            bucket_idx = _round_bucket_idx % len(bucket_batcher.buckets)
            _round_bucket_idx += 1
        qs, rs, fbs, wts = [], [], [], []
        for _ in range(total_shards):
            if bucket_idx is not None:
                batch_tuple = bucket_batcher.next_batch(bucket_idx=bucket_idx)
                b_unpack = _unpack_batch(batch_tuple)
            else:
                b_unpack = _next_batch()
            pb, tb = b_unpack[0], b_unpack[1]
            fb = b_unpack[2] if len(b_unpack) > 2 else None
            wb = b_unpack[3] if len(b_unpack) > 3 else None
            qs.append(np.asarray(pb, dtype=np.int32))
            rs.append(np.asarray(tb, dtype=np.int32))
            if fb is not None:
                fbs.append(np.asarray(fb, dtype=np.int32))
            if wb is not None:
                wts.append(np.asarray(wb, dtype=np.float32))
        b_p, l_p = qs[0].shape
        b_t, l_t = rs[0].shape
        q_arr = np.stack(qs, axis=0).reshape(eff_num_devices, eff_local_steps, b_p, l_p)
        r_arr = np.stack(rs, axis=0).reshape(eff_num_devices, eff_local_steps, b_t, l_t)
        fb_arr = (
            np.stack(fbs, axis=0).reshape(eff_num_devices, eff_local_steps, b_t, l_t)
            if len(fbs) == total_shards
            else None
        )
        if len(wts) == total_shards:
            w_stacked = np.stack(wts, axis=0)
            if wts[0].ndim == 2:
                wt_arr = w_stacked.reshape(eff_num_devices, eff_local_steps, b_t, l_t)
            else:
                wt_arr = w_stacked.reshape(eff_num_devices, eff_local_steps, b_t)
        else:
            wt_arr = None
        return q_arr, r_arr, fb_arr, wt_arr

    _multi_tpu_queue = None
    _multi_tpu_stop = None
    _multi_tpu_thread = None
    if use_multi_tpu_local_sgd:
        import queue as _py_queue
        import threading as _py_threading
        _multi_tpu_queue = _py_queue.Queue(maxsize=3)
        _multi_tpu_stop = _py_threading.Event()

        def _multi_tpu_prefetch_worker():
            while not _multi_tpu_stop.is_set():
                try:
                    q_a, r_a, fb_a, wt_a = _build_multi_tpu_round()
                    if len(devices) == eff_num_devices:
                        q_a = jax.device_put_sharded(list(q_a), devices)
                        r_a = jax.device_put_sharded(list(r_a), devices)
                        if fb_a is not None:
                            fb_a = jax.device_put_sharded(list(fb_a), devices)
                        if wt_a is not None:
                            wt_a = jax.device_put_sharded(list(wt_a), devices)
                    item = (q_a, r_a, fb_a, wt_a)
                    while not _multi_tpu_stop.is_set():
                        try:
                            _multi_tpu_queue.put(item, timeout=0.2)
                            break
                        except _py_queue.Full:
                            continue
                except Exception as exc:
                    if not _multi_tpu_stop.is_set():
                        _multi_tpu_queue.put(exc)
                    break

        _multi_tpu_thread = _py_threading.Thread(
            target=_multi_tpu_prefetch_worker, name="MultiTpuRoundPrefetcher", daemon=True
        )
        _multi_tpu_thread.start()

    def _next_multi_tpu_round():
        if _multi_tpu_queue is not None:
            item = _multi_tpu_queue.get()
            if isinstance(item, Exception):
                raise item
            return item
        return _build_multi_tpu_round()

    def _get_host_params():
        if use_multi_tpu_local_sgd:
            return jax.tree_util.tree_map(lambda x: jax.device_get(x[0]), rep_params)
        return params

    def _get_host_opt_state():
        if use_multi_tpu_local_sgd and rep_opt_state is not None:
            return jax.tree_util.tree_map(lambda x: jax.device_get(x[0]), rep_opt_state)
        return opt_state

    def _count_real_tokens(prompts: Any, targets: Any) -> Tuple[int, int, int, int]:
        """Count non-padding input and loss-bearing target tokens separately.

        Static TPU shapes are an implementation detail, not training-data
        volume. Reporting valid prompt and loss-bearing tokens separately ensures
        honest token accounting.
        """
        p = np.asarray(prompts)
        t = np.asarray(targets)
        valid_prompt = int(np.count_nonzero(p != pad_id))
        valid_loss = int(np.count_nonzero(t >= 0))
        total_slots = int(p.size + t.size)
        real_tokens = valid_prompt + valid_loss
        padded_slots = total_slots - real_tokens
        return real_tokens, valid_prompt, valid_loss, padded_slots

    step = start_step
    t_start = time.time()
    t_last_log = time.time()
    last_cloud_save_time = time.time()
    step_window_tokens = 0
    step_window_count = 0
    ema_loss: Optional[float] = initial_ema_loss
    last_synced_loss_val: float = float(initial_ema_loss) if initial_ema_loss is not None else 3.0
    steps_since_shape_change: int = 0
    consistent_target_count = 0

    # Diagnostic Tracking for Optimizer Health & Stability
    cumulative_skips = 0
    window_clipped_count = 0
    window_step_count = 0

    # Restore per-dataset token attribution if present
    if not getattr(args, "reset_step_counter", False):
        loaded_ds_tokens = loaded_ckpt.get("tokens_by_dataset", {})
        if loaded_ds_tokens and hasattr(batcher, "set_tokens_by_dataset"):
            batcher.set_tokens_by_dataset(loaded_ds_tokens)
            print(f"📦 [Checkpoint Restore] Restored per-dataset token stats: {loaded_ds_tokens}")

    last_successful_step = step
    training_error: Optional[Exception] = None
    target_token_budget = getattr(args, "target_tokens", None)
    in_deep_reasoning_phase = False
    reasoning_transition_step = None
    auto_reasoning_trans = getattr(args, "auto_reasoning_transition", True)
    reasoning_trans_tokens = getattr(args, "reasoning_transition_tokens", 41_400_000_000)
    final_campaign_budget = getattr(args, "final_campaign_tokens", 50_000_000_000)

    # Recalibrate total_steps schedule on resumption if remote/checkpoint token counter was updated
    if not getattr(args, "total_steps_explicit", False) and target_token_budget is not None and target_token_budget > 0:
        rem_budget = max(0.0, float(target_token_budget) - float(cumulative_tokens))
        if rem_budget <= 0.0:
            print(f"🎯 [Target Token Budget Satisfied] Cumulative tokens ({cumulative_tokens:,}) >= budget ({int(target_token_budget):,}).")
        else:
            recal_steps = calculate_curriculum_aware_target_steps(
                target_budget=rem_budget,
                eff_batch_size=eff_batch_size,
                base_prompt_len=real_prompt_slots,
                base_seq_len=real_target_slots,
                context_curriculum=getattr(args, "context_curriculum", False),
                context_stages=context_stages if "context_stages" in locals() else None,
                context_stage_steps=getattr(args, "context_stage_steps", 100_000),
                start_step=step,
            )
            args.total_steps = step + recal_steps
            print(
                f"🎯 [Schedule Alignment] Adjusted campaign schedule to Step {args.total_steps:,} "
                f"(+{recal_steps:,} remaining steps) to ingest {rem_budget/1e9:.2f}B remaining tokens."
            )

    # Check if resumed checkpoint already passed the 15.7B milestone
    if auto_reasoning_trans and cumulative_tokens >= reasoning_trans_tokens:
        in_deep_reasoning_phase = True
        reasoning_transition_step = step
        print("\n" + "=" * 80)
        print(f"🧠 [Deep Reasoning Active on Resumption] Cumulative tokens ({cumulative_tokens:,}) >= {reasoning_trans_tokens/1e9:.2f}B.")
        print(f"   Activating High-Density Reasoning Mixture (35% Math, 35% Cosmopedia v2, 20% Code, 10% Edu Academic).")
        print(f"   Applying Conservative Stage 2 Learning Rate ({args.reasoning_max_lr:g} -> {args.reasoning_min_lr:g}) to protect Stage 1 foundations.")
        print("=" * 80 + "\n", flush=True)
        if target_token_budget is not None and target_token_budget <= reasoning_trans_tokens:
            target_token_budget = final_campaign_budget
            args.target_tokens = final_campaign_budget
            if not getattr(args, "total_steps_explicit", False):
                args.total_steps = calibrate_curriculum_total_steps(
                    step=step,
                    start_step=start_step,
                    cumulative_tokens=cumulative_tokens,
                    resumed_cum_tokens=resumed_cum_tokens if "resumed_cum_tokens" in locals() else cumulative_tokens,
                    target_token_budget=target_token_budget,
                    eff_batch_size=eff_batch_size,
                    base_prompt_len=real_prompt_slots,
                    base_seq_len=real_target_slots,
                    context_curriculum=getattr(args, "context_curriculum", False),
                    context_stages=context_stages if "context_stages" in locals() else None,
                    context_stage_steps=getattr(args, "context_stage_steps", 100_000),
                )
        if local_streamer is not None and hasattr(local_streamer, "transition_to_deep_reasoning"):
            local_streamer.transition_to_deep_reasoning()
        elif batcher is not None and hasattr(batcher, "transition_to_deep_reasoning"):
            batcher.transition_to_deep_reasoning()
        generate_held_out_split("deep_reasoning", args.total_window if is_pretrain_source else args.seq_len)
        if smart_mixture_controller is not None:
            reasoning_priors = {"math": 0.35, "cosmopedia": 0.35, "code": 0.20, "edu_academic": 0.10}
            smart_mixture_controller.prior_shares = dict(reasoning_priors)
            smart_mixture_controller.current_shares = dict(reasoning_priors)

    budget_active = (target_token_budget is not None and not getattr(args, "total_steps_explicit", False))
    try:
        while not shutdown_requested:
            if budget_active:
                if cumulative_tokens >= target_token_budget:
                    print(
                        f"\n🎯 [Budget Target Reached] Cumulative tokens ({cumulative_tokens:,}) "
                        f"reached configured target budget ({target_token_budget:,}). Cleanly stopping training."
                    )
                    break
            else:
                if step >= args.total_steps:
                    break
            # ──────────────────────────────────────────────────────────────────
            # Automatic Deep Reasoning Transition Check (Milestone: 15.7B Tokens)
            # ──────────────────────────────────────────────────────────────────
            if auto_reasoning_trans and not in_deep_reasoning_phase and cumulative_tokens >= reasoning_trans_tokens:
                in_deep_reasoning_phase = True
                reasoning_transition_step = step
                best_eval_loss = float("inf")
                print("\n" + "=" * 80)
                print("🧠 [AUTOMATIC CURRICULUM TRANSITION: PRETRAIN -> DEEP REASONING]")
                print("=" * 80)
                print(f"🎯 Milestone Reached   : {cumulative_tokens:,} tokens ingested (>= {reasoning_trans_tokens/1e9:.2f}B token threshold)")
                print(f"🌊 Phase Transition    : PRETRAIN (Broad Web) -> DEEP REASONING (High-Density Formal Reasoning)")
                print(f"📚 Data Mixture Shift   : 35% Math (FineMath-4+) | 35% Cosmopedia v2 | 20% Python Code | 10% Edu Academic")
                print(f"⚡ Context Horizon      : Full 4096 tokens contiguous reasoning trajectory")
                print(f"🛡️ Learning Rate Guard  : Conservative Stage 2 Schedule ({args.reasoning_max_lr:g} -> {args.reasoning_min_lr:g}) active to prevent catastrophic forgetting.")
                print("=" * 80 + "\n", flush=True)

                # Extend target token budget so training continues smoothly
                if target_token_budget is not None and target_token_budget <= reasoning_trans_tokens:
                    target_token_budget = final_campaign_budget
                    args.target_tokens = final_campaign_budget
                    if not getattr(args, "total_steps_explicit", False):
                        args.total_steps = calibrate_curriculum_total_steps(
                            step=step,
                            start_step=start_step,
                            cumulative_tokens=cumulative_tokens,
                            resumed_cum_tokens=resumed_cum_tokens if "resumed_cum_tokens" in locals() else cumulative_tokens,
                            target_token_budget=target_token_budget,
                            eff_batch_size=eff_batch_size,
                            base_prompt_len=real_prompt_slots,
                            base_seq_len=real_target_slots,
                            context_curriculum=getattr(args, "context_curriculum", False),
                            context_stages=context_stages if "context_stages" in locals() else None,
                            context_stage_steps=getattr(args, "context_stage_steps", 100_000),
                        )
                    print(f"📈 [Extended Target Budget] Target token budget extended to {final_campaign_budget/1e9:.2f}B tokens for Deep Reasoning.")

                # Switch active streamer to deep reasoning mixture
                if local_streamer is not None and hasattr(local_streamer, "transition_to_deep_reasoning"):
                    local_streamer.transition_to_deep_reasoning()
                elif batcher is not None and hasattr(batcher, "transition_to_deep_reasoning"):
                    batcher.transition_to_deep_reasoning()

                # Rebuild held-out evaluation split for Stage 2 (Deep Reasoning)
                generate_held_out_split("deep_reasoning", args.total_window if is_pretrain_source else args.seq_len)
                if smart_mixture_controller is not None:
                    reasoning_priors = {"math": 0.35, "cosmopedia": 0.35, "code": 0.20, "edu_academic": 0.10}
                    smart_mixture_controller.prior_shares = dict(reasoning_priors)
                    smart_mixture_controller.current_shares = dict(reasoning_priors)

                # Archive milestone checkpoint locally and to bioflo.in cloud
                try:
                    milestone_ckpt_name = f"{config_preset}_pretrain_15b_complete.pt"
                    milestone_dest = os.path.join(args.checkpoint_dir, milestone_ckpt_name)
                    curr_eval = last_eval_loss if last_eval_loss != float("inf") else (initial_ema_loss or 2.70)
                    save_jax_to_torch_pt(
                        params,
                        buffers,
                        milestone_dest,
                        meta={
                            "step": step,
                            "eval_loss": curr_eval,
                            "cumulative_tokens": cumulative_tokens,
                            "phase": "pretrain_15b_complete",
                            "preset": config_preset,
                            "config": dataclasses.asdict(config) if dataclasses.is_dataclass(config) else None,
                        },
                        opt_state=opt_state if args.save_optimizer_state else None,
                        bare_bf16=not args.save_optimizer_state,
                    )
                    print(f"💾 [Milestone Checkpoint] Archived 15.7B pretraining completion snapshot: {milestone_dest}")
                    if args.cloud_sync:
                        push_checkpoint_async(
                            milestone_dest,
                            step,
                            curr_eval,
                            target_type="milestone",
                            model=config_preset,
                            url=args.cloud_url,
                            token=args.cloud_token,
                        )
                except Exception as e_ms:
                    print(f"ℹ️ [Milestone Archive Notice] {e_ms}")

            if target_token_budget is not None and cumulative_tokens >= target_token_budget:
                print(
                    f"\n🎯 [Budget Target Reached] Cumulative tokens ({cumulative_tokens:,}) "
                    f"reached configured target budget ({target_token_budget:,}). Cleanly stopping training."
                )
                break

            prev_step = step
            step += step_stride
            current_step_holder[0] = step
            is_champion = False
            save_latest = False

            # Context Stage Curriculum Transition Check
            if getattr(args, "context_curriculum", False):
                cur_stage_idx = min(len(context_stages) - 1, step // context_stage_steps)
                if cur_stage_idx != active_stage_idx:
                    active_stage_idx = cur_stage_idx
                    active_window = context_stages[active_stage_idx]
                    args.total_window = active_window
                    if is_pretrain_source:
                        args.seq_len = active_window
                    new_buckets = get_default_elastic_buckets(active_window)
                    print(
                        f"\n🔄 [Window Expansion Transition (Step-Based)] Step {step:,}: Advancing to Window Stage {active_stage_idx} ({active_window} tokens)!\n"
                        f"   Active Buckets: {new_buckets}\n"
                    )
                    if args.batch_size * active_window > 65536:
                        new_bs = max(1, 65536 // active_window)
                        if new_bs >= 16:
                            new_bs = (new_bs // 16) * 16
                        elif new_bs >= 8:
                            new_bs = (new_bs // 8) * 8
                        if new_bs < args.batch_size:
                            print(f"🛡️ [Phase HBM Guard] Scaling micro batch_size {args.batch_size} -> {new_bs} for {active_window}-token phase.")
                            args.batch_size = new_bs
                            if local_streamer is not None and hasattr(local_streamer, "batch_size"):
                                local_streamer.batch_size = new_bs
                            if elastic_batcher is not None:
                                if hasattr(elastic_batcher, "batch_size"):
                                    elastic_batcher.batch_size = new_bs
                                if hasattr(elastic_batcher, "batcher") and hasattr(elastic_batcher.batcher, "batch_size"):
                                    elastic_batcher.batcher.batch_size = new_bs
                    if "_multi_tpu_queue" in locals() and _multi_tpu_queue is not None:
                        while not _multi_tpu_queue.empty():
                            try:
                                _multi_tpu_queue.get_nowait()
                            except Exception:
                                break
                    if local_streamer is not None and hasattr(local_streamer, "update_block_size"):
                        local_streamer.update_block_size(active_window)
                    b_target = elastic_batcher
                    if b_target is not None:
                        if hasattr(b_target, "update_buckets"):
                            b_target.update_buckets(new_buckets, new_total_window=active_window)
                        elif hasattr(b_target, "batcher") and hasattr(b_target.batcher, "update_buckets"):
                            b_target.batcher.update_buckets(new_buckets, new_total_window=active_window)
                    generate_held_out_split("deep_reasoning" if in_deep_reasoning_phase else "pretrain", active_window)
                    steps_since_shape_change = 0

            # Dynamic Sequence Length Transition Check (if seq_transition_steps > 0)
            if target_seq_len is not None and target_seq_len > initial_seq_len and seq_trans_steps > 0:
                elapsed_trans = max(0, step - start_step)
                trans_frac = min(1.0, float(elapsed_trans) / float(seq_trans_steps))
                cur_target_len = initial_seq_len + int((target_seq_len - initial_seq_len) * trans_frac)
                cur_target_len = min(target_seq_len, max(initial_seq_len, (cur_target_len // 64) * 64))
                if cur_target_len != args.seq_len:
                    old_len = args.seq_len
                    args.seq_len = cur_target_len
                    if is_pretrain_source:
                        args.total_window = cur_target_len
                        if local_streamer is not None and hasattr(local_streamer, "update_block_size"):
                            local_streamer.update_block_size(cur_target_len)
                    print(f"\n🔄 [Sequence Transition] Step {step:,}: Progressing target sequence length {old_len} -> {args.seq_len} (Final Target: {target_seq_len})")
                    if elastic_batcher is not None and hasattr(elastic_batcher, "update_target_len"):
                        elastic_batcher.update_target_len(args.seq_len)
                    generate_held_out_split(
                        "deep_reasoning" if in_deep_reasoning_phase else "pretrain",
                        args.total_window if is_pretrain_source else args.seq_len,
                    )

            if in_deep_reasoning_phase and reasoning_transition_step is not None:
                # Conservative Stage 2 Learning Rate: guards against catastrophic forgetting of Stage 1
                r_peak = getattr(args, "reasoning_max_lr", 2e-4)
                r_min = getattr(args, "reasoning_min_lr", 3e-5)
                if budget_active and target_token_budget is not None and target_token_budget > reasoning_trans_tokens:
                    r_ratio = min(1.0, max(0.0, float(cumulative_tokens - reasoning_trans_tokens) / float(max(1.0, target_token_budget - reasoning_trans_tokens))))
                else:
                    r_total_steps = max(1, args.total_steps - reasoning_transition_step)
                    r_elapsed = max(0, step - reasoning_transition_step)
                    r_ratio = min(1.0, float(r_elapsed) / float(r_total_steps))
                r_coeff = 0.5 * (1.0 + math.cos(math.pi * r_ratio))
                cur_lr = float(r_min + (r_peak - r_min) * r_coeff)
            else:
                cur_lr = get_learning_rate(
                    step, args.total_steps, args.lr, args.min_lr, args.warmup_steps,
                    cur_tokens=cumulative_tokens if budget_active else None,
                    target_toks=target_token_budget if budget_active else None,
                )

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

            is_past_freeze_limit = True
            is_frozen_active = False
            is_anneal = False

            crossed_log_step = (prev_step == start_step) or ((step // max(1, args.log_every)) > (prev_step // max(1, args.log_every)))
            crossed_eval_step = bool(eval_batches) and ((step // max(1, args.eval_interval)) > (prev_step // max(1, args.eval_interval)))
            crossed_save_step = ((step // max(1, args.save_interval)) > (prev_step // max(1, args.save_interval))) or (
                args.cloud_save_hours is not None and (time.time() - last_cloud_save_time) >= (args.cloud_save_hours * 3600.0)
            )
            crossed_sync_step = (step % max(1, getattr(args, "cloud_sync_interval", 100)) == 0)
            crossed_periodic_step = (step // max(1, args.periodic_every)) > (prev_step // max(1, args.periodic_every))
            need_host_sync = (
                steps_since_shape_change < 3
                or crossed_log_step
                or crossed_eval_step
                or crossed_save_step
                or crossed_sync_step
                or crossed_periodic_step
            )

            step_executed = False
            while not step_executed:
                prompt_b, target_b, feedback_b, weights_b = None, None, None, None
                try:
                    if use_multi_tpu_local_sgd:
                        q_shard, r_shard, fb_shard, wt_shard = _next_multi_tpu_round()
                        rep_params, rep_opt_state, loss_rep = pmap_round_fn(
                            rep_params,
                            rep_opt_state,
                            q_shard,
                            r_shard,
                            cur_lr,
                            feedback_micro=fb_shard,
                            sampling_p=cur_sampling_p,
                            target_weights_micro=wt_shard,
                        )
                        loss = loss_rep[0]
                        step_tokens, step_prompt, step_loss, step_pad = _count_real_tokens(q_shard, r_shard)
                        step_executed = True
                    else:
                        consumed_micro_batches: List[Tuple[Any, ...]] = []
                        if compute_grad_step is not None and grad_accum > 1:
                            acc_grads = None
                            loss_acc = 0.0
                            step_tokens, step_prompt, step_loss, step_pad = 0, 0, 0, 0
                            inv_accum = 1.0 / float(grad_accum)
                            for _ in range(grad_accum):
                                batch_data = oom_handler.slicer.get_micro_batch(args.batch_size, _next_batch)
                                consumed_micro_batches.append(batch_data)
                                if len(batch_data) == 4:
                                    prompt_b, target_b, feedback_b, weights_b = batch_data
                                else:
                                    prompt_b, target_b, feedback_b = batch_data
                                    weights_b = None
                                g_loss, micro_grads = compute_grad_step(
                                    params, prompt_b, target_b, feedback_tokens=feedback_b, sampling_p=cur_sampling_p, deliberation_mode=args.deliberation_mode, encephalization_alpha=enc_alpha, target_weights=weights_b
                                )
                                acc_grads = (
                                    micro_grads
                                    if acc_grads is None
                                    else jax.tree_util.tree_map(lambda a, b: a + b, acc_grads, micro_grads)
                                )
                                loss_acc = loss_acc + g_loss
                                m_tok, m_p, m_l, m_pad = _count_real_tokens(prompt_b, target_b)
                                step_tokens += m_tok
                                step_prompt += m_p
                                step_loss += m_l
                                step_pad += m_pad
                            acc_grads = jax.tree_util.tree_map(lambda a: a * inv_accum, acc_grads)
                            loss = loss_acc * inv_accum
                            cur_apply_opt = apply_opt_step_anneal if (not is_frozen_active or is_anneal or param_masks is None) else (apply_opt_step_frozen or apply_opt_step_anneal)
                            params, opt_state = cur_apply_opt(params, opt_state, acc_grads, cur_lr)
                        else:
                            batch_data = oom_handler.slicer.get_micro_batch(args.batch_size, _next_batch)
                            consumed_micro_batches.append(batch_data)
                            if len(batch_data) == 4:
                                prompt_b, target_b, feedback_b, weights_b = batch_data
                            else:
                                prompt_b, target_b, feedback_b = batch_data
                                weights_b = None
                            cur_train_step = train_step_fn_anneal if (not is_frozen_active or is_anneal or param_masks is None) else (train_step_fn_frozen or train_step_fn_anneal)
                            params, opt_state, loss = cur_train_step(
                                params,
                                opt_state,
                                prompt_b,
                                target_b,
                                cur_lr,
                                feedback_tokens=feedback_b,
                                sampling_p=cur_sampling_p,
                                encephalization_alpha=enc_alpha,
                                target_weights=weights_b,
                            )
                            step_tokens, step_prompt, step_loss, step_pad = _count_real_tokens(prompt_b, target_b)
                        # Synchronize device execution on initial shape steps (for OOM catching) and log/eval/save steps;
                        # otherwise allow XLA async dispatch to overlap TPU execution with host batch prep.
                        if need_host_sync:
                            loss_val = float(loss)
                            last_synced_loss_val = loss_val
                        else:
                            loss_val = last_synced_loss_val
                        step_executed = True
                except Exception as e_step:
                    if not is_oom_error(e_step) or (args.batch_size <= 1 and remat_choice):
                        raise
                    failed_batches_to_stash = consumed_micro_batches if consumed_micro_batches else (
                        [(prompt_b, target_b, feedback_b, weights_b)] if weights_b is not None else ([(prompt_b, target_b, feedback_b)] if prompt_b is not None else None)
                    )
                    args.batch_size, grad_accum = oom_handler.handle_oom(
                        e_step, step=step, failed_batch=failed_batches_to_stash, current_remat=remat_choice
                    )
                    remat_changed = (oom_handler.remat != remat_choice)
                    remat_choice = oom_handler.remat
                    oom_handler.update_streamer(elastic_batcher or local_streamer)
                    if use_multi_tpu_local_sgd:
                        eff_local_steps = max(1, int(getattr(args, "local_steps", 4)) * grad_accum)
                        step_stride = eff_num_devices * eff_local_steps
                        pmap_round_fn = build_pmap_local_sgd_step_spheroid(
                            config,
                            buffers,
                            pad_id=pad_id,
                            weight_decay=args.weight_decay,
                            repetition_penalty=args.repetition_penalty,
                            remat=remat_choice,
                            local_steps=eff_local_steps,
                            devices=devices,
                            scmer_lambda=eff_scmer_lambda,
                            scmer_subsample=eff_scmer_subsample,
                            scmer_interval=eff_scmer_interval,
                            scmer_margin=eff_scmer_margin,
                            scmer_pair_samples=eff_scmer_pairs,
                        )
                        host_p = _get_host_params()
                        host_opt = _get_host_opt_state()
                        rep_params = replicate_for_pmap(host_p, devices)
                        if host_opt is not None:
                            rep_opt_state = replicate_for_pmap(host_opt, devices)
                    elif compute_grad_step is None or remat_changed:
                        compute_grad_step, apply_opt_step_anneal = build_spheroid_train_engine(
                            config, buffers, pad_id=pad_id, weight_decay=args.weight_decay,
                            repetition_penalty=args.repetition_penalty, remat=remat_choice,
                            deliberation_mode=args.deliberation_mode, param_masks=None,
                            scmer_lambda=eff_scmer_lambda, scmer_subsample=eff_scmer_subsample,
                            scmer_interval=eff_scmer_interval, scmer_margin=eff_scmer_margin,
                            scmer_pair_samples=eff_scmer_pairs,
                        )
                        _, apply_opt_step_frozen = build_spheroid_train_engine(
                            config, buffers, pad_id=pad_id, weight_decay=args.weight_decay,
                            repetition_penalty=args.repetition_penalty, remat=remat_choice,
                            deliberation_mode=args.deliberation_mode, param_masks=param_masks,
                            scmer_lambda=eff_scmer_lambda, scmer_subsample=eff_scmer_subsample,
                            scmer_interval=eff_scmer_interval, scmer_margin=eff_scmer_margin,
                            scmer_pair_samples=eff_scmer_pairs,
                        ) if param_masks is not None else (None, apply_opt_step_anneal)
                        train_step_fn_anneal = build_train_step_spheroid(
                            config, buffers, pad_id=pad_id, weight_decay=args.weight_decay,
                            repetition_penalty=args.repetition_penalty, remat=remat_choice,
                            deliberation_mode=args.deliberation_mode, param_masks=None,
                            scmer_lambda=eff_scmer_lambda, scmer_subsample=eff_scmer_subsample,
                            scmer_interval=eff_scmer_interval, scmer_margin=eff_scmer_margin,
                            scmer_pair_samples=eff_scmer_pairs,
                        )
                        train_step_fn_frozen = build_train_step_spheroid(
                            config, buffers, pad_id=pad_id, weight_decay=args.weight_decay,
                            repetition_penalty=args.repetition_penalty, remat=remat_choice,
                            deliberation_mode=args.deliberation_mode, param_masks=param_masks,
                            scmer_lambda=eff_scmer_lambda, scmer_subsample=eff_scmer_subsample,
                            scmer_interval=eff_scmer_interval, scmer_margin=eff_scmer_margin,
                            scmer_pair_samples=eff_scmer_pairs,
                        ) if param_masks is not None else train_step_fn_anneal

            steps_since_shape_change += 1
            if need_host_sync:
                loss_val = float(loss)
                last_synced_loss_val = loss_val
                if math.isfinite(loss_val):
                    # Smooth 0.96 / 0.04 EMA update to prevent single-batch spikes from oscillating the governor
                    ema_loss = loss_val if ema_loss is None else (0.96 * ema_loss + 0.04 * loss_val)
                else:
                    print(f"\n⚠️ [Divergence Watchdog] Non-finite loss ({loss_val}) detected at Step {step:,}! Stabilizing optimizer momentum...")
                    ema_loss = last_eval_loss if math.isfinite(last_eval_loss) else 3.00
                    cur_lr = cur_lr * 0.75
            else:
                loss_val = last_synced_loss_val

            # Extract optimizer diagnostics only on host-sync steps in one batched device_get
            cur_diag = None
            if need_host_sync:
                if use_multi_tpu_local_sgd and rep_opt_state is not None:
                    raw_diag = rep_opt_state.get("diag")
                    if raw_diag is not None:
                        host_diag = jax.device_get(raw_diag)
                        cur_diag = {k: float(v[0]) for k, v in host_diag.items()}
                elif opt_state is not None and isinstance(opt_state, dict) and "diag" in opt_state:
                    raw_diag = opt_state["diag"]
                    host_diag = jax.device_get(raw_diag)
                    cur_diag = {k: float(v) for k, v in host_diag.items()}

            if cur_diag is not None:
                was_skip = cur_diag.get("was_skipped", 0.0)
                if was_skip > 0.0:
                    cumulative_skips += 1
                    print(
                        f"\n⚠️ [Optimizer Skip @ Step {step:,}] Step was SKIPPED by optimizer! "
                        f"(Non-finite gradient or parameters detected. Total skips: {cumulative_skips})"
                    )
                was_clip = cur_diag.get("was_clipped", 0.0)
                if was_clip > 0.0:
                    window_clipped_count += 1
            window_step_count += step_stride

            # Step & token throughput accounting
            step_window_tokens += step_tokens
            step_window_count += step_stride
            cumulative_tokens += step_tokens
            cumulative_tokens_holder[0] = cumulative_tokens
            cumulative_prompt_tokens += step_prompt
            cumulative_loss_tokens += step_loss
            cumulative_padded_tokens += step_pad
            lifetime_tokens = initial_lifetime_tokens + cumulative_tokens

            # Console Progress Logging
            crossed_log = (prev_step == start_step) or ((step // max(1, args.log_every)) > (prev_step // max(1, args.log_every)))
            if crossed_log:
                elapsed_log = time.time() - t_last_log
                steps_per_sec = step_window_count / max(elapsed_log, 1e-6)
                tok_per_sec = step_window_tokens / max(elapsed_log, 1e-6)

                # Dynamic Step Schedule Calibration: project remaining tokens across curriculum stages
                # using empirical token density rather than extrapolating Stage-0 tokens/step linearly.
                if budget_active and not getattr(args, "total_steps_explicit", False) and step > start_step:
                    args.total_steps = calibrate_curriculum_total_steps(
                        step=step,
                        start_step=start_step,
                        cumulative_tokens=cumulative_tokens,
                        resumed_cum_tokens=resumed_cum_tokens,
                        target_token_budget=target_token_budget,
                        eff_batch_size=eff_batch_size,
                        base_prompt_len=real_prompt_slots,
                        base_seq_len=real_target_slots,
                        context_curriculum=getattr(args, "context_curriculum", False),
                        context_stages=context_stages if "context_stages" in locals() else None,
                        context_stage_steps=getattr(args, "context_stage_steps", 100_000),
                    )

                gov_str = f" | Gov: {alpha_gov:.2f}" if getattr(args, "loss_window_governor", True) and alpha_gov > 0.01 else ""

                pad_ratio = (cumulative_padded_tokens / max(cumulative_tokens + cumulative_padded_tokens, 1)) * 100.0
                tgt_tok = getattr(args, "target_tokens", 200_000_000_000)
                if tgt_tok and tgt_tok > 0:
                    pct = (cumulative_tokens / tgt_tok) * 100.0
                    eta_sec = max(0.0, tgt_tok - cumulative_tokens) / max(tok_per_sec, 1e-6)
                    eta_h = eta_sec / 3600.0
                    tok_str = (
                        f"RealTok: {cumulative_tokens/1e9:5.3f}B/{tgt_tok/1e9:4.2f}B ({pct:5.2f}%) "
                        f"[Loss: {cumulative_loss_tokens/1e9:5.3f}B, Prompt: {cumulative_prompt_tokens/1e9:5.3f}B | Pad: {pad_ratio:4.1f}%] | "
                        f"ETA: {eta_h:4.1f}h | "
                    )
                else:
                    tok_str = (
                        f"RealTok: {cumulative_tokens/1e9:5.3f}B "
                        f"[Loss: {cumulative_loss_tokens/1e9:5.3f}B, Prompt: {cumulative_prompt_tokens/1e9:5.3f}B | Pad: {pad_ratio:4.1f}%] | "
                    )

                phase_tag = " [ALL_TRAINABLE]"

                clip_pct = (window_clipped_count / max(1, window_step_count)) * 100.0
                diag_info = ""
                if cur_diag is not None:
                    gn = cur_diag.get("grad_norm", 0.0)
                    u_m = cur_diag.get("u_ratio_manifold", 0.0)
                    u_v = cur_diag.get("u_ratio_vocab", 0.0)
                    u_1d = cur_diag.get("u_ratio_1d", 0.0)
                    drift = cur_diag.get("replica_drift", 0.0)
                    drift_str = f" | Drift: {drift:.2e}" if use_multi_tpu_local_sgd else ""
                    diag_info = (
                        f"\n       ↳ [Optimizer Diag] GradNorm: {gn:5.2f} (Clip: {clip_pct:4.1f}%) | "
                        f"Skips: {cumulative_skips} | U/W [M: {u_m:.2e}, Vocab: {u_v:.2e}, 1D: {u_1d:.2e}]{drift_str}"
                    )

                rss_gb = get_current_rss_gb()
                mem_str = f"Host RSS: {rss_gb:.2f} GB | "
                devs = jax.devices()
                if devs and hasattr(devs[0], "memory_stats"):
                    stats = devs[0].memory_stats()
                    if stats and "bytes_in_use" in stats:
                        used_hbm = stats["bytes_in_use"] / (1024.0 ** 3)
                        limit_hbm = stats.get("bytes_limit", 0) / (1024.0 ** 3)
                        mem_str = f"Host RSS: {rss_gb:.2f} GB | HBM: {used_hbm:.1f}/{limit_hbm:.1f} GB | "

                print(
                    f"[Step {step:6d}/{args.total_steps}]{phase_tag} "
                    f"{tok_str}"
                    f"Loss: {loss_val:6.4f} | "
                    f"Steps/s: {steps_per_sec:5.2f} | "
                    f"Tok/s: {tok_per_sec:7,.0f} | "
                    f"{mem_str}"
                    f"LR: {cur_lr:.2e} | "
                    f"Samp_p: {cur_sampling_p:.2f}{gov_str}"
                    f"{diag_info}"
                )
                t_last_log = time.time()
                step_window_tokens = 0
                step_window_count = 0
                window_step_count = 0
                window_clipped_count = 0

            # Held-Out Validation Evaluation
            crossed_eval = bool(eval_batches) and ((step // max(1, args.eval_interval)) > (prev_step // max(1, args.eval_interval)))
            if crossed_eval:
                expected_eval_len = args.total_window if is_pretrain_source else args.seq_len
                cur_stage_name = "deep_reasoning" if in_deep_reasoning_phase else ("midtrain" if getattr(args, "dataset_source", "") in ("midtrain", "midtraining") else "pretrain")
                curriculum_shifted = False
                if is_pretrain_source:
                    try:
                        from src.smart_mixture import map_domain_label_to_tag
                        cur_eval_map = {map_domain_label_to_tag(d): w for d, w in zip(eval_domains, eval_weights)} if (eval_domains and eval_weights and len(eval_domains) == len(eval_weights)) else {}
                    except Exception:
                        cur_eval_map = {}
                    if local_streamer is not None and hasattr(local_streamer, "shares") and local_streamer.shares:
                        target_shares = dict(local_streamer.shares)
                        if set(cur_eval_map.keys()) != set(target_shares.keys()) or any(abs(cur_eval_map.get(k, 0.0) - float(target_shares[k])) > 0.01 for k in target_shares):
                            curriculum_shifted = True
                    elif in_deep_reasoning_phase and set(cur_eval_map.keys()) != {"math", "cosmopedia", "code", "edu_academic"}:
                        curriculum_shifted = True

                if (eval_batches and eval_batches[0][1].shape[1] != expected_eval_len) or curriculum_shifted:
                    if is_pretrain_source:
                        generate_held_out_split(cur_stage_name, expected_eval_len)
                    if eval_batches and eval_batches[0][1].shape[1] > expected_eval_len:
                        eval_batches = [
                            (b[0], b[1][:, :expected_eval_len], b[2][:, :expected_eval_len] if len(b) > 2 and b[2] is not None else None)
                            for b in eval_batches
                        ]
                eval_p = jax.device_put(_get_host_params(), devices[0]) if use_multi_tpu_local_sgd else params
                val_losses = [
                    float(eval_step_loss(eval_p, b[0], b[1], b[2] if len(b) > 2 else None, enc_alpha=enc_alpha))
                    for b in eval_batches
                ]
                if eval_weights and len(eval_weights) == len(val_losses):
                    eval_loss = float(sum(w * l for w, l in zip(eval_weights, val_losses)) / sum(eval_weights))
                else:
                    eval_loss = float(np.mean(val_losses)) if val_losses else float("inf")
                last_eval_loss = eval_loss
                eval_ppl = math.exp(min(eval_loss, 20.0))
                is_champion = eval_loss < best_eval_loss

                champ_tag = "🏆 (NEW CHAMPION RECORD)" if is_champion else ""
                print(f"📊 [Eval @ Step {step:,}] Composite Held-Out Loss: {eval_loss:.4f} | PPL: {eval_ppl:.2f} {champ_tag}")
                if len(val_losses) == len(eval_domains):
                    for idx, (d_name, d_val) in enumerate(zip(eval_domains, val_losses)):
                        d_ppl = math.exp(min(d_val, 20.0))
                        wt_str = f" ({eval_weights[idx]*100:.0f}%)" if (eval_weights and idx < len(eval_weights)) else ""
                        print(f"   ├─ {d_name + wt_str:<44}: Loss {d_val:.4f} (PPL: {d_ppl:.2f})")
                if hasattr(batcher, "get_tokens_by_dataset"):
                    ds_stats = batcher.get_tokens_by_dataset()
                    if ds_stats:
                        parts = [f"{k}: {v/1e9:.3f}B" for k, v in sorted(ds_stats.items())]
                        print(f"   └─ Ingested Tokens Breakdown: " + " | ".join(parts))

                # Dynamic Smart Mixture Rebalancing
                if smart_mixture_controller is not None and len(val_losses) == len(eval_domains):
                    try:
                        from src.smart_mixture import map_domain_label_to_tag
                        dom_losses = {map_domain_label_to_tag(d_name): float(d_val) for d_name, d_val in zip(eval_domains, val_losses)}
                        dom_toks = {}
                        if hasattr(batcher, "quota_mgr"):
                            snap = batcher.quota_mgr.get_snapshot()
                            for k, v in snap.items():
                                if isinstance(v, dict) and "tokens" in v:
                                    dom_toks[k] = v["tokens"]
                        elif hasattr(batcher, "get_tokens_by_dataset"):
                            dom_toks = batcher.get_tokens_by_dataset()

                        new_shares, mix_metrics = smart_mixture_controller.step(
                            step=step,
                            domain_val_losses=dom_losses,
                            domain_tokens=dom_toks,
                        )
                        if hasattr(batcher, "update_shares"):
                            batcher.update_shares(new_shares)

                        print(smart_mixture_controller.format_status_report(mix_metrics))
                    except Exception as e_mix_step:
                        print(f"⚠️ [Smart Mixture Engine] Dynamic step update notice: {e_mix_step}")

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
                        "use_rope": config.use_rope,
                        "max_phase_steps": config.max_phase_steps,
                        "num_sensory_layers": config.num_sensory_layers,
                        "use_centralized_readout": config.use_centralized_readout,
                        "use_shell_tap": config.use_shell_tap,
                        "use_bidirectional_tracts": config.use_bidirectional_tracts,
                        "use_sensory_core_coupling": config.use_sensory_core_coupling,
                        "use_parallel_plan_scan": config.use_parallel_plan_scan,
                        "preserve_shell_norm_clamp": config.preserve_shell_norm_clamp,
                        "top_k_resonance": config.top_k_resonance,
                        "sync_model": sync_model_name,
                        "cumulative_tokens": cumulative_tokens,
                        "cumulative_prompt_tokens": cumulative_prompt_tokens,
                        "cumulative_loss_tokens": cumulative_loss_tokens,
                        "cumulative_padded_tokens": cumulative_padded_tokens,
                        "lifetime_tokens": lifetime_tokens,
                        "target_tokens": getattr(args, "target_tokens", 200_000_000_000),
                        "effective_batch_size": args.batch_size * grad_accum,
                        "tokens_by_dataset": batcher.get_tokens_by_dataset() if hasattr(batcher, "get_tokens_by_dataset") else {},
                        "converged": bool(converged_target_met),
                        "timestamp": time.time(),
                    }
                    ckpt_mgr.dispatch(_get_host_params(), buffers, best_fname, meta_dict=meta_best, block_if_busy=False, opt_state=_get_host_opt_state())
                    if converged_target_met:
                        converged_fname = f"{args.preset}_converged_1.6.pt"
                        ckpt_mgr.dispatch(_get_host_params(), buffers, converged_fname, meta_dict=meta_best, block_if_busy=False, opt_state=_get_host_opt_state())

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
                            push_remote_loss_history_async(
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
                    "use_rope": config.use_rope,
                    "max_phase_steps": config.max_phase_steps,
                    "num_sensory_layers": config.num_sensory_layers,
                    "use_centralized_readout": config.use_centralized_readout,
                    "use_shell_tap": config.use_shell_tap,
                    "use_bidirectional_tracts": config.use_bidirectional_tracts,
                    "use_sensory_core_coupling": config.use_sensory_core_coupling,
                    "use_parallel_plan_scan": config.use_parallel_plan_scan,
                    "preserve_shell_norm_clamp": config.preserve_shell_norm_clamp,
                    "top_k_resonance": config.top_k_resonance,
                    "sync_model": sync_model_name,
                    "cumulative_tokens": cumulative_tokens,
                    "cumulative_prompt_tokens": cumulative_prompt_tokens,
                    "cumulative_loss_tokens": cumulative_loss_tokens,
                    "cumulative_padded_tokens": cumulative_padded_tokens,
                    "lifetime_tokens": lifetime_tokens,
                    "target_tokens": getattr(args, "target_tokens", 200_000_000_000),
                    "effective_batch_size": args.batch_size * grad_accum,
                    "tokens_by_dataset": batcher.get_tokens_by_dataset() if hasattr(batcher, "get_tokens_by_dataset") else {},
                    "timestamp": time.time(),
                }
                ckpt_mgr.dispatch(_get_host_params(), buffers, latest_fname, meta_dict=meta_latest, opt_state=_get_host_opt_state())

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
                        push_remote_loss_history_async(
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
                    push_remote_token_tracker_async(
                        model=sync_model_name or args.preset,
                        cumulative_tokens=cumulative_tokens,
                        target_tokens=getattr(args, "target_tokens", 200_000_000_000),
                        tokens_by_dataset=ds_stats,
                        effective_batch_size=args.batch_size * grad_accum,
                        step=step,
                        loss=float(loss_val),
                        tok_per_sec=float(tok_per_sec) if 'tok_per_sec' in locals() else 0.0,
                        run_id=run_id,
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
                    opt_state=_get_host_opt_state(),
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

            # Record step as fully verified and completed
            last_successful_step = step

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
    except Exception as e:
        training_error = e
        print(f"\n❌ [Fatal Error] Training encountered exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        step_val = last_successful_step if "last_successful_step" in locals() else (step if "step" in locals() else start_step)
        print(f"\n💾 Flushing final checkpoint state for last verified Step {step_val:,}...")
        final_fname = f"{args.preset}_latest.pt"
        b_ref = batcher if ("batcher" in locals() and batcher is not None) else (elastic_batcher if ("elastic_batcher" in locals() and elastic_batcher is not None) else None)
        ds_stats = b_ref.get_tokens_by_dataset() if (b_ref is not None and hasattr(b_ref, "get_tokens_by_dataset")) else {}
        final_meta = {
            "step": step_val,
            "loss": float(loss) if "loss" in locals() else 0.0,
            "ema_loss": ema_loss if ("ema_loss" in locals() and ema_loss is not None) else (float(loss) if "loss" in locals() else 0.0),
            "eval_loss": best_eval_loss if "best_eval_loss" in locals() else float("inf"),
            "preset": args.preset,
            "use_rope": config.use_rope,
            "max_phase_steps": config.max_phase_steps,
            "sync_model": sync_model_name if "sync_model_name" in locals() else args.preset,
            "cumulative_tokens": cumulative_tokens if "cumulative_tokens" in locals() else 0,
            "cumulative_prompt_tokens": cumulative_prompt_tokens if "cumulative_prompt_tokens" in locals() else 0,
            "cumulative_loss_tokens": cumulative_loss_tokens if "cumulative_loss_tokens" in locals() else 0,
            "cumulative_padded_tokens": cumulative_padded_tokens if "cumulative_padded_tokens" in locals() else 0,
            "lifetime_tokens": lifetime_tokens if "lifetime_tokens" in locals() else 0,
            "target_tokens": getattr(args, "target_tokens", 200_000_000_000),
            "effective_batch_size": args.batch_size * (grad_accum if "grad_accum" in locals() else 1),
            "tokens_by_dataset": ds_stats,
            "timestamp": time.time(),
        }
        if "ckpt_mgr" in locals() and ckpt_mgr is not None:
            ckpt_mgr.flush(_get_host_params(), buffers, final_fname, meta_dict=final_meta, opt_state=_get_host_opt_state())
            ckpt_mgr.shutdown(wait=True)

        if args.cloud_sync:
            wait_for_telemetry_uploads(timeout=10.0)

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
                    run_id=run_id if "run_id" in locals() else None,
                    url=args.cloud_url,
                    token=args.cloud_token,
                )
            except Exception:
                pass
        if "_multi_tpu_stop" in locals() and _multi_tpu_stop is not None:
            _multi_tpu_stop.set()
            if "_multi_tpu_thread" in locals() and _multi_tpu_thread is not None:
                _multi_tpu_thread.join(timeout=2.0)
        if "local_streamer" in locals() and local_streamer is not None:
            local_streamer.stop()
        if b_ref is not None:
            if hasattr(b_ref, "stop"):
                b_ref.stop()
            elif hasattr(b_ref, "shutdown"):
                b_ref.shutdown()
        if "training_error" in locals() and training_error is not None:
            print(f"❌ Training aborted due to fatal error at Step {step_val:,}: {training_error}")
            raise training_error
        else:
            print(f"🎉 Training session completed at Step {step_val:,}!")


def run_training(**kwargs):
    """Programmatic entrypoint for Jupyter / Colab notebooks."""
    args = parse_args([])
    for k, v in kwargs.items():
        attr = k.replace("-", "_")
        if attr in ("effective_batch_size", "effective_batch"):
            args.effective_batch_size_explicit = True
        if hasattr(args, attr):
            setattr(args, attr, v)
        elif hasattr(args, k):
            setattr(args, k, v)
        elif k in ("checkpoint_dir", "ckpt_dir"):
            setattr(args, "checkpoint_dir", v)
        elif k in ("max_steps", "total_steps", "steps"):
            setattr(args, "total_steps", v)
            args.total_steps_explicit = True
        elif k == "remat":
            setattr(args, "remat", str(v).lower())
    return main(args)


if __name__ == "__main__":
    main()
