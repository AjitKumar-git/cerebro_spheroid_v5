#!/usr/bin/env python3
"""
CerebroSpheroid V5: Graded Curriculum Pre-Training & Agentic Fine-Tuning Pipeline
==================================================================================

Full 1,000,000-Step Developmental Curriculum:
  • Phase 1 (0 -> 50k steps)      : Basic Syntax, Grammar & Low-Entropy Structure
  • Phase 2 (50k -> 500k steps)   : Knowledge, Reasoning & Calculation (GSM8k, OpenHermes)
  • Phase 3 (500k -> 900k steps)  : Modern Agentic Workflows & Multi-Turn Tool-Use
  • Phase 4 (900k -> 1,000k steps): Model Consolidation, Replay & Energy Annealing

Storage-Optimized:
  • Zero upfront token disk dumps.
  • Streams memory-mapped row batches from Parquets on-the-fly.
  • Adafactor optimizer with 15 MB state tracking.
"""

import argparse
import contextlib
import datetime
import gc
import math
import os
import random
import shutil
import signal
import sys
import time
from typing import Optional, Dict, Any, List, Union, Tuple

# Kaggle / Python 3.12+ PyArrow compatibility shim for HuggingFace `datasets`
try:
    import pyarrow as _pa
    if not hasattr(_pa, "PyExtensionType") and hasattr(_pa, "ExtensionType"):
        _pa.PyExtensionType = _pa.ExtensionType
except ImportError:
    pass

# Set PyTorch memory optimization flag before torch initialization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.optimization import Adafactor
from transformers import AutoTokenizer

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    PRESETS_V5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.agentic_harness import ActionType, ToolActionHead
from src.cloud_sync import (
    check_remote_manifest,
    detect_source_environment,
    notify_training_start,
    pull_checkpoint,
    push_checkpoint_async,
    push_remote_loss_history,
    select_highest_step_slot,
    wait_for_in_flight_uploads,
)
from src.graded_curriculum import (
    CurriculumPhase,
    CurriculumScheduleConfig,
    GradedCurriculumStreamer,
    PHASE_NAMES,
    compute_curriculum_lr,
    get_curriculum_phase,
)

_STOP_REQUESTED = False


def _handle_shutdown_signal(signum, frame):
    """Record SIGINT/SIGTERM so distributed ranks can save and exit cleanly."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _has_native_bf16(device_index: int = 0) -> bool:
    """Return True only for hardware Tensor Core BF16 (Ampere sm_80+ or newer)."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device_index)
    if major < 8:
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return bool(torch.cuda.is_bf16_supported())


def _select_amp_dtype(use_cuda: bool, device_index: int = 0, preferred_dtype: str = "auto") -> torch.dtype:
    """Pick bfloat16 on sm_80+, native FP16 on pre-Ampere GPUs (e.g., Tesla T4), else FP32."""
    if not use_cuda:
        return torch.float32
    pref = (preferred_dtype or "auto").lower()
    if pref in ("float16", "fp16", "half"):
        return torch.float16
    elif pref in ("bfloat16", "bf16"):
        return torch.bfloat16
    elif pref in ("float32", "fp32"):
        return torch.float32
    return torch.bfloat16 if _has_native_bf16(device_index) else torch.float16


def _sync_stop_signal(local_stop: bool, is_distributed: bool, device: torch.device) -> bool:
    """Ensure every DDP rank observes the exact same stop decision to avoid collective hangs."""
    if not is_distributed:
        return local_stop
    flag = torch.tensor([1 if local_stop else 0], dtype=torch.int64, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


class _CerebroDDPModule(nn.Module):
    """Wraps CerebroSpheroidV5 + ToolActionHead under one DDP reducer."""

    def __init__(self, model: CerebroSpheroidV5, action_head: ToolActionHead):
        super().__init__()
        self.model = model
        self.action_head = action_head

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        feedback_tokens: Optional[torch.Tensor] = None,
        sampling_p: float = 1.0,
        pitfall_prob: float = 0.0,
        pad_id: int = 0,
        perceptual_inputs: Optional[torch.Tensor] = None,
        compute_action: bool = False,
        persona_id: Optional[int] = None,
    ):
        ce_loss, logits = self.model(
            input_ids,
            targets=targets,
            feedback_tokens=feedback_tokens,
            sampling_p=sampling_p,
            pitfall_prob=pitfall_prob,
            pad_id=pad_id,
            perceptual_inputs=perceptual_inputs,
            persona_id=persona_id,
        )
        action_logits = None
        if compute_action:
            cortical_mantle = self.model.current_spheroid_state[:, -1, :]
            action_logits, _ = self.action_head(cortical_mantle)
        return ce_loss, logits, action_logits


def parse_args():
    parser = argparse.ArgumentParser(description="CerebroSpheroid V5 Graded Curriculum Training")
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--test-steps", type=int, default=None, help="If set, stops after test-steps (for local dry run)")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU micro-batch size (default: 8)")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps (default: auto-computed to hit --target-batch-size)")
    parser.add_argument("--target-batch-size", "--prompts-per-step", dest="target_batch_size", type=int, default=64, help="Global prompts per optimizer step across all GPUs (default: 64, matching TPU)")
    parser.add_argument("--prompt-len", type=int, default=32, help="Prompt context length in TPU-parity static mode (default: 32)")
    parser.add_argument("--seq-len", type=int, default=256, help="Target sequence length in TPU-parity static mode (default: 256)")
    parser.add_argument("--target-seq-len", "--target_seq_len", dest="target_seq_len", type=int, default=None, help="Target sequence length to transition toward (e.g. 512 or 1024)")
    parser.add_argument("--seq-transition-steps", "--seq_transition_steps", dest="seq_transition_steps", type=int, default=0, help="Steps over which to transition sequence length (0 = immediate transition to target_seq_len)")
    parser.add_argument("--deliberation-mode", "--deliberation_mode", dest="deliberation_mode", type=str, default="teacher_forcing", choices=["teacher_forcing", "scheduled", "pitfall_recovery", "internal"], help="Deliberation feedback mode: 'teacher_forcing' uses pure ground truth (peak throughput); 'scheduled' anneals sampling_p; 'pitfall_recovery' trains error recovery")
    parser.add_argument("--schedule-steps", "--schedule_steps", dest="schedule_steps", type=int, default=0, help="Steps to anneal from teacher-forcing down to min_sampling_p (default: 0)")
    parser.add_argument("--min-sampling-p", "--min_sampling_p", dest="min_sampling_p", type=float, default=1.0, help="Minimum teacher sampling floor (default: 1.0 for teacher-forcing)")
    parser.add_argument("--pitfall-prob", "--pitfall_prob", dest="pitfall_prob", type=float, default=0.15, help="Probability of exploring counterfactual/pitfall thoughts during deliberation (default: 0.15)")
    parser.add_argument("--loss-window-governor", action=argparse.BooleanOptionalAction, default=True, help="Automatically clamp deliberation and float sampling_p back to 1.0 if loss climbs above window ceiling")
    parser.add_argument("--loss-ceiling", type=float, default=3.25, help="Upper loss boundary: if smoothed loss exceeds this, restore full (100%%) teacher forcing to prevent loss climb (default: 3.25)")
    parser.add_argument("--loss-floor", type=float, default=2.80, help="Lower loss boundary: if smoothed loss is at/below this, permit full deliberation at min_sampling_p (default: 2.80)")
    parser.add_argument("--loss-lr-damping", action=argparse.BooleanOptionalAction, default=True, help="Damp learning rate when loss is climbing towards ceiling to prevent optimizer overshoot")
    parser.add_argument("--max-lr", type=float, default=5e-4, help="Peak learning rate (default: 5e-4 matching TPU)")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Linear warmup steps (default: 500 matching TPU)")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between log lines")
    parser.add_argument("--eval-interval", type=int, default=250, help="Steps between fixed held-out validation passes (default: 250)")
    parser.add_argument("--save-interval", type=int, default=1000, help="Steps between checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/graded_curriculum", help="Checkpoint directory")
    parser.add_argument("--cloud-sync", action="store_true", help="Enable bi-directional checkpoint sync with bioflo.in")
    parser.add_argument("--cloud-sync-interval", type=int, default=100, help="Steps between background cloud sync uploads")
    parser.add_argument("--offline", action="store_true", help="Force offline mode using local datasets & cache")
    parser.add_argument("--elastic-window", action=argparse.BooleanOptionalAction, default=False, help="Enable multi-bucket elastic sliding window instead of static TPU (32, 256) shape")
    parser.add_argument("--total-window", type=int, default=288, help="Total sequence budget (prompt + response) for elastic window")
    parser.add_argument("--tail-prob", type=float, default=0.25, help="V4 tail probability for EOS representation in window sampling")
    parser.add_argument("--num-mtp-heads", "--num_mtp_heads", dest="num_mtp_heads", type=int, default=1, choices=[1, 2, 3, 4], help="Multi-Token Prediction heads (1: standard single-token emission; 2 or 3: speculative decoding & future-token predictive training)")
    parser.add_argument("--persona-id", "--persona_id", dest="persona_id", type=int, default=None, choices=[0, 1, 2, 3], help="Cognitive Persona Steering basis (None: unbiased representation; 0: Reasoner, 1: Explorer, 2: Concise, 3: Verifier)")
    parser.add_argument("--use-adaptive-thinking", "--use_adaptive_thinking", dest="use_adaptive_thinking", action=argparse.BooleanOptionalAction, default=True, help="Toggle dynamic subconscious deliberation halting gate (True: adaptive thinking up to max cycles; False: fast fixed cycles)")
    parser.add_argument("--no-adaptive-thinking", "--no_adaptive_thinking", dest="use_adaptive_thinking", action="store_false", help="Disable adaptive thinking (use fast fixed cycles)")
    parser.add_argument("--thinking-cycles", "--thinking_cycles", dest="thinking_cycles", type=int, default=4, help="Subconscious deliberation iterations per token (default: 4)")
    parser.add_argument("--max-thinking-cycles", "--max_thinking_cycles", dest="max_thinking_cycles", type=int, default=16, help="Maximum thinking cycles when adaptive thinking is active (default: 16)")
    parser.add_argument("--model-preset", type=str, default="v5_small", choices=["v5_small", "v5_compact", "v5_base"], help="Model preset architecture tier (v5_small: 101M, v5_compact: 244M, v5_base: 617M)")
    parser.add_argument("--mix", type=str, default="auto", choices=["auto", "foundation", "reasoning", "code", "agentic", "consolidation"], help="Curriculum data mixture selection")
    parser.add_argument("--amp-dtype", "--precision", dest="amp_dtype", type=str, default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"], help="Mixed precision dtype: 'auto' (native BF16 on Ampere+, FP16 on Turing/T4), or explicitly 'float16' / 'bfloat16' / 'float32'")
    parser.add_argument("--optimizer", type=str, default="spheroid", choices=["spheroid", "adafactor", "adamw"], help="Optimizer to use (default: custom SpheroidOptimizer with Polar Newton-Schulz & spectral clamp)")
    parser.add_argument("--from-scratch", "--from_scratch", dest="from_scratch", action="store_true", default=False, help="Start clean from Step 0 without restoring local/cloud checkpoints")
    parser.add_argument("--max-hours", type=float, default=None, help="Wall-clock budget in hours for clean checkpoint save before timeout")
    parser.add_argument("--seed", type=int, default=42, help="Base PRNG seed (offset per DDP rank for data streaming)")
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=0, help="Local device rank when launched via distributed runner")
    return parser.parse_args()


def load_local_tokenizer(offline: bool = True):
    """Loads modern 32,000-vocabulary LLaMA tokenizer from bundled data/tokenizer directory."""
    from src.corpus import build_tokenizer
    return build_tokenizer("llama")


def _atomic_save_and_link(payload: dict, primary_path: str, legacy_path: str) -> None:
    """Save checkpoint atomically to primary_path and hardlink/copy to legacy_path."""
    tmp_primary = f"{primary_path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_primary)
    os.replace(tmp_primary, primary_path)
    if os.path.abspath(legacy_path) == os.path.abspath(primary_path):
        return
    tmp_legacy = f"{legacy_path}.tmp.{os.getpid()}"
    if os.path.exists(tmp_legacy):
        os.unlink(tmp_legacy)
    try:
        os.link(primary_path, tmp_legacy)
    except OSError:
        shutil.copy2(primary_path, tmp_legacy)
    os.replace(tmp_legacy, legacy_path)


def _to_bfloat16_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Converts floating-point tensors (float32/float64) in a state dict to CPU bfloat16 for compact serialization and upload."""
    converted = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point() and v.dtype in (torch.float32, torch.float64):
                converted[k] = v.detach().to(device="cpu", dtype=torch.bfloat16)
            else:
                converted[k] = v.detach().to(device="cpu")
        else:
            converted[k] = v
    return converted


def _save_and_push_checkpoints(
    checkpoint_dir: str,
    model_preset: str,
    raw_model: CerebroSpheroidV5,
    raw_action_head: ToolActionHead,
    config: ModelConfigV5,
    step: int,
    loss_val: float,
    best_loss: float,
    need_local_save: bool,
    need_cloud_sync: bool,
    eval_loss: Optional[float] = None,
    final_save: bool = False,
) -> float:
    """Persist local latest/best/step checkpoints (in bfloat16) and enqueue optional bioflo.in upload."""
    metric_loss = eval_loss if (eval_loss is not None and math.isfinite(eval_loss) and eval_loss > 0.0) else loss_val
    model_sd = _to_bfloat16_state(raw_model.state_dict())
    action_sd = _to_bfloat16_state(raw_action_head.state_dict())
    ckpt_payload = {
        "step": step,
        "model_state": model_sd,
        "action_head_state": action_sd,
        "config": config,
        "loss": loss_val,
        "eval_loss": eval_loss if eval_loss is not None else loss_val,
        "model_preset": model_preset,
    }
    latest_path = os.path.join(checkpoint_dir, f"{model_preset}_latest.pt")
    best_path = os.path.join(checkpoint_dir, f"{model_preset}_best.pt")
    legacy_latest = os.path.join(checkpoint_dir, "latest.pt")
    legacy_best = os.path.join(checkpoint_dir, "best.pt")

    if final_save:
        final_ckpt = os.path.join(checkpoint_dir, f"{model_preset}_graded_final.pt")
        torch.save(ckpt_payload, final_ckpt)
    elif need_local_save:
        ckpt_path = os.path.join(checkpoint_dir, f"{model_preset}_step_{step}.pt")
        torch.save(ckpt_payload, ckpt_path)
        print(f"--> Saved checkpoint: {ckpt_path}")

    _atomic_save_and_link(ckpt_payload, latest_path, legacy_latest)

    is_best = False
    if math.isfinite(metric_loss) and metric_loss > 0.0 and metric_loss < best_loss:
        best_loss = metric_loss
        _atomic_save_and_link(ckpt_payload, best_path, legacy_best)
        is_best = True
        loss_label = f"eval loss: {eval_loss:.4f}" if eval_loss is not None else f"loss: {best_loss:.4f}"
        print(f"🌟 New best {loss_label} -> saved {best_path}")

    if need_cloud_sync:
        push_checkpoint_async(latest_path, step=step, loss=metric_loss, target_type="latest", model=model_preset)
        if is_best:
            push_checkpoint_async(best_path, step=step, loss=best_loss, target_type="best", model=model_preset)
        try:
            import datetime
            rec = {
                "step": step,
                "loss": float(loss_val),
                "eval_loss": float(eval_loss) if (eval_loss is not None and math.isfinite(eval_loss)) else None,
                "status": "NEW_CHAMPION" if is_best else "SMOOTH_DESCENT",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source": detect_source_environment(),
            }
            push_remote_loss_history(model=model_preset, records=[rec])
        except Exception:
            pass

    return best_loss


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    is_distributed = world_size > 1
    is_main = (rank == 0)

    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    if torch.cuda.is_available():
        gpu_idx = local_rank if is_distributed else 0
        torch.cuda.set_device(gpu_idx)
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        gpu_idx = 0
        device = torch.device("cpu")
    use_cuda = (device.type == "cuda") and torch.cuda.is_available()

    if is_distributed:
        backend = "nccl" if use_cuda else "gloo"
        pg_kwargs = {"backend": backend, "timeout": datetime.timedelta(minutes=30)}
        if use_cuda:
            pg_kwargs["device_id"] = device
        try:
            dist.init_process_group(**pg_kwargs)
        except TypeError:
            pg_kwargs.pop("device_id", None)
            dist.init_process_group(**pg_kwargs)

    try:
        _run_training(
            args=args,
            world_size=world_size,
            rank=rank,
            gpu_idx=gpu_idx,
            device=device,
            use_cuda=use_cuda,
            is_distributed=is_distributed,
            is_main=is_main,
        )
    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


def _run_training(
    args,
    world_size: int,
    rank: int,
    gpu_idx: int,
    device: torch.device,
    use_cuda: bool,
    is_distributed: bool,
    is_main: bool,
):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set offline flags if requested
    if args.offline:
        args.cloud_sync = False
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    amp_dtype = _select_amp_dtype(use_cuda, gpu_idx, preferred_dtype=getattr(args, "amp_dtype", "auto"))
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and amp_dtype == torch.float16))
    if use_cuda and amp_dtype == torch.bfloat16:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Sequence Length Transition & Context Budget Progression
    if getattr(args, "target_seq_len", None) is not None and args.target_seq_len > args.seq_len:
        if is_main:
            print(f"\n🔄 [Sequence Transition] Advancing target sequence length from {args.seq_len} -> {args.target_seq_len} tokens for deep reasoning!")
        args.seq_len = args.target_seq_len

    if args.total_window == 288 and not args.elastic_window and (args.seq_len != 256 or args.prompt_len != 32):
        args.total_window = args.prompt_len + args.seq_len

    # Resolve effective grad_accum for global prompt/token parity with TPU (64 prompts/step)
    if args.grad_accum is None:
        denom = max(1, world_size * max(1, args.batch_size))
        args.grad_accum = max(1, math.ceil(args.target_batch_size / denom))
    effective_prompts = world_size * args.batch_size * args.grad_accum
    step_window = args.total_window if args.elastic_window else (args.prompt_len + args.seq_len)
    effective_tokens = effective_prompts * step_window

    if is_main:
        print(f"=== CerebroSpheroid V5 Graded Curriculum Engine ===")
        world_str = f" | DDP World Size: {world_size}" if is_distributed else ""
        print(f"Device: {device} | Total Target Steps: {args.total_steps}{world_str}")
        if use_cuda:
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            vram_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
            if amp_dtype == torch.bfloat16:
                prec_label = "bfloat16 (Tensor Core BF16)"
            elif amp_dtype == torch.float16:
                prec_label = "float16 (Hardware-Native sm_75 Tensor Cores + GradScaler)"
            else:
                prec_label = "float32 (Full FP32)"
            print(f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM/GPU) | Precision: {prec_label}")
        else:
            print(f"Precision: float32 (CPU)")
        shape_str = f"Elastic(budget={args.total_window})" if args.elastic_window else f"Prompt={args.prompt_len}, Target={args.seq_len}"
        print(f"TPU Parity Target : {effective_prompts} prompts/step ({world_size}x{args.batch_size}x{args.grad_accum}) | {shape_str} = {effective_tokens:,} tok/step")
        data_desc = "Offline Local Corpus (data/*.parquet)" if args.offline else "Online HF Stream (Cosmopedia-v2 + FineWeb-Edu + OpenHermes/SmolTalk)"
        print(f"Data Source       : {data_desc}")

    # 1. Initialize Tokenizer & Adjust Checkpoint Dir
    tokenizer = load_local_tokenizer(offline=args.offline)
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_id is None:
        pad_id = 0

    if args.checkpoint_dir == "checkpoints/graded_curriculum":
        args.checkpoint_dir = f"checkpoints/{args.model_preset}"
    if is_main:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    if is_distributed:
        dist.barrier()

    # 1b. Checkpoint Discovery & Cloud Sync Pull (before model init to match checkpoint topology)
    start_step = 0
    best_loss = float("inf")
    local_latest = os.path.join(args.checkpoint_dir, f"{args.model_preset}_latest.pt")
    tpu_latest = os.path.join(args.checkpoint_dir, "tpu_v5e_1_latest.pt")
    legacy_latest = os.path.join(args.checkpoint_dir, "latest.pt")

    if args.cloud_sync and not args.from_scratch and is_main:
        print(f"[cloud_sync] Checking remote manifest on bioflo.in for {args.model_preset}...")
        notify_training_start(step=start_step, model=args.model_preset)
        manifest = check_remote_manifest(model=args.model_preset)
        slot_model, remote_step, remote_loss = select_highest_step_slot(
            manifest, args.model_preset, "latest"
        )
        if manifest.get("has_latest", False) or remote_step > 0:
            print(
                f"🔎 [cloud_sync] Selected remote slot model={slot_model or 'root'} "
                f"(Step {remote_step:,}, Loss {remote_loss:.4f}) for latest.pt"
            )
            pulled = pull_checkpoint(local_latest, target_type="latest", model=slot_model)
            if pulled:
                print(f"✓ Restored {args.model_preset} latest checkpoint from cloud hub!")
    if is_distributed:
        dist.barrier()

    load_cand = None
    if not args.from_scratch:
        for cand in [local_latest, tpu_latest, legacy_latest]:
            if os.path.exists(cand):
                load_cand = cand
                break

    ckpt = None
    max_seq_budget = max(32768, getattr(args, "total_window", 1024) + 512)
    if load_cand is not None:
        try:
            ckpt = torch.load(load_cand, map_location="cpu", weights_only=False)
            config = infer_v5_config_from_state_dict(
                ckpt,
                default_preset=args.model_preset,
                vocab_size_override=getattr(tokenizer, "vocab_size", 32000),
                max_seq_len=max_seq_budget,
            )
        except Exception as e:
            if is_main:
                print(f"⚠️ Could not inspect checkpoint from {load_cand}: {e}")
            ckpt = None

    if ckpt is None:
        preset_kwargs = PRESETS_V5[args.model_preset].copy()
        preset_kwargs["max_seq_len"] = max_seq_budget
        preset_kwargs["use_sparse_mot"] = True
        preset_kwargs["use_energy_regularization"] = True
        if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size:
            preset_kwargs["vocab_size"] = tokenizer.vocab_size
        config = ModelConfigV5(**preset_kwargs)

    if getattr(args, "num_mtp_heads", None) is not None:
        config.num_mtp_heads = args.num_mtp_heads
    if getattr(args, "use_adaptive_thinking", None) is not None:
        config.use_adaptive_thinking = args.use_adaptive_thinking
    if getattr(args, "thinking_cycles", None) is not None:
        config.thinking_cycles = args.thinking_cycles
    if getattr(args, "max_thinking_cycles", None) is not None:
        config.max_thinking_cycles = args.max_thinking_cycles

    torch.manual_seed(args.seed)
    raw_model = CerebroSpheroidV5(config).to(device)
    raw_action_head = ToolActionHead(surface_dim=config.surface_dim, num_actions=len(ActionType)).to(device)
    raw_combined = _CerebroDDPModule(raw_model, raw_action_head)

    total_params = sum(p.numel() for p in raw_model.parameters()) + sum(p.numel() for p in raw_action_head.parameters())
    if is_main:
        print(f"Model Preset: {args.model_preset} ({config.num_shells} shells, {config.surface_dim} surface dim, tied={config.use_tied_embeddings})")
        print(f"Total Parameters (Model + ActionHead): {total_params:,} (~{total_params/1e6:.1f}M)")

    if ckpt is not None and load_cand is not None:
        try:
            load_v5_checkpoint_into_torch(raw_model, ckpt, verbose=is_main)
            act_state = ckpt.get("action_head_state", ckpt.get("action_head_state_dict"))
            if isinstance(act_state, dict):
                raw_action_head.load_state_dict(act_state, strict=False)
            start_step = int(ckpt.get("step", 0) or 0)
            ckpt_eval_loss = ckpt.get("eval_loss", ckpt.get("best_eval_loss", None))
            ckpt_loss = ckpt.get("loss", None)
            if ckpt_eval_loss is not None and math.isfinite(float(ckpt_eval_loss)) and float(ckpt_eval_loss) > 0.0:
                best_loss = float(ckpt_eval_loss)
            elif ckpt_loss is not None and math.isfinite(float(ckpt_loss)) and float(ckpt_loss) > 0.0:
                best_loss = float(ckpt_loss)
            else:
                best_loss = float("inf")
            if is_main:
                print(f"✓ Resumed from {os.path.basename(load_cand)} at step {start_step:,} (best loss: {best_loss:.4f})")
        except Exception as e:
            if is_main:
                print(f"⚠️ Could not load checkpoint from {load_cand}: {e}")
        finally:
            del ckpt
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
    elif args.from_scratch and is_main:
        print("🌱 [--from-scratch] Starting clean training from Step 0.")

    if is_distributed:
        ddp_kwargs = {"find_unused_parameters": True}
        if use_cuda:
            ddp_kwargs["device_ids"] = [gpu_idx]
            ddp_kwargs["output_device"] = gpu_idx
        train_module = DDP(raw_combined, **ddp_kwargs)
    else:
        train_module = raw_combined

    # 2. Setup Memory-Saving Curriculum Streamer (distinct seed & HF shard per DDP rank)
    schedule_config = CurriculumScheduleConfig(
        total_steps=args.total_steps,
        phase_1_end=50_000,
        phase_2_end=500_000,
        phase_3_end=900_000,
        warmup_steps=args.warmup_steps,
        max_lr=args.max_lr,
        prompt_len=args.prompt_len,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        multimodal_sample_ratio=0.0,
    )

    streamer = GradedCurriculumStreamer(
        tokenizer=tokenizer,
        config=schedule_config,
        seed=args.seed + rank * 10007,
        mix=args.mix,
        search_dirs=["data"],
        offline=args.offline,
        rank=rank,
        world_size=world_size,
    )

    # 3. Setup Optimizer (Default: Custom SpheroidOptimizer with Polar Newton-Schulz)
    all_params = list(raw_model.parameters()) + list(raw_action_head.parameters())
    if args.optimizer == "spheroid":
        from src.spheroid_optimizer import build_spheroid_optimizer
        optimizer = build_spheroid_optimizer(
            model=raw_combined,
            lr=schedule_config.max_lr,
            lr_1d=0.2 * schedule_config.max_lr,
            lr_vocab=min(schedule_config.max_lr, 0.0006),
            spectral_clamp=1.0,
        )
        if is_main:
            print(f"[Optimizer] Configured SpheroidOptimizer (Polar Newton-Schulz NS-5 + Persistent Spectral Clamp)")
    elif args.optimizer == "adafactor":
        optimizer = Adafactor(
            all_params,
            lr=schedule_config.max_lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
        if is_main:
            print(f"[Optimizer] Configured Adafactor Optimizer (~15MB memory footprint)")
    else:
        optimizer = torch.optim.AdamW(all_params, lr=schedule_config.max_lr, weight_decay=0.01)
        if is_main:
            print(f"[Optimizer] Configured standard AdamW Optimizer")

    for pg in optimizer.param_groups:
        pg.setdefault("initial_lr", pg["lr"])

    action_loss_fn = nn.CrossEntropyLoss()

    if is_main:
        print("\n--- Starting Graded Pre-Training Pipeline ---")
    start_time = time.time()
    max_seconds = (args.max_hours * 3600.0) if (args.max_hours is not None and args.max_hours > 0) else None
    accum_loss = 0.0
    accum_act_loss = 0.0
    steps_since_log = 0
    last_avg_loss = best_loss if math.isfinite(best_loss) else 0.0
    tokens_processed = 0
    step = start_step
    max_steps = (start_step + args.test_steps) if args.test_steps else args.total_steps

    if step >= max_steps:
        if is_main:
            print(f"✓ Already reached target step ({step:,} >= {max_steps:,}). Nothing to train.")
        return

    current_phase = get_curriculum_phase(step, schedule_config)
    if is_main:
        print(f"Entering {PHASE_NAMES[current_phase]} (Step {step:,} -> {max_steps:,})")

    train_module.train()

    static_buckets = None if args.elastic_window else [("static", args.prompt_len, args.seq_len)]
    window_budget = args.total_window if args.elastic_window else (args.prompt_len + args.seq_len)

    fixed_eval_batches = []
    if args.eval_interval and args.eval_interval > 0:
        try:
            eval_streamer = GradedCurriculumStreamer(
                tokenizer=tokenizer,
                config=schedule_config,
                seed=99999,
                mix=args.mix,
                search_dirs=["data"],
                offline=args.offline,
                rank=0,
                world_size=1,
            )
            for _ in range(2):
                ev_p, ev_t, ev_fb, _, _, _, _ = eval_streamer.get_elastic_batch(
                    step=0,
                    device=device,
                    batch_size=min(8, args.batch_size),
                    total_window=window_budget,
                    tail_probability=args.tail_prob,
                    buckets=static_buckets,
                )
                fixed_eval_batches.append((ev_p, ev_t, ev_fb))
            eval_streamer.close()
            if is_main:
                print(f"✓ Prepared {len(fixed_eval_batches)} fixed held-out evaluation batches for periodic tracking")
        except Exception as e:
            if is_main:
                print(f"ℹ️ Could not pre-generate held-out eval batches: {e}")

    ema_loss = None
    latest_eval_loss = None
    best_eval_loss = best_loss if math.isfinite(best_loss) else float("inf")

    def get_sampling_p(
        step_idx: int,
        start_s: int,
        sched_steps: int,
        min_p: float,
        mode: str,
        nominal_pitfall: float = 0.0,
        current_loss: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        if mode == "teacher_forcing" or min_p >= 1.0 - 1e-4:
            return 1.0, 0.0, 0.0
        if sched_steps <= 0:
            nominal_p = min_p
        elif mode == "internal":
            nominal_p = min_p
        else:
            elapsed = max(0, step_idx - start_s)
            ratio = min(1.0, float(elapsed) / float(sched_steps))
            coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
            nominal_p = float(min_p + (1.0 - min_p) * coeff)

        # Loss-Window Governor: automatically keeps loss strictly within [loss_floor, loss_ceiling]
        alpha_gov = 0.0
        if getattr(args, "loss_window_governor", True) and current_loss is not None and math.isfinite(current_loss):
            c_floor = float(getattr(args, "loss_floor", 2.80))
            c_ceil = float(getattr(args, "loss_ceiling", 3.25))
            if c_ceil > c_floor:
                alpha_gov = max(0.0, min(1.0, (float(current_loss) - c_floor) / (c_ceil - c_floor)))

        effective_p = nominal_p + alpha_gov * (1.0 - nominal_p)
        effective_pitfall = nominal_pitfall * (1.0 - alpha_gov)
        if effective_p >= 1.0 - 1e-4:
            effective_p = 1.0
        return float(effective_p), float(effective_pitfall), float(alpha_gov)

    ema_loss: Optional[float] = None

    try:
        while step < max_steps:
            local_stop = _STOP_REQUESTED or (
                is_main and max_seconds is not None and (time.time() - start_time) >= max_seconds
            )
            if _sync_stop_signal(local_stop, is_distributed, device):
                if is_main:
                    print(f"\n⏳ Graceful shutdown triggered at step {step:,}; flushing latest checkpoints...")
                break

            # Check for phase transitions
            new_phase = get_curriculum_phase(step, schedule_config)
            if new_phase != current_phase:
                current_phase = new_phase
                if is_main:
                    print(f"\n=======================================================")
                    print(f"★ PHASE TRANSITION at Step {step}: Entering {PHASE_NAMES[new_phase]}")
                    print(f"=======================================================\n")
                    # Save checkpoint at phase boundary
                    boundary_ckpt = os.path.join(args.checkpoint_dir, f"v5_phase_{int(new_phase)}_boundary_{step}.pt")
                    torch.save({
                        "step": step,
                        "model_state": _to_bfloat16_state(raw_model.state_dict()),
                        "action_head_state": _to_bfloat16_state(raw_action_head.state_dict()),
                        "config": config,
                    }, boundary_ckpt)
                    print(f"[Checkpoint Saved] Phase boundary snapshot: {boundary_ckpt}")

            cur_sampling_p, cur_pitfall_prob, alpha_gov = get_sampling_p(
                step,
                start_step,
                args.schedule_steps,
                args.min_sampling_p,
                args.deliberation_mode,
                nominal_pitfall=args.pitfall_prob if args.deliberation_mode in ("pitfall_recovery", "scheduled") else 0.0,
                current_loss=ema_loss,
            )

            # Update dynamic learning rate while preserving per-group LR ratios (lr_1d, lr_vocab)
            lr = compute_curriculum_lr(step, schedule_config)
            if getattr(args, "loss_lr_damping", True) and alpha_gov > 0.0:
                lr = lr * (1.0 - 0.4 * alpha_gov)
            lr_scale = lr / max(schedule_config.max_lr, 1e-12)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * lr_scale

            optimizer.zero_grad(set_to_none=True)

            # Gradient accumulation loop (every micro-batch preserves Strategy-A prompt/target/feedback parity)
            for micro in range(args.grad_accum):
                sync_ctx = (
                    train_module.no_sync()
                    if (is_distributed and micro < args.grad_accum - 1)
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    prompts, targets, feedback, action_labels, perceptual_inputs, batch_phase, _ = streamer.get_elastic_batch(
                        step=step,
                        device=device,
                        batch_size=args.batch_size,
                        total_window=window_budget,
                        tail_probability=args.tail_prob,
                        buckets=static_buckets,
                    )
                    micro_tokens = prompts.numel() + targets.numel()
                    compute_action = batch_phase in (
                        CurriculumPhase.PHASE_3_WORKFLOWS,
                        CurriculumPhase.PHASE_4_CONSOLIDATION,
                    )
                    with torch.amp.autocast("cuda", enabled=use_cuda, dtype=amp_dtype):
                        ce_loss, logits, action_logits = train_module(
                            prompts,
                            targets=targets,
                            feedback_tokens=feedback,
                            sampling_p=cur_sampling_p,
                            pitfall_prob=cur_pitfall_prob,
                            pad_id=pad_id,
                            perceptual_inputs=perceptual_inputs,
                            compute_action=compute_action,
                            persona_id=args.persona_id,
                        )
                        if compute_action and action_logits is not None:
                            act_loss = action_loss_fn(action_logits, action_labels)
                        else:
                            act_loss = torch.tensor(0.0, device=device)

                        total_batch_loss = (ce_loss + 0.1 * act_loss) / args.grad_accum
                    scaler.scale(total_batch_loss).backward()

                    accum_loss += ce_loss.item() / args.grad_accum
                    accum_act_loss += act_loss.item() / args.grad_accum
                    tokens_processed += micro_tokens

            # Clip gradients & optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            steps_since_log += 1

            # Periodic Held-Out Evaluation
            if args.eval_interval and step % args.eval_interval == 0 and fixed_eval_batches:
                raw_model.eval()
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=use_cuda, dtype=amp_dtype):
                        v_losses = []
                        for ev_p, ev_t, ev_fb in fixed_eval_batches:
                            ev_out, _ = raw_model(
                                ev_p,
                                targets=ev_t,
                                feedback_tokens=ev_fb,
                                pad_id=pad_id,
                                persona_id=args.persona_id,
                            )
                            v_losses.append(float(ev_out.item()))
                        cur_eval_loss = sum(v_losses) / len(v_losses)
                if is_distributed:
                    ev_t = torch.tensor([cur_eval_loss], dtype=torch.float64, device=device)
                    dist.all_reduce(ev_t, op=dist.ReduceOp.SUM)
                    cur_eval_loss = float(ev_t.item()) / world_size
                raw_model.train()
                latest_eval_loss = cur_eval_loss
                if cur_eval_loss < best_eval_loss:
                    best_eval_loss = cur_eval_loss

            # Periodic Logging
            if step % args.log_interval == 0:
                elapsed = time.time() - start_time
                avg_loss_local = accum_loss / max(1, steps_since_log)
                avg_act_local = accum_act_loss / max(1, steps_since_log)
                global_tok = float(tokens_processed)
                if is_distributed:
                    stats = torch.tensor([avg_loss_local, avg_act_local, global_tok], dtype=torch.float64, device=device)
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    avg_loss = float(stats[0].item()) / world_size
                    avg_act_loss = float(stats[1].item()) / world_size
                    global_tok = float(stats[2].item())
                else:
                    avg_loss = avg_loss_local
                    avg_act_loss = avg_act_local

                last_avg_loss = avg_loss
                if ema_loss is None:
                    ema_loss = avg_loss
                else:
                    ema_loss = 0.9 * ema_loss + 0.1 * avg_loss

                if is_main:
                    tok_per_sec = global_tok / max(1.0, elapsed)
                    vram_mb = torch.cuda.memory_reserved(device) / (1024**2) if use_cuda else 0.0
                    act_str = f" | ActLoss: {avg_act_loss:.4f}" if current_phase >= CurriculumPhase.PHASE_3_WORKFLOWS else ""
                    eval_str = f" | Eval: {latest_eval_loss:.4f} (Best: {best_eval_loss:.4f})" if latest_eval_loss is not None else ""
                    gov_str = f" (Gov: {alpha_gov:.2f})" if getattr(args, "loss_window_governor", True) and alpha_gov > 0.01 else ""
                    samp_str = f" | Samp_p: {cur_sampling_p:.2f}{gov_str}"
                    print(
                        f"[Step {step:6d}/{max_steps}] Phase: {int(current_phase)} ({PHASE_NAMES[current_phase][:16]}...) | "
                        f"Loss: {avg_loss:.4f} (EMA: {ema_loss:.4f}){eval_str}{samp_str}{act_str} | LR: {lr:.2e} | "
                        f"Tok/s: {tok_per_sec:6.1f} | VRAM: {vram_mb:5.0f}MB"
                    )
                accum_loss = 0.0
                accum_act_loss = 0.0
                steps_since_log = 0

            # Periodic Checkpointing & Cloud Sync (rank 0 only)
            need_local_save = (step % args.save_interval == 0 and step > start_step)
            need_cloud_sync = (args.cloud_sync and step > start_step and (step % args.cloud_sync_interval == 0))
            if (need_local_save or need_cloud_sync) and is_main:
                cur_loss = (accum_loss / steps_since_log) if steps_since_log > 0 else (ema_loss or last_avg_loss)
                best_loss = _save_and_push_checkpoints(
                    checkpoint_dir=args.checkpoint_dir,
                    model_preset=args.model_preset,
                    raw_model=raw_model,
                    raw_action_head=raw_action_head,
                    config=config,
                    step=step,
                    loss_val=cur_loss,
                    best_loss=best_loss,
                    need_local_save=need_local_save,
                    need_cloud_sync=need_cloud_sync,
                    eval_loss=latest_eval_loss,
                    final_save=False,
                )

        # Final Checkpoint
        final_loss_local = (accum_loss / steps_since_log) if steps_since_log > 0 else (ema_loss or last_avg_loss)
        if is_distributed:
            loss_t = torch.tensor([final_loss_local], dtype=torch.float64, device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            final_loss = float(loss_t.item()) / world_size
        else:
            final_loss = final_loss_local

        if is_main:
            best_loss = _save_and_push_checkpoints(
                checkpoint_dir=args.checkpoint_dir,
                model_preset=args.model_preset,
                raw_model=raw_model,
                raw_action_head=raw_action_head,
                config=config,
                step=step,
                loss_val=final_loss,
                best_loss=best_loss,
                need_local_save=False,
                need_cloud_sync=args.cloud_sync,
                eval_loss=latest_eval_loss,
                final_save=True,
            )
            final_ckpt = os.path.join(args.checkpoint_dir, f"{args.model_preset}_graded_final.pt")
            print(f"\n✓ Graded training completed {step} steps! Final checkpoint: {final_ckpt}")

            if args.cloud_sync:
                print(f"[cloud_sync] Flushing final {args.model_preset} checkpoints to cloud hub...")
                wait_for_in_flight_uploads(timeout=180.0)

        if is_distributed:
            dist.barrier()
    finally:
        streamer.close()


if __name__ == "__main__":
    main()
