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
import math
import os
import sys
import time
from typing import Optional

# Set PyTorch memory optimization flag before torch initialization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def parse_args():
    parser = argparse.ArgumentParser(description="CerebroSpheroid V5 Graded Curriculum Training")
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--test-steps", type=int, default=None, help="If set, stops after test-steps (for local dry run)")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--max-lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between log lines")
    parser.add_argument("--save-interval", type=int, default=1000, help="Steps between checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/graded_curriculum", help="Checkpoint directory")
    parser.add_argument("--cloud-sync", action="store_true", help="Enable bi-directional checkpoint sync with bioflo.in")
    parser.add_argument("--cloud-sync-interval", type=int, default=100, help="Steps between background cloud sync uploads")
    parser.add_argument("--offline", action="store_true", help="Force offline mode using local datasets & cache")
    parser.add_argument("--elastic-window", action=argparse.BooleanOptionalAction, default=True, help="Enable V4-style elastic complementary sliding window")
    parser.add_argument("--total-window", type=int, default=1024, help="Total sequence budget (prompt + response) for elastic window")
    parser.add_argument("--tail-prob", type=float, default=0.25, help="V4 tail probability for EOS representation in window sampling")
    parser.add_argument("--model-preset", type=str, default="v5_small", choices=["v5_small", "v5_compact", "v5_base"], help="Model preset architecture tier (v5_small: 101M, v5_compact: 244M, v5_base: 617M)")
    parser.add_argument("--mix", type=str, default="auto", choices=["auto", "foundation", "reasoning", "code", "agentic", "consolidation"], help="Curriculum data mixture selection")
    parser.add_argument("--optimizer", type=str, default="spheroid", choices=["spheroid", "adafactor", "adamw"], help="Optimizer to use (default: custom SpheroidOptimizer with Polar Newton-Schulz & spectral clamp)")
    parser.add_argument("--from-scratch", "--from_scratch", dest="from_scratch", action="store_true", default=False, help="Start clean from Step 0 without restoring local/cloud checkpoints")
    return parser.parse_args()


def load_local_tokenizer(offline: bool = True):
    """Loads modern 32,000-vocabulary LLaMA tokenizer from bundled data/tokenizer directory."""
    from src.corpus import build_tokenizer
    return build_tokenizer("llama")


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set offline flags if requested
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda") and torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and amp_dtype == torch.float16))

    print(f"=== CerebroSpheroid V5 Graded Curriculum Engine ===")
    print(f"Device: {device} | Total Target Steps: {args.total_steps}")
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        prec_label = "bfloat16 (Native Tensor Core)" if amp_dtype == torch.bfloat16 else "float16 (AMP + GradScaler)"
        print(f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM) | Precision: {prec_label}")
    else:
        print(f"Precision: float32 (CPU)")

    # 1. Initialize Tokenizer & Adjust Checkpoint Dir
    tokenizer = load_local_tokenizer(offline=args.offline)
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_id is None:
        pad_id = 0

    if args.checkpoint_dir == "checkpoints/graded_curriculum":
        args.checkpoint_dir = f"checkpoints/{args.model_preset}"
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1b. Checkpoint Discovery & Cloud Sync Pull (before model init to match checkpoint topology)
    start_step = 0
    best_loss = float("inf")
    local_latest = os.path.join(args.checkpoint_dir, f"{args.model_preset}_latest.pt")
    local_best = os.path.join(args.checkpoint_dir, f"{args.model_preset}_best.pt")
    tpu_latest = os.path.join(args.checkpoint_dir, "tpu_v5e_1_latest.pt")
    legacy_latest = os.path.join(args.checkpoint_dir, "latest.pt")

    if args.cloud_sync and not args.from_scratch:
        print(f"[cloud_sync] Checking remote manifest on bioflo.in for {args.model_preset}...")
        notify_training_start(step=start_step, model=args.model_preset)
        manifest = check_remote_manifest(model=args.model_preset)
        if manifest.get("has_latest", False):
            pulled = pull_checkpoint(local_latest, target_type="latest", model=args.model_preset)
            if pulled:
                print(f"✓ Restored {args.model_preset} latest checkpoint from cloud hub!")

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
                use_sparse_mot=True,
                use_energy_regularization=True,
            )
        except Exception as e:
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

    model = CerebroSpheroidV5(config).to(device)
    action_head = ToolActionHead(surface_dim=config.surface_dim, num_actions=len(ActionType)).to(device)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in action_head.parameters())
    print(f"Model Preset: {args.model_preset} ({config.num_shells} shells, {config.surface_dim} surface dim, tied={config.use_tied_embeddings})")
    print(f"Total Parameters (Model + ActionHead): {total_params:,} (~{total_params/1e6:.1f}M)")

    if ckpt is not None and load_cand is not None:
        try:
            load_v5_checkpoint_into_torch(model, ckpt, verbose=True)
            act_state = ckpt.get("action_head_state", ckpt.get("action_head_state_dict"))
            if isinstance(act_state, dict):
                action_head.load_state_dict(act_state, strict=False)
            start_step = int(ckpt.get("step", 0) or 0)
            ckpt_loss = ckpt.get("loss", None)
            best_loss = float(ckpt_loss) if ckpt_loss is not None else float("inf")
            print(f"✓ Resumed from {os.path.basename(load_cand)} at step {start_step:,} (loss: {best_loss:.4f})")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint from {load_cand}: {e}")
        finally:
            del ckpt
    elif args.from_scratch:
        print("🌱 [--from-scratch] Starting clean training from Step 0.")

    # 2. Setup Memory-Saving Curriculum Streamer
    schedule_config = CurriculumScheduleConfig(
        total_steps=args.total_steps,
        phase_1_end=50_000,
        phase_2_end=500_000,
        phase_3_end=900_000,
        warmup_steps=2_000,
        max_lr=args.max_lr,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    streamer = GradedCurriculumStreamer(
        tokenizer=tokenizer,
        config=schedule_config,
        mix=args.mix,
        search_dirs=["data"],
    )

    # 3. Setup Optimizer (Default: Custom SpheroidOptimizer with Polar Newton-Schulz)
    all_params = list(model.parameters()) + list(action_head.parameters())
    if args.optimizer == "spheroid":
        from src.spheroid_optimizer import build_spheroid_optimizer
        combined_module = nn.ModuleDict({"model": model, "action_head": action_head})
        optimizer = build_spheroid_optimizer(
            model=combined_module,
            lr=schedule_config.max_lr,
            lr_1d=0.2 * schedule_config.max_lr,
            lr_vocab=min(schedule_config.max_lr, 0.0006),
            spectral_clamp=1.0,
        )
        print(f"[Optimizer] Configured SpheroidOptimizer (Polar Newton-Schulz NS-5 + Persistent Spectral Clamp)")
    elif args.optimizer == "adafactor":
        optimizer = Adafactor(
            all_params,
            lr=schedule_config.max_lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
        print(f"[Optimizer] Configured Adafactor Optimizer (~15MB memory footprint)")
    else:
        optimizer = torch.optim.AdamW(all_params, lr=schedule_config.max_lr, weight_decay=0.01)
        print(f"[Optimizer] Configured standard AdamW Optimizer")

    for pg in optimizer.param_groups:
        pg.setdefault("initial_lr", pg["lr"])

    action_loss_fn = nn.CrossEntropyLoss()

    print("\n--- Starting Graded Pre-Training Pipeline ---")
    start_time = time.time()
    accum_loss = 0.0
    accum_act_loss = 0.0
    steps_since_log = 0
    last_avg_loss = best_loss if math.isfinite(best_loss) else 0.0
    tokens_processed = 0
    step = start_step
    max_steps = (start_step + args.test_steps) if args.test_steps else args.total_steps

    if step >= max_steps:
        print(f"✓ Already reached target step ({step:,} >= {max_steps:,}). Nothing to train.")
        return

    current_phase = get_curriculum_phase(step, schedule_config)
    print(f"Entering {PHASE_NAMES[current_phase]} (Step {step:,} -> {max_steps:,})")

    model.train()
    action_head.train()

    while step < max_steps:
        # Check for phase transitions
        new_phase = get_curriculum_phase(step, schedule_config)
        if new_phase != current_phase:
            print(f"\n=======================================================")
            print(f"★ PHASE TRANSITION at Step {step}: Entering {PHASE_NAMES[new_phase]}")
            print(f"=======================================================\n")
            current_phase = new_phase
            # Save checkpoint at phase boundary
            boundary_ckpt = os.path.join(args.checkpoint_dir, f"v5_phase_{int(new_phase)}_boundary_{step}.pt")
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "action_head_state": action_head.state_dict(),
                "config": config,
            }, boundary_ckpt)
            print(f"[Checkpoint Saved] Phase boundary snapshot: {boundary_ckpt}")

        # Update dynamic learning rate while preserving per-group LR ratios (lr_1d, lr_vocab)
        lr = compute_curriculum_lr(step, schedule_config)
        lr_scale = lr / max(schedule_config.max_lr, 1e-12)
        for pg in optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * lr_scale

        optimizer.zero_grad()

        # Gradient accumulation loop
        for micro in range(args.grad_accum):
            if args.elastic_window:
                prompts, targets, feedback, action_labels, perceptual_inputs, batch_phase, b_name = streamer.get_elastic_batch(
                    step=step,
                    device=device,
                    batch_size=args.batch_size,
                    total_window=args.total_window,
                    tail_probability=args.tail_prob,
                )
                with torch.amp.autocast("cuda", enabled=use_cuda, dtype=amp_dtype):
                    ce_loss, logits = model(
                        prompts,
                        targets=targets,
                        feedback_tokens=feedback,
                        pad_id=pad_id,
                        perceptual_inputs=perceptual_inputs,
                    )
                    act_loss = torch.tensor(0.0, device=device)
                    if batch_phase in (CurriculumPhase.PHASE_3_WORKFLOWS, CurriculumPhase.PHASE_4_CONSOLIDATION):
                        cortical_mantle = model.current_spheroid_state[:, -1, :]
                        action_logits, _ = action_head(cortical_mantle)
                        act_loss = action_loss_fn(action_logits, action_labels)

                    total_batch_loss = (ce_loss + 0.1 * act_loss) / args.grad_accum
                scaler.scale(total_batch_loss).backward()

                accum_loss += ce_loss.item() / args.grad_accum
                accum_act_loss += act_loss.item() / args.grad_accum
                tokens_processed += (prompts.numel() + targets.numel())
            else:
                input_ids, targets, action_labels, perceptual_inputs, batch_phase = streamer.get_batch(
                    step=step,
                    device=device,
                    batch_size=args.batch_size,
                )
                with torch.amp.autocast("cuda", enabled=use_cuda, dtype=amp_dtype):
                    ce_loss, logits = model(
                        input_ids,
                        targets=targets,
                        pad_id=pad_id,
                        perceptual_inputs=perceptual_inputs,
                    )
                    act_loss = torch.tensor(0.0, device=device)
                    if batch_phase in (CurriculumPhase.PHASE_3_WORKFLOWS, CurriculumPhase.PHASE_4_CONSOLIDATION):
                        cortical_mantle = model.current_spheroid_state[:, -1, :]
                        action_logits, _ = action_head(cortical_mantle)
                        act_loss = action_loss_fn(action_logits, action_labels)

                    total_batch_loss = (ce_loss + 0.1 * act_loss) / args.grad_accum
                scaler.scale(total_batch_loss).backward()

                accum_loss += ce_loss.item() / args.grad_accum
                accum_act_loss += act_loss.item() / args.grad_accum
                tokens_processed += input_ids.numel()

        # Clip gradients & optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        steps_since_log += 1

        # Periodic Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = tokens_processed / max(1.0, elapsed)
            vram_mb = torch.cuda.memory_reserved(device) / (1024**2) if use_cuda else 0.0

            avg_loss = accum_loss / max(1, steps_since_log)
            avg_act_loss = accum_act_loss / max(1, steps_since_log)
            last_avg_loss = avg_loss
            act_str = f" | ActLoss: {avg_act_loss:.4f}" if current_phase >= CurriculumPhase.PHASE_3_WORKFLOWS else ""
            print(
                f"[Step {step:6d}/{max_steps}] Phase: {int(current_phase)} ({PHASE_NAMES[current_phase][:16]}...) | "
                f"Loss: {avg_loss:.4f}{act_str} | LR: {lr:.2e} | "
                f"Tok/s: {tok_per_sec:6.1f} | VRAM: {vram_mb:5.0f}MB"
            )
            accum_loss = 0.0
            accum_act_loss = 0.0
            steps_since_log = 0

        # Periodic Checkpointing & Cloud Sync
        need_local_save = (step % args.save_interval == 0 and step > start_step)
        need_cloud_sync = (args.cloud_sync and step > start_step and (step % args.cloud_sync_interval == 0))
        if need_local_save or need_cloud_sync:
            cur_loss = (accum_loss / steps_since_log) if steps_since_log > 0 else last_avg_loss
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model_preset}_step_{step}.pt")
            latest_path = os.path.join(args.checkpoint_dir, f"{args.model_preset}_latest.pt")
            best_path = os.path.join(args.checkpoint_dir, f"{args.model_preset}_best.pt")
            legacy_latest = os.path.join(args.checkpoint_dir, "latest.pt")
            legacy_best = os.path.join(args.checkpoint_dir, "best.pt")

            ckpt_payload = {
                "step": step,
                "model_state": model.state_dict(),
                "action_head_state": action_head.state_dict(),
                "config": config,
                "loss": cur_loss,
                "model_preset": args.model_preset,
            }
            if need_local_save:
                torch.save(ckpt_payload, ckpt_path)
            torch.save(ckpt_payload, latest_path)
            torch.save(ckpt_payload, legacy_latest)
            if need_local_save:
                print(f"--> Saved checkpoint: {ckpt_path}")

            is_best = False
            if cur_loss < best_loss:
                best_loss = cur_loss
                torch.save(ckpt_payload, best_path)
                torch.save(ckpt_payload, legacy_best)
                is_best = True
                print(f"🌟 New best loss: {best_loss:.4f} -> saved {best_path}")

            if need_cloud_sync:
                push_checkpoint_async(latest_path, step=step, loss=cur_loss, target_type="latest", model=args.model_preset)
                if is_best:
                    push_checkpoint_async(best_path, step=step, loss=best_loss, target_type="best", model=args.model_preset)

    # Final Checkpoint
    final_loss = (accum_loss / steps_since_log) if steps_since_log > 0 else last_avg_loss
    final_ckpt = os.path.join(args.checkpoint_dir, f"{args.model_preset}_graded_final.pt")
    latest_path = os.path.join(args.checkpoint_dir, f"{args.model_preset}_latest.pt")
    legacy_latest = os.path.join(args.checkpoint_dir, "latest.pt")
    final_payload = {
        "step": step,
        "model_state": model.state_dict(),
        "action_head_state": action_head.state_dict(),
        "config": config,
        "loss": final_loss,
        "model_preset": args.model_preset,
    }
    torch.save(final_payload, final_ckpt)
    torch.save(final_payload, latest_path)
    torch.save(final_payload, legacy_latest)
    print(f"\n✓ Graded training completed {step} steps! Final checkpoint: {final_ckpt}")

    if args.cloud_sync:
        print(f"[cloud_sync] Flushing final {args.model_preset} checkpoints to cloud hub...")
        push_checkpoint_async(latest_path, step=step, loss=final_payload["loss"], target_type="latest", model=args.model_preset)
        wait_for_in_flight_uploads()


if __name__ == "__main__":
    main()
