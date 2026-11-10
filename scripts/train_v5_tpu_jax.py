#!/usr/bin/env python3
"""
CerebroSpheroid V5: Pure JAX Cloud TPU High-Throughput Pretraining Engine
========================================================================

Designed specifically for Google Cloud TPUs (v2-8, v3-8, v4-8, v5e-1) & Colab TPU:
- Full XLA Ahead-Of-Time compilation via jax.jit.
- Direct hardware Bfloat16 Matrix Multiply Unit (MXU) systolic acceleration.
- Native chunkwise parallel scan for ~50,000 to 120,000 tokens/sec.
- Automatic bi-directional synchronization with https://bioflo.in/api/cerebro/sync.php.
"""

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import jax
import jax.numpy as jnp
from jax import lax

try:
    import optax
except ImportError:
    optax = None

from src.jax.v5_model import (
    ModelConfigV5 as JAXConfigV5,
    PRESETS_V5,
    init_params_v5,
    training_rollout,
)
from src.jax.spheroid_optimizer import (
    init_spheroid_opt_state,
    build_train_step_spheroid,
)
from src.cloud_sync import (
    check_remote_manifest,
    pull_checkpoint,
    push_checkpoint_async,
    wait_for_in_flight_uploads,
)
from src.jax.data_streamer import LinguisticDataStreamer


def parse_args(args_list=None):
    if args_list is None:
        args_list = sys.argv[1:]
    parser = argparse.ArgumentParser(description="CerebroSpheroid V5 Cloud TPU JAX Trainer")
    parser.add_argument("--total-steps", "--steps", "--max-steps", dest="total_steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-device batch size (default: 32 for v5_small, fits comfortably in TPU v5e-1 16GB HBM)")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length (default: 256)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate (default: 5e-4)")
    parser.add_argument("--save-interval", "--save-every", dest="save_interval", type=int, default=500, help="Steps between local checkpoints")
    parser.add_argument("--cloud-sync", action="store_true", default=True, help="Enable live sync with bioflo.in")
    parser.add_argument("--cloud-sync-interval", type=int, default=100, help="Steps between cloud sync uploads")
    parser.add_argument("--checkpoint-dir", "--ckpt-dir", dest="checkpoint_dir", type=str, default="/content/checkpoints" if os.path.exists("/content") else "checkpoints/tpu_jax", help="Checkpoint directory")
    parser.add_argument("--preset", type=str, default="v5_small", choices=["v5_small", "v5_compact", "v5_base", "v5_test"], help="Model preset (default: v5_small)")
    parser.add_argument("--optimizer", type=str, default="spheroid", choices=["spheroid", "adamw"], help="Optimizer choice (default: spheroid)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp32"], help="Matrix execution precision (default: auto, bf16 on TPU)")
    parser.add_argument("--scratch", "--from-scratch", dest="from_scratch", action="store_true", default=False, help="Start training strictly from scratch (step 0, clean initialization)")
    parser.add_argument("--auto-tune", dest="auto_tune", action="store_true", default=True, help="Automatically test environment to discover maximum tok/s (default: True)")
    parser.add_argument("--no-auto-tune", dest="auto_tune", action="store_false", help="Disable automatic environment tuning")
    parser.add_argument("--remat", type=str, default="auto", help="Activation checkpointing (auto, true, false)")
    return parser.parse_args(args_list)


def run_training(**kwargs):
    """Programmatic in-process entrypoint for Google Colab TPU notebooks (avoids TPU process lock conflicts)."""
    args = parse_args([])
    for k, v in kwargs.items():
        if hasattr(args, k):
            setattr(args, k, v)
        elif k in ("ckpt_dir", "drive_dir"):
            setattr(args, "checkpoint_dir", v)
        elif k in ("max_steps", "steps"):
            setattr(args, "total_steps", v)
        elif k == "save_every":
            setattr(args, "save_interval", v)
        elif k in ("from_scratch", "scratch", "clean"):
            setattr(args, "from_scratch", bool(v))
        elif k in ("auto_tune", "tune"):
            setattr(args, "auto_tune", bool(v))
        elif k == "remat":
            setattr(args, "remat", v)
    return main(args)


def main(args=None):
    if args is None:
        args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 75)
    print("⚡ CEREBROSPHEROID V5: PURE JAX CLOUD TPU ENGINE")
    print("=" * 75)
    print(f"JAX Version     : {jax.__version__}")
    print(f"JAX Backend     : {jax.default_backend().upper()}")
    devices = jax.devices()
    print(f"Active Devices  : {len(devices)} x {devices[0].device_kind}")
    is_tpu = any("TPU" in d.device_kind.upper() for d in devices)

    if is_tpu:
        dev_kind = devices[0].device_kind.lower()
        if "v5" in dev_kind or "lite" in dev_kind:
            print("🚀 Google Colab TPU v5e-1 Native BF16 MXU ACTIVE! (~50,000 to 120,000 tok/s)")
        else:
            print(f"🚀 Native TPU Bfloat16 Systolic Matrix Units (MXU) ACTIVE! ({devices[0].device_kind})")
    else:
        print("ℹ️ Running on GPU/CPU fallback.")

    # 0. Automatic Environment Benchmark & Tuning (Maximum tok/s Discovery)
    remat_choice = True
    if getattr(args, "auto_tune", True):
        from src.jax.auto_tuner import auto_tune_environment
        best_cfg = auto_tune_environment(
            preset=args.preset,
            seq_len=args.seq_len,
            dtype=args.dtype,
            verbose=True,
        )
        if best_cfg.get("batch_size"):
            print(f"⚡ Applying Auto-Tuned Hyperparameters: batch_size={best_cfg['batch_size']}, remat={best_cfg.get('remat', True)}")
            args.batch_size = best_cfg["batch_size"]
            remat_choice = best_cfg.get("remat", True)
    elif hasattr(args, "remat") and args.remat != "auto":
        remat_choice = str(args.remat).lower() in ("true", "1", "yes")

    # 1. Config & Parameter Initialization
    cfg_dict = PRESETS_V5.get(args.preset, PRESETS_V5["v5_base"]).copy()
    cfg_dict["max_phase_steps"] = max(1024, args.seq_len * 2)
    config = JAXConfigV5(**cfg_dict)

    # Precision setup (TPU v5e-1 runs native bfloat16)
    if args.dtype == "auto":
        target_dtype = jnp.bfloat16 if is_tpu else jnp.float32
        dtype_desc = "bfloat16 (Native TPU v5e-1 MXU)" if is_tpu else "float32"
    elif args.dtype == "bf16":
        target_dtype = jnp.bfloat16
        dtype_desc = "bfloat16"
    else:
        target_dtype = jnp.float32
        dtype_desc = "float32"

    print(f"\nModel Config    : {args.preset} | Shells: {config.num_shells} | Surface Dim: {config.surface_dim} | Radial Dim: {config.radial_dim}")
    print(f"Precision       : {dtype_desc} | Remat: {remat_choice}")
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    params, buffers = init_params_v5(init_key, config)

    # Cast parameters to target execution dtype
    if target_dtype != jnp.float32:
        params = {k: v.astype(target_dtype) if jnp.issubdtype(v.dtype, jnp.floating) else v for k, v in params.items()}

    # Count parameters
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    vocab_params = params["sensory.embedding.weight"].size
    active_compute = total_params - vocab_params
    print(f"Total Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    print(f"Active Compute  : {active_compute:,} (~{active_compute/1e6:.1f}M) ({active_compute/total_params*100:.2f}%)")

    # 2. Setup Optimizer (SpheroidOptimizer vs AdamW)
    print(f"Optimizer       : {args.optimizer.upper()} ({'Polar Newton-Schulz NS-5 + Spectral Clamp' if args.optimizer == 'spheroid' else 'Optax AdamW'})")
    if args.optimizer == "spheroid":
        opt_state = init_spheroid_opt_state(params)
        train_step_fn = build_train_step_spheroid(config, buffers, pad_id=0, weight_decay=1e-5, remat=remat_choice)
    else:
        learning_rate_sched = optax.cosine_decay_schedule(
            init_value=args.lr,
            decay_steps=args.total_steps,
            alpha=0.03,
        )
        if optax is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=learning_rate_sched, b1=0.9, b2=0.95, weight_decay=1e-4),
            )
            opt_state = optimizer.init(params)
        else:
            raise ImportError("optax is required for AdamW TPU training. Please install via: pip install optax")

        def loss_fn(p, prompt_batch, target_batch):
            return training_rollout(
                params=p,
                prompts=prompt_batch,
                target_len=args.seq_len,
                buffers=buffers,
                config=config,
                targets=target_batch,
                remat=remat_choice,
            )

        @jax.jit
        def train_step_adamw(p, opt_s, prompt_b, target_b, cur_lr):
            loss, grads = jax.value_and_grad(loss_fn)(p, prompt_b, target_b)
            updates, next_opt_s = optimizer.update(grads, opt_s, p)
            next_p = optax.apply_updates(p, updates)
            return next_p, next_opt_s, loss

        train_step_fn = train_step_adamw

    # 3. Cloud Sync & Local Checkpoint Resumption
    start_step = 0
    best_loss = float("inf")
    local_latest = os.path.join(args.checkpoint_dir, f"{args.preset}_latest.pt")
    local_best = os.path.join(args.checkpoint_dir, f"{args.preset}_best.pt")

    if args.from_scratch:
        print(f"\n🌱 Starting [{args.preset}] pretraining strictly from scratch (Step 0)!")
        for stale in (local_latest, local_best):
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                    print(f"🧹 Removed stale checkpoint to enforce fresh start: {os.path.basename(stale)}")
                except Exception:
                    pass
    else:
        if args.cloud_sync:
            print(f"\n[cloud_sync] Checking remote manifest on bioflo.in for {args.preset}...")
            try:
                m_resp = check_remote_manifest(model=args.preset)
                models_info = m_resp.get("models", {})
                model_tier_info = models_info.get(args.preset, {})
                tier_manifest = m_resp.get("manifest", {}).get(f"{args.preset}_latest", {})

                has_tier_latest = model_tier_info.get("has_latest", False) or bool(tier_manifest)
                if has_tier_latest:
                    pulled = pull_checkpoint(local_latest, target_type="latest", model=args.preset)
                    if pulled:
                        print(f"✓ Restored latest {args.preset} checkpoint from cloud hub!")
                else:
                    print(f"ℹ️ No existing cloud checkpoint found for [{args.preset}]. Starting fresh {args.preset} training frontier!")
            except Exception as e:
                print(f"⚠️ Cloud sync notice: {e}")

        # Check local checkpoint
        if os.path.exists(local_latest):
            try:
                import torch
                ckpt = torch.load(local_latest, map_location="cpu", weights_only=False)
                ckpt_preset = ckpt.get("model_preset")
                if ckpt_preset is not None and ckpt_preset != args.preset:
                    print(f"⚠️ Checkpoint preset mismatch: found '{ckpt_preset}', but running '{args.preset}'. Starting clean!")
                    os.remove(local_latest)
                elif "step" in ckpt:
                    start_step = int(ckpt.get("step", 0))
                    best_loss = float(ckpt.get("loss", float("inf")))
                    print(f"✓ Resumed from {os.path.basename(local_latest)} at Step {start_step:,} (loss: {best_loss:.4f})")
                    if "params" in ckpt and isinstance(ckpt["params"], dict):
                        restored_cnt = 0
                        for k, v in ckpt["params"].items():
                            if k in params and params[k].shape == v.shape:
                                params[k] = jnp.asarray(v, dtype=target_dtype)
                                restored_cnt += 1
                        print(f"✓ Restored {restored_cnt}/{len(params)} parameter arrays into JAX engine.")
                else:
                    print(f"⚠️ Incompatible checkpoint structure in {local_latest}. Starting fresh!")
                    os.remove(local_latest)
            except Exception as e:
                print(f"⚠️ Local checkpoint inspection: {e}")

    if start_step >= args.total_steps:
        print(f"✓ Current Step ({start_step:,}) >= Target Steps ({args.total_steps:,}). Target reached!")
        return

    def get_lr(s: int, total_s: int, peak_lr: float, warmup_steps: int = 500) -> float:
        if s <= warmup_steps:
            return float(peak_lr * (float(s) / max(1.0, float(warmup_steps))))
        ratio = min(1.0, float(s - warmup_steps) / max(1.0, float(total_s - warmup_steps)))
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return float(peak_lr * (0.05 + 0.95 * coeff))

    # 4. Warmup JIT Trace & Compile
    print("\n⚡ Compiling XLA computation graph on TPU...")
    dummy_prompts = jnp.zeros((args.batch_size, 32), dtype=jnp.int32)
    dummy_targets = jnp.zeros((args.batch_size, args.seq_len), dtype=jnp.int32)
    t0_compile = time.time()
    params, opt_state, initial_loss = train_step_fn(params, opt_state, dummy_prompts, dummy_targets, args.lr)
    jax.block_until_ready(initial_loss)
    print(f"✓ XLA Compilation complete in {time.time() - t0_compile:.2f}s! Initial Loss: {float(initial_loss):.4f}")

    # 5. Initialize Continuous Linguistic Streamer
    print("\n📦 Initializing background linguistic streaming queue...")
    streamer = LinguisticDataStreamer(
        batch_size=args.batch_size,
        prompt_len=32,
        target_len=args.seq_len,
        vocab_size=config.vocab_size,
        queue_size=8,
    )

    # 6. High-Throughput Training Loop
    print("\n" + "=" * 75)
    print(f"🚀 Training {args.total_steps:,} Steps (Batch: {args.batch_size}, SeqLen: {args.seq_len})")
    print("=" * 75)

    step = start_step
    running_loss = 0.0
    start_time = time.time()
    tokens_per_step = args.batch_size * args.seq_len

    while step < args.total_steps:
        step += 1
        prompt_b, target_b = streamer.get_batch()

        cur_lr = get_lr(step, args.total_steps, args.lr)
        params, opt_state, loss = train_step_fn(params, opt_state, prompt_b, target_b, cur_lr)
        loss_val = float(loss)
        running_loss += loss_val

        if step % 10 == 0:
            jax.block_until_ready(loss)
            elapsed = time.time() - start_time
            steps_completed = step - start_step
            tok_per_sec = (steps_completed * tokens_per_step) / max(elapsed, 1e-4)
            avg_loss = running_loss / 10.0
            running_loss = 0.0
            print(f"[Step {step:6d}/{args.total_steps}] Loss: {avg_loss:.4f} | Tok/s: {tok_per_sec:,.0f} | Elapsed: {elapsed:.1f}s")

        if step % args.save_interval == 0:
            print(f"--> [Checkpoint] Saved Step {step:,} (Loss: {loss_val:.4f})")
            params_numpy = {k: np.asarray(v) for k, v in params.items()}
            save_payload = {
                "step": step,
                "loss": loss_val,
                "model_preset": args.preset,
                "timestamp": time.time(),
                "params": params_numpy,
            }
            try:
                import torch
                torch.save(save_payload, local_latest)
                if loss_val < best_loss:
                    best_loss = loss_val
                    torch.save(save_payload, local_best)
                    if args.cloud_sync:
                        push_checkpoint_async(local_best, step=step, loss=loss_val, target_type="best", model=args.preset)

                if args.cloud_sync:
                    push_checkpoint_async(local_latest, step=step, loss=loss_val, target_type="latest", model=args.preset)
            except Exception as e:
                print(f"⚠️ Checkpoint save notice: {e}")

    print(f"\n✓ TPU Training completed {step} steps in {time.time() - start_time:.1f}s!")


if __name__ == "__main__":
    main()
