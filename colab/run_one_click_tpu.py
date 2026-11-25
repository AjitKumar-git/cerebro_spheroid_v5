#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 CerebroSpheroid V5: One-Click Cloud TPU Runner
=================================================
Complete, self-contained one-click training script for Google Colab TPU v5e/lite.
Automatically handles:
  1. Cloud TPU device verification (JAX + native TPU v5 lite MXU bfloat16)
  2. Hugging Face Hub authentication (eliminates 429 rate limits)
  3. Bioflo.in cloud synchronization token setup
  4. Repository synchronization to latest origin/main
  5. Checkpoint directory resolution (local scratch storage, zero Google Drive dependency)
  6. Autonomous resumption from highest step milestone or remote latest.pt
  7. Pure 100% Loss-Bearing Contiguous Pretraining:
     (85% FineWeb sample-10BT broad web + 15% Code The Stack v2 / Python code)
     with contiguous unbroken sequence blocks and next-token supervision
  8. Option A lightweight cloud sync (~229 MB weights-only, 2.9s non-blocking upload)
"""

import os
import sys
import json
import time
import subprocess
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def setup_colab_environment():
    """Configures system hugepages, TPU backend, Hugging Face auth, and directories."""
    print("=" * 80)
    print("⚡ CEREBROSPHEROID V5: PURE JAX HIGH-THROUGHPUT CLOUD TPU RUNNER")
    print("=" * 80)

    # 1. Enable Linux Transparent Hugepages
    os.system("sudo sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled' 2>/dev/null || true")

    # 2. Verify JAX and TPU acceleration
    try:
        import jax
        devices = jax.devices()
        dev_kind = devices[0].device_kind if devices else "CPU"
        is_tpu = any("TPU" in d.device_kind.upper() for d in devices)
        print(f"JAX Version     : {jax.__version__}")
        print(f"JAX Backend     : {jax.default_backend().upper()}")
        print(f"Active Devices  : {len(devices)} x {dev_kind}")
        if is_tpu:
            print("Precision       : bfloat16 (Native TPU v5 lite MXU)")
        else:
            print("⚠️ Warning: Running on CPU/GPU. In Colab, select: Runtime -> Change runtime type -> TPU v5e.")
    except Exception as e_jax:
        print(f"⚠️ JAX device detection notice: {e_jax}")

    # 3. Setup Bioflo Cloud Sync Token
    from src.cloud_sync import get_sync_token
    from src.hf_utils import get_hf_token

    bioflo_tok = get_sync_token()
    os.environ["BIOFLO_SYNC_TOKEN"] = bioflo_tok
    os.environ["CEREBRO_SYNC_TOKEN"] = bioflo_tok
    print("🌐 Bioflo Cloud Sync Token configured.")

    # 4. Authenticate Hugging Face Hub (Dynamically fetched from website vault)
    hf_tok = get_hf_token()
    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_tok
        try:
            from huggingface_hub import login
            login(token=hf_tok, add_to_git_credential=False)
            print(f"🔑 Hugging Face Hub dynamically authenticated from website vault ({hf_tok[:4]}...{hf_tok[-4:]}).")
        except Exception as e_login:
            print(f"ℹ️ Hugging Face login notice: {e_login}")
    else:
        print("⚠️ Hugging Face Hub token could not be resolved.")

    # 5. Resolve Local Checkpoint Directory (Zero Google Drive dependency, synced directly to bioflo.in)
    if os.path.exists("/content"):
        ckpt_dir = "/content/checkpoints/v5_small_scratch"
        print(f"📁 Checkpoint Destination: Colab Scratch Storage ({ckpt_dir})")
    else:
        ckpt_dir = os.path.join(REPO_ROOT, "checkpoints", "v5_small_scratch")
        print(f"📁 Checkpoint Destination: Local Directory ({ckpt_dir})")

    os.makedirs(ckpt_dir, exist_ok=True)
    print("=" * 80 + "\n")
    return ckpt_dir, bioflo_tok


def main():
    ckpt_dir, bioflo_tok = setup_colab_environment()

    # Import training runner
    from training.train_v5_tpu_jax import run_training

    # Launch with verified production configuration
    from_scratch = os.environ.get("FROM_SCRATCH", "false").lower() in ("true", "1")
    preset = os.environ.get("PRESET", "v5_compact")
    run_training(
        preset=preset,                         # Model preset (default: v5_compact, 250M Params)
        from_scratch=from_scratch,             # True: Clean start from Step 0; False: Resume
        auto_tune=True,                        # Adopts verified B=32, remat=True (192,360 tok/s on TPU v5 lite)
        force_tune=False,                      # Bypasses micro-benchmarks; adopts verified config
        effective_batch_size=256,              # Effective batch size (gradient accumulation over optimal micro-batch)
        save_interval=100,                     # Back up every 100 steps
        target_tokens=15_700_000_000,          # 15.70 Billion Tokens target budget
        dataset_source=os.environ.get("DATASET_SOURCE", "pretrain"),  # Pure Contiguous Pretraining (SmolLM2 4-stage curriculum: 90/10 -> 75/20/5 -> 70/20/10 -> 58/24/14/4)
        elastic_window=False,                  # Pure contiguous 100% loss-bearing sequences (zero prompt masking)
        thinking_cycles=4,                     # Matches v5_compact preset deliberation depth
        min_context_window=512,                # Stage 0: 512 tokens
        context_stage_steps=100_000,           # Curriculum stages: 512, 1024, 2048, 4096 (100k steps/stage)
        total_window=4096,                     # Max horizon
        sensory_position_steps=4096,           # Learned position step horizon
        max_phase_steps=4096,                  # Learned phase clock horizon
        checkpoint_dir=ckpt_dir,               # Checkpoint destination
        cloud_sync=True,                       # Bioflo.in bi-directional live sync
        cloud_weights_only=True,               # Option A: ~229 MB lightweight fast uploads (2.9s)
        save_optimizer_state=True,             # Preserves Spheroid moments locally on disk
        cloud_token=bioflo_tok,
        resume_target="latest",                # Prioritizes latest checkpoint
        resume_warmup_steps=100,               # Smooth 100-step warmup on resume
        scmer_lambda=0.05,                     # SCMER anti-anisotropy regularizer strength
        scmer_margin=0.12,                     # Cosine margin threshold m (3.04 sigma)
        scmer_subsample=512,                   # Subsampled vocabulary rows per event
        scmer_pair_samples=8192,               # Subsampled unique token pairs
        run_id=os.environ.get("RUN_ID", None), # Explicit run identifier (or auto-generated and persistent)
        scmer_interval=8,                      # SCMER execution interval (every 8 steps)
    )


if __name__ == "__main__":
    main()
