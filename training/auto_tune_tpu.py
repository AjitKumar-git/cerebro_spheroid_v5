#!/usr/bin/env python3
"""
CerebroSpheroid V5: Standalone TPU Auto-Tuner & Hardware Benchmark
===================================================================

Runs micro-benchmarks on the current hardware environment (Colab TPU v5e-1, GPU, or CPU)
to determine the optimal batch size and activation remat strategy for maximum tokens/second.

Usage:
    python scripts/auto_tune_tpu.py --preset v5_small
    python scripts/auto_tune_tpu.py --preset v5_small --launch-training
"""

import argparse
import os
import sys

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.jax.auto_tuner import auto_tune_environment


def parse_args():
    parser = argparse.ArgumentParser(description="CerebroSpheroid V5 TPU Auto-Tuner")
    parser.add_argument("--preset", type=str, default="v5_small", choices=["v5_small", "v5_compact", "v5_base", "v5_test"], help="Model preset (default: v5_small)")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length (default: 256)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp32"], help="Precision (default: auto, bf16 on TPU)")
    parser.add_argument("--force-tune", "--force_tune", dest="force_tune", action="store_true", default=False, help="Force re-benchmark even if cached on bioflo.in")
    parser.add_argument("--launch-training", action="store_true", default=False, help="Launch pretraining with winning configuration immediately")
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Total training steps if launching")
    parser.add_argument("--scratch", "--from-scratch", dest="from_scratch", action="store_true", default=True, help="Start from scratch (step 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    winner = auto_tune_environment(
        preset=args.preset,
        seq_len=args.seq_len,
        dtype=args.dtype,
        force_tune=args.force_tune,
        verbose=True,
    )

    if args.launch_training:
        print(f"\n🚀 Launching CerebroSpheroid V5 training with optimal parameters...")
        from training.train_v5_tpu_jax import run_training
        run_training(
            preset=args.preset,
            batch_size=winner["batch_size"],
            remat=winner.get("remat", True),
            seq_len=args.seq_len,
            dtype=args.dtype,
            total_steps=args.total_steps,
            from_scratch=args.from_scratch,
            auto_tune=False,  # Already tuned!
        )


if __name__ == "__main__":
    main()
