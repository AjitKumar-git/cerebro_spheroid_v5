"""
Autotune Command-Line Interface.
================================

CLI utility to inspect accelerator hardware, profile available memory,
and test candidate search ladders without running training.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .hardware import get_hardware_info, resolve_jax_compute_dtype
from .search import generate_vram_maximizing_candidates
from .objectives import TuneObjective


def main(args: Optional[list] = None):
    parser = argparse.ArgumentParser(
        prog="autotune",
        description="High-Performance Accelerator & Architecture Auto-Tuning Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Command: info
    info_parser = subparsers.add_parser("info", help="Inspect local accelerator hardware & memory capacity")
    info_parser.add_argument("--json", action="store_true", help="Output hardware profiling as JSON")

    # Command: ladder
    ladder_parser = subparsers.add_parser("ladder", help="Display candidate batch size search ladder for hardware")
    ladder_parser.add_argument("--preset", type=str, default="v5_small", help="Model preset / architecture name")
    ladder_parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    ladder_parser.add_argument("--max-batch-size", type=int, default=None, help="Batch size ceiling")

    parsed = parser.parse_args(args)

    if parsed.command == "info" or parsed.command is None:
        hw = get_hardware_info()
        if getattr(parsed, "json", False):
            print(json.dumps(hw, indent=2))
        else:
            print("=" * 60)
            print("⚡ AUTOTUNE ACCELERATOR HARDWARE PROFILE")
            print("=" * 60)
            print(f"Device Kind        : {hw['device_kind']}")
            print(f"Platform           : {hw['platform']}")
            print(f"Device Count       : {hw['device_count']}")
            print(f"Memory Type        : {hw['memory_type']}")
            print(f"Total Memory       : {hw['total_gb']:.2f} GB ({hw['total_memory_bytes']:,} bytes)")
            print(f"Hardware Signature : {hw['hardware_signature']}")
            _, dtype_str = resolve_jax_compute_dtype("auto")
            print(f"Optimal Precision  : {dtype_str}")
            print("=" * 60)

    elif parsed.command == "ladder":
        hw = get_hardware_info()
        candidates = generate_vram_maximizing_candidates(
            preset=parsed.preset,
            total_gb=hw["total_gb"],
            seq_len=parsed.seq_len,
            max_batch_size=parsed.max_batch_size,
        )
        print("=" * 60)
        print(f"🔍 CANDIDATE SEARCH LADDER: {parsed.preset} ({hw['total_gb']:.1f}GB {hw['memory_type']})")
        print("=" * 60)
        print(f"{'Idx':<4} | {'Batch Size':<12} | {'Remat / Checkpointing':<24}")
        print("-" * 60)
        for idx, c in enumerate(candidates, 1):
            print(f"{idx:<4} | {c['batch_size']:<12} | {str(c['remat']):<24}")
        print("=" * 60)


if __name__ == "__main__":
    main()
