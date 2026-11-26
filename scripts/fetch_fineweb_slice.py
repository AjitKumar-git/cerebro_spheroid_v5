#!/usr/bin/env python3
"""
Stream a slice of HuggingFaceFW/fineweb-edu to CSV for DEE data integrity & outlier profiling.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import time
import argparse
from datasets import load_dataset
from src.hf_utils import get_hf_token

def stream_fineweb_slice(
    target_mb: float = 10.0,
    output_path: str = "data/fineweb_slice_10mb.csv",
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    include_full_text: bool = True,
):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    target_bytes = int(target_mb * 1024 * 1024)
    token = get_hf_token()

    print(f"🌐 Connecting to {dataset_name} ({dataset_config})...")
    ds = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True, token=token)
    it = iter(ds)

    print(f"📥 Streaming to {output_path} (target size: {target_mb:.1f} MB)...")
    t0 = time.time()
    total_bytes = 0
    row_count = 0

    fieldnames = [
        "id",
        "dump",
        "url",
        "language",
        "language_score",
        "token_count",
        "score",
        "int_score",
        "text_len",
    ]
    if include_full_text:
        fieldnames.append("text")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)
        header_len = f.tell()
        total_bytes += header_len

        for sample in it:
            text = sample.get("text", "")
            row = [
                sample.get("id", ""),
                sample.get("dump", ""),
                sample.get("url", ""),
                sample.get("language", ""),
                sample.get("language_score", 0.0),
                sample.get("token_count", 0),
                sample.get("score", 0.0),
                sample.get("int_score", 0),
                len(text),
            ]
            if include_full_text:
                row.append(text)

            writer.writerow(row)
            row_count += 1
            total_bytes = f.tell()

            if row_count % 500 == 0:
                elapsed = time.time() - t0
                mb_so_far = total_bytes / (1024 * 1024)
                print(f"  Processed {row_count} rows ({mb_so_far:.2f} MB) in {elapsed:.1f}s...")

            if total_bytes >= target_bytes:
                break

    elapsed = time.time() - t0
    final_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Finished! Wrote {row_count:,} rows ({final_mb:.2f} MB) in {elapsed:.2f}s to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch FineWeb slice for DEE testing")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu", help="HuggingFace dataset name")
    parser.add_argument("--config", type=str, default="sample-10BT", help="Dataset configuration")
    parser.add_argument("--size-mb", type=float, default=10.0, help="Target file size in MB")
    parser.add_argument("--output", type=str, default="data/fineweb_slice_10mb.csv", help="Output CSV path")
    parser.add_argument("--no-full-text", action="store_true", help="Exclude full text column")
    args = parser.parse_args()

    stream_fineweb_slice(
        target_mb=args.size_mb,
        output_path=args.output,
        dataset_name=args.dataset,
        dataset_config=args.config,
        include_full_text=not args.no_full_text,
    )
