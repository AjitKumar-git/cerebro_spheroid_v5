#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Google Gemma 4 E4B QAT Q4_0 GGUF Model
===============================================
Downloads the official 4-bit Quantization-Aware Trained (QAT) GGUF variant
of Google Gemma 4 E4B (multimodal edge model) into local Hugging Face cache.
"""

import os
import sys
import time
from huggingface_hub import snapshot_download, get_token

REPO_ID = "google/gemma-4-E4B-it-qat-q4_0-gguf"
CACHE_DIR = os.path.expanduser("~/.hf_cache")

def main():
    print("=" * 80)
    print("📥 DOWNLOADING GOOGLE GEMMA 4 E4B (QAT Q4_0 GGUF)")
    print("=" * 80)
    print(f"  Repository : {REPO_ID}")
    print(f"  Cache Dir  : {CACHE_DIR}")
    print("=" * 80 + "\n")

    token = get_token()
    t_start = time.time()

    print("🚀 Starting download via snapshot_download (resumable, multi-threaded)...")
    model_dir = snapshot_download(
        repo_id=REPO_ID,
        cache_dir=CACHE_DIR,
        resume_download=True,
        max_workers=4,
        token=token,
    )

    t_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print("✨ DOWNLOAD COMPLETE")
    print(f"  Local Snapshot Path: {model_dir}")
    print(f"  Time Elapsed       : {t_elapsed / 60:.2f} minutes")
    print("\nFiles downloaded:")
    total_size = 0
    for root, _, files in os.walk(model_dir):
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            total_size += sz
            print(f"  - {f} ({sz / (1024**2):.2f} MB)")
    print(f"\nTotal footprint: {total_size / (1024**3):.2f} GB")
    print("=" * 80)

if __name__ == "__main__":
    main()
