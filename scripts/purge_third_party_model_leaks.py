#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Purging & Decontamination CLI
=====================================
Purges third-party model leaks (OpenAI, ChatGPT, Claude, Anthropic, DeepSeek,
Qwen, Baichuan, ChatGLM, etc.) across SFT, reasoning, and pretraining datasets.

Usage:
  python3 scripts/purge_third_party_model_leaks.py --target data/frontier_100k/frontier_100k_spheroid_format.jsonl
  python3 scripts/purge_third_party_model_leaks.py --all
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.identity_sanitizer import (
    is_identity_contaminated,
    sanitize_conversation_turn,
    purge_dataset,
)


def purge_jsonl_file(file_path: str, inplace: bool = True) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        print(f"⚠️ File does not exist: {file_path}")
        return {}

    print(f"\n🧹 [Sanitizer] Scanning & purging third-party leaks in: {file_path}")

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    records.append(json.loads(line_str))
                except Exception:
                    pass

    total_in = len(records)
    if total_in == 0:
        print("  Empty file, skipping.")
        return {}

    # Detect prompt/response keys
    first = records[0]
    prompt_key = "prompt" if "prompt" in first else "formatted_input"
    text_key = "formatted_output" if "formatted_output" in first else ("response" if "response" in first else "answer")

    cleaned_records, stats = purge_dataset(
        records,
        text_key=text_key,
        prompt_key=prompt_key,
        allow_adversarial=True
    )

    out_path = file_path if inplace else file_path.replace(".jsonl", "_purged.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in cleaned_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  • Scanned:  {stats['total_in']:,} records")
    print(f"  • Retained: {stats['retained']:,} records ({stats['retained']/max(1, stats['total_in'])*100:.2f}%)")
    print(f"  • Cleaned:  {stats['sanitized']:,} preambles sanitized")
    print(f"  • Purged:   {stats['purged_identity_leak']:,} poisoned third-party identity declarations dropped")
    print(f"  💾 Saved cleanly to: {out_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Purge third-party model references from datasets")
    parser.add_argument("--target", type=str, help="Specific JSONL file to decontaminate")
    parser.add_argument("--all", action="store_true", help="Scan and decontaminate all core datasets in data/")
    parser.add_argument("--no-inplace", action="store_true", help="Save to _purged.jsonl instead of in-place")
    args = parser.parse_args()

    inplace = not args.no_inplace

    if args.target:
        purge_jsonl_file(args.target, inplace=inplace)
    elif args.all:
        scan_targets = [
            "data/doty_identity/doty_identity_spheroid.jsonl",
            "data/doty_identity/doty_identity_sharegpt.jsonl",
            "data/anti_sycophancy_truth_dataset.jsonl",
            "data/cerebro_cognitive_8k/cerebro_cognitive_10k.jsonl",
            "data/chat_thinking_sft_15k.jsonl",
            "data/diverse_sft_corpus_25k.jsonl",
            "data/gemini_golden_reasoning_sft.jsonl",
            "data/frontier_100k/frontier_100k_spheroid_format.jsonl",
            "data/frontier_100k/frontier_100k_industry_sharegpt.jsonl",
        ]
        total_purged = 0
        total_sanitized = 0
        total_scanned = 0

        for path in scan_targets:
            full_path = os.path.join(PROJECT_ROOT, path)
            if os.path.exists(full_path):
                s = purge_jsonl_file(full_path, inplace=inplace)
                total_scanned += s.get("total_in", 0)
                total_purged += s.get("purged_identity_leak", 0)
                total_sanitized += s.get("sanitized", 0)

        print("\n" + "=" * 60)
        print(f"🎉 MASTER DECONTAMINATION COMPLETE:")
        print(f"   • Total Records Scanned:   {total_scanned:,}")
        print(f"   • Toxic Leaks Purged:      {total_purged:,}")
        print(f"   • Preambles Sanitized:     {total_sanitized:,}")
        print("=" * 60)
    else:
        print("Please specify --target <file> or --all.")


if __name__ == "__main__":
    main()
