#!/usr/bin/env python3
"""
🧪 CerebroSpheroid V5: Comprehensive Factual & Deliberation Test Battery
========================================================================
Rigorous verification of the calibrated CerebroSpheroid V5 checkpoint across:
  1. Analogy Completion
  2. Antonyms & Lexical Opposites
  3. Irregular Verb Forms
  4. World Capitals & Chemical Symbols
  5. Arithmetic & Numerical Reasoning
  6. Subconscious Deliberation Cycles Sweep (Cycles = 1, 2, 4)
  7. Temperature Scaling Diagnostics (T = 0.01 Greedy vs T = 0.3 vs T = 0.7)
"""

import os
import sys
import time
import argparse
from typing import List, Tuple, Dict, Any

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.corpus import format_stimulus_prompt, clean_motor_response, encode_no_special

# Benchmark battery across 6 factual and cognitive domains
TEST_BATTERY = [
    # A. Analogies
    ("Analogy", "Complete the analogy: hot is to cold as day is to ____.", "night"),
    ("Analogy", "Complete the analogy: cold is to hot as night is to ____.", "day"),
    ("Analogy", "Complete the analogy: bird is to fly as fish is to ____.", "swim"),
    ("Analogy", "Complete the analogy: pen is to write as knife is to ____.", "cut"),
    ("Analogy", "Complete the analogy: hammer is to nail as screwdriver is to ____.", "screw"),
    ("Analogy", "Complete the analogy: author is to book as artist is to ____.", "painting"),
    ("Analogy", "Complete the analogy: sun is to day as moon is to ____.", "night"),
    ("Analogy", "Complete the analogy: dog is to puppy as cat is to ____.", "kitten"),

    # B. Antonyms
    ("Antonym", "What is the exact antonym of the word \"author\"?", "reader"),
    ("Antonym", "What is the exact antonym of the word \"strong\"?", "weak"),
    ("Antonym", "What is the exact antonym of the word \"dark\"?", "light"),
    ("Antonym", "What is the exact antonym of the word \"begin\"?", "end"),
    ("Antonym", "What is the exact antonym of the word \"empty\"?", "full"),
    ("Antonym", "What is the exact antonym of the word \"noisy\"?", "quiet"),
    ("Antonym", "What is the exact antonym of the word \"expand\"?", "contract"),

    # C. Irregular Verbs
    ("Verb", "What is the simple past tense of the irregular verb \"drink\"?", "drank"),
    ("Verb", "What is the simple past tense of the irregular verb \"run\"?", "ran"),
    ("Verb", "What is the simple past tense of the irregular verb \"eat\"?", "ate"),
    ("Verb", "What is the simple past tense of the irregular verb \"grow\"?", "grew"),
    ("Verb", "What is the simple past tense of the irregular verb \"write\"?", "wrote"),
    ("Verb", "What is the simple past tense of the irregular verb \"choose\"?", "chose"),
    ("Verb", "What is the simple past tense of the irregular verb \"swim\"?", "swam"),

    # D. World Capitals & Science
    ("Fact", "What is the capital of France?", "Paris"),
    ("Fact", "What is the capital of Germany?", "Berlin"),
    ("Fact", "What is the capital of Japan?", "Tokyo"),
    ("Fact", "What is the capital of Italy?", "Rome"),
    ("Fact", "What is the capital of the United Kingdom?", "London"),
    ("Fact", "What is the capital of Canada?", "Ottawa"),
    ("Fact", "What is the chemical symbol for water?", "H2O"),
    ("Fact", "What planet is known as the Red Planet?", "Mars"),

    # E. Elementary Arithmetic
    ("Arithmetic", "If John has 3 apples and buys 2 more, how many apples does he have?", "5"),
    ("Arithmetic", "What is 15 + 27?", "42"),
    ("Arithmetic", "What is 8 multiplied by 7?", "56"),
    ("Arithmetic", "What is 100 divided by 4?", "25"),
    ("Arithmetic", "What is 9 squared?", "81"),
    ("Arithmetic", "What is 12 + 13?", "25"),
]


def run_test_suite(
    checkpoint_path: str,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    test_cycles: List[int] = [1, 2, 4],
):
    print("=" * 85)
    print("🔬 CEREBRO-SPHEROID V5: COMPREHENSIVE CALIBRATION & REASONING BATTERY")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Device:     {device_str.upper()}")
    print("=" * 85)

    device = torch.device(device_str)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = infer_v5_config_from_state_dict(raw, default_preset="v5_small")
    model = CerebroSpheroidV5(cfg).to(torch.bfloat16).to(device)

    # Load weights
    sd = raw.get("model_state", raw.get("model_state_dict", raw.get("model", raw)))
    if any(k.startswith(("sensory.", "core.", "motor.")) for k in sd.keys()):
        load_v5_checkpoint_into_torch(model, raw, verbose=False)
    else:
        model.load_state_dict(sd, strict=False)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(WORKSPACE_ROOT, "data", "tokenizer"))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture: {cfg.num_shells} Concentric Shells | Surface Dim: {cfg.surface_dim} | Params: {total_params:,} ({total_params/1e6:.1f}M)")

    # 1. Main Benchmark Battery (Greedy Decoding: T=0.01)
    print("\n" + "=" * 85)
    print("📋 PART 1: COMPREHENSIVE FACTUAL & LOGICAL BATTERY (Greedy Decoding T=0.01, Cycles=2)")
    print("=" * 85)

    category_stats = {}
    total_hits = 0
    t0_suite = time.time()

    for idx, (cat, prompt, target) in enumerate(TEST_BATTERY, 1):
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        category_stats[cat]["total"] += 1

        t0 = time.time()
        with torch.no_grad():
            output_strict = model.generate_text(
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=12,
                temperature=0.01,
                top_k=5,
                device=device,
                strict_act=True,
                thinking_cycles=2,
            ).strip()

            output_raw = model.generate_text(
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=12,
                temperature=0.01,
                top_k=5,
                device=device,
                strict_act=False,
                thinking_cycles=2,
            ).strip()
        lat_ms = (time.time() - t0) * 1000.0

        # Verification: check if target is cleanly matched
        is_hit = (target.lower() in output_strict.lower().split() or output_strict.lower() == target.lower())
        if is_hit:
            category_stats[cat]["correct"] += 1
            total_hits += 1
            status_icon = "✅"
        else:
            status_icon = "❌"

        print(f"[{idx:2d}/{len(TEST_BATTERY)}] {status_icon} [{cat:10s}] Target: [{target:10s}] -> Output: {repr(output_strict):15s} | Raw: {repr(output_raw):15s} ({lat_ms:.0f}ms)")

    total_duration = time.time() - t0_suite
    accuracy_pct = (total_hits / len(TEST_BATTERY)) * 100.0

    print("\n" + "-" * 85)
    print(f"📊 OVERALL BATTERY ACCURACY: {total_hits}/{len(TEST_BATTERY)} ({accuracy_pct:.1f}%) | Total Time: {total_duration:.1f}s")
    print("-" * 85)
    for cat, stats in category_stats.items():
        cat_acc = (stats["correct"] / stats["total"]) * 100.0
        print(f"  • {cat:12s}: {stats['correct']:2d}/{stats['total']:2d} ({cat_acc:5.1f}%)")
    print("-" * 85)

    # 2. Subconscious Deliberation Cycles Sweep
    print("\n" + "=" * 85)
    print("🌀 PART 2: SUBCONSCIOUS DELIBERATION CYCLES DYNAMICS (Cycles = 1, 2, 4)")
    print("=" * 85)
    sample_queries = [
        ("Complete the analogy: hot is to cold as day is to ____.", "night"),
        ("What is the capital of France?", "Paris"),
        ("If John has 3 apples and buys 2 more, how many apples does he have?", "5"),
    ]

    for q, target in sample_queries:
        print(f"\nQuery: \"{q}\" (Target: [{target}])")
        for cycles in test_cycles:
            t0 = time.time()
            with torch.no_grad():
                res = model.generate_text(
                    prompt=q,
                    tokenizer=tokenizer,
                    max_new_tokens=10,
                    temperature=0.01,
                    device=device,
                    strict_act=True,
                    thinking_cycles=cycles,
                ).strip()
            dt_ms = (time.time() - t0) * 1000.0
            match = "✅" if res.lower() == target.lower() else "❌"
            print(f"  Cycles = {cycles:1d} | Output: {repr(res):12s} | Match: {match} | Latency: {dt_ms:.1f}ms")

    # 3. Temperature Scaling Sweep
    print("\n" + "=" * 85)
    print("🌡️ PART 3: TEMPERATURE SCALING DYNAMICS (T = 0.01 vs T = 0.3 vs T = 0.7)")
    print("=" * 85)
    sample_q = "Complete the analogy: sun is to day as moon is to ____."
    print(f"Prompt: \"{sample_q}\" (Target: [night])")
    for temp in [0.01, 0.3, 0.7]:
        with torch.no_grad():
            res = model.generate_text(
                prompt=sample_q,
                tokenizer=tokenizer,
                max_new_tokens=10,
                temperature=temp,
                top_k=20,
                device=device,
                strict_act=True,
                thinking_cycles=2,
            ).strip()
        print(f"  T = {temp:4.2f} | Output: {repr(res)}")

    print("\n" + "=" * 85)
    print("✨ TEST BATTERY COMPLETED SUCCESSFULLY")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test calibrated CerebroSpheroid checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_small_sft_factual_calibrated.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_test_suite(checkpoint_path=args.checkpoint, device_str=args.device)
