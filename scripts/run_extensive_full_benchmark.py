#!/usr/bin/env python3
"""
🏆 CerebroSpheroid V5: Master Extensive Benchmark Orchestrator
==============================================================
Runs a comprehensive multi-model, multi-suite benchmark campaign comparing:
  1. Native V5 Pre-trained Champion (TPU Step 900,000)
  2. Native V5 SFT Champion (Supervised Fine-Tuned Step 50,000)
  3. Converged SmolLM-V5 (135M Tier Weight Transfer, Step 1,038)
  4. Converged Qwen-V5 (0.5B Tier Weight Transfer, Step 262)

Dual Benchmark Batteries:
  Battery A: Official Academic & Industry Standard Suite (EleutherAI / Lighteval Standard)
             - ARC-Challenge, ARC-Easy, HellaSwag, MMLU, Winogrande, OpenBookQA, GSM8K
  Battery B: Cerebro-1000 Cognitive Architecture Suite
             - Linguistic & Syntax, Factual & Counterfactual, Math & Logic, Code & Algorithms, Needle & Agentic
"""

import os
import sys
import json
import time
import math
import argparse
from typing import Dict, Any, List

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
from scripts.benchmark_industry_standard import (
    run_academic_benchmark_suite,
    SMOLLM2_135M_REF,
    RANDOM_BASELINE,
)
from eval.run_benchmark import evaluate_benchmark


MODELS_REGISTRY = [
    {
        "id": "qwen_0_5b_champion",
        "name": "Harmonized Champion Qwen2.5-V5",
        "path": os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_qwen_0_5b_champion.pt"),
        "tier": "280M Transferred (0.5B Source)",
        "protocol": "stim_act",
    },
    {
        "id": "qwen_0_5b_refined_sft",
        "name": "Refined SFT Qwen2.5-V5 (Step 400)",
        "path": os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_qwen_0_5b_refined_sft.pt"),
        "tier": "280M Transferred (0.5B Source)",
        "protocol": "stim_act",
    },
    {
        "id": "native_v5_sft_master_diverse",
        "name": "Diverse SFT Native V5",
        "path": os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_sft_master_diverse.pt"),
        "tier": "102M Native SFT",
        "protocol": "stim_act",
    },
    {
        "id": "native_v5_sft_best",
        "name": "Native V5 SFT Best (Step 50k)",
        "path": os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_sft_best.pt"),
        "tier": "102M Native SFT",
        "protocol": "stim_act",
    },
    {
        "id": "native_v5_pretrained_900k",
        "name": "Native V5 Pretrained (Step 900k)",
        "path": os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_latest_step900k.pt"),
        "tier": "102M Native TPU",
        "protocol": "raw",
    },
]


def run_extensive_benchmark(
    academic_samples: int = 50,
    cerebro_samples: int = 100,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "eval/results",
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    master_report_path = os.path.join(output_dir, f"extensive_master_benchmark_{timestamp}.json")

    print("\n" + "=" * 105)
    print("🌟 CEREBROSPHEROID V5: EXTENSIVE MULTI-TIER BENCHMARK CAMPAIGN")
    print(f"Device: {device_str} | Academic Samples/Task: {academic_samples} | Cerebro-1000 Samples: {cerebro_samples}")
    print("=" * 105)

    all_academic_results = {}
    all_cerebro_results = {}

    tasks = ["arc_challenge", "arc_easy", "hellaswag", "mmlu", "winogrande", "openbookqa", "gsm8k"]
    benchmark_1000_path = os.path.join(WORKSPACE_ROOT, "eval", "benchmark_1000.jsonl")

    campaign_start_time = time.time()

    for m_info in MODELS_REGISTRY:
        m_id = m_info["id"]
        m_name = m_info["name"]
        m_path = m_info["path"]
        m_protocol = m_info["protocol"]

        if not os.path.exists(m_path):
            print(f"⚠️ Checkpoint not found: {m_path}, skipping.")
            continue

        print("\n" + "#" * 105)
        print(f"🚀 EVALUATING MODEL: {m_name}")
        print(f"Path: {m_path} | Tier: {m_info['tier']}")
        print("#" * 105)

        # ----------------------------------------------------------------------
        # Battery A: Official Academic Suite
        # ----------------------------------------------------------------------
        print(f"\n--- [Battery A: Official Academic & Industry Suite] ---")
        acad_export = os.path.join(output_dir, f"academic_{m_id}_{timestamp}.json")
        try:
            acad_summary, _ = run_academic_benchmark_suite(
                checkpoint_path=m_path,
                tasks=tasks,
                samples_per_task=academic_samples,
                protocol=m_protocol,
                device_str=device_str,
                export_json=acad_export,
            )
            all_academic_results[m_id] = acad_summary
        except Exception as e:
            print(f"❌ Error in academic battery for {m_id}: {e}")
            all_academic_results[m_id] = {}

        # ----------------------------------------------------------------------
        # Battery B: Cerebro-1000 Cognitive Suite
        # ----------------------------------------------------------------------
        print(f"\n--- [Battery B: Cerebro-1000 Cognitive Pillar Suite] ---")
        try:
            cerebro_summary = evaluate_benchmark(
                checkpoint_path=m_path,
                benchmark_path=benchmark_1000_path,
                num_samples=cerebro_samples,
                device_str=device_str,
                output_dir=output_dir,
                verbose=False,
            )
            all_cerebro_results[m_id] = cerebro_summary
        except Exception as e:
            print(f"❌ Error in Cerebro-1000 battery for {m_id}: {e}")
            all_cerebro_results[m_id] = {}

    total_time = time.time() - campaign_start_time

    # ==========================================================================
    # CONSOLIDATED SCORECARDS
    # ==========================================================================
    print("\n" + "=" * 105)
    print("📊 MASTER BENCHMARK SHOWDOWN: OFFICIAL ACADEMIC BENCHMARKS (acc_norm / EM)")
    print("=" * 105)

    header = f"{'Benchmark Task':<16} | {'Random':<7} | {'SmolLM-Ref':<10} | " + " | ".join([f"{m['name'][:18]:<18}" for m in MODELS_REGISTRY if m['id'] in all_academic_results])
    print(header)
    print("-" * len(header))

    task_keys = ["arc_challenge", "arc_easy", "hellaswag", "mmlu", "winogrande", "openbookqa", "gsm8k"]
    model_averages = {m["id"]: [] for m in MODELS_REGISTRY if m["id"] in all_academic_results}

    for t in task_keys:
        rnd = f"{RANDOM_BASELINE.get(t, 0.0):.1f}%"
        ref = f"{SMOLLM2_135M_REF.get(t, 0.0):.1f}%"
        row = f"{t:<16} | {rnd:<7} | {ref:<10} | "
        for m in MODELS_REGISTRY:
            m_id = m["id"]
            if m_id not in all_academic_results:
                continue
            val = all_academic_results[m_id].get(t, {}).get("acc_norm", all_academic_results[m_id].get(t, {}).get("exact_match", 0.0))
            model_averages[m_id].append(val)
            row += f"{val:6.1f}%             | "
        print(row)

    print("-" * len(header))
    comp_row = f"{'COMPOSITE AVERAGE':<16} | {'—':<7} | {'36.3%':<10} | "
    for m in MODELS_REGISTRY:
        m_id = m["id"]
        if m_id not in all_academic_results:
            continue
        vals = model_averages[m_id]
        comp = sum(vals) / len(vals) if vals else 0.0
        comp_row += f"{comp:6.1f}%             | "
    print(comp_row)
    print("=" * len(header))

    # Cerebro-1000 Summary Scorecard
    print("\n" + "=" * 105)
    print("🧠 MASTER BENCHMARK SHOWDOWN: CEREBRO-1000 COGNITIVE SUITE (Regex Accuracy / Perplexity)")
    print("=" * 105)

    c_header = f"{'Cognitive Pillar':<28} | " + " | ".join([f"{m['name'][:16]:<16}" for m in MODELS_REGISTRY if m['id'] in all_cerebro_results])
    print(c_header)
    print("-" * len(c_header))

    pillars = [
        "linguistic_and_syntax",
        "factual_and_counterfactual",
        "mathematical_and_logic",
        "code_and_algorithms",
        "needle_and_agentic",
    ]

    for p in pillars:
        row = f"{p:<28} | "
        for m in MODELS_REGISTRY:
            m_id = m["id"]
            if m_id not in all_cerebro_results:
                continue
            p_data = all_cerebro_results[m_id].get("by_category", {}).get(p, {})
            reg_acc = p_data.get("regex_accuracy", 0.0)
            row += f"{reg_acc:5.1f}%          | "
        print(row)

    print("-" * len(c_header))
    overall_row = f"{'OVERALL ACCURACY / PERPLEX':<28} | "
    for m in MODELS_REGISTRY:
        m_id = m["id"]
        if m_id not in all_cerebro_results:
            continue
        tot = all_cerebro_results[m_id].get("overall", {})
        acc = tot.get("regex_accuracy", 0.0)
        ppl = tot.get("perplexity", 999.0)
        overall_row += f"{acc:4.1f}% (PPL:{ppl:<5.1f})| "
    print(overall_row)
    print("=" * len(c_header))

    print(f"\n⏱️ Total Campaign Evaluation Elapsed: {total_time:.1f}s ({total_time/60:.2f} mins)")

    # Save consolidated master payload
    master_dict = {
        "timestamp": timestamp,
        "academic_samples_per_task": academic_samples,
        "cerebro_1000_samples": cerebro_samples,
        "total_elapsed_seconds": total_time,
        "academic_results": all_academic_results,
        "cerebro_results": all_cerebro_results,
    }
    with open(master_report_path, "w") as f:
        json.dump(master_dict, f, indent=2)

    print(f"💾 Master Benchmark Report Saved to: {master_report_path}")
    return master_report_path


def main():
    parser = argparse.ArgumentParser(description="Master Extensive Benchmark Orchestrator")
    parser.add_argument("--academic-samples", type=int, default=50, help="Samples per academic task")
    parser.add_argument("--cerebro-samples", type=int, default=100, help="Samples from Cerebro-1000 benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output-dir", type=str, default="eval/results", help="Output directory")

    args = parser.parse_args()
    run_extensive_benchmark(
        academic_samples=args.academic_samples,
        cerebro_samples=args.cerebro_samples,
        device_str=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
