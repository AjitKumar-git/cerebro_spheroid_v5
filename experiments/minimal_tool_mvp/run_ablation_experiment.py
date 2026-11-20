"""
Master Experiment Runner: Minimal Tool MVP & SCMER Controlled Ablation
=====================================================================
Executes the full build sequence:
1. Generate synthetic traces across 3 cases (Case 1, Case 2, Case 3).
2. Evaluate 3 systems:
   - System A: Direct-answer baseline
   - System B: Tool-enabled training WITHOUT SCMER
   - System C: Tool-enabled training WITH SCMER
3. Report all required metrics in an audited comparison table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from experiments.minimal_tool_mvp.dataset_generator import TraceDatasetGenerator
from experiments.minimal_tool_mvp.evaluate import EvaluationReport, evaluate_system
from experiments.minimal_tool_mvp.model import MinimalModelConfig, MinimalToolModel
from experiments.minimal_tool_mvp.tokenizer import MinimalTokenizer
from experiments.minimal_tool_mvp.train import TraceDataset, train_model


def run_experiment(
    num_train: int = 2400,
    num_test: int = 600,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 3e-3,
    scmer_lambda: float = 0.005,
    output_json: str = "experiments/minimal_tool_mvp/mvp_results.json",
    seed: int = 42,
) -> Dict[str, EvaluationReport]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting Minimal Tool MVP Experiment ===")
    print(f"Device: {device} | Seed: {seed} | Epochs: {epochs} | Batch size: {batch_size}")

    # 1. Initialize Tokenizer & Datasets
    tokenizer = MinimalTokenizer()
    gen = TraceDatasetGenerator(seed=seed)

    print(f"\n[1/4] Generating {num_train} train and {num_test} held-out test traces...")
    train_traces = gen.generate_dataset(num_train)
    test_traces = gen.generate_dataset(num_test)

    train_ds = TraceDataset(train_traces, tokenizer)
    cfg = MinimalModelConfig(vocab_size=tokenizer.vocab_size)

    # 2. System A: Direct-Answer Baseline (no tool access)
    print("\n[2/4] Training System A: Direct-Answer Baseline...", flush=True)
    torch.manual_seed(seed)
    model_baseline = MinimalToolModel(cfg)
    train_model(
        model=model_baseline,
        train_dataset=train_ds,
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        scmer_enabled=False,
        device=device,
        verbose=True,
    )
    report_baseline = evaluate_system(
        system_name="Direct-Answer Baseline",
        model=model_baseline,
        test_traces=test_traces,
        tokenizer=tokenizer,
        force_direct_baseline=True,
        device=device,
    )

    # 3. System B: Tool-Enabled Training WITHOUT SCMER
    print("\n[3/4] Training System B: Tool-Enabled (SCMER Disabled)...", flush=True)
    torch.manual_seed(seed)
    model_no_scmer = MinimalToolModel(cfg)
    train_model(
        model=model_no_scmer,
        train_dataset=train_ds,
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        scmer_enabled=False,
        device=device,
        verbose=True,
    )
    report_no_scmer = evaluate_system(
        system_name="Tool-Enabled (No SCMER)",
        model=model_no_scmer,
        test_traces=test_traces,
        tokenizer=tokenizer,
        force_direct_baseline=False,
        device=device,
    )

    # 4. System C: Tool-Enabled Training WITH SCMER
    print("\n[4/4] Training System C: Tool-Enabled (SCMER Enabled, lambda=0.005)...", flush=True)
    torch.manual_seed(seed)
    model_with_scmer = MinimalToolModel(cfg)
    train_model(
        model=model_with_scmer,
        train_dataset=train_ds,
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        scmer_enabled=True,
        scmer_lambda=scmer_lambda,
        device=device,
        verbose=True,
    )
    report_with_scmer = evaluate_system(
        system_name="Tool-Enabled (With SCMER)",
        model=model_with_scmer,
        test_traces=test_traces,
        tokenizer=tokenizer,
        force_direct_baseline=False,
        device=device,
    )

    results = {
        "Direct-Answer Baseline": report_baseline,
        "Tool-Enabled (No SCMER)": report_no_scmer,
        "Tool-Enabled (With SCMER)": report_with_scmer,
    }

    # Print Formatted Markdown Comparison Table
    print("\n" + "=" * 90)
    print("                      MINIMAL TOOL MVP ABLATION REPORT")
    print("=" * 90)
    header = (
        f"| {'Metric':<35} | {'Baseline (No Tool)':<20} | "
        f"{'Tool (No SCMER)':<18} | {'Tool (With SCMER)':<18} |"
    )
    separator = f"|{'-'*37}|{'-'*22}|{'-'*20}|{'-'*20}|"
    print(header)
    print(separator)

    metrics_keys = [
        ("Correct Tool-Choice Rate (%)", "correct_tool_choice_rate"),
        ("Unnecessary-Call Rate (%)", "unnecessary_call_rate"),
        ("Missed-Call Rate (%)", "missed_call_rate"),
        ("Tool Schema Validity (%)", "tool_call_schema_validity"),
        ("Tool Execution Success (%)", "tool_execution_success"),
        ("Accuracy Post Tool-Result (%)", "accuracy_after_tool_results"),
        ("Direct-Answer Accuracy (%)", "direct_answer_accuracy"),
        ("Overall Final Accuracy (%)", "overall_accuracy"),
    ]

    for label, key in metrics_keys:
        v_base = getattr(report_baseline, key) * 100
        v_no = getattr(report_no_scmer, key) * 100
        v_scm = getattr(report_with_scmer, key) * 100
        print(f"| {label:<35} | {v_base:>18.2f}% | {v_no:>16.2f}% | {v_scm:>16.2f}% |")
    print("=" * 90 + "\n")

    # Save to JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)
    print(f"Saved audit results to {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Minimal Tool MVP ablation experiment.")
    parser.add_argument("--num_train", type=int, default=2400)
    parser.add_argument("--num_test", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--scmer_lambda", type=float, default=0.005)
    parser.add_argument("--output_json", type=str, default="experiments/minimal_tool_mvp/mvp_results.json")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_experiment(
        num_train=args.num_train,
        num_test=args.num_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        scmer_lambda=args.scmer_lambda,
        output_json=args.output_json,
        seed=args.seed,
    )
