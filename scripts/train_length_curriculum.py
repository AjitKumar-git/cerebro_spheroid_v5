#!/usr/bin/env python3
"""
📈 Multi-Stage Length Curriculum Orchestrator for CerebroSpheroid V5
===================================================================
Executes Step 5 of the architectural and empirical recovery plan:
Trains length curriculum: 256 -> 512 -> 1024 -> 2048 -> 4096 tokens
with explicit passkey/needle, multi-hop retrieval, and lost-in-the-middle examples.

Evaluates empirical passkey retrieval accuracy after each stage.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
from transformers import AutoTokenizer

from src.spheroid_v5 import CerebroSpheroidV5, infer_v5_config_from_state_dict
from scripts.run_refined_sft_pytorch import run_refined_sft
from scripts.eval_passkey_retrieval import evaluate_passkey_matrix

CURRICULUM_STAGES = [
    {
        "stage": 1,
        "name": "Stage 1: 256 Tokens (Short-Context Anchor)",
        "horizon": 256,
        "prompt_len": 256,
        "target_len": 512,
        "eval_lengths": [128, 256],
    },
    {
        "stage": 2,
        "name": "Stage 2: 512 Tokens (Standard CoT Expansion)",
        "horizon": 512,
        "prompt_len": 512,
        "target_len": 512,
        "eval_lengths": [128, 256, 512],
    },
    {
        "stage": 3,
        "name": "Stage 3: 1024 Tokens (Kilobyte Document Reasoning)",
        "horizon": 1024,
        "prompt_len": 1024,
        "target_len": 512,
        "eval_lengths": [128, 256, 512, 1024],
    },
    {
        "stage": 4,
        "name": "Stage 4: 2048 Tokens (Extended Needle Retrieval)",
        "horizon": 2048,
        "prompt_len": 2048,
        "target_len": 512,
        "eval_lengths": [256, 512, 1024, 2048],
    },
    {
        "stage": 5,
        "name": "Stage 5: 4096 Tokens (Full-Context Horizon)",
        "horizon": 4096,
        "prompt_len": 3584,
        "target_len": 512,
        "eval_lengths": [512, 1024, 2048, 4096],
    },
]


def run_length_curriculum(
    initial_checkpoint: str = "checkpoints/v5_passkey_focused_test.pt",
    steps_per_stage: int = 150,
    batch_size: int = 2,
    accum_steps: int = 8,
    lr: float = 4e-5,
    min_lr: float = 4e-6,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "checkpoints/curriculum",
    start_stage: int = 1,
    max_stage: int = 5,
    eval_trials: int = 2,
):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "curriculum_eval_report.json")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(WORKSPACE_ROOT, "data", "tokenizer"))
    device = torch.device(device_str)

    print("\n" + "=" * 80)
    print("🎓 CEREBRO-SPHEROID V5 LENGTH CURRICULUM TRAINING SUITE")
    print(f"   Initial Checkpoint : {initial_checkpoint}")
    print(f"   Stages to Execute  : Stage {start_stage} -> Stage {max_stage}")
    print(f"   Steps per Stage    : {steps_per_stage}")
    print(f"   Effective Batch    : {batch_size * accum_steps}")
    print(f"   Device             : {device}")
    print("=" * 80 + "\n")

    current_checkpoint = initial_checkpoint
    curriculum_history = []

    for stage_info in CURRICULUM_STAGES:
        s_num = stage_info["stage"]
        if s_num < start_stage or s_num > max_stage:
            continue

        s_name = stage_info["name"]
        p_len = stage_info["prompt_len"]
        t_len = stage_info["target_len"]
        eval_lens = stage_info["eval_lengths"]
        stage_ckpt = os.path.join(output_dir, f"v5_curriculum_stage{s_num}_{stage_info['horizon']}.pt")

        print("\n" + "#" * 80)
        print(f"▶️  STARTING {s_name.upper()}")
        print(f"    Horizon: {stage_info['horizon']} (Prompt <= {p_len}, Target <= {t_len})")
        print(f"    Input Checkpoint:  {current_checkpoint}")
        print(f"    Target Checkpoint: {stage_ckpt}")
        print("#" * 80 + "\n")

        # 1. Train Stage SFT
        run_refined_sft(
            base_checkpoint=current_checkpoint,
            output_checkpoint=stage_ckpt,
            optimizer_type="adamw",
            max_steps=steps_per_stage,
            batch_size=batch_size,
            accum_steps=accum_steps,
            lr=lr,
            min_lr=min_lr,
            weight_decay=0.01,
            warmup_steps=min(20, steps_per_stage // 5),
            probe_interval=max(25, steps_per_stage // 2),
            device_str=device_str,
            use_rope=True,
            max_prompt_len=p_len,
            max_target_len=t_len,
        )

        current_checkpoint = stage_ckpt

        # 2. Evaluate Passkey Retrieval on Stage Checkpoint
        print(f"\n🔍 Evaluating Passkey Matrix for Stage {s_num} Checkpoint: {stage_ckpt}...")
        raw = torch.load(stage_ckpt, map_location="cpu", weights_only=False)
        cfg = infer_v5_config_from_state_dict(raw)
        cfg.use_rope = True
        eval_model = CerebroSpheroidV5(cfg)
        model_state = raw.get("model_state", raw.get("state_dict", raw))
        eval_model.load_state_dict(model_state, strict=False)
        eval_model.to(device)
        eval_model.eval()

        eval_res_2d = evaluate_passkey_matrix(
            model=eval_model,
            tokenizer=tokenizer,
            lengths=eval_lens,
            depths=[0.1, 0.5, 0.9],
            trials_per_cell=eval_trials,
            device=device,
            digits=2,
        )
        eval_res_5d = evaluate_passkey_matrix(
            model=eval_model,
            tokenizer=tokenizer,
            lengths=eval_lens,
            depths=[0.1, 0.5, 0.9],
            trials_per_cell=eval_trials,
            device=device,
            digits=5,
        )
        del eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stage_record = {
            "stage": s_num,
            "horizon": stage_info["horizon"],
            "checkpoint": stage_ckpt,
            "eval_lengths": eval_lens,
            "overall_accuracy_2d": eval_res_2d["overall_accuracy"],
            "matrix_2d": eval_res_2d["matrix"],
            "overall_accuracy_5d": eval_res_5d["overall_accuracy"],
            "matrix_5d": eval_res_5d["matrix"],
        }
        curriculum_history.append(stage_record)

        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump({"curriculum_history": curriculum_history}, rf, indent=2)
        print(f"   💾 Saved Curriculum Progress Report -> {report_path}")

    print("\n" + "=" * 80)
    print("🏆 ALL CURRICULUM STAGES COMPLETE!")
    print("   Summary of Stage Retrieval Accuracies:")
    for rec in curriculum_history:
        print(f"   • Stage {rec['stage']} ({rec['horizon']} tokens): 2-digit Acc={rec['overall_accuracy_2d']:.1f}%, 5-digit Acc={rec['overall_accuracy_5d']:.1f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Stage Length Curriculum for CerebroSpheroid V5")
    parser.add_argument("--initial-checkpoint", type=str, default="checkpoints/v5_passkey_focused_test.pt")
    parser.add_argument("--steps-per-stage", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--min-lr", type=float, default=4e-6)
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--max-stage", type=int, default=5)
    parser.add_argument("--eval-trials", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="checkpoints/curriculum")
    args = parser.parse_args()

    run_length_curriculum(
        initial_checkpoint=args.initial_checkpoint,
        steps_per_stage=args.steps_per_stage,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        device_str=args.device,
        output_dir=args.output_dir,
        start_stage=args.start_stage,
        max_stage=args.max_stage,
        eval_trials=args.eval_trials,
    )
