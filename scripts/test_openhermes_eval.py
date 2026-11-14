#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: OpenHermes Evaluation & Generation Probe
==============================================================
Evaluates checkpoint(s) on genuine OpenHermes-2.5 questions:
1. Computes Cross-Entropy Evaluation Loss & Perplexity on ground truth responses.
2. Runs greedy and stochastic motor generation to inspect text outputs.
3. Compares behavior across checkpoints (e.g. Best 586k vs Latest 593k / 614k).

Usage:
  python3 scripts/test_openhermes_eval.py [checkpoint_path]
"""

import sys
import os
import math
import argparse
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from src.jax.checkpoint import adapt_jax_to_torch_state_dict
from src.corpus import (
    format_stimulus_prompt,
    format_motor_response,
    clean_motor_response,
    encode_no_special,
    get_bio_token_ids,
)


def load_openhermes_bench_questions(parquet_path: str, count: int = 8):
    """Loads a curated diverse set of questions from OpenHermes parquet."""
    table = pq.read_table(parquet_path)
    df = table.slice(0, 100).to_pandas()
    
    selected = []
    # Pick diverse indices
    target_indices = [0, 1, 2, 3, 4, 15, 20, 25]
    for idx in target_indices:
        if idx >= len(df):
            continue
        row = df.iloc[idx]
        convs = row.get("conversations", [])
        q = next((c["value"] for c in convs if c["from"] in ("human", "user")), None)
        a = next((c["value"] for c in convs if c["from"] in ("gpt", "assistant")), None)
        category = row.get("category", "general")
        if q and a:
            selected.append({
                "index": idx,
                "category": category,
                "question": q.strip(),
                "ground_truth": a.strip(),
            })
        if len(selected) >= count:
            break
            
    return selected


def evaluate_checkpoint_on_openhermes(ckpt_path: str, questions: list, device: torch.device):
    print("=" * 80)
    print(f"🧠 CEREBRO-SPHEROID V5: OPENHERMES-2.5 BENCHMARK EVALUATION")
    print(f"📁 Checkpoint: {ckpt_path}")
    print(f"🖥️  Compute Device: {device}")
    print("=" * 80)

    # 1. Load Tokenizer
    tok_dir = os.path.join(REPO_ROOT, "data", "tokenizer")
    if not os.path.exists(os.path.join(tok_dir, "tokenizer.json")):
        tok_dir = "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/data/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    print(f"🔤 Tokenizer: {tok_dir} (Vocab: {len(tokenizer):,})")

    # 2. Load Model
    raw_cfg = PRESETS_V5.get("v5_small")
    cfg = ModelConfigV5(**raw_cfg)
    model = CerebroSpheroidV5(cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    step = ckpt.get("step", "N/A")
    train_loss = ckpt.get("loss", "N/A")
    print(f"📍 Checkpoint Step: {step} | Training Loss: {train_loss:.5f}" if isinstance(train_loss, (int, float)) else f"📍 Checkpoint Step: {step}")

    if any(k.startswith("core.intra_swiglu") for k in state.keys()) or any("proj_in" in k for k in state.keys()):
        state = adapt_jax_to_torch_state_dict(state, cfg)

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    print("✓ Weights loaded and synchronized successfully!\n")

    losses = []
    print("=" * 80)
    print("🧪 EVALUATION ON OPENHERMES QUESTIONS")
    print("=" * 80)

    for i, item in enumerate(questions, 1):
        q = item["question"]
        gt = item["ground_truth"]
        cat = item["category"]

        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📌 [Test {i}/{len(questions)}] Category: {cat}")
        print(f"❓ QUESTION:\n{q}")
        print(f"📖 GROUND TRUTH (snippet):\n{gt[:180]}...")

        # 1. Compute Cross-Entropy Loss on Ground Truth
        formatted_prompt = format_stimulus_prompt(q)
        formatted_target = format_motor_response(gt)

        p_ids = encode_no_special(tokenizer, formatted_prompt, max_length=256)
        t_ids = encode_no_special(tokenizer, formatted_target, max_length=512)

        p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
        t_tensor = torch.tensor([t_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(p_tensor, targets=t_tensor)
            if isinstance(out, tuple):
                loss_val = float(out[0].item())
            else:
                loss_val = float(out.item())
        ppl = math.exp(min(loss_val, 20.0))
        losses.append(loss_val)
        print(f"\n📊 Teacher-Forcing Cross-Entropy Loss: {loss_val:.4f} (Perplexity: {ppl:.2f})")

        # 2. Autonomous Motor Generation (Greedy T=0.2)
        with torch.no_grad():
            gen_greedy = model.generate_text(
                prompt=q,
                tokenizer=tokenizer,
                max_new_tokens=64,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.2,
                device=device,
                strict_act=False,
            )
        clean_greedy = clean_motor_response(gen_greedy, strict=False).strip()
        print(f"\n🤖 MODEL GENERATION (Greedy T=0.2):\n{clean_greedy if clean_greedy else repr(gen_greedy)}")

        # 3. Autonomous Motor Generation (Sampling T=0.6)
        with torch.no_grad():
            gen_sample = model.generate_text(
                prompt=q,
                tokenizer=tokenizer,
                max_new_tokens=64,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                device=device,
                strict_act=False,
            )
        clean_sample = clean_motor_response(gen_sample, strict=False).strip()
        print(f"\n🎲 MODEL GENERATION (Sampling T=0.6):\n{clean_sample if clean_sample else repr(gen_sample)}")

    avg_loss = sum(losses) / len(losses)
    avg_ppl = math.exp(min(avg_loss, 20.0))
    print("\n" + "=" * 80)
    print("📈 OPENHERMES BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Target Checkpoint         : {ckpt_path}")
    print(f"Checkpoint Step           : {step}")
    print(f"Average OpenHermes Loss   : {avg_loss:.4f}")
    print(f"Average Perplexity        : {avg_ppl:.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CerebroSpheroid on OpenHermes questions")
    parser.add_argument("checkpoint", nargs="?", default=None, help="Path to checkpoint .pt file")
    args = parser.parse_args()

    ckpt_target = args.checkpoint
    if ckpt_target is None:
        candidates = [
            "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_latest_step664k.pt",
            "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_best_step586k.pt",
            "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_latest_step593k.pt",
            "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_latest_step612k.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_target = c
                break

    parquet_candidates = [
        "/home/ajit/Documents/Startup/Active_Development/reasoning_agent/data/openhermes/openhermes_2_5.parquet",
        os.path.join(REPO_ROOT, "data", "openhermes", "openhermes_2_5.parquet"),
    ]
    parquet_file = next((p for p in parquet_candidates if os.path.exists(p)), None)
    if not parquet_file:
        print("❌ OpenHermes parquet not found!")
        sys.exit(1)
    questions = load_openhermes_bench_questions(parquet_file, count=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_checkpoint_on_openhermes(ckpt_target, questions, device)
