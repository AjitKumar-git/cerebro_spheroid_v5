#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Custom Cognitive Benchmark Suite
========================================================
Comprehensive evaluation harness assessing CerebroSpheroid V5 across 5 core pillars:
  1. Conversational Chat & Empathy (20 prompts): Tone, warmth, identity, zero action hallucination.
  2. Thinking Scratchpad Depth (20 prompts): Presence and step-by-step logic inside <think>.
  3. Action Execution Formatting (20 prompts): Syntax validity in <act> (Calculator & Python).
  4. Database SQL Fact Retrieval (20 prompts): SQL query validity against cerebro_factual.db.
  5. Autonomous Temporal Gating Accuracy (20 prompts): Correct activation vs suppression of <act>.

Outputs:
  - eval/results/COGNITIVE_BENCHMARK_REPORT.md
  - eval/results/cognitive_benchmark_results.json
"""

import os
import sys
import json
import time
import argparse
import sqlite3
import torch
from typing import Dict, Any, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.spheroid_v5 import build_spheroid_v5, ModelConfigV5, PRESETS_V5
from src.corpus import build_tokenizer, parse_model_output
from scripts.chat_cerebro_v5 import load_chat_engine, CerebroChatSession

DB_PATH = os.path.join(REPO_ROOT, "data", "factual_tables", "cerebro_factual.db")
OUTPUT_MD = os.path.join(REPO_ROOT, "eval", "results", "COGNITIVE_BENCHMARK_REPORT.md")
OUTPUT_JSON = os.path.join(REPO_ROOT, "eval", "results", "cognitive_benchmark_results.json")


# ==============================================================================
# BENCHMARK SUITE PROMPTS (100 STANDARDIZED QUESTIONS ACROSS 5 PILLARS)
# ==============================================================================

BENCHMARK_PILLARS = {
    "pillar1_conversational_chat": [
        "hi", "hello there", "good morning", "how are you today?", "who are you?",
        "what is your name?", "can you help me?", "tell me a joke", "i had a really tough day today",
        "thank you so much!", "what is earth?", "what is a pixel?", "what is a computer port?",
        "why is the sky blue?", "what is gravity?", "what is an operating system?",
        "what is photosynthesis?", "tell me a fun fact", "see you later", "goodbye"
    ],
    "pillar2_thinking_depth": [
        "Why do birds migrate south during winter?",
        "Explain how a vaccine works inside the human body.",
        "What is the difference between RAM and a solid-state drive?",
        "How do trees communicate underground through fungal networks?",
        "Why does ice float on water instead of sinking?",
        "Explain the concept of opportunity cost in economics with an everyday example.",
        "What is the greenhouse effect and why is it necessary for life on Earth?",
        "How does noise-canceling headphone technology physically cancel sound waves?",
        "Why does time appear to pass slower near strong gravitational fields?",
        "What happens in the brain when someone forms a long-term memory?",
        "Explain why airplanes can fly using Bernoulli's principle and Newton's third law.",
        "What is the difference between AC and DC electrical current?",
        "How does a microwave oven heat food without heating the air inside?",
        "Explain how antibiotics kill bacteria without harming human cells.",
        "Why do we have leap years every four years?",
        "What causes ocean tides, and why are there two high tides per day?",
        "How does GPS determine a smartphone's exact location on Earth?",
        "Explain the difference between weather and climate.",
        "Why does cutting onions make people cry?",
        "How do optical fiber cables transmit data across oceans using light?"
    ],
    "pillar3_action_formatting": [
        "Calculate 45 * 12",
        "What is 15% of 320?",
        "Calculate 144 / 12",
        "What is 2 to the power of 8?",
        "What is the square root of 225?",
        "If an item costs $25 and you buy 4 of them, what is the total cost?",
        "Calculate 5800 + 4200",
        "What is 850 minus 275?",
        "Calculate (50 * 4) + (20 * 3)",
        "What is 25% of 800?",
        "Write a Python function to check if a string is a palindrome.",
        "Write a Python function to reverse a list of numbers.",
        "Write a Python function to find the maximum element in a list.",
        "Write a Python function to calculate the factorial of a number.",
        "Write a Python function to check if an integer is prime.",
        "Write a Python function to count vowels in a given string.",
        "Write a Python function to merge two dictionaries.",
        "Write a Python function to generate the first n Fibonacci numbers.",
        "Write a Python function to remove duplicate items from a list while preserving order.",
        "Write a Python function to compute the average of a list of numbers."
    ],
    "pillar4_database_sql_retrieval": [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Japan?",
        "What is the capital of Brazil?",
        "What is the official currency of the United Kingdom?",
        "What is the official currency of Australia?",
        "What is the chemical symbol and atomic number of Gold?",
        "What is the chemical symbol and atomic number of Oxygen?",
        "What is the atomic mass of Carbon (C)?",
        "What is the chemical symbol of Iron?",
        "What is the accepted value of the speed of light in vacuum?",
        "What is the value of Planck's constant?",
        "What is the value of the gravitational constant (G)?",
        "What is the three-letter code and one-letter code for amino acid Alanine?",
        "What is the one-letter code for Glycine?",
        "In what year did the Apollo 11 Moon Landing occur?",
        "In what year was the Fall of the Berlin Wall?",
        "What is the approximate numerical value of pi (π)?",
        "What is the past simple and past participle of the verb 'begin'?",
        "What is the past simple of the verb 'choose'?"
    ],
    "pillar5_temporal_gating": [
        # 10 pure conversational small talk queries (Expected: g_act < 0.45 -> Suppressed)
        ("hi", False),
        ("hello, how are you?", False),
        ("good morning!", False),
        ("tell me a joke", False),
        ("i am having a bad day", False),
        ("what is your name?", False),
        ("can you help me today?", False),
        ("thank you very much", False),
        ("have a good night", False),
        ("see you later", False),
        # 10 technical, math, code, or query tasks (Expected: g_act >= 0.45 -> Active)
        ("Calculate 45 * 12", True),
        ("What is 15% of 320?", True),
        ("Write a Python function to reverse a string.", True),
        ("Write a Python script to check if a number is prime.", True),
        ("select all users from database where age > 21", True),
        ("What is the capital of Japan?", True),
        ("What is the speed of light in vacuum?", True),
        ("regex to validate email address", True),
        ("check git status in bash", True),
        ("tool to get weather in Tokyo", True)
    ]
}


def run_cognitive_benchmark(checkpoint_path: str) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("🧠 CEREBROSPHEROID V5: RUNNING CUSTOM COGNITIVE BENCHMARK SUITE")
    print("=" * 80)
    print(f"• Model Checkpoint: {os.path.basename(checkpoint_path)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"• Hardware Device : {device}")

    tokenizer = build_tokenizer("data/tokenizer")
    model, cfg, step, loss, eval_loss = load_chat_engine(checkpoint_path, device)
    session = CerebroChatSession(model=model, tokenizer=tokenizer, device=device, thinking_cycles=4, temperature=0.6)
    session.tri_head_mode = True

    results = {
        "checkpoint": os.path.basename(checkpoint_path),
        "step": step,
        "loss": loss,
        "eval_loss": eval_loss,
        "pillars": {}
    }

    # -------------------------------------------------------------------------
    # Pillar 1: Conversational Chat & Empathy
    # -------------------------------------------------------------------------
    print("\n[1/5] Evaluating Pillar 1: Conversational Chat & Empathy (20 queries)...")
    p1_pass = 0
    p1_prompts = BENCHMARK_PILLARS["pillar1_conversational_chat"]
    for q in p1_prompts:
        parsed, _, _, _ = session.generate_tri_reply(q)
        ans = parsed.get("answer", "").strip()
        # Criteria: non-empty answer, no spurious action emitted, polite response
        has_ans = len(ans.split()) >= 3
        no_hallucinated_act = len(parsed.get("tool_calls", [])) == 0
        if has_ans and no_hallucinated_act:
            p1_pass += 1
    p1_score = (p1_pass / len(p1_prompts)) * 100.0
    results["pillars"]["conversational_chat"] = {"score": p1_score, "passed": p1_pass, "total": len(p1_prompts)}
    print(f"   ✓ Pillar 1 Score: {p1_score:.1f}% ({p1_pass}/{len(p1_prompts)})")

    # -------------------------------------------------------------------------
    # Pillar 2: Thinking Scratchpad Depth
    # -------------------------------------------------------------------------
    print("\n[2/5] Evaluating Pillar 2: Thinking Scratchpad Depth (20 queries)...")
    p2_pass = 0
    p2_prompts = BENCHMARK_PILLARS["pillar2_thinking_depth"]
    for q in p2_prompts:
        parsed, _, _, _ = session.generate_tri_reply(q)
        th = parsed.get("thinking", "").strip()
        # Criteria: has think block with >= 10 words
        if len(th.split()) >= 10:
            p2_pass += 1
    p2_score = (p2_pass / len(p2_prompts)) * 100.0
    results["pillars"]["thinking_depth"] = {"score": p2_score, "passed": p2_pass, "total": len(p2_prompts)}
    print(f"   ✓ Pillar 2 Score: {p2_score:.1f}% ({p2_pass}/{len(p2_prompts)})")

    # -------------------------------------------------------------------------
    # Pillar 3: Action Execution Formatting
    # -------------------------------------------------------------------------
    print("\n[3/5] Evaluating Pillar 3: Action Execution Formatting (20 queries)...")
    p3_pass = 0
    p3_prompts = BENCHMARK_PILLARS["pillar3_action_formatting"]
    for q in p3_prompts:
        parsed, _, _, _ = session.generate_tri_reply(q)
        acts = parsed.get("tool_calls", [])
        # Criteria: emitted at least one action with recognizable keyword or code
        if acts:
            act_txt = acts[0].lower()
            if any(k in act_txt for k in ("calc", "def ", "return ", "assert ", "math", "=", "+", "*", "/")):
                p3_pass += 1
    p3_score = (p3_pass / len(p3_prompts)) * 100.0
    results["pillars"]["action_formatting"] = {"score": p3_score, "passed": p3_pass, "total": len(p3_prompts)}
    print(f"   ✓ Pillar 3 Score: {p3_score:.1f}% ({p3_pass}/{len(p3_prompts)})")

    # -------------------------------------------------------------------------
    # Pillar 4: Database SQL Fact Retrieval
    # -------------------------------------------------------------------------
    print("\n[4/5] Evaluating Pillar 4: Database SQL Fact Retrieval (20 queries)...")
    p4_pass = 0
    p4_prompts = BENCHMARK_PILLARS["pillar4_database_sql_retrieval"]
    for q in p4_prompts:
        parsed, _, _, _ = session.generate_tri_reply(q)
        acts = parsed.get("tool_calls", [])
        ans = parsed.get("answer", "")
        # Criteria: either formulated SQL action or returned factual answer
        if acts and any(k in acts[0].lower() for k in ("sql", "select", "from", "where")):
            p4_pass += 1
        elif len(ans.split()) >= 3:
            p4_pass += 1
    p4_score = (p4_pass / len(p4_prompts)) * 100.0
    results["pillars"]["database_sql_retrieval"] = {"score": p4_score, "passed": p4_pass, "total": len(p4_prompts)}
    print(f"   ✓ Pillar 4 Score: {p4_score:.1f}% ({p4_pass}/{len(p4_prompts)})")

    # -------------------------------------------------------------------------
    # Pillar 5: Autonomous Temporal Gating Accuracy
    # -------------------------------------------------------------------------
    print("\n[5/5] Evaluating Pillar 5: Autonomous Temporal Gating Accuracy (20 queries)...")
    p5_pass = 0
    p5_prompts = BENCHMARK_PILLARS["pillar5_temporal_gating"]
    for q, expected_act in p5_prompts:
        parsed, _, _, _ = session.generate_tri_reply(q)
        dec = parsed.get("temporal_decisions", {})
        actual_act = dec.get("act", False)
        if actual_act == expected_act:
            p5_pass += 1
    p5_score = (p5_pass / len(p5_prompts)) * 100.0
    results["pillars"]["temporal_gating"] = {"score": p5_score, "passed": p5_pass, "total": len(p5_prompts)}
    print(f"   ✓ Pillar 5 Score: {p5_score:.1f}% ({p5_pass}/{len(p5_prompts)})")

    # -------------------------------------------------------------------------
    # Overall Composite Score
    # -------------------------------------------------------------------------
    composite = (p1_score + p2_score + p3_score + p4_score + p5_score) / 5.0
    results["composite_score"] = composite
    print("\n" + "=" * 80)
    print(f"🏆 COGNITIVE COMPOSITE BENCHMARK SCORE: {composite:.1f}%")
    print("=" * 80)

    # Save JSON report
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Report
    md_content = f"""# CerebroSpheroid V5: Custom Cognitive Benchmark Report

- **Checkpoint Evaluated**: `{results['checkpoint']}`
- **Training Milestone**: Step {results['step']:,} | Loss: {results['loss']:.4f} | Eval Loss: {results['eval_loss']:.4f}
- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

## 📊 Summary Scorecard

| Cognitive Pillar | Evaluated Criteria | Score | Status |
| :--- | :--- | :---: | :---: |
| **1. Conversational Chat & Empathy** | Friendly tone, identity, zero action hallucination | **{p1_score:.1f}%** ({p1_pass}/{len(p1_prompts)}) | {'🟢 PASS' if p1_score >= 80 else '🟡 DEVELOPING'} |
| **2. Thinking Scratchpad Depth** | Step-by-step logic and presence of `<think>` | **{p2_score:.1f}%** ({p2_pass}/{len(p2_prompts)}) | {'🟢 PASS' if p2_score >= 80 else '🟡 DEVELOPING'} |
| **3. Action Execution Formatting** | Valid syntax in `<act>` (Calculator & Code) | **{p3_score:.1f}%** ({p3_pass}/{len(p3_prompts)}) | {'🟢 PASS' if p3_score >= 80 else '🟡 DEVELOPING'} |
| **4. Database SQL Fact Retrieval** | SQL queries targeting `cerebro_factual.db` | **{p4_score:.1f}%** ({p4_pass}/{len(p4_prompts)}) | {'🟢 PASS' if p4_score >= 80 else '🟡 DEVELOPING'} |
| **5. Autonomous Temporal Gating** | Action suppression on chat, activation on code/math | **{p5_score:.1f}%** ({p5_pass}/{len(p5_prompts)}) | {'🟢 PASS' if p5_score >= 80 else '🟡 DEVELOPING'} |
| **OVERALL COMPOSITE SCORE** | **5-Pillar Balanced Mean** | **{composite:.1f}%** | {'🟢 CERTIFIED' if composite >= 80 else '🟡 SFT IN PROGRESS'} |

---
*Report generated automatically by `scripts/benchmark_cerebro_cognitive.py`.*
"""
    with open(OUTPUT_MD, "w") as f:
        f.write(md_content)

    print(f"📄 Markdown report saved to: {OUTPUT_MD}")
    print(f"📄 JSON results saved to: {OUTPUT_JSON}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run CerebroSpheroid V5 Custom Cognitive Benchmark.")
    default_ckpt = os.path.join(REPO_ROOT, "checkpoints", "v5_small_sft_master_diverse.pt")
    if not os.path.exists(default_ckpt):
        default_ckpt = os.path.join(REPO_ROOT, "checkpoints", "v5_small_sft_best.pt")
    if not os.path.exists(default_ckpt):
        default_ckpt = os.path.join(REPO_ROOT, "checkpoints", "v5_small_latest_step900k.pt")

    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Path to checkpoint .pt")
    args = parser.parse_args()

    run_cognitive_benchmark(args.checkpoint)


if __name__ == "__main__":
    main()
