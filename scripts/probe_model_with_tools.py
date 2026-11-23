#!/usr/bin/env python3
"""
🔬 CerebroSpheroid V5: Extended Tool-Assisted Evaluation & Diagnostic Probe
==========================================================================
Evaluates CerebroSpheroid V5 checkpoints across a comprehensive 15-test battery of
multi-step arithmetic, physical/chemical constants, geography, and algorithms:
  1. Zero-Tool Condition: Raw model output from internal weights alone
  2. Autonomous Tool Invocation: Model generates <think> and <tool_call>
  3. Closed-Loop Execution: Executes tool via UnifiedToolDispatcher and validates answer
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    PRESETS_V5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.jax.checkpoint import adapt_jax_to_torch_state_dict
from src.tools.unified_tool_dispatcher import UnifiedToolDispatcher

EXTENDED_TEST_BATTERY = [
    # 1. Multi-step Arithmetic
    {
        "category": "Arithmetic (Multi-step)",
        "prompt": "What is (17 * 24) + 85?",
        "canonical_call": "calc((17 * 24) + 85)",
        "ground_truth": "493",
    },
    # 2. Exponentiation
    {
        "category": "Arithmetic (Powers & Roots)",
        "prompt": "What is 2^10 - 24?",
        "canonical_call": "calc(2**10 - 24)",
        "ground_truth": "1000",
    },
    # 3. Square Roots & Division
    {
        "category": "Arithmetic (Square Roots)",
        "prompt": "Calculate sqrt(1089) / 3",
        "canonical_call": "calc(sqrt(1089) / 3)",
        "ground_truth": "11",
    },
    # 4. Large Multiplication
    {
        "category": "Arithmetic (Complex Products)",
        "prompt": "Compute 125 * 32",
        "canonical_call": "calc(125 * 32)",
        "ground_truth": "4000",
    },
    # 5. Physics: Speed of Light
    {
        "category": "Physics Constants (c)",
        "prompt": "What is the speed of light in vacuum?",
        "canonical_call": "factual('speed of light')",
        "ground_truth": "299,792,458 m/s",
    },
    # 6. Physics: Planck's Constant
    {
        "category": "Physics Constants (h)",
        "prompt": "What is Planck's constant?",
        "canonical_call": "factual('planck')",
        "ground_truth": "6.626",
    },
    # 7. Chemistry: Gold
    {
        "category": "Chemistry (Periodic Table: Au)",
        "prompt": "What is the atomic number of Gold?",
        "canonical_call": "factual('atomic number of Gold')",
        "ground_truth": "79",
    },
    # 8. Chemistry: Carbon
    {
        "category": "Chemistry (Periodic Table: C)",
        "prompt": "What is the atomic number of Carbon?",
        "canonical_call": "factual('atomic number of Carbon')",
        "ground_truth": "6",
    },
    # 9. Chemistry: Water
    {
        "category": "Chemistry (Molecular Formula)",
        "prompt": "What is the chemical formula for water?",
        "canonical_call": "factual('chemical formula for water')",
        "ground_truth": "H2O",
    },
    # 10. Geography: France
    {
        "category": "Geography (Capitals)",
        "prompt": "What is the capital of France?",
        "canonical_call": "factual('capital of France')",
        "ground_truth": "Paris",
    },
    # 11. Geography: Japan
    {
        "category": "Geography (Capitals)",
        "prompt": "What is the capital of Japan?",
        "canonical_call": "factual('capital of Japan')",
        "ground_truth": "Tokyo",
    },
    # 12. Geography: Germany
    {
        "category": "Geography (Capitals)",
        "prompt": "What is the capital of Germany?",
        "canonical_call": "factual('capital of Germany')",
        "ground_truth": "Berlin",
    },
    # 13. Algorithm: Prime Count
    {
        "category": "Algorithm (Python Prime Counter)",
        "prompt": "How many primes are strictly below 20?",
        "canonical_call": 'python(print(len([n for n in range(2, 20) if all(n % d != 0 for d in range(2, int(n**0.5) + 1))])))',
        "ground_truth": "8",
    },
    # 14. Algorithm: String Reversal
    {
        "category": "Algorithm (Python String Reversal)",
        "prompt": "Reverse the word 'cerebro' in Python",
        "canonical_call": "python(print('cerebro'[::-1]))",
        "ground_truth": "orberec",
    },
    # 15. Algorithm: Sum of Squares
    {
        "category": "Algorithm (Python Math Series)",
        "prompt": "Calculate the sum of squares from 1 to 5 in Python",
        "canonical_call": "python(print(sum(i**2 for i in range(1, 6))))",
        "ground_truth": "55",
    },
]


def load_model(checkpoint_path: str, device: str = "cuda"):
    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model", ckpt))

    cfg_dict = PRESETS_V5["v5_small"].copy()
    cfg_dict["sensory_position_steps"] = 4096
    cfg_dict["max_phase_steps"] = 4096
    cfg_dict["max_seq_len"] = 4096
    cfg_dict["use_rope"] = True
    cfg_dict["top_k_resonance"] = 0
    cfg_dict["use_deliberation_probe"] = False
    cfg_dict["use_adaptive_thinking"] = False
    cfg = ModelConfigV5(**cfg_dict)

    if any(k.startswith(("sensory.", "core.", "motor.")) for k in state.keys()):
        state = adapt_jax_to_torch_state_dict(state, cfg)

    model = CerebroSpheroidV5(cfg)
    matched, total = load_v5_checkpoint_into_torch(model, state, verbose=False)
    model.to(dev)
    model.eval()
    return model, tokenizer, ckpt.get("step", "unknown"), dev


def run_extended_probe(checkpoint_path: str, device: str = "cuda"):
    model, tokenizer, step, dev = load_model(checkpoint_path, device)
    dispatcher = UnifiedToolDispatcher()

    print(f"\n{'='*85}")
    print(f"🔬 CEREBRO-SPHEROID V5: EXTENDED 15-TASK AGENTIC PROBE")
    print(f"Checkpoint : {os.path.basename(checkpoint_path)} (Step: {step})")
    print(f"Device     : {dev}")
    print(f"{'='*85}\n")

    zero_tool_hits = 0
    auto_tool_calls_detected = 0
    closed_loop_hits = 0
    forced_hits = 0
    total = len(EXTENDED_TEST_BATTERY)

    for idx, test in enumerate(EXTENDED_TEST_BATTERY, 1):
        cat = test["category"]
        prompt = test["prompt"]
        gt = test["ground_truth"]
        canonical_call = test["canonical_call"]

        print(f"[{idx:02d}/{total:02d}] {cat}")
        print(f"  Prompt       : \"{prompt}\"")
        print(f"  Ground Truth : {gt}")

        # 1. Zero-Tool Condition (Model raw forward generation)
        p_text = f"<s> {prompt}\n"
        ids = tokenizer.encode(p_text, add_special_tokens=False)
        inp = torch.tensor([ids], dtype=torch.long, device=dev)

        with torch.no_grad():
            out_ids = model.generate(inp, max_new_tokens=64, temperature=0.1)
        raw_output = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=False).strip()
        raw_snippet = raw_output.replace("\n", " ")[:65]

        is_direct_hit = gt.lower() in raw_output.lower()
        if is_direct_hit:
            zero_tool_hits += 1
        print(f"  Zero-Tool    : \"{raw_snippet}...\" [{'✓ PASS' if is_direct_hit else '✗ FAIL'}]")

        # 2. Check Autonomous Tool-Calling Emergence
        m_call = None
        if "<tool_call>" in raw_output:
            m = re.search(r"<tool_call>(.*?)</tool_call>", raw_output, re.DOTALL)
            if m and len(m.group(1).strip()) > 3:
                m_call = m.group(1).strip()
        elif any(raw_output.strip().startswith(p) for p in ("calc(", "python(", "factual(", "facts(")):
            m_call = raw_output.strip()

        if m_call:
            auto_tool_calls_detected += 1
            print(f"  Tool Call    : Model emitted: {m_call[:60]}")
            tool_result = dispatcher.execute_raw_call_string(m_call)
            is_tool_hit = gt.lower() in tool_result.lower()

            # Ingest tool response back into sensory context for Stage 2 Answer Generation
            tool_block = f"\n<tool_response> {tool_result} </tool_response>\nAnswer:"
            stage2_text = f"{p_text}<tool_call> {m_call} </tool_call>{tool_block}"
            stage2_ids = tokenizer.encode(stage2_text, add_special_tokens=False)
            stage2_inp = torch.tensor([stage2_ids], dtype=torch.long, device=dev)

            with torch.no_grad():
                stage2_out = model.generate(stage2_inp, max_new_tokens=64, temperature=0.1)
            final_answer = tokenizer.decode(stage2_out[0].tolist(), skip_special_tokens=False).strip()
            final_snippet = final_answer.replace("\n", " ")[:80]

            # Honest evaluation: final answer must contain ground truth
            is_closed_hit = gt.lower() in final_answer.lower()
            if is_closed_hit:
                closed_loop_hits += 1

            print(f"  Tool Exec    : {m_call[:40]} ➔ \"{tool_result[:35]}\" [{'✓ PASS' if is_tool_hit else '✗ FAIL'}]")
            print(f"  Final Answer : \"{final_snippet}...\" [{'✓ PASS' if is_closed_hit else '✗ FAIL'}]")
        else:
            print(f"  Tool Call    : [No valid tool call emitted by model] [✗ FAIL]")
            print(f"  Tool Exec    : [Skipped - zero model call]")
            
            # Also test Forced-Sensory-KV Stage 2 Grounding to diagnose whether the motor decoder can extract from sensory KV
            canon_exec = dispatcher.execute_raw_call_string(canonical_call)
            forced_tool_block = f"\n<tool_response> {canon_exec} </tool_response>\nAnswer:"
            forced_stage2_text = f"{p_text}<tool_call> {canonical_call} </tool_call>{forced_tool_block}"
            forced_ids = tokenizer.encode(forced_stage2_text, add_special_tokens=False)
            forced_inp = torch.tensor([forced_ids], dtype=torch.long, device=dev)
            with torch.no_grad():
                forced_out = model.generate(forced_inp, max_new_tokens=64, temperature=0.1)
            forced_ans = tokenizer.decode(forced_out[0].tolist(), skip_special_tokens=False).strip()
            forced_snippet = forced_ans.replace("\n", " ")[:80]
            is_forced_hit = gt.lower() in forced_ans.lower()
            if is_forced_hit:
                forced_hits += 1
            print(f"  Sensory-KV Extraction Probe: \"{forced_snippet}...\" [{'✓ PASS' if is_forced_hit else '✗ FAIL'}]")
        print()

    print(f"{'='*85}")
    print(f"📊 EXTENDED 15-TASK AGENTIC BENCHMARK SUMMARY:")
    print(f"  Zero-Tool Accuracy (Weights Alone)     : {zero_tool_hits:02d}/{total:02d} ({zero_tool_hits/total*100:5.1f}%)")
    print(f"  Cognitive Tool Syntax Emergence        : {auto_tool_calls_detected:02d}/{total:02d} ({auto_tool_calls_detected/total*100:5.1f}%)")
    print(f"  Autonomous End-to-End Accuracy         : {closed_loop_hits:02d}/{total:02d} ({closed_loop_hits/total*100:5.1f}%)")
    print(f"  Sensory-KV Grounded Extraction Rate    : {forced_hits:02d}/{total:02d} ({forced_hits/total*100:5.1f}%)")
    print(f"{'='*85}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_small_latest.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_extended_probe(args.checkpoint, args.device)
