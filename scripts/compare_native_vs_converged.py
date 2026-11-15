#!/usr/bin/env python3
"""
Direct Head-to-Head Benchmark: Native V5 Champion vs. Converged SmolLM-Transferred V5
=====================================================================================
Compares:
1. Native Pre-trained V5 (Step 900,000, 102.4M params, TPU Pretrained)
2. Converged SmolLM2-Transferred V5 (Step 1,038, 102.4M params, GPU Aligned to Loss 2.65)
Across:
- MMLU Academic Knowledge (Log-Likelihood Multiple Choice)
- ARC-Challenge Scientific Reasoning (Log-Likelihood Multiple Choice)
- HellaSwag Physical / Common Sense Logic
- GSM8K Mathematical Word Problems (Generative CoT)
- HumanEval Python Code Synthesis (AST Validity)
- Canonical Test Prompts (Factual, Arithmetic, Definition, Code)
"""

import os
import sys
import json
import math
import time
import ast
import re
import torch
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from transformers import AutoTokenizer
from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from src.jax.checkpoint import adapt_jax_to_torch_state_dict
from scripts.benchmark_frontier_suite import (
    MMLU_TASKS,
    ARC_TASKS,
    HELLASWAG_TASKS,
    GSM8K_TASKS,
    HUMANEVAL_TASKS,
    score_multiple_choice_loglik,
    evaluate_generative_response,
    extract_gsm8k_number,
)


def load_model(checkpoint_path: str, cfg: ModelConfigV5, device: torch.device):
    model = CerebroSpheroidV5(cfg).to(device)
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = raw_ckpt.get("model_state", raw_ckpt.get("model", raw_ckpt))
    
    # If TPU/JAX keys
    if any(k.startswith(("sensory.", "core.", "motor.")) for k in state.keys()):
        state = adapt_jax_to_torch_state_dict(state, cfg)
        
    model.load_state_dict(state, strict=False)
    model.eval()
    step = raw_ckpt.get("step", "unknown")
    loss = raw_ckpt.get("calibration_loss", raw_ckpt.get("loss", "unknown"))
    return model, step, loss


def evaluate_suite_on_model(model, tokenizer, device, name: str):
    print(f"\n" + "=" * 70)
    print(f"📊 Running Benchmark Battery on: {name}")
    print("=" * 70)

    results = {}
    letters = ["A", "B", "C", "D"]

    # 1. MMLU
    mmlu_correct = 0
    for item in MMLU_TASKS:
        prompt = f"Question: {item['question']}\nChoices:\n"
        for i, ch in enumerate(item["choices"]):
            prompt += f"{letters[i]}) {ch}\n"
        prompt += "Answer:"
        pred_idx, _ = score_multiple_choice_loglik(model, tokenizer, prompt, item["choices"], device)
        if letters[pred_idx] == item["answer"]:
            mmlu_correct += 1
    mmlu_score = (mmlu_correct / len(MMLU_TASKS)) * 100.0
    results["MMLU"] = mmlu_score
    print(f"   ✓ MMLU Score:        {mmlu_score:.1f}% ({mmlu_correct}/{len(MMLU_TASKS)})")

    # 2. ARC-Challenge
    arc_correct = 0
    for item in ARC_TASKS:
        prompt = f"Question: {item['question']}\nChoices:\n"
        for i, ch in enumerate(item["choices"]):
            prompt += f"{letters[i]}) {ch}\n"
        prompt += "Answer:"
        pred_idx, _ = score_multiple_choice_loglik(model, tokenizer, prompt, item["choices"], device)
        if letters[pred_idx] == item["answer"]:
            arc_correct += 1
    arc_score = (arc_correct / len(ARC_TASKS)) * 100.0
    results["ARC"] = arc_score
    print(f"   ✓ ARC-Challenge:     {arc_score:.1f}% ({arc_correct}/{len(ARC_TASKS)})")

    # 3. HellaSwag
    hella_correct = 0
    for item in HELLASWAG_TASKS:
        prompt = f"Context: {item['context']}\nContinuations:\n"
        for i, ch in enumerate(item["choices"]):
            prompt += f"{letters[i]}) {ch}\n"
        prompt += "Answer:"
        pred_idx, _ = score_multiple_choice_loglik(model, tokenizer, prompt, item["choices"], device)
        if letters[pred_idx] == item["answer"]:
            hella_correct += 1
    hella_score = (hella_correct / len(HELLASWAG_TASKS)) * 100.0
    results["HellaSwag"] = hella_score
    print(f"   ✓ HellaSwag:         {hella_score:.1f}% ({hella_correct}/{len(HELLASWAG_TASKS)})")

    # 4. GSM8K Exact Match
    gsm_correct = 0
    for item in GSM8K_TASKS:
        prompt = f"Question: {item['question']}\nAnswer:"
        generated = evaluate_generative_response(model, tokenizer, prompt, max_tokens=48, temperature=0.1, device=device)
        extracted = extract_gsm8k_number(generated)
        if extracted == item["target_number"]:
            gsm_correct += 1
    gsm_score = (gsm_correct / len(GSM8K_TASKS)) * 100.0
    results["GSM8K"] = gsm_score
    print(f"   ✓ GSM8K Math:        {gsm_score:.1f}% ({gsm_correct}/{len(GSM8K_TASKS)})")

    # 5. HumanEval-Mini Code Syntax
    code_valid = 0
    for item in HUMANEVAL_TASKS:
        prompt = item["prompt"]
        generated = evaluate_generative_response(model, tokenizer, prompt, max_tokens=48, temperature=0.1, device=device)
        full_code = prompt + generated
        is_valid = False
        try:
            clean_code = re.sub(r"^```python\s*", "", full_code)
            clean_code = re.sub(r"```.*$", "", clean_code)
            ast.parse(clean_code)
            is_valid = True
        except Exception:
            is_valid = False
        if is_valid:
            code_valid += 1
    code_score = (code_valid / len(HUMANEVAL_TASKS)) * 100.0
    results["HumanEval"] = code_score
    print(f"   ✓ HumanEval Code:    {code_score:.1f}% ({code_valid}/{len(HUMANEVAL_TASKS)})")

    # Composite Score
    composite = sum(results.values()) / len(results)
    results["Composite"] = round(composite, 2)
    print(f"   ⭐ Composite Score:  {composite:.2f}%")

    return results


def run_comparative_prompts(native_model, converged_model, tokenizer, device):
    test_prompts = [
        "The capital of France is",
        "Question: If 2 + 2 = 4, then 4 + 4 equals what? Answer:",
        "What is an atom?",
        "def is_even(n):",
    ]

    print("\n" + "=" * 70)
    print("🔬 CANONICAL PROMPT HEAD-TO-HEAD GENERATION")
    print("=" * 70)

    comparison_outputs = []
    for idx, prompt in enumerate(test_prompts, 1):
        print(f"\n[Prompt {idx}] \"{prompt}\"")
        with torch.no_grad():
            nat_text = native_model.generate_text(
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=48,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                device=device,
                strict_act=False,
            ).strip()

            conv_text = converged_model.generate_text(
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=48,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                device=device,
                strict_act=False,
            ).strip()

        print(f"  🧠 [Native Step 900k]:     \"{nat_text}\"")
        print(f"  🎯 [Converged Step 1038]:  \"{conv_text}\"")

        comparison_outputs.append({
            "prompt": prompt,
            "native_output": nat_text,
            "converged_output": conv_text,
        })

    return comparison_outputs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("🥊 HEAD-TO-HEAD SHOWDOWN: NATIVE V5 vs CONVERGED SMOL-ALIGNED V5")
    print(f"Device: {device}")
    print("=" * 80)

    v5_tok = AutoTokenizer.from_pretrained(os.path.join(WORKSPACE_ROOT, "data", "tokenizer"))
    cfg = ModelConfigV5(**PRESETS_V5["tpu_v5e_1"])

    # Load 1: Native V5
    native_ckpt = os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_best.pt")
    native_model, nat_step, nat_loss = load_model(native_ckpt, cfg, device)
    print(f"Loaded Native Model: Step {nat_step} | Loss: {nat_loss}")

    # Load 2: Converged V5
    converged_ckpt = os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_smollm_converged.pt")
    converged_model, conv_step, conv_loss = load_model(converged_ckpt, cfg, device)
    print(f"Loaded Converged Model: Step {conv_step} | Loss: {conv_loss:.4f}")

    # Evaluate Suites
    nat_scores = evaluate_suite_on_model(native_model, v5_tok, device, f"Native V5 (Step {nat_step:,})")
    conv_scores = evaluate_suite_on_model(converged_model, v5_tok, device, f"Converged SmolLM-V5 (Step {conv_step:,})")

    # Run side-by-side prompt generation
    prompt_showdown = run_comparative_prompts(native_model, converged_model, v5_tok, device)

    # Save summary report
    report = {
        "timestamp": time.time(),
        "native": {
            "checkpoint": "v5_small_best.pt",
            "step": nat_step,
            "pretraining_loss": float(nat_loss) if isinstance(nat_loss, (int, float)) else nat_loss,
            "scores": nat_scores,
        },
        "converged": {
            "checkpoint": "v5_small_smollm_converged.pt",
            "step": conv_step,
            "calibration_loss": float(conv_loss),
            "scores": conv_scores,
        },
        "prompt_showdown": prompt_showdown,
    }

    report_path = os.path.join(WORKSPACE_ROOT, "eval", "results", "native_vs_converged_benchmark.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Full comparative report saved to: {report_path}")


if __name__ == "__main__":
    main()
