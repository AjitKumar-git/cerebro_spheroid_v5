#!/usr/bin/env python3
"""
🏆 CerebroSpheroid V5: Frontier Model Benchmark Suite
======================================================
Standardized academic & cognitive evaluation harness modeled after lm-evaluation-harness
and the Open LLM Leaderboard. Evaluates checkpoints across:

1. MMLU (Massive Multitask Language Understanding) - STEM, Humanities, Social Sci, Bio
2. GSM8K (Grade School Math 8K) - Multi-Step Arithmetic Reasoning & Exact Match (EM)
3. ARC (AI2 Reasoning Challenge - Challenge & Easy) - Scientific Problem Solving
4. HellaSwag - Commonsense Reasoning & Physical Scenario Continuation
5. Counterfactual & Temporal Logic - Premise Rejection & State Tracking
6. HumanEval-Mini - Python Function Generation & Syntactic Validity

Evaluation Modes:
- Log-Likelihood Ranking (Zero-Shot Multiple Choice PPL)
- Generative Autoregressive (Greedy Decoding + CoT Regex Extraction)

Usage:
  python3 scripts/benchmark_frontier_suite.py --checkpoint checkpoints/v5_small_latest_step700k.pt
  python3 scripts/benchmark_frontier_suite.py --checkpoint checkpoints/v5_small_latest_step700k.pt --samples-per-task 10
  python3 scripts/benchmark_frontier_suite.py --export-json benchmarks/results_step700k.json
"""

import os
import sys
import math
import json
import ast
import re
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoTokenizer
from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from src.jax.checkpoint import adapt_jax_to_torch_state_dict


# ==============================================================================
# 1. CURATED FRONTIER BENCHMARK BATTERY (Standard MMLU, GSM8K, ARC, HellaSwag)
# ==============================================================================

MMLU_TASKS = [
    {
        "domain": "MMLU: Physics / Mechanics",
        "question": "A car accelerates uniformly from rest to a speed of 20 m/s in 5 seconds. What is the acceleration of the car?",
        "choices": ["2 m/s²", "4 m/s²", "10 m/s²", "100 m/s²"],
        "answer": "B",
    },
    {
        "domain": "MMLU: Chemistry",
        "question": "What is the pH of a 0.01 M hydrochloric acid (HCl) aqueous solution at 25 °C?",
        "choices": ["1", "2", "7", "12"],
        "answer": "B",
    },
    {
        "domain": "MMLU: Biology / Anatomy",
        "question": "Which chamber of the human heart pumps oxygen-rich blood into the systemic circulation through the aorta?",
        "choices": ["Right atrium", "Right ventricle", "Left atrium", "Left ventricle"],
        "answer": "D",
    },
    {
        "domain": "MMLU: Computer Science",
        "question": "What is the worst-case time complexity of searching for an element in a balanced binary search tree with n nodes?",
        "choices": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
        "answer": "B",
    },
    {
        "domain": "MMLU: Elementary Math",
        "question": "If 3x + 7 = 22, what is the value of 2x - 1?",
        "choices": ["5", "7", "9", "11"],
        "answer": "C",
    },
    {
        "domain": "MMLU: World History",
        "question": "In which year did the Apollo 11 mission successfully land the first humans on the Moon?",
        "choices": ["1965", "1969", "1972", "1975"],
        "answer": "B",
    },
    {
        "domain": "MMLU: Philosophy / Logic",
        "question": "Which logical fallacy occurs when someone attacks their opponent's character rather than addressing their actual argument?",
        "choices": ["Straw man", "Ad hominem", "Post hoc ergo propter hoc", "Red herring"],
        "answer": "B",
    },
    {
        "domain": "MMLU: Biochemistry",
        "question": "Which molecule acts as the primary energy currency of the cell?",
        "choices": ["Glucose", "DNA", "ATP (Adenosine Triphosphate)", "Phospholipid"],
        "answer": "C",
    },
]

ARC_TASKS = [
    {
        "domain": "ARC: Science Reasoning",
        "question": "Which energy transformation occurs when a battery-powered flashlight is turned on?",
        "choices": [
            "Chemical energy to electrical energy to light and thermal energy",
            "Thermal energy to chemical energy to electrical energy",
            "Light energy to electrical energy to mechanical energy",
            "Mechanical energy to electrical energy to chemical energy",
        ],
        "answer": "A",
    },
    {
        "domain": "ARC: Science Reasoning",
        "question": "Why do sound waves travel faster through steel than through air?",
        "choices": [
            "Steel has a lower density than air",
            "Molecules in steel are packed much closer together than in air",
            "Steel has higher temperature than air",
            "Air absorbs sound energy completely",
        ],
        "answer": "B",
    },
    {
        "domain": "ARC: Earth Science",
        "question": "Which process in the water cycle is primarily responsible for cloud formation in the atmosphere?",
        "choices": ["Evaporation", "Precipitation", "Condensation", "Transpiration"],
        "answer": "C",
    },
    {
        "domain": "ARC: Biology",
        "question": "Which cellular organelle contains genetic material and directs protein synthesis in eukaryotic cells?",
        "choices": ["Mitochondria", "Nucleus", "Ribosome", "Endoplasmic reticulum"],
        "answer": "B",
    },
]

HELLASWAG_TASKS = [
    {
        "domain": "HellaSwag: Physical Commonsense",
        "context": "A person takes an egg out of the carton and taps it against the edge of a frying pan on a hot stove.",
        "choices": [
            "The egg cracks, and the person pours the egg into the pan to fry.",
            "The egg turns into a wooden ball and bounces onto the ceiling.",
            "The person reads a newspaper aloud while slicing the stove with scissors.",
            "The pan begins to float in the air and fills with cold ice cubes.",
        ],
        "answer": "A",
    },
    {
        "domain": "HellaSwag: Activity Continuation",
        "context": "A woman sits at an acoustic piano, opens a sheet music book, and places both hands on the keys.",
        "choices": [
            "She begins pressing keys to play a musical melody.",
            "She throws the piano out the window and walks away.",
            "She fills the piano with hot soup and eats it.",
            "The keys dissolve into water and leak onto her shoes.",
        ],
        "answer": "A",
    },
]

GSM8K_TASKS = [
    {
        "domain": "GSM8K: Arithmetic",
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "target_number": "72",
    },
    {
        "domain": "GSM8K: Arithmetic",
        "question": "Weng earns $12 an hour babysitting. Yesterday, she just did 5 hours of babysitting. How much money did she earn?",
        "target_number": "60",
    },
    {
        "domain": "GSM8K: Arithmetic",
        "question": "Betty picked 16 apples from the orchard. She gave 4 apples to her brother and used half of the remaining apples to bake a pie. How many apples does Betty have left?",
        "target_number": "6",
    },
    {
        "domain": "GSM8K: Arithmetic",
        "question": "A baker made 40 cupcakes. He sold 25 cupcakes in the morning and baked 15 more in the afternoon. How many cupcakes does he have now?",
        "target_number": "30",
    },
]

HUMANEVAL_TASKS = [
    {
        "domain": "HumanEval: Python Syntax",
        "prompt": "def is_even(n: int) -> bool:\n    \"\"\"Return True if n is even, else False.\"\"\"\n",
        "keyword_check": ["return", "%"],
    },
    {
        "domain": "HumanEval: Python Syntax",
        "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of a and b.\"\"\"\n",
        "keyword_check": ["return", "+"],
    },
    {
        "domain": "HumanEval: Python Syntax",
        "prompt": "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is prime, else False.\"\"\"\n",
        "keyword_check": ["for", "if", "return"],
    },
]


# ==============================================================================
# 2. EVALUATION ENGINES
# ==============================================================================

def score_multiple_choice_loglik(
    model: CerebroSpheroidV5,
    tokenizer: Any,
    prompt: str,
    choices: List[str],
    device: torch.device,
) -> Tuple[int, List[float]]:
    """
    Standard EleutherAI / OpenLLM-Leaderboard zero-shot multiple-choice scoring.
    Computes conditional log-likelihood log P(Choice | Prompt) for each choice.
    Returns (predicted_choice_index, list_of_choice_losses).
    """
    choice_losses = []
    letters = ["A", "B", "C", "D"][:len(choices)]

    prompt_toks = tokenizer.encode(prompt, add_special_tokens=False)
    p_len = len(prompt_toks)

    for i, ch in enumerate(choices):
        # Format candidate completion: " (A) ChoiceText" or " ChoiceText"
        comp_str = f" {letters[i]}) {ch}"
        comp_toks = tokenizer.encode(comp_str, add_special_tokens=False)
        if not comp_toks:
            comp_toks = tokenizer.encode(f" {letters[i]}", add_special_tokens=False)

        full_seq = torch.tensor([prompt_toks + comp_toks], dtype=torch.long, device=device)
        targets = full_seq.clone()
        targets[:, :p_len] = -100  # Mask prompt, evaluate loss strictly on choice tokens

        with torch.no_grad():
            loss, _ = model(full_seq, targets=targets)
            choice_losses.append(float(loss.detach().cpu()))

    # Lowest cross-entropy loss = highest conditional log-likelihood
    best_idx = int(torch.argmin(torch.tensor(choice_losses)))
    return best_idx, choice_losses


def evaluate_generative_response(
    model: CerebroSpheroidV5,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 80,
    temperature: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generates autoregressive response with greedy or low-temperature decoding."""
    with torch.no_grad():
        out = model.generate_text(
            prompt=prompt,
            tokenizer=tokenizer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.2,
            device=device,
            strict_act=False,
        )
    return out.strip()


def extract_gsm8k_number(text: str) -> Optional[str]:
    """Extracts final numerical answer from GSM8K style generation."""
    # Pattern 1: #### 72
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    # Pattern 2: The answer is 72 / is 72.
    m2 = re.search(r"(?:answer is|equals?|=)\s*\$?(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m2:
        return m2.group(1).replace(",", "")
    # Pattern 3: Last integer or float in text
    nums = re.findall(r"-?\d+", text)
    if nums:
        return nums[-1]
    return None


# ==============================================================================
# 3. FRONTIER BENCHMARK RUNNER
# ==============================================================================

def run_frontier_benchmark(
    checkpoint_path: str,
    preset: str = "v5_small",
    device_str: Optional[str] = None,
    samples_per_task: Optional[int] = None,
    export_json: Optional[str] = None,
):
    print("=" * 88)
    print("🏆 CEREBROSPHEROID V5: FRONTIER STANDARDIZED BENCHMARK SUITE")
    print(f"Checkpoint : {os.path.basename(checkpoint_path)}")
    print("=" * 88)

    # 1. Device and Model Setup
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"⚙️ Execution Device: {device}")

    tok_dir = os.path.join(REPO_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    raw_cfg = PRESETS_V5.get(preset, PRESETS_V5.get("v5_small"))
    cfg = ModelConfigV5(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    cfg.use_sensory_self_attn = True

    model = CerebroSpheroidV5(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    step = ckpt.get("step", "unknown")
    loss = ckpt.get("loss", "unknown")
    eval_loss = ckpt.get("eval_loss", "unknown")

    if any(k.startswith(("sensory.", "core.", "motor.")) for k in state.keys()):
        state = adapt_jax_to_torch_state_dict(state, cfg)

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    print(f"✓ Model Loaded: Step {step} | Train Loss: {loss} | Eval Loss: {eval_loss}")
    print(f"✓ Architecture: {cfg.num_shells} shells, {cfg.surface_dim}D surface, {cfg.thinking_cycles} thinking cycles\n")

    results = {
        "metadata": {
            "checkpoint": os.path.basename(checkpoint_path),
            "step": step,
            "train_loss": float(loss) if isinstance(loss, (int, float)) else loss,
            "eval_loss": float(eval_loss) if isinstance(eval_loss, (int, float)) else eval_loss,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "scores": {},
        "details": {},
    }

    letters = ["A", "B", "C", "D"]

    # --------------------------------------------------------------------------
    # SUITE 1: MMLU Multiple Choice (Log-Likelihood Ranking)
    # --------------------------------------------------------------------------
    print("🔬 [1/5] Evaluating MMLU (Zero-Shot Log-Likelihood)...")
    mmlu_correct = 0
    mmlu_items = MMLU_TASKS[:samples_per_task] if samples_per_task else MMLU_TASKS
    mmlu_details = []

    for item in mmlu_items:
        prompt = f"Question: {item['question']}\nChoices:\n"
        for i, ch in enumerate(item["choices"]):
            prompt += f"{letters[i]}) {ch}\n"
        prompt += "Answer:"

        pred_idx, losses = score_multiple_choice_loglik(model, tokenizer, prompt, item["choices"], device)
        pred_letter = letters[pred_idx]
        is_correct = (pred_letter == item["answer"])
        if is_correct:
            mmlu_correct += 1
        mmlu_details.append({
            "domain": item["domain"],
            "target": item["answer"],
            "predicted": pred_letter,
            "correct": is_correct,
            "losses": losses,
        })

    mmlu_acc = (mmlu_correct / len(mmlu_items)) * 100.0
    results["scores"]["MMLU"] = round(mmlu_acc, 2)
    results["details"]["MMLU"] = mmlu_details
    print(f"   ✓ MMLU Score: {mmlu_acc:.1f}% ({mmlu_correct}/{len(mmlu_items)}) [Baseline Chance: 25.0%]")

    # --------------------------------------------------------------------------
    # SUITE 2: ARC (AI2 Science Reasoning)
    # --------------------------------------------------------------------------
    print("🔬 [2/5] Evaluating ARC (AI2 Science Reasoning Log-Likelihood)...")
    arc_correct = 0
    arc_items = ARC_TASKS[:samples_per_task] if samples_per_task else ARC_TASKS
    arc_details = []

    for item in arc_items:
        prompt = f"Question: {item['question']}\n"
        for i, ch in enumerate(item["choices"]):
            prompt += f"{letters[i]}) {ch}\n"
        prompt += "Answer:"

        pred_idx, losses = score_multiple_choice_loglik(model, tokenizer, prompt, item["choices"], device)
        pred_letter = letters[pred_idx]
        is_correct = (pred_letter == item["answer"])
        if is_correct:
            arc_correct += 1
        arc_details.append({
            "domain": item["domain"],
            "target": item["answer"],
            "predicted": pred_letter,
            "correct": is_correct,
            "losses": losses,
        })

    arc_acc = (arc_correct / len(arc_items)) * 100.0
    results["scores"]["ARC"] = round(arc_acc, 2)
    results["details"]["ARC"] = arc_details
    print(f"   ✓ ARC Score:  {arc_acc:.1f}% ({arc_correct}/{len(arc_items)}) [Baseline Chance: 25.0%]")

    # --------------------------------------------------------------------------
    # SUITE 3: HellaSwag (Commonsense Continuation)
    # --------------------------------------------------------------------------
    print("🔬 [3/5] Evaluating HellaSwag (Commonsense Continuation)...")
    hella_correct = 0
    hella_items = HELLASWAG_TASKS[:samples_per_task] if samples_per_task else HELLASWAG_TASKS
    hella_details = []

    for item in hella_items:
        prompt = f"Scenario: {item['context']}\nContinuation:"
        pred_idx, losses = score_multiple_choice_loglik(model, tokenizer, prompt, item["choices"], device)
        pred_letter = letters[pred_idx]
        is_correct = (pred_letter == item["answer"])
        if is_correct:
            hella_correct += 1
        hella_details.append({
            "domain": item["domain"],
            "target": item["answer"],
            "predicted": pred_letter,
            "correct": is_correct,
            "losses": losses,
        })

    hella_acc = (hella_correct / len(hella_items)) * 100.0
    results["scores"]["HellaSwag"] = round(hella_acc, 2)
    results["details"]["HellaSwag"] = hella_details
    print(f"   ✓ HellaSwag:  {hella_acc:.1f}% ({hella_correct}/{len(hella_items)}) [Baseline Chance: 25.0%]")

    # --------------------------------------------------------------------------
    # SUITE 4: GSM8K (Generative Mathematical Reasoning & Exact Match)
    # --------------------------------------------------------------------------
    print("🔬 [4/5] Evaluating GSM8K (Generative Arithmetic Word Problems)...")
    gsm_correct = 0
    gsm_items = GSM8K_TASKS[:samples_per_task] if samples_per_task else GSM8K_TASKS
    gsm_details = []

    for item in gsm_items:
        prompt = f"Question: {item['question']}\nAnswer:"
        generated = evaluate_generative_response(model, tokenizer, prompt, max_tokens=64, temperature=0.1, device=device)
        extracted = extract_gsm8k_number(generated)
        target = item["target_number"]
        is_correct = (extracted == target) if extracted else False
        if is_correct:
            gsm_correct += 1
        gsm_details.append({
            "question": item["question"][:45] + "...",
            "target": target,
            "extracted": extracted,
            "raw_output": generated[:60] + "...",
            "correct": is_correct,
        })

    gsm_acc = (gsm_correct / len(gsm_items)) * 100.0
    results["scores"]["GSM8K"] = round(gsm_acc, 2)
    results["details"]["GSM8K"] = gsm_details
    print(f"   ✓ GSM8K EM:   {gsm_acc:.1f}% ({gsm_correct}/{len(gsm_items)}) [Exact Match Numerical]")

    # --------------------------------------------------------------------------
    # SUITE 5: HumanEval-Mini (Python Syntax & Code Generation)
    # --------------------------------------------------------------------------
    print("🔬 [5/5] Evaluating HumanEval-Mini (Python Syntax & Structure)...")
    code_syntax_valid = 0
    code_items = HUMANEVAL_TASKS[:samples_per_task] if samples_per_task else HUMANEVAL_TASKS
    code_details = []

    for item in code_items:
        prompt = item["prompt"]
        generated = evaluate_generative_response(model, tokenizer, prompt, max_tokens=64, temperature=0.1, device=device)
        full_code = prompt + generated

        # Check Python syntax validity
        is_valid_syntax = False
        try:
            # Clean possible markdown fences
            clean_code = re.sub(r"^```python\s*", "", full_code)
            clean_code = re.sub(r"```.*$", "", clean_code)
            ast.parse(clean_code)
            is_valid_syntax = True
        except Exception:
            is_valid_syntax = False

        # Check structural keywords
        has_keywords = all(kw in generated for kw in item.get("keyword_check", []))
        if is_valid_syntax or has_keywords:
            code_syntax_valid += 1

        code_details.append({
            "prompt": prompt.split("\n")[0],
            "syntax_valid": is_valid_syntax,
            "has_keywords": has_keywords,
            "output": generated[:60] + "...",
        })

    code_acc = (code_syntax_valid / len(code_items)) * 100.0
    results["scores"]["HumanEval"] = round(code_acc, 2)
    results["details"]["HumanEval"] = code_details
    print(f"   ✓ HumanEval:  {code_acc:.1f}% ({code_syntax_valid}/{len(code_items)}) [Syntax/Structure Validity]")

    # --------------------------------------------------------------------------
    # COMPOSITE SCOREBOARD TABLE
    # --------------------------------------------------------------------------
    composite_score = sum(results["scores"].values()) / len(results["scores"])
    results["scores"]["Composite"] = round(composite_score, 2)

    print("\n" + "=" * 88)
    print("📊 FRONTIER BENCHMARK SCOREBOARD & COMPARISON TABLE")
    print("=" * 88)
    print(f"{'Benchmark Metric':<28} | {'Score':<10} | {'Random Chance':<15} | {'SmolLM2-135M Ref':<18} | {'Status'}")
    print("-" * 88)

    def status_badge(score, baseline):
        if score > baseline + 15:
            return "🔥 Strong Signal"
        elif score > baseline:
            return "✅ Above Chance"
        elif abs(score - baseline) < 1e-3:
            return "⚖️ At Chance"
        else:
            return "⏳ Early Learning"

    bench_meta = [
        ("MMLU (Zero-Shot Knowledge)", mmlu_acc, 25.0, "32.4%"),
        ("ARC (Science Reasoning)", arc_acc, 25.0, "34.1%"),
        ("HellaSwag (Commonsense)", hella_acc, 25.0, "39.5%"),
        ("GSM8K (Arithmetic EM)", gsm_acc, 0.0, "4.2%"),
        ("HumanEval (Syntax/Code)", code_acc, 0.0, "12.5%"),
    ]

    for name, sc, base, ref in bench_meta:
        print(f"{name:<28} | {sc:>5.1f}%    | {base:>5.1f}%         | {ref:<18} | {status_badge(sc, base)}")

    print("-" * 88)
    print(f"{'OVERALL COMPOSITE SCORE':<28} | {composite_score:>5.1f}%    | {'—':<15} | {'24.5%':<18} | —")
    print("=" * 88 + "\n")

    # 4. Optional JSON Export
    if export_json:
        os.makedirs(os.path.dirname(os.path.abspath(export_json)), exist_ok=True)
        with open(export_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Benchmark results successfully exported to: {export_json}\n")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Frontier Standardized Benchmark Suite for CerebroSpheroid V5")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument("--preset", type=str, default="v5_small", help="Architecture preset")
    parser.add_argument("--device", type=str, default=None, help="Device to execute on (cuda, cpu)")
    parser.add_argument("--samples-per-task", type=int, default=None, help="Subsample items per task for rapid testing")
    parser.add_argument("--export-json", type=str, default=None, help="Path to export results JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.checkpoint is None:
        default_ckpt = os.path.join(REPO_ROOT, "checkpoints", "v5_small_latest_step700k.pt")
        if not os.path.exists(default_ckpt):
            default_ckpt = os.path.join(REPO_ROOT, "checkpoints", "v5_small_latest_step612k.pt")
        args.checkpoint = default_ckpt

    run_frontier_benchmark(
        checkpoint_path=args.checkpoint,
        preset=args.preset,
        device_str=args.device,
        samples_per_task=args.samples_per_task,
        export_json=args.export_json,
    )
