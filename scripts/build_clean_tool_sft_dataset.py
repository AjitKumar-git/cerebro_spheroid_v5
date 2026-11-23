#!/usr/bin/env python3
"""
🛠️ CerebroSpheroid V5: Closed-Loop Agentic SFT Dataset Generator
================================================================
Generates 5,000 high-density, multi-turn closed-loop trajectories with verified
tool executions and grounded final answers:
  1. Arithmetic & Symbolic Math (40% - 2,000 samples)
  2. Physics & Chemistry Constants / Geography (30% - 1,500 samples)
  3. Python Code Execution & Algorithms (20% - 1,000 samples)
  4. Conversational Direct Queries (10% - 500 samples)

Outputs:
  - closed_loop_trajectories.jsonl : Multi-turn conversation format
  - closed_loop_sft_pairs.jsonl    : Teacher-forced (input, output) training pairs
                                     including both Stage 1 (Tool Call) and
                                     Stage 2 (Tool Response -> Grounded Answer)
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.tools.unified_tool_dispatcher import UnifiedToolDispatcher

random.seed(42)
dispatcher = UnifiedToolDispatcher()


def generate_arithmetic_sample() -> Dict[str, Any]:
    """Generates complex arithmetic problems with verified calculator calls."""
    pattern = random.choice([
        "mult_add", "sub_mult", "power_sub", "sqrt_div", "poly", "bracket_div", "percentage", "trig"
    ])

    if pattern == "mult_add":
        a, b, c = random.randint(11, 89), random.randint(11, 49), random.randint(15, 250)
        expr = f"({a} * {b}) + {c}"
        prompt = random.choice([
            f"What is ({a} * {b}) + {c}?",
            f"Calculate ({a} * {b}) + {c}",
            f"Evaluate ({a} * {b}) + {c}.",
        ])
    elif pattern == "percentage":
        pct = random.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 75])
        val = random.randint(100, 2500)
        expr = f"({pct} / 100) * {val}"
        prompt = random.choice([
            f"What is {pct}% of {val}?",
            f"Calculate {pct}% of {val}.",
            f"Find {pct} percent of {val}."
        ])
    elif pattern == "trig":
        deg = random.choice([0, 30, 45, 60])
        expr = f"round(math.sin(math.radians({deg})), 4)"
        prompt = f"Calculate sin({deg} degrees)"
    elif pattern == "sub_mult":
        a, b, c, d = random.randint(30, 99), random.randint(5, 25), random.randint(20, 60), random.randint(5, 15)
        expr = f"({a} - {b}) * ({c} + {d})"
        prompt = f"Calculate ({a} - {b}) * ({c} + {d})"
    elif pattern == "power_sub":
        p = random.choice([8, 9, 10, 11, 12])
        sub = random.randint(10, 99)
        expr = f"2**{p} - {sub}"
        prompt = f"What is 2^{p} - {sub}?"
    elif pattern == "sqrt_div":
        root = random.randint(15, 65)
        square = root * root
        div = random.choice([2, 3, 5])
        if root % div == 0:
            expr = f"int(math.sqrt({square})) // {div}"
            prompt = f"Calculate sqrt({square}) / {div}"
        else:
            expr = f"math.sqrt({square})"
            prompt = f"What is the square root of {square}?"
    elif pattern == "poly":
        x = random.randint(3, 15)
        expr = f"3 * ({x}**2) + (5 * {x}) - 12"
        prompt = f"Evaluate the polynomial 3*x^2 + 5*x - 12 when x = {x}"
    else:
        a, b = random.randint(500, 2500), random.randint(12, 48)
        expr = f"{a} * {b}"
        prompt = f"Compute {a} * {b}"

    # Verify execution via dispatcher
    raw_res = dispatcher.dispatch_call("calculator", {"expression": expr})
    result = str(raw_res).strip()

    call_str = f"calc({expr})"
    think1 = f"<think>Calculate {expr}.</think>"
    step1_out = f"{think1}<tool_call>{call_str}</tool_call>"

    step2_out = f"The result is {result}.</s>"

    return {
        "category": "arithmetic",
        "prompt": prompt,
        "tool_call": call_str,
        "tool_response": result,
        "step1_output": step1_out,
        "step2_output": step2_out,
        "final_answer": f"The result is {result}.",
    }


def generate_factual_sample() -> Dict[str, Any]:
    """Generates physical, chemical, and geographical queries with facts tool."""
    facts_type = random.choice(["constant", "element", "capital"])

    if facts_type == "constant":
        constants = [
            ("speed of light in vacuum", "c", "299,792,458 m/s"),
            ("Planck's constant", "h", "6.62607015e-34 J*s"),
            ("gravitational constant", "G", "6.67430e-11 N*m^2/kg^2"),
            ("elementary charge", "e", "1.602176634e-19 C"),
            ("Avogadro constant", "N_A", "6.02214076e23 mol^-1"),
            ("Boltzmann constant", "k_B", "1.380649e-23 J/K"),
            ("gas constant", "R", "8.314462618 J/(mol*K)"),
        ]
        q_text, sym, val = random.choice(constants)
        prompt = f"What is the {q_text}?"
        call_obj = {"name": "facts", "arguments": {"query": q_text}}
        call_json = json.dumps(call_obj)
        tool_res = dispatcher.dispatch_call("facts", {"query": q_text})
        answer_text = f"The {q_text} ({sym}) is {val}."

    elif facts_type == "element":
        elements = [
            ("Hydrogen", "H", 1, 1.008),
            ("Helium", "He", 2, 4.0026),
            ("Carbon", "C", 6, 12.011),
            ("Nitrogen", "N", 7, 14.007),
            ("Oxygen", "O", 8, 15.999),
            ("Sodium", "Na", 11, 22.990),
            ("Iron", "Fe", 26, 55.845),
            ("Copper", "Cu", 29, 63.546),
            ("Gold", "Au", 79, 196.97),
            ("Silver", "Ag", 47, 107.87),
            ("Lead", "Pb", 82, 207.2),
            ("Uranium", "U", 92, 238.03),
        ]
        elem, sym, num, mass = random.choice(elements)
        attr = random.choice(["atomic number", "symbol", "atomic mass"])
        if attr == "atomic number":
            prompt = f"What is the atomic number of {elem}?"
            answer_text = f"The atomic number of {elem} ({sym}) is {num}."
        elif attr == "symbol":
            prompt = f"What is the chemical symbol for {elem}?"
            answer_text = f"The chemical symbol for {elem} is {sym}."
        else:
            prompt = f"What is the atomic mass of {elem}?"
            answer_text = f"The atomic mass of {elem} is {mass} u."

        call_obj = {"name": "facts", "arguments": {"query": f"{attr} of {elem}"}}
        call_json = json.dumps(call_obj)
        tool_res = dispatcher.dispatch_call("facts", {"query": f"{attr} of {elem}"})

    else:
        capitals = [
            ("France", "Paris"),
            ("Japan", "Tokyo"),
            ("Germany", "Berlin"),
            ("United Kingdom", "London"),
            ("Italy", "Rome"),
            ("Spain", "Madrid"),
            ("Canada", "Ottawa"),
            ("Australia", "Canberra"),
            ("India", "New Delhi"),
            ("Brazil", "Brasilia"),
            ("South Korea", "Seoul"),
            ("Egypt", "Cairo"),
        ]
        country, cap = random.choice(capitals)
        prompt = f"What is the capital of {country}?"
        call_obj = {"name": "facts", "arguments": {"query": f"capital of {country}"}}
        call_json = json.dumps(call_obj)
        tool_res = dispatcher.dispatch_call("facts", {"query": f"capital of {country}"})
        answer_text = f"The capital of {country} is {cap}."

    q_val = call_obj['arguments']['query']
    call_str = f'facts("{q_val}")'
    think1 = f"<think>Query {q_val}.</think>"
    step1_out = f"{think1}<tool_call>{call_str}</tool_call>"

    step2_out = f"{answer_text}</s>"

    return {
        "category": "factual",
        "prompt": prompt,
        "tool_call": call_str,
        "tool_response": str(tool_res),
        "step1_output": step1_out,
        "step2_output": step2_out,
        "final_answer": answer_text,
    }


def generate_python_sample() -> Dict[str, Any]:
    """Generates Python algorithmic problems with verified REPL execution."""
    tasks = ["prime_count", "string_reverse", "sum_squares", "factorial", "palindrome"]
    task = random.choice(tasks)

    if task == "prime_count":
        limit = random.choice([20, 30, 50, 100])
        code = f"print(len([n for n in range(2, {limit}) if all(n % d != 0 for d in range(2, int(n**0.5) + 1))]))"
        prompt = f"How many primes are strictly below {limit} in Python?"
        call_obj = {"name": "python", "arguments": {"code": code}}
        tool_res = dispatcher.dispatch_call("python", {"code": code}).strip()
        answer_text = f"There are {tool_res} prime numbers strictly below {limit}."

    elif task == "string_reverse":
        word = random.choice(["cerebro", "spheroid", "algorithmic", "quantum", "neural", "tensor", "matrix"])
        code = f"print('{word}'[::-1])"
        prompt = f"Reverse the word '{word}' in Python"
        call_obj = {"name": "python", "arguments": {"code": code}}
        tool_res = dispatcher.dispatch_call("python", {"code": code}).strip()
        answer_text = f"The reverse of '{word}' is '{tool_res}'."

    elif task == "sum_squares":
        n = random.choice([5, 8, 10, 12, 15])
        code = f"print(sum(i**2 for i in range(1, {n+1})))"
        prompt = f"Calculate the sum of squares from 1 to {n} in Python"
        call_obj = {"name": "python", "arguments": {"code": code}}
        tool_res = dispatcher.dispatch_call("python", {"code": code}).strip()
        answer_text = f"The sum of squares from 1 to {n} is {tool_res}."

    elif task == "factorial":
        n = random.choice([5, 6, 7, 8, 9])
        code = f"import math; print(math.factorial({n}))"
        prompt = f"Compute {n} factorial ({n}!) in Python"
        call_obj = {"name": "python", "arguments": {"code": code}}
        tool_res = dispatcher.dispatch_call("python", {"code": code}).strip()
        answer_text = f"{n}! is {tool_res}."

    else:
        word = random.choice(["racecar", "level", "deified", "python", "neural"])
        code = f"w = '{word}'; print(w == w[::-1])"
        prompt = f"Check if the word '{word}' is a palindrome in Python"
        call_obj = {"name": "python", "arguments": {"code": code}}
        tool_res = dispatcher.dispatch_call("python", {"code": code}).strip()
        is_p = tool_res == "True"
        answer_text = f"The word '{word}' {'is' if is_p else 'is not'} a palindrome."

    call_str = f"python({code})"
    think1 = f"<think>Execute code in python.</think>"
    step1_out = f"{think1}<tool_call>{call_str}</tool_call>"

    step2_out = f"{answer_text}</s>"

    return {
        "category": "python",
        "prompt": prompt,
        "tool_call": call_str,
        "tool_response": str(tool_res),
        "step1_output": step1_out,
        "step2_output": step2_out,
        "final_answer": answer_text,
    }


def generate_conversational_sample() -> Dict[str, Any]:
    """Generates direct conversational responses that must NOT invoke tools."""
    dialogues = [
        (
            "Hello, who are you?",
            "I am Doty, an autonomous reasoning model built on the Cerebro Spheroid architecture by Ajit Kumar. How can I assist you today?"
        ),
        (
            "What architecture do you use?",
            "I am powered by the Cerebro Spheroid V5 architecture, which features 16 concentric spherical shells, 40 axonal shortcut tracts, and Arcuate Fasciculus cross-resonance heads."
        ),
        (
            "What is photosynthesis?",
            "Photosynthesis is the biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy, synthesizing glucose from carbon dioxide and water while releasing oxygen."
        ),
        (
            "What is the difference between supervised and unsupervised learning?",
            "Supervised learning trains models on labeled input-output pairs to learn a mapping function, whereas unsupervised learning discovers hidden patterns, clusters, or representations in unlabeled data without explicit targets."
        ),
        (
            "Can you explain Newton's First Law of Motion?",
            "Newton's First Law of Motion states that an object at rest will remain at rest, and an object in motion will continue in motion with a constant velocity, unless acted upon by a net external force."
        ),
    ]
    prompt, ans = random.choice(dialogues)
    think = f"<think>\nThis is a conversational/conceptual query. No computational or factual lookup tool is required.\n</think>"
    out = f"{think}\n{ans}"
    return {
        "category": "conversational",
        "prompt": prompt,
        "tool_call": None,
        "tool_response": None,
        "step1_output": out,
        "step2_output": None,
        "final_answer": ans,
    }


def main():
    target_dir = os.path.join(WORKSPACE_ROOT, "data", "consolidated", "02_agentic_tools")
    os.makedirs(target_dir, exist_ok=True)

    traj_path = os.path.join(target_dir, "closed_loop_trajectories.jsonl")
    pairs_path = os.path.join(target_dir, "closed_loop_sft_pairs.jsonl")

    total_samples = 10000
    n_arith = 4000
    n_facts = 3000
    n_python = 2000
    n_conv = 1000

    print(f"\n{'='*80}")
    print(f"🛠️ GENERATING {total_samples:,} CLOSED-LOOP AGENTIC SFT TRAJECTORIES")
    print(f"Target Directory: {target_dir}")
    print(f"Distribution: {n_arith} Arithmetic | {n_facts} Facts | {n_python} Python | {n_conv} Conversational")
    print(f"{'='*80}\n")

    trajectories = []
    # 1. Arithmetic
    print("Generating arithmetic samples...")
    for _ in range(n_arith):
        trajectories.append(generate_arithmetic_sample())

    # 2. Facts
    print("Generating factual samples...")
    for _ in range(n_facts):
        trajectories.append(generate_factual_sample())

    # 3. Python
    print("Generating python samples...")
    for _ in range(n_python):
        trajectories.append(generate_python_sample())

    # 4. Conversational
    print("Generating conversational samples...")
    for _ in range(n_conv):
        trajectories.append(generate_conversational_sample())

    random.shuffle(trajectories)

    # Write multi-turn trajectory format
    with open(traj_path, "w", encoding="utf-8") as f:
        for idx, t in enumerate(trajectories, 1):
            t["id"] = f"traj_{idx:05d}"
            f.write(json.dumps(t) + "\n")
    print(f"✓ Saved {len(trajectories):,} trajectories to {traj_path}")

    # Generate teacher-forcing training pairs:
    # Native ChatML format aligned with tokenizer special tokens (IDs 27 & 28)
    training_pairs = []
    for t in trajectories:
        cat = t["category"]
        p = t["prompt"]
        base_inp = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"

        if cat == "conversational":
            clean_out = t["step1_output"].replace("</s>", "").strip()
            training_pairs.append({
                "type": "conversational",
                "category": cat,
                "input": base_inp,
                "output": f"{clean_out}<|im_end|>",
            })
        else:
            # Stage 1: Learn to emit <tool_call>
            clean_s1 = t["step1_output"].replace("</s>", "").strip()
            training_pairs.append({
                "type": "stage1_tool_call",
                "category": cat,
                "input": base_inp,
                "output": f"{clean_s1}<|im_end|>",
            })
            # Stage 2: Learn to assimilate <tool_response> and emit grounded final answer
            stage2_inp = f"{base_inp}{clean_s1}<|im_end|>\n<tool_response> {t['tool_response']} </tool_response>\n<|im_start|>assistant\n"
            clean_s2 = t["step2_output"].replace("</s>", "").strip()
            training_pairs.append({
                "type": "stage2_grounded_answer",
                "category": cat,
                "input": stage2_inp,
                "output": f"{clean_s2}<|im_end|>",
            })

    random.shuffle(training_pairs)
    with open(pairs_path, "w", encoding="utf-8") as f:
        for pair in training_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"✓ Saved {len(training_pairs):,} paired training samples to {pairs_path}")
    print(f"\n🎉 Dataset Generation Complete!\n")


if __name__ == "__main__":
    main()
