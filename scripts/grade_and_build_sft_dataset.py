#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Model Resilient Reasoning & Tool Calling Dataset Synthesizer
==================================================================
Uses a distributed pool of Gemini & Gemma models with intelligent rate-limit
pacing and automatic 503/429 failover to:
1. Synthesize & grade candidate prompts for mathematical, algorithmic, and scientific accuracy.
2. Filter for high-accuracy instances (grade >= 8.5/10).
3. Align outputs strictly with CerebroSpheroid V5 cognitive tokens:
   - Deliberation scratchpad: `<think> ... </think>`
   - External tool actions:   `<act> calc(...) </act>` or `<act> python(code='...') </act>`
   - Environment feedback:    `<obs> ... </obs>`
   - Final plain text answer: Outside `<act>`
4. Closed-loop sandbox verification: all tool outputs are executed locally to prevent hallucinations.
"""

import os
import sys
import json
import time
import re
import argparse
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agentic_harness import execute_agentic_tool


def get_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    bashrc_path = os.path.expanduser("~/.bashrc")
    if os.path.exists(bashrc_path):
        with open(bashrc_path, "r", encoding="utf-8") as f:
            for line in f:
                if "export GEMINI_API_KEY=" in line:
                    return line.split("export GEMINI_API_KEY=")[1].strip().strip("\"'")
                if "export GOOGLE_API_KEY=" in line:
                    return line.split("export GOOGLE_API_KEY=")[1].strip().strip("\"'")
    raise RuntimeError("GEMINI_API_KEY not found in environment or ~/.bashrc")


class MultiModelDispatcher:
    """Manages a pool of models with cooldowns, failover, and rate-limiting."""

    def __init__(self, models: List[str], min_request_interval: float = 3.5):
        from google import genai
        self.client = genai.Client(api_key=get_gemini_api_key())
        self.models = models
        self.cooldowns: Dict[str, float] = {m: 0.0 for m in models}
        self.model_success_count: Dict[str, int] = {m: 0 for m in models}
        self.min_request_interval = min_request_interval
        self.last_call_time = 0.0
        self.cursor = 0

    def _get_next_available_model(self) -> str:
        now = time.time()
        for i in range(len(self.models)):
            idx = (self.cursor + i) % len(self.models)
            m = self.models[idx]
            if now >= self.cooldowns[m]:
                self.cursor = (idx + 1) % len(self.models)
                return m
        # If all in cooldown, wait until earliest cooldown expires
        earliest = min(self.cooldowns.values())
        wait_s = max(0.5, earliest - now)
        time.sleep(wait_s)
        return self.models[0]

    def generate(self, instruction: str, max_retries: int = 6) -> List[Dict[str, Any]]:
        for attempt in range(max_retries):
            # Enforce rate limit pacing
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)

            model_name = self._get_next_available_model()
            self.last_call_time = time.time()

            try:
                resp = self.client.models.generate_content(
                    model=model_name,
                    contents=instruction,
                    config={"response_mime_type": "application/json"}
                )
                raw = resp.text.strip()
                data = json.loads(raw)
                self.model_success_count[model_name] += 1
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    for v in data.values():
                        if isinstance(v, list):
                            return v
            except Exception as e:
                err = str(e)
                now = time.time()
                if "503" in err or "crowded" in err.lower():
                    # 503 Overloaded: Put model on 30s cooldown
                    self.cooldowns[model_name] = now + 30.0
                    print(f"  ⏳ [{model_name}] 503 Crowded. Cooldown 30s -> Failover to next model.")
                elif "429" in err or "quota" in err.lower():
                    # 429 Rate limit: Put model on 45s cooldown
                    self.cooldowns[model_name] = now + 45.0
                    print(f"  ⏳ [{model_name}] 429 Rate limited. Cooldown 45s -> Failover to next model.")
                else:
                    self.cooldowns[model_name] = now + 15.0
                    print(f"  ⚠️ [{model_name}] Error: {err[:60]}... Failing over.")
                time.sleep(2.0)

        return []


PROMPT_DOMAINS = [
    {
        "domain": "arithmetic_word_problem",
        "description": "Multi-step arithmetic word problems requiring careful calculation.",
        "instruction": """Generate {num_samples} challenging multi-step arithmetic word problems (e.g. compound percentages, currency exchange, rate/distance/work, inventory depreciation).
For each problem, evaluate its clarity and solvability, then provide:
- "prompt": The word problem statement.
- "accuracy_score": Numeric score from 1.0 to 10.0 on whether the problem is mathematically sound and unambiguous.
- "thinking": A step-by-step reasoning plan.
- "tool_call": A single calculator call formatted as calc(expression), e.g. calc(150 * (1 - 0.20) / 3).
- "final_answer": The final natural language response stating the result clearly.

Format strictly as a JSON list of objects with keys: "prompt", "accuracy_score", "thinking", "tool_call", "final_answer"."""
    },
    {
        "domain": "algorithmic_code_execution",
        "description": "Algorithmic logic, string manipulations, and data structures requiring Python.",
        "instruction": """Generate {num_samples} algorithmic problem statements (e.g. prime factorization, string anagrams, sorting/filtering arrays, palindrome detection, combinatorics).
For each problem, evaluate its accuracy and provide:
- "prompt": The problem statement.
- "accuracy_score": Numeric score from 1.0 to 10.0.
- "thinking": Deliberation on the algorithm and edge cases.
- "tool_call": A Python execution call formatted as python(code='...') that prints the exact answer.
- "final_answer": Concise text explaining the algorithmic solution with the computed value.

Format strictly as a JSON list of objects with keys: "prompt", "accuracy_score", "thinking", "tool_call", "final_answer"."""
    },
    {
        "domain": "closed_loop_react",
        "description": "Two-turn ReAct problem solving with tool execution and observation feedback.",
        "instruction": """Generate {num_samples} multi-stage problem solving scenarios requiring an intermediate tool calculation before reaching the final conclusion.
For each problem, evaluate its soundness and provide:
- "prompt": The question.
- "accuracy_score": Numeric score from 1.0 to 10.0.
- "initial_thinking": Internal scratchpad on what needs to be calculated first.
- "tool_call": calc(...) or python(...) invocation.
- "post_tool_thinking": Scratchpad reflecting on the tool observation.
- "final_answer": The verified comprehensive conclusion.

Format strictly as a JSON list of objects with keys: "prompt", "accuracy_score", "initial_thinking", "tool_call", "post_tool_thinking", "final_answer"."""
    },
    {
        "domain": "deductive_science_reasoning",
        "description": "Pure step-by-step scientific deduction without external tools.",
        "instruction": """Generate {num_samples} rigorous multi-step deduction questions across biology, chemistry, physics, or formal logic.
For each question, evaluate its factual precision and provide:
- "prompt": The scientific or logical question.
- "accuracy_score": Numeric score from 1.0 to 10.0.
- "thinking": Deep step-by-step deduction breaking down the mechanism or syllogism.
- "final_answer": The clear, authoritative scientific conclusion (no tool call needed).

Format strictly as a JSON list of objects with keys: "prompt", "accuracy_score", "thinking", "final_answer"."""
    }
]


def build_and_verify_sample(raw_item: Dict[str, Any], domain: str, min_accuracy: float = 8.5) -> List[Dict[str, Any]]:
    """
    Grades prompt accuracy, executes tool in local sandbox, and formats with
    CerebroSpheroid <think> and <act> tokens.
    """
    acc_score = float(raw_item.get("accuracy_score", 0.0))
    if acc_score < min_accuracy:
        return []

    prompt = raw_item.get("prompt", "").strip()
    if not prompt:
        return []

    verified_samples = []

    # Case 1: Closed-Loop ReAct
    if domain == "closed_loop_react":
        init_think = raw_item.get("initial_thinking", "").strip()
        tool_call = raw_item.get("tool_call", "").strip()
        post_think = raw_item.get("post_tool_thinking", "").strip()
        final_ans = raw_item.get("final_answer", "").strip()

        if tool_call:
            obs = execute_agentic_tool(tool_call)
            if obs is not None and not str(obs).startswith("["):
                clean_obs = str(obs).strip()
                try:
                    val = float(clean_obs)
                    clean_obs = f"{val:.4f}".rstrip("0").rstrip(".") if "." in f"{val:.4f}" else str(int(val))
                except Exception:
                    pass

                # Turn 1: Thought + Action (<act>)
                t1_response = f"<think>\n{init_think}\n</think>\n<act> {tool_call} </act>"
                verified_samples.append({
                    "prompt": prompt,
                    "response": t1_response,
                    "category": "closed_loop_react_act",
                    "accuracy_score": acc_score,
                    "domain": domain
                })

                # Turn 2: Thought + Observation + Final Answer outside <act>
                t2_prompt = f"{prompt}\n{t1_response}\n<obs> {clean_obs} </obs>"
                t2_response = f"<think>\n{post_think}\nThe sandbox verified computation returned {clean_obs}.\n</think>\nTherefore, the final result is {clean_obs}."
                verified_samples.append({
                    "prompt": t2_prompt,
                    "response": t2_response,
                    "category": "closed_loop_react_resolve",
                    "accuracy_score": acc_score,
                    "domain": domain
                })
        else:
            response = f"<think>\n{init_think}\n</think>\n{final_ans}"
            verified_samples.append({
                "prompt": prompt,
                "response": response,
                "category": "reasoning_deduction",
                "accuracy_score": acc_score,
                "domain": domain
            })

    # Case 2: Tool Calling (Arithmetic / Algorithmic)
    elif "tool_call" in raw_item and raw_item["tool_call"]:
        tool_call = raw_item.get("tool_call", "").strip()
        thinking = raw_item.get("thinking", "").strip()
        final_ans = raw_item.get("final_answer", "").strip()

        obs = execute_agentic_tool(tool_call)
        if obs is not None and not str(obs).startswith("["):
            clean_obs = str(obs).strip()
            response = f"<think>\n{thinking}\n</think>\n<act> {tool_call} </act>\n{final_ans}"
            verified_samples.append({
                "prompt": prompt,
                "response": response,
                "category": f"tool_{domain}",
                "accuracy_score": acc_score,
                "domain": domain
            })

    # Case 3: Pure Step-by-Step Reasoning (Deductive Science)
    else:
        thinking = raw_item.get("thinking", "").strip()
        final_ans = raw_item.get("final_answer", "").strip()
        if thinking and final_ans:
            response = f"<think>\n{thinking}\n</think>\n{final_ans}"
            verified_samples.append({
                "prompt": prompt,
                "response": response,
                "category": "reasoning_deductive_science",
                "accuracy_score": acc_score,
                "domain": domain
            })

    return verified_samples


def run_multi_model_pipeline(
    target_count: int = 100,
    output_path: str = "data/gemma31_graded_reasoning_sft.jsonl",
    min_accuracy: float = 8.5,
    custom_models: Optional[List[str]] = None
):
    print("=" * 80)
    print("🧠 CerebroSpheroid V5: Multi-Model Reasoning & Tool Calling Synthesizer")
    print("=" * 80)

    # Balanced pool: Fast & uncrowded flash-lite models + flagship models when available
    default_pool = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite-preview",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash",
        "gemma-4-31b-it",
    ]
    model_pool = custom_models or default_pool

    dispatcher = MultiModelDispatcher(models=model_pool, min_request_interval=3.5)
    print("✓ Successfully authenticated with Gemini API.")
    print(f"🌐 Active Model Pool  : {', '.join(model_pool)}")
    print(f"🎯 Target Samples     : {target_count}")
    print(f"🎯 Minimum Grade Cutoff: {min_accuracy}/10.0 accuracy")
    print(f"💾 Output File        : {output_path}\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    verified_total = 0
    domain_counts = {}
    batch_idx = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        while verified_total < target_count:
            d_spec = PROMPT_DOMAINS[batch_idx % len(PROMPT_DOMAINS)]
            domain_name = d_spec["domain"]
            num_req = min(12, max(4, target_count - verified_total))
            instruction = d_spec["instruction"].format(num_samples=num_req)

            print(f"[{batch_idx+1}] Generating & grading '{domain_name}' candidates across model pool...")
            raw_candidates = dispatcher.generate(instruction)

            accepted_batch = 0
            for item in raw_candidates:
                formatted_list = build_and_verify_sample(item, domain=domain_name, min_accuracy=min_accuracy)
                for f_sample in formatted_list:
                    out_f.write(json.dumps(f_sample, ensure_ascii=False) + "\n")
                    verified_total += 1
                    accepted_batch += 1
                    domain_counts[domain_name] = domain_counts.get(domain_name, 0) + 1
                    if verified_total >= target_count:
                        break
                if verified_total >= target_count:
                    break

            out_f.flush()
            print(f"   -> Accepted & Verified: {accepted_batch} samples (Total: {verified_total}/{target_count})")
            batch_idx += 1

    print("\n" + "=" * 80)
    print("🎉 Pipeline Complete! High-Quality Dataset Built.")
    print("=" * 80)
    print(f"Total Verified Samples: {verified_total}")
    for d, c in domain_counts.items():
        print(f"  • {d:<30}: {c} samples")
    print("\nModel Contribution Breakdown:")
    for m, c in dispatcher.model_success_count.items():
        print(f"  • {m:<32}: {c} successful batches")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Reasoning & Tool Calling Synthesizer")
    parser.add_argument("--count", type=int, default=100, help="Number of verified samples (default: 100)")
    parser.add_argument("--min-accuracy", type=float, default=8.5, help="Minimum accuracy grade cutoff (default: 8.5)")
    parser.add_argument("--output", type=str, default="data/gemma31_graded_reasoning_sft.jsonl", help="Output path")
    args = parser.parse_args()

    run_multi_model_pipeline(target_count=args.count, output_path=args.output, min_accuracy=args.min_accuracy)
