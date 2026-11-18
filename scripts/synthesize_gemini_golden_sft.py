#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini Frontier Golden SFT Synthesizer
======================================
Leverages Gemini API (gemini-3.5-flash) to synthesize high-density,
verified reasoning trajectories for fine-tuning Cerebro Spheroid v5:
1. Multi-step Chain-of-Thought (<think> ... </think>)
2. Deterministic Tool Calls (<act> calc(...) </act> / <act> python(...) </act>)
3. Closed-Loop ReAct (Turn 1: Act -> Sandbox Observation -> Turn 2: Verified Conclusion)
4. Factual Knowledge (crisp, direct answers without unnecessary thinking)
"""

import os
import sys
import json
import time
import random
from typing import List, Dict, Any

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agentic_harness import execute_agentic_tool


def get_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    bashrc_path = os.path.expanduser("~/.bashrc")
    if os.path.exists(bashrc_path):
        with open(bashrc_path, "r", encoding="utf-8") as f:
            for line in f:
                if "export GEMINI_API_KEY=" in line:
                    return line.split("export GEMINI_API_KEY=")[1].strip().strip("\"'")
    raise RuntimeError("GEMINI_API_KEY not found in environment or ~/.bashrc")


def init_gemini_client():
    from google import genai
    key = get_gemini_api_key()
    return genai.Client(api_key=key)


PROMPT_TEMPLATES = [
    # 1. Multi-step word problems requiring calc
    """Generate {num_samples} distinct multi-step arithmetic word problems (e.g. inventory, split bills, rate/time, discounts).
For each, provide:
- "prompt": The problem statement.
- "category": "word_problem_arithmetic"
- "response": Start with brief deliberative thinking inside <think> ... </think>, then emit a single tool call inside <act> calc(...) </act>.
Format strictly as a JSON list of objects with keys "prompt", "category", "response".""",

    # 2. Complex numerical comparisons or algorithms requiring python
    """Generate {num_samples} distinct questions requiring numerical comparison, min/max finding, string manipulation, or conditional logic.
For each, provide:
- "prompt": The question.
- "category": "algorithmic_logic"
- "response": Brief deliberation inside <think> ... </think>, then emit a single tool call inside <act> python(code='...') </act> that prints the result.
Format strictly as a JSON list of objects with keys "prompt", "category", "response".""",

    # 3. Full closed-loop ReAct demonstrations
    """Generate {num_samples} distinct complete closed-loop problem-solving demonstrations.
For each, provide:
- "prompt": The user question.
- "category": "closed_loop_react"
- "action": <think> Thought on what to compute </think> <act> calc(...) or python(code='...') </act>
- "observation": The exact computed output from the tool
- "conclusion": <think> Verification of observation </think> The final answer is ...
Format strictly as a JSON list of objects with keys "prompt", "category", "action", "observation", "conclusion".""",

    # 4. Factual / direct QA (to anchor representations and prevent overthinking)
    """Generate {num_samples} concise factual QA pairs across science, geography, history, and computing.
For each, provide:
- "prompt": The user question.
- "category": "direct_factual"
- "response": A direct, accurate answer (no tools, no fake mental math, 1-2 sentences).
Format strictly as a JSON list of objects with keys "prompt", "category", "response"."""
]


def synthesize_batch(client, template_idx: int, num_samples: int = 25, max_retries: int = 5) -> List[Dict[str, Any]]:
    prompt = PROMPT_TEMPLATES[template_idx].format(num_samples=num_samples)
    models = ["gemini-3.5-flash-lite", "gemini-flash-latest"]
    
    for attempt in range(max_retries):
        model_name = models[attempt % len(models)]
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            raw = resp.text.strip()
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, list):
                        return val
        except Exception as e:
            err_msg = str(e)
            delay = 16.0
            if "retryDelay" in err_msg:
                try:
                    import re
                    match = re.search(r"retryDelay':\s*'(\d+)", err_msg)
                    if match:
                        delay = float(match.group(1)) + 1.0
                except Exception:
                    pass
            print(f"\n  ⏳ Rate limit on {model_name} (Attempt {attempt+1}/{max_retries}). Backing off for {delay:.1f}s...")
            time.sleep(delay)
    return []


def verify_and_format_sample(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Verifies tool calls using local sandbox and standardizes output format."""
    cat = item.get("category", "general")
    formatted = []

    if cat == "closed_loop_react":
        prompt = item.get("prompt", "").strip()
        action = item.get("action", "").strip()
        conclusion = item.get("conclusion", "").strip()
        
        # Verify action
        if "<act>" in action:
            act_str = action.split("<act>")[1].split("</act>")[0].strip()
            res = execute_agentic_tool(act_str)
            if res is not None:
                # Turn 1: Tool invocation
                formatted.append({
                    "prompt": prompt,
                    "response": action,
                    "category": "agentic_tool_invoke"
                })
                # Turn 2: Multi-turn continuation with observation
                full_turn_prompt = f"{prompt}\n{action}\n<obs> {res} </obs>"
                formatted.append({
                    "prompt": full_turn_prompt,
                    "response": conclusion,
                    "category": "agentic_react_completion"
                })
    else:
        prompt = item.get("prompt", "").strip()
        resp = item.get("response", "").strip()
        if not prompt or not resp:
            return []

        if "<act>" in resp:
            act_str = resp.split("<act>")[1].split("</act>")[0].strip()
            res = execute_agentic_tool(act_str)
            if res is None:
                return []  # Discard unverified or malformed tool calls
            # Valid tool call
            formatted.append({
                "prompt": prompt,
                "response": resp,
                "category": cat
            })
            # Also synthesize the direct resolution turn
            formatted.append({
                "prompt": f"{prompt}\n{resp}\n<obs> {res} </obs>",
                "response": f"<think> The tool verified the result as {res}. </think>\nTherefore, the answer is {res}.",
                "category": "react_verified_resolution"
            })
        else:
            # Factual slot
            formatted.append({
                "prompt": prompt,
                "response": resp,
                "category": cat
            })

    return formatted


def run_synthesis(target_samples: int = 1200, output_file: str = "data/gemini_golden_reasoning_sft.jsonl"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    client = init_gemini_client()
    print(f"🚀 [GEMINI DISTILLER] Connected to Gemini 3.5 Flash API.")
    print(f"🎯 Target: {target_samples} golden reasoning samples -> {output_file}")

    total_valid = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        batch_id = 0
        while total_valid < target_samples:
            template_idx = batch_id % len(PROMPT_TEMPLATES)
            batch = synthesize_batch(client, template_idx, num_samples=15)
            
            for item in batch:
                valid_items = verify_and_format_sample(item)
                for v in valid_items:
                    out_f.write(json.dumps(v, ensure_ascii=False) + "\n")
                    total_valid += 1
                    if total_valid >= target_samples:
                        break
                if total_valid >= target_samples:
                    break

            batch_id += 1
            print(f"  ✨ Progress: {total_valid:,}/{target_samples:,} verified samples written (Batch {batch_id})...", flush=True)
            out_f.flush()
            time.sleep(14.0)  # Free Tier: Stay below 5 RPM

    print(f"\n✅ [SYNTHESIS COMPLETE] {total_valid:,} golden verified samples written to {output_file}")


if __name__ == "__main__":
    count = 1000
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    run_synthesis(target_samples=count)
