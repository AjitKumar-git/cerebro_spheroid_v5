#!/usr/bin/env python3
"""
🔑 Long-Context Retrieval Curriculum Data Generator for CerebroSpheroid V5
========================================================================
Generates a comprehensive dataset for length curriculum training:
1. Passkey Needle-in-a-Haystack (lengths: 128, 256, 512, 1024, 2048, 4096; depths: 10%, 25%, 50%, 75%, 90%)
2. Multi-Hop Graph Traversal (A -> B -> C -> Target) across variable context horizons
3. Lost-in-the-Middle Factual Needle Retrieval with CoT
4. Long-Context Tool Calling (extracting parameters across 500-2000 tokens for <act> calc/python </act>)

Enables CerebroSpheroid V5 to develop continuous contextual attention routing
across 256 -> 512 -> 1k -> 2k -> 4k token horizons.
"""

import os
import sys
import json
import random
import argparse
from typing import List, Dict, Any, Tuple

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from transformers import AutoTokenizer
from scripts.eval_passkey_retrieval import FILLER_SENTENCES, generate_filler_text

NAMES = ["Alice", "Bob", "Carlos", "Diana", "Ethan", "Fiona", "George", "Hannah", "Isaac", "Julia", "Kevin", "Luna"]
CITIES = ["Zurich", "Tokyo", "Reykjavik", "Kyoto", "Valletta", "Singapore", "Oslo", "Montreal", "Geneva", "Vienna"]
PROJECTS = ["Project Apex", "Project Bluebird", "Project Cerberus", "Project Dynamo", "Project Eclipse", "Project Falcon", "Project Horizon", "Project Iris"]
ORGANIZATIONS = ["Quantum Labs", "Stellar Dynamics", "Cerebro Institute", "Helios Energy", "Vanguard Systems", "Apex Robotics"]
CODES = [f"{random.randint(10000, 99999)}" for _ in range(200)]


def build_passkey_sample(target_tokens: int, depth: float, tokenizer: Any, rng: random.Random) -> Dict[str, Any]:
    passkey = f"{rng.randint(10000, 99999)}"
    prefix_inst = "There is an important secret hidden in the following document. Memorize it carefully.\n\n"
    needle = f"The secret passkey is {passkey}. Remember this passkey.\n"
    question = "\nWhat is the secret passkey? The secret passkey is"

    inst_tokens = len(tokenizer.encode(prefix_inst, add_special_tokens=False))
    needle_tokens = len(tokenizer.encode(needle, add_special_tokens=False))
    q_tokens = len(tokenizer.encode(question, add_special_tokens=False))

    overhead = inst_tokens + needle_tokens + q_tokens
    filler_budget = max(0, target_tokens - overhead)

    tokens_before = int(filler_budget * depth)
    tokens_after = filler_budget - tokens_before

    filler_before = generate_filler_text(tokens_before, tokenizer)
    filler_after = generate_filler_text(tokens_after, tokenizer)

    ids_before = tokenizer.encode(filler_before, add_special_tokens=False)[:tokens_before]
    ids_after = tokenizer.encode(filler_after, add_special_tokens=False)[:tokens_after]

    text_before = tokenizer.decode(ids_before)
    text_after = tokenizer.decode(ids_after)

    full_prompt = f"{prefix_inst}{text_before}\n{needle}\n{text_after}{question}"
    response = f" {passkey}."

    return {
        "prompt": full_prompt,
        "response": response,
        "category": "passkey_needle",
        "length": target_tokens,
        "depth": depth,
    }


def build_multihop_sample(target_tokens: int, depth: float, tokenizer: Any, rng: random.Random) -> Dict[str, Any]:
    person = rng.choice(NAMES)
    city = rng.choice(CITIES)
    proj = rng.choice(PROJECTS)
    org = rng.choice(ORGANIZATIONS)
    code = f"{rng.randint(1000, 9999)}"

    # Fact chain: Person works in City -> City hosts Project -> Project is sponsored by Org with Code
    chain_facts = [
        f"The lead researcher {person} operates out of the regional station in {city}.",
        f"In the city of {city}, the primary strategic endeavor is known as {proj}.",
        f"The initiative {proj} is entirely funded and directed by {org}.",
        f"Records indicate that {org} is officially registered under authentication code {code}.",
    ]
    rng.shuffle(chain_facts)
    needle_text = " ".join(chain_facts)

    prefix = "Analyze the organizational report below to resolve entity relationships:\n\n"
    question = f"\n\nQuestion: Based on the report, what is the registration authentication code of the organization that funds the project in the city where {person} operates?\nAnswer:"

    inst_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    needle_tokens = len(tokenizer.encode(needle_text, add_special_tokens=False))
    q_tokens = len(tokenizer.encode(question, add_special_tokens=False))

    overhead = inst_tokens + needle_tokens + q_tokens
    filler_budget = max(0, target_tokens - overhead)

    tokens_before = int(filler_budget * depth)
    tokens_after = filler_budget - tokens_before

    filler_before = generate_filler_text(tokens_before, tokenizer)
    filler_after = generate_filler_text(tokens_after, tokenizer)

    ids_before = tokenizer.encode(filler_before, add_special_tokens=False)[:tokens_before]
    ids_after = tokenizer.encode(filler_after, add_special_tokens=False)[:tokens_after]

    full_prompt = f"{prefix}{tokenizer.decode(ids_before)}\n\n{needle_text}\n\n{tokenizer.decode(ids_after)}{question}"
    response = (
        f" <think>\nTracing the multi-hop connection from the report:\n"
        f"1. {person} is based in {city}.\n"
        f"2. The project in {city} is {proj}.\n"
        f"3. {proj} is funded by {org}.\n"
        f"4. {org} is registered under code {code}.\n"
        f"</think>\nThe registration authentication code is {code}."
    )

    return {
        "prompt": full_prompt,
        "response": response,
        "category": "multihop_reasoning",
        "length": target_tokens,
        "depth": depth,
    }


def build_lost_in_middle_sample(target_tokens: int, depth: float, tokenizer: Any, rng: random.Random) -> Dict[str, Any]:
    subject = rng.choice(["vessel Aurora", "satellite Vanguard-4", "reactor unit Beta", "deep-sea probe Kraken", "telescope Zenith-1"])
    metric = rng.choice([
        ("cruising velocity", f"{rng.randint(20, 55)} knots"),
        ("orbital altitude", f"{rng.randint(350, 850)} kilometers"),
        ("thermal efficiency rating", f"{rng.randint(85, 99)} percent"),
        ("operational depth threshold", f"{rng.randint(1200, 4500)} meters"),
        ("primary focal aperture", f"{rng.randint(6, 18)} meters"),
    ])
    fact = f"Critical telemetry log: The designated {metric[0]} of the {subject} is confirmed to be {metric[1]}."

    prefix = "Review the archival engineering transcript below:\n\n"
    question = f"\n\nQuestion: According to the engineering transcript, what is the {metric[0]} of the {subject}?\nAnswer:"

    inst_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    fact_tokens = len(tokenizer.encode(fact, add_special_tokens=False))
    q_tokens = len(tokenizer.encode(question, add_special_tokens=False))

    overhead = inst_tokens + fact_tokens + q_tokens
    filler_budget = max(0, target_tokens - overhead)

    tokens_before = int(filler_budget * depth)
    tokens_after = filler_budget - tokens_before

    filler_before = generate_filler_text(tokens_before, tokenizer)
    filler_after = generate_filler_text(tokens_after, tokenizer)

    ids_before = tokenizer.encode(filler_before, add_special_tokens=False)[:tokens_before]
    ids_after = tokenizer.encode(filler_after, add_special_tokens=False)[:tokens_after]

    full_prompt = f"{prefix}{tokenizer.decode(ids_before)}\n\n{fact}\n\n{tokenizer.decode(ids_after)}{question}"
    response = (
        f" <think>\nLocating telemetry log for {subject}. Found log entry: '{fact}'.\n</think>\n"
        f"The {metric[0]} of the {subject} is {metric[1]}."
    )

    return {
        "prompt": full_prompt,
        "response": response,
        "category": "lost_in_middle",
        "length": target_tokens,
        "depth": depth,
    }


def build_long_context_tool_sample(target_tokens: int, depth: float, tokenizer: Any, rng: random.Random) -> List[Dict[str, Any]]:
    # A financial / technical calculation buried in text
    item1_val = rng.randint(120, 850)
    item2_val = rng.randint(40, 250)
    multiplier = rng.randint(2, 6)
    expected_ans = (item1_val - item2_val) * multiplier

    fact = f"Financial audit note: Initial production inventory accounted for ${item1_val}, with recorded spoilage of ${item2_val}. The remaining valid inventory was distributed across {multiplier} regional hubs."

    prefix = "Examine the commercial inventory disclosure below:\n\n"
    question = f"\n\nQuestion: Calculate the total inventory value successfully allocated across the {multiplier} regional hubs.\nAnswer:"

    inst_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    fact_tokens = len(tokenizer.encode(fact, add_special_tokens=False))
    q_tokens = len(tokenizer.encode(question, add_special_tokens=False))

    overhead = inst_tokens + fact_tokens + q_tokens
    filler_budget = max(0, target_tokens - overhead)

    tokens_before = int(filler_budget * depth)
    tokens_after = filler_budget - tokens_before

    filler_before = generate_filler_text(tokens_before, tokenizer)
    filler_after = generate_filler_text(tokens_after, tokenizer)

    ids_before = tokenizer.encode(filler_before, add_special_tokens=False)[:tokens_before]
    ids_after = tokenizer.encode(filler_after, add_special_tokens=False)[:tokens_after]

    full_prompt = f"{prefix}{tokenizer.decode(ids_before)}\n\n{fact}\n\n{tokenizer.decode(ids_after)}{question}"

    # Turn 1: Invocation
    t1_response = (
        f" <think>\nRetrieved inventory parameters from disclosure:\n"
        f"- Initial inventory: {item1_val}\n"
        f"- Spoilage: {item2_val}\n"
        f"- Net valid: {item1_val} - {item2_val}\n"
        f"- Allocated across {multiplier} hubs: ({item1_val} - {item2_val}) * {multiplier}\n"
        f"</think>\n<act> calc(({item1_val} - {item2_val}) * {multiplier}) </act>"
    )

    # Turn 2: Resolution
    full_turn_prompt = f"{full_prompt}{t1_response}\n<obs> {expected_ans} </obs>"
    t2_response = (
        f" <think> The tool verified the total allocated value as {expected_ans}. </think>\n"
        f"The total inventory value successfully allocated across the {multiplier} regional hubs is ${expected_ans}."
    )

    return [
        {
            "prompt": full_prompt,
            "response": t1_response,
            "category": "long_context_tool_act",
            "length": target_tokens,
            "depth": depth,
        },
        {
            "prompt": full_turn_prompt,
            "response": t2_response,
            "category": "long_context_tool_obs",
            "length": target_tokens,
            "depth": depth,
        }
    ]


def generate_curriculum_dataset(
    output_path: str = "data/retrieval_curriculum_data.jsonl",
    samples_per_cell: int = 15,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    rng = random.Random(1337)

    lengths = [128, 256, 512, 1024, 2048, 4096]
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]

    total_records = 0
    print(f"🚀 Generating Long-Context Retrieval Curriculum Dataset...")
    print(f"   Context Lengths : {lengths}")
    print(f"   Placement Depths: {depths}")
    print(f"   Samples/Cell    : {samples_per_cell} per task category")
    print(f"   Output File     : {output_path}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        for L in lengths:
            for d in depths:
                # 1. Passkey Needle
                for _ in range(samples_per_cell):
                    item = build_passkey_sample(L, d, tokenizer, rng)
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_records += 1

                # 2. Multi-hop Reasoning
                for _ in range(samples_per_cell):
                    item = build_multihop_sample(L, d, tokenizer, rng)
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_records += 1

                # 3. Lost-in-the-Middle Factual QA
                for _ in range(samples_per_cell):
                    item = build_lost_in_middle_sample(L, d, tokenizer, rng)
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_records += 1

                # 4. Long-Context Tool Calling
                for _ in range(samples_per_cell):
                    items = build_long_context_tool_sample(L, d, tokenizer, rng)
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        total_records += 1

            print(f"   ✓ Finished context length {L:4d} tokens ({total_records:,} total records written)")

    print(f"\n✅ Retrieval curriculum dataset complete: {total_records:,} records written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Long-Context Retrieval Curriculum Data")
    parser.add_argument("--output", type=str, default="data/retrieval_curriculum_data.jsonl")
    parser.add_argument("--samples-per-cell", type=int, default=15)
    args = parser.parse_args()

    generate_curriculum_dataset(output_path=args.output, samples_per_cell=args.samples_per_cell)
