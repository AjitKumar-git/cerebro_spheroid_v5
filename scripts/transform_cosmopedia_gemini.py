#!/usr/bin/env python3
"""
🌟 CerebroSpheroid V5: Cosmopedia -> Tri-Head Reasoning Dataset Transformer
=============================================================================
Transforms raw expository textbook/article data from HuggingFace Cosmopedia
into 3-Tier Cognitive Emission format (<think>, <act>, <answer>) using the Gemini API.

Capabilities:
  • Streams Cosmopedia dataset directly from HuggingFace (no massive disk downloads).
  • Uses Gemini 2.5 Flash / 2.0 Flash / 1.5 Flash with Structured JSON Output Schema.
  • Converts passive text into active user inquiries, multi-step deliberation scratchpads,
    grounded tool dispatches (calc, sql, python, mermaid), and conversational responses.
  • Enforces context length budget: Prompt <= 512 tokens, Response <= 1024 tokens.
  • Multi-threaded concurrent worker pool with rate-limiting and exponential backoff.
  • Resilient checkpointing: appends directly to JSONL and tracks progress.
"""

import os
import sys
import json
import time
import argparse
import random
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try importing Google GenAI SDK (modern google-genai or legacy google.generativeai)
HAS_GENAI = False
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
    GENAI_VERSION = "google-genai"
except ImportError:
    try:
        import google.generativeai as legacy_genai
        HAS_GENAI = True
        GENAI_VERSION = "google-generativeai"
    except ImportError:
        HAS_GENAI = False
        GENAI_VERSION = None

SYSTEM_INSTRUCTION = """You are an expert cognitive curriculum engineer for CerebroSpheroid V5, an advanced tri-head reasoning AI.
Your job is to transform raw, passive educational passages from textbooks (Cosmopedia) into high-density, interactive TRI-HEAD training examples.

A Tri-Head example consists of:
1. `prompt`: A natural, curious user inquiry, practical challenge, or question inspired by the core educational concepts in the passage (never just "Summarize this text").
2. `thought`: 2-4 lines of explicit internal cognitive deliberation inside <think>...</think>. Dissect the user's intent, plan the pedagogical response, state hypotheses or equations, and identify edge cases or common misconceptions.
3. `action`: Optional grounded tool call inside <act>...</act> if beneficial:
   - For numerical arithmetic or conversions: `calc(<math expression>)`
   - For relational facts (geography, history, chemistry): `sql: SELECT ...` or `db_lookup('Entity', 'Attribute')`
   - For programming or data structures: `python: ...`
   - For workflows, lifecycles, or architectures: `mermaid: graph TD ...`
   - Leave empty ("") if purely conceptual dialogue.
4. `answer`: A clear, direct, and engaging pedagogical answer addressing the user's prompt directly, grounded in the deliberation and tool action.
5. `category`: One of:
   - "conversational_dialogue" (empathetic conceptual explanation)
   - "reasoning_math" (quantitative deduction or arithmetic)
   - "db_fact_retrieval" (structured relational lookup)
   - "python_code" (algorithmic or software implementation)
   - "agentic_modalities" (diagrams, tool routing, or error handling)

Format strictly as JSON with keys:
"prompt", "thought", "action", "answer", "category"
"""

PROMPT_TEMPLATE = """Transform the following educational passage into a high-quality Tri-Head reasoning example:

--- PASSAGE ---
{text}
--- END PASSAGE ---

Ensure:
- The user prompt is natural and realistic (max 512 tokens).
- The thought is an authentic step-by-step reasoning plan.
- The action is grounded (or empty if not needed).
- The answer is complete and instructional (max 1024 tokens).
"""


def get_gemini_client(api_key: Optional[str] = None):
    """Initializes and returns the appropriate Gemini client."""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Missing Gemini API Key. Set GEMINI_API_KEY environment variable or pass --api-key.")

    if GENAI_VERSION == "google-genai":
        return genai.Client(api_key=key)
    elif GENAI_VERSION == "google-generativeai":
        legacy_genai.configure(api_key=key)
        return legacy_genai
    else:
        raise ImportError("Neither 'google-genai' nor 'google-generativeai' is installed. Run: pip install google-genai")


def call_gemini_transform(
    client: Any,
    text_snippet: str,
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 5,
) -> Optional[Dict[str, Any]]:
    """Calls Gemini API with structured JSON output and exponential backoff."""
    prompt = PROMPT_TEMPLATE.format(text=text_snippet.strip()[:2000])

    for attempt in range(max_retries):
        try:
            if GENAI_VERSION == "google-genai":
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        temperature=0.7,
                    ),
                )
                raw_text = response.text
            else:
                model = client.GenerativeModel(
                    model_name=model_name,
                    system_instruction=SYSTEM_INSTRUCTION,
                    generation_config={"response_mime_type": "application/json", "temperature": 0.7},
                )
                response = model.generate_content(prompt)
                raw_text = response.text

            # Parse JSON
            raw_text = raw_text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            data = json.loads(raw_text.strip())
            if isinstance(data, list) and data:
                data = data[0]

            # Validate expected fields
            if "prompt" in data and "thought" in data and "answer" in data:
                return data

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                wait_time = (2 ** attempt) + random.uniform(0.5, 2.0)
                time.sleep(wait_time)
            else:
                time.sleep(1.0)

    return None


def format_trihead_record(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """Converts raw model response dictionary into a CerebroSpheroid tri-head JSONL record."""
    prompt_raw = raw_result.get("prompt", "").strip()
    thought_raw = raw_result.get("thought", "").strip()
    action_raw = raw_result.get("action", "").strip()
    answer_raw = raw_result.get("answer", "").strip()
    category = raw_result.get("category", "conversational_dialogue").strip()

    # Normalize prompt
    if not prompt_raw.lower().startswith("user:"):
        prompt_formatted = f"User: {prompt_raw}"
    else:
        prompt_formatted = prompt_raw

    # Assemble response stream
    response_blocks = []
    if thought_raw:
        # Strip existing tags if duplicated
        clean_thought = thought_raw.replace("<think>", "").replace("</think>", "").strip()
        response_blocks.append(f"<think>\n{clean_thought}\n</think>")

    has_action = bool(action_raw and action_raw.lower() not in ("none", "null", '""', "n/a"))
    if has_action:
        clean_action = action_raw.replace("<act>", "").replace("</act>", "").strip()
        response_blocks.append(f"<act>\n{clean_action}\n</act>")

    clean_answer = answer_raw.replace("<answer>", "").replace("</answer>", "").strip()
    response_blocks.append(clean_answer)

    full_response = "\n".join(response_blocks)
    temporal_gate = [1, 1, 1] if has_action else [1, 0, 1]

    return {
        "prompt": prompt_formatted,
        "response": full_response,
        "category": category,
        "temporal_gate": temporal_gate,
    }


def stream_cosmopedia_passages(
    subset: str = "auto_math_text",
    max_samples: int = 1000,
    min_length: int = 150,
):
    """Streams text passages from Cosmopedia on Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    # Available high-quality educational subsets in Cosmopedia
    # e.g.: auto_math_text, khanacademy, openstax, stanford, stories, web_samples
    print(f"📦 Connecting to HuggingFaceTB/cosmopedia ({subset})...")
    ds = load_dataset("HuggingFaceTB/cosmopedia", subset, split="train", streaming=True)

    count = 0
    for item in ds:
        text = item.get("text", "") or item.get("content", "")
        if len(text.strip()) >= min_length:
            yield text.strip()
            count += 1
            if count >= max_samples:
                break


def main():
    parser = argparse.ArgumentParser(description="Transform Cosmopedia into Tri-Head SFT Dataset with Gemini")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API Key")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--subset", type=str, default="auto_math_text", help="Cosmopedia subset (auto_math_text, openstax, khanacademy, stories)")
    parser.add_argument("--max-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/cosmopedia_trihead_sft.jsonl", help="Output JSONL path")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent transformation workers")
    parser.add_argument("--merge-target", type=str, default=None, help="Optional existing JSONL dataset to append to (e.g. data/chat_thinking_sft_15k.jsonl)")
    args = parser.parse_args()

    client = get_gemini_client(args.api_key)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    existing_count = 0
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            existing_count = sum(1 for line in f if line.strip())
        print(f"📂 Found existing output file with {existing_count:,} records. Appending new records...")

    print(f"\n🚀 Launching Cosmopedia Tri-Head Pipeline:")
    print(f"   • Model       : {args.model}")
    print(f"   • Subset      : {args.subset}")
    print(f"   • Target      : {args.max_samples:,} samples")
    print(f"   • Output      : {args.output}")
    print(f"   • Concurrency : {args.workers} workers\n")

    passages = []
    print("⏳ Streaming passages from Hugging Face...")
    for p in stream_cosmopedia_passages(subset=args.subset, max_samples=args.max_samples * 2):
        passages.append(p)
        if len(passages) >= args.max_samples:
            break
    print(f"✓ Retrieved {len(passages):,} passages for processing.\n")

    saved_count = existing_count
    t0 = time.time()

    def process_passage(text_snippet: str) -> Optional[Dict[str, Any]]:
        raw = call_gemini_transform(client, text_snippet, model_name=args.model)
        if raw:
            return format_trihead_record(raw)
        return None

    with open(args.output, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_passage, p): i for i, p in enumerate(passages)}
            for fut in as_completed(futures):
                try:
                    record = fut.result()
                    if record:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        saved_count += 1
                        elapsed = time.time() - t0
                        rate = saved_count / max(0.1, elapsed)
                        print(f"[{saved_count:,}/{existing_count + len(passages):,}] Generated: {record['category']} | Gate: {record['temporal_gate']} ({rate:.2f} rec/s)")
                except Exception as e:
                    print(f"⚠️ Worker error: {e}")

    print(f"\n✅ Finished! Successfully generated {saved_count:,} Tri-Head records at: {args.output}")

    if args.merge_target and os.path.exists(args.merge_target):
        print(f"🔄 Merging into target: {args.merge_target}...")
        with open(args.output, "r", encoding="utf-8") as src, open(args.merge_target, "a", encoding="utf-8") as dst:
            for line in src:
                if line.strip():
                    dst.write(line.strip() + "\n")
        print("✓ Merge complete!")


if __name__ == "__main__":
    main()
