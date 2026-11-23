#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Consolidated Dataset Builder for Frontier Agent "Doty"
============================================================
Consolidates, standardizes, decontaminates, and exports training datasets
for each of the 6 core curriculum types of CerebroSpheroid V5:

  1. doty_identity       (Persona, Creator Attribution: Ajit Kumar, Architecture)
  2. agentic_tools       (APIGen & Hermes Function Calling, Multi-turn execution)
  3. complex_reasoning   (OpenThoughts, Sky-T1 Olympiad, NuminaMath, Golden CoT)
  4. domain_knowledge    (Diverse SFT, Chat Thinking, Custom Train, Retrieval)
  5. anti_sycophancy     (Anti-sycophancy, Epistemic Humility, Counterfactuals)
  6. lexical_dictionary  (Webster's 111k unabridged lexicon & rare-token anchors)

Key Architectural Refinements (from Handover Specification):
  • Complete elimination of ChatML tokens (<|im_start|>, <|im_end|>):
    Prevents tokenizer sub-token fragmentation [529, 29989, 326, ...] ("im" token hallucination).
  • Native special tokens contract:
    - Input: <s> {prompt}\\n
    - Output: <think>...</think><tool_call>...</tool_call>\\n<tool_response>...</tool_response>\\n{answer}</s>
  • Functional tool syntax normalization:
    Converts nested JSON calls into functional syntax: calc(...), facts(...), python(...), func(k=v)
    to eliminate small-model JSON squishing hallucinations.
  • Pointer-copy sensory grounding & closed-loop agentic trajectories.
  • Parity across (formatted_input, formatted_output) and (input, output) fields.

Tri-Format Parity:
  • <type>_spheroid.jsonl : CerebroSpheroid 3-tier format (<think>, <tool_call>, answer)
  • <type>_sharegpt.jsonl : Industry ShareGPT / OpenAI chat messages format
  • <type>.parquet        : Fast columnar storage with ZSTD level 7 compression

Developer & Creator: Ajit Kumar
Target Agent: Doty
Architecture: CerebroSpheroid V5 (100% unlocked shells, SCMER regularization)
"""

import os
import sys
import json
import re
import time
import hashlib
import argparse
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.identity_sanitizer import (
    sanitize_conversation_turn,
    is_identity_contaminated,
)

DOTY_SYSTEM_PROMPT = (
    "You are Doty, an autonomous frontier AI agent developed by Ajit Kumar. "
    "You are built upon the CerebroSpheroid V5 concentric cognitive architecture, "
    "featuring biologically inspired multi-scale radial pathways, 100% unlocked cortical shells, "
    "and SCMER embedding regularization. You deliberate thoughtfully within <think>...</think>, "
    "invoke deterministic tools via standard schemas when precision is required, "
    "and deliver objective, truthful, and helpful answers."
)

OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "consolidated")


PARQUET_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("type", pa.string()),
    ("category", pa.string()),
    ("source", pa.string()),
    ("persona", pa.string()),
    ("developer", pa.string()),
    ("prompt", pa.string()),
    ("thinking", pa.string()),
    ("tool_call", pa.string()),
    ("tool_response", pa.string()),
    ("answer", pa.string()),
    ("formatted_input", pa.string()),
    ("formatted_output", pa.string()),
    ("input", pa.string()),
    ("output", pa.string()),
    ("messages_json", pa.string()),
])


def json_tool_call_to_functional(tool_str: str) -> str:
    """
    Converts nested JSON function calls into clean functional syntax:
    e.g. [{"name": "facts", "arguments": {"query": "atomic mass"}}] -> facts("atomic mass")
    calc, facts, python, and arbitrary APIs are converted to func(k=v) or func(arg).
    """
    raw = str(tool_str or "").strip()
    if not raw or raw in ("None", "null", "{}"):
        return ""

    # If already functional syntax like calc(...) or facts(...)
    if re.match(r"^[a-zA-Z0-9_]+\(.*?\)$", raw, re.DOTALL):
        return raw

    try:
        data = json.loads(raw)

        def _fmt(item: Any) -> str:
            if not isinstance(item, dict):
                return str(item)
            name = item.get("name") or item.get("tool") or "tool"
            args = item.get("arguments") or item.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    return f'{name}("{args}")'
            if not isinstance(args, dict):
                return f"{name}({args})"
            if len(args) == 1:
                k, v = next(iter(args.items()))
                if isinstance(v, str):
                    return f'{name}("{v}")'
                return f"{name}({v})"
            parts = [f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in args.items()]
            return f"{name}({', '.join(parts)})"

        if isinstance(data, list):
            return "\n".join(_fmt(x) for x in data if x)
        elif isinstance(data, dict):
            return _fmt(data)
    except Exception:
        pass

    return raw


def parse_model_tiers(text: str) -> Dict[str, str]:
    """
    Extracts 3-tier components from formatted text:
    thinking (<think>...</think>), tool_call (<tool_call>... or <act>...),
    tool_response (<tool_response>...), and answer (clean talking text).
    """
    raw = str(text or "").strip()

    # 1. Thinking
    thinking = ""
    m_think = re.search(r"<think>(.*?)(?:</think>|$)", raw, re.DOTALL)
    if m_think:
        thinking = m_think.group(1).strip()

    # 2. Tool call
    tool_call = ""
    m_tool = re.search(r"<(?:tool_call|act)>(.*?)(?:</(?:tool_call|act)>|$)", raw, re.DOTALL)
    if m_tool:
        tool_call = m_tool.group(1).strip()

    # 3. Tool response
    tool_response = ""
    m_resp = re.search(r"<tool_response>(.*?)(?:</tool_response>|$)", raw, re.DOTALL)
    if m_resp:
        tool_response = m_resp.group(1).strip()

    # 4. Clean answer (text outside think and tool tags)
    clean_text = re.sub(r"<think>.*?(?:</think>|$)", "", raw, flags=re.DOTALL)
    clean_text = re.sub(r"<(?:tool_call|act)>.*?(?:</(?:tool_call|act)>|$)", "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(r"<tool_response>.*?(?:</tool_response>|$)", "", clean_text, flags=re.DOTALL)

    # Strip legacy and ChatML tokens
    for stray in (
        "<|im_end|>", "<|im_start|>", "</s>", "<s>",
        "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_start|>system\n",
        "<stim>", "</stim>", "[INST]", "[/INST]"
    ):
        clean_text = clean_text.replace(stray, "")

    answer = clean_text.strip()
    return {
        "thinking": thinking,
        "tool_call": tool_call,
        "tool_response": tool_response,
        "answer": answer if answer else (raw if not thinking and not tool_call else "")
    }


def clean_raw_prompt(prompt: str) -> str:
    """Strips all ChatML and legacy tokens from the prompt."""
    if not isinstance(prompt, str):
        prompt = str(prompt or "")
    for stray in (
        "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_start|>system\n",
        "<|im_start|>", "<|im_end|>",
        "<stim>", "</stim>", "[INST]", "[/INST]"
    ):
        prompt = prompt.replace(stray, "")

    prompt = prompt.strip()
    # Strip leading <s> so we don't duplicate
    if prompt.startswith("<s>"):
        prompt = prompt[len("<s>"):].strip()
    # Strip trailing </s>
    if prompt.endswith("</s>"):
        prompt = prompt[:-len("</s>")].strip()
    return prompt.strip()


def standardize_record(
    raw_item: Dict[str, Any],
    type_name: str,
    prefix_id: str,
    idx: int,
    default_source: str = "cerebro_v5_corpus"
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    """
    Standardizes a raw record into (spheroid_dict, sharegpt_dict, parquet_dict).
    Applies identity decontamination and attribution to Ajit Kumar.
    Returns None if record is contaminated or empty.
    """
    # 1. Extract prompt
    prompt = (
        raw_item.get("prompt")
        or raw_item.get("question")
        or raw_item.get("input")
        or raw_item.get("formatted_input")
        or ""
    )
    prompt = clean_raw_prompt(str(prompt))

    # 2. Extract response
    response = (
        raw_item.get("formatted_output")
        or raw_item.get("response")
        or raw_item.get("output")
        or raw_item.get("target")
        or raw_item.get("answer")
        or raw_item.get("completion")
        or ""
    )

    # If raw_item has messages (ShareGPT)
    if "messages" in raw_item and isinstance(raw_item["messages"], list):
        user_msgs = [m.get("content", "") for m in raw_item["messages"] if m.get("role") in ("user", "human")]
        asst_msgs = [m.get("content", "") for m in raw_item["messages"] if m.get("role") in ("assistant", "gpt", "model")]
        if user_msgs and not prompt:
            prompt = clean_raw_prompt("\n".join(user_msgs))
        if asst_msgs and not response:
            response = asst_msgs[-1].strip()

    if not prompt or not response:
        return None

    # Strip ChatML from response
    if isinstance(response, str):
        for stray in (
            "<|im_start|>assistant\n", "<|im_start|>user\n", "<|im_start|>system\n",
            "<|im_start|>", "<|im_end|>",
            "<stim>", "</stim>", "[INST]", "[/INST]"
        ):
            response = response.replace(stray, "")
        response = response.strip()

    # 3. Decontaminate third-party model leaks
    clean_res = sanitize_conversation_turn(prompt, response, allow_adversarial_attribution=True)
    if clean_res is None:
        return None  # Purged toxic leak

    sanitized_response = clean_res["response"]

    # 4. Parse 3-tier components
    tiers = parse_model_tiers(sanitized_response)
    thinking = raw_item.get("thinking") or tiers["thinking"]
    tool_call = raw_item.get("tool_call") or raw_item.get("action") or tiers["tool_call"]
    tool_response = raw_item.get("tool_response") or tiers["tool_response"]
    answer = raw_item.get("answer") or tiers["answer"]
    if not answer and not thinking and not tool_call:
        answer = sanitized_response

    # Convert tool call to functional syntax
    if tool_call:
        tool_call = json_tool_call_to_functional(tool_call)

    # 5. Build canonical formatted_output
    def to_str(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    thinking = to_str(thinking).strip()
    tool_call = to_str(tool_call).strip()
    tool_response = to_str(tool_response).strip()
    answer = to_str(answer).strip()
    prompt = to_str(prompt).strip()

    out_parts = []
    if thinking:
        out_parts.append(f"<think>\n{thinking}\n</think>")
    if tool_call:
        out_parts.append(f"<tool_call>\n{tool_call}\n</tool_call>")
    if tool_response:
        out_parts.append(f"<tool_response>\n{tool_response}\n</tool_response>")
    if answer:
        out_parts.append(answer)
    formatted_output = "\n".join(out_parts).strip()

    # Clean legacy ChatML tokens from formatted_output if any slipped through
    for stray in ("<|im_start|>", "<|im_end|>"):
        formatted_output = formatted_output.replace(stray, "")

    # Ensure clean terminal EOS token
    if answer or not tool_call:
        if not formatted_output.endswith("</s>"):
            formatted_output = formatted_output + "</s>"
    else:
        # Tool call emission only (Stage 1)
        if not formatted_output.endswith("</tool_call>") and not formatted_output.endswith("</s>"):
            formatted_output = formatted_output + "</s>"

    # Native input format
    formatted_input = f"<s> {prompt}\n"

    uid = f"{prefix_id}_{idx:06d}"
    category = to_str(raw_item.get("category") or raw_item.get("domain") or type_name)
    source = to_str(raw_item.get("source") or default_source)

    # Spheroid record (both formatted_input/formatted_output and input/output for full compatibility)
    spheroid_record = {
        "id": uid,
        "type": type_name,
        "category": category,
        "source": source,
        "persona": "Doty",
        "developer": "Ajit Kumar",
        "prompt": prompt,
        "thinking": thinking,
        "tool_call": tool_call,
        "tool_response": tool_response,
        "answer": answer,
        "formatted_input": formatted_input,
        "formatted_output": formatted_output,
        "input": formatted_input,
        "output": formatted_output,
    }

    # ShareGPT record (clean messages without raw <s> or ChatML tokens)
    clean_user_content = prompt
    if clean_user_content.startswith("<s>"):
        clean_user_content = clean_user_content[len("<s>"):].strip()
    clean_asst_content = formatted_output.rstrip("</s>").strip()

    messages = [
        {"role": "system", "content": DOTY_SYSTEM_PROMPT},
        {"role": "user", "content": clean_user_content},
        {"role": "assistant", "content": clean_asst_content},
    ]
    sharegpt_record = {
        "id": uid,
        "type": type_name,
        "category": category,
        "source": source,
        "developer": "Ajit Kumar",
        "messages": messages,
    }

    # Parquet record
    parquet_record = {
        "id": uid,
        "type": type_name,
        "category": category,
        "source": source,
        "persona": "Doty",
        "developer": "Ajit Kumar",
        "prompt": prompt,
        "thinking": thinking,
        "tool_call": tool_call,
        "tool_response": tool_response,
        "answer": answer,
        "formatted_input": formatted_input,
        "formatted_output": formatted_output,
        "input": formatted_input,
        "output": formatted_output,
        "messages_json": json.dumps(messages, ensure_ascii=False),
    }

    return spheroid_record, sharegpt_record, parquet_record


def export_consolidated_type(
    type_name: str,
    dir_name: str,
    prefix_id: str,
    sources_config: List[Dict[str, Any]],
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """
    Consolidates multiple input sources into a single validated type directory
    using streaming I/O, atomic tmp replacement, and low-memory batched Parquet writing.
    """
    target_dir = os.path.join(OUTPUT_BASE_DIR, dir_name)
    os.makedirs(target_dir, exist_ok=True)

    spheroid_path = os.path.join(target_dir, f"{type_name}_spheroid.jsonl")
    sharegpt_path = os.path.join(target_dir, f"{type_name}_sharegpt.jsonl")
    parquet_path = os.path.join(target_dir, f"{type_name}.parquet")

    # Temp files for atomic write
    spheroid_tmp = f"{spheroid_path}.tmp"
    sharegpt_tmp = f"{sharegpt_path}.tmp"
    parquet_tmp = f"{parquet_path}.tmp"

    print(f"\n============================================================")
    print(f"🚀 [Consolidator] Processing Type: {type_name.upper()} -> {dir_name}/")
    print(f"============================================================")

    f_spheroid = open(spheroid_tmp, "w", encoding="utf-8")
    f_sharegpt = open(sharegpt_tmp, "w", encoding="utf-8")
    pq_writer = pq.ParquetWriter(parquet_tmp, schema=PARQUET_SCHEMA, compression="zstd", compression_level=7)

    parquet_batch: List[Dict[str, Any]] = []
    BATCH_SIZE = 2500

    total_scanned = 0
    total_purged = 0
    total_retained = 0
    total_token_chars = 0
    seen_prompts = set()
    category_counts = Counter()

    for src in sources_config:
        path = os.path.join(PROJECT_ROOT, src["path"])
        if not os.path.exists(path):
            print(f"⚠️ Source not found: {path}, skipping.")
            continue

        default_source = src.get("source_name", "cerebro_v5_corpus")
        filter_fn = src.get("filter_fn")
        limit = src.get("limit")

        print(f"  • Ingesting: {src['path']} (limit={limit})...")
        src_count = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue

                try:
                    item = json.loads(line_str)
                except Exception:
                    continue

                total_scanned += 1

                # Apply optional filtering
                if filter_fn and not filter_fn(item):
                    continue

                std = standardize_record(
                    item,
                    type_name=type_name,
                    prefix_id=prefix_id,
                    idx=total_retained,
                    default_source=default_source
                )
                if std is None:
                    total_purged += 1
                    continue

                sph, sgpt, pq_row = std

                # Exact prompt deduplication using 8-byte hash (low RAM)
                p_hash = hashlib.md5(" ".join(sph["prompt"].lower().split()).encode("utf-8")).digest()[:8]
                if p_hash in seen_prompts:
                    continue
                seen_prompts.add(p_hash)

                # Write directly to stream
                f_spheroid.write(json.dumps(sph, ensure_ascii=False) + "\n")
                f_sharegpt.write(json.dumps(sgpt, ensure_ascii=False) + "\n")
                parquet_batch.append(pq_row)

                total_retained += 1
                total_token_chars += len(sph["prompt"]) + len(sph["formatted_output"])
                category_counts[sph["category"]] += 1
                src_count += 1

                if len(parquet_batch) >= BATCH_SIZE:
                    table = pa.Table.from_pylist(parquet_batch, schema=PARQUET_SCHEMA)
                    pq_writer.write_table(table)
                    parquet_batch = []

                if limit and src_count >= limit:
                    break

        print(f"    -> Retained {src_count:,} records from this source.")

    # Flush remaining batch
    if parquet_batch:
        table = pa.Table.from_pylist(parquet_batch, schema=PARQUET_SCHEMA)
        pq_writer.write_table(table)
        parquet_batch = []

    f_spheroid.close()
    f_sharegpt.close()
    pq_writer.close()

    # Atomic rename tmp files to target paths
    os.replace(spheroid_tmp, spheroid_path)
    os.replace(sharegpt_tmp, sharegpt_path)
    os.replace(parquet_tmp, parquet_path)

    # Compute metrics & sizes
    sph_mb = os.path.getsize(spheroid_path) / (1024 * 1024)
    sgpt_mb = os.path.getsize(sharegpt_path) / (1024 * 1024)
    pq_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    est_tokens = total_token_chars // 4

    print(f"\n📊 Summary for {type_name}:")
    print(f"  • Scanned:          {total_scanned:,} raw records")
    print(f"  • Purged Leaks:     {total_purged:,}")
    print(f"  • Retained Clean:   {total_retained:,} records")
    print(f"  • Estimated Tokens: ~{est_tokens:,} tokens ({est_tokens/1e9:.3f}B)")
    print(f"  • Spheroid JSONL:   {spheroid_path} ({sph_mb:.2f} MB)")
    print(f"  • ShareGPT JSONL:   {sharegpt_path} ({sgpt_mb:.2f} MB)")
    print(f"  • Parquet (ZSTD):   {parquet_path} ({pq_mb:.2f} MB)")
    print(f"  • Top Categories:   {dict(category_counts.most_common(5))}")

    return {
        "type": type_name,
        "directory": dir_name,
        "records": total_retained,
        "estimated_tokens": est_tokens,
        "spheroid_path": os.path.relpath(spheroid_path, PROJECT_ROOT),
        "sharegpt_path": os.path.relpath(sharegpt_path, PROJECT_ROOT),
        "parquet_path": os.path.relpath(parquet_path, PROJECT_ROOT),
        "spheroid_size_mb": round(sph_mb, 2),
        "sharegpt_size_mb": round(sgpt_mb, 2),
        "parquet_size_mb": round(pq_mb, 2),
        "categories": dict(category_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Build and refine consolidated datasets for each curriculum type")
    parser.add_argument("--type", type=str, help="Specific type to build (identity, tools, reasoning, knowledge, sycophancy, dictionary)")
    parser.add_argument("--all", action="store_true", help="Build all 6 consolidated dataset types")
    parser.add_argument("--skip-existing", action="store_true", help="Skip types that are already fully exported")
    args = parser.parse_args()

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    CONFIGS = {
        "identity": {
            "type_name": "doty_identity",
            "dir_name": "01_doty_identity",
            "prefix_id": "doty_id",
            "sources": [
                {
                    "path": "data/consolidated/01_doty_identity/doty_identity_spheroid.jsonl",
                    "source_name": "cerebro_v5_doty_grounding",
                }
            ]
        },
        "tools": {
            "type_name": "agentic_tools",
            "dir_name": "02_agentic_tools",
            "prefix_id": "doty_tool",
            "sources": [
                {
                    "path": "data/consolidated/02_agentic_tools/closed_loop_sft_pairs.jsonl",
                    "source_name": "closed_loop_sensory_kv_pairs",
                },
                {
                    "path": "data/consolidated/02_agentic_tools/closed_loop_trajectories.jsonl",
                    "source_name": "closed_loop_trajectories",
                },
                {
                    "path": "data/consolidated/02_agentic_tools/agentic_tools_sensory_grounded.jsonl",
                    "source_name": "agentic_tools_sensory_grounded",
                },
                {
                    "path": "data/consolidated/02_agentic_tools/agentic_tools_spheroid.jsonl",
                    "source_name": "agentic_tools_spheroid",
                }
            ]
        },
        "reasoning": {
            "type_name": "complex_reasoning",
            "dir_name": "03_complex_reasoning",
            "prefix_id": "doty_cot",
            "sources": [
                {
                    "path": "data/consolidated/03_complex_reasoning/complex_reasoning_spheroid.jsonl",
                    "source_name": "complex_reasoning_curriculum",
                }
            ]
        },
        "knowledge": {
            "type_name": "domain_knowledge",
            "dir_name": "04_domain_knowledge",
            "prefix_id": "doty_knw",
            "sources": [
                {
                    "path": "data/consolidated/04_domain_knowledge/domain_knowledge_spheroid.jsonl",
                    "source_name": "domain_knowledge_curriculum",
                }
            ]
        },
        "sycophancy": {
            "type_name": "anti_sycophancy",
            "dir_name": "05_anti_sycophancy",
            "prefix_id": "doty_syc",
            "sources": [
                {
                    "path": "data/consolidated/05_anti_sycophancy/anti_sycophancy_spheroid.jsonl",
                    "source_name": "anti_sycophancy_curriculum",
                }
            ]
        },
        "dictionary": {
            "type_name": "lexical_dictionary",
            "dir_name": "06_lexical_dictionary",
            "prefix_id": "doty_lex",
            "sources": [
                {
                    "path": "data/consolidated/06_lexical_dictionary/lexical_dictionary_spheroid.jsonl",
                    "source_name": "websters_unabridged_dictionary",
                }
            ]
        }
    }

    selected_types = list(CONFIGS.keys()) if (args.all or not args.type) else [args.type]

    manifest = {
        "manifest_version": "1.1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent": "Doty",
        "developer": "Ajit Kumar",
        "architecture": "CerebroSpheroid V5",
        "cognitive_boundary": "3-Tier: <think> -> <tool_call> -> plain text",
        "special_tokens_contract": {
            "input_prefix": "<s> {prompt}\\n",
            "output_suffix": "</s>",
            "chatml_tokens_purged": True,
            "functional_tool_syntax": True,
            "sensory_kv_grounded": True,
        },
        "total_types": len(selected_types),
        "types": {},
        "summary": {
            "total_records": 0,
            "total_estimated_tokens": 0,
            "total_parquet_size_mb": 0.0,
        }
    }

    t0 = time.time()
    for t in selected_types:
        if t not in CONFIGS:
            print(f"Unknown type: {t}. Choose from: {list(CONFIGS.keys())}")
            continue

        cfg = CONFIGS[t]
        res = export_consolidated_type(
            type_name=cfg["type_name"],
            dir_name=cfg["dir_name"],
            prefix_id=cfg["prefix_id"],
            sources_config=cfg["sources"],
            skip_existing=args.skip_existing,
        )
        manifest["types"][cfg["type_name"]] = res
        manifest["summary"]["total_records"] += res["records"]
        manifest["summary"]["total_estimated_tokens"] += res["estimated_tokens"]
        manifest["summary"]["total_parquet_size_mb"] += res["parquet_size_mb"]

    # Save manifest
    manifest_path = os.path.join(OUTPUT_BASE_DIR, "DATASET_MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"🎉 MASTER CONSOLIDATION PIPELINE COMPLETE ({elapsed:.1f}s)")
    print(f"   • Total Types Consolidated: {len(manifest['types'])}")
    print(f"   • Grand Total Records:       {manifest['summary']['total_records']:,}")
    print(f"   • Estimated Token Volume:    ~{manifest['summary']['total_estimated_tokens']:,} tokens ({manifest['summary']['total_estimated_tokens']/1e9:.3f}B)")
    print(f"   • Total Parquet Size:        {manifest['summary']['total_parquet_size_mb']:.2f} MB")
    print(f"   • Manifest Saved:            {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
