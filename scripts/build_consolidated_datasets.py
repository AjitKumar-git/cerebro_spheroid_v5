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

Each consolidated type is exported with tri-format parity:
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
    ("messages_json", pa.string()),
])


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

    # Strip legacy tokens
    for stray in ("<|im_end|>", "<|im_start|>", "</s>", "<s>", "<stim>", "</stim>"):
        clean_text = clean_text.replace(stray, "")

    answer = clean_text.strip()
    return {
        "thinking": thinking,
        "tool_call": tool_call,
        "tool_response": tool_response,
        "answer": answer if answer else (raw if not thinking and not tool_call else "")
    }


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
        or raw_item.get("formatted_input")
        or ""
    )
    if isinstance(prompt, str):
        prompt = prompt.replace("<|im_start|>user\n", "").replace("<|im_end|>\n<|im_start|>assistant\n", "").strip()

    # 2. Extract response
    response = (
        raw_item.get("formatted_output")
        or raw_item.get("response")
        or raw_item.get("answer")
        or raw_item.get("completion")
        or raw_item.get("output")
        or raw_item.get("target")
        or ""
    )

    # If raw_item has messages (ShareGPT)
    if "messages" in raw_item and isinstance(raw_item["messages"], list):
        user_msgs = [m.get("content", "") for m in raw_item["messages"] if m.get("role") in ("user", "human")]
        asst_msgs = [m.get("content", "") for m in raw_item["messages"] if m.get("role") in ("assistant", "gpt", "model")]
        if user_msgs and not prompt:
            prompt = "\n".join(user_msgs).strip()
        if asst_msgs and not response:
            response = asst_msgs[-1].strip()

    if not prompt or not response:
        return None

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

    # 5. Build canonical formatted_output
    def to_str(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    thinking = to_str(thinking)
    tool_call = to_str(tool_call)
    tool_response = to_str(tool_response)
    answer = to_str(answer)
    prompt = to_str(prompt)

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

    formatted_input = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    uid = f"{prefix_id}_{idx:06d}"
    category = to_str(raw_item.get("category") or raw_item.get("domain") or type_name)
    source = to_str(raw_item.get("source") or default_source)

    # Spheroid record
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
    }

    # ShareGPT record
    messages = [
        {"role": "system", "content": DOTY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": formatted_output},
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
    using streaming I/O and low-memory batched Parquet writing.
    """
    target_dir = os.path.join(OUTPUT_BASE_DIR, dir_name)
    os.makedirs(target_dir, exist_ok=True)

    spheroid_path = os.path.join(target_dir, f"{type_name}_spheroid.jsonl")
    sharegpt_path = os.path.join(target_dir, f"{type_name}_sharegpt.jsonl")
    parquet_path = os.path.join(target_dir, f"{type_name}.parquet")

    # If skip_existing and all files exist with size > 0:
    if skip_existing and os.path.exists(spheroid_path) and os.path.exists(sharegpt_path) and os.path.exists(parquet_path):
        if os.path.getsize(spheroid_path) > 0 and os.path.getsize(parquet_path) > 0:
            rec_cnt = 0
            with open(spheroid_path, "r", encoding="utf-8") as f:
                for _ in f:
                    rec_cnt += 1
            sph_mb = os.path.getsize(spheroid_path) / (1024 * 1024)
            sgpt_mb = os.path.getsize(sharegpt_path) / (1024 * 1024)
            pq_mb = os.path.getsize(parquet_path) / (1024 * 1024)
            print(f"⏩ [Consolidator] Skipping existing {type_name}: {rec_cnt:,} records, {pq_mb:.2f} MB parquet.")
            return {
                "type": type_name,
                "directory": dir_name,
                "records": rec_cnt,
                "estimated_tokens": int(sph_mb * 1024 * 1024 // 4),
                "spheroid_path": os.path.relpath(spheroid_path, PROJECT_ROOT),
                "sharegpt_path": os.path.relpath(sharegpt_path, PROJECT_ROOT),
                "parquet_path": os.path.relpath(parquet_path, PROJECT_ROOT),
                "spheroid_size_mb": round(sph_mb, 2),
                "sharegpt_size_mb": round(sgpt_mb, 2),
                "parquet_size_mb": round(pq_mb, 2),
                "categories": {},
            }

    print(f"\n============================================================")
    print(f"🚀 [Consolidator] Building Type: {type_name.upper()} -> {dir_name}/")
    print(f"============================================================")

    f_spheroid = open(spheroid_path, "w", encoding="utf-8")
    f_sharegpt = open(sharegpt_path, "w", encoding="utf-8")
    pq_writer = pq.ParquetWriter(parquet_path, schema=PARQUET_SCHEMA, compression="zstd", compression_level=7)

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

    return {
        "type": type_name,
        "directory": dir_name,
        "records": len(spheroid_records),
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
    parser = argparse.ArgumentParser(description="Build consolidated datasets for each curriculum type")
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
                    "path": "data/doty_identity/doty_identity_spheroid.jsonl",
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
                    "path": "data/frontier_100k/frontier_100k_spheroid_format.jsonl",
                    "source_name": "frontier_function_calling",
                    "filter_fn": lambda item: item.get("category") in ("apigen_function_calling", "hermes_function_calling")
                                            or item.get("domain") in ("tool_calling_api", "tool_calling_conversational")
                },
                {
                    "path": "data/cerebro_v5_custom_dataset/factual_generated_1k.jsonl",
                    "source_name": "cerebro_v5_factual_sql",
                }
            ]
        },
        "reasoning": {
            "type_name": "complex_reasoning",
            "dir_name": "03_complex_reasoning",
            "prefix_id": "doty_cot",
            "sources": [
                {
                    "path": "data/frontier_100k/frontier_100k_spheroid_format.jsonl",
                    "source_name": "frontier_deliberation",
                    "filter_fn": lambda item: item.get("category") in ("open_thoughts_deepseek_cot", "sky_t1_olympiad_cot", "numina_math_cot")
                },
                {
                    "path": "data/gemini_golden_reasoning_sft.jsonl",
                    "source_name": "gemini_golden_reasoning",
                },
                {
                    "path": "data/gemma31_graded_reasoning_sft.jsonl",
                    "source_name": "gemma31_graded_reasoning",
                },
                {
                    "path": "data/layman_conversational_cot.jsonl",
                    "source_name": "layman_conversational_cot",
                }
            ]
        },
        "knowledge": {
            "type_name": "domain_knowledge",
            "dir_name": "04_domain_knowledge",
            "prefix_id": "doty_knw",
            "sources": [
                {
                    "path": "data/diverse_sft_corpus_25k.jsonl",
                    "source_name": "diverse_sft_stem_arc",
                },
                {
                    "path": "data/chat_thinking_sft_15k.jsonl",
                    "source_name": "chat_thinking_generalist",
                },
                {
                    "path": "data/cerebro_v5_custom_dataset/train.jsonl",
                    "source_name": "cerebro_v5_custom_train",
                    "limit": 25000,
                },
                {
                    "path": "data/retrieval_curriculum_data.jsonl",
                    "source_name": "retrieval_passkey_needle",
                }
            ]
        },
        "sycophancy": {
            "type_name": "anti_sycophancy",
            "dir_name": "05_anti_sycophancy",
            "prefix_id": "doty_syc",
            "sources": [
                {
                    "path": "data/anti_sycophancy_truth_dataset.jsonl",
                    "source_name": "anti_sycophancy_truth_set",
                },
                {
                    "path": "data/cerebro_cognitive_8k/cerebro_cognitive_10k.jsonl",
                    "source_name": "cerebro_cognitive_counterfactual",
                }
            ]
        },
        "dictionary": {
            "type_name": "lexical_dictionary",
            "dir_name": "06_lexical_dictionary",
            "prefix_id": "doty_lex",
            "sources": [
                {
                    "path": "data/dictionary/dictionary_spheroid_format.jsonl",
                    "source_name": "websters_unabridged_dictionary",
                }
            ]
        }
    }

    selected_types = list(CONFIGS.keys()) if (args.all or not args.type) else [args.type]

    manifest = {
        "manifest_version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent": "Doty",
        "developer": "Ajit Kumar",
        "architecture": "CerebroSpheroid V5",
        "cognitive_boundary": "3-Tier: <think> -> <tool_call> -> plain text",
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
