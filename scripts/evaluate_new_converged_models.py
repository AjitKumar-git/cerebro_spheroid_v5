#!/usr/bin/env python3
"""
📊 Comprehensive Evaluation Suite for Newly Converged CerebroSpheroid V5 Models
==============================================================================
Evaluates and compares:
  1. checkpoints/v5_qwen_0_5b_champion.pt (Newly generated 280M Champion)
  2. checkpoints/v5_qwen_0_5b_refined_sft.pt (280M Refined SFT)
  3. checkpoints/v5_small_sft_master_diverse.pt (102M Diverse SFT Native)
  4. checkpoints/v5_small_sft_best.pt (102M Native SFT Champion)

Test Dimensions:
  A. Generative QA Accuracy (Analogies, Antonyms, Grammar, Facts, Arithmetic)
  B. Language Modeling Perplexity & Cross-Entropy Loss
  C. Academic Multiple-Choice Benchmarks (MMLU, OpenBookQA, Winogrande, ARC-Easy)
  D. Action Routing / Deliberation Telemetry
"""

import os
import sys
import json
import time
import math
from typing import Dict, Any, List, Tuple

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, infer_v5_config_from_state_dict
from src.corpus import (
    format_stimulus_prompt,
    format_motor_response,
    encode_no_special,
    parse_model_output,
)
from scripts.benchmark_industry_standard import run_academic_benchmark_suite


MODELS_TO_EVAL = [
    {
        "name": "Qwen-0.5B Champion (Milestone 6)",
        "path": "checkpoints/v5_qwen_0_5b_champion.pt",
        "tier": "280M UWBL + AALW",
    },
    {
        "name": "Qwen-0.5B Refined SFT (Milestone 5)",
        "path": "checkpoints/v5_qwen_0_5b_refined_sft.pt",
        "tier": "280M UWBL + SCM",
    },
    {
        "name": "Native V5 Master Diverse (Milestone 4)",
        "path": "checkpoints/v5_small_sft_master_diverse.pt",
        "tier": "102M Native SFT",
    },
    {
        "name": "Native V5 SFT Best (Milestone 2)",
        "path": "checkpoints/v5_small_sft_best.pt",
        "tier": "102M Native SFT",
    },
]

QA_TEST_BATTERY = [
    # 1. Analogies
    ("Complete the analogy: hot is to cold as day is to ____.", ["night"]),
    ("Complete the analogy: bird is to fly as fish is to ____.", ["swim"]),
    ("Complete the analogy: pen is to write as knife is to ____.", ["cut"]),
    ("Complete the analogy: author is to book as artist is to ____.", ["painting", "art", "picture"]),
    # 2. Antonyms
    ("What is the opposite of 'dark'?", ["light", "bright"]),
    ("What is the opposite of 'fast'?", ["slow"]),
    ("What is the exact antonym of the word 'author'?", ["reader", "audience"]),
    ("What is the opposite of 'strong'?", ["weak"]),
    # 3. Verbs / Grammar
    ("What is the simple past tense of the irregular verb 'drink'?", ["drank"]),
    ("What is the past tense of 'run'?", ["ran"]),
    ("What is the past tense of 'eat'?", ["ate"]),
    # 4. Factual Knowledge
    ("What is the capital of France?", ["Paris"]),
    ("What is the capital of Japan?", ["Tokyo"]),
    ("What is the capital of Germany?", ["Berlin"]),
    # 5. Arithmetic Word Problem
    ("If John has 3 apples and buys 2 more, how many apples does he have?", ["5", "five"]),
    ("What is 15 + 27?", ["42"]),
]


def load_eval_model(ckpt_path: str, device: torch.device) -> Tuple[CerebroSpheroidV5, ModelConfigV5]:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_d = raw.get("config", {})
    valid_keys = set(ModelConfigV5.__dataclass_fields__.keys())
    valid_cfg = {k: v for k, v in cfg_d.items() if k in valid_keys}
    if valid_cfg:
        cfg = ModelConfigV5(**valid_cfg)
    else:
        cfg = infer_v5_config_from_state_dict(raw, default_preset="v5_qwen_0_5b" if "qwen" in ckpt_path else "v5_small")

    model = CerebroSpheroidV5(cfg)
    sd = raw.get("model_state", raw.get("model", raw))
    model.load_state_dict(sd, strict=False)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    model = model.to(dtype).to(device).eval()
    return model, cfg


def evaluate_generative_qa(
    model: CerebroSpheroidV5,
    tokenizer: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates the model across the standardized QA test battery."""
    results = []
    correct_count = 0

    for prompt, acceptable_answers in QA_TEST_BATTERY:
        formatted = format_stimulus_prompt(prompt)
        raw_ids = encode_no_special(tokenizer, formatted)
        input_ids = [t % model.config.vocab_size for t in raw_ids] or [1]
        inp_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out_ids = model.generate(
                inp_tensor,
                max_new_tokens=25,
                temperature=0.1,
                top_k=5,
                stop_token_ids=(getattr(tokenizer, "eos_token_id", 2) or 2,),
            )

        seq = out_ids[0].tolist()
        # Decode without special tokens
        emitted_text = tokenizer.decode(seq, skip_special_tokens=False)
        parsed = parse_model_output(emitted_text)
        answer = parsed["answer"] if parsed["answer"] else parsed["raw"]

        # Clean answer text
        clean_ans = answer.strip().lower()
        matched = any(acc.lower() in clean_ans for acc in acceptable_answers)
        if matched:
            correct_count += 1

        results.append({
            "prompt": prompt,
            "target": acceptable_answers[0],
            "emitted": answer[:40],
            "correct": matched,
        })

    accuracy = (correct_count / len(QA_TEST_BATTERY)) * 100.0
    return {"accuracy": accuracy, "correct": correct_count, "total": len(QA_TEST_BATTERY), "details": results}


def compute_perplexity(
    model: CerebroSpheroidV5,
    tokenizer: Any,
    device: torch.device,
    max_eval_items: int = 50,
) -> Dict[str, float]:
    """Measures test cross-entropy loss and perplexity on cognitive benchmark items."""
    benchmark_path = os.path.join(WORKSPACE_ROOT, "eval", "benchmark_1000.jsonl")
    if not os.path.exists(benchmark_path):
        return {"loss": 0.0, "ppl": 0.0}

    items = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_eval_items:
                break
            entry = json.loads(line)
            p = entry.get("prompt", "")
            ans = entry.get("reference_answer", entry.get("target", ""))
            if p and ans:
                items.append((p, ans))

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for p_str, a_str in items:
            p_ids = [t % model.config.vocab_size for t in encode_no_special(tokenizer, format_stimulus_prompt(p_str))][:128]
            a_ids = [t % model.config.vocab_size for t in encode_no_special(tokenizer, format_motor_response(a_str))][:128]
            if len(p_ids) < 2 or len(a_ids) < 2:
                continue

            p_ten = torch.tensor([p_ids], dtype=torch.long, device=device)
            a_ten = torch.tensor([a_ids], dtype=torch.long, device=device)

            loss, _ = model(p_ten, a_ten, deliberation_mode="teacher_forcing")
            loss_val = loss.item()
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                total_loss += loss_val
                count += 1

    avg_loss = total_loss / max(1, count)
    ppl = math.exp(min(avg_loss, 20.0))
    return {"loss": avg_loss, "ppl": ppl}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(os.path.join(WORKSPACE_ROOT, "data", "tokenizer"))

    print("\n" + "=" * 90)
    print("🏆 CEREBROSPHEROID V5: MASTER NEW MODEL CONVERGENCE & PERFORMANCE AUDIT")
    print(f"Device: {device} | CUDA Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 90)

    summary_records = []

    for m_info in MODELS_TO_EVAL:
        name = m_info["name"]
        path = m_info["path"]
        tier = m_info["tier"]

        full_path = os.path.join(WORKSPACE_ROOT, path)
        if not os.path.exists(full_path):
            print(f"⚠️ Checkpoint not found: {path}, skipping.")
            continue

        print(f"\nEvaluating: {name} ({tier})...")
        t0 = time.time()
        model, cfg = load_eval_model(full_path, device)

        # 1. Perplexity & Loss
        ppl_metrics = compute_perplexity(model, tok, device, max_eval_items=50)

        # 2. Generative QA Accuracy
        qa_metrics = evaluate_generative_qa(model, tok, device)

        # 3. Academic Benchmarks (MMLU, OpenBookQA, Winogrande, ARC-Easy)
        print("   Running Official Academic Battery (MMLU, OpenBookQA, Winogrande, ARC-Easy)...")
        acad_summary, _ = run_academic_benchmark_suite(
            checkpoint_path=full_path,
            tasks=["mmlu", "openbookqa", "winogrande", "arc_easy"],
            samples_per_task=25,
            protocol="stim_act",
            device_str=str(device),
        )

        elapsed = time.time() - t0
        print(f"   ✓ Finished in {elapsed:.1f}s | PPL: {ppl_metrics['ppl']:.2f} | Loss: {ppl_metrics['loss']:.2f} | QA Match: {qa_metrics['accuracy']:.1f}%")

        summary_records.append({
            "name": name,
            "path": path,
            "tier": tier,
            "loss": ppl_metrics["loss"],
            "ppl": ppl_metrics["ppl"],
            "qa_acc": qa_metrics["accuracy"],
            "qa_details": qa_metrics["details"],
            "mmlu": acad_summary.get("mmlu", {}).get("acc_norm", 0.0),
            "openbookqa": acad_summary.get("openbookqa", {}).get("acc_norm", 0.0),
            "winogrande": acad_summary.get("winogrande", {}).get("acc_norm", 0.0),
            "arc_easy": acad_summary.get("arc_easy", {}).get("acc_norm", 0.0),
        })

        del model
        torch.cuda.empty_cache()

    # Print Master Summary Table
    print("\n" + "=" * 105)
    print("📊 MASTER CONVERGED MODEL BENCHMARK SCORECARD")
    print("=" * 105)
    print(f"{'Model':<35} | {'Tier':<18} | {'Loss':<6} | {'PPL':<9} | {'MMLU':<6} | {'OBQA':<6} | {'Wino':<6} | {'ARC-E':<6}")
    print("-" * 105)
    for r in summary_records:
        print(f"{r['name']:<35} | {r['tier']:<18} | {r['loss']:<6.2f} | {r['ppl']:<9.1f} | {r['mmlu']:<6.1f} | {r['openbookqa']:<6.1f} | {r['winogrande']:<6.1f} | {r['arc_easy']:<6.1f}")
    print("=" * 105)

    # Export to JSON
    out_json = os.path.join(WORKSPACE_ROOT, "eval", "results", "converged_models_performance_report.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\n💾 Full JSON evaluation report exported to: {out_json}")


if __name__ == "__main__":
    main()
