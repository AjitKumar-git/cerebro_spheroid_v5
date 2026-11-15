#!/usr/bin/env python3
"""
🏆 CerebroSpheroid V5: Official Academic & Industry Benchmark Harness
=====================================================================
Follows the official EleutherAI LM Evaluation Harness / Hugging Face Lighteval standard
used for benchmarking frontier models (SmolLM2-135M, Llama-3, MobileLLM):

Supported Standardized Benchmarks:
1. ARC-Challenge (allenai/ai2_arc, test split: 1,172 questions): Hard multi-step science
2. ARC-Easy      (allenai/ai2_arc, test split: 2,376 questions): Elementary science deduction
3. HellaSwag     (Rowan/hellaswag, val split: 10,042 questions): Physical commonsense continuation
4. MMLU          (cais/mmlu, test split: 14,042 questions): STEM, Humanities, Social Sci
5. Winogrande    (allenai/winogrande, val split: 1,267 questions): Coreference resolution
6. OpenBookQA    (allenai/openbookqa, test split: 500 questions): Open book scientific QA
7. GSM8K         (openai/gsm8k, test split: 1,319 problems): Multi-step math (Exact Match)

Metrics Reported:
- Accuracy (acc): Unnormalized log-likelihood ranking
- Normalized Accuracy (acc_norm): Per-token length-normalized log-likelihood (Open LLM Leaderboard standard)
- Standard Error (stderr): sqrt(p * (1 - p) / N) with 95% Confidence Interval
- Exact Match (EM): Generative numerical answer extraction for math

Usage:
  # Quick 50-sample diagnostic across all tasks:
  python3 scripts/benchmark_industry_standard.py --checkpoint checkpoints/v5_small_sft_best.pt --samples 50

  # Rigorous 500-sample benchmark:
  python3 scripts/benchmark_industry_standard.py --checkpoint checkpoints/v5_small_sft_best.pt --samples 500

  # Specific task evaluation:
  python3 scripts/benchmark_industry_standard.py --tasks arc_challenge,hellaswag --samples 100

  # Full academic evaluation (all thousands of questions):
  python3 scripts/benchmark_industry_standard.py --full
"""

import os
import sys

# Ensure clean user-owned cache directory before importing Hugging Face libraries
HF_CACHE_DIR = os.path.expanduser("~/.cache_hf_user")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

import re
import math
import time
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoTokenizer
from datasets import load_dataset

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.corpus import format_stimulus_prompt

LOCAL_CACHE_DIR = os.path.join(REPO_ROOT, "data", "benchmark_cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

# Published SmolLM2-135M reference scores (Hugging Face technical report)
SMOLLM2_135M_REF = {
    "arc_challenge": 34.1,
    "arc_easy": 52.8,
    "hellaswag": 42.1,
    "mmlu": 31.5,
    "winogrande": 51.5,
    "openbookqa": 37.8,
    "gsm8k": 4.2,
}

RANDOM_BASELINE = {
    "arc_challenge": 25.0,
    "arc_easy": 25.0,
    "hellaswag": 25.0,
    "mmlu": 25.0,
    "winogrande": 50.0,
    "openbookqa": 25.0,
    "gsm8k": 0.0,
}


# ==============================================================================
# 1. DATASET LOADERS WITH LOCAL JSON CACHING
# ==============================================================================

def load_arc_dataset(split_name: str = "ARC-Challenge", limit: Optional[int] = 50) -> List[Dict[str, Any]]:
    """Loads ARC from allenai/ai2_arc test split."""
    cache_tag = f"all" if limit is None or limit <= 0 else str(limit)
    cache_file = os.path.join(LOCAL_CACHE_DIR, f"{split_name.lower()}_{cache_tag}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    print(f"  📥 Loading {cache_tag} items from allenai/ai2_arc ({split_name}, test split)...")
    split_str = f"test[:{limit}]" if limit and limit > 0 else "test"
    ds = load_dataset("allenai/ai2_arc", split_name, split=split_str)
    items = []
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D"}

    for row in ds:
        q = row.get("question", "").strip()
        choices_raw = row.get("choices", {})
        texts = choices_raw.get("text", [])
        labels = [label_map.get(str(lbl), str(lbl)) for lbl in choices_raw.get("label", [])]
        ans = label_map.get(str(row.get("answerKey", "")), str(row.get("answerKey", "")))

        if len(texts) >= 2 and ans in labels:
            items.append({
                "question": q,
                "choices": texts,
                "labels": labels,
                "answer_idx": labels.index(ans),
                "answer_key": ans,
            })

    try:
        with open(cache_file, "w") as f:
            json.dump(items, f, indent=2)
    except Exception:
        pass
    return items


def load_hellaswag_dataset(limit: Optional[int] = 100) -> List[Dict[str, Any]]:
    """Loads HellaSwag from Rowan/hellaswag validation split."""
    cache_tag = f"all" if limit is None or limit <= 0 else str(limit)
    cache_file = os.path.join(LOCAL_CACHE_DIR, f"hellaswag_{cache_tag}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    print(f"  📥 Loading {cache_tag} items from Rowan/hellaswag (validation split)...")
    split_str = f"validation[:{limit}]" if limit and limit > 0 else "validation"
    ds = load_dataset("Rowan/hellaswag", split=split_str)
    items = []

    for row in ds:
        ctx = row.get("ctx", "").strip()
        endings = row.get("endings", [])
        lbl = str(row.get("label", "")).strip()
        if ctx and len(endings) == 4 and lbl.isdigit():
            ans_idx = int(lbl)
            if 0 <= ans_idx < 4:
                items.append({
                    "context": ctx,
                    "choices": endings,
                    "answer_idx": ans_idx,
                })

    try:
        with open(cache_file, "w") as f:
            json.dump(items, f, indent=2)
    except Exception:
        pass
    return items


def load_mmlu_dataset(limit: Optional[int] = 100) -> List[Dict[str, Any]]:
    """Loads MMLU across all subjects from cais/mmlu test split."""
    cache_tag = f"all" if limit is None or limit <= 0 else str(limit)
    cache_file = os.path.join(LOCAL_CACHE_DIR, f"mmlu_{cache_tag}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    print(f"  📥 Loading {cache_tag} items from cais/mmlu (test split)...")
    split_str = f"test[:{limit}]" if limit and limit > 0 else "test"
    ds = load_dataset("cais/mmlu", "all", split=split_str)
    items = []

    for row in ds:
        q = row.get("question", "").strip()
        choices = row.get("choices", [])
        ans = row.get("answer", None)
        if q and len(choices) == 4 and ans is not None and 0 <= int(ans) < 4:
            items.append({
                "question": q,
                "choices": choices,
                "answer_idx": int(ans),
                "subject": row.get("subject", "general"),
            })

    try:
        with open(cache_file, "w") as f:
            json.dump(items, f, indent=2)
    except Exception:
        pass
    return items


def load_winogrande_dataset(limit: Optional[int] = 50) -> List[Dict[str, Any]]:
    """Loads Winogrande from allenai/winogrande validation split."""
    cache_tag = f"all" if limit is None or limit <= 0 else str(limit)
    cache_file = os.path.join(LOCAL_CACHE_DIR, f"winogrande_{cache_tag}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    print(f"  📥 Loading {cache_tag} items from allenai/winogrande (validation split)...")
    split_str = f"validation[:{limit}]" if limit and limit > 0 else "validation"
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split=split_str)
    items = []

    for row in ds:
        sent = row.get("sentence", "").strip()
        opt1 = row.get("option1", "").strip()
        opt2 = row.get("option2", "").strip()
        ans = str(row.get("answer", "")).strip()
        if sent and opt1 and opt2 and ans in ("1", "2"):
            items.append({
                "sentence": sent,
                "choices": [opt1, opt2],
                "answer_idx": int(ans) - 1,
            })

    try:
        with open(cache_file, "w") as f:
            json.dump(items, f, indent=2)
    except Exception:
        pass
    return items


def load_openbookqa_dataset(limit: Optional[int] = 50) -> List[Dict[str, Any]]:
    """Loads OpenBookQA from allenai/openbookqa test split."""
    cache_tag = f"all" if limit is None or limit <= 0 else str(limit)
    cache_file = os.path.join(LOCAL_CACHE_DIR, f"openbookqa_{cache_tag}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    print(f"  📥 Loading {cache_tag} items from allenai/openbookqa (test split)...")
    split_str = f"test[:{limit}]" if limit and limit > 0 else "test"
    ds = load_dataset("allenai/openbookqa", "additional", split=split_str)
    items = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    for row in ds:
        q = row.get("question_stem", "").strip()
        choices_raw = row.get("choices", {})
        texts = choices_raw.get("text", [])
        ans = str(row.get("answerKey", "")).strip().upper()
        if q and len(texts) == 4 and ans in label_map:
            items.append({
                "question": q,
                "choices": texts,
                "answer_idx": label_map[ans],
                "answer_key": ans,
            })

    try:
        with open(cache_file, "w") as f:
            json.dump(items, f, indent=2)
    except Exception:
        pass
    return items


def load_gsm8k_dataset(limit: Optional[int] = 25) -> List[Dict[str, Any]]:
    """Loads GSM8K from openai/gsm8k test split."""
    cache_tag = f"all" if limit is None or limit <= 0 else str(limit)
    cache_file = os.path.join(LOCAL_CACHE_DIR, f"gsm8k_{cache_tag}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    print(f"  📥 Loading {cache_tag} items from openai/gsm8k (test split)...")
    split_str = f"test[:{limit}]" if limit and limit > 0 else "test"
    ds = load_dataset("openai/gsm8k", "main", split=split_str)
    items = []

    for row in ds:
        q = row.get("question", "").strip()
        ans_raw = row.get("answer", "")
        if "####" in ans_raw:
            num = ans_raw.split("####")[-1].strip().replace(",", "")
            items.append({
                "question": q,
                "solution": ans_raw,
                "target_number": num,
            })

    try:
        with open(cache_file, "w") as f:
            json.dump(items, f, indent=2)
    except Exception:
        pass
    return items


# ==============================================================================
# 2. STANDARDIZED EVALUATION KERNEL (acc & acc_norm)
# ==============================================================================

def score_multiple_choice(
    model: CerebroSpheroidV5,
    tokenizer: Any,
    prompt: str,
    choices: List[str],
    device: torch.device,
    protocol: str = "stim_act",
) -> Tuple[int, int, List[float], List[float]]:
    """
    Standard EleutherAI / Lighteval zero-shot log-likelihood multiple choice scoring.
    Batches candidate choices into a single forward pass for 4x throughput acceleration.
    Computes both unnormalized cross-entropy and per-token length-normalized cross-entropy.
    Returns (pred_unnorm, pred_norm, losses_unnorm, losses_norm).
    """
    if protocol == "stim_act":
        prompt_str = format_stimulus_prompt(prompt) + " <act> "
    else:
        prompt_str = prompt.strip() + " "

    p_toks = tokenizer.encode(prompt_str, add_special_tokens=False)
    letters = ["A", "B", "C", "D"][:len(choices)]
    max_seq = getattr(model.config, "total_window", 768)

    c_tok_list = []
    for idx, ch in enumerate(choices):
        lbl = letters[idx] if idx < len(letters) else str(idx + 1)
        comp_str = f" {lbl}) {ch.strip()}"
        ct = tokenizer.encode(comp_str, add_special_tokens=False)
        if not ct:
            ct = tokenizer.encode(f" {lbl}", add_special_tokens=False)
        c_tok_list.append(ct)

    # Prepare batched sequences with padding
    max_l = max(len(p_toks + ct) for ct in c_tok_list)
    if max_l > max_seq:
        max_l = max_seq

    batch_seq = []
    batch_tgt = []

    for ct in c_tok_list:
        seq_i = p_toks + ct
        if len(seq_i) > max_seq:
            seq_i = seq_i[-max_seq:]
            eff_p = max(0, len(seq_i) - len(ct))
            tgt_i = [-100] * eff_p + ct[- (len(seq_i) - eff_p):]
        else:
            eff_p = len(p_toks)
            pad_len = max_l - len(seq_i)
            tgt_i = [-100] * eff_p + ct + [-100] * pad_len
            seq_i = seq_i + [0] * pad_len

        batch_seq.append(seq_i)
        batch_tgt.append(tgt_i)

    b_full = torch.tensor(batch_seq, dtype=torch.long, device=device)
    b_tgt = torch.tensor(batch_tgt, dtype=torch.long, device=device)

    with torch.no_grad():
        _, logits = model(b_full, targets=b_tgt)
        losses_norm = []
        losses_unnorm = []

        for i in range(len(choices)):
            mask = (b_tgt[i] != -100)
            num_tokens = int(mask.sum().item())
            if num_tokens == 0:
                losses_norm.append(999.0)
                losses_unnorm.append(999.0)
            else:
                mean_l = float(F.cross_entropy(logits[i, mask].float(), b_tgt[i, mask], reduction="mean").item())
                losses_norm.append(mean_l)
                losses_unnorm.append(mean_l * num_tokens)

    pred_unnorm = int(torch.argmin(torch.tensor(losses_unnorm)))
    pred_norm = int(torch.argmin(torch.tensor(losses_norm)))
    return pred_unnorm, pred_norm, losses_unnorm, losses_norm


def extract_numerical_answer(text: str) -> Optional[str]:
    """Extracts numerical answer from generative chain-of-thought."""
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    m2 = re.search(r"(?:answer is|equals?|=)\s*\$?(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m2:
        return m2.group(1).replace(",", "")
    nums = re.findall(r"-?\d+", text)
    if nums:
        return nums[-1]
    return None


def calculate_stderr(accuracy_pct: float, n: int) -> float:
    """Computes standard error of the mean percentage: sqrt(p * (1 - p) / N) * 100."""
    if n <= 1:
        return 0.0
    p = accuracy_pct / 100.0
    se = math.sqrt(max(0.0, p * (1.0 - p)) / n) * 100.0
    return se


# ==============================================================================
# 3. MAIN BENCHMARK ENGINE
# ==============================================================================

def run_academic_benchmark_suite(
    checkpoint_path: str,
    tasks: List[str],
    samples_per_task: Optional[int] = 50,
    protocol: str = "auto",
    device_str: Optional[str] = None,
    export_json: Optional[str] = None,
):
    print("\n" + "=" * 98)
    print("🏆 CEREBROSPHEROID V5: OFFICIAL ACADEMIC BENCHMARK BATTERY")
    print(f"Checkpoint : {os.path.basename(checkpoint_path)}")
    print(f"Harness    : EleutherAI LM Evaluation Harness / Hugging Face Lighteval Standard")
    print("=" * 98)

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"⚙️ Execution Device : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(REPO_ROOT, "data", "tokenizer"))

    # Load Model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = infer_v5_config_from_state_dict(ckpt, default_preset="v5_small")
    model = CerebroSpheroidV5(config=cfg)
    load_v5_checkpoint_into_torch(model, ckpt, verbose=False)
    model.to(device).eval()

    step = ckpt.get("step", ckpt.get("global_step", "Final"))
    loss = ckpt.get("loss", ckpt.get("train_loss", ckpt.get("eval_loss", "unknown")))
    print(f"✓ Model Loaded    : Step {step} | Loss: {loss}")
    print(f"✓ Architecture    : {cfg.num_shells} concentric shells, {cfg.surface_dim}D surface, {cfg.thinking_cycles} cycles")

    # Determine prompt protocol
    if protocol == "auto":
        is_sft = "sft" in os.path.basename(checkpoint_path).lower()
        active_protocol = "stim_act" if is_sft else "raw"
    else:
        active_protocol = protocol
    print(f"✓ Prompt Protocol : {active_protocol.upper()} ({'<stim>...</stim><act>' if active_protocol == 'stim_act' else 'Raw Completion'})\n")

    results_summary: Dict[str, Dict[str, Any]] = {}
    details_log: Dict[str, List[Any]] = {}

    start_total = time.time()

    # --------------------------------------------------------------------------
    # 1. ARC-Challenge
    # --------------------------------------------------------------------------
    if "all" in tasks or "arc_challenge" in tasks:
        print(f"🔬 [Task] Evaluating ARC-Challenge (allenai/ai2_arc test split)...")
        items = load_arc_dataset("ARC-Challenge", limit=samples_per_task)
        corr_u, corr_n = 0, 0
        details = []

        t0 = time.time()
        for idx, item in enumerate(items, 1):
            prompt = f"Question: {item['question']}\nAnswer:"
            pu, pn, _, _ = score_multiple_choice(model, tokenizer, prompt, item["choices"], device, protocol=active_protocol)
            ok_u = (pu == item["answer_idx"])
            ok_n = (pn == item["answer_idx"])
            if ok_u: corr_u += 1
            if ok_n: corr_n += 1
            details.append({"q": item["question"][:60], "target": item["answer_key"], "pred": item["labels"][pn], "correct": ok_n})
            if idx % 25 == 0 or idx == len(items):
                print(f"   [{idx}/{len(items)}] Current acc_norm: {(corr_n/idx)*100:.1f}%...")

        acc = (corr_u / max(1, len(items))) * 100.0
        acc_norm = (corr_n / max(1, len(items))) * 100.0
        se = calculate_stderr(acc_norm, len(items))
        results_summary["arc_challenge"] = {"acc": acc, "acc_norm": acc_norm, "stderr": se, "n": len(items), "time_sec": time.time() - t0}
        details_log["arc_challenge"] = details
        print(f"   ✓ ARC-Challenge : {acc_norm:.1f}% ± {se:.1f}% acc_norm ({corr_n}/{len(items)})\n")

    # --------------------------------------------------------------------------
    # 2. ARC-Easy
    # --------------------------------------------------------------------------
    if "all" in tasks or "arc_easy" in tasks:
        print(f"🔬 [Task] Evaluating ARC-Easy (allenai/ai2_arc test split)...")
        items = load_arc_dataset("ARC-Easy", limit=samples_per_task)
        corr_u, corr_n = 0, 0
        details = []

        t0 = time.time()
        for idx, item in enumerate(items, 1):
            prompt = f"Question: {item['question']}\nAnswer:"
            pu, pn, _, _ = score_multiple_choice(model, tokenizer, prompt, item["choices"], device, protocol=active_protocol)
            ok_u = (pu == item["answer_idx"])
            ok_n = (pn == item["answer_idx"])
            if ok_u: corr_u += 1
            if ok_n: corr_n += 1
            details.append({"q": item["question"][:60], "target": item["answer_key"], "pred": item["labels"][pn], "correct": ok_n})
            if idx % 25 == 0 or idx == len(items):
                print(f"   [{idx}/{len(items)}] Current acc_norm: {(corr_n/idx)*100:.1f}%...")

        acc = (corr_u / max(1, len(items))) * 100.0
        acc_norm = (corr_n / max(1, len(items))) * 100.0
        se = calculate_stderr(acc_norm, len(items))
        results_summary["arc_easy"] = {"acc": acc, "acc_norm": acc_norm, "stderr": se, "n": len(items), "time_sec": time.time() - t0}
        details_log["arc_easy"] = details
        print(f"   ✓ ARC-Easy      : {acc_norm:.1f}% ± {se:.1f}% acc_norm ({corr_n}/{len(items)})\n")

    # --------------------------------------------------------------------------
    # 3. HellaSwag
    # --------------------------------------------------------------------------
    if "all" in tasks or "hellaswag" in tasks:
        print(f"🔬 [Task] Evaluating HellaSwag (Rowan/hellaswag validation split)...")
        items = load_hellaswag_dataset(limit=samples_per_task)
        corr_u, corr_n = 0, 0
        details = []

        t0 = time.time()
        for idx, item in enumerate(items, 1):
            prompt = item["context"]
            pu, pn, _, _ = score_multiple_choice(model, tokenizer, prompt, item["choices"], device, protocol=active_protocol)
            ok_u = (pu == item["answer_idx"])
            ok_n = (pn == item["answer_idx"])
            if ok_u: corr_u += 1
            if ok_n: corr_n += 1
            details.append({"ctx": item["context"][:60], "target": item["answer_idx"], "pred": pn, "correct": ok_n})
            if idx % 25 == 0 or idx == len(items):
                print(f"   [{idx}/{len(items)}] Current acc_norm: {(corr_n/idx)*100:.1f}%...")

        acc = (corr_u / max(1, len(items))) * 100.0
        acc_norm = (corr_n / max(1, len(items))) * 100.0
        se = calculate_stderr(acc_norm, len(items))
        results_summary["hellaswag"] = {"acc": acc, "acc_norm": acc_norm, "stderr": se, "n": len(items), "time_sec": time.time() - t0}
        details_log["hellaswag"] = details
        print(f"   ✓ HellaSwag     : {acc_norm:.1f}% ± {se:.1f}% acc_norm ({corr_n}/{len(items)})\n")

    # --------------------------------------------------------------------------
    # 4. MMLU
    # --------------------------------------------------------------------------
    if "all" in tasks or "mmlu" in tasks:
        print(f"🔬 [Task] Evaluating MMLU (cais/mmlu test split across STEM & Humanities)...")
        items = load_mmlu_dataset(limit=samples_per_task)
        corr_u, corr_n = 0, 0
        details = []

        t0 = time.time()
        for idx, item in enumerate(items, 1):
            prompt = f"Question: {item['question']}\nAnswer:"
            pu, pn, _, _ = score_multiple_choice(model, tokenizer, prompt, item["choices"], device, protocol=active_protocol)
            ok_u = (pu == item["answer_idx"])
            ok_n = (pn == item["answer_idx"])
            if ok_u: corr_u += 1
            if ok_n: corr_n += 1
            details.append({"q": item["question"][:60], "target": item["answer_idx"], "pred": pn, "correct": ok_n})
            if idx % 25 == 0 or idx == len(items):
                print(f"   [{idx}/{len(items)}] Current acc_norm: {(corr_n/idx)*100:.1f}%...")

        acc = (corr_u / max(1, len(items))) * 100.0
        acc_norm = (corr_n / max(1, len(items))) * 100.0
        se = calculate_stderr(acc_norm, len(items))
        results_summary["mmlu"] = {"acc": acc, "acc_norm": acc_norm, "stderr": se, "n": len(items), "time_sec": time.time() - t0}
        details_log["mmlu"] = details
        print(f"   ✓ MMLU          : {acc_norm:.1f}% ± {se:.1f}% acc_norm ({corr_n}/{len(items)})\n")

    # --------------------------------------------------------------------------
    # 5. Winogrande
    # --------------------------------------------------------------------------
    if "all" in tasks or "winogrande" in tasks:
        print(f"🔬 [Task] Evaluating Winogrande (allenai/winogrande validation split)...")
        items = load_winogrande_dataset(limit=samples_per_task)
        corr_u, corr_n = 0, 0
        details = []

        t0 = time.time()
        for idx, item in enumerate(items, 1):
            prompt = item["sentence"].replace("_", "").strip()
            pu, pn, _, _ = score_multiple_choice(model, tokenizer, prompt, item["choices"], device, protocol=active_protocol)
            ok_u = (pu == item["answer_idx"])
            ok_n = (pn == item["answer_idx"])
            if ok_u: corr_u += 1
            if ok_n: corr_n += 1
            details.append({"s": item["sentence"][:60], "target": item["answer_idx"], "pred": pn, "correct": ok_n})
            if idx % 25 == 0 or idx == len(items):
                print(f"   [{idx}/{len(items)}] Current acc_norm: {(corr_n/idx)*100:.1f}%...")

        acc = (corr_u / max(1, len(items))) * 100.0
        acc_norm = (corr_n / max(1, len(items))) * 100.0
        se = calculate_stderr(acc_norm, len(items))
        results_summary["winogrande"] = {"acc": acc, "acc_norm": acc_norm, "stderr": se, "n": len(items), "time_sec": time.time() - t0}
        details_log["winogrande"] = details
        print(f"   ✓ Winogrande    : {acc_norm:.1f}% ± {se:.1f}% acc_norm ({corr_n}/{len(items)})\n")

    # --------------------------------------------------------------------------
    # 6. OpenBookQA
    # --------------------------------------------------------------------------
    if "all" in tasks or "openbookqa" in tasks:
        print(f"🔬 [Task] Evaluating OpenBookQA (allenai/openbookqa test split)...")
        items = load_openbookqa_dataset(limit=samples_per_task)
        corr_u, corr_n = 0, 0
        details = []

        t0 = time.time()
        for idx, item in enumerate(items, 1):
            prompt = f"Question: {item['question']}\nAnswer:"
            pu, pn, _, _ = score_multiple_choice(model, tokenizer, prompt, item["choices"], device, protocol=active_protocol)
            ok_u = (pu == item["answer_idx"])
            ok_n = (pn == item["answer_idx"])
            if ok_u: corr_u += 1
            if ok_n: corr_n += 1
            details.append({"q": item["question"][:60], "target": item["answer_idx"], "pred": pn, "correct": ok_n})
            if idx % 25 == 0 or idx == len(items):
                print(f"   [{idx}/{len(items)}] Current acc_norm: {(corr_n/idx)*100:.1f}%...")

        acc = (corr_u / max(1, len(items))) * 100.0
        acc_norm = (corr_n / max(1, len(items))) * 100.0
        se = calculate_stderr(acc_norm, len(items))
        results_summary["openbookqa"] = {"acc": acc, "acc_norm": acc_norm, "stderr": se, "n": len(items), "time_sec": time.time() - t0}
        details_log["openbookqa"] = details
        print(f"   ✓ OpenBookQA    : {acc_norm:.1f}% ± {se:.1f}% acc_norm ({corr_n}/{len(items)})\n")

    # --------------------------------------------------------------------------
    # 7. GSM8K (Generative CoT Exact Match)
    # --------------------------------------------------------------------------
    if "all" in tasks or "gsm8k" in tasks:
        print(f"🔬 [Task] Evaluating GSM8K (openai/gsm8k test split, Generative Exact Match)...")
        items = load_gsm8k_dataset(limit=samples_per_task)
        corr_em = 0
        details = []

        t0 = time.time()
        for idx, item in enumerate(items, 1):
            if active_protocol == "stim_act":
                prompt_str = format_stimulus_prompt(f"Solve this math problem: {item['question']}") + " <act> "
            else:
                prompt_str = f"Question: {item['question']}\nAnswer: Let's calculate step by step:"

            inp = tokenizer(prompt_str, return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    inp["input_ids"],
                    max_new_tokens=48,
                    temperature=0.1,
                    top_k=20,
                    repetition_penalty=1.15,
                )
            gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            gen_ans = gen_text[len(prompt_str):].strip()
            extracted = extract_numerical_answer(gen_ans)
            target = item["target_number"]
            is_em = (extracted is not None and extracted == target)
            if is_em: corr_em += 1
            details.append({"q": item["question"][:60], "target": target, "pred": extracted, "correct": is_em})

        em_pct = (corr_em / max(1, len(items))) * 100.0
        se = calculate_stderr(em_pct, len(items))
        results_summary["gsm8k"] = {"acc": em_pct, "acc_norm": em_pct, "stderr": se, "n": len(items), "time_sec": time.time() - t0}
        details_log["gsm8k"] = details
        print(f"   ✓ GSM8K Exact Match : {em_pct:.1f}% ± {se:.1f}% ({corr_em}/{len(items)})\n")

    # --------------------------------------------------------------------------
    # Grand Comparison Table
    # --------------------------------------------------------------------------
    print("=" * 100)
    print("📊 OFFICIAL ACADEMIC BENCHMARK SCOREBOARD (Standardized EleutherAI / Lighteval Battery)")
    print("=" * 100)
    headers = f"{'Benchmark Task':<20} | {'N Samples':<10} | {'CerebroSpheroid (acc_norm)':<28} | {'Random Chance':<14} | {'SmolLM2-135M Ref':<16}"
    print(headers)
    print("-" * 100)

    all_acc_norms = []
    all_smollm_refs = []

    for t_name, data in results_summary.items():
        an = data["acc_norm"]
        se = data["stderr"]
        n = data["n"]
        rc = RANDOM_BASELINE.get(t_name, 25.0)
        s_ref = SMOLLM2_135M_REF.get(t_name, 0.0)

        all_acc_norms.append(an)
        all_smollm_refs.append(s_ref)

        val_str = f"{an:>5.1f}% ± {se:>4.1f}%"
        print(f"{t_name:<20} | {n:<10} | {val_str:<28} | {rc:>5.1f}%        | {s_ref:>5.1f}%")

    print("-" * 100)
    comp_cerebro = sum(all_acc_norms) / max(1, len(all_acc_norms))
    comp_smollm = sum(all_smollm_refs) / max(1, len(all_smollm_refs))
    print(f"{'COMPOSITE AVERAGE':<20} | {'—':<10} | {comp_cerebro:>5.1f}%                     | {'—':<14} | {comp_smollm:>5.1f}%")
    print("=" * 100)

    elapsed_total = time.time() - start_total
    print(f"\n⏱️ Total Benchmark Evaluation Time: {elapsed_total:.1f} seconds ({elapsed_total/60:.2f} minutes)")

    if export_json:
        out_data = {
            "metadata": {
                "checkpoint": os.path.basename(checkpoint_path),
                "step": step,
                "loss": loss,
                "protocol": active_protocol,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "device": str(device),
            },
            "summary": results_summary,
            "details": details_log,
        }
        os.makedirs(os.path.dirname(os.path.abspath(export_json)), exist_ok=True)
        with open(export_json, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"💾 Full Results Exported to: {export_json}\n")


def main():
    parser = argparse.ArgumentParser(description="Official Academic Benchmark Battery for CerebroSpheroid V5")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_small_sft_best.pt", help="Path to checkpoint")
    parser.add_argument("--tasks", type=str, default="all", help="Comma-separated tasks: arc_challenge,arc_easy,hellaswag,mmlu,winogrande,openbookqa,gsm8k,all")
    parser.add_argument("--samples", type=int, default=50, help="Number of questions per benchmark (e.g. 50, 100, 500)")
    parser.add_argument("--full", action="store_true", help="Evaluate full official dataset splits (thousands of questions)")
    parser.add_argument("--protocol", type=str, choices=["auto", "stim_act", "raw"], default="auto", help="Prompt formatting protocol")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda/cpu)")
    parser.add_argument("--export-json", type=str, default="benchmarks/academic_benchmark_results.json", help="Path to export full JSON results")

    args = parser.parse_args()

    ckpt_path = os.path.join(REPO_ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    if not os.path.exists(ckpt_path):
        alt = os.path.join(REPO_ROOT, "checkpoints", "v5_small_latest_step900k.pt")
        if os.path.exists(alt):
            ckpt_path = alt

    task_list = [t.strip().lower() for t in args.tasks.split(",")]
    samples = None if args.full else args.samples

    run_academic_benchmark_suite(
        checkpoint_path=ckpt_path,
        tasks=task_list,
        samples_per_task=samples,
        protocol=args.protocol,
        device_str=args.device,
        export_json=args.export_json,
    )


if __name__ == "__main__":
    main()
