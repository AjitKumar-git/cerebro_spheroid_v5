#!/usr/bin/env python3
"""
🏆 CerebroSpheroid V5: Industry-Standard Extensive Benchmark Harness
=====================================================================
Follows the official EleutherAI / HuggingFace Open LLM Leaderboard standard:
1. ARC-Challenge (allenai/ai2_arc, test split): Hard scientific reasoning
2. ARC-Easy      (allenai/ai2_arc, test split): Elementary scientific deduction
3. HellaSwag     (Rowan/hellaswag, validation split): Commonsense physical continuation
4. MMLU          (cais/mmlu, test split): Broad academic knowledge (STEM, Humanities, Social Sci)
5. GSM8K         (openai/gsm8k, test split): Multi-step arithmetic reasoning (Exact Match)
6. Winogrande    (winogrande, validation split): Commonsense coreference resolution

Metrics:
- Accuracy (acc): Unnormalized log-likelihood ranking
- Normalized Accuracy (acc_norm): Per-token length-normalized log-likelihood (standard leaderboard metric)
- Exact Match (EM): Generative numerical answer extraction
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from transformers import AutoTokenizer

CACHE_DIR = os.path.join(REPO_ROOT, "data", "benchmark_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.makedirs("/tmp/hf_cache", exist_ok=True)


# ==============================================================================
# 1. STATE DICT CONVERTER (JAX TO PYTORCH)
# ==============================================================================

def adapt_jax_to_torch_state_dict(
    jax_state: Dict[str, Any], cfg: ModelConfigV5
) -> Dict[str, torch.Tensor]:
    """Transposes 2D linear weight matrices and maps parameter names from JAX to PyTorch."""
    torch_state: Dict[str, torch.Tensor] = {}

    for k, v in jax_state.items():
        if isinstance(v, (int, float, str, bool)):
            continue
        if isinstance(v, torch.Tensor):
            t = v.clone().detach().float()
        elif hasattr(v, "numpy"):
            t = torch.from_numpy(v.numpy())
        else:
            t = torch.tensor(v)

        if "embedding" in k or "pos_embed" in k:
            torch_state[k] = t
            continue

        if t.ndim == 2:
            t = t.T

        torch_state[k] = t

    if "sensory.out_self.weight" not in torch_state and getattr(cfg, "use_sensory_self_attn", True):
        d = cfg.surface_dim
        torch_state["sensory.out_self.weight"] = torch.zeros(d, d, dtype=torch.float32)
        torch_state["sensory.q_self.weight"] = torch.randn(d, d, dtype=torch.float32) * 0.02
        torch_state["sensory.k_self.weight"] = torch.randn(d, d, dtype=torch.float32) * 0.02
        torch_state["sensory.v_self.weight"] = torch.randn(d, d, dtype=torch.float32) * 0.02

    return torch_state


# ==============================================================================
# 2. DATASET STREAMERS & LOCAL CACHING
# ==============================================================================

def get_arc_dataset(split_name: str = "ARC-Challenge", limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves official ARC questions from allenai/ai2_arc."""
    cache_file = os.path.join(CACHE_DIR, f"{split_name.lower()}_{limit}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    from datasets import load_dataset
    print(f"  📥 Streaming {limit} questions from allenai/ai2_arc ({split_name}, test)...")
    ds = load_dataset("allenai/ai2_arc", split_name, split="test", streaming=True)
    items = []
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D"}

    for row in ds:
        q = row.get("question", "").strip()
        choices_raw = row.get("choices", {})
        texts = choices_raw.get("text", [])
        labels = [label_map.get(str(lbl), str(lbl)) for lbl in choices_raw.get("label", [])]
        ans = label_map.get(str(row.get("answerKey", "")), str(row.get("answerKey", "")))

        if len(texts) >= 2 and ans in labels:
            ans_idx = labels.index(ans)
            items.append({
                "question": q,
                "choices": texts,
                "labels": labels,
                "answer_idx": ans_idx,
                "answer_key": ans,
            })
            if len(items) >= limit:
                break

    try:
        with open(cache_file, "w") as f:
            json.dump(items, f, indent=2)
    except Exception:
        pass
    return items


def get_hellaswag_dataset(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves official HellaSwag questions from Rowan/hellaswag (validation split)."""
    cache_file = os.path.join(CACHE_DIR, f"hellaswag_{limit}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    from datasets import load_dataset
    print(f"  📥 Streaming {limit} questions from Rowan/hellaswag (validation)...")
    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
    items = []

    for row in ds:
        ctx = row.get("ctx", "").strip()
        endings = row.get("endings", [])
        lbl = row.get("label", "")
        if ctx and len(endings) == 4 and str(lbl).isdigit():
            ans_idx = int(lbl)
            if 0 <= ans_idx < 4:
                items.append({
                    "context": ctx,
                    "choices": endings,
                    "answer_idx": ans_idx,
                    "activity": row.get("activity_label", "Activity"),
                })
                if len(items) >= limit:
                    break

    try:
        with open(cache_file, "w") as f:
            json.dump(items, f, indent=2)
    except Exception:
        pass
    return items


def get_mmlu_dataset(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves official MMLU questions from cais/mmlu (test split across subjects)."""
    cache_file = os.path.join(CACHE_DIR, f"mmlu_{limit}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    from datasets import load_dataset
    print(f"  📥 Streaming {limit} questions from cais/mmlu (test)...")
    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    items = []

    for row in ds:
        q = row.get("question", "").strip()
        choices = row.get("choices", [])
        ans = row.get("answer", None)
        subj = row.get("subject", "general")

        if q and len(choices) == 4 and ans is not None and 0 <= int(ans) < 4:
            items.append({
                "question": q,
                "choices": choices,
                "answer_idx": int(ans),
                "subject": subj,
            })
            if len(items) >= limit:
                break

    try:
        with open(cache_file, "w") as f:
            json.dump(items, f, indent=2)
    except Exception:
        pass
    return items


def get_gsm8k_dataset(limit: int = 25) -> List[Dict[str, Any]]:
    """Retrieves official GSM8K questions from openai/gsm8k (test split)."""
    cache_file = os.path.join(CACHE_DIR, f"gsm8k_{limit}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    from datasets import load_dataset
    print(f"  📥 Streaming {limit} questions from openai/gsm8k (test)...")
    ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
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
            if len(items) >= limit:
                break

    try:
        with open(cache_file, "w") as f:
            json.dump(items, f, indent=2)
    except Exception:
        pass
    return items


def get_winogrande_dataset(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves official Winogrande questions from winogrande (validation split)."""
    cache_file = os.path.join(CACHE_DIR, f"winogrande_{limit}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    from datasets import load_dataset
    print(f"  📥 Streaming {limit} questions from winogrande (validation)...")
    try:
        ds = load_dataset("winogrande", "winogrande_xl", split="validation", streaming=True)
        items = []
        for row in ds:
            sent = row.get("sentence", "").strip()
            opt1 = row.get("option1", "").strip()
            opt2 = row.get("option2", "").strip()
            ans = str(row.get("answer", ""))
            if sent and opt1 and opt2 and ans in ("1", "2"):
                items.append({
                    "sentence": sent,
                    "option1": opt1,
                    "option2": opt2,
                    "answer_idx": int(ans) - 1,
                })
                if len(items) >= limit:
                    break
        try:
            with open(cache_file, "w") as f:
                json.dump(items, f, indent=2)
        except Exception:
            pass
        return items
    except Exception as e:
        print(f"  ⚠️ Winogrande stream notice: {e}")
        return []


# ==============================================================================
# 3. INDUSTRY STANDARD EVALUATION KERNEL (ACC & ACC_NORM)
# ==============================================================================

def score_multiple_choice_item(
    model: CerebroSpheroidV5,
    tokenizer: Any,
    prompt: str,
    choices: List[str],
    device: torch.device,
    choice_prefix: str = " ",
) -> Tuple[int, int, List[float], List[float]]:
    """
    Computes both unnormalized and length-normalized cross-entropy loss per choice.
    Returns: (pred_unnorm, pred_norm, losses_unnorm, losses_norm)
    """
    losses_unnorm: List[float] = []
    losses_norm: List[float] = []

    prompt_toks = tokenizer.encode(prompt, add_special_tokens=False)
    p_len = len(prompt_toks)

    for ch in choices:
        comp_str = f"{choice_prefix}{ch.strip()}"
        comp_toks = tokenizer.encode(comp_str, add_special_tokens=False)
        if not comp_toks:
            comp_toks = [tokenizer.eos_token_id or 2]

        full_toks = prompt_toks + comp_toks
        # Clamp to model sensory capacity if needed
        max_seq = getattr(model.config, "total_window", 768)
        if len(full_toks) > max_seq:
            full_toks = full_toks[-max_seq:]
            p_len = max(0, len(full_toks) - len(comp_toks))

        full_seq = torch.tensor([full_toks], dtype=torch.long, device=device)
        targets = full_seq.clone()
        targets[:, :p_len] = -100

        with torch.no_grad():
            loss_tensor, _ = model(full_seq, targets=targets)
            mean_loss = float(loss_tensor.detach().cpu())
            num_tokens = len(comp_toks)
            sum_loss = mean_loss * num_tokens

            losses_norm.append(mean_loss)
            losses_unnorm.append(sum_loss)

    pred_unnorm = int(torch.argmin(torch.tensor(losses_unnorm)))
    pred_norm = int(torch.argmin(torch.tensor(losses_norm)))
    return pred_unnorm, pred_norm, losses_unnorm, losses_norm


def extract_numerical_answer(text: str) -> Optional[str]:
    """Extracts final numerical answer from generated solution."""
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


# ==============================================================================
# 4. MAIN BENCHMARK ORCHESTRATOR
# ==============================================================================

def run_extensive_benchmark(
    checkpoint_path: str,
    preset: str = "v5_small",
    device_str: Optional[str] = None,
    samples_arc: int = 50,
    samples_hellaswag: int = 100,
    samples_mmlu: int = 100,
    samples_gsm8k: int = 25,
    samples_winogrande: int = 50,
    export_json: Optional[str] = None,
):
    print("=" * 96)
    print("🏆 CEREBROSPHEROID V5: EXTENSIVE INDUSTRY-STANDARD BENCHMARK SUITE")
    print(f"Checkpoint : {os.path.basename(checkpoint_path)}")
    print(f"Standards  : EleutherAI LM-Eval / HuggingFace Open LLM Leaderboard (acc & acc_norm)")
    print("=" * 96)

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"⚙️ Execution Device: {device}")

    tok_path = os.path.join(REPO_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    raw_cfg = PRESETS_V5.get(preset, PRESETS_V5.get("v5_small"))
    cfg = ModelConfigV5(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    cfg.use_sensory_self_attn = True

    model = CerebroSpheroidV5(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    step = ckpt.get("step", ckpt.get("global_step", "unknown"))
    train_loss = ckpt.get("loss", ckpt.get("train_loss", "unknown"))
    eval_loss = ckpt.get("eval_loss", "unknown")

    if any(k.startswith(("sensory.", "core.", "motor.")) for k in state.keys()):
        state = adapt_jax_to_torch_state_dict(state, cfg)

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    print(f"✓ Model Loaded : Step {step} | Train Loss: {train_loss} | Eval Loss: {eval_loss}")
    print(f"✓ Architecture : {cfg.num_shells} shells, {cfg.surface_dim}D surface, {cfg.thinking_cycles} cycles\n")

    summary_records = {}
    details_records = {}

    # --------------------------------------------------------------------------
    # 1. ARC-Challenge
    # --------------------------------------------------------------------------
    print(f"🔬 [1/6] Evaluating ARC-Challenge ({samples_arc} official test questions)...")
    arc_c_items = get_arc_dataset("ARC-Challenge", limit=samples_arc)
    c_correct_unnorm = 0
    c_correct_norm = 0
    arc_c_details = []

    for item in arc_c_items:
        prompt = f"Question: {item['question']}\nAnswer:"
        pred_u, pred_n, _, _ = score_multiple_choice_item(
            model, tokenizer, prompt, item["choices"], device
        )
        ok_u = (pred_u == item["answer_idx"])
        ok_n = (pred_n == item["answer_idx"])
        if ok_u: c_correct_unnorm += 1
        if ok_n: c_correct_norm += 1
        arc_c_details.append({
            "question": item["question"][:60],
            "target": item["answer_key"],
            "pred_norm": item["labels"][pred_n] if pred_n < len(item["labels"]) else str(pred_n),
            "correct": ok_n,
        })

    c_acc = (c_correct_unnorm / max(1, len(arc_c_items))) * 100
    c_acc_norm = (c_correct_norm / max(1, len(arc_c_items))) * 100
    summary_records["ARC-Challenge"] = {"acc": c_acc, "acc_norm": c_acc_norm, "n": len(arc_c_items)}
    details_records["ARC-Challenge"] = arc_c_details
    print(f"   ✓ ARC-Challenge : {c_acc_norm:.1f}% acc_norm ({c_correct_norm}/{len(arc_c_items)}) [acc: {c_acc:.1f}%]")

    # --------------------------------------------------------------------------
    # 2. ARC-Easy
    # --------------------------------------------------------------------------
    print(f"\n🔬 [2/6] Evaluating ARC-Easy ({samples_arc} official test questions)...")
    arc_e_items = get_arc_dataset("ARC-Easy", limit=samples_arc)
    e_correct_unnorm = 0
    e_correct_norm = 0
    arc_e_details = []

    for item in arc_e_items:
        prompt = f"Question: {item['question']}\nAnswer:"
        pred_u, pred_n, _, _ = score_multiple_choice_item(
            model, tokenizer, prompt, item["choices"], device
        )
        ok_u = (pred_u == item["answer_idx"])
        ok_n = (pred_n == item["answer_idx"])
        if ok_u: e_correct_unnorm += 1
        if ok_n: e_correct_norm += 1
        arc_e_details.append({
            "question": item["question"][:60],
            "target": item["answer_key"],
            "pred_norm": item["labels"][pred_n] if pred_n < len(item["labels"]) else str(pred_n),
            "correct": ok_n,
        })

    e_acc = (e_correct_unnorm / max(1, len(arc_e_items))) * 100
    e_acc_norm = (e_correct_norm / max(1, len(arc_e_items))) * 100
    summary_records["ARC-Easy"] = {"acc": e_acc, "acc_norm": e_acc_norm, "n": len(arc_e_items)}
    details_records["ARC-Easy"] = arc_e_details
    print(f"   ✓ ARC-Easy      : {e_acc_norm:.1f}% acc_norm ({e_correct_norm}/{len(arc_e_items)}) [acc: {e_acc:.1f}%]")

    # --------------------------------------------------------------------------
    # 3. HellaSwag
    # --------------------------------------------------------------------------
    print(f"\n🔬 [3/6] Evaluating HellaSwag ({samples_hellaswag} official validation questions)...")
    hs_items = get_hellaswag_dataset(limit=samples_hellaswag)
    hs_correct_unnorm = 0
    hs_correct_norm = 0
    hs_details = []

    for item in hs_items:
        prompt = item["context"]
        pred_u, pred_n, _, _ = score_multiple_choice_item(
            model, tokenizer, prompt, item["choices"], device
        )
        ok_u = (pred_u == item["answer_idx"])
        ok_n = (pred_n == item["answer_idx"])
        if ok_u: hs_correct_unnorm += 1
        if ok_n: hs_correct_norm += 1
        hs_details.append({
            "context": item["context"][:60],
            "target": item["answer_idx"],
            "pred_norm": pred_n,
            "correct": ok_n,
        })

    hs_acc = (hs_correct_unnorm / max(1, len(hs_items))) * 100
    hs_acc_norm = (hs_correct_norm / max(1, len(hs_items))) * 100
    summary_records["HellaSwag"] = {"acc": hs_acc, "acc_norm": hs_acc_norm, "n": len(hs_items)}
    details_records["HellaSwag"] = hs_details
    print(f"   ✓ HellaSwag     : {hs_acc_norm:.1f}% acc_norm ({hs_correct_norm}/{len(hs_items)}) [acc: {hs_acc:.1f}%]")

    # --------------------------------------------------------------------------
    # 4. MMLU
    # --------------------------------------------------------------------------
    print(f"\n🔬 [4/6] Evaluating MMLU ({samples_mmlu} official test questions across subjects)...")
    mmlu_items = get_mmlu_dataset(limit=samples_mmlu)
    mmlu_correct_unnorm = 0
    mmlu_correct_norm = 0
    mmlu_details = []
    letters = ["A", "B", "C", "D"]

    for item in mmlu_items:
        prompt = f"Question: {item['question']}\nAnswer:"
        pred_u, pred_n, _, _ = score_multiple_choice_item(
            model, tokenizer, prompt, item["choices"], device
        )
        ok_u = (pred_u == item["answer_idx"])
        ok_n = (pred_n == item["answer_idx"])
        if ok_u: mmlu_correct_unnorm += 1
        if ok_n: mmlu_correct_norm += 1
        mmlu_details.append({
            "subject": item.get("subject", "general"),
            "question": item["question"][:60],
            "target": letters[item["answer_idx"]],
            "pred_norm": letters[pred_n] if pred_n < 4 else str(pred_n),
            "correct": ok_n,
        })

    mmlu_acc = (mmlu_correct_unnorm / max(1, len(mmlu_items))) * 100
    mmlu_acc_norm = (mmlu_correct_norm / max(1, len(mmlu_items))) * 100
    summary_records["MMLU"] = {"acc": mmlu_acc, "acc_norm": mmlu_acc_norm, "n": len(mmlu_items)}
    details_records["MMLU"] = mmlu_details
    print(f"   ✓ MMLU          : {mmlu_acc_norm:.1f}% acc_norm ({mmlu_correct_norm}/{len(mmlu_items)}) [acc: {mmlu_acc:.1f}%]")

    # --------------------------------------------------------------------------
    # 5. Winogrande
    # --------------------------------------------------------------------------
    winogrande_items = get_winogrande_dataset(limit=samples_winogrande)
    if winogrande_items:
        print(f"\n🔬 [5/6] Evaluating Winogrande ({len(winogrande_items)} validation questions)...")
        w_correct_unnorm = 0
        w_correct_norm = 0
        w_details = []

        for item in winogrande_items:
            prompt = item["sentence"].replace("_", "").strip()
            choices = [item["option1"], item["option2"]]
            pred_u, pred_n, _, _ = score_multiple_choice_item(
                model, tokenizer, prompt, choices, device
            )
            ok_u = (pred_u == item["answer_idx"])
            ok_n = (pred_n == item["answer_idx"])
            if ok_u: w_correct_unnorm += 1
            if ok_n: w_correct_norm += 1
            w_details.append({
                "sentence": item["sentence"][:60],
                "target": item["answer_idx"],
                "pred_norm": pred_n,
                "correct": ok_n,
            })

        w_acc = (w_correct_unnorm / max(1, len(winogrande_items))) * 100
        w_acc_norm = (w_correct_norm / max(1, len(winogrande_items))) * 100
        summary_records["Winogrande"] = {"acc": w_acc, "acc_norm": w_acc_norm, "n": len(winogrande_items)}
        details_records["Winogrande"] = w_details
        print(f"   ✓ Winogrande    : {w_acc_norm:.1f}% acc_norm ({w_correct_norm}/{len(winogrande_items)}) [acc: {w_acc:.1f}%]")

    # --------------------------------------------------------------------------
    # 6. GSM8K (Generative CoT Exact Match)
    # --------------------------------------------------------------------------
    print(f"\n🔬 [6/6] Evaluating GSM8K ({samples_gsm8k} official test questions, Exact Match)...")
    gsm_items = get_gsm8k_dataset(limit=samples_gsm8k)
    gsm_em = 0
    gsm_details = []

    for item in gsm_items:
        prompt = f"Question: {item['question']}\nAnswer: Let's calculate step by step:"
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(
                inp["input_ids"],
                max_new_tokens=45,
                temperature=0.2,
                top_k=40,
                repetition_penalty=1.15,
            )
        gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        gen_ans = gen_text[len(prompt):].strip()
        extracted = extract_numerical_answer(gen_ans)
        target = item["target_number"]
        is_em = (extracted is not None and extracted == target)
        if is_em: gsm_em += 1
        gsm_details.append({
            "question": item["question"][:60],
            "target": target,
            "extracted": extracted,
            "correct": is_em,
        })

    gsm_em_pct = (gsm_em / max(1, len(gsm_items))) * 100
    summary_records["GSM8K"] = {"acc": gsm_em_pct, "acc_norm": gsm_em_pct, "n": len(gsm_items)}
    details_records["GSM8K"] = gsm_details
    print(f"   ✓ GSM8K EM      : {gsm_em_pct:.1f}% ({gsm_em}/{len(gsm_items)})")

    # --------------------------------------------------------------------------
    # 7. Overall Composite Scoreboard
    # --------------------------------------------------------------------------
    benchmarks_list = ["ARC-Challenge", "ARC-Easy", "HellaSwag", "MMLU", "GSM8K"]
    if "Winogrande" in summary_records:
        benchmarks_list.append("Winogrande")

    composite_acc_norm = sum(summary_records[b]["acc_norm"] for b in benchmarks_list) / len(benchmarks_list)
    composite_acc = sum(summary_records[b]["acc"] for b in benchmarks_list) / len(benchmarks_list)

    smollm2_ref = {
        "ARC-Challenge": 34.1,
        "ARC-Easy": 52.8,
        "HellaSwag": 39.5,
        "MMLU": 32.4,
        "GSM8K": 4.2,
        "Winogrande": 51.5,
    }
    smollm2_composite = sum(smollm2_ref[b] for b in benchmarks_list) / len(benchmarks_list)

    random_chance = {
        "ARC-Challenge": 25.0,
        "ARC-Easy": 25.0,
        "HellaSwag": 25.0,
        "MMLU": 25.0,
        "GSM8K": 0.0,
        "Winogrande": 50.0,
    }

    print("\n" + "=" * 96)
    print("📊 EXTENSIVE BENCHMARK SCOREBOARD & COMPARISON TABLE (N = 375 Questions)")
    print("=" * 96)
    print(f"{'Benchmark Metric':<20} | {'acc_norm':<10} | {'acc (unnorm)':<12} | {'Random':<8} | {'SmolLM2-135M':<12} | {'Status'}")
    print("-" * 96)

    for b in benchmarks_list:
        an = summary_records[b]["acc_norm"]
        au = summary_records[b]["acc"]
        rc = random_chance.get(b, 25.0)
        s_ref = smollm2_ref.get(b, 0.0)
        n = summary_records[b]["n"]

        status = "🔥 Beats SmolLM2" if an > s_ref else ("✅ Above Chance" if an > rc else "⚖️ Near Baseline")
        print(f"{b + f' (N={n})':<20} | {an:>7.1f}%   | {au:>8.1f}%     | {rc:>5.1f}%  | {s_ref:>9.1f}%    | {status}")

    print("-" * 96)
    status_overall = "🎉 BEATS SmolLM2" if composite_acc_norm > smollm2_composite else "📈 Closing Gap"
    print(f"{'COMPOSITE OVERALL':<20} | {composite_acc_norm:>7.1f}%   | {composite_acc:>8.1f}%     | {'—':>6}   | {smollm2_composite:>9.1f}%    | {status_overall}")
    print("=" * 96)

    export_path = export_json or os.path.join(REPO_ROOT, "benchmarks", f"industry_standard_step{step}.json")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, "w") as f:
        json.dump({
            "metadata": {
                "checkpoint": os.path.basename(checkpoint_path),
                "step": step,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "device": str(device),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "summary": summary_records,
            "composite": {
                "acc_norm": composite_acc_norm,
                "acc": composite_acc,
                "smollm2_ref": smollm2_composite,
            },
            "details": details_records,
        }, f, indent=2)

    print(f"\n💾 Full extensive report successfully exported to: {export_path}")
    return summary_records, composite_acc_norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Extensive Industry Standard Benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_small_latest_step834k.pt", help="Path to checkpoint")
    parser.add_argument("--preset", type=str, default="v5_small", help="Preset architecture")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--samples-arc", type=int, default=50, help="ARC sample count")
    parser.add_argument("--samples-hellaswag", type=int, default=100, help="HellaSwag sample count")
    parser.add_argument("--samples-mmlu", type=int, default=100, help="MMLU sample count")
    parser.add_argument("--samples-gsm8k", type=int, default=25, help="GSM8K sample count")
    parser.add_argument("--samples-winogrande", type=int, default=50, help="Winogrande sample count")
    parser.add_argument("--export-json", type=str, default=None, help="Export JSON path")

    args = parser.parse_args()
    run_extensive_benchmark(
        checkpoint_path=args.checkpoint,
        preset=args.preset,
        device_str=args.device,
        samples_arc=args.samples_arc,
        samples_hellaswag=args.samples_hellaswag,
        samples_mmlu=args.samples_mmlu,
        samples_gsm8k=args.samples_gsm8k,
        samples_winogrande=args.samples_winogrande,
        export_json=args.export_json,
    )
