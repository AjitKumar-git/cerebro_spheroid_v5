#!/usr/bin/env python3
"""
Cerebro-1000 Benchmark Evaluation Runner
========================================
Runs standardized evaluation of CerebroSpheroid V5 models against the 1,000-prompt
benchmark suite across 5 cognitive pillars:

1. linguistic_and_syntax (200 prompts)
2. factual_and_counterfactual (200 prompts)
3. mathematical_and_logic (200 prompts)
4. code_and_algorithms (200 prompts)
5. needle_and_agentic (200 prompts)

Outputs metrics: Exact-match accuracy, regex accuracy, action classification accuracy,
cross-entropy loss, perplexity, and token generation throughput.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from src.agentic_harness import AgenticSpheroidHarness, ActionType, ACTION_NAMES
from src.corpus import format_stimulus_prompt, format_motor_response, encode_no_special

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


class FallbackTokenizer:
    """Byte-Fallback ASCII Tokenizer if transformers is absent."""
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = [b % self.vocab_size for b in text.encode("utf-8", errors="replace")]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        clean = [t for t in tokens if t not in (0, 1, 2)] if skip_special_tokens else tokens
        return bytes([t % 256 for t in clean]).decode("utf-8", errors="replace")


def get_tokenizer():
    try:
        from src.corpus import build_tokenizer
        return build_tokenizer("llama")
    except Exception:
        return FallbackTokenizer(vocab_size=32000)


def evaluate_benchmark(
    checkpoint_path: str,
    benchmark_path: str,
    num_samples: Optional[int] = None,
    category_filter: Optional[str] = None,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "eval/results",
    max_gen_tokens: int = 32,
    verbose: bool = False,
) -> Dict[str, Any]:
    device = torch.device(device_str)
    print("=" * 75)
    print("🧠 CEREBRO-1000 BENCHMARK EVALUATION ENGINE")
    print("=" * 75)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Benchmark Suite: {benchmark_path}")

    tokenizer = get_tokenizer()
    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    print(f"Tokenizer Vocab Size: {vocab_size}")

    # Load prompts
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}. Run build_benchmark_1000.py first!")

    prompts_data = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if category_filter and item.get("category") != category_filter:
                    continue
                prompts_data.append(item)

    if num_samples is not None and num_samples > 0:
        prompts_data = prompts_data[:num_samples]

    print(f"Loaded {len(prompts_data)} benchmark evaluation prompts.")

    # Load Model (auto-detect preset & architecture from checkpoint if available)
    from src.spheroid_v5 import infer_v5_config_from_state_dict, load_v5_checkpoint_into_torch

    ckpt = None
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = infer_v5_config_from_state_dict(ckpt, default_preset="v5_small", vocab_size_override=vocab_size)
    else:
        preset = PRESETS_V5["v5_small"].copy()
        config = ModelConfigV5(**preset)
        config.vocab_size = vocab_size

    model = CerebroSpheroidV5(config).to(device)
    harness = AgenticSpheroidHarness(model).to(device)

    if ckpt is not None:
        load_v5_checkpoint_into_torch(model, ckpt, verbose=True)
        act_state = ckpt.get("action_head_state_dict", ckpt.get("action_head_state"))
        if isinstance(act_state, dict):
            harness.action_head.load_state_dict(act_state, strict=False)
        print("✓ Model weights loaded successfully.")
    else:
        print(f"⚠️ Checkpoint not found at {checkpoint_path}. Evaluating base initialized weights.")

    model.eval()
    harness.eval()

    # Metrics containers
    category_stats = defaultdict(lambda: {"total": 0, "regex_matches": 0, "exact_matches": 0, "action_matches": 0, "losses": []})
    total_start_time = time.time()
    total_generated_tokens = 0

    results_log = []

    print("-" * 75)
    print(f"Executing evaluation across {len(prompts_data)} items...")
    print("-" * 75)

    for idx, item in enumerate(prompts_data, 1):
        pid = item["id"]
        cat = item["category"]
        prompt_text = item["prompt"]
        ref_ans = item["reference_answer"]
        v_regex = item.get("verification_regex", "")
        exp_action = item.get("expected_action", "chat")

        stats = category_stats[cat]
        stats["total"] += 1

        # 1. Target Cross-Entropy Loss computation (Strategy A delimited format matching training)
        formatted_p = format_stimulus_prompt(prompt_text)
        formatted_a = format_motor_response(ref_ans)
        ans_tokens = encode_no_special(tokenizer, formatted_a) if hasattr(tokenizer, "encode") else [1]
        p_tokens = encode_no_special(tokenizer, formatted_p) if hasattr(tokenizer, "encode") else [1]
        
        loss_val = None
        if ans_tokens and p_tokens:
            p_tensor = torch.tensor([[t % vocab_size for t in p_tokens]], dtype=torch.long, device=device)
            t_tensor = torch.tensor([[t % vocab_size for t in ans_tokens]], dtype=torch.long, device=device)
            with torch.no_grad():
                try:
                    loss_out, _ = model(prompts=p_tensor, targets=t_tensor)
                    loss_val = float(loss_out.item())
                    stats["losses"].append(loss_val)
                except Exception:
                    pass

        # 2. Text Generation
        gen_start = time.time()
        with torch.no_grad():
            gen_text = model.generate_text(
                prompt=prompt_text,
                tokenizer=tokenizer,
                max_new_tokens=max_gen_tokens,
                temperature=0.3,
                top_k=20,
                device=device,
            )
        gen_duration = time.time() - gen_start
        gen_tok_count = len(gen_text.split())
        total_generated_tokens += gen_tok_count

        # 3. Action Classification via ToolActionHead
        with torch.no_grad():
            # State mantle representation after prompt ingestion
            state = model.init_spheroid_state(1, device)
            sensory_flux, _ = model.sensory_dnn(p_tensor)
            mantle = sensory_flux.mean(dim=1)
            action_logits, action_probs = harness.action_head(mantle)
            pred_action_id = ActionType(int(torch.argmax(action_probs, dim=-1).item()))
            pred_action_name = ACTION_NAMES.get(pred_action_id, "chat")

        # 4. Evaluation Matches
        is_exact = ref_ans.strip().lower() in gen_text.strip().lower()
        is_regex = bool(re.search(v_regex, gen_text, re.IGNORECASE)) if v_regex else is_exact
        is_action_match = (pred_action_name == exp_action.lower())

        if is_exact:
            stats["exact_matches"] += 1
        if is_regex:
            stats["regex_matches"] += 1
        if is_action_match:
            stats["action_matches"] += 1

        results_log.append({
            "id": pid,
            "category": cat,
            "prompt": prompt_text,
            "reference": ref_ans,
            "generated": gen_text,
            "loss": loss_val,
            "exact_match": is_exact,
            "regex_match": is_regex,
            "expected_action": exp_action,
            "predicted_action": pred_action_name,
            "action_match": is_action_match,
        })

        if verbose or idx % 50 == 0 or idx == len(prompts_data):
            loss_str = f"Loss: {loss_val:.3f}" if loss_val is not None else "Loss: N/A"
            reg_sym = "✓" if is_regex else "✗"
            act_sym = "✓" if is_action_match else "✗"
            print(f"[{idx:4d}/{len(prompts_data):4d}] [{cat[:16]:16s}] {loss_str} | Ans: {reg_sym} | Act: {act_sym} | {gen_text[:40]}")

    total_time = time.time() - total_start_time
    tok_per_sec = total_generated_tokens / max(total_time, 1e-4)

    # Compile Summary
    print("\n" + "=" * 75)
    print("📊 CEREBRO-1000 BENCHMARK FINAL REPORT")
    print("=" * 75)
    print(f"{'Pillar Category':<30s} | {'Count':<5s} | {'Regex Acc':<10s} | {'Exact Acc':<10s} | {'Act Acc':<8s} | {'Avg Loss':<8s} | {'Perplexity':<8s}")
    print("-" * 92)

    total_prompts = 0
    total_regex = 0
    total_exact = 0
    total_action = 0
    all_losses = []

    summary_by_category = {}

    for cat, s in category_stats.items():
        n = s["total"]
        r_acc = (s["regex_matches"] / n) * 100 if n else 0
        e_acc = (s["exact_matches"] / n) * 100 if n else 0
        a_acc = (s["action_matches"] / n) * 100 if n else 0
        avg_l = sum(s["losses"]) / len(s["losses"]) if s["losses"] else 0
        ppl = math.exp(min(avg_l, 20.0)) if avg_l > 0 else 0

        total_prompts += n
        total_regex += s["regex_matches"]
        total_exact += s["exact_matches"]
        total_action += s["action_matches"]
        all_losses.extend(s["losses"])

        summary_by_category[cat] = {
            "count": n,
            "regex_accuracy_percent": round(r_acc, 2),
            "exact_accuracy_percent": round(e_acc, 2),
            "action_accuracy_percent": round(a_acc, 2),
            "avg_loss": round(avg_l, 4),
            "perplexity": round(ppl, 2),
        }
        print(f"{cat:<30s} | {n:<5d} | {r_acc:>8.1f}% | {e_acc:>8.1f}% | {a_acc:>6.1f}% | {avg_l:>8.3f} | {ppl:>8.2f}")

    print("-" * 92)
    overall_r_acc = (total_regex / total_prompts) * 100 if total_prompts else 0
    overall_e_acc = (total_exact / total_prompts) * 100 if total_prompts else 0
    overall_a_acc = (total_action / total_prompts) * 100 if total_prompts else 0
    overall_loss = sum(all_losses) / len(all_losses) if all_losses else 0
    overall_ppl = math.exp(min(overall_loss, 20.0)) if overall_loss > 0 else 0

    print(f"{'OVERALL BENCHMARK':<30s} | {total_prompts:<5d} | {overall_r_acc:>8.1f}% | {overall_e_acc:>8.1f}% | {overall_a_acc:>6.1f}% | {overall_loss:>8.3f} | {overall_ppl:>8.2f}")
    print(f"\nThroughput: {tok_per_sec:.2f} words/sec | Total Elapsed: {total_time:.1f}s")
    print("=" * 75)

    # Save output artifacts
    os.makedirs(output_dir, exist_ok=True)
    report_data = {
        "checkpoint": checkpoint_path,
        "total_prompts": total_prompts,
        "elapsed_seconds": round(total_time, 2),
        "overall": {
            "regex_accuracy_percent": round(overall_r_acc, 2),
            "exact_accuracy_percent": round(overall_e_acc, 2),
            "action_accuracy_percent": round(overall_a_acc, 2),
            "avg_loss": round(overall_loss, 4),
            "perplexity": round(overall_ppl, 2),
        },
        "by_category": summary_by_category,
        "item_results": results_log,
    }

    report_json_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save Markdown summary
    report_md_path = os.path.join(output_dir, "BENCHMARK_REPORT.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Cerebro-1000 Benchmark Evaluation Report\n\n")
        f.write(f"- **Evaluated Checkpoint:** `{checkpoint_path}`\n")
        f.write(f"- **Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total Prompts Evaluated:** {total_prompts}\n")
        f.write(f"- **Overall Loss:** {overall_loss:.4f} (Perplexity: {overall_ppl:.2f})\n")
        f.write(f"- **Overall Accuracy (Regex):** {overall_r_acc:.2f}%\n")
        f.write(f"- **Overall Action Accuracy:** {overall_a_acc:.2f}%\n\n")
        f.write("## Performance by Pillar\n\n")
        f.write("| Pillar Category | Evaluated | Regex Acc | Exact Acc | Action Acc | Loss | Perplexity |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for cat, s in summary_by_category.items():
            f.write(f"| **{cat}** | {s['count']} | {s['regex_accuracy_percent']}% | {s['exact_accuracy_percent']}% | {s['action_accuracy_percent']}% | {s['avg_loss']} | {s['perplexity']} |\n")
        f.write(f"| **TOTAL** | {total_prompts} | **{overall_r_acc:.2f}%** | **{overall_e_acc:.2f}%** | **{overall_a_acc:.2f}%** | **{overall_loss:.4f}** | **{overall_ppl:.2f}** |\n\n")

    print(f"✓ Detailed JSON Report saved to: {report_json_path}")
    print(f"✓ Markdown Summary saved to: {report_md_path}")
    return report_data


def main():
    parser = argparse.ArgumentParser(description="Run Cerebro-1000 Benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_all_parts_multimodal_tested.pt")
    parser.add_argument("--benchmark", type=str, default="eval/benchmark_1000.jsonl")
    parser.add_argument("--samples", type=int, default=50, help="Number of prompts to evaluate (default: 50, use 1000 for full suite)")
    parser.add_argument("--category", type=str, default=None, help="Filter to specific pillar category")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="eval/results")
    parser.add_argument("--verbose", action="store_true", help="Print details for every prompt")
    args = parser.parse_args()

    evaluate_benchmark(
        checkpoint_path=args.checkpoint,
        benchmark_path=args.benchmark,
        num_samples=args.samples,
        category_filter=args.category,
        device_str=args.device,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
