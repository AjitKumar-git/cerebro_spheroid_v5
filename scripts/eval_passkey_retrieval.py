#!/usr/bin/env python3
"""
🔑 Long-Context Passkey Retrieval Evaluation Suite for CerebroSpheroid V5
========================================================================
Rigorous empirical benchmark testing contextual information retention across
multiple context lengths (256 -> 512 -> 1024 -> 2048 -> 4096 tokens)
and needle placement depths (Beginning: 10%, Middle: 50%, End: 90%).

Prevents unsubstantiated long-context claims by measuring exact retrieval accuracy.
"""

import os
import sys
import argparse
import random
import re
import time
from typing import Dict, Any, List, Tuple

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
from transformers import AutoTokenizer

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, infer_v5_config_from_state_dict
from src.corpus import format_stimulus_prompt, parse_model_output

FILLER_SENTENCES = [
    "The sky above the port was the color of television, tuned to a dead channel.",
    "The quick brown fox jumps over the lazy dog in the tranquil afternoon breeze.",
    "A journey of a thousand miles begins with a single deliberate step forward.",
    "Deep inside the concentric spherical laminae, cortical representations evolve smoothly.",
    "Scientific inquiry demands transparent empirical verification rather than assumptions.",
    "The invariant manifold preserves relational topology across continuous phase transformations.",
    "Geometric deep learning leverages non-Euclidean symmetries for compact neural representation.",
    "Information flow through axonal shortcut tracts balances local convection and global integration.",
]


def generate_filler_text(target_tokens: int, tokenizer: Any) -> str:
    """Generates coherent filler text of approximately target_tokens length."""
    words_per_token_approx = 0.75
    text_chunks = []
    current_tokens = 0
    rng = random.Random(42)

    while current_tokens < target_tokens:
        sentence = rng.choice(FILLER_SENTENCES)
        text_chunks.append(sentence)
        current_tokens += len(tokenizer.encode(" " + sentence, add_special_tokens=False))

    return " ".join(text_chunks)


def build_passkey_prompt(
    total_tokens: int,
    depth: float,
    passkey: str,
    tokenizer: Any,
) -> Tuple[str, str]:
    """
    Constructs a document of exact target token length containing a hidden passkey at depth.
    Depth is in [0.0, 1.0]: 0.1 = beginning, 0.5 = middle, 0.9 = end.
    """
    prefix_inst = "There is an important secret hidden in the following document. Memorize it carefully.\n\n"
    needle = f"The secret passkey is {passkey}. Remember this passkey.\n"
    question = "\nWhat is the secret passkey? The secret passkey is"

    inst_tokens = len(tokenizer.encode(prefix_inst, add_special_tokens=False))
    needle_tokens = len(tokenizer.encode(needle, add_special_tokens=False))
    q_tokens = len(tokenizer.encode(question, add_special_tokens=False))

    overhead = inst_tokens + needle_tokens + q_tokens
    filler_budget = max(0, total_tokens - overhead)

    tokens_before = int(filler_budget * depth)
    tokens_after = filler_budget - tokens_before

    filler_before = generate_filler_text(tokens_before, tokenizer)
    filler_after = generate_filler_text(tokens_after, tokenizer)

    # Trim filler tokens to exact count
    ids_before = tokenizer.encode(filler_before, add_special_tokens=False)[:tokens_before]
    ids_after = tokenizer.encode(filler_after, add_special_tokens=False)[:tokens_after]

    text_before = tokenizer.decode(ids_before)
    text_after = tokenizer.decode(ids_after)

    full_prompt = f"{prefix_inst}{text_before}\n{needle}\n{text_after}{question}"
    return full_prompt, passkey


def evaluate_passkey_matrix(
    model: CerebroSpheroidV5,
    tokenizer: Any,
    lengths: List[int],
    depths: List[float],
    trials_per_cell: int = 3,
    device: torch.device = torch.device("cpu"),
    digits: int = 5,
) -> Dict[str, Any]:
    """Runs the full (length x depth) evaluation matrix."""
    depth_labels = {0.1: "Beginning (10%)", 0.5: "Middle (50%)", 0.9: "End (90%)"}
    results_matrix: Dict[int, Dict[float, float]] = {}
    details: List[Dict[str, Any]] = []

    print("\n" + "=" * 78)
    print(f"🔑 CerebroSpheroid V5 Long-Context Passkey Retrieval Benchmark ({digits}-digit passkeys)")
    print(f"   Tested Context Lengths: {lengths}")
    print(f"   Tested Placement Depths: {[depth_labels.get(d, f'{int(d*100)}%') for d in depths]}")
    print(f"   Trials per Cell: {trials_per_cell}")
    print("=" * 78)

    model.eval()
    total_cells = len(lengths) * len(depths)
    cell_idx = 0

    for length in lengths:
        results_matrix[length] = {}
        for depth in depths:
            cell_idx += 1
            label = depth_labels.get(depth, f"{int(depth*100)}%")
            correct_count = 0
            trial_times = []

            for trial in range(trials_per_cell):
                if digits == 1:
                    passkey = f"{random.randint(1, 9)}"
                elif digits == 2:
                    passkey = f"{random.randint(10, 99)}"
                elif digits == 3:
                    passkey = f"{random.randint(100, 999)}"
                elif digits == 4:
                    passkey = f"{random.randint(1000, 9999)}"
                else:
                    passkey = f"{random.randint(10000, 99999)}"
                prompt, target = build_passkey_prompt(length, depth, passkey, tokenizer)

                t0 = time.time()
                with torch.no_grad():
                    output = model.generate_text(
                        prompt=prompt,
                        tokenizer=tokenizer,
                        max_new_tokens=16,
                        temperature=0.01,
                        device=device,
                        return_parsed=False,
                    )
                elapsed = time.time() - t0
                trial_times.append(elapsed)

                response_text = str(output).strip()
                # Check for exact passkey presence
                hit = (target in response_text) or bool(re.search(rf"\b{target}\b", response_text))
                if hit:
                    correct_count += 1

                details.append({
                    "length": length,
                    "depth": depth,
                    "trial": trial + 1,
                    "passkey": passkey,
                    "response": response_text,
                    "hit": hit,
                    "latency": elapsed,
                })

            acc = (correct_count / trials_per_cell) * 100.0
            results_matrix[length][depth] = acc
            mean_time = sum(trial_times) / max(len(trial_times), 1)
            print(f"[{cell_idx:2d}/{total_cells}] Length: {length:5d} tokens | Depth: {label:<16} | Acc: {acc:5.1f}% ({correct_count}/{trials_per_cell}) | {mean_time:.2f}s/trial")

    # Display Final Markdown Table
    print("\n" + "=" * 78)
    print("📊 PASSKEY RETRIEVAL ACCURACY MATRIX (%)")
    print("=" * 78)
    headers = ["Context Length"] + [depth_labels.get(d, f"{int(d*100)}%") for d in depths] + ["Mean Acc"]
    row_fmt = "{:<16}" + "{:<18}" * len(depths) + "{:<12}"
    print(row_fmt.format(*headers))
    print("-" * 78)

    all_accs = []
    for length in lengths:
        row_accs = [results_matrix[length][d] for d in depths]
        mean_row = sum(row_accs) / len(row_accs)
        all_accs.append(mean_row)
        formatted_row = [f"{length} tokens"] + [f"{acc:5.1f}%" for acc in row_accs] + [f"{mean_row:5.1f}%"]
        print(row_fmt.format(*formatted_row))

    print("-" * 78)
    overall_mean = sum(all_accs) / max(len(all_accs), 1)
    print(f"Overall Retrieval Score: {overall_mean:5.1f}%\n")

    return {
        "matrix": results_matrix,
        "overall_accuracy": overall_mean,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate passkey retrieval on CerebroSpheroid V5.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_small_agentic_tool_aligned.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--lengths", type=str, default="256,512,1024,2048,4096", help="Comma-separated context lengths (default: 256,512,1024,2048,4096)")
    parser.add_argument("--depths", type=str, default="0.1,0.5,0.9", help="Comma-separated needle depths (default: 0.1,0.5,0.9)")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per cell")
    parser.add_argument("--digits", type=int, default=5, choices=[1, 2, 3, 4, 5], help="Number of digits in passkey (default: 5)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--tokenizer", type=str, default="auto", help="Tokenizer name, path, or 'auto' (detects 32k vs Qwen from config)")
    parser.add_argument("--use-rope", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=None, help="Override use_rope setting in config (True/False)")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save JSON results")
    args = parser.parse_args()

    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    depths = [float(x.strip()) for x in args.depths.split(",") if x.strip()]
    device = torch.device(args.device)

    ckpt_path = os.path.join(WORKSPACE_ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"⚙️ Loading model checkpoint: {ckpt_path} onto {device}...")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_d = raw.get("config", {})
    if isinstance(cfg_d, dict):
        valid_keys = set(ModelConfigV5.__dataclass_fields__.keys())
        valid_cfg = {k: v for k, v in cfg_d.items() if k in valid_keys}
        cfg = ModelConfigV5(**valid_cfg)
    elif isinstance(cfg_d, ModelConfigV5):
        cfg = cfg_d
    else:
        cfg = infer_v5_config_from_state_dict(raw, default_preset="v5_small")

    if args.use_rope is not None:
        cfg.use_rope = args.use_rope

    # Resolve Tokenizer matching model vocab_size
    if args.tokenizer == "auto":
        if cfg.vocab_size == 32000:
            print("⚙️ Auto-detected vocab_size 32,000 -> Loading LLaMA 32k tokenizer...")
            from src.corpus import build_tokenizer
            tokenizer = build_tokenizer("llama")
        else:
            print(f"⚙️ Auto-detected vocab_size {cfg.vocab_size} -> Loading Qwen/Qwen2.5-0.5B tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    else:
        print(f"⚙️ Loading tokenizer: {args.tokenizer}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            print(f"⚠️ Failed to load {args.tokenizer}, falling back to build_tokenizer(): {e}")
            from src.corpus import build_tokenizer
            tokenizer = build_tokenizer("llama")

    model = CerebroSpheroidV5(cfg)
    model_state = raw.get("model_state", raw.get("state_dict", raw))
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully: {cfg.surface_dim}D, {cfg.num_shells} shells, {cfg.num_heads} heads, use_rope={cfg.use_rope}")

    eval_res = evaluate_passkey_matrix(
        model=model,
        tokenizer=tokenizer,
        lengths=lengths,
        depths=depths,
        trials_per_cell=args.trials,
        device=device,
        digits=args.digits,
    )

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        import json
        with open(args.output_json, "w") as f:
            json.dump(eval_res, f, indent=2)
        print(f"💾 Results saved to JSON: {args.output_json}")


if __name__ == "__main__":
    main()
