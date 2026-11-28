#!/usr/bin/env python3
"""
scratch/controlled_readout_finetune_comparison.py

Controlled Head-to-Head Fine-Tuning Comparison:
Branch A: Baseline Legacy Readout (use_centralized_readout=False)
Branch B: Centralized Bottleneck Readout (use_centralized_readout=True)

Starting from identical checkpoint weights (checkpoints/fresh_scratch/v5_small_latest.pt),
both models receive the exact identical batches of reasoning & domain knowledge data.
Tracks:
1. Training loss trajectory over 150 steps.
2. Held-out validation loss across 4 domains (Python, STEM, Math, Prose).
3. Gradient routing: does the centralized bottleneck increase learning signal to Shell 0?
4. Inward radial flux dynamics before and after training.
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import copy
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.corpus import build_tokenizer

# Fixed evaluation domains
DOMAINS = {
    "python_code": (
        "def quicksort(arr: list[int]) -> list[int]:\n"
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    pivot = arr[len(arr) // 2]\n"
        "    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n"
        "    right = [x for x in arr if x > pivot]\n"
        "    return quicksort(left) + middle + quicksort(right)\n"
    ),
    "stem_biochem": (
        "ATP synthase catalyzes the synthesis of adenosine triphosphate from ADP and inorganic phosphate. "
        "The enzyme operates via a rotary catalytic mechanism driven by the transmembrane proton-motive force. "
        "The F0 domain conducts protons across the mitochondrial inner membrane, inducing mechanical torque "
        "transmitted through the gamma-epsilon central stalk to the catalytic beta subunits of F1."
    ),
    "mathematics": (
        "Let V be a finite-dimensional inner product space over the complex field. "
        "A linear operator T on V is normal if and only if it commutes with its adjoint. "
        "By the spectral theorem, T is normal if and only if there exists an orthonormal basis of V "
        "consisting of eigenvectors of T, with corresponding eigenvalues representing diagonal entries."
    ),
    "general_prose": (
        "The historical development of urban infrastructure in nineteenth-century Europe "
        "reflected deep socio-economic transformations. As industrial centers expanded rapidly, "
        "municipal governments were forced to implement centralized sewage, clean water aqueducts, "
        "and wide boulevards to mitigate public health crises and facilitate commercial transport."
    ),
}

def load_training_samples(jsonl_path: str, tokenizer, max_samples: int = 400, max_len: int = 128) -> list:
    """Loads and tokenizes prompt-target pairs from spheroid formatted jsonl."""
    samples = []
    if not os.path.exists(jsonl_path):
        print(f"Warning: {jsonl_path} not found.")
        return samples
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                prompt = row.get("prompt", "")
                answer = row.get("answer", "")
                text = row.get("text", "")
                if not text:
                    if prompt or answer:
                        text = (prompt + "\n" + answer).strip()
                    elif "messages" in row:
                        text = "\n".join([m.get("content", "") for m in row["messages"]])
                if len(text) < 40:
                    continue
                toks = tokenizer.encode(text, add_special_tokens=True)
                if len(toks) > max_len:
                    toks = toks[:max_len]
                if len(toks) >= 32:
                    samples.append(torch.tensor(toks, dtype=torch.long))
                    if len(samples) >= max_samples:
                        break
            except Exception:
                continue
    return samples

def evaluate_domains(model: CerebroSpheroidV5, tokenizer, device: torch.device) -> Dict[str, float]:
    """Evaluates cross-entropy loss across the four test domains."""
    model.eval()
    losses = {}
    with torch.no_grad():
        for name, text in DOMAINS.items():
            toks = torch.tensor([tokenizer.encode(text, add_special_tokens=True)], dtype=torch.long, device=device)
            prompts = toks[:, :-1]
            targets = toks[:, 1:]
            loss, _ = model(prompts, targets=targets)
            losses[name] = float(loss.item())
    return losses

def measure_grad_norms(model: CerebroSpheroidV5) -> Dict[str, float]:
    """Measures component gradient norms."""
    norms = {}
    # 1. Total norm
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().data.norm(2).item() ** 2
    norms["total"] = math.sqrt(total_sq)

    # 2. Motor Query Proj norm
    if model.motor_decoder.q_proj.weight.grad is not None:
        norms["motor_q_proj"] = float(model.motor_decoder.q_proj.weight.grad.norm(2).item())

    # 3. Core Radial Inward norm
    if model.volumetric_core.radial_in_proj.weight.grad is not None:
        norms["radial_in_proj"] = float(model.volumetric_core.radial_in_proj.weight.grad.norm(2).item())

    # 4. Core Shell 0 Intra SwiGLU norm
    s0_grad_sq = 0.0
    sw0 = model.volumetric_core.intra_shell_swiglu[0]
    for p in sw0.parameters():
        if p.grad is not None:
            s0_grad_sq += p.grad.detach().data.norm(2).item() ** 2
    norms["shell_0_swiglu"] = math.sqrt(s0_grad_sq)

    # 5. Core Shell 15 Intra SwiGLU norm
    s15_grad_sq = 0.0
    sw15 = model.volumetric_core.intra_shell_swiglu[-1]
    for p in sw15.parameters():
        if p.grad is not None:
            s15_grad_sq += p.grad.detach().data.norm(2).item() ** 2
    norms["shell_15_swiglu"] = math.sqrt(s15_grad_sq)

    return norms

def train_branch(
    branch_name: str,
    cfg,
    raw_ckpt,
    training_data: list,
    batch_indices: list,
    device: torch.device,
    tokenizer,
    steps: int = 150,
    lr: float = 1e-4,
) -> Tuple[Dict[str, float], Dict[str, float], list]:
    """Trains a single model branch, measuring initial and final held-out losses, and gradient norms."""
    print(f"\n=======================================================")
    print(f"🚀 TRAINING {branch_name.upper()} (use_centralized_readout={cfg.use_centralized_readout})")
    print(f"=======================================================")

    model = CerebroSpheroidV5(config=cfg).to(device)
    load_v5_checkpoint_into_torch(model, raw_ckpt, verbose=False)

    eval_init = evaluate_domains(model, tokenizer, device)
    print(f"Initial held-out losses: {eval_init}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()

    logs = []
    t0 = time.time()
    for step in range(1, steps + 1):
        idxs = batch_indices[step - 1]
        batch_seqs = [training_data[i] for i in idxs]
        max_l = max(s.size(0) for s in batch_seqs)
        padded = torch.zeros(len(batch_seqs), max_l, dtype=torch.long, device=device)
        for b_i, s in enumerate(batch_seqs):
            padded[b_i, :s.size(0)] = s

        prompts = padded[:, :-1]
        targets = padded[:, 1:]

        opt.zero_grad(set_to_none=True)
        loss, _ = model(prompts, targets=targets, pad_id=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        norms = measure_grad_norms(model)
        opt.step()

        logs.append({"step": step, "loss": float(loss.item()), **norms})

        if step % 25 == 0 or step == steps:
            dt = time.time() - t0
            print(f"  Step {step:3d}/{steps} | Loss: {loss.item():.4f} | "
                  f"Grad S0: {norms['shell_0_swiglu']:.4f} | Grad S15: {norms['shell_15_swiglu']:.4f} | {dt:.1f}s")

    eval_final = evaluate_domains(model, tokenizer, device)
    print(f"Final held-out losses: {eval_final}")

    # Clean up GPU memory completely
    del model
    del opt
    torch.cuda.empty_cache()

    return eval_init, eval_final, logs

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"================================================================================")
    print(f"🧠 CONTROLLED EXPERIMENT D: CENTRALIZED READOUT VS BASELINE LEGACY")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"================================================================================\n")

    ckpt_path = "checkpoints/fresh_scratch/v5_small_latest.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/v5_small_latest.pt"
    print(f"Loading base checkpoint: {ckpt_path}")
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    tokenizer = build_tokenizer("llama")

    # Load shared training data
    data_path = "data/consolidated/03_complex_reasoning/complex_reasoning_spheroid.jsonl"
    training_data = load_training_samples(data_path, tokenizer, max_samples=400, max_len=128)
    if not training_data:
        data_path = "data/consolidated/04_domain_knowledge/domain_knowledge_spheroid.jsonl"
        training_data = load_training_samples(data_path, tokenizer, max_samples=400, max_len=128)
    print(f"Loaded {len(training_data)} training sequences.")

    steps = 150
    batch_size = 2
    lr = 1e-4

    torch.manual_seed(42)
    batch_indices = [torch.randint(0, len(training_data), (batch_size,)).tolist() for _ in range(steps)]

    # Branch A config: Baseline Legacy Readout
    cfg_a = infer_v5_config_from_state_dict(raw_ckpt)
    cfg_a.use_per_token_efference = False
    cfg_a.use_centralized_readout = False

    # Branch B config: Centralized Bottleneck Readout
    cfg_b = infer_v5_config_from_state_dict(raw_ckpt)
    cfg_b.use_per_token_efference = False
    cfg_b.use_centralized_readout = True

    # Run Branch A sequentially
    eval_a_0, eval_a_end, logs_a = train_branch(
        "Branch A (Legacy)", cfg_a, raw_ckpt, training_data, batch_indices, device, tokenizer, steps=steps, lr=lr
    )

    # Run Branch B sequentially with identical batches and seed
    eval_b_0, eval_b_end, logs_b = train_branch(
        "Branch B (Centralized Bottleneck)", cfg_b, raw_ckpt, training_data, batch_indices, device, tokenizer, steps=steps, lr=lr
    )

    # Comparative Summary
    print("\n" + "=" * 95)
    print("FINAL POST-FINE-TUNING COMPARATIVE EVALUATION RESULTS (150 STEPS)")
    print("=" * 95)
    print(f"{'Domain':<18s} | {'Branch A Init':<14s} -> {'Branch A Final':<14s} | {'Branch B Init':<14s} -> {'Branch B Final':<14s} | {'Advantage (B vs A)':<20s}")
    print("-" * 95)
    for d in DOMAINS.keys():
        d_a = ((eval_a_end[d] - eval_a_0[d]) / eval_a_0[d]) * 100
        d_b = ((eval_b_end[d] - eval_b_0[d]) / eval_b_0[d]) * 100
        final_b_vs_a = ((eval_b_end[d] - eval_a_end[d]) / eval_a_end[d]) * 100
        print(f"{d:<18s} | {eval_a_0[d]:<6.4f} -> {eval_a_end[d]:<6.4f} ({d_a:+.2f}%) | "
              f"{eval_b_0[d]:<6.4f} -> {eval_b_end[d]:<6.4f} ({d_b:+.2f}%) | "
              f"{final_b_vs_a:+.2f}%")
    print("=" * 95)

    # Aggregate Gradient Flux Analysis
    avg_s0_grad_a = sum(l["shell_0_swiglu"] for l in logs_a) / len(logs_a)
    avg_s0_grad_b = sum(l["shell_0_swiglu"] for l in logs_b) / len(logs_b)
    avg_s15_grad_a = sum(l["shell_15_swiglu"] for l in logs_a) / len(logs_a)
    avg_s15_grad_b = sum(l["shell_15_swiglu"] for l in logs_b) / len(logs_b)
    ratio_a = avg_s0_grad_a / (avg_s15_grad_a + 1e-8)
    ratio_b = avg_s0_grad_b / (avg_s15_grad_b + 1e-8)

    print("\n--- Structural Optimization Gradient Routing ---")
    print(f"Shell 0 Mean Gradient Norm   : Branch A = {avg_s0_grad_a:.4f}  | Branch B = {avg_s0_grad_b:.4f}  ({(avg_s0_grad_b/avg_s0_grad_a - 1)*100:+.2f}%)")
    print(f"Shell 15 Mean Gradient Norm  : Branch A = {avg_s15_grad_a:.4f}  | Branch B = {avg_s15_grad_b:.4f}  ({(avg_s15_grad_b/avg_s15_grad_a - 1)*100:+.2f}%)")
    print(f"Core/Surface Gradient Ratio  : Branch A = {ratio_a:.4f}  | Branch B = {ratio_b:.4f}  ({(ratio_b/ratio_a - 1)*100:+.2f}%)")
    print("=" * 95)

if __name__ == "__main__":
    main()
