#!/usr/bin/env python3
"""
scratch/deliberation_dynamics_audit.py

Phase 1 Diagnostic Audit: Deliberation Dynamics & Iterative State Estimation
Measures mathematical behavior of internal thinking cycles (c = 0..8):
1. Relative Step Norm: delta_c = ||z_c - z_{c-1}|| / (||z_{c-1}|| + eps)
2. Directional Momentum & Oscillation: cos(Delta z_c, Delta z_{c-1})
3. Empirical Local Lipschitz / Spectral Bound: L_c = ||F(S + eta) - F(S)|| / ||eta||
4. Prediction Cross-Entropy Loss L_c and Token Entropy H_c per cycle.
Evaluated across 5 prompt complexities on the current checkpoint.
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.corpus import build_tokenizer

PROMPTS = {
    "trivial_fact": (
        "The capital of France is Paris. The city is renowned for its iconic Eiffel Tower, "
        "historic museums, and vibrant cultural heritage spanning centuries of European history."
    ),
    "general_prose": (
        "The historical development of urban infrastructure in nineteenth-century Europe "
        "reflected deep socio-economic transformations. As industrial centers expanded rapidly, "
        "municipal governments were forced to implement centralized sewage, clean water aqueducts, "
        "and wide boulevards to mitigate public health crises and facilitate commercial transport."
    ),
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
    "mathematics": (
        "Let V be a finite-dimensional inner product space over the complex field. "
        "A linear operator T on V is normal if and only if it commutes with its adjoint. "
        "By the spectral theorem, T is normal if and only if there exists an orthonormal basis of V "
        "consisting of eigenvectors of T, with corresponding eigenvalues representing diagonal entries."
    ),
    "stem_biochem": (
        "ATP synthase catalyzes the synthesis of adenosine triphosphate from ADP and inorganic phosphate. "
        "The enzyme operates via a rotary catalytic mechanism driven by the transmembrane proton-motive force. "
        "The F0 domain conducts protons across the mitochondrial inner membrane, inducing mechanical torque "
        "transmitted through the gamma-epsilon central stalk to the catalytic beta subunits of F1."
    ),
}

def measure_cycle_dynamics(
    model: CerebroSpheroidV5,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    max_cycles: int = 8,
    perturbation_scale: float = 1e-3,
) -> List[Dict[str, float]]:
    """Evaluates core state evolution, convergence metrics, and prediction loss at each cycle c in 0..max_cycles."""
    model.eval()
    dev = prompt_tokens.device
    B, P = prompt_tokens.shape
    T = target_tokens.shape[1]

    with torch.no_grad():
        # 1. Sensory Encoding
        sensory_flux, sensory_kv = model.sensory_dnn(prompt_tokens)
        prompt_centroid = model._recency_weighted_pool(sensory_flux)

        # Initialize laminar profile
        num_shells = model.config.num_shells
        radial_profile = (
            torch.arange(1, num_shells + 1, device=dev, dtype=torch.float32) / float(num_shells)
        ).view(1, -1, 1)
        state_0 = (prompt_centroid.unsqueeze(1).repeat(1, num_shells, 1).float() * radial_profile).to(
            prompt_centroid.dtype
        )

        # Track trajectory
        states = [state_0]
        cores = [state_0[:, 0, :].clone()]  # Shell 0 is deliberated core

        current_state = state_0.clone()
        lipschitz_bounds = [float("nan")]  # c=0 has no update

        for c in range(1, max_cycles + 1):
            # Evaluate Local Lipschitz / Sensitivity before applying update
            eta = torch.randn_like(current_state)
            eta = eta / (eta.norm(p=2, dim=-1, keepdim=True) + 1e-8) * perturbation_scale
            perturbed_out = model.volumetric_core.deliberate_step(current_state + eta)
            clean_out = model.volumetric_core.deliberate_step(current_state)

            diff_out = (perturbed_out - clean_out).norm(p=2, dim=-1).mean().item()
            l_bound = diff_out / perturbation_scale
            lipschitz_bounds.append(l_bound)

            # Apply sensory KV probe if configured
            if sensory_kv is not None and model.deliberation_probe is not None:
                keys, values = sensory_kv
                L_kv = keys.shape[2]
                probe_q = model.deliberation_probe(current_state.mean(dim=1)).view(
                    B, model.config.num_heads, 1, model.sensory_dnn.head_dim
                )
                scores = torch.matmul(probe_q.to(keys.dtype), keys.transpose(-1, -2)) / math.sqrt(
                    model.sensory_dnn.head_dim
                )
                if model.config.top_k_resonance > 0 and L_kv > model.config.top_k_resonance:
                    topk_v, _ = torch.topk(scores, model.config.top_k_resonance, dim=-1)
                    scores = torch.where(scores < topk_v[..., -1, None], torch.full_like(scores, -1e4), scores)
                attn_p = F.softmax(scores.float(), dim=-1).to(scores.dtype)
                retrieved = torch.matmul(attn_p, values).transpose(1, 2).contiguous().view(B, model.config.surface_dim)
                state_mantle = current_state[:, -1:, :] + 0.1 * model.deliberation_probe_norm(retrieved).unsqueeze(1)
                current_state = torch.cat([current_state[:, :-1, :], state_mantle], dim=1)

            # Deliberate step
            current_state = model.volumetric_core.deliberate_step(current_state)
            states.append(current_state)
            cores.append(current_state[:, 0, :].clone())

        # 2. Prepare target teacher-forcing feedback
        t_len = target_tokens.size(1)
        pad_col = torch.full((B, 1), -100, dtype=target_tokens.dtype, device=dev)
        prev_tokens = torch.cat([pad_col, target_tokens[:, : max(0, t_len - 1)]], dim=1)[:, :t_len]
        safe_feedback = torch.clamp(prev_tokens, min=0)
        prompt_pos_offset = prompt_tokens.shape[1]
        tf_flux, (tf_keys, tf_values) = model.sensory_dnn(
            safe_feedback, is_causal=True, pos_offset=prompt_pos_offset
        )
        fb_valid_mask = prev_tokens >= 0
        tf_flux = tf_flux * fb_valid_mask.unsqueeze(-1).to(tf_flux.dtype)

        # 3. Evaluate held-out cross-entropy loss and metrics at each cycle
        cycle_metrics = []
        delta_vecs = []

        for c in range(max_cycles + 1):
            st = states[c]
            z_c = cores[c]
            norm_z = float(z_c.norm(p=2, dim=-1).mean().item())

            # Decode target sequence using current deliberation state
            logits, _, _ = model.motor_decoder.forward_sequence(
                state=st,
                sensory_kv=sensory_kv,
                target_feedback=tf_flux,
                target_kv=(tf_keys, tf_values),
                tied_embedding_weight=model.sensory_dnn.embedding.weight,
                efference_gate=model.volumetric_core.efference_gate,
                radial_norm=model.volumetric_core.radial_norm,
                plan=None,
                target_kv_mask=fb_valid_mask,
                prompt_valid_mask=None,
            )
            # targets is target_tokens
            loss_c = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1)).item()

            # Prediction token entropy
            probs = F.softmax(logits.float(), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean().item()

            if c == 0:
                delta_norm = 0.0
                rel_delta = 0.0
                cos_sim = float("nan")
            else:
                delta_z = z_c - cores[c - 1]
                delta_vecs.append(delta_z)
                delta_norm = float(delta_z.norm(p=2, dim=-1).mean().item())
                rel_delta = delta_norm / (cores[c - 1].norm(p=2, dim=-1).mean().item() + 1e-8)

                if len(delta_vecs) >= 2:
                    dz_curr = delta_vecs[-1]
                    dz_prev = delta_vecs[-2]
                    dot = (dz_curr * dz_prev).sum(dim=-1)
                    cos_sim = float(
                        (dot / (dz_curr.norm(p=2, dim=-1) * dz_prev.norm(p=2, dim=-1) + 1e-8)).mean().item()
                    )
                else:
                    cos_sim = float("nan")

            cycle_metrics.append({
                "cycle": c,
                "loss": loss_c,
                "entropy": entropy,
                "core_norm": norm_z,
                "delta_norm": delta_norm,
                "rel_delta": rel_delta,
                "cos_sim": cos_sim,
                "lipschitz": lipschitz_bounds[c],
            })

    return cycle_metrics

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("🧠 PHASE 1 DIAGNOSTIC AUDIT: DELIBERATION DYNAMICS & ITERATIVE STATE ESTIMATION")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85 + "\n")

    ckpt_path = "checkpoints/fresh_scratch/v5_small_latest.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/v5_small_latest.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    tokenizer = build_tokenizer("llama")
    cfg = infer_v5_config_from_state_dict(ckpt)
    cfg.use_per_token_efference = False
    cfg.use_centralized_readout = True
    model = CerebroSpheroidV5(config=cfg).to(device)
    load_v5_checkpoint_into_torch(model, ckpt, verbose=False)

    max_cycles = 8

    print(f"✓ Model loaded with use_centralized_readout={model.config.use_centralized_readout}")
    print(f"Tracking cycles c = 0..{max_cycles} across 5 domain complexities:\n")

    results = {}
    for d_name, text in PROMPTS.items():
        toks = torch.tensor([tokenizer.encode(text, add_special_tokens=True)], dtype=torch.long, device=device)
        p_toks = toks[:, :min(48, toks.size(1) // 2)]
        t_toks = toks[:, min(48, toks.size(1) // 2):]

        metrics = measure_cycle_dynamics(model, p_toks, t_toks, max_cycles=max_cycles)
        results[d_name] = metrics

        print(f"--- Domain: {d_name.upper()} ---")
        print(f"{'Cycle':<5s} | {'Loss':<8s} | {'Entropy':<8s} | {'Core Norm':<10s} | {'Rel Delta (dc)':<14s} | {'Cos Sim':<10s} | {'Lipschitz (Lc)':<14s}")
        print("-" * 85)
        for m in metrics:
            c = m["cycle"]
            cos_str = f"{m['cos_sim']:+.4f}" if not math.isnan(m["cos_sim"]) else "  N/A  "
            lip_str = f"{m['lipschitz']:.4f}" if not math.isnan(m["lipschitz"]) else "  N/A  "
            print(f"{c:<5d} | {m['loss']:<8.4f} | {m['entropy']:<8.4f} | {m['core_norm']:<10.4f} | {m['rel_delta']:<14.4f} | {cos_str:<10s} | {lip_str:<14s}")
        print("-" * 85 + "\n")

    # Global Summary Analysis across domains
    print("=" * 85)
    print("GLOBAL SYNTHESIS OF INFERENCE STABILITY METRICS")
    print("=" * 85)
    print(f"{'Domain':<16s} | {'Min Loss Cycle':<15s} | {'L(0) -> L(min) -> L(8)':<26s} | {'Mean Cos Sim':<14s} | {'Mean Lipschitz':<14s}")
    print("-" * 85)
    for d_name, metrics in results.items():
        losses = [m["loss"] for m in metrics]
        min_c = int(torch.tensor(losses).argmin().item())
        cos_sims = [m["cos_sim"] for m in metrics if not math.isnan(m["cos_sim"])]
        mean_cos = sum(cos_sims) / len(cos_sims) if cos_sims else float("nan")
        lips = [m["lipschitz"] for m in metrics if not math.isnan(m["lipschitz"])]
        mean_lip = sum(lips) / len(lips) if lips else float("nan")

        loss_str = f"{losses[0]:.3f} -> {losses[min_c]:.3f} -> {losses[8]:.3f}"
        print(f"{d_name:<16s} | Cycle {min_c:<9d} | {loss_str:<26s} | {mean_cos:+.4f}{' ':8s} | {mean_lip:.4f}")
    print("=" * 85)

if __name__ == "__main__":
    main()
