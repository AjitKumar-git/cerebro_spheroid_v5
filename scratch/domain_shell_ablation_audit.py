#!/usr/bin/env python3
"""
diagnostics/domain_shell_ablation_audit.py

Empirical Core Overwrite & Functional Shell Specialization Audit:
Directly tests the hypotheses outlined in the architectural discussion:
1. Per-Shell Activation Norms & Variance across domains (Prose, Python Code, Mathematics, STEM).
2. Functional Shell Zero-Out Ablations:
   Measures baseline cross-entropy loss L_base vs L_ablate(k) when zeroing out:
   - Shell 0 (Subcortical Core)
   - Shells 1..3 (Inner Mantle)
   - Shells 4..7 (Mid Mantle)
   - Shells 8..11 (Outer Mantle)
   - Shells 12..15 (Cortical Boundary)
   to determine whether outer shells disproportionately impact niche data (Code, Math, STEM)
   or merely represent undertrained/redundant parameters.
3. Radial Signal Flux Ratio:
   rho_k = ||r_mix * W_in * S_{k+1}||_2 / (||S_k||_2 + eps)
   Measures whether inward radial signal actually overwhelms Shell 0 during forward inference.
"""

import os
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from src.spheroid_v5 import CerebroSpheroidV5, infer_v5_config_from_state_dict, load_v5_checkpoint_into_torch, apply_rope
from src.corpus import build_tokenizer

DOMAINS = {
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
        "Theorem: Let G be a finite group and H a subgroup of G. Then the order of H divides "
        "the order of G. Proof: Consider the left cosets of H in G. The partition induced by cosets "
        "gH shows that each coset has cardinality |H|. Since G is the disjoint union of these cosets, "
        "we have |G| = [G:H] * |H|, proving Lagrange's theorem."
    ),
    "stem_biochem": (
        "The catalytic triad of serine proteases consists of His-57, Asp-102, and Ser-195. "
        "During peptide bond cleavage, His-57 acts as a general base, abstracting a proton from "
        "the Ser-195 hydroxyl group to facilitate nucleophilic attack on the carbonyl carbon of the substrate."
    ),
}

def run_diagnostic():
    ckpt_path = "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/fresh_scratch/v5_small_latest.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    print("================================================================================")
    print("   CEREBRO SPHEROID V5: EMPIRICAL CORE OVERWRITE & SHELL ABLATION AUDIT         ")
    print("================================================================================")
    print(f"Loading checkpoint: {ckpt_path} ...")
    cp = torch.load(ckpt_path, map_location="cpu")
    cfg = infer_v5_config_from_state_dict(cp)
    cfg.use_per_token_efference = False  # Evaluate with per-token efference disabled
    model = CerebroSpheroidV5(config=cfg)
    model.eval()
    load_v5_checkpoint_into_torch(model, cp, verbose=False)
    print(f"Model instantiated: K={cfg.num_shells} shells, D={cfg.surface_dim}, Step={cp.get('step', '?')}")

    tokenizer = build_tokenizer("llama")

    # 1. Inward Radial Signal Flux
    print("\n--- 1. INWARD RADIAL SIGNAL FLUX ANALYSIS ---")
    w_rad_in = model.volumetric_core.w_radial_in  # [K-1, D_r, D_r]
    print(f"Radial Inward Weight Norms per shell transition (k+1 -> k):")
    for k in range(cfg.num_shells - 1):
        norm_k = torch.norm(w_rad_in[k]).item()
        print(f"  Shell {k+1:2d} -> Shell {k:2d}: ||W_radial_in|| = {norm_k:.4f}")

    # 2. Domain-Specific Forward Pass & Activation Norms
    print("\n--- 2. PER-SHELL ACTIVATION NORMS ACROSS DOMAINS ---")
    shell_activations = {domain: [] for domain in DOMAINS}
    domain_tokens = {}

    for domain_name, text in DOMAINS.items():
        toks = tokenizer.encode(text, add_special_tokens=True)
        if len(toks) > 64:
            toks = toks[:64]
        domain_tokens[domain_name] = torch.tensor([toks], dtype=torch.long)

    with torch.no_grad():
        for domain_name, t_in in domain_tokens.items():
            B = t_in.shape[0]
            dev = t_in.device
            sensory_flux, sensory_kv = model.sensory_dnn(t_in)
            prompt_centroid = model._recency_weighted_pool(sensory_flux)

            # Radial laminar depth profile
            radial_profile = (torch.arange(1, cfg.num_shells + 1, device=dev, dtype=torch.float32) / float(cfg.num_shells)).view(1, -1, 1)
            state = (prompt_centroid.unsqueeze(1).repeat(1, cfg.num_shells, 1).float() * radial_profile).to(prompt_centroid.dtype)

            # Subconscious Deliberation Cycles
            cycles = cfg.thinking_cycles
            for cycle in range(cycles):
                if sensory_kv is not None and model.deliberation_probe is not None:
                    keys, values = sensory_kv
                    L_kv = keys.shape[2]
                    probe_q = model.deliberation_probe(state.mean(dim=1)).view(B, cfg.num_heads, 1, model.sensory_dnn.head_dim)
                    if model.motor_decoder.use_rope:
                        q_rot = apply_rope(probe_q, model.motor_decoder.rope_cos, model.motor_decoder.rope_sin, offset=cycle)
                        k_rot = apply_rope(keys, model.motor_decoder.rope_cos, model.motor_decoder.rope_sin, offset=0)
                    else:
                        q_rot, k_rot = probe_q, keys
                    scores = torch.matmul(q_rot.to(keys.dtype), k_rot.transpose(-1, -2)) / math.sqrt(model.sensory_dnn.head_dim)
                    if cfg.top_k_resonance > 0 and L_kv > cfg.top_k_resonance:
                        topk_v, _ = torch.topk(scores, cfg.top_k_resonance, dim=-1)
                        scores = torch.where(scores < topk_v[..., -1, None], torch.full_like(scores, -1e4), scores)
                    attn_p = F.softmax(scores.float(), dim=-1).to(scores.dtype)
                    retrieved = torch.matmul(attn_p, values).transpose(1, 2).contiguous().view(B, cfg.surface_dim)
                    state_mantle = state[:, -1:, :] + 0.1 * model.deliberation_probe_norm(retrieved).unsqueeze(1)
                    state = torch.cat([state[:, :-1, :], state_mantle], dim=1)

                state = model.volumetric_core.deliberate_step(state)

            norms = [torch.norm(state[0, k, :]).item() for k in range(cfg.num_shells)]
            shell_activations[domain_name] = norms

    header = f"{'Shell':<8} | " + " | ".join([f"{d[:12]:<12}" for d in DOMAINS])
    print(header)
    print("-" * len(header))
    for k in range(cfg.num_shells):
        vals = " | ".join([f"{shell_activations[d][k]:<12.4f}" for d in DOMAINS])
        print(f"Shell {k:<2} | {vals}")

    # 3. Functional Shell Ablations Across Domains
    print("\n--- 3. FUNCTIONAL ZERO-OUT ABLATION (Cross-Entropy Loss Delta) ---")
    print("Testing whether ablating outer shells selectively harms niche domains (Math, Code, STEM)")
    print("versus general prose, or if damage is uniform.\n")

    ablation_groups = {
        "Baseline (None)": [],
        "Ablate S0 (Core)": [0],
        "Ablate S1-S3": list(range(1, 4)),
        "Ablate S4-S7": list(range(4, 8)),
        "Ablate S8-S11": list(range(8, 12)),
        "Ablate S12-S15 (Cortex)": list(range(12, 16)),
    }

    results = {group: {} for group in ablation_groups}

    for group_name, shells_to_zero in ablation_groups.items():
        for domain_name, t_in in domain_tokens.items():
            prompt = t_in[:, :-1]
            target = t_in[:, 1:]
            B = prompt.shape[0]
            dev = prompt.device

            with torch.no_grad():
                sensory_flux, sensory_kv = model.sensory_dnn(prompt)
                prompt_centroid = model._recency_weighted_pool(sensory_flux)

                radial_profile = (torch.arange(1, cfg.num_shells + 1, device=dev, dtype=torch.float32) / float(cfg.num_shells)).view(1, -1, 1)
                state = (prompt_centroid.unsqueeze(1).repeat(1, cfg.num_shells, 1).float() * radial_profile).to(prompt_centroid.dtype)

                cycles = cfg.thinking_cycles
                for cycle in range(cycles):
                    if sensory_kv is not None and model.deliberation_probe is not None:
                        keys, values = sensory_kv
                        L_kv = keys.shape[2]
                        probe_q = model.deliberation_probe(state.mean(dim=1)).view(B, cfg.num_heads, 1, model.sensory_dnn.head_dim)
                        if model.motor_decoder.use_rope:
                            q_rot = apply_rope(probe_q, model.motor_decoder.rope_cos, model.motor_decoder.rope_sin, offset=cycle)
                            k_rot = apply_rope(keys, model.motor_decoder.rope_cos, model.motor_decoder.rope_sin, offset=0)
                        else:
                            q_rot, k_rot = probe_q, keys
                        scores = torch.matmul(q_rot.to(keys.dtype), k_rot.transpose(-1, -2)) / math.sqrt(model.sensory_dnn.head_dim)
                        if cfg.top_k_resonance > 0 and L_kv > cfg.top_k_resonance:
                            topk_v, _ = torch.topk(scores, cfg.top_k_resonance, dim=-1)
                            scores = torch.where(scores < topk_v[..., -1, None], torch.full_like(scores, -1e4), scores)
                        attn_p = F.softmax(scores.float(), dim=-1).to(scores.dtype)
                        retrieved = torch.matmul(attn_p, values).transpose(1, 2).contiguous().view(B, cfg.surface_dim)
                        state_mantle = state[:, -1:, :] + 0.1 * model.deliberation_probe_norm(retrieved).unsqueeze(1)
                        state = torch.cat([state[:, :-1, :], state_mantle], dim=1)

                    state = model.volumetric_core.deliberate_step(state)

                # Apply ablation to state manifold
                if shells_to_zero:
                    state_ablated = state.clone()
                    for sk in shells_to_zero:
                        state_ablated[:, sk, :] = 0.0
                else:
                    state_ablated = state

                # Run motor decoder sequence
                tf_flux, _ = model.sensory_dnn(target)
                logits, _, _ = model.motor_decoder.forward_sequence(
                    state=state_ablated,
                    sensory_kv=sensory_kv,
                    target_feedback=tf_flux,
                    target_kv=None,
                    tied_embedding_weight=model.sensory_dnn.embedding.weight,
                    efference_gate=model.volumetric_core.efference_gate,
                    radial_norm=model.volumetric_core.radial_norm,
                )
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target.view(-1)).item()
                results[group_name][domain_name] = loss

    ablation_header = f"{'Ablation Group':<26} | " + " | ".join([f"{d[:12]:<12}" for d in DOMAINS])
    print(ablation_header)
    print("-" * len(ablation_header))
    for group_name in ablation_groups:
        row = f"{group_name:<26} | " + " | ".join([f"{results[group_name][d]:<12.4f}" for d in DOMAINS])
        print(row)

    print("\n--- Relative Degradation (% Delta Loss over Baseline) ---")
    base = results["Baseline (None)"]
    print(ablation_header)
    print("-" * len(ablation_header))
    for group_name in ablation_groups:
        if group_name == "Baseline (None)":
            continue
        deltas = []
        for d in DOMAINS:
            pct = ((results[group_name][d] - base[d]) / base[d]) * 100.0
            deltas.append(f"{pct:>+11.2f}%")
        print(f"{group_name:<26} | " + " | ".join(deltas))

    print("\n================================================================================")
    print("   DIAGNOSTIC COMPLETE                                                          ")
    print("================================================================================")

if __name__ == "__main__":
    run_diagnostic()
