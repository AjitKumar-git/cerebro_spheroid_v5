#!/usr/bin/env python3
"""
diagnostics/readout_integration_and_flux_audit.py

Diagnostic Suite for:
1. True Inward Radial Signal Flux:
   rho_k = ||rad_scale * delta_in[k]||_2 / (||S_k||_2 + eps)
   Evaluates whether inward radial activations actually overwhelm recipient shells during deliberation.
2. Readout Contribution Decomposition for S12-S15:
   Under the exact same checkpoint and prompt/target sequences, evaluates:
   - Baseline (Full 100% S12-S15 contribution)
   - Zero contribution (0% S12-S15)
   - 25% contribution scale
   - 50% contribution scale
   - Norm-matched shuffled noise replacement (preserves activation magnitude, destroys learned feature semantics)
   This directly isolates whether the -12.46% (Code) and -5.79% (STEM) loss improvement from removing S12-S15
   is due to:
   (A) Activation magnitude/scale imbalance in motor input
   (B) Destructive feature interference from poorly learned representations
"""

import os
import math
import torch
import torch.nn.functional as F
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

def run_audit():
    ckpt_path = "checkpoints/fresh_scratch/v5_small_latest.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    print("================================================================================")
    print("   CEREBRO SPHEROID V5: READOUT INTEGRATION & RADIAL FLUX AUDIT                 ")
    print("================================================================================")
    cp = torch.load(ckpt_path, map_location="cpu")
    cfg = infer_v5_config_from_state_dict(cp)
    cfg.use_per_token_efference = False
    model = CerebroSpheroidV5(config=cfg)
    model.eval()
    load_v5_checkpoint_into_torch(model, cp, verbose=False)
    tokenizer = build_tokenizer("llama")

    domain_tokens = {}
    for d, text in DOMAINS.items():
        toks = tokenizer.encode(text, add_special_tokens=True)
        if len(toks) > 64:
            toks = toks[:64]
        domain_tokens[d] = torch.tensor([toks], dtype=torch.long)

    # -------------------------------------------------------------------------
    # PART 1: TRUE INWARD & OUTWARD RADIAL SIGNAL FLUX
    # -------------------------------------------------------------------------
    print("\n--- 1. TRUE RADIAL ACTIVATION SIGNAL FLUX RATIOS (rho_k) ---")
    print(f"Model radial_mix_scale: {model.volumetric_core.radial_mix_scale.item():.4f}")
    
    with torch.no_grad():
        # Evaluate on general prose and python code
        for d in ["general_prose", "python_code"]:
            t_in = domain_tokens[d]
            dev = t_in.device
            sensory_flux, _ = model.sensory_dnn(t_in)
            prompt_centroid = model._recency_weighted_pool(sensory_flux)

            radial_profile = (torch.arange(1, cfg.num_shells + 1, device=dev, dtype=torch.float32) / float(cfg.num_shells)).view(1, -1, 1)
            state = (prompt_centroid.unsqueeze(1).repeat(1, cfg.num_shells, 1).float() * radial_profile).to(prompt_centroid.dtype)

            # Measure during deliberate_step
            state_intra = []
            for k in range(cfg.num_shells):
                s_k = state[:, k, :]
                h_k = model.volumetric_core.intra_norms[k](s_k)
                u_k = model.volumetric_core.intra_shell_swiglu[k](h_k) * model.volumetric_core.shell_gains[k]
                state_intra.append(s_k + u_k)
            state_intra = torch.stack(state_intra, dim=1)

            h_comm = model.volumetric_core.radial_norm(state_intra)
            rad_in = model.volumetric_core.radial_in_proj(h_comm[:, 1:, :])
            in_flux = torch.einsum("bkd,kde->bke", rad_in, model.volumetric_core.w_radial_in)
            in_up = model.volumetric_core.radial_up(in_flux)
            rad_scale = torch.clamp(model.volumetric_core.radial_mix_scale, 0.01, 1.0).to(state_intra.dtype)
            delta_in = F.pad(rad_scale * in_up, (0, 0, 0, 1))

            print(f"\nDomain: {d}")
            print(f"{'Transition':<20} | {'||S_k||_2':<12} | {'||Delta_in[k]||_2':<16} | {'rho_k (Flux Ratio)':<18}")
            print("-" * 72)
            for k in range(cfg.num_shells - 1):
                norm_sk = torch.norm(state_intra[0, k, :]).item()
                norm_din = torch.norm(delta_in[0, k, :]).item()
                rho_k = norm_din / (norm_sk + 1e-6)
                print(f"Shell {k+1:2d} -> Shell {k:2d} | {norm_sk:<12.4f} | {norm_din:<16.4f} | {rho_k:<18.4%}")

    # -------------------------------------------------------------------------
    # PART 2: S12-S15 MOTOR READOUT DECOMPOSITION
    # -------------------------------------------------------------------------
    print("\n--- 2. S12-S15 READOUT CONTRIBUTION DECOMPOSITION ---")
    print("Isolating Magnitude Imbalance vs Destructive Feature Interference:\n"
          "  (1) Baseline (100% S12-15)\n"
          "  (2) Zeroed (0% S12-15)\n"
          "  (3) Scaled 25% (S12-15 * 0.25)\n"
          "  (4) Scaled 50% (S12-15 * 0.50)\n"
          "  (5) Norm-Matched Shuffled Noise (S12-15 replaced by feature-scrambled noise of identical norm)")

    interventions = [
        "Baseline (100%)",
        "Zeroed (0%)",
        "Scaled 25%",
        "Scaled 50%",
        "Norm-Matched Shuffled",
    ]

    results = {iv: {} for iv in interventions}

    for iv in interventions:
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

                for cycle in range(cfg.thinking_cycles):
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

                # Intervene on S12-S15
                state_mod = state.clone()
                if iv == "Zeroed (0%)":
                    state_mod[:, 12:16, :] = 0.0
                elif iv == "Scaled 25%":
                    state_mod[:, 12:16, :] = state_mod[:, 12:16, :] * 0.25
                elif iv == "Scaled 50%":
                    state_mod[:, 12:16, :] = state_mod[:, 12:16, :] * 0.50
                elif iv == "Norm-Matched Shuffled":
                    for sk in range(12, 16):
                        orig = state_mod[:, sk, :]
                        orig_norm = torch.norm(orig, dim=-1, keepdim=True)
                        noise = torch.randn_like(orig)
                        noise_norm = torch.norm(noise, dim=-1, keepdim=True).clamp_min(1e-6)
                        state_mod[:, sk, :] = (noise / noise_norm) * orig_norm

                tf_flux, _ = model.sensory_dnn(target)
                logits, _, _ = model.motor_decoder.forward_sequence(
                    state=state_mod,
                    sensory_kv=sensory_kv,
                    target_feedback=tf_flux,
                    target_kv=None,
                    tied_embedding_weight=model.sensory_dnn.embedding.weight,
                    efference_gate=model.volumetric_core.efference_gate,
                    radial_norm=model.volumetric_core.radial_norm,
                )
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target.view(-1)).item()
                results[iv][domain_name] = loss

    header = f"{'Intervention on S12-15':<24} | " + " | ".join([f"{d[:12]:<12}" for d in DOMAINS])
    print("\nCross-Entropy Loss:")
    print(header)
    print("-" * len(header))
    for iv in interventions:
        row = f"{iv:<24} | " + " | ".join([f"{results[iv][d]:<12.4f}" for d in DOMAINS])
        print(row)

    print("\n% Change relative to Baseline (Negative is Better):")
    print(header)
    print("-" * len(header))
    base = results["Baseline (100%)"]
    for iv in interventions:
        if iv == "Baseline (100%)":
            continue
        deltas = []
        for d in DOMAINS:
            pct = ((results[iv][d] - base[d]) / base[d]) * 100.0
            deltas.append(f"{pct:>+11.2f}%")
        print(f"{iv:<24} | " + " | ".join(deltas))

    print("\n================================================================================")
    print("   AUDIT COMPLETE                                                               ")
    print("================================================================================")

if __name__ == "__main__":
    run_audit()
