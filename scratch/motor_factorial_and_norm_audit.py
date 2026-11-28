#!/usr/bin/env python3
"""
diagnostics/motor_factorial_and_norm_audit.py

Phase A & Phase B Audit:
Full Motor-Slot Factorial Interaction & Normalization Confound Audit:
1. Slot Norms & Gated Magnitudes:
   Measures ||c_i||_2 before FusedNorm and after FusedNorm for each slot:
   - Slot 0: gate[0] * motor_surface (Shell 15 direct)
   - Slot 1: gate[1] * cur_core      (Shell 0 deliberate core)
   - Slot 2: gate[2] * resonance     (Arcuate cross-resonance from sensory_kv)
   - Slot 3: gate[3] * shell_pool    (Multi-shell pooling tap)
   - Slot 4: gate[4] * cur_plan      (Executive recurrent plan)
2. Normalization Controls:
   Distinguishes feature removal from normalization scaling artifacts:
   (A) Standard Zeroing: c_i = 0 (changes RMS denominator of FusedNorm)
   (B) Variance-Preserved Shuffled Noise: replaces c_i with random noise of the exact same norm,
       keeping FusedNorm denominator constant while destroying features.
   (C) Re-normalized Zeroing: re-normalizes remaining slots so total RMS variance is preserved.
3. Full Factorial Matrix across all 4 domains:
   - Baseline [1, 1, 1, 1, 1]
   - Mute Slot 0: Direct surface pathway
   - Mute Slot 2: Surface-derived resonance pathway
   - Mute Slot 3: Shell pooling pathway
   - Mute Slots 0 + 3: Outer direct bypasses
   - Mute Slots 0 + 2: Surface pathways (surface + resonance)
   - Mute Slots 2 + 3: Resonance + pool
   - Mute Slots 0 + 2 + 3: All outer-derived direct pathways
   - Core + Plan Only [0, 1, 0, 0, 1]: Core sufficiency test
"""

import os
import math
import torch
import torch.nn.functional as F
from src.spheroid_v5 import CerebroSpheroidV5, infer_v5_config_from_state_dict, load_v5_checkpoint_into_torch, apply_rope, RMSNorm
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

def run_factorial_audit():
    ckpt_path = "checkpoints/fresh_scratch/v5_small_latest.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    print("================================================================================")
    print("   CEREBRO SPHEROID V5: FULL MOTOR-SLOT FACTORIAL & NORMALIZATION AUDIT         ")
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
    # PART 1: MEASURE UNGATED & GATED SLOT NORMS BEFORE & AFTER FUSED NORM
    # -------------------------------------------------------------------------
    print("\n--- 1. SLOT NORMS & RELATIVE CONTRIBUTION (General Prose & Python Code) ---")
    gate = torch.clamp(model.motor_decoder.input_gate.float(), -5.0, 5.0)
    print(f"Learned input_gate values: {gate.tolist()}")

    with torch.no_grad():
        for d in ["general_prose", "python_code"]:
            t_in = domain_tokens[d]
            prompt = t_in[:, :-1]
            target = t_in[:, 1:]
            B = prompt.shape[0]
            dev = prompt.device

            sensory_flux, sensory_kv = model.sensory_dnn(prompt)
            prompt_centroid = model._recency_weighted_pool(sensory_flux)

            rad_prof = (torch.arange(1, cfg.num_shells + 1, device=dev, dtype=torch.float32) / float(cfg.num_shells)).view(1, -1, 1)
            state = (prompt_centroid.unsqueeze(1).repeat(1, cfg.num_shells, 1).float() * rad_prof).to(prompt_centroid.dtype)

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

            tf_flux, _ = model.sensory_dnn(target)
            T_seq = tf_flux.shape[1]
            surface = state[:, -1, :]
            cur_core = state[:, 0, :]
            dt = surface.dtype
            plan = model.motor_decoder.init_plan(B, state.device).to(dt)
            p_scaled = model.motor_decoder._prepare_phase_vectors(torch.arange(T_seq, device=state.device), dt)
            motor_surface = surface.unsqueeze(1) + p_scaled.unsqueeze(0) + tf_flux.to(dt)

            resonance = model.motor_decoder._parallel_cross_resonance(
                motor_surface=motor_surface, sensory_kv=sensory_kv, target_kv=None,
                target_kv_mask=None, prompt_valid_mask=None, chunk_offset=0
            )

            norm_factor = math.sqrt(float(cfg.num_shells))
            shell_weights = F.softmax(model.motor_decoder.shell_score.float(), dim=0).to(dt)
            shell_scale = model.motor_decoder.shell_scale.to(dt) * norm_factor
            shell_proj_rest = model.motor_decoder.shell_proj(state[:, 1:, :])
            shell_pool_rest = torch.sum(shell_weights[1:].view(1, -1, 1) * shell_proj_rest, dim=1) * shell_scale
            w0s = shell_weights[0] * shell_scale
            sp_t = shell_pool_rest + w0s * model.motor_decoder.shell_proj(cur_core)

            # Slot components at t=0
            s0_raw = motor_surface[:, 0, :]
            s1_raw = cur_core
            s2_raw = resonance[:, 0, :]
            s3_raw = sp_t
            s4_raw = plan

            raw_norms = [torch.norm(s).item() for s in [s0_raw, s1_raw, s2_raw, s3_raw, s4_raw]]
            gated_norms = [torch.norm(gate[i] * s).item() for i, s in enumerate([s0_raw, s1_raw, s2_raw, s3_raw, s4_raw])]

            fused_cat = torch.cat([gate[i] * s for i, s in enumerate([s0_raw, s1_raw, s2_raw, s3_raw, s4_raw])], dim=-1)
            fused_normed = model.motor_decoder.fused_norm(fused_cat)
            normed_chunks = fused_normed.chunk(5, dim=-1)
            normed_norms = [torch.norm(c).item() for c in normed_chunks]

            print(f"\nDomain: {d}")
            print(f"{'Slot':<18} | {'Raw Norm ||s||':<16} | {'Gated Norm ||g*s||':<20} | {'Post-FusedNorm Norm':<20}")
            print("-" * 80)
            slot_names = ["0: motor_surface", "1: cur_core", "2: resonance", "3: shell_pool", "4: cur_plan"]
            for i, name in enumerate(slot_names):
                print(f"{name:<18} | {raw_norms[i]:<16.4f} | {gated_norms[i]:<20.4f} | {normed_norms[i]:<20.4f}")

    # -------------------------------------------------------------------------
    # PART 2: FULL FACTORIAL INTERACTION MATRIX WITH NORMALIZATION CONTROLS
    # -------------------------------------------------------------------------
    print("\n--- 2. COMPLETE FACTORIAL INTERACTION MATRIX ---")
    print("Testing all slot interaction combinations across all 4 domains:\n")

    FACTORIAL_CONFIGS = {
        "Baseline [1,1,1,1,1]":         [1, 1, 1, 1, 1],
        "Mute S0 (Surface)":            [0, 1, 1, 1, 1],
        "Mute S2 (Resonance)":          [1, 1, 0, 1, 1],
        "Mute S3 (Shell Pool)":         [1, 1, 1, 0, 1],
        "Mute S0 + S3 (Outer Bypasses)":[0, 1, 1, 0, 1],
        "Mute S0 + S2 (Surface Paths)": [0, 1, 0, 1, 1],
        "Mute S2 + S3 (Res + Pool)":    [1, 1, 0, 0, 1],
        "Mute S0 + S2 + S3 (All Outer)":[0, 1, 0, 0, 1],
        "Core + Plan Only [0,1,0,0,1]": [0, 1, 0, 0, 1],
    }

    def eval_with_mask(model, prompt_tokens, target_tokens, mask, norm_control="zero"):
        prompt = prompt_tokens
        target = target_tokens
        B = prompt.shape[0]
        dev = prompt.device

        with torch.no_grad():
            sensory_flux, sensory_kv = model.sensory_dnn(prompt)
            prompt_centroid = model._recency_weighted_pool(sensory_flux)

            rad_prof = (torch.arange(1, cfg.num_shells + 1, device=dev, dtype=torch.float32) / float(cfg.num_shells)).view(1, -1, 1)
            state = (prompt_centroid.unsqueeze(1).repeat(1, cfg.num_shells, 1).float() * rad_prof).to(prompt_centroid.dtype)

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

            tf_flux, _ = model.sensory_dnn(target)
            T_seq = tf_flux.shape[1]
            surface = state[:, -1, :]
            cur_core = state[:, 0, :]
            dt = surface.dtype
            plan = model.motor_decoder.init_plan(B, state.device).to(dt)
            p_scaled = model.motor_decoder._prepare_phase_vectors(torch.arange(T_seq, device=state.device), dt)
            motor_surface = surface.unsqueeze(1) + p_scaled.unsqueeze(0) + tf_flux.to(dt)

            resonance = model.motor_decoder._parallel_cross_resonance(
                motor_surface=motor_surface, sensory_kv=sensory_kv, target_kv=None,
                target_kv_mask=None, prompt_valid_mask=None, chunk_offset=0
            )

            norm_factor = math.sqrt(float(cfg.num_shells))
            shell_weights = F.softmax(model.motor_decoder.shell_score.float(), dim=0).to(dt)
            shell_scale = model.motor_decoder.shell_scale.to(dt) * norm_factor
            shell_proj_rest = model.motor_decoder.shell_proj(state[:, 1:, :])
            shell_pool_rest = torch.sum(shell_weights[1:].view(1, -1, 1) * shell_proj_rest, dim=1) * shell_scale
            w0s = shell_weights[0] * shell_scale
            sp_t = shell_pool_rest + w0s * model.motor_decoder.shell_proj(cur_core)

            cur_plan = plan.to(dt)
            pd = torch.clamp(model.motor_decoder.plan_decay.float(), 0.0, 0.92).to(dt) if model.motor_decoder.use_plan else None
            pw = model.motor_decoder.plan_write.to(dt) if model.motor_decoder.use_plan else None
            g = torch.clamp(model.motor_decoder.input_gate.float(), -5.0, 5.0).to(dt)

            hs = []
            for t in range(T_seq):
                parts = [
                    motor_surface[:, t, :] * g[0],
                    cur_core * g[1],
                    resonance[:, t, :] * g[2],
                    sp_t * g[3],
                    cur_plan * g[4],
                ]

                # Apply Mask & Normalization Control
                active_parts = []
                for idx, (m_val, p_val) in enumerate(zip(mask, parts)):
                    if m_val == 1:
                        active_parts.append(p_val)
                    else:
                        if norm_control == "zero":
                            active_parts.append(torch.zeros_like(p_val))
                        elif norm_control == "shuffled_noise":
                            # Random noise with identical norm to keep FusedNorm denominator constant
                            noise = torch.randn_like(p_val)
                            p_norm = torch.norm(p_val, dim=-1, keepdim=True)
                            n_norm = torch.norm(noise, dim=-1, keepdim=True).clamp_min(1e-6)
                            active_parts.append((noise / n_norm) * p_norm)
                        elif norm_control == "renorm_scale":
                            # Rescale remaining parts so total variance sum is preserved
                            active_parts.append(torch.zeros_like(p_val))

                fused_in = torch.cat(active_parts, dim=-1)
                if norm_control == "renorm_scale":
                    active_count = sum(mask)
                    total_count = len(mask)
                    fused_in = fused_in * math.sqrt(float(total_count) / float(max(1, active_count)))

                fused = model.motor_decoder.fused_norm(fused_in)
                h_t = model.motor_decoder.norm_mid(model.motor_decoder.motor_swiglu(fused))
                hs.append(h_t)

                if model.motor_decoder.use_plan and cur_plan is not None:
                    p_in = torch.cat([h_t, p_scaled[t].unsqueeze(0).expand(B, -1)], dim=-1)
                    cur_plan = torch.clamp(pd * cur_plan + pw * torch.sigmoid(model.motor_decoder.plan_gate(p_in)) * model.motor_decoder.plan_update(p_in), -50.0, 50.0)

            H = torch.stack(hs, dim=1)
            temp = torch.clamp(model.motor_decoder.temperature_scale.float(), 0.1, 5.0).to(dt)
            H_scaled = (H * model.motor_decoder.affine_scale.to(dt).view(1, 1, -1)) / temp
            logits = F.linear(H_scaled, model.sensory_dnn.embedding.weight.to(dt))
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target.view(-1)).item()
            return loss

    # Run Standard Zeroing Factorial
    results_zero = {cfg_name: {} for cfg_name in FACTORIAL_CONFIGS}
    for cfg_name, mask in FACTORIAL_CONFIGS.items():
        for d_name, t_in in domain_tokens.items():
            loss = eval_with_mask(model, t_in[:, :-1], t_in[:, 1:], mask, norm_control="zero")
            results_zero[cfg_name][d_name] = loss

    header = f"{'Configuration':<32} | " + " | ".join([f"{d[:12]:<12}" for d in DOMAINS])
    print(header)
    print("-" * len(header))
    for cfg_name in FACTORIAL_CONFIGS:
        row = f"{cfg_name:<32} | " + " | ".join([f"{results_zero[cfg_name][d]:<12.4f}" for d in DOMAINS])
        print(row)

    print("\n% Change relative to Baseline (Negative is Better):")
    print(header)
    print("-" * len(header))
    base = results_zero["Baseline [1,1,1,1,1]"]
    for cfg_name in FACTORIAL_CONFIGS:
        if cfg_name == "Baseline [1,1,1,1,1]":
            continue
        deltas = []
        for d in DOMAINS:
            pct = ((results_zero[cfg_name][d] - base[d]) / base[d]) * 100.0
            deltas.append(f"{pct:>+11.2f}%")
        print(f"{cfg_name:<32} | " + " | ".join(deltas))

    # -------------------------------------------------------------------------
    # PART 3: NORMALIZATION CONFOUND AUDIT (Python Code & Prose)
    # -------------------------------------------------------------------------
    print("\n--- 3. NORMALIZATION CONFOUND DISSECTION (Python Code & Prose) ---")
    print("Comparing (A) Zeroing vs (B) Shuffled Noise (Same Norm) vs (C) Re-normalized Scale")
    print("For: Mute S0+S3 (Outer Bypasses) and Core+Plan Only:")

    test_masks = {
        "Mute S0 + S3 (Outer Bypasses)": [0, 1, 1, 0, 1],
        "Core + Plan Only [0,1,0,0,1]":  [0, 1, 0, 0, 1],
    }

    norm_header = f"{'Mask & Normalization Mode':<40} | {'Prose Loss':<14} | {'Code Loss':<14}"
    print("\n" + norm_header)
    print("-" * len(norm_header))

    p_in_prose = domain_tokens["general_prose"]
    p_in_code = domain_tokens["python_code"]

    for mask_name, mask in test_masks.items():
        l_zero_prose = eval_with_mask(model, p_in_prose[:, :-1], p_in_prose[:, 1:], mask, norm_control="zero")
        l_zero_code = eval_with_mask(model, p_in_code[:, :-1], p_in_code[:, 1:], mask, norm_control="zero")

        l_shuf_prose = eval_with_mask(model, p_in_prose[:, :-1], p_in_prose[:, 1:], mask, norm_control="shuffled_noise")
        l_shuf_code = eval_with_mask(model, p_in_code[:, :-1], p_in_code[:, 1:], mask, norm_control="shuffled_noise")

        l_rnm_prose = eval_with_mask(model, p_in_prose[:, :-1], p_in_prose[:, 1:], mask, norm_control="renorm_scale")
        l_rnm_code = eval_with_mask(model, p_in_code[:, :-1], p_in_code[:, 1:], mask, norm_control="renorm_scale")

        print(f"{mask_name + ' (Zero)':<40} | {l_zero_prose:<14.4f} | {l_zero_code:<14.4f}")
        print(f"{mask_name + ' (Shuffled Noise)':<40} | {l_shuf_prose:<14.4f} | {l_shuf_code:<14.4f}")
        print(f"{mask_name + ' (Re-norm Scaled)':<40} | {l_rnm_prose:<14.4f} | {l_rnm_code:<14.4f}")
        print("-" * len(norm_header))

    print("\n================================================================================")
    print("   FACTORIAL AUDIT COMPLETE                                                     ")
    print("================================================================================")

if __name__ == "__main__":
    run_factorial_audit()
