#!/usr/bin/env python3
"""
scratch/sensory_kv_2_pilot.py

Pilot evaluation of Sensory KV 2.0 (Persistent Evidence Field):
Testing:
1. Baseline: Undifferentiated probe query = state.mean(), injection = mantle only (0.1 * retrieved).
2. Core-Driven Inquiry: probe query = S0 (Shell 0 deliberated state), injection = mantle only.
3. Core-Driven Dual-Injection: probe query = S0, evidence injected into both Mantle (S15) and Core (S0).
4. Multi-Channel Evidence Field: Raw KV (token-level) + Relational KV (depthwise conv-3 on flux).
5. Gated Core Evidence Adaptation on Math (15 optimization steps).
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
    apply_rope,
    RMSNorm,
)
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

def evaluate_sensory_kv_regime(
    model, cfg, prompt, target, regime: str = "baseline"
) -> float:
    B = prompt.shape[0]
    dev = prompt.device

    with torch.no_grad():
        sensory_flux, sensory_kv = model.sensory_dnn(prompt)
        prompt_centroid = model._recency_weighted_pool(sensory_flux)

        if regime == "multichannel":
            flux_t = sensory_flux.transpose(1, 2)  # [B, D, L]
            rel_flux = F.avg_pool1d(flux_t, kernel_size=3, stride=1, padding=1).transpose(1, 2)
            rel_k = model.sensory_dnn.k_proj(rel_flux).view(B, -1, cfg.num_heads, model.sensory_dnn.head_dim).transpose(1, 2)
            rel_v = model.sensory_dnn.v_proj(rel_flux).view(B, -1, cfg.num_heads, model.sensory_dnn.head_dim).transpose(1, 2)

        rad_prof = (torch.arange(1, cfg.num_shells + 1, device=dev, dtype=torch.float32) / float(cfg.num_shells)).view(1, -1, 1)
        state = (prompt_centroid.unsqueeze(1).repeat(1, cfg.num_shells, 1).float() * rad_prof).to(prompt_centroid.dtype)

        cycles = cfg.thinking_cycles
        init_scale = state.norm(p=2, dim=-1, keepdim=True) + 1e-8

        for cycle in range(cycles):
            if sensory_kv is not None and model.deliberation_probe is not None:
                keys, values = sensory_kv
                L_kv = keys.shape[2]

                if regime == "baseline":
                    query_vec = state.mean(dim=1)
                else:
                    query_vec = state[:, 0, :]

                probe_q = model.deliberation_probe(query_vec).view(B, cfg.num_heads, 1, model.sensory_dnn.head_dim)
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

                if regime == "multichannel":
                    scores_rel = torch.matmul(q_rot.to(rel_k.dtype), rel_k.transpose(-1, -2)) / math.sqrt(model.sensory_dnn.head_dim)
                    attn_rel = F.softmax(scores_rel.float(), dim=-1).to(scores_rel.dtype)
                    ret_rel = torch.matmul(attn_rel, rel_v).transpose(1, 2).contiguous().view(B, cfg.surface_dim)
                    retrieved = 0.5 * retrieved + 0.5 * ret_rel

                ret_normed = model.deliberation_probe_norm(retrieved)

                if regime in ("baseline", "core_inquiry"):
                    state_mantle = state[:, -1:, :] + 0.1 * ret_normed.unsqueeze(1)
                    state = torch.cat([state[:, :-1, :], state_mantle], dim=1)
                elif regime in ("core_dual_injection", "multichannel"):
                    state_core = state[:, :1, :] + 0.05 * ret_normed.unsqueeze(1)
                    state_mantle = state[:, -1:, :] + 0.05 * ret_normed.unsqueeze(1)
                    if cfg.num_shells > 2:
                        state = torch.cat([state_core, state[:, 1:-1, :], state_mantle], dim=1)
                    else:
                        state = torch.cat([state_core, state_mantle], dim=1)

            raw_next = model.volumetric_core.deliberate_step(state)
            alpha = float(0.75 ** cycle)
            updated = state + alpha * (raw_next - state)
            state = (updated / (updated.norm(p=2, dim=-1, keepdim=True) + 1e-8)) * init_scale

        tf_flux, target_kv = model.sensory_dnn(target, is_causal=True)
        logits, _, _ = model.motor_decoder.forward_sequence(
            state=state,
            sensory_kv=sensory_kv,
            target_feedback=tf_flux,
            target_kv=target_kv,
            tied_embedding_weight=model.sensory_dnn.embedding.weight,
            efference_gate=model.volumetric_core.efference_gate,
            radial_norm=model.volumetric_core.radial_norm,
        )
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target.view(-1))
        return loss.item()

def main():
    ckpt_path = "checkpoints/fresh_scratch/v5_small_latest.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found")
        return

    print("================================================================================")
    print("   PILOT AUDIT: SENSORY KV 2.0 (PERSISTENT EVIDENCE FIELD)                     ")
    print("================================================================================")
    cp = torch.load(ckpt_path, map_location="cpu")
    cfg = infer_v5_config_from_state_dict(cp)
    cfg.use_per_token_efference = False
    cfg.use_deliberation_probe = True
    model = CerebroSpheroidV5(config=cfg)
    load_v5_checkpoint_into_torch(model, cp, verbose=False)
    if model.deliberation_probe is not None:
        nn.init.orthogonal_(model.deliberation_probe.weight, gain=0.1)
    model.eval()

    tok = build_tokenizer("data/tokenizer")

    regimes = [
        ("baseline", "Regime 0: Baseline (state.mean query, mantle injection)"),
        ("core_inquiry", "Regime 1: Core-Driven Inquiry (S0 query, mantle injection)"),
        ("core_dual_injection", "Regime 2: Core-Driven Dual-Injection (S0 query, S0 + S15 injection)"),
        ("multichannel", "Regime 3: Multi-Channel Evidence Field (Raw + Relational KV)"),
    ]

    results = {r[0]: {} for r in regimes}

    for d_name, d_text in DOMAINS.items():
        ids = tok.encode(d_text)
        mid = len(ids) // 2
        p_ids = torch.tensor([ids[:mid]], dtype=torch.long)
        t_ids = torch.tensor([ids[mid:]], dtype=torch.long)

        for reg_key, _ in regimes:
            loss = evaluate_sensory_kv_regime(model, cfg, p_ids, t_ids, reg_key)
            results[reg_key][d_name] = loss

    print(f"\n{'Regime':<60} | {'General':<10} | {'Python':<10} | {'Math':<10} | {'Biochem':<10}")
    print("-" * 110)
    for reg_key, reg_desc in regimes:
        res = results[reg_key]
        print(f"{reg_desc:<60} | {res['general_prose']:<10.4f} | {res['python_code']:<10.4f} | {res['mathematics']:<10.4f} | {res['stem_biochem']:<10.4f}")

    print("\n--- % Change relative to Baseline (Regime 0) ---")
    print(f"{'Regime':<60} | {'General':<10} | {'Python':<10} | {'Math':<10} | {'Biochem':<10}")
    print("-" * 110)
    base = results["baseline"]
    for reg_key, reg_desc in regimes[1:]:
        res = results[reg_key]
        d_gen = (res['general_prose'] - base['general_prose']) / base['general_prose'] * 100
        d_py = (res['python_code'] - base['python_code']) / base['python_code'] * 100
        d_math = (res['mathematics'] - base['mathematics']) / base['mathematics'] * 100
        d_bio = (res['stem_biochem'] - base['stem_biochem']) / base['stem_biochem'] * 100
        print(f"{reg_desc:<60} | {d_gen:>+9.2f}% | {d_py:>+9.2f}% | {d_math:>+9.2f}% | {d_bio:>+9.2f}%")

    print("\n--- Gated Core Evidence Adaptation Test on Math (15 steps) ---")
    gate_param = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    opt = torch.optim.Adam([gate_param, model.deliberation_probe.weight], lr=1e-3)
    math_ids = tok.encode(DOMAINS["mathematics"])
    m_p = torch.tensor([math_ids[:len(math_ids)//2]], dtype=torch.long)
    m_t = torch.tensor([math_ids[len(math_ids)//2:]], dtype=torch.long)

    # Initial loss with beta = 0.0
    with torch.no_grad():
        tf_flux, target_kv = model.sensory_dnn(m_t, is_causal=True)
        sensory_flux, sensory_kv = model.sensory_dnn(m_p)
        prompt_centroid = model._recency_weighted_pool(sensory_flux)
        rad_prof = (torch.arange(1, cfg.num_shells + 1, dtype=torch.float32) / float(cfg.num_shells)).view(1, -1, 1)
        state = (prompt_centroid.unsqueeze(1).repeat(1, cfg.num_shells, 1).float() * rad_prof).to(prompt_centroid.dtype)
        init_scale = state.norm(p=2, dim=-1, keepdim=True) + 1e-8
        for cycle in range(cfg.thinking_cycles):
            raw_next = model.volumetric_core.deliberate_step(state)
            alpha = float(0.75 ** cycle)
            updated = state + alpha * (raw_next - state)
            state = (updated / (updated.norm(p=2, dim=-1, keepdim=True) + 1e-8)) * init_scale
        logits, _, _ = model.motor_decoder.forward_sequence(
            state=state, sensory_kv=sensory_kv, target_feedback=tf_flux, target_kv=target_kv,
            tied_embedding_weight=model.sensory_dnn.embedding.weight,
            efference_gate=model.volumetric_core.efference_gate, radial_norm=model.volumetric_core.radial_norm,
        )
        base_math_loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), m_t.view(-1)).item()

    print(f"Initial Math Loss (Gated beta=0): {base_math_loss:.4f}")

    for step in range(15):
        opt.zero_grad()
        sensory_flux, sensory_kv = model.sensory_dnn(m_p)
        prompt_centroid = model._recency_weighted_pool(sensory_flux)
        rad_prof = (torch.arange(1, cfg.num_shells + 1, dtype=torch.float32) / float(cfg.num_shells)).view(1, -1, 1)
        state = (prompt_centroid.unsqueeze(1).repeat(1, cfg.num_shells, 1).float() * rad_prof).to(prompt_centroid.dtype)
        init_scale = state.norm(p=2, dim=-1, keepdim=True) + 1e-8
        beta = torch.tanh(gate_param)

        for cycle in range(cfg.thinking_cycles):
            keys, values = sensory_kv
            probe_q = model.deliberation_probe(state[:, 0, :]).view(1, cfg.num_heads, 1, model.sensory_dnn.head_dim)
            if model.motor_decoder.use_rope:
                q_rot = apply_rope(probe_q, model.motor_decoder.rope_cos, model.motor_decoder.rope_sin, offset=cycle)
                k_rot = apply_rope(keys, model.motor_decoder.rope_cos, model.motor_decoder.rope_sin, offset=0)
            else:
                q_rot, k_rot = probe_q, keys
            scores = torch.matmul(q_rot.to(keys.dtype), k_rot.transpose(-1, -2)) / math.sqrt(model.sensory_dnn.head_dim)
            attn_p = F.softmax(scores.float(), dim=-1).to(scores.dtype)
            retrieved = torch.matmul(attn_p, values).transpose(1, 2).contiguous().view(1, cfg.surface_dim)
            ret_normed = model.deliberation_probe_norm(retrieved)

            state_core = state[:, :1, :] + beta * 0.1 * ret_normed.unsqueeze(1)
            state_mantle = state[:, -1:, :] + 0.1 * ret_normed.unsqueeze(1)
            state = torch.cat([state_core, state[:, 1:-1, :], state_mantle], dim=1)

            raw_next = model.volumetric_core.deliberate_step(state)
            alpha = float(0.75 ** cycle)
            updated = state + alpha * (raw_next - state)
            state = (updated / (updated.norm(p=2, dim=-1, keepdim=True) + 1e-8)) * init_scale

        tf_flux, target_kv = model.sensory_dnn(m_t, is_causal=True)
        logits, _, _ = model.motor_decoder.forward_sequence(
            state=state, sensory_kv=sensory_kv, target_feedback=tf_flux, target_kv=target_kv,
            tied_embedding_weight=model.sensory_dnn.embedding.weight,
            efference_gate=model.volumetric_core.efference_gate, radial_norm=model.volumetric_core.radial_norm,
        )
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), m_t.view(-1))
        loss.backward()
        opt.step()

    print(f"Adapted Math Loss: {loss.item():.4f} (Delta: {loss.item() - base_math_loss:+.4f})")
    print(f"Learned Core Evidence Gate (beta): {torch.tanh(gate_param).item():.4f}")
    print("================================================================================")

if __name__ == "__main__":
    main()
