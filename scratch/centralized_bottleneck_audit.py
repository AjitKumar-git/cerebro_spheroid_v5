#!/usr/bin/env python3
"""
diagnostics/centralized_bottleneck_audit.py

Direct Benchmark for Centralized Readout Bottleneck Hypotheses:
Compares 4 precise architectures on the exact same checkpoint (Step 12,200) across all 4 domains:

1. Current Baseline:
   - Shell-to-motor access: All direct paths (Slot 0 Surface + Slot 1 Core + Slot 3 Shell Pool)
   - Resonance query: motor_surface (Shell 15)
   - Core representation: Raw S0 (magnitude ~1.7)

2. Architecture A:
   - Shell-to-motor access: S0 only (Slots 0 and 3 muted: mask = [0, 1, 1, 0, 1])
   - Resonance query: motor_surface (Shell 15)
   - Core representation: Raw S0

3. Architecture B:
   - Shell-to-motor access: S0 only (Slots 0 and 3 muted: mask = [0, 1, 1, 0, 1])
   - Resonance query: motor_core = cur_core + phase + feedback (Core S0 generates the retrieval query)
   - Core representation: Raw S0

4. Architecture C (Core-Normalized / Energy Rebalanced):
   - Shell-to-motor access: S0 only (Slots 0 and 3 muted: mask = [0, 1, 1, 0, 1])
   - Resonance query: motor_core (Core S0 generates the retrieval query)
   - Core representation: Core S0 re-scaled/normalized to match expected motor slot energy (~25.0 norm)
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

def eval_architecture(model, cfg, prompt_tokens, target_tokens, arch_type="Current"):
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

        # -------------------------------------------------------------
        # Determine Query Source for Cross-Resonance:
        # In Current & Arch A: Query from motor_surface (Shell 15)
        # In Arch B & Arch C:  Query from motor_core (Shell 0 deliberated)
        # -------------------------------------------------------------
        motor_surface = surface.unsqueeze(1) + p_scaled.unsqueeze(0) + tf_flux.to(dt)
        motor_core = cur_core.unsqueeze(1) + p_scaled.unsqueeze(0) + tf_flux.to(dt)

        if arch_type in ("Current", "Arch_A"):
            query_input = motor_surface
        else:  # Arch_B or Arch_C
            query_input = motor_core

        resonance = model.motor_decoder._parallel_cross_resonance(
            motor_surface=query_input, sensory_kv=sensory_kv, target_kv=None,
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

        # Core scaling for Architecture C (Rebalanced to surface magnitude)
        if arch_type == "Arch_C":
            # Surface norm is ~35, core norm is ~2.4; boost core to parity
            core_boost = 35.0 / max(1e-4, cur_core.norm(dim=-1, keepdim=True).mean().item())
            effective_core = cur_core * core_boost
        else:
            effective_core = cur_core

        hs = []
        for t in range(T_seq):
            if arch_type == "Current":
                parts = [
                    motor_surface[:, t, :] * g[0],
                    cur_core * g[1],
                    resonance[:, t, :] * g[2],
                    sp_t * g[3],
                    cur_plan * g[4],
                ]
            elif arch_type == "Arch_A":
                # S0 only to motor, Surface query for resonance
                parts = [
                    torch.zeros_like(motor_surface[:, t, :]),
                    cur_core * g[1],
                    resonance[:, t, :] * g[2],
                    torch.zeros_like(sp_t),
                    cur_plan * g[4],
                ]
            elif arch_type == "Arch_B":
                # S0 only to motor, Core query for resonance
                parts = [
                    torch.zeros_like(motor_surface[:, t, :]),
                    cur_core * g[1],
                    resonance[:, t, :] * g[2],
                    torch.zeros_like(sp_t),
                    cur_plan * g[4],
                ]
            elif arch_type == "Arch_C":
                # S0 only to motor, Core query for resonance, Core energy normalized
                parts = [
                    torch.zeros_like(motor_surface[:, t, :]),
                    effective_core * g[1],
                    resonance[:, t, :] * g[2],
                    torch.zeros_like(sp_t),
                    cur_plan * g[4],
                ]

            fused_in = torch.cat(parts, dim=-1)
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

def run_benchmark():
    ckpt_path = "checkpoints/fresh_scratch/v5_small_latest.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    print("================================================================================")
    print("   CEREBRO SPHEROID V5: CENTRALIZED READOUT BOTTLENECK AUDIT (A, B, C)          ")
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

    architectures = [
        ("Current Baseline", "Current"),
        ("Architecture A (S0 only, Surface Query)", "Arch_A"),
        ("Architecture B (S0 only, Core Query)", "Arch_B"),
        ("Architecture C (S0 only, Core Query, Re-norm Core)", "Arch_C"),
    ]

    results = {name: {} for name, _ in architectures}

    for name, code in architectures:
        for d, t_in in domain_tokens.items():
            loss = eval_architecture(model, cfg, t_in[:, :-1], t_in[:, 1:], arch_type=code)
            results[name][d] = loss

    header = f"{'Architecture':<42} | " + " | ".join([f"{d[:12]:<12}" for d in DOMAINS])
    print("\n" + header)
    print("-" * len(header))
    for name, _ in architectures:
        row = f"{name:<42} | " + " | ".join([f"{results[name][d]:<12.4f}" for d in DOMAINS])
        print(row)

    print("\n% Change relative to Current Baseline (Negative is Better):")
    print(header)
    print("-" * len(header))
    base = results["Current Baseline"]
    for name, _ in architectures:
        if name == "Current Baseline":
            continue
        deltas = []
        for d in DOMAINS:
            pct = ((results[name][d] - base[d]) / base[d]) * 100.0
            deltas.append(f"{pct:>+11.2f}%")
        print(f"{name:<42} | " + " | ".join(deltas))

    print("\n================================================================================")
    print("   BENCHMARK COMPLETE                                                           ")
    print("================================================================================")

if __name__ == "__main__":
    run_benchmark()
