#!/usr/bin/env python3
"""
diagnostics/spectral_core_readout_audit.py

Empirical Spectral Core Readout Diagnostic Suite:
Testing the Spectral vs. Orthogonal vs. Raw Centralized Core Readout Hypotheses:

The 4 Evaluated Configurations:
-----------------------------------------------------------------------------
1. Experiment A (Current Baseline):
   - Motor inputs: [motor_surface, cur_core, resonance(Q=surface), shell_pool, plan]
   - Raw coordinate system, direct outer bypasses active.

2. Experiment B (Centralized Bottleneck - Raw S0):
   - Motor inputs: [0, cur_core, resonance(Q=core), 0, plan]
   - Outer bypasses disabled, cross-resonance query anchored in deliberated core S0.

3. Experiment C (Spectral Band Decomposition via DCT / Real-FFT):
   - Motor inputs: [0, Phi_spec(cur_core), resonance(Q=core), 0, plan]
   - Discrete Cosine Transform (DCT Type-II / Orthonormal Chebyshev basis) decomposes S0 into:
     * Low-Frequency Band (DC / broad semantic attractor): dims 0..D/4
     * Mid-Frequency Band (Harmonics / relational structure): dims D/4..3D/4
     * High-Frequency Band (Fine detail / syntax variations): dims 3D/4..D
   - Orthonormality ensures ||Phi_spec(S0)||_2 == ||S0||_2 (Parseval's theorem), preserving energy.

4. Experiment D (Learned / Deterministic Orthogonal Basis Control - Haar Wavelet / Random Orthogonal):
   - Motor inputs: [0, W_orth @ cur_core, resonance(Q=core), 0, plan]
   - Pure orthonormal coordinate rotation with W_orth @ W_orth.T = I.
   - Evaluates whether coordinate rotation alone changes motor projection versus true frequency ordering.
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

def create_dct_matrix(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal Discrete Cosine Transform (DCT-II) matrix."""
    n = torch.arange(dim, device=device, dtype=torch.float32)
    k = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(1)
    M = torch.cos(math.pi / float(dim) * (n + 0.5) * k)
    M[0, :] = M[0, :] * (1.0 / math.sqrt(float(dim)))
    M[1:, :] = M[1:, :] * math.sqrt(2.0 / float(dim))
    return M.to(dtype)

def create_orthogonal_matrix(dim: int, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic random orthogonal matrix via QR decomposition."""
    gen = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn((dim, dim), generator=gen, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(A)
    # Ensure determinant is +1 (proper rotation)
    d = torch.diag(r)
    ph = d.sign()
    q = q * ph.unsqueeze(0)
    return q.to(dtype)

def eval_readout_experiment(
    model, cfg, prompt_tokens, target_tokens,
    exp_type: str = "Exp_A",
    dct_mat: torch.Tensor = None,
    orth_mat: torch.Tensor = None
) -> float:
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
        # 1. Cross-Resonance Query Selection
        # Exp A: Surface Query (S15 + phase + tf)
        # Exp B, C, D: Core Query (S0 + phase + tf)
        # -------------------------------------------------------------
        motor_surface = surface.unsqueeze(1) + p_scaled.unsqueeze(0) + tf_flux.to(dt)
        motor_core = cur_core.unsqueeze(1) + p_scaled.unsqueeze(0) + tf_flux.to(dt)

        query_input = motor_surface if exp_type == "Exp_A" else motor_core

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

        # -------------------------------------------------------------
        # 2. Core Representation Transformation:
        # Exp A & B: Raw cur_core
        # Exp C: DCT spectral transform Phi_spec(S0) = DCT @ S0
        # Exp D: Orthonormal basis rotation W_orth @ S0
        # -------------------------------------------------------------
        if exp_type in ("Exp_A", "Exp_B"):
            transformed_core = cur_core
        elif exp_type == "Exp_C":
            # Discrete Cosine Transform (Spectral frequency decomposition)
            transformed_core = F.linear(cur_core, dct_mat)
        elif exp_type == "Exp_D":
            # Arbitrary orthonormal coordinate rotation (Control)
            transformed_core = F.linear(cur_core, orth_mat)
        else:
            raise ValueError(f"Unknown experiment: {exp_type}")

        hs = []
        for t in range(T_seq):
            if exp_type == "Exp_A":
                # Baseline: All bypasses enabled
                parts = [
                    motor_surface[:, t, :] * g[0],
                    cur_core * g[1],
                    resonance[:, t, :] * g[2],
                    sp_t * g[3],
                    cur_plan * g[4],
                ]
            else:
                # Exp B, C, D: Centralized Bottleneck (Slots 0 and 3 muted)
                parts = [
                    torch.zeros_like(motor_surface[:, t, :]),
                    transformed_core * g[1],
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

def run_spectral_audit():
    ckpt_path = "checkpoints/fresh_scratch/v5_small_latest.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    print("================================================================================")
    print("   CEREBRO SPHEROID V5: SPECTRAL CORE READOUT DIAGNOSTIC (A, B, C, D)           ")
    print("================================================================================")
    cp = torch.load(ckpt_path, map_location="cpu")
    cfg = infer_v5_config_from_state_dict(cp)
    cfg.use_per_token_efference = False
    model = CerebroSpheroidV5(config=cfg)
    model.eval()
    load_v5_checkpoint_into_torch(model, cp, verbose=False)
    tokenizer = build_tokenizer("llama")

    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    dct_mat = create_dct_matrix(cfg.surface_dim, dev, dt)
    orth_mat = create_orthogonal_matrix(cfg.surface_dim, seed=42, device=dev, dtype=dt)

    # Verify orthonormality (Parseval energy preservation)
    dct_err = torch.norm(dct_mat @ dct_mat.T - torch.eye(cfg.surface_dim, device=dev, dtype=dt)).item()
    orth_err = torch.norm(orth_mat @ orth_mat.T - torch.eye(cfg.surface_dim, device=dev, dtype=dt)).item()
    print(f"Spectral DCT-II Matrix: D={cfg.surface_dim}, Orthonormality Error: {dct_err:.2e}")
    print(f"Orthogonal Rotation Matrix: D={cfg.surface_dim}, Orthonormality Error: {orth_err:.2e}")

    domain_tokens = {}
    for d, text in DOMAINS.items():
        toks = tokenizer.encode(text, add_special_tokens=True)
        if len(toks) > 64:
            toks = toks[:64]
        domain_tokens[d] = torch.tensor([toks], dtype=torch.long)

    experiments = [
        ("Experiment A: Baseline (All direct paths)", "Exp_A"),
        ("Experiment B: Bottleneck Raw S0 (Core Query)", "Exp_B"),
        ("Experiment C: Spectral DCT S0 (Core Query)", "Exp_C"),
        ("Experiment D: Orthogonal Basis S0 (Core Query)", "Exp_D"),
    ]

    results = {name: {} for name, _ in experiments}

    for name, code in experiments:
        for d, t_in in domain_tokens.items():
            loss = eval_readout_experiment(
                model, cfg, t_in[:, :-1], t_in[:, 1:],
                exp_type=code, dct_mat=dct_mat, orth_mat=orth_mat
            )
            results[name][d] = loss

    header = f"{'Experiment':<46} | " + " | ".join([f"{d[:12]:<12}" for d in DOMAINS])
    print("\n" + header)
    print("-" * len(header))
    for name, _ in experiments:
        row = f"{name:<46} | " + " | ".join([f"{results[name][d]:<12.4f}" for d in DOMAINS])
        print(row)

    print("\n% Change relative to Experiment A (Current Baseline):")
    print(header)
    print("-" * len(header))
    base = results["Experiment A: Baseline (All direct paths)"]
    for name, _ in experiments:
        if name == "Experiment A: Baseline (All direct paths)":
            continue
        deltas = []
        for d in DOMAINS:
            pct = ((results[name][d] - base[d]) / base[d]) * 100.0
            deltas.append(f"{pct:>+11.2f}%")
        print(f"{name:<46} | " + " | ".join(deltas))

    print("\n% Change relative to Experiment B (Bottleneck Raw S0):")
    print(header)
    print("-" * len(header))
    base_b = results["Experiment B: Bottleneck Raw S0 (Core Query)"]
    for name in ["Experiment C: Spectral DCT S0 (Core Query)", "Experiment D: Orthogonal Basis S0 (Core Query)"]:
        deltas = []
        for d in DOMAINS:
            pct = ((results[name][d] - base_b[d]) / base_b[d]) * 100.0
            deltas.append(f"{pct:>+11.2f}%")
        print(f"{name:<46} | " + " | ".join(deltas))

    print("\n================================================================================")
    print("   SPECTRAL AUDIT COMPLETE                                                      ")
    print("================================================================================")

if __name__ == "__main__":
    run_spectral_audit()
