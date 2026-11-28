#!/usr/bin/env python3
"""
scratch/spectral_core_readout_pilot.py

Pilot evaluation of Core Spectral Readout formulations:
1. Baseline: Raw S0 (Centralized Bottleneck)
2. Direct DCT Spectral S0
3. Multi-Band Gain DCT (g_low, g_mid, g_high)
4. Residual Gated Spectral Readout: S0 + beta * Norm(Phi_spec(S0))
5. Fine-tuning calibration test: Short gradient update step to measure adaptation rate.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    infer_v5_config_from_state_dict,
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

def create_dct_matrix(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal Discrete Cosine Transform (DCT-II) matrix."""
    n = torch.arange(dim, device=device, dtype=torch.float32)
    k = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(1)
    M = torch.cos(math.pi / float(dim) * (n + 0.5) * k)
    M[0, :] = M[0, :] * (1.0 / math.sqrt(float(dim)))
    M[1:, :] = M[1:, :] * math.sqrt(2.0 / float(dim))
    return M.to(dtype)

class SpectralCoreReadoutLayer(nn.Module):
    """
    Structured Spectral Core Readout:
    Decomposes S0 into orthonormal frequency bands via DCT basis U.
    Applies learned multi-band gain filtering and residual gating.
    """
    def __init__(self, dim: int, low_dim: int = 160, mid_dim: int = 320, high_dim: int = 160):
        super().__init__()
        self.dim = dim
        self.low_dim = low_dim
        self.mid_dim = mid_dim
        self.high_dim = high_dim
        assert low_dim + mid_dim + high_dim == dim

        # Fixed orthonormal DCT basis
        dct_mat = create_dct_matrix(dim, torch.device("cpu"), torch.float32)
        self.register_buffer("dct_basis", dct_mat)

        # Learnable spectral band gain parameters
        self.band_gains = nn.Parameter(torch.ones(3, dtype=torch.float32))
        
        # Learnable residual mixing gate (initially zero for exact backward compatibility)
        self.residual_gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.norm = RMSNorm(dim)

    def forward(self, s0: torch.Tensor, mode: str = "residual") -> torch.Tensor:
        # s0: [B, D]
        dt = s0.dtype
        # Compute orthonormal spectrum: S_hat = S0 @ U^T
        s_hat = F.linear(s0, self.dct_basis.to(dt))  # [B, D]

        # Multi-band slicing
        z_low = s_hat[:, :self.low_dim] * self.band_gains[0].to(dt)
        z_mid = s_hat[:, self.low_dim:self.low_dim + self.mid_dim] * self.band_gains[1].to(dt)
        z_high = s_hat[:, self.low_dim + self.mid_dim:] * self.band_gains[2].to(dt)

        filtered_spectrum = torch.cat([z_low, z_mid, z_high], dim=-1)

        if mode == "direct_dct":
            return s_hat
        elif mode == "multiband_dct":
            return filtered_spectrum
        elif mode == "residual":
            beta = torch.tanh(self.residual_gate).to(dt)
            return self.norm(s0 + beta * filtered_spectrum)
        else:
            return s0

def evaluate_readout(model, cfg, prompt, target, spectral_layer, mode: str) -> float:
    B = prompt.shape[0]
    dev = prompt.device
    with torch.no_grad():
        sensory_flux, sensory_kv = model.sensory_dnn(prompt)
        prompt_centroid = model._recency_weighted_pool(sensory_flux)
        rad_prof = (torch.arange(1, cfg.num_shells + 1, device=dev, dtype=torch.float32) / float(cfg.num_shells)).view(1, -1, 1)
        state = (prompt_centroid.unsqueeze(1).repeat(1, cfg.num_shells, 1).float() * rad_prof).to(prompt_centroid.dtype)

        cycles = cfg.thinking_cycles
        init_scale = state.norm(p=2, dim=-1, keepdim=True) + 1e-8
        for cycle in range(cycles):
            raw_next = model.volumetric_core.deliberate_step(state)
            alpha = float(0.75 ** cycle)
            updated = state + alpha * (raw_next - state)
            state = (updated / (updated.norm(p=2, dim=-1, keepdim=True) + 1e-8)) * init_scale

        tf_flux, _ = model.sensory_dnn(target)
        T_seq = tf_flux.shape[1]
        cur_core = state[:, 0, :]
        dt = cur_core.dtype

        if mode == "baseline":
            effective_core = cur_core
        else:
            effective_core = spectral_layer(cur_core, mode=mode)

        plan = model.motor_decoder.init_plan(B, state.device).to(dt)
        p_scaled = model.motor_decoder._prepare_phase_vectors(torch.arange(T_seq, device=state.device), dt)

        # In Architecture B, cross-resonance query is driven by core
        motor_core = effective_core.unsqueeze(1) + p_scaled.unsqueeze(0) + tf_flux.to(dt)
        resonance = model.motor_decoder._parallel_cross_resonance(
            motor_surface=motor_core, sensory_kv=sensory_kv, target_kv=None,
            target_kv_mask=None, prompt_valid_mask=None, chunk_offset=0
        )

        cur_plan = plan.to(dt)
        pd = torch.clamp(model.motor_decoder.plan_decay.float(), 0.0, 0.92).to(dt) if model.motor_decoder.use_plan else None
        pw = model.motor_decoder.plan_write.to(dt) if model.motor_decoder.use_plan else None
        g = torch.clamp(model.motor_decoder.input_gate.float(), -5.0, 5.0).to(dt)

        hs = []
        for t in range(T_seq):
            parts = [
                torch.zeros_like(motor_core[:, t, :]),
                effective_core * g[1],
                resonance[:, t, :] * g[2],
                torch.zeros_like(effective_core),
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

def main():
    ckpt_path = "checkpoints/fresh_scratch/v5_small_latest.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found")
        return

    print("================================================================================")
    print("   PILOT AUDIT: CORE SPECTRAL READOUT FORMULATIONS                             ")
    print("================================================================================")
    cp = torch.load(ckpt_path, map_location="cpu")
    cfg = infer_v5_config_from_state_dict(cp)
    cfg.use_per_token_efference = False
    model = CerebroSpheroidV5(config=cfg)
    model.eval()

    tok = build_tokenizer("data/tokenizer")
    spectral_layer = SpectralCoreReadoutLayer(cfg.surface_dim).eval()

    modes = [
        ("baseline", "Architecture B Baseline (Raw S0)"),
        ("direct_dct", "Direct DCT Transform (Pure Spectrum)"),
        ("multiband_dct", "Multi-band Gain DCT (g_L, g_M, g_H)"),
        ("residual", "Gated Residual Spectral Readout (S0 + beta*Phi_spec)"),
    ]

    results = {m[0]: {} for m in modes}

    for d_name, d_text in DOMAINS.items():
        ids = tok.encode(d_text)
        mid = len(ids) // 2
        p_ids = torch.tensor([ids[:mid]], dtype=torch.long)
        t_ids = torch.tensor([ids[mid:]], dtype=torch.long)

        for mode_key, _ in modes:
            loss = evaluate_readout(model, cfg, p_ids, t_ids, spectral_layer, mode_key)
            results[mode_key][d_name] = loss

    print(f"\n{'Mode':<45} | {'General':<10} | {'Python':<10} | {'Math':<10} | {'Biochem':<10}")
    print("-" * 95)
    for mode_key, mode_desc in modes:
        res = results[mode_key]
        print(f"{mode_desc:<45} | {res['general_prose']:<10.4f} | {res['python_code']:<10.4f} | {res['mathematics']:<10.4f} | {res['stem_biochem']:<10.4f}")

    print("\n--- Adaptation Test: 10 Steps of Fine-Tuning Band Gains on Code & Math ---")
    # Test how fast the multiband / residual layer optimizes its band gains
    spectral_trainable = SpectralCoreReadoutLayer(cfg.surface_dim)
    # Start with slight positive beta for adaptation
    with torch.no_grad():
        spectral_trainable.residual_gate.fill_(0.1)
    optimizer = torch.optim.Adam(spectral_trainable.parameters(), lr=1e-3)

    math_ids = tok.encode(DOMAINS["mathematics"])
    m_p = torch.tensor([math_ids[:len(math_ids)//2]], dtype=torch.long)
    m_t = torch.tensor([math_ids[len(math_ids)//2:]], dtype=torch.long)

    init_loss = evaluate_readout(model, cfg, m_p, m_t, spectral_trainable, "residual")
    for step in range(15):
        optimizer.zero_grad()
        # forward pass with grad
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

        tf_flux, _ = model.sensory_dnn(m_t)
        cur_core = state[:, 0, :]
        eff_core = spectral_trainable(cur_core, mode="residual")
        p_scaled = model.motor_decoder._prepare_phase_vectors(torch.arange(tf_flux.shape[1]), cur_core.dtype)
        motor_core = eff_core.unsqueeze(1) + p_scaled.unsqueeze(0) + tf_flux.to(cur_core.dtype)
        resonance = model.motor_decoder._parallel_cross_resonance(
            motor_surface=motor_core, sensory_kv=sensory_kv, target_kv=None,
            target_kv_mask=None, prompt_valid_mask=None, chunk_offset=0
        )
        plan = model.motor_decoder.init_plan(1, state.device).to(cur_core.dtype)
        g = torch.clamp(model.motor_decoder.input_gate.float(), -5.0, 5.0).to(cur_core.dtype)
        hs = []
        for t in range(tf_flux.shape[1]):
            parts = [
                torch.zeros_like(motor_core[:, t, :]),
                eff_core * g[1],
                resonance[:, t, :] * g[2],
                torch.zeros_like(eff_core),
                plan * g[4],
            ]
            fused_in = torch.cat(parts, dim=-1)
            fused = model.motor_decoder.fused_norm(fused_in)
            h_t = model.motor_decoder.norm_mid(model.motor_decoder.motor_swiglu(fused))
            hs.append(h_t)
        H = torch.stack(hs, dim=1)
        temp = torch.clamp(model.motor_decoder.temperature_scale.float(), 0.1, 5.0).to(cur_core.dtype)
        H_scaled = (H * model.motor_decoder.affine_scale.to(cur_core.dtype).view(1, 1, -1)) / temp
        logits = F.linear(H_scaled, model.sensory_dnn.embedding.weight.to(cur_core.dtype))
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), m_t.view(-1))
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    print(f"Math Loss Adaptation: {init_loss:.4f} -> {final_loss:.4f} (Delta: {final_loss - init_loss:+.4f})")
    print(f"Learned Band Gains: Low={spectral_trainable.band_gains[0].item():.4f}, Mid={spectral_trainable.band_gains[1].item():.4f}, High={spectral_trainable.band_gains[2].item():.4f}")
    print(f"Learned Residual Gate (beta): {torch.tanh(spectral_trainable.residual_gate).item():.4f}")
    print("================================================================================")

if __name__ == "__main__":
    main()
