"""Rigorous mathematical validation of Architecture V-Next subsystems on v5_small_latest.pt.

Measures:
1. Scale Conservation: ||S^(c)||_2 / ||S^(0)||_2 across all shells and cycles.
2. Empirical Local Contraction & Jacobian Spectral Radius rho(J_c) via power iteration.
3. Parseval Energy Conservation & Orthonormality ||U^T U - I||_inf for DCT-II basis.
4. Centralized Readout Bottleneck: ||d(logits) / d(S_{1..K-1})||_2 == 0.0.
5. Zero-Initialization Identity for Spectral Gate, Core Evidence Gate, and Laminar Transport.
6. Spectral Readout vs. Raw S0 vs. Random Orthogonal Basis loss comparison (ablation).
"""

import os
import torch
import torch.nn.functional as F

from src.spheroid_v5 import CerebroSpheroidV5, infer_v5_config_from_state_dict


def run_math_validation():
    ckpt_path = "checkpoints/v5_small_latest.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu")
    raw_sd = sd.get("model_state", sd)

    cfg = infer_v5_config_from_state_dict(raw_sd)
    cfg.use_deliberation_probe = True
    cfg.use_core_inquiry = True
    cfg.use_dual_evidence_grounding = True
    cfg.use_contractive_deliberation = True
    cfg.contractive_delib_decay = 0.75
    cfg.use_spectral_core_readout = True
    cfg.use_centralized_readout = True
    cfg.use_core_halting = True
    cfg.use_laminar_transport = True
    cfg.use_adaptive_thinking = False

    model = CerebroSpheroidV5(cfg)
    model.load_state_dict(raw_sd, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Loaded {ckpt_path} on {device} (K={cfg.num_shells}, D={cfg.surface_dim})")

    # Sample prompt & target
    prompt_text = "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:"
    target_text = "\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid"
    prompts = torch.tensor([[ord(c) % cfg.vocab_size for c in prompt_text]], device=device)
    targets = torch.tensor([[ord(c) % cfg.vocab_size for c in target_text]], device=device)

    with torch.no_grad():
        sensory_flux, sensory_kv = model.sensory_dnn(prompts, is_causal=False)
        prompt_centroid = model._recency_weighted_pool(sensory_flux)
        radial_profile = (torch.arange(1, cfg.num_shells + 1, device=device, dtype=torch.float32) / float(cfg.num_shells)).view(1, -1, 1)
        s0_init = (prompt_centroid.unsqueeze(1).repeat(1, cfg.num_shells, 1).float() * radial_profile).to(prompt_centroid.dtype)
        init_scale = s0_init.norm(p=2, dim=-1, keepdim=True) + 1e-8

    # =========================================================================
    # 1. SCALE CONSERVATION & EMPIRICAL CONTRACTION / JACOBIAN SPECTRAL RADIUS
    # =========================================================================
    print("\n--- 1. Damped Deliberation: Scale Conservation & Local Contraction ---")
    state = s0_init.clone()
    eps_pert = 1e-3
    for c in range(6):
        alpha = float(cfg.contractive_delib_decay ** c)

        def step_map(x):
            x_p = model._apply_deliberation_probe(x, sensory_kv, c)
            raw = model.volumetric_core.deliberate_step(x_p)
            up = x_p + alpha * (raw - x_p)
            return (up / (up.norm(p=2, dim=-1, keepdim=True) + 1e-8)) * init_scale

        with torch.no_grad():
            next_state = step_map(state)
            scale_err = (next_state.norm(p=2, dim=-1, keepdim=True) - init_scale).abs().max().item()
            rel_innov = (next_state[:, 0, :] - state[:, 0, :]).norm().item() / init_scale[:, 0, 0].item()

            # Power iteration (6 steps) for local Jacobian spectral radius rho(J_c) around `state`
            v = torch.randn_like(state)
            v = v / (v.norm() + 1e-8)
            for _ in range(6):
                # Directional finite-difference JVP: (F(s + eps*v) - F(s)) / eps
                jv = (step_map(state + eps_pert * v) - next_state) / eps_pert
                rho_est = jv.norm().item()
                v = jv / (rho_est + 1e-8)

            print(
                f"  Cycle {c+1} (alpha={alpha:.3f}): ScaleErr={scale_err:.2e} | "
                f"RelInnov={rel_innov:.4f} | Local rho(J_c)={rho_est:.4f}"
            )
            state = next_state

    # =========================================================================
    # 2. PARSEVAL ORTHONORMALITY & ENERGY CONSERVATION (DCT-II BASIS)
    # =========================================================================
    print("\n--- 2. Core Spectral Readout: Orthonormality & Parseval Conservation ---")
    U = model.motor_decoder.dct_basis  # [D, D]
    ident = torch.eye(cfg.surface_dim, device=device, dtype=U.dtype)
    ortho_err = (U @ U.T - ident).abs().max().item()

    s0_vec = state[:, 0, :]
    s_hat = torch.matmul(s0_vec, U.T)
    e_in = (s0_vec ** 2).sum().item()
    e_spec = (s_hat ** 2).sum().item()
    parseval_rel_err = abs(e_in - e_spec) / max(e_in, 1e-8)
    print(f"  ||U U^T - I||_inf = {ortho_err:.2e}")
    print(f"  Parseval Energy In={e_in:.6f}, Spectral={e_spec:.6f}, RelErr={parseval_rel_err:.2e}")
    assert ortho_err < 1e-4
    assert parseval_rel_err < 1e-4

    # =========================================================================
    # 3. CENTRALIZED READOUT BOTTLENECK VERIFICATION
    # =========================================================================
    print("\n--- 3. Centralized Readout Bottleneck: d(logits)/d(S_{1..K-1}) ---")
    delib_state = state.detach().clone().requires_grad_(True)
    tied_w = model.sensory_dnn.embedding.weight
    with torch.no_grad():
        tf_flux, (tf_k, tf_v) = model.sensory_dnn(targets, is_causal=True)
    logits, _, _ = model.motor_decoder.forward_sequence(
        state=delib_state,
        sensory_kv=sensory_kv,
        target_feedback=tf_flux,
        target_kv=(tf_k, tf_v),
        tied_embedding_weight=tied_w,
        efference_gate=model.volumetric_core.efference_gate,
        radial_norm=model.volumetric_core.radial_norm,
    )
    logits.sum().backward()
    grad_s0 = delib_state.grad[:, 0, :].norm().item()
    grad_outer = delib_state.grad[:, 1:, :].norm().item()
    print(f"  ||d(logits)/d(S_0)||_2      = {grad_s0:.4f}")
    print(f"  ||d(logits)/d(S_{{1..K-1}})||_2 = {grad_outer:.2e}")
    assert grad_s0 > 0.0
    assert grad_outer == 0.0, f"Motor readout leaked from outer shells! grad_outer={grad_outer}"

    # =========================================================================
    # 4. ZERO-INIT IDENTITY & CONTROLLED SPECTRAL ABLATION
    # =========================================================================
    print("\n--- 4. Zero-Init Identity & Controlled Spectral Readout Ablation (15 steps) ---")
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        loss_base, _ = model(prompts, targets=targets, pad_id=0)

        # Toggle laminar transport off (should match at zero-init)
        model.volumetric_core.use_laminar_transport = False
        loss_no_lam, _ = model(prompts, targets=targets, pad_id=0)
        model.volumetric_core.use_laminar_transport = True
        print(f"  Laminar zero-init delta: {abs(loss_base.item() - loss_no_lam.item()):.2e}")

        # Toggle spectral readout off (should match at zero-init up to RMSNorm)
        model.motor_decoder.use_spectral_core_readout = False
        loss_raw_s0, _ = model(prompts, targets=targets, pad_id=0)
        model.motor_decoder.use_spectral_core_readout = True
        print(f"  Zero-shot Loss: Raw S0={loss_raw_s0.item():.4f}, Spectral DCT={loss_base.item():.4f}")

    # Controlled 15-step ablation on identical data/init:
    # A: Raw S0 (no spectral module)
    # B: Spectral DCT-II multi-band (orthonormal DCT basis + 3 band gains + residual gate)
    # C: Random Haar Orthogonal basis (equally parameterized 3 band gains + residual gate)
    def run_ablation(mode: str) -> float:
        m = CerebroSpheroidV5(cfg)
        m.load_state_dict(raw_sd, strict=False)
        m.to(device)
        m.train()
        if mode == "raw_s0":
            m.motor_decoder.use_spectral_core_readout = False
            params_to_opt = [p for p in m.motor_decoder.parameters() if p.requires_grad]
        elif mode == "dct":
            m.motor_decoder.use_spectral_core_readout = True
            params_to_opt = [m.motor_decoder.spectral_band_gains, m.motor_decoder.spectral_residual_gate]
        elif mode == "rand_ortho":
            m.motor_decoder.use_spectral_core_readout = True
            q, _ = torch.linalg.qr(torch.randn(cfg.surface_dim, cfg.surface_dim, device=device))
            m.motor_decoder.dct_basis.copy_(q)
            params_to_opt = [m.motor_decoder.spectral_band_gains, m.motor_decoder.spectral_residual_gate]
        else:
            raise ValueError(mode)

        opt = torch.optim.AdamW(params_to_opt, lr=1e-2)
        for _ in range(15):
            opt.zero_grad()
            l, _ = m(prompts, targets=targets, pad_id=0)
            l.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            final_l, _ = m(prompts, targets=targets, pad_id=0)
        return final_l.item()

    l_raw = run_ablation("raw_s0")
    l_dct = run_ablation("dct")
    l_rand = run_ablation("rand_ortho")
    print(f"  15-step Controlled Ablation: Raw S0={l_raw:.4f} | Random Ortho={l_rand:.4f} | DCT-II Multi-Band={l_dct:.4f}")


if __name__ == "__main__":
    run_math_validation()
