"""Pilot test for Phase 5: Controlled Inter-Shell Information Transport (Laminar Transport Dynamics)."""

import torch
import torch.nn.functional as F

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5, infer_v5_config_from_state_dict


def test_laminar_transport():
    ckpt_path = "checkpoints/v5_small_latest.pt"
    print("Loading checkpoint...")
    sd = torch.load(ckpt_path, map_location="cpu")
    raw_sd = sd.get("model_state", sd)

    cfg = infer_v5_config_from_state_dict(raw_sd)
    cfg.use_deliberation_probe = True
    cfg.use_core_inquiry = True
    cfg.use_dual_evidence_grounding = True
    cfg.use_contractive_deliberation = True
    cfg.use_spectral_core_readout = True
    cfg.use_core_halting = True
    cfg.use_laminar_transport = True

    model = CerebroSpheroidV5(cfg)
    model.load_state_dict(raw_sd, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1. Verification of identity with baseline
    # Temporarily compute step with and without laminar gates
    K = cfg.num_shells
    assert hasattr(model.volumetric_core, "laminar_in_gate")
    assert hasattr(model.volumetric_core, "laminar_out_gate")
    assert model.volumetric_core.laminar_in_gate.shape == (K - 1,)
    assert model.volumetric_core.laminar_out_gate.shape == (K - 1,)

    state = torch.randn(2, K, cfg.surface_dim, device=device)
    with torch.no_grad():
        step_laminar = model.volumetric_core.deliberate_step(state)

        # Baseline computation without laminar gates
        orig_use = model.volumetric_core.use_laminar_transport
        model.volumetric_core.use_laminar_transport = False
        step_base = model.volumetric_core.deliberate_step(state)
        model.volumetric_core.use_laminar_transport = orig_use

        diff = (step_laminar - step_base).abs().max().item()
        print(f"Max absolute difference between baseline and zero-init laminar transport: {diff:.2e}")
        assert diff < 1e-6, f"Parity broken! diff={diff}"
        print("✓ Exact zero-init numerical parity verified (diff < 1e-6)!")

    # 2. Gradient flow check
    model.train()
    prompts = torch.randint(1, cfg.vocab_size, (2, 8), device=device)
    targets = torch.randint(1, cfg.vocab_size, (2, 6), device=device)

    model.zero_grad(set_to_none=True)
    loss, _ = model(prompts, targets=targets, pad_id=0)
    loss.backward()

    assert model.volumetric_core.laminar_in_gate.grad is not None
    assert torch.isfinite(model.volumetric_core.laminar_in_gate.grad).all()
    assert model.volumetric_core.laminar_out_gate.grad is not None
    assert torch.isfinite(model.volumetric_core.laminar_out_gate.grad).all()
    print("✓ Laminar in/out gates receive clean finite gradients!")

    # 3. 15-step optimization on Math
    print("\nRunning 15-step fine-tuning on Math Reasoning...")
    optimizer = torch.optim.AdamW([
        {"params": [model.volumetric_core.laminar_in_gate, model.volumetric_core.laminar_out_gate], "lr": 1e-2},
    ])
    math_text = "Problem: Solve for x: 5*x + 15 = 40. Step 1: 5*x = 25. Step 2: x = 5. Final Answer: 5."
    tokens = torch.tensor([[ord(c) % cfg.vocab_size for c in math_text]], device=device)
    p = tokens[:, :30]
    t = tokens[:, 30:]

    init_loss = None
    for step in range(15):
        optimizer.zero_grad()
        loss, _ = model(p, targets=t, pad_id=0)
        if init_loss is None:
            init_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    print(f"Initial Math Loss: {init_loss:.4f} -> Final Math Loss: {final_loss:.4f} (Delta: {final_loss - init_loss:+.4f})")

    tau_in = 2.0 * torch.sigmoid(model.volumetric_core.laminar_in_gate).detach().cpu()
    tau_out = 2.0 * torch.sigmoid(model.volumetric_core.laminar_out_gate).detach().cpu()
    print("Learned Inward Laminar Transport Factors (Shells 1..15 -> 0..14):")
    print(" ", [f"{v:.3f}" for v in tau_in.tolist()])
    print("Learned Outward Laminar Transport Factors (Shells 0..14 -> 1..15):")
    print(" ", [f"{v:.3f}" for v in tau_out.tolist()])


if __name__ == "__main__":
    test_laminar_transport()
