"""Verification of Phase 4: Multi-Criteria Deliberation Halting & Centralized Core Halting Gate."""

import torch
import torch.nn.functional as F

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5


def test_phase4_mechanics():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    cfg.use_adaptive_thinking = True
    cfg.use_core_halting = True
    cfg.max_thinking_cycles = 6
    cfg.thinking_halt_threshold = 0.70
    cfg.deliberation_early_halt_thresh = 0.05
    cfg.deliberation_oscillation_halt = True

    model = CerebroSpheroidV5(cfg)
    model.train()

    B = 2
    prompts = torch.randint(1, cfg.vocab_size, (B, 8))
    targets = torch.randint(1, cfg.vocab_size, (B, 6))

    # Test training forward & backward with centralized core halting gate
    loss, extra = model(prompts, targets=targets, pad_id=0)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    loss.backward()

    assert model.halting_gate.weight.grad is not None, "Halting gate weight received no gradients!"
    assert torch.isfinite(model.halting_gate.weight.grad).all(), "Halting gate weight gradients have NaN/Inf!"
    print("✓ Training forward and backward through centralized core halting gate passed.")

    # Test autoregressive generation with early halting
    model.eval()
    with torch.no_grad():
        out = model.generate(prompts, max_new_tokens=4)
        assert out.shape == (B, 4)
    print("✓ Autoregressive generation with multi-criteria early halting passed.")


if __name__ == "__main__":
    test_phase4_mechanics()
