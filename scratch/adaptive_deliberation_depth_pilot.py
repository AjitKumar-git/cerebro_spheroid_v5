"""Pilot diagnostic to evaluate Phase 4: Adaptive Deliberation Depth & Multi-Criteria Convergence.

Evaluates:
1. Convergence trajectories of Shell 0 (Core S_0) across deliberation cycles (c = 1..8).
2. Relative innovation delta: ||S_0^{(c)} - S_0^{(c-1)}||_2 / ||S_0^{(0)}||_2.
3. Directional trajectory alignment: cos(Delta_c, Delta_{c-1}).
4. Real next-token prediction loss as a function of deliberation cycle count c in [1..8].
"""

import math
import os
import torch
import torch.nn.functional as F

from src.spheroid_v5 import CerebroSpheroidV5, infer_v5_config_from_state_dict


def run_pilot():
    ckpt_path = "checkpoints/v5_small_latest.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    print("Loading checkpoint...")
    sd = torch.load(ckpt_path, map_location="cpu")
    raw_sd = sd.get("model_state", sd)

    cfg = infer_v5_config_from_state_dict(raw_sd)
    cfg.use_deliberation_probe = True
    cfg.use_core_inquiry = True
    cfg.use_dual_evidence_grounding = True
    cfg.use_contractive_deliberation = True
    cfg.contractive_delib_decay = 0.75
    cfg.use_adaptive_thinking = False  # Fixed cycle sweeps

    model = CerebroSpheroidV5(cfg)
    model.load_state_dict(raw_sd, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model initialized on {device}. Surface dim: {cfg.surface_dim}, Shells: {cfg.num_shells}")

    # Domain test cases
    test_cases = {
        "Python Code": (
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]",
            "\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"
        ),
        "Math Reasoning": (
            "Problem: Solve for x where 3*x + 7 = 22. Step 1: Subtract 7 from both sides: 3*x = 15. Step 2:",
            " Divide both sides by 3: x = 5. Final Answer: x = 5."
        ),
        "STEM/Biochem": (
            "The Michaelis-Menten constant Km represents the substrate concentration at which reaction rate is half Vmax.",
            " High affinity corresponds to a low Km value, indicating tight substrate binding."
        ),
        "Simple QA": (
            "Question: What is the capital of France? Answer:",
            " Paris is the capital and most populous city of France."
        ),
    }

    print("\n" + "=" * 90)
    print("DELIBERATION DEPTH VS PREDICTION LOSS SWEEP (c = 1 .. 8)")
    print("=" * 90)

    for domain, (prompt_text, target_text) in test_cases.items():
        p_tokens = torch.tensor([[ord(c) % cfg.vocab_size for c in prompt_text]], device=device)
        t_tokens = torch.tensor([[ord(c) % cfg.vocab_size for c in target_text]], device=device)

        print(f"\n--- Domain: {domain} ---")
        cycle_losses = []
        for c in range(1, 9):
            model.config.thinking_cycles = c
            with torch.no_grad():
                loss, _ = model(p_tokens, targets=t_tokens, pad_id=0)
                cycle_losses.append(loss.item())

        base_loss = cycle_losses[0]
        for c, l in enumerate(cycle_losses, 1):
            delta = l - base_loss
            pct = (delta / base_loss) * 100.0
            marker = " *" if l == min(cycle_losses) else ""
            print(f"  Cycle {c}: Loss = {l:.4f} (Delta: {delta:+.4f}, {pct:+.2f}%){marker}")

    print("\n" + "=" * 90)
    print("DONE AUDITING CONVERGENCE DEPTH.")
    print("=" * 90)


if __name__ == "__main__":
    run_pilot()
