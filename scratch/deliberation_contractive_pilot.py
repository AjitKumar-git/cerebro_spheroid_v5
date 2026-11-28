#!/usr/bin/env python3
"""
scratch/deliberation_contractive_pilot.py

Phase 1 Pilot Ablation: Contractive Deliberation Operator
Tests 4 candidate regimes across thinking cycles c = 0..8:
  Regime 0 (Baseline): Raw unconstrained residual addition S_{c+1} = S_c + Delta S_c
  Regime 1 (Damped Step): Alpha_c = 0.6^{c-1} step decay
  Regime 2 (Scale Conservation): S_{c+1} = RMSNorm(S_c + Delta S_c) * ||S_0||
  Regime 3 (Damped + Scale Conservation): S_{c+1} = RMSNorm(S_c + Alpha_c Delta S_c) * ||S_0|| with Alpha_c = 0.75^{c-1}

Measures:
- Core Norm trajectory ||z_c||
- Local Lipschitz bound L_c
- Step convergence delta_c
- Loss trajectory on Python Code, Mathematics, and General Prose across c = 0..8
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import copy
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.corpus import build_tokenizer

PROMPTS = {
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
        "Let V be a finite-dimensional inner product space over the complex field. "
        "A linear operator T on V is normal if and only if it commutes with its adjoint. "
        "By the spectral theorem, T is normal if and only if there exists an orthonormal basis of V "
        "consisting of eigenvectors of T, with corresponding eigenvalues representing diagonal entries."
    ),
    "general_prose": (
        "The historical development of urban infrastructure in nineteenth-century Europe "
        "reflected deep socio-economic transformations. As industrial centers expanded rapidly, "
        "municipal governments were forced to implement centralized sewage, clean water aqueducts, "
        "and wide boulevards to mitigate public health crises and facilitate commercial transport."
    ),
}

def run_regime(
    model: CerebroSpheroidV5,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    regime: str,
    max_cycles: int = 8,
    perturbation_scale: float = 1e-3,
) -> List[Dict[str, float]]:
    """Runs deliberation under a specified mathematical contraction regime."""
    model.eval()
    dev = prompt_tokens.device
    B, P = prompt_tokens.shape

    with torch.no_grad():
        sensory_flux, sensory_kv = model.sensory_dnn(prompt_tokens)
        prompt_centroid = model._recency_weighted_pool(sensory_flux)

        num_shells = model.config.num_shells
        radial_profile = (
            torch.arange(1, num_shells + 1, device=dev, dtype=torch.float32) / float(num_shells)
        ).view(1, -1, 1)
        state_0 = (prompt_centroid.unsqueeze(1).repeat(1, num_shells, 1).float() * radial_profile).to(
            prompt_centroid.dtype
        )
        initial_scale = state_0.norm(p=2, dim=-1, keepdim=True)  # [B, K, 1]

        # Prepare target teacher-forcing feedback
        t_len = target_tokens.size(1)
        pad_col = torch.full((B, 1), -100, dtype=target_tokens.dtype, device=dev)
        prev_tokens = torch.cat([pad_col, target_tokens[:, : max(0, t_len - 1)]], dim=1)[:, :t_len]
        safe_feedback = torch.clamp(prev_tokens, min=0)
        prompt_pos_offset = prompt_tokens.shape[1]
        tf_flux, (tf_keys, tf_values) = model.sensory_dnn(
            safe_feedback, is_causal=True, pos_offset=prompt_pos_offset
        )
        fb_valid_mask = prev_tokens >= 0
        tf_flux = tf_flux * fb_valid_mask.unsqueeze(-1).to(tf_flux.dtype)

        states = [state_0]
        cores = [state_0[:, 0, :].clone()]
        current_state = state_0.clone()
        lipschitz_bounds = [float("nan")]

        for c in range(1, max_cycles + 1):
            # Compute step size based on regime
            if regime == "regime_0_baseline":
                alpha_c = 1.0
                preserve_scale = False
            elif regime == "regime_1_damped":
                alpha_c = 0.6 ** (c - 1)
                preserve_scale = False
            elif regime == "regime_2_scale_norm":
                alpha_c = 1.0
                preserve_scale = True
            elif regime == "regime_3_damped_scale_norm":
                alpha_c = 0.75 ** (c - 1)
                preserve_scale = True
            else:
                raise ValueError(f"Unknown regime: {regime}")

            # Define step function
            def _step_fn(st: torch.Tensor, scale_factor: float, norm_conserve: bool) -> torch.Tensor:
                raw_next = model.volumetric_core.deliberate_step(st)
                delta = raw_next - st
                updated = st + scale_factor * delta
                if norm_conserve:
                    # Conserve the baseline laminar energy distribution
                    norm_updated = updated.norm(p=2, dim=-1, keepdim=True) + 1e-8
                    updated = (updated / norm_updated) * initial_scale
                return updated

            # Measure Local Lipschitz
            eta = torch.randn_like(current_state)
            eta = eta / (eta.norm(p=2, dim=-1, keepdim=True) + 1e-8) * perturbation_scale
            perturbed_out = _step_fn(current_state + eta, alpha_c, preserve_scale)
            clean_out = _step_fn(current_state, alpha_c, preserve_scale)
            diff_out = (perturbed_out - clean_out).norm(p=2, dim=-1).mean().item()
            lipschitz_bounds.append(diff_out / perturbation_scale)

            # Apply deliberate step
            current_state = clean_out
            states.append(current_state)
            cores.append(current_state[:, 0, :].clone())

        # Evaluate loss and step dynamics
        metrics = []
        for c in range(max_cycles + 1):
            st = states[c]
            z_c = cores[c]
            norm_z = float(z_c.norm(p=2, dim=-1).mean().item())

            logits, _, _ = model.motor_decoder.forward_sequence(
                state=st,
                sensory_kv=sensory_kv,
                target_feedback=tf_flux,
                target_kv=(tf_keys, tf_values),
                tied_embedding_weight=model.sensory_dnn.embedding.weight,
                efference_gate=model.volumetric_core.efference_gate,
                radial_norm=model.volumetric_core.radial_norm,
                plan=None,
                target_kv_mask=fb_valid_mask,
                prompt_valid_mask=None,
            )
            loss_c = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1)).item()

            if c == 0:
                rel_delta = 0.0
            else:
                delta_z = z_c - cores[c - 1]
                rel_delta = float(delta_z.norm(p=2, dim=-1).mean().item()) / (
                    cores[c - 1].norm(p=2, dim=-1).mean().item() + 1e-8
                )

            metrics.append({
                "cycle": c,
                "loss": loss_c,
                "core_norm": norm_z,
                "rel_delta": rel_delta,
                "lipschitz": lipschitz_bounds[c],
            })

    return metrics

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("🧠 PHASE 1 PILOT: BENCHMARKING CONTRACTIVE DELIBERATION OPERATORS")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 95 + "\n")

    ckpt_path = "checkpoints/fresh_scratch/v5_small_latest.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/v5_small_latest.pt"
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    tokenizer = build_tokenizer("llama")
    cfg = infer_v5_config_from_state_dict(raw_ckpt)
    cfg.use_per_token_efference = False
    cfg.use_centralized_readout = True
    model = CerebroSpheroidV5(config=cfg).to(device)
    load_v5_checkpoint_into_torch(model, raw_ckpt, verbose=False)

    regimes = [
        ("regime_0_baseline", "Regime 0: Baseline Raw Unconstrained"),
        ("regime_1_damped", "Regime 1: Damped Step (alpha_c = 0.6^{c-1})"),
        ("regime_2_scale_norm", "Regime 2: Scale Conservation Norm"),
        ("regime_3_damped_scale_norm", "Regime 3: Damped + Scale Conservation"),
    ]

    for d_name, text in PROMPTS.items():
        toks = torch.tensor([tokenizer.encode(text, add_special_tokens=True)], dtype=torch.long, device=device)
        p_toks = toks[:, :min(48, toks.size(1) // 2)]
        t_toks = toks[:, min(48, toks.size(1) // 2):]

        print(f"\n{'=' * 95}")
        print(f"DOMAIN: {d_name.upper()}")
        print(f"{'=' * 95}")

        regime_results = {}
        for r_key, r_title in regimes:
            res = run_regime(model, p_toks, t_toks, regime=r_key, max_cycles=8)
            regime_results[r_key] = res

        # Table comparing Loss trajectories
        print(f"\n--- Held-Out Cross-Entropy Loss Across Cycles (c = 0..8) ---")
        print(f"{'Regime':<35s} | {'c=0':<7s} | {'c=1':<7s} | {'c=2':<7s} | {'c=4':<7s} | {'c=8':<7s} | {'Min Loss (Cycle)':<16s}")
        print("-" * 95)
        for r_key, r_title in regimes:
            losses = [m["loss"] for m in regime_results[r_key]]
            min_c = int(torch.tensor(losses).argmin().item())
            min_loss = losses[min_c]
            print(f"{r_title:<35s} | {losses[0]:<7.4f} | {losses[1]:<7.4f} | {losses[2]:<7.4f} | {losses[4]:<7.4f} | {losses[8]:<7.4f} | {min_loss:.4f} (c={min_c})")

        # Table comparing Core Norm trajectories
        print(f"\n--- Core Norm Trajectory ||z_c|| Across Cycles (c = 0..8) ---")
        print(f"{'Regime':<35s} | {'c=0':<7s} | {'c=1':<7s} | {'c=2':<7s} | {'c=4':<7s} | {'c=8':<7s} | {'Scale Factor':<12s}")
        print("-" * 95)
        for r_key, r_title in regimes:
            norms = [m["core_norm"] for m in regime_results[r_key]]
            scale = norms[8] / (norms[0] + 1e-8)
            print(f"{r_title:<35s} | {norms[0]:<7.4f} | {norms[1]:<7.4f} | {norms[2]:<7.4f} | {norms[4]:<7.4f} | {norms[8]:<7.4f} | {scale:.2f}x")

        # Table comparing Lipschitz Bounds
        print(f"\n--- Mean Lipschitz Bound L_c (Contraction if < 1.0) ---")
        print(f"{'Regime':<35s} | {'Mean Lipschitz':<18s} | {'Final Step Delta (d_8)':<22s}")
        print("-" * 95)
        for r_key, r_title in regimes:
            lips = [m["lipschitz"] for m in regime_results[r_key] if not math.isnan(m["lipschitz"])]
            mean_lip = sum(lips) / len(lips)
            last_delta = regime_results[r_key][8]["rel_delta"]
            status = "CONTRACTIVE" if mean_lip < 1.0 else "EXPANSIVE"
            print(f"{r_title:<35s} | {mean_lip:<8.4f} ({status:<11s}) | {last_delta:<22.6f}")

    print("\n" + "=" * 95)
    print("PILOT BENCHMARK COMPLETE")
    print("=" * 95)

if __name__ == "__main__":
    main()
