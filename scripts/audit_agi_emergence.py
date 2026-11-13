#!/usr/bin/env python3
"""
CerebroSpheroid V5: AGI Emergence Audit CLI Tool
=================================================

Inspects, visualizes, and audits continuous spheroidal brain manifolds
for emergent cognitive features, macro-circuits, and dynamical safety.

Usage:
    python3 scripts/audit_agi_emergence.py --checkpoint /path/to/checkpoint.pt
    python3 scripts/audit_agi_emergence.py --preset v5_small --device cuda
"""

import argparse
import json
import os
import sys
import torch

# Ensure repository root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    PRESETS_V5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.emergence_detector import AGIEmergenceAuditor, EmergenceReport


def parse_args():
    parser = argparse.ArgumentParser(description="Audit AGI Emergence & Manifold Safety in CerebroSpheroid V5")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--preset", type=str, default="v5_small", help="Architecture preset name if no checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--output-json", type=str, default=None, help="Path to export report JSON")
    parser.add_argument("--num-samples", type=int, default=4, help="Batch size of audit stimuli")
    parser.add_argument("--seq-len", type=int, default=64, help="Stimulus sequence length for probe")
    return parser.parse_args()


def render_ascii_bar(val: float, max_val: float = 1.5, width: int = 24) -> str:
    """Renders a normalized ASCII energy bar."""
    ratio = min(1.0, max(0.0, val / max_val))
    filled = int(round(ratio * width))
    empty = width - filled
    return "█" * filled + "░" * empty


def print_dashboard(report: EmergenceReport):
    """Prints a beautiful, rich terminal dashboard of the emergence audit."""
    print("\n" + "=" * 80)
    print("🧠 CEREBROSPHEROID V5: AGI EMERGENCE & MANIFOLD SAFETY AUDIT")
    print("=" * 80)
    print(f"Timestamp       : {report.timestamp}")
    print(f"Checkpoint      : {report.checkpoint_path or 'Uninitialized / Scratch'}")
    print(f"Preset Target   : {report.preset}")
    print(f"Total Parameters: {report.total_params:,} (Active Compute: {report.active_params:,} | {report.active_ratio * 100:.1f}%)")
    print(f"Manifold Layout : {report.num_shells} Concentric Shells | Surface Dim: {report.surface_dim} | Radial Dim: {report.radial_dim}")
    print("=" * 80)

    # 1. Laminar Profile
    lp = report.laminar_profile
    print("\n[1/5] 🌐 LAMINAR DIFFERENTIATION & PROTOBRAIN PROFILE")
    print("-" * 80)
    print(f"Effective Dimensional Rank : {lp['effective_dimensional_rank']} / {report.surface_dim} ({lp['rank_utilization_ratio']*100:.1f}% capacity)")
    print(f"Asymmetric Inversion Delta : {lp['inversion_sensitivity_delta']} (Biological Radial Hierarchy: {lp['has_laminar_differentiation']})")
    print(f"Mean Inter-Shell Coupling  : {lp['mean_adjacent_cosine']} cos-sim")
    print("\nConcentric Shell Kinetic Energy Profile (RMS):")
    for k, rms in enumerate(lp["shell_rms_energy"]):
        tag = ""
        if k == 0:
            tag = " [DEEP CORE / HIPPOCAMPUS]"
        elif k == report.num_shells - 1:
            tag = " [MOTOR SURFACE / BROCA]"
        elif k == report.num_shells // 2:
            tag = " [ASSOCIATION MANTLE]"
        bar = render_ascii_bar(rms, max_val=1.4, width=22)
        print(f"  Shell {k:02d} [{bar}] {rms:.4f} RMS{tag}")

    # 2. Connectome Topology
    ct = report.connectome_topology
    print("\n[2/5] ⚡ CONNECTOME TOPOLOGY & AXONAL HIGHWAYS")
    print("-" * 80)
    print(f"Total Axonal Tracts       : {ct['num_tracts']}")
    print(f"Connectome Asymmetry Ratio: {ct['asymmetry_ratio']}x (Peak vs Median Tract Strength)")
    print(f"Reciprocal Macro-Circuit  : {'DETECTED (Papez / Fornix Equivalent) ✅' if ct['has_reciprocal_macro_circuit'] else 'Not yet formed'}")
    print("\nDominant White-Matter Bundles:")
    for t in ct["top_tracts"]:
        print(f"  #{t['rank']}: Tract {t['tract_idx']:02d} (Shell {t['source_shell']:02d} -> Shell {t['destination_shell']:02d}) | Norm: {t['frobenius_norm']:.4f} | Cond: {t['conductance']:.4f} | Eff: {t['effective_strength']:.4f}")

    # 3. Proprioceptive Autonomy
    pa = report.proprioceptive_autonomy
    print("\n[3/5] 🔄 PROPRIOCEPTIVE AUTONOMY & COROLLARY DISCHARGE")
    print("-" * 80)
    print(f"Sensory Input Norm        : {pa['sensory_centroid_norm']:.4f}")
    print(f"Motor Efference Norm      : {pa['motor_efference_norm']:.4f}")
    print(f"Deep Core Displacement    : {pa['shell0_displacement']:.4f}")
    print(f"Corollary Discharge Ratio : {pa['corollary_discharge_ratio']:.4f}")
    print(f"Autonomy Classification   : {pa['autonomy_classification']}")

    # 4. Deliberation Dynamics
    dd = report.deliberation_dynamics
    print("\n[4/5] 💭 SUBCONSCIOUS RECURRENT DELIBERATION")
    print("-" * 80)
    print(f"Multi-Step Trajectory Drift : {dd['total_deliberation_drift']:.4f}")
    print(f"Trajectory Convergence      : {'Contractive Attractor (Stable) ✅' if dd['is_trajectory_contractive'] else 'Expanding / Fluctuating'}")
    if dd["has_adaptive_halting"]:
        print(f"Mean Halting Probability    : {dd['mean_halting_probability']:.4f}")

    # 5. Lyapunov & Spectral Stability
    ls = report.lyapunov_stability
    print("\n[5/5] ⚖️ LYAPUNOV STABILITY & CRITICALITY BOUNDARY")
    print("-" * 80)
    print(f"Peak Radial Spectral Radius : {ls['peak_spectral_radius']:.4f} (Contractive Guard <= 1.05: {ls['is_spectral_contractive']})")
    print(f"Lyapunov Exponent (lambda)  : {ls['finite_time_lyapunov_exponent']:.4f}")
    print(f"Dynamical Regime            : {ls['dynamical_regime']}")

    # Final Verdict
    print("\n" + "=" * 80)
    print("🏁 FINAL AGI EMERGENCE & SCALE-UP SAFETY VERDICT")
    print("=" * 80)
    print(f"AGI Emergence Score  : {report.agi_emergence_score} / 100")
    print(f"Safety Status        : {report.safety_classification}")
    print(f"Scale-Up Permission  : {'GRANTED ✅' if report.is_safe_to_scale else 'HOLD / HAZARD DETECTED ❌'}")
    print("\nRecommendations for Model Scaling:")
    for rec in report.scale_up_recommendations:
        print(f"  • {rec}")
    print("=" * 80 + "\n")


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from: {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = infer_v5_config_from_state_dict(ckpt, default_preset=args.preset)
        model = CerebroSpheroidV5(cfg).to(device)
        loaded, total = load_v5_checkpoint_into_torch(model, ckpt, verbose=False)
        print(f"Model ready on {device}! Loaded {loaded}/{total} tensors.")
    else:
        preset_name = args.preset
        cfg_dict = PRESETS_V5.get(preset_name, PRESETS_V5["v5_small"])
        cfg = ModelConfigV5(**cfg_dict)
        model = CerebroSpheroidV5(cfg).to(device)
        print(f"Initialized fresh CerebroSpheroidV5 ({preset_name}) on {device}.")

    auditor = AGIEmergenceAuditor(model, device=device)
    sample_ids = torch.randint(10, min(1000, model.config.vocab_size), (args.num_samples, args.seq_len), device=device)

    report = auditor.compute_full_agi_emergence_report(
        checkpoint_path=args.checkpoint,
        sample_input_ids=sample_ids,
    )

    print_dashboard(report)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"✓ Emergence Report JSON saved to: {args.output_json}")


if __name__ == "__main__":
    main()
