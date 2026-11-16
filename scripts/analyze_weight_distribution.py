#!/usr/bin/env python3
"""
Deep Weight Distribution & Spectral Analysis across Native Trained and Converted Spheroid V5 Models.
Handles both native packed 3D JAX/PyTorch layout and expanded ModuleList layouts.
"""

import os
import sys
import math
import torch
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.spheroid_v5 import (
    infer_v5_config_from_state_dict,
)


def analyze_model_weights(model_path, label):
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return None

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model", ckpt))
    cfg = infer_v5_config_from_state_dict(ckpt)

    total_params = 0
    subsystem_params = {
        "sensory_embedding": [],
        "sensory_layers": [],
        "volumetric_swiglu": [],
        "volumetric_radial": [],
        "volumetric_shortcuts": [],
        "volumetric_gains": [],
        "motor_decoder": [],
        "multimodal_stems": [],
    }

    shell_swiglu_norms = {}
    shell_gains = []

    for name, param in state.items():
        if not isinstance(param, torch.Tensor):
            continue
        p_float = param.float()
        n_elem = p_float.numel()
        total_params += n_elem
        f_norm = float(p_float.norm().item())
        rms = f_norm / math.sqrt(max(1, n_elem))
        mean = float(p_float.mean().item())
        std = float(p_float.std().item()) if n_elem > 1 else 0.0

        stat = {
            "name": name,
            "shape": list(p_float.shape),
            "numel": n_elem,
            "f_norm": f_norm,
            "rms": rms,
            "mean": mean,
            "std": std,
            "min": float(p_float.min().item()),
            "max": float(p_float.max().item()),
        }

        # Subsystem classification (supporting both native packed 'core.*' / 'sensory.*' / 'motor.*'
        # and converted 'volumetric_core.*' / 'sensory_dnn.*' / 'motor_decoder.*')
        
        # 1. Sensory Embedding
        if name in ("sensory.embedding.weight", "sensory_dnn.embedding.weight"):
            subsystem_params["sensory_embedding"].append(stat)

        # 2. Sensory Attention / Layers
        elif name.startswith("sensory.") or name.startswith("sensory_dnn."):
            subsystem_params["sensory_layers"].append(stat)

        # 3. Core SwiGLU Mantle
        elif "intra_swiglu" in name or "intra_shell_swiglu" in name:
            subsystem_params["volumetric_swiglu"].append(stat)
            # Handle native packed 3D tensor: [K, D, 2D] or [K, 2D, D]
            if p_float.ndim == 3 and p_float.shape[0] == cfg.num_shells:
                for s in range(cfg.num_shells):
                    s_norm = float(p_float[s].norm().item())
                    if s not in shell_swiglu_norms:
                        shell_swiglu_norms[s] = []
                    shell_swiglu_norms[s].append(s_norm)
            else:
                # Converted expanded layout
                parts = name.split(".")
                for p in parts:
                    if p.isdigit():
                        s_idx = int(p)
                        if s_idx not in shell_swiglu_norms:
                            shell_swiglu_norms[s_idx] = []
                        shell_swiglu_norms[s_idx].append(f_norm)
                        break

        # 4. Core Radial Waves
        elif "radial" in name or "w_radial" in name:
            subsystem_params["volumetric_radial"].append(stat)

        # 5. Axonal Shortcut Tracts
        elif "shortcut" in name:
            subsystem_params["volumetric_shortcuts"].append(stat)

        # 6. Shell Gains
        elif "shell_gains" in name:
            subsystem_params["volumetric_gains"].append(stat)
            shell_gains = p_float.tolist()

        # 7. Motor Decoder / Readout
        elif name.startswith("motor.") or name.startswith("motor_decoder.") or "head" in name:
            subsystem_params["motor_decoder"].append(stat)

        # 8. Multimodal Stems (Audio / Vision)
        elif "audio_stem" in name or "image_stem" in name or "perceptual" in name:
            subsystem_params["multimodal_stems"].append(stat)

    # Shell isotropy metric
    shell_avg_norms = {}
    for s_idx, norms in shell_swiglu_norms.items():
        shell_avg_norms[s_idx] = sum(norms) / len(norms)
    
    if shell_avg_norms:
        min_sn = min(shell_avg_norms.values())
        max_sn = max(shell_avg_norms.values())
        isotropy_ratio = min_sn / max(1e-6, max_sn)
    else:
        min_sn, max_sn, isotropy_ratio = 0.0, 0.0, 1.0

    step_num = ckpt.get("step", "N/A")
    loss_val = ckpt.get("loss", ckpt.get("eval_loss", ckpt.get("calibration_loss", "N/A")))

    return {
        "label": label,
        "path": model_path,
        "config": cfg,
        "step": step_num,
        "loss": loss_val,
        "total_params": total_params,
        "subsystems": subsystem_params,
        "shell_avg_norms": shell_avg_norms,
        "min_shell_norm": min_sn,
        "max_shell_norm": max_sn,
        "isotropy_ratio": isotropy_ratio,
        "shell_gains": shell_gains,
    }


def print_comparison_report(results):
    print("\n" + "=" * 115)
    print("📊 CEREBROSPHEROID V5 WEIGHT DISTRIBUTION & SPECTRAL TOPOLOGY AUDIT")
    print("=" * 115)

    print(f"\n1. MODEL OVERVIEW & GLOBAL PARAMETER SCALE:")
    print("-" * 115)
    print(f"{'Model Label':<33} | {'Class':<14} | {'K':<3} | {'Dim':<5} | {'Step':<8} | {'Loss':<8} | {'Total Params':<14} | {'Isotropy (Min/Max)'}")
    print("-" * 115)
    for r in results:
        if r is None:
            continue
        cfg = r["config"]
        loss_str = f"{r['loss']:.4f}" if isinstance(r['loss'], (int, float)) else str(r['loss'])
        is_native = "Native" in r["label"]
        class_str = "Native Trained" if is_native else "Converted"
        print(f"{r['label']:<33} | {class_str:<14} | {cfg.num_shells:<3d} | {cfg.surface_dim:<5d} | {str(r['step']):<8} | {loss_str:<8} | {r['total_params']:,} (~{r['total_params']/1e6:.1f}M) | {r['isotropy_ratio']:.3f} ({r['min_shell_norm']:.1f} / {r['max_shell_norm']:.1f})")

    print("\n2. SUBSYSTEM WEIGHT STATISTICS (Mean RMS = ||W||_F / sqrt(numel)):")
    print("-" * 115)
    subsystems = [
        ("sensory_embedding", "Sensory Embedding"),
        ("sensory_layers", "Sensory Transformer"),
        ("volumetric_swiglu", "Core SwiGLU Mantle"),
        ("volumetric_radial", "Radial Advection-Diffusion"),
        ("volumetric_shortcuts", "Axonal Shortcut Tracts"),
        ("motor_decoder", "Premotor Motor Decoder"),
    ]

    for key, title in subsystems:
        print(f"\n👉 [{title}]")
        print(f"{'Model Label':<33} | {'Param Count':<12} | {'Mean Weight':<14} | {'Mean Std':<14} | {'Mean Normalized RMS'}")
        print("-" * 95)
        for r in results:
            if r is None:
                continue
            items = r["subsystems"].get(key, [])
            if not items:
                continue
            cnt = sum(it["numel"] for it in items)
            mean_w = sum(it["mean"] * it["numel"] for it in items) / max(1, cnt)
            mean_std = sum(it["std"] * it["numel"] for it in items) / max(1, cnt)
            mean_rms = sum(it["rms"] * it["numel"] for it in items) / max(1, cnt)
            print(f"{r['label']:<33} | {cnt:<12,} | {mean_w:<+14.6f} | {mean_std:<14.6f} | {mean_rms:.6f}")

    print("\n3. CONCENTRIC SHELL-BY-SHELL SWIGLU FROBENIUS NORM TOPOLOGY:")
    print("-" * 115)
    print("Shows how weight magnitude varies across concentric shells from subcortical core (Shell 0) to cortical surface:")
    for r in results:
        if r is None or not r["shell_avg_norms"]:
            continue
        K = r["config"].num_shells
        norms = [r["shell_avg_norms"].get(k, 0.0) for k in range(K)]
        formatted = " ".join(f"{n:5.1f}" for n in norms)
        print(f"\n[{r['label']}] (K={K}, Isotropy={r['isotropy_ratio']:.3f}):")
        if K <= 16:
            header = " ".join(f" S{k:<2d} " for k in range(K))
            print(f"  Shells : {header}")
            print(f"  Norms  : {formatted}")
        else:
            h1 = " ".join(f" S{k:<2d} " for k in range(12))
            f1 = " ".join(f"{norms[k]:5.1f}" for k in range(12))
            h2 = " ".join(f" S{k:<2d} " for k in range(12, K))
            f2 = " ".join(f"{norms[k]:5.1f}" for k in range(12, K))
            print(f"  S0-11  : {h1}\n           {f1}")
            print(f"  S12-{K-1}: {h2}\n           {f2}")


if __name__ == "__main__":
    checkpoints = [
        ("/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_latest_step900k.pt", "Native V5 Pretrained (900k)"),
        ("/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_sft_best.pt", "Native V5 SFT Best (100)"),
        ("/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_best_step911k.pt", "Native V5 Website Best (911.6k)"),
        ("/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_latest_step920k.pt", "Native V5 Website Latest (920.5k)"),
        ("/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_qwen_0_5b_uwbl_scm.pt", "Converted Qwen (UWBL SCM)"),
        ("/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_qwen_0_5b_converged.pt", "Converted Qwen (Converged)"),
        ("/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_smollm_converged.pt", "Converted SmolLM (Converged)"),
    ]

    all_res = []
    for path, label in checkpoints:
        if os.path.exists(path):
            all_res.append(analyze_model_weights(path, label))
        else:
            print(f"Pending/Missing: {path}")

    print_comparison_report(all_res)
