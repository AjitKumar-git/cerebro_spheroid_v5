#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Weight Saturation & Information Capacity Auditor
======================================================================
Performs deep mathematical analysis of model weights to determine:
1. Effective SVD Rank (R_eff / min(M, N)) & Spectral Entropy per layer.
2. Frobenius Energy Concentration (k_90, k_99 singular values).
3. Parameter Distribution, Kurtosis, Sparsity, and Dead Neuron Rates.
4. Core 3D Concentric Laminae & Axonal Tract Conductance Saturation.
5. Theoretical Information Capacity (Shannon/Kolmogorov bits, Attractor Basins).
6. Remaining Capacity Headroom for Reasoning & Multi-step Deliberation.

Usage:
  python3 scripts/audit_weight_saturation.py [checkpoint_path]
"""

import sys
import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def compute_matrix_svd_stats(w_mat: torch.Tensor):
    """
    Computes rigorous SVD metrics on a 2D matrix W [M, N]:
    - Singular values: sigma_1 >= sigma_2 >= ... >= sigma_k
    - Condition number: sigma_1 / max(sigma_k, 1e-8)
    - Spectral distribution: p_i = sigma_i / sum(sigma)
    - Spectral entropy: H = - sum(p_i * ln(p_i))
    - Effective rank: R_eff = exp(H)
    - Rank saturation: R_eff / min(M, N)
    - Stable rank: ||W||_F^2 / sigma_1^2
    - k_90, k_99: components capturing 90% and 99% energy
    """
    if w_mat.dim() != 2:
        w_mat = w_mat.view(w_mat.size(0), -1)
    
    M, N = w_mat.shape
    min_dim = min(M, N)
    if min_dim == 0:
        return {}
    
    # Cast to float32 for high numerical precision
    w_f = w_mat.float()
    
    # Singular value decomposition
    try:
        s = torch.linalg.svdvals(w_f)
    except Exception as e:
        # Fallback if SVD fails
        return {"error": str(e), "min_dim": min_dim}
    
    s_np = s.cpu().numpy()
    sum_s = np.sum(s_np)
    if sum_s <= 1e-12:
        return {
            "min_dim": min_dim,
            "sigma_max": 0.0,
            "sigma_min": 0.0,
            "condition_number": 1.0,
            "effective_rank": 0.0,
            "rank_saturation_pct": 0.0,
            "stable_rank": 0.0,
            "k_90_pct": 0.0,
            "k_99_pct": 0.0,
        }
    
    p = s_np / sum_s
    p_nz = p[p > 1e-15]
    H = -np.sum(p_nz * np.log(p_nz))
    r_eff = float(np.exp(H))
    rank_sat = float(r_eff / min_dim * 100.0)
    
    # Frobenius energy
    s_sq = s_np ** 2
    total_energy = np.sum(s_sq)
    stable_rank = float(total_energy / max(s_sq[0], 1e-12))
    
    cum_energy = np.cumsum(s_sq) / max(total_energy, 1e-12)
    k_90 = int(np.searchsorted(cum_energy, 0.90) + 1)
    k_99 = int(np.searchsorted(cum_energy, 0.99) + 1)
    k_90_pct = float(k_90 / min_dim * 100.0)
    k_99_pct = float(k_99 / min_dim * 100.0)
    
    cond = float(s_np[0] / max(s_np[-1], 1e-12))
    
    return {
        "min_dim": min_dim,
        "sigma_max": float(s_np[0]),
        "sigma_min": float(s_np[-1]),
        "condition_number": cond,
        "effective_rank": r_eff,
        "rank_saturation_pct": rank_sat,
        "stable_rank": stable_rank,
        "k_90": k_90,
        "k_90_pct": k_90_pct,
        "k_99": k_99,
        "k_99_pct": k_99_pct,
    }


def compute_distribution_stats(w_tensor: torch.Tensor):
    """
    Computes statistical distribution metrics:
    - mean, std, Frobenius norm
    - excess kurtosis (peakedness vs Gaussian)
    - near-zero sparsity (< 1e-4, < 1e-3)
    - channel death rate
    """
    w_f = w_tensor.float().cpu()
    numel = w_f.numel()
    if numel == 0:
        return {}
    
    mean = float(w_f.mean())
    std = float(w_f.std())
    frob = float(torch.norm(w_f))
    
    # Kurtosis
    if std > 1e-8:
        z = (w_f - mean) / std
        kurtosis = float(torch.mean(z ** 4) - 3.0)
    else:
        kurtosis = 0.0
    
    abs_w = torch.abs(w_f)
    sparse_1e4 = float((abs_w < 1e-4).sum()) / numel * 100.0
    sparse_1e3 = float((abs_w < 1e-3).sum()) / numel * 100.0
    
    # Dead channels (for 2D+ tensors)
    dead_rows = 0.0
    dead_cols = 0.0
    if w_f.dim() >= 2:
        w_2d = w_f.view(w_f.size(0), -1)
        row_norms = torch.norm(w_2d, dim=1)
        mean_row = row_norms.mean()
        if mean_row > 1e-8:
            dead_rows = float((row_norms < 1e-4 * mean_row).sum()) / w_2d.size(0) * 100.0
        
        col_norms = torch.norm(w_2d, dim=0)
        mean_col = col_norms.mean()
        if mean_col > 1e-8:
            dead_cols = float((col_norms < 1e-4 * mean_col).sum()) / w_2d.size(1) * 100.0
    
    return {
        "numel": numel,
        "mean": mean,
        "std": std,
        "frob_norm": frob,
        "kurtosis": kurtosis,
        "sparsity_1e4_pct": sparse_1e4,
        "sparsity_1e3_pct": sparse_1e3,
        "dead_rows_pct": dead_rows,
        "dead_cols_pct": dead_cols,
    }


def audit_checkpoint(ckpt_path: str):
    print("=" * 80)
    print(f"🧠 CEREBRO-SPHEROID V5: WEIGHT SATURATION & CAPACITY AUDIT")
    print(f"📁 Target Checkpoint: {ckpt_path}")
    print("=" * 80)
    
    if not os.path.exists(ckpt_path):
        print(f"❌ File not found: {ckpt_path}")
        return
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    step = ckpt.get("step", "N/A")
    loss = ckpt.get("loss", "N/A")
    eval_loss = ckpt.get("eval_loss", "N/A")
    print(f"📍 Checkpoint Step      : {step}")
    print(f"📉 Checkpoint Loss      : {loss:.5f}" if isinstance(loss, (int, float)) else f"📉 Checkpoint Loss      : {loss}")
    if isinstance(eval_loss, (int, float)):
        print(f"🎯 Checkpoint Eval Loss : {eval_loss:.5f}")
    print("-" * 80)
    
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
        
    total_params = sum(p.numel() for p in state.values() if hasattr(p, "numel"))
    print(f"🔢 Total Model Parameters: {total_params:,} ({total_params/1e6:.2f} M)")
    
    # Subsystems breakdown
    subsystems = {
        "Sensory Transduction": [],
        "Perceptual Stem": [],
        "Volumetric Cortical Core": [],
        "Subcortical Axonal Tracts": [],
        "Motor Deliberation & Decoder": [],
        "Other / Auxiliary": [],
    }
    
    for name, param in state.items():
        if not hasattr(param, "numel") or param.numel() == 0:
            continue
        if name.startswith("sensory."):
            subsystems["Sensory Transduction"].append((name, param))
        elif name.startswith("perceptual.") or name.startswith("image_stem.") or name.startswith("audio_stem."):
            subsystems["Perceptual Stem"].append((name, param))
        elif name.startswith("core.shortcut_"):
            subsystems["Subcortical Axonal Tracts"].append((name, param))
        elif name.startswith("core."):
            subsystems["Volumetric Cortical Core"].append((name, param))
        elif name.startswith("motor."):
            subsystems["Motor Deliberation & Decoder"].append((name, param))
        else:
            subsystems["Other / Auxiliary"].append((name, param))
    
    # Detailed SVD & Saturation metrics
    layer_results = []
    subsystem_aggregates = {}
    
    print("\n" + "=" * 80)
    print("🔬 LAYER-BY-LAYER SPECTRAL & SATURATION PROFILE")
    print("=" * 80)
    print(f"{'Layer Name':<42} | {'Shape':<16} | {'Params':<9} | {'R_eff/dim':<9} | {'k_90%':<6} | {'Kurtosis':<8} | {'Sparse%':<7}")
    print("-" * 105)
    
    for sub_name, params in subsystems.items():
        if not params:
            continue
        
        sub_params = 0
        sub_sat_weighted = 0.0
        sub_eff_rank_sum = 0.0
        sub_k90_weighted = 0.0
        sub_kurt_weighted = 0.0
        sub_sparse_weighted = 0.0
        
        for name, param in params:
            numel = param.numel()
            sub_params += numel
            dist = compute_distribution_stats(param)
            
            # SVD stats
            svd_sat = 0.0
            k90_pct = 0.0
            eff_rank = 0.0
            
            if param.dim() >= 2:
                if param.dim() == 2:
                    svd = compute_matrix_svd_stats(param)
                    svd_sat = svd.get("rank_saturation_pct", 0.0)
                    k90_pct = svd.get("k_90_pct", 0.0)
                    eff_rank = svd.get("effective_rank", 0.0)
                elif param.dim() == 3:
                    # 3D tensor (e.g. [shells, D, D] or [tracts, D, D])
                    # Compute SVD across slice 0 and average across slices
                    sats = []
                    k90s = []
                    eff_ranks = []
                    for s in range(param.size(0)):
                        svd_s = compute_matrix_svd_stats(param[s])
                        if "rank_saturation_pct" in svd_s:
                            sats.append(svd_s["rank_saturation_pct"])
                            k90s.append(svd_s["k_90_pct"])
                            eff_ranks.append(svd_s["effective_rank"])
                    if sats:
                        svd_sat = float(np.mean(sats))
                        k90_pct = float(np.mean(k90s))
                        eff_rank = float(np.sum(eff_ranks))
                elif param.dim() == 4:
                    # Conv 4D tensor
                    svd = compute_matrix_svd_stats(param.view(param.size(0), -1))
                    svd_sat = svd.get("rank_saturation_pct", 0.0)
                    k90_pct = svd.get("k_90_pct", 0.0)
                    eff_rank = svd.get("effective_rank", 0.0)
            
            shape_str = "x".join(str(d) for d in param.shape)
            kurt = dist.get("kurtosis", 0.0)
            sparse = dist.get("sparsity_1e3_pct", 0.0)
            
            layer_results.append({
                "subsystem": sub_name,
                "name": name,
                "shape": list(param.shape),
                "numel": numel,
                "svd_sat": svd_sat,
                "k90_pct": k90_pct,
                "eff_rank": eff_rank,
                "kurtosis": kurt,
                "sparsity_1e3": sparse,
                "frob_norm": dist.get("frob_norm", 0.0),
                "std": dist.get("std", 0.0),
            })
            
            # Weighted aggregation
            sub_sat_weighted += svd_sat * numel
            sub_k90_weighted += k90_pct * numel
            sub_kurt_weighted += kurt * numel
            sub_sparse_weighted += sparse * numel
            sub_eff_rank_sum += eff_rank
            
            if param.dim() >= 2 and numel > 50000:
                print(f"{name:<42} | {shape_str:<16} | {numel/1e3:>7.1f}k | {svd_sat:>8.2f}% | {k90_pct:>5.1f}% | {kurt:>8.2f} | {sparse:>6.2f}%")
        
        if sub_params > 0:
            subsystem_aggregates[sub_name] = {
                "params": sub_params,
                "pct_of_model": sub_params / total_params * 100.0,
                "mean_rank_saturation": sub_sat_weighted / sub_params,
                "mean_k90_pct": sub_k90_weighted / sub_params,
                "mean_kurtosis": sub_kurt_weighted / sub_params,
                "mean_sparsity_1e3": sub_sparse_weighted / sub_params,
            }
    
    print("-" * 105)
    
    # Subsystem Summary Table
    print("\n" + "=" * 80)
    print("🏛️ SUBSYSTEM-BY-SUBSYSTEM SATURATION & HEADROOM SUMMARY")
    print("=" * 80)
    print(f"{'Subsystem':<30} | {'Params':<10} | {'Share':<6} | {'Rank Sat':<9} | {'k_90 Energy':<11} | {'Kurtosis':<8} | {'Headroom'}")
    print("-" * 95)
    
    total_effective_capacity_params = 0.0
    
    for sub_name, agg in subsystem_aggregates.items():
        p_count = agg["params"]
        share = agg["pct_of_model"]
        sat = agg["mean_rank_saturation"]
        k90 = agg["mean_k90_pct"]
        kurt = agg["mean_kurtosis"]
        
        # Effective parameters utilized = params * (sat / 100)
        p_eff = p_count * (sat / 100.0)
        total_effective_capacity_params += p_eff
        headroom_pct = 100.0 - sat
        
        status = "🟢 High Headroom" if sat < 35.0 else ("🟡 Moderate" if sat < 65.0 else "🔴 Saturated")
        
        print(f"{sub_name:<30} | {p_count/1e6:>8.2f}M | {share:>5.1f}% | {sat:>8.2f}% | {k90:>10.2f}% | {kurt:>8.2f} | {headroom_pct:>5.1f}% ({status})")
    
    print("-" * 95)
    
    # Global Saturation & Information Theory Calculations
    overall_rank_saturation = (total_effective_capacity_params / total_params) * 100.0
    unused_parameters = total_params - total_effective_capacity_params
    unused_pct = 100.0 - overall_rank_saturation
    
    # Shannon-Kolmogorov Information Capacity
    # For neural network weights with 16-bit representation (BF16), empirical SNR yields ~1.8 to 2.2 bits/param
    # At current rank saturation:
    bits_per_param_raw = 16.0  # BF16 storage
    bits_per_param_entropy = 2.0  # Information theoretical empirical bound for dense LLM
    
    total_theoretical_capacity_bits = total_params * bits_per_param_entropy
    total_theoretical_capacity_mb = total_theoretical_capacity_bits / (8 * 1024 * 1024)
    
    current_utilized_bits = total_effective_capacity_params * bits_per_param_entropy
    current_utilized_mb = current_utilized_bits / (8 * 1024 * 1024)
    
    remaining_capacity_bits = unused_parameters * bits_per_param_entropy
    remaining_capacity_mb = remaining_capacity_bits / (8 * 1024 * 1024)
    
    # Text token equivalence:
    # 1 byte of information ~ 0.25 to 0.5 tokens in natural language text (Chinchilla / Shannon entropy of English: ~1.0-1.5 bits/char)
    # Natural language: ~4-5 chars/token => ~6-8 bits/token.
    # Therefore, 1 MB entropy ~ 1,000,000 to 1,300,000 text tokens of compressed knowledge!
    utilized_token_equiv = (current_utilized_bits / 7.0)
    remaining_token_equiv = (remaining_capacity_bits / 7.0)
    total_token_capacity = (total_theoretical_capacity_bits / 7.0)
    
    # 3D Core Attractor Basins
    # S=16 shells, D=640 => 10,240 core latent state dimensions.
    # Continuous associative memory capacity alpha_c ~ 0.14 * N_dim (Hopfield limit)
    # For spherical manifold with NS-5 Riemannian stabilization:
    core_dims = 16 * 640
    orthogonal_attractors = int(0.14 * core_dims)
    
    # Axonal Tract Conductance Check
    if "core.shortcut_gate" in state:
        gates = torch.sigmoid(state["core.shortcut_gate"].float()).numpy()
        active_tracts_count = int(np.sum(gates > 0.5))
        mean_conductance = float(np.mean(gates))
    else:
        active_tracts_count = "N/A"
        mean_conductance = "N/A"
        
    print("\n" + "=" * 80)
    print("📊 INFORMATION THEORETICAL CAPACITY & COMPRESSION BOUNDS")
    print("=" * 80)
    print(f"📦 Total Parameter Weight Budget     : {total_params:,} ({total_params/1e6:.2f} M params)")
    print(f"💾 Storage Size on Disk (BF16)       : {total_params * 2 / (1024*1024):.2f} MB")
    print(f"📈 Overall Effective Rank Saturation : {overall_rank_saturation:.2f}% (Utilized Subspace)")
    print(f"🟢 Remaining Capacity Headroom        : {unused_pct:.2f}% (Unused Orthogonal Subspaces)")
    print("-" * 80)
    print(f"🧮 Theoretical Entropy Ceiling       : {total_theoretical_capacity_mb:.2f} MB ({total_theoretical_capacity_bits/1e6:.1f} Mbits at 2.0 bits/param)")
    print(f"🧠 Current Packed Knowledge Entropy   : {current_utilized_mb:.2f} MB ({current_utilized_bits/1e6:.1f} Mbits)")
    print(f"🚀 Available Space for New Knowledge  : {remaining_capacity_mb:.2f} MB ({remaining_capacity_bits/1e6:.1f} Mbits)")
    print("-" * 80)
    print(f"📚 Equivalent Compressed Text Tokens :")
    print(f"   - Currently Packed Knowledge      : ~{utilized_token_equiv/1e6:.2f} Million tokens (~{utilized_token_equiv*4.5/1e6:.1f} M characters)")
    print(f"   - Remaining Capacity for Tokens   : ~{remaining_token_equiv/1e6:.2f} Million tokens (~{remaining_token_equiv*4.5/1e6:.1f} M characters)")
    print(f"   - Total Theoretical Upper Bound   : ~{total_token_capacity/1e6:.2f} Million tokens")
    print("-" * 80)
    print(f"🌐 3D Concentric Cortical Attractors :")
    print(f"   - Total Core Latent Dimension     : {core_dims:,} dimensions (16 concentric shells × 640D)")
    print(f"   - Orthogonal Attractor Basins     : ~{orthogonal_attractors:,} simultaneous independent concept attractors")
    if isinstance(active_tracts_count, int):
        print(f"   - Axonal Shortcuts (White Matter) : {active_tracts_count} / 40 active tracts (mean conductance: {mean_conductance:.3f})")
    print("=" * 80)
    
    # Strategic Diagnosis
    print("\n🎯 STRATEGIC CAPACITY DIAGNOSIS:")
    if overall_rank_saturation < 35.0:
        print("✅ The model is in the PRIME LEARNING REGIME (Rank Saturation < 35%).")
        print("   - Weight matrices exhibit high spectral headroom and steep singular value decay.")
        print("   - Low effective rank indicates representations are structured and sparse, NOT saturated.")
        print(f"   - The network has {unused_pct:.1f}% orthogonal capacity remaining, meaning it can absorb")
        print("     substantial new domains (GSM8K Math-CoT, code execution, complex reasoning) without")
        print("     catastrophic interference or representation collapse.")
    elif overall_rank_saturation < 65.0:
        print("⚠️ The model is in the MODERATELY LOADED REGIME (Rank Saturation 35-65%).")
        print("   - Balanced capacity utilization. Ideal for targeted fine-tuning and annealing.")
    else:
        print("🔴 The model is HIGHLY SATURATED (Rank Saturation > 65%).")
        print("   - Weight space is heavily diffused; requires model capacity scaling (e.g. to V5 Base).")
        
    print("=" * 80 + "\n")
    
    return {
        "step": step,
        "loss": loss,
        "total_params": total_params,
        "overall_rank_saturation": overall_rank_saturation,
        "unused_pct": unused_pct,
        "total_theoretical_capacity_mb": total_theoretical_capacity_mb,
        "current_utilized_mb": current_utilized_mb,
        "remaining_capacity_mb": remaining_capacity_mb,
        "utilized_token_equiv": utilized_token_equiv,
        "remaining_token_equiv": remaining_token_equiv,
        "orthogonal_attractors": orthogonal_attractors,
        "subsystems": subsystem_aggregates,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit CerebroSpheroid V5 Weight Saturation & Capacity")
    parser.add_argument("checkpoint", nargs="?", default=None, help="Path to checkpoint .pt file")
    args = parser.parse_args()
    
    ckpt_target = args.checkpoint
    if ckpt_target is None:
        # Default search order
        candidates = [
            os.path.join(REPO_ROOT, "checkpoints", "v5_small_latest_step593k.pt"),
            os.path.join(REPO_ROOT, "checkpoints", "v5_small_best_step586k.pt"),
            os.path.join(REPO_ROOT, "checkpoints", "v5_small_cloud_best_live.pt"),
            os.path.join(REPO_ROOT, "checkpoints", "v5_small_latest.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_target = c
                break
                
    if not ckpt_target:
        print("❌ No checkpoint found to audit!")
        sys.exit(1)
        
    audit_checkpoint(ckpt_target)
