#!/usr/bin/env python3
"""
Compute and compare comprehensive Data Density & Information Density metrics
across:
1. Trained Spheroid V5 (v5_small_latest_step900k.pt)
2. Parent Transformer (Qwen/Qwen2.5-0.5B-Instruct)
3. Converted Spheroid V5 (v5_qwen_0_5b_converged.pt)
4. Refined SFT Spheroid V5 (v5_qwen_0_5b_refined_sft.pt)
"""

import os
import math
import json
import torch
import numpy as np

# Set custom HF cache directory
os.environ["HF_HOME"] = "/home/ajit/.hf_cache"

def compute_matrix_spectral_metrics(W_tensor):
    """
    Given a 2D weight matrix (M, N) as a PyTorch tensor:
    Computes:
      - Singular values via SVD
      - Frobenius norm, Spectral norm (max singular value)
      - Effective rank: r_eff = exp(-sum p_i * ln p_i) where p_i = s_i / sum s
      - Stable rank: ||W||_F^2 / s_max^2
      - Rank utilization: r_eff / min(M, N)
      - Inactive / Near-zero weight fractions (|W| < 1e-3, |W| < 1e-4)
      - Weight Shannon entropy (in 256 uniform bins over [-3*sigma, +3*sigma])
    """
    W = W_tensor.float().detach().cpu()
    M, N = W.shape
    min_dim = min(M, N)
    if min_dim < 2:
        return None
    
    # Weight statistics
    w_flat = W.flatten()
    std = torch.std(w_flat).item()
    mean = torch.mean(w_flat).item()
    frob = torch.norm(W, p='fro').item()
    sparsity_1e3 = (torch.abs(w_flat) < 1e-3).float().mean().item()
    sparsity_1e4 = (torch.abs(w_flat) < 1e-4).float().mean().item()
    
    # Weight entropy (Bits per parameter across 256 bins)
    if std > 1e-9:
        hist = torch.histc(w_flat, bins=256, min=mean - 3.5*std, max=mean + 3.5*std)
        probs = hist / (hist.sum() + 1e-12)
        probs = probs[probs > 0]
        weight_entropy = -torch.sum(probs * torch.log2(probs)).item()
    else:
        weight_entropy = 0.0

    # SVD
    # For large matrices, take up to 2048 x 2048 or compute SVD
    try:
        # Perform SVD
        # torch.linalg.svdvals is faster and memory efficient
        s = torch.linalg.svdvals(W)
        s_max = s[0].item()
        s_sum = s.sum().item()
        s_norm = s / (s_sum + 1e-12)
        
        # Spectral entropy and effective rank
        nz = s_norm[s_norm > 1e-12]
        spectral_entropy = -torch.sum(nz * torch.log(nz)).item()
        effective_rank = math.exp(spectral_entropy)
        
        # Stable rank
        stable_rank = (frob ** 2) / (s_max ** 2 + 1e-12)
        
        # Rank utilization
        rank_utilization = effective_rank / min_dim
        
        return {
            "M": M,
            "N": N,
            "min_dim": min_dim,
            "numel": M * N,
            "frob_norm": frob,
            "spectral_norm": s_max,
            "effective_rank": effective_rank,
            "stable_rank": stable_rank,
            "rank_utilization": rank_utilization,
            "weight_entropy": weight_entropy,
            "sparsity_1e3": sparsity_1e3,
            "sparsity_1e4": sparsity_1e4,
            "std": std
        }
    except Exception as e:
        print(f"Error computing SVD for shape {M}x{N}: {e}")
        return None

def analyze_model_weights(name, weight_dict_or_model, is_state_dict=True):
    """
    Extracts all 2D weight matrices (or 3D batched matrices like shortcut/radial weights)
    and aggregates information density metrics.
    """
    print(f"\n=======================================================")
    print(f"Analyzing {name}...")
    print(f"=======================================================")
    
    total_params = 0
    embedding_params = 0
    core_params = 0
    
    matrix_metrics = []
    
    if is_state_dict:
        state_dict = weight_dict_or_model
        items = state_dict.items()
    else:
        state_dict = {k: v for k, v in weight_dict_or_model.named_parameters()}
        items = state_dict.items()
        
    for k, v in items:
        if not hasattr(v, 'shape'):
            continue
        numel = v.numel()
        total_params += numel
        
        k_lower = k.lower()
        if 'embed' in k_lower or 'wte' in k_lower or 'lm_head' in k_lower:
            embedding_params += numel
        else:
            core_params += numel
            
        shape = list(v.shape)
        # Check if 2D matrix
        if len(shape) == 2:
            # Skip tiny biases or 1xN vectors
            if shape[0] >= 16 and shape[1] >= 16:
                m = compute_matrix_spectral_metrics(v)
                if m is not None:
                    m['name'] = k
                    matrix_metrics.append(m)
        elif len(shape) == 3:
            # Batched matrices e.g. [shells, D, D] or [tracts, D, D]
            # Analyze each slice or sample up to 10 slices
            B, M, N = shape
            if M >= 16 and N >= 16:
                sample_indices = np.linspace(0, B - 1, min(B, 8), dtype=int)
                for idx in sample_indices:
                    slice_tensor = v[idx]
                    m = compute_matrix_spectral_metrics(slice_tensor)
                    if m is not None:
                        m['name'] = f"{k}[{idx}]"
                        matrix_metrics.append(m)
                        
    # Aggregated Metrics
    num_mats = len(matrix_metrics)
    if num_mats == 0:
        return {}
        
    avg_eff_rank = np.mean([m['effective_rank'] for m in matrix_metrics])
    avg_stable_rank = np.mean([m['stable_rank'] for m in matrix_metrics])
    avg_rank_utilization = np.mean([m['rank_utilization'] for m in matrix_metrics])
    avg_weight_entropy = np.mean([m['weight_entropy'] for m in matrix_metrics])
    avg_sparsity_1e3 = np.mean([m['sparsity_1e3'] for m in matrix_metrics])
    avg_sparsity_1e4 = np.mean([m['sparsity_1e4'] for m in matrix_metrics])
    avg_frob = np.mean([m['frob_norm'] for m in matrix_metrics])
    
    total_effective_rank_volume = sum(m['effective_rank'] for m in matrix_metrics)
    
    # Information Density Ratio:
    # 1. Rank per Million Parameters (how much geometric manifold capacity per param)
    rank_per_million_params = (total_effective_rank_volume / (total_params / 1e6))
    
    # 2. Core Information Density: Effective Rank Volume / Core Params (M)
    core_rank_per_million = (total_effective_rank_volume / (max(1, core_params) / 1e6))
    
    # 3. Active Parameter Ratio:
    # Transformers activate ~100% of weights on every token pass.
    # Spheroid V5 has concentric shells + localized shortcuts.
    
    results = {
        "name": name,
        "total_params": total_params,
        "embedding_params": embedding_params,
        "core_params": core_params,
        "embedding_ratio": embedding_params / total_params,
        "num_analyzed_matrices": num_mats,
        "avg_effective_rank": avg_eff_rank,
        "avg_stable_rank": avg_stable_rank,
        "avg_rank_utilization": avg_rank_utilization,
        "avg_weight_entropy_bits": avg_weight_entropy,
        "avg_sparsity_1e3": avg_sparsity_1e3,
        "avg_sparsity_1e4": avg_sparsity_1e4,
        "total_effective_rank_volume": total_effective_rank_volume,
        "rank_per_million_params": rank_per_million_params,
        "core_rank_per_million": core_rank_per_million
    }
    
    print(f"Total Params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Embedding / Head Params: {embedding_params:,} ({embedding_params/total_params*100:.1f}%)")
    print(f"  Core Latent Params: {core_params:,} ({core_params/total_params*100:.1f}%)")
    print(f"Analyzed Matrices: {num_mats}")
    print(f"  Avg Effective Rank: {avg_eff_rank:.2f}")
    print(f"  Avg Stable Rank: {avg_stable_rank:.2f}")
    print(f"  Avg Rank Utilization: {avg_rank_utilization*100:.2f}%")
    print(f"  Avg Weight Entropy: {avg_weight_entropy:.2f} bits/param (max 8.0)")
    print(f"  Inactive / Near-Zero Weights (|w|<1e-3): {avg_sparsity_1e3*100:.2f}%")
    print(f"  Total Effective Rank Volume: {total_effective_rank_volume:.1f}")
    print(f"  Information Density (Rank / M-Params): {rank_per_million_params:.2f}")
    print(f"  Core Information Density (Rank / M-Core-Params): {core_rank_per_million:.2f}")
    
    return results

def main():
    print("Loading checkpoints and models for comparative density evaluation...")
    
    summary = {}
    
    # 1. Trained Spheroid V5 Small
    p1 = '/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_latest_step900k.pt'
    if os.path.exists(p1):
        d1 = torch.load(p1, map_location='cpu', weights_only=False)
        summary['trained_spheroid_v5'] = analyze_model_weights(
            "Trained Spheroid V5 Small (Step 900k)",
            d1['model_state'],
            is_state_dict=True
        )
        
    # 2. Parent Transformer: Qwen2.5-0.5B-Instruct
    try:
        from transformers import AutoModelForCausalLM
        print("\nLoading parent model Qwen/Qwen2.5-0.5B-Instruct...")
        qwen = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        summary['parent_qwen_0_5b'] = analyze_model_weights(
            "Parent Transformer (Qwen2.5-0.5B-Instruct)",
            qwen,
            is_state_dict=False
        )
        del qwen
    except Exception as e:
        print(f"Could not load Qwen: {e}")
        
    # 3. Converted Qwen Spheroid V5 (Step 262 Converged)
    p3 = '/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_qwen_0_5b_converged.pt'
    if os.path.exists(p3):
        d3 = torch.load(p3, map_location='cpu', weights_only=False)
        summary['converted_qwen_spheroid'] = analyze_model_weights(
            "Converted Model (Qwen -> Spheroid V5, Step 262)",
            d3['model_state'],
            is_state_dict=True
        )

    # 4. Refined SFT Spheroid V5 (Optional comparison)
    p4 = '/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_qwen_0_5b_refined_sft.pt'
    if os.path.exists(p4):
        d4 = torch.load(p4, map_location='cpu', weights_only=False)
        summary['refined_sft_spheroid'] = analyze_model_weights(
            "Refined SFT Model (Qwen -> Spheroid V5 SFT)",
            d4['model_state'],
            is_state_dict=True
        )

    # Save output to JSON
    out_path = '/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/metrics/data_density_comparison.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full quantitative metrics to {out_path}")

if __name__ == '__main__':
    main()
