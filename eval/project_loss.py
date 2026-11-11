#!/usr/bin/env python3
"""
Empirical Loss Trajectory & Horizon Projection for CerebroSpheroid V5.
=====================================================================
Fits Chinchilla / Kaplan-style power-law and log-linear dynamics to the
empirical training run on Cloud TPU v5e-1 to project when the model will
break:
  • Loss < 5.0
  • Loss < 4.0
  • Loss < 3.0 (Primary Milestone)
  • Loss < 2.5 (High Perplexity Milestone)
"""

import sys
import numpy as np

def run_loss_projection(empirical_points, current_step, step_time_sec=1.5, tokens_per_step=98304):
    print("=" * 80)
    print("📈 CEREBROSPHEROID V5: LEARNING TRAJECTORY & LOSS HORIZON PROJECTION")
    print("=" * 80)
    print(f"Hardware Engine     : Cloud TPU v5e-1 (Pure JAX Native BF16 MXU)")
    print(f"Batch Size          : 384 sequences (SeqLen: 256) = {tokens_per_step:,} tokens/step")
    print(f"Step Velocity       : ~{step_time_sec:.2f}s per step (~{tokens_per_step/step_time_sec:,.0f} tokens/second)")
    print(f"Current Model Step  : Step {current_step:,} ({current_step * tokens_per_step / 1e6:.2f}M tokens consumed)")
    print("-" * 80)
    print("Recorded Empirical Convergence Points:")
    print("  Step       Loss       Phase / Description")
    print("  -------------------------------------------------------------")
    for s, l, desc in empirical_points:
        print(f"  {s:5d}      {l:6.4f}     {desc}")
    print("-" * 80)

    # Filter post-warmup points (where learning rate is stable and descent follows power law)
    post_warmup = [(s, l) for s, l, desc in empirical_points if s >= 500]
    
    # Fit power law: (L - L_inf) = A * s^(-gamma)
    # L_inf for 32k vocab on dense educational text is empirically ~1.8 - 2.2
    L_inf = 2.0
    steps = np.array([p[0] for p in post_warmup], dtype=np.float64)
    losses = np.array([p[1] for p in post_warmup], dtype=np.float64)
    
    # Log-log linear regression on y = ln(L - L_inf), x = ln(s)
    y = np.log(losses - L_inf)
    x = np.log(steps)
    
    slope, intercept = np.polyfit(x, y, 1)
    gamma = -slope
    A = np.exp(intercept)
    
    print(f"Model Power-Law Parameters:")
    print(f"  • Assumed Irreducible Loss Floor (L_inf) : {L_inf:.2f}")
    print(f"  • Power-Law Decay Exponent (gamma)       : {gamma:.4f}")
    print(f"  • Scaling Coefficient (A)                : {A:.2f}")
    print(f"  • Dynamic Equation                       : L(s) = {L_inf:.2f} + {A:.2f} · s^(-{gamma:.4f})")
    print("-" * 80)

    # Forecast target milestones
    targets = [5.0, 4.0, 3.5, 3.0, 2.5]
    print(f"{'Target Loss':<12} | {'Projected Step':<15} | {'Tokens Needed':<16} | {'Remaining Time':<16} | {'Total Time'}")
    print("-" * 80)
    
    for t_loss in targets:
        if t_loss <= L_inf:
            print(f"{t_loss:<12.1f} | {'Asymptotic Floor':<15} | {'N/A':<16} | {'N/A':<16} | N/A")
            continue
            
        # s_target = ( (t_loss - L_inf) / A )^(-1/gamma)
        s_target = int(round(((t_loss - L_inf) / A) ** (-1.0 / gamma)))
        rem_steps = max(0, s_target - current_step)
        rem_tokens = rem_steps * tokens_per_step
        rem_sec = rem_steps * step_time_sec
        tot_sec = s_target * step_time_sec
        
        rem_hours = rem_sec / 3600.0
        tot_hours = tot_sec / 3600.0
        
        rem_str = f"{rem_hours:.1f} hrs" if rem_hours >= 1.0 else f"{rem_sec/60:.0f} mins"
        tot_str = f"{tot_hours:.1f} hrs" if tot_hours >= 1.0 else f"{tot_sec/60:.0f} mins"
        
        status = "PASSED ✓" if current_step >= s_target else f"in {rem_str}"
        print(f"Loss < {t_loss:<6.1f} | Step {s_target:<10,d} | {rem_tokens/1e6:6.1f}M tokens  | {status:<16} | {tot_str}")
        
    print("=" * 80)

if __name__ == "__main__":
    empirical_pts = [
        (0, 10.4950, "Random Orthogonal Initialization (ln(32000))"),
        (40, 10.4528, "Early warmup (LR = 4e-5)"),
        (300, 8.7242, "Mid warmup (LR = 3e-4)"),
        (500, 7.5000, "Warmup completion (LR = 5e-4 peak)"),
        (750, 5.9623, "Active NS-5 Newton-Schulz Polar descent"),
        (900, 5.6742, "Active NS-5 Newton-Schulz Polar descent"),
    ]
    cur_s = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    run_loss_projection(empirical_pts, cur_s)
