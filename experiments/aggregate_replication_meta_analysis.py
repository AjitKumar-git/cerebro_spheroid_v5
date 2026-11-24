#!/usr/bin/env python3
"""
📊 CerebroSpheroid V5: Replication & Validation Meta-Analysis
Synthesizes 3 Arms (T0, T1, T3) × 3 Seeds (42, 137, 2026) × 100 Steps
Evaluated on 16 Strictly Held-Out Multi-Domain Sequences.
"""

import os
import sys
import json
import math
import numpy as np
from typing import Dict, List, Tuple

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "replication_checkpoints")

SEEDS = [42, 137, 2026]
ARMS = ["T0", "T1", "T3"]
MODES = ["Frozen", "Periodic", "Adaptive", "Dense"]


def load_metrics():
    train_metrics = {s: {} for s in SEEDS}
    eval_summaries = {}

    for s in SEEDS:
        eval_path = os.path.join(CHECKPOINT_DIR, f"eval_summary_seed{s}.json")
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                eval_summaries[s] = json.load(f)

        for arm in ARMS:
            m_path = os.path.join(CHECKPOINT_DIR, f"pilot_seed{s}_{arm}_metrics.json")
            if os.path.exists(m_path):
                with open(m_path, "r") as f:
                    train_metrics[s][arm] = json.load(f)

    return train_metrics, eval_summaries


def aggregate_meta_analysis():
    train_metrics, eval_summaries = load_metrics()

    completed_seeds = [s for s in SEEDS if s in eval_summaries]
    print(f"\n===============================================================================================")
    print(f"📊 REPLICATION & VALIDATION PILOT: MULTI-SEED META-ANALYSIS ({len(completed_seeds)}/3 SEEDS COMPLETED)")
    print(f"===============================================================================================")

    if not completed_seeds:
        print("No completed seed evaluations found yet.")
        return

    # 1. Training Dynamics Aggregation
    print("\n" + "=" * 95)
    print("📈 1. TRAINING DYNAMICS (100 Steps per Arm per Seed)")
    print("=" * 95)
    print(f"{'Arm':<6} | {'Final Loss (Mean ± Std)':<24} | {'Core |Δθ|':<14} | {'Mid |Δθ|':<14} | {'Outer |Δθ|':<14} | {'Wall Time (s)':<14}")
    print("-" * 95)

    for arm in ARMS:
        losses = []
        d_cores = []
        d_mids = []
        d_outers = []
        wall_times = []

        for s in completed_seeds:
            if arm in train_metrics[s]:
                m = train_metrics[s][arm]
                losses.append(m["final_loss"])
                d_cores.append(m["delta_theta"][0])
                d_mids.append(m["delta_theta"][1])
                d_outers.append(m["delta_theta"][2])
                wall_times.append(m["wall_clock_sec"])

        if losses:
            l_mean, l_std = np.mean(losses), np.std(losses)
            dc_mean, dc_std = np.mean(d_cores), np.std(d_cores)
            dm_mean, dm_std = np.mean(d_mids), np.std(d_mids)
            do_mean, do_std = np.mean(d_outers), np.std(d_outers)
            wt_mean, wt_std = np.mean(wall_times), np.std(wall_times)
            print(f"{arm:<6} | {l_mean:7.4f} ± {l_std:6.4f}             | {dc_mean:10.2e}   | {dm_mean:10.2e}   | {do_mean:10.2e}   | {wt_mean:6.1f}s ± {wt_std:4.1f}s")

    # 2. Held-Out Cross-Evaluation Matrix Grand Means
    print("\n" + "=" * 95)
    print("🎯 2. GRAND MEAN HELD-OUT CROSS-EVALUATION MATRIX (16 Unseen Sequences)")
    print("=" * 95)
    header = f"{'Training Arm':<16} | " + " | ".join([f"{m:<16}" for m in MODES])
    print(header)
    print("-" * 95)

    grand_matrix = {arm: {m: [] for m in MODES} for arm in ARMS}

    for s in completed_seeds:
        matrix = eval_summaries[s]["matrix"]
        for arm in ARMS:
            for m in MODES:
                grand_matrix[arm][m].append(matrix[arm][m])

    for arm in ARMS:
        row_str = f"{arm:<16} | "
        cols = []
        for m in MODES:
            vals = grand_matrix[arm][m]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            cols.append(f"{mean_val:6.4f} ± {std_val:6.4f}")
        row_str += " | ".join(cols)
        print(row_str)
    print("=" * 95)

    # 3. Seed-Paired Bootstrap Aggregation
    print("\n" + "=" * 95)
    print("🔬 3. HIERARCHICAL PAIRED BOOTSTRAP: SEED-PAIRED REPLICATION")
    print("=" * 95)

    t1_t0_diffs = []
    t3_t1_diffs = []

    for s in completed_seeds:
        p_t1_t0 = eval_summaries[s]["paired_t1_t0"]
        p_t3_t1 = eval_summaries[s]["paired_t3_t1"]
        t1_t0_diffs.append(p_t1_t0["mean"])
        t3_t1_diffs.append(p_t3_t1["mean"])
        print(f"Seed {s:4d}:")
        print(f"  • T1 vs T0: ΔNLL = {p_t1_t0['mean']:+.4f} ± {p_t1_t0['se']:.4f} | 95% CI: [{p_t1_t0['ci_95'][0]:+.4f}, {p_t1_t0['ci_95'][1]:+.4f}] | p = {p_t1_t0['p_val']:.4e}")
        print(f"  • T3 vs T1: ΔNLL = {p_t3_t1['mean']:+.4f} ± {p_t3_t1['se']:.4f} | 95% CI: [{p_t3_t1['ci_95'][0]:+.4f}, {p_t3_t1['ci_95'][1]:+.4f}] | p = {p_t3_t1['p_val']:.4e}")

    print("-" * 95)
    print(f"Grand Mean Seed-Paired Differences across {len(completed_seeds)} Seeds:")
    print(f"  ★ T1 (Periodic) vs T0 (Static Control):      ΔNLL = {np.mean(t1_t0_diffs):+.4f} ± {np.std(t1_t0_diffs):.4f}")
    print(f"  ★ T3 (Hierarchical) vs T1 (Periodic Baseline): ΔNLL = {np.mean(t3_t1_diffs):+.4f} ± {np.std(t3_t1_diffs):.4f}")
    print("=" * 95)


if __name__ == "__main__":
    aggregate_meta_analysis()
