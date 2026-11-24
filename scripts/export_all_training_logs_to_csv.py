#!/usr/bin/env python3
"""
Exports full CerebroSpheroid V5 Cloud TPU pretraining logs (from download.txt)
and all empirical experimental benchmark results to standardized CSV files.

Output CSV files generated in metrics/:
  1. metrics/cerebro_v5_training_steps.csv: Full 1,082 step-level training log (loss, lr, throughput, memory, optimizer diag)
  2. metrics/cerebro_v5_evaluations.csv: Full 692 evaluation checkpoints (composite loss/PPL, FineWeb loss/PPL, The Stack loss/PPL)
  3. metrics/cerebro_v5_unified_training_and_eval.csv: Unified master timeline aligned by training step
  4. metrics/cerebro_v5_audit_shell_gradients.csv: Shell-by-shell gradient norms, SNR, state sensitivity, relative norms
  5. metrics/cerebro_v5_audit_optimizer_dynamics.csv: Shell-by-shell Polar Newton-Schulz update step, relative updates, alignment
  6. metrics/cerebro_v5_audit_temporal_displacement.csv: Autoregressive state displacement across emission steps
  7. metrics/cerebro_v5_audit_radial_transmission.csv: Inter-shell Jacobian transmission matrix across r_mix sweep
  8. metrics/cerebro_v5_condition_d_benchmark.csv: Condition D pilot/benchmark results across arms and seeds
  9. metrics/cerebro_v5_frozen_readout_rescue.csv: Frozen readout rescue sweep (gain beta, core exclusion, loss delta)
"""

import os
import re
import csv
import json
from typing import Dict, Any, List

REPO_ROOT = "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5"
DOWNLOAD_LOG = os.path.join(REPO_ROOT, "download.txt")
METRICS_DIR = os.path.join(REPO_ROOT, "metrics")
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")

os.makedirs(METRICS_DIR, exist_ok=True)


def parse_download_txt():
    print(f"Parsing Cloud TPU training log: {DOWNLOAD_LOG}...")
    with open(DOWNLOAD_LOG, "r") as f:
        lines = f.readlines()

    step_pat = re.compile(
        r'\[Step\s+(?P<step>\d+)/(?P<tot_steps>\d+)\]\s+\[(?P<mode>[^\]]+)\]\s+RealTok:\s+(?P<realtok>[\d\.]+)B/(?P<tottok>[\d\.]+)B\s+\((?P<prog>[\d\.]+)%\)\s+\[Loss:\s+(?P<loss_tok>[\d\.]+)B,\s+Prompt:\s+(?P<prompt_tok>[\d\.]+)B\s+\|\s+Pad:\s+(?P<pad>[\d\.]+)%\]\s+\|\s+ETA:\s+(?P<eta>[\d\.]+)h\s+\|\s+Loss:\s+(?P<loss>[\d\.]+)\s+\|\s+Steps/s:\s+(?P<steps_per_s>[\d\.]+)\s+\|\s+Tok/s:\s+(?P<tok_per_s>[\d,]+)\s+\|\s+Host RSS:\s+(?P<host_rss>[\d\.]+)\s+GB\s+\|\s+HBM:\s+(?P<hbm_used>[\d\.]+)/(?P<hbm_tot>[\d\.]+)\s+GB\s+\|\s+LR:\s+(?P<lr>[\d\.e\-\+]+)\s+\|\s+Samp_p:\s+(?P<samp_p>[\d\.]+)'
    )

    diag_pat = re.compile(
        r'↳\s+\[Optimizer Diag\]\s+GradNorm:\s+(?P<grad_norm>[\d\.]+)\s+\(Clip:\s+(?P<clip>[\d\.]+)%\)\s+\|\s+Skips:\s+(?P<skips>\d+)\s+\|\s+U/W\s+\[M:\s+(?P<uw_m>[\d\.e\-\+]+),\s+Vocab:\s+(?P<uw_vocab>[\d\.e\-\+]+),\s+1D:\s+(?P<uw_1d>[\d\.e\-\+]+)\]\s+\|\s+Drift:\s+(?P<drift>[\d\.e\-\+]+)'
    )

    eval_pat = re.compile(
        r'\[Eval\s+@\s+Step\s+(?P<step>[\d,]+)\]\s+Composite Held-Out Loss:\s+(?P<loss>[\d\.]+)\s+\|\s+PPL:\s+(?P<ppl>[\d\.]+)'
    )
    fineweb_pat = re.compile(r'Broad Web.*Loss\s+(?P<loss>[\d\.]+)\s+\(PPL:\s+(?P<ppl>[\d\.]+)\)')
    stack_pat = re.compile(r'Code Repositories.*Loss\s+(?P<loss>[\d\.]+)\s+\(PPL:\s+(?P<ppl>[\d\.]+)\)')

    training_steps = []
    evaluations = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Extract timestamp if present at beginning
        ts_match = re.match(r'^([\d\.]+)s\s+(\d+)', line)
        ts_sec = float(ts_match.group(1)) if ts_match else None
        line_no = int(ts_match.group(2)) if ts_match else (i + 1)

        # Check step
        s_m = step_pat.search(line)
        if s_m:
            step_data = s_m.groupdict()
            row = {
                "step": int(step_data["step"]),
                "total_steps": int(step_data["tot_steps"]),
                "elapsed_time_sec": ts_sec,
                "train_mode": step_data["mode"],
                "realtok_billion": float(step_data["realtok"]),
                "target_tokens_billion": float(step_data["tottok"]),
                "progress_pct": float(step_data["prog"]),
                "loss_tokens_billion": float(step_data["loss_tok"]),
                "prompt_tokens_billion": float(step_data["prompt_tok"]),
                "pad_pct": float(step_data["pad"]),
                "eta_hours": float(step_data["eta"]),
                "train_loss": float(step_data["loss"]),
                "steps_per_sec": float(step_data["steps_per_s"]),
                "tokens_per_sec": int(step_data["tok_per_s"].replace(",", "")),
                "host_rss_gb": float(step_data["host_rss"]),
                "hbm_used_gb": float(step_data["hbm_used"]),
                "hbm_total_gb": float(step_data["hbm_tot"]),
                "learning_rate": float(step_data["lr"]),
                "sampling_p": float(step_data["samp_p"]),
                "grad_norm": None,
                "clip_pct": None,
                "skips": None,
                "uw_manifold": None,
                "uw_vocab": None,
                "uw_1d": None,
                "drift": None,
            }

            # Check next line for Optimizer Diag
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                d_m = diag_pat.search(next_line)
                if d_m:
                    diag_data = d_m.groupdict()
                    row["grad_norm"] = float(diag_data["grad_norm"])
                    row["clip_pct"] = float(diag_data["clip"])
                    row["skips"] = int(diag_data["skips"])
                    row["uw_manifold"] = float(diag_data["uw_m"])
                    row["uw_vocab"] = float(diag_data["uw_vocab"])
                    row["uw_1d"] = float(diag_data["uw_1d"])
                    row["drift"] = float(diag_data["drift"])
                    i += 1  # consumed next line

            training_steps.append(row)
            i += 1
            continue

        # Check eval
        e_m = eval_pat.search(line)
        if e_m:
            e_data = e_m.groupdict()
            eval_row = {
                "step": int(e_data["step"].replace(",", "")),
                "elapsed_time_sec": ts_sec,
                "composite_eval_loss": float(e_data["loss"]),
                "composite_ppl": float(e_data["ppl"]),
                "fineweb_loss": None,
                "fineweb_ppl": None,
                "code_loss": None,
                "code_ppl": None,
            }

            # Inspect subsequent lines for FineWeb and The Stack sub-losses
            sub_j = i + 1
            while sub_j < min(i + 4, len(lines)):
                sub_line = lines[sub_j]
                fw_m = fineweb_pat.search(sub_line)
                st_m = stack_pat.search(sub_line)
                if fw_m:
                    eval_row["fineweb_loss"] = float(fw_m.group("loss"))
                    eval_row["fineweb_ppl"] = float(fw_m.group("ppl"))
                elif st_m:
                    eval_row["code_loss"] = float(st_m.group("loss"))
                    eval_row["code_ppl"] = float(st_m.group("ppl"))
                sub_j += 1

            evaluations.append(eval_row)
            i += 1
            continue

        i += 1

    print(f"Extracted {len(training_steps):,} training steps and {len(evaluations):,} evaluation checkpoints.")

    # 1. Write metrics/cerebro_v5_training_steps.csv
    train_csv_path = os.path.join(METRICS_DIR, "cerebro_v5_training_steps.csv")
    if training_steps:
        fieldnames = list(training_steps[0].keys())
        with open(train_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_steps)
        print(f"Saved: {train_csv_path} ({len(training_steps):,} rows)")

    # 2. Write metrics/cerebro_v5_evaluations.csv
    eval_csv_path = os.path.join(METRICS_DIR, "cerebro_v5_evaluations.csv")
    if evaluations:
        fieldnames = list(evaluations[0].keys())
        with open(eval_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(evaluations)
        print(f"Saved: {eval_csv_path} ({len(evaluations):,} rows)")

    # 3. Write unified outer-join master timeline
    unified_csv_path = os.path.join(METRICS_DIR, "cerebro_v5_unified_training_and_eval.csv")
    all_steps = sorted(set([r["step"] for r in training_steps] + [r["step"] for r in evaluations]))
    train_by_step = {r["step"]: r for r in training_steps}
    eval_by_step = {r["step"]: r for r in evaluations}

    unified_rows = []
    for s in all_steps:
        tr = train_by_step.get(s, {})
        ev = eval_by_step.get(s, {})
        merged = {
            "step": s,
            "elapsed_time_sec": tr.get("elapsed_time_sec") or ev.get("elapsed_time_sec"),
            "train_loss": tr.get("train_loss"),
            "composite_eval_loss": ev.get("composite_eval_loss"),
            "composite_ppl": ev.get("composite_ppl"),
            "fineweb_loss": ev.get("fineweb_loss"),
            "fineweb_ppl": ev.get("fineweb_ppl"),
            "code_loss": ev.get("code_loss"),
            "code_ppl": ev.get("code_ppl"),
            "realtok_billion": tr.get("realtok_billion"),
            "progress_pct": tr.get("progress_pct"),
            "tokens_per_sec": tr.get("tokens_per_sec"),
            "steps_per_sec": tr.get("steps_per_sec"),
            "learning_rate": tr.get("learning_rate"),
            "grad_norm": tr.get("grad_norm"),
            "uw_manifold": tr.get("uw_manifold"),
            "drift": tr.get("drift"),
            "host_rss_gb": tr.get("host_rss_gb"),
            "hbm_used_gb": tr.get("hbm_used_gb"),
        }
        unified_rows.append(merged)

    with open(unified_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(unified_rows[0].keys()))
        writer.writeheader()
        writer.writerows(unified_rows)
    print(f"Saved: {unified_csv_path} ({len(unified_rows):,} rows)")


def export_audit_results():
    audit_json = os.path.join(EXPERIMENTS_DIR, "comprehensive_read_only_audit_results.json")
    if not os.path.exists(audit_json):
        print(f"File not found: {audit_json}")
        return

    print(f"Exporting comprehensive audit results from: {audit_json}...")
    with open(audit_json, "r") as f:
        d = json.load(f)

    # 4. Audit Shell Gradients
    shell_metrics = d.get("phase_2_gradient_distribution", {}).get("shell_metrics", [])
    if shell_metrics:
        p = os.path.join(METRICS_DIR, "cerebro_v5_audit_shell_gradients.csv")
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(shell_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(shell_metrics)
        print(f"Saved: {p} ({len(shell_metrics)} rows)")

    # 5. Audit Optimizer Dynamics
    opt_audit = d.get("phase_5_optimizer_audit", {}).get("opt_audit", [])
    if opt_audit:
        p = os.path.join(METRICS_DIR, "cerebro_v5_audit_optimizer_dynamics.csv")
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(opt_audit[0].keys()))
            writer.writeheader()
            writer.writerows(opt_audit)
        print(f"Saved: {p} ({len(opt_audit)} rows)")

    # 6. Audit Temporal Displacement
    disp_matrix = d.get("phase_3_temporal_credit_assignment", {}).get("displacement_matrix", [])
    if disp_matrix:
        p = os.path.join(METRICS_DIR, "cerebro_v5_audit_temporal_displacement.csv")
        fieldnames = ["time_step"] + [f"S{k:02d}" for k in range(len(disp_matrix[0]))]
        rows = []
        for t, row in enumerate(disp_matrix):
            entry = {"time_step": t}
            for k, val in enumerate(row):
                entry[f"S{k:02d}"] = val
            rows.append(entry)
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {p} ({len(rows)} rows)")

    # 7. Audit Radial Transmission
    radial_sweep = d.get("phase_4_radial_communication", {})
    if radial_sweep:
        p = os.path.join(METRICS_DIR, "cerebro_v5_audit_radial_transmission.csv")
        rows = []
        for r_val, data in radial_sweep.items():
            rows.append({
                "r_mix": float(r_val),
                "mean_self_coupling": data["mean_self_coupling"],
                "mean_outward_coupling": data["mean_outward_coupling"],
                "mean_inward_coupling": data["mean_inward_coupling"],
                "outward_to_self_ratio": data["outward_to_self_ratio"],
            })
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {p} ({len(rows)} rows)")


def export_other_benchmarks():
    # Frozen readout rescue
    f_json = os.path.join(EXPERIMENTS_DIR, "frozen_readout_rescue_benchmark_results.json")
    if os.path.exists(f_json):
        with open(f_json) as f:
            data = json.load(f)
        sweep = data.get("pool_gain_sweep", [])
        if sweep:
            p = os.path.join(METRICS_DIR, "cerebro_v5_frozen_readout_rescue.csv")
            with open(p, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
                writer.writeheader()
                writer.writerows(sweep)
            print(f"Saved: {p} ({len(sweep)} rows)")

    # Condition D benchmark
    cond_d_json = os.path.join(EXPERIMENTS_DIR, "condition_d_full_benchmark_results.json")
    if os.path.exists(cond_d_json):
        with open(cond_d_json) as f:
            data = json.load(f)
        rows = []
        for arm_name, arm_data in data.items():
            if isinstance(arm_data, dict):
                r_mix = arm_data.get("r_mix")
                for seed_key, seed_data in arm_data.get("seeds", {}).items():
                    steps = seed_data.get("steps", {})
                    for step_num, step_info in steps.items():
                        rows.append({
                            "arm": arm_name,
                            "r_mix": r_mix,
                            "seed": seed_key,
                            "step": int(step_num),
                            "val_nll": step_info.get("val_nll"),
                            "interior_cka": step_info.get("interior_cka"),
                            "interior_utility": step_info.get("interior_utility"),
                            "pool_utility": step_info.get("pool_utility"),
                            "radial_update_norm": step_info.get("radial_update_norm"),
                        })
        if rows:
            p = os.path.join(METRICS_DIR, "cerebro_v5_condition_d_benchmark.csv")
            with open(p, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved: {p} ({len(rows)} rows)")


def main():
    print("=" * 60)
    print("EXPORTING ALL TRAINING & AUDIT LOGS TO CSV")
    print("=" * 60)
    parse_download_txt()
    export_audit_results()
    export_other_benchmarks()
    print("=" * 60)
    print("ALL LOGS SUCCESSFULLY EXPORTED TO CSV IN metrics/")
    print("=" * 60)


if __name__ == "__main__":
    main()
