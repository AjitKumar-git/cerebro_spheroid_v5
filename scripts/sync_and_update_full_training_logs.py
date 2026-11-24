#!/usr/bin/env python3
"""
Synchronizes full training and evaluation history from bioflo.in and merges with
local Cloud TPU telemetry from download.txt.

Updates:
  1. metrics/cerebro_v5_training_steps.csv
  2. metrics/cerebro_v5_evaluations.csv
  3. metrics/cerebro_v5_unified_training_and_eval.csv
Also archives raw remote data into:
  - metrics/bioflo_remote_loss_history.json
  - metrics/bioflo_manifest.json
"""

import os
import sys
import re
import csv
import json
import math
from typing import Dict, Any, List

REPO_ROOT = "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5"
DOWNLOAD_LOG = os.path.join(REPO_ROOT, "download.txt")
METRICS_DIR = os.path.join(REPO_ROOT, "metrics")

sys.path.insert(0, REPO_ROOT)
from src.cloud_sync import (
    fetch_remote_loss_history,
    fetch_remote_token_tracker,
    check_remote_manifest,
)

os.makedirs(METRICS_DIR, exist_ok=True)


def fetch_from_bioflo():
    print("Connecting to bioflo.in to retrieve full pretraining telemetry...")
    manifest = check_remote_manifest("v5_small")
    token_tracker = fetch_remote_token_tracker("v5_small")
    loss_history = fetch_remote_loss_history("v5_small")

    print(f"  Remote status: {manifest.get('status')}")
    print(f"  Retrieved {len(loss_history or []):,} loss history records from bioflo.in")
    print(f"  Remote cumulative tokens: {token_tracker.get('cumulative_tokens', 0):,} ({token_tracker.get('pct_complete', 0.0):.2f}%)")

    # Save raw archives
    with open(os.path.join(METRICS_DIR, "bioflo_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if loss_history:
        with open(os.path.join(METRICS_DIR, "bioflo_remote_loss_history.json"), "w") as f:
            json.dump(loss_history, f, indent=2)

    return manifest, token_tracker, loss_history or []


def parse_download_txt():
    print(f"Parsing local high-resolution Cloud TPU logs from: {DOWNLOAD_LOG}...")
    if not os.path.exists(DOWNLOAD_LOG):
        print(f"Warning: {DOWNLOAD_LOG} not found.")
        return {}, {}

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

    training_steps_by_step = {}
    evaluations_by_step = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        ts_match = re.match(r'^([\d\.]+)s\s+(\d+)', line)
        ts_sec = float(ts_match.group(1)) if ts_match else None

        # Match step
        s_m = step_pat.search(line)
        if s_m:
            sd = s_m.groupdict()
            step = int(sd["step"])
            row = {
                "step": step,
                "total_steps": int(sd["tot_steps"]),
                "elapsed_time_sec": ts_sec,
                "train_mode": sd["mode"],
                "realtok_billion": float(sd["realtok"]),
                "target_tokens_billion": float(sd["tottok"]),
                "progress_pct": float(sd["prog"]),
                "loss_tokens_billion": float(sd["loss_tok"]),
                "prompt_tokens_billion": float(sd["prompt_tok"]),
                "pad_pct": float(sd["pad"]),
                "eta_hours": float(sd["eta"]),
                "train_loss": float(sd["loss"]),
                "steps_per_sec": float(sd["steps_per_s"]),
                "tokens_per_sec": int(sd["tok_per_s"].replace(",", "")),
                "host_rss_gb": float(sd["host_rss"]),
                "hbm_used_gb": float(sd["hbm_used"]),
                "hbm_total_gb": float(sd["hbm_tot"]),
                "learning_rate": float(sd["lr"]),
                "sampling_p": float(sd["samp_p"]),
                "grad_norm": None,
                "clip_pct": None,
                "skips": None,
                "uw_manifold": None,
                "uw_vocab": None,
                "uw_1d": None,
                "drift": None,
            }

            if i + 1 < len(lines):
                d_m = diag_pat.search(lines[i + 1])
                if d_m:
                    dd = d_m.groupdict()
                    row["grad_norm"] = float(dd["grad_norm"])
                    row["clip_pct"] = float(dd["clip"])
                    row["skips"] = int(dd["skips"])
                    row["uw_manifold"] = float(dd["uw_m"])
                    row["uw_vocab"] = float(dd["uw_vocab"])
                    row["uw_1d"] = float(dd["uw_1d"])
                    row["drift"] = float(dd["drift"])
                    i += 1

            training_steps_by_step[step] = row
            i += 1
            continue

        # Match eval
        e_m = eval_pat.search(line)
        if e_m:
            ed = e_m.groupdict()
            step = int(ed["step"].replace(",", ""))
            ev_row = {
                "step": step,
                "elapsed_time_sec": ts_sec,
                "composite_eval_loss": float(ed["loss"]),
                "composite_ppl": float(ed["ppl"]),
                "fineweb_loss": None,
                "fineweb_ppl": None,
                "code_loss": None,
                "code_ppl": None,
            }

            sub_j = i + 1
            while sub_j < min(i + 4, len(lines)):
                sub_line = lines[sub_j]
                fw_m = fineweb_pat.search(sub_line)
                st_m = stack_pat.search(sub_line)
                if fw_m:
                    ev_row["fineweb_loss"] = float(fw_m.group("loss"))
                    ev_row["fineweb_ppl"] = float(fw_m.group("ppl"))
                elif st_m:
                    ev_row["code_loss"] = float(st_m.group("loss"))
                    ev_row["code_ppl"] = float(st_m.group("ppl"))
                sub_j += 1

            evaluations_by_step[step] = ev_row
            i += 1
            continue

        i += 1

    print(f"  Parsed {len(training_steps_by_step):,} steps and {len(evaluations_by_step):,} evals from download.txt")
    return training_steps_by_step, evaluations_by_step


def merge_and_save(bioflo_records: List[Dict[str, Any]], local_steps: Dict[int, Any], local_evals: Dict[int, Any]):
    print("\nMerging bioflo.in cloud records with local high-resolution logs...")

    # Organize bioflo records by step
    bioflo_by_step = {}
    for r in bioflo_records:
        if "step" in r:
            bioflo_by_step[int(r["step"])] = r

    all_steps = sorted(set(list(bioflo_by_step.keys()) + list(local_steps.keys())))
    print(f"Total unified unique training steps: {len(all_steps):,} (from Step {min(all_steps)} to Step {max(all_steps)})")

    # 1. Update metrics/cerebro_v5_training_steps.csv
    train_rows = []
    for s in all_steps:
        b_rec = bioflo_by_step.get(s, {})
        l_rec = local_steps.get(s, {})

        t_loss = l_rec.get("train_loss")
        if t_loss is None:
            t_loss = b_rec.get("loss")

        e_loss = b_rec.get("eval_loss")

        merged = {
            "step": s,
            "timestamp": b_rec.get("timestamp"),
            "source": b_rec.get("source") or ("kaggle" if l_rec else "unknown"),
            "status": b_rec.get("status"),
            "train_loss": round(float(t_loss), 4) if t_loss is not None else None,
            "eval_loss": round(float(e_loss), 4) if e_loss is not None else None,
            "total_steps": l_rec.get("total_steps") or 300840,
            "elapsed_time_sec": l_rec.get("elapsed_time_sec"),
            "train_mode": l_rec.get("train_mode") or "ALL_TRAINABLE",
            "realtok_billion": l_rec.get("realtok_billion"),
            "target_tokens_billion": l_rec.get("target_tokens_billion") or 15.70,
            "progress_pct": l_rec.get("progress_pct"),
            "loss_tokens_billion": l_rec.get("loss_tokens_billion"),
            "prompt_tokens_billion": l_rec.get("prompt_tokens_billion"),
            "pad_pct": l_rec.get("pad_pct"),
            "eta_hours": l_rec.get("eta_hours"),
            "steps_per_sec": l_rec.get("steps_per_sec"),
            "tokens_per_sec": l_rec.get("tokens_per_sec"),
            "host_rss_gb": l_rec.get("host_rss_gb"),
            "hbm_used_gb": l_rec.get("hbm_used_gb"),
            "hbm_total_gb": l_rec.get("hbm_total_gb"),
            "learning_rate": l_rec.get("learning_rate"),
            "sampling_p": l_rec.get("sampling_p"),
            "grad_norm": l_rec.get("grad_norm"),
            "clip_pct": l_rec.get("clip_pct"),
            "skips": l_rec.get("skips"),
            "uw_manifold": l_rec.get("uw_manifold"),
            "uw_vocab": l_rec.get("uw_vocab"),
            "uw_1d": l_rec.get("uw_1d"),
            "drift": l_rec.get("drift"),
        }
        train_rows.append(merged)

    train_csv = os.path.join(METRICS_DIR, "cerebro_v5_training_steps.csv")
    with open(train_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
        writer.writeheader()
        writer.writerows(train_rows)
    print(f"✅ Successfully updated: {train_csv} ({len(train_rows):,} rows)")

    # 2. Update metrics/cerebro_v5_evaluations.csv
    # Combine bioflo eval records with local granular domain evals
    eval_steps = sorted(set(
        [s for s, r in bioflo_by_step.items() if r.get("eval_loss") is not None] +
        list(local_evals.keys())
    ))
    print(f"Total unified evaluation checkpoints: {len(eval_steps):,} (from Step {min(eval_steps)} to Step {max(eval_steps)})")

    eval_rows = []
    for s in eval_steps:
        b_rec = bioflo_by_step.get(s, {})
        l_ev = local_evals.get(s, {})

        c_loss = l_ev.get("composite_eval_loss")
        if c_loss is None:
            c_loss = b_rec.get("eval_loss")

        c_ppl = l_ev.get("composite_ppl")
        if c_ppl is None and c_loss is not None:
            try:
                c_ppl = round(math.exp(min(c_loss, 20.0)), 2)
            except OverflowError:
                c_ppl = None

        ev_entry = {
            "step": s,
            "timestamp": b_rec.get("timestamp"),
            "source": b_rec.get("source") or ("kaggle" if l_ev else "unknown"),
            "composite_eval_loss": round(float(c_loss), 4) if c_loss is not None else None,
            "composite_ppl": c_ppl,
            "fineweb_loss": l_ev.get("fineweb_loss"),
            "fineweb_ppl": l_ev.get("fineweb_ppl"),
            "code_loss": l_ev.get("code_loss"),
            "code_ppl": l_ev.get("code_ppl"),
            "elapsed_time_sec": l_ev.get("elapsed_time_sec"),
            "status": b_rec.get("status"),
        }
        eval_rows.append(ev_entry)

    eval_csv = os.path.join(METRICS_DIR, "cerebro_v5_evaluations.csv")
    with open(eval_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(eval_rows)
    print(f"✅ Successfully updated: {eval_csv} ({len(eval_rows):,} rows)")

    # 3. Update metrics/cerebro_v5_unified_training_and_eval.csv
    # Full master timeline aligning step, train_loss, eval_loss, domain breakdowns, and optimizer diagnostics
    unified_rows = []
    eval_dict_by_step = {r["step"]: r for r in eval_rows}
    train_dict_by_step = {r["step"]: r for r in train_rows}
    master_steps = sorted(set(list(train_dict_by_step.keys()) + list(eval_dict_by_step.keys())))

    for s in master_steps:
        tr = train_dict_by_step.get(s, {})
        ev = eval_dict_by_step.get(s, {})

        u_entry = {
            "step": s,
            "timestamp": tr.get("timestamp") or ev.get("timestamp"),
            "source": tr.get("source") or ev.get("source"),
            "train_loss": tr.get("train_loss"),
            "eval_loss": ev.get("composite_eval_loss"),
            "eval_ppl": ev.get("composite_ppl"),
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
            "clip_pct": tr.get("clip_pct"),
            "uw_manifold": tr.get("uw_manifold"),
            "uw_vocab": tr.get("uw_vocab"),
            "drift": tr.get("drift"),
            "host_rss_gb": tr.get("host_rss_gb"),
            "hbm_used_gb": tr.get("hbm_used_gb"),
            "status": tr.get("status") or ev.get("status"),
        }
        unified_rows.append(u_entry)

    unified_csv = os.path.join(METRICS_DIR, "cerebro_v5_unified_training_and_eval.csv")
    with open(unified_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(unified_rows[0].keys()))
        writer.writeheader()
        writer.writerows(unified_rows)
    print(f"✅ Successfully updated: {unified_csv} ({len(unified_rows):,} rows)")


def main():
    print("=" * 60)
    print("SYNCING FULL TRAINING LOGS FROM BIOFLO.IN & MERGING CSVs")
    print("=" * 60)
    manifest, tokens, bioflo_records = fetch_from_bioflo()
    local_steps, local_evals = parse_download_txt()
    merge_and_save(bioflo_records, local_steps, local_evals)
    print("=" * 60)
    print("SYNC AND CSV UPDATE COMPLETE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
