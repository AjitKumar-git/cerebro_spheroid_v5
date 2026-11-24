#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Multi-Sequence Adaptive Frequency Sweep

Executes the Phase 3 Multi-Timescale Frequency Sweep across diverse domains:
- Variant A:    Frozen baseline (Core every token; Mid & Outer frozen)
- Variant B_4:  Fixed periodic every 4 tokens (Mid /4; Outer /16)
- Variant B_8:  Fixed periodic every 8 tokens (Mid /8; Outer /32)
- Variant B_16: Fixed periodic every 16 tokens (Mid /16; Outer /64)
- Variant C_loose:   Adaptive loose threshold (tau = 2.8, frequent event triggers)
- Variant C_current: Adaptive current threshold (tau = 3.2, balanced triggers)
- Variant C_strict:  Adaptive strict threshold (tau = 3.8, sparse high-surprisal triggers)
- Variant D:    Full dense recurrence (All 16 shells updated at every single token)

Statistical Reporting:
1. Paired delta NLL: Delta_i = NLL(V, i) - NLL(A, i) for every sequence i.
2. Mean, median, and std dev of Delta_i.
3. 95% Bootstrap Confidence Interval on paired differences.
4. Win rate (% sequences where V beats Baseline A).
5. Breakdown by domain (Code, Natural Language, Science, Mathematics).
6. State displacement D_k and FLOPs/tok scaling.
"""

import os
import sys
import time
import math
import argparse
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import jax
import jax.numpy as jnp

from src.spheroid_v5 import infer_v5_config_from_state_dict
from src.jax.v5_model import (
    ModelConfigV5,
    deliberate_step,
    deliberate_cycles_jax,
    encode_sensory,
    emit_step,
)
from src.jax.checkpoint import load_torch_pt_to_jax


# Standardized Multi-Domain Evaluation Suite
CORPUS = [
    # --- Python / Code ---
    {"domain": "Code", "text": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n"},
    {"domain": "Code", "text": "class BinarySearchTree:\n    def __init__(self, value):\n        self.val = value\n        self.left = None\n        self.right = None\n    def insert(self, key):\n        if key < self.val:\n            if self.left:\n                self.left.insert(key)\n            else:\n                self.left = BinarySearchTree(key)\n"},
    {"domain": "Code", "text": "def depth_first_search(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    for neighbor in graph.get(start, []):\n        if neighbor not in visited:\n            depth_first_search(graph, neighbor, visited)\n    return visited\n"},
    {"domain": "Code", "text": "import threading\nlock = threading.Lock()\ndef synchronized_counter(shared_state, increment):\n    with lock:\n        current = shared_state.value\n        shared_state.value = current + increment\n    return shared_state.value\n"},

    # --- Natural Language / Encyclopedic ---
    {"domain": "Language", "text": "The Library of Alexandria was one of the largest and most significant libraries of the ancient world. Dedicated to the Muses, the nine goddesses of the arts, it flourished under the patronage of the Ptolemaic dynasty."},
    {"domain": "Language", "text": "Linguistic typology studies and classifies languages according to their structural features. Its goal is to describe and explain the common properties and the structural diversity of the world's languages."},
    {"domain": "Language", "text": "The industrial revolution marked a period of development in the latter half of the eighteenth century that transformed largely rural, agrarian societies in Europe and America into industrialized, urban centers."},
    {"domain": "Language", "text": "Geographic information systems integrate spatial data with attribute information, enabling analysts to visualize, question, analyze, and interpret location-based data to understand relationships and patterns."},

    # --- Scientific Text (Biology & Chemistry) ---
    {"domain": "Science", "text": "Ribosomes are macromolecular machines found within all living cells that perform biological protein synthesis. Ribosomes link amino acids together in the order specified by codons of messenger RNA molecules."},
    {"domain": "Science", "text": "In biochemistry, Michaelis-Menten kinetics is one of the best-known models of enzyme kinetics. It takes the form of an equation describing the rate of enzymatic reactions by relating reaction rate to substrate concentration."},
    {"domain": "Science", "text": "CRISPR-Cas9 is a specialized bacterial defense mechanism adapted into a revolutionary gene-editing technology. The endonuclease creates double-strand breaks at targeted genomic loci specified by guide RNA sequences."},
    {"domain": "Science", "text": "Cellular respiration breaks down glucose molecules to extract energy in the form of adenosine triphosphate through glycolysis, the tricarboxylic acid cycle, and oxidative phosphorylation."},

    # --- Mathematics & Physics ---
    {"domain": "Math/Physics", "text": "The Navier-Stokes equations describe how the velocity, pressure, temperature, and density of a moving fluid are related. They are differential equations derived by applying Newton's second law to fluid motion."},
    {"domain": "Math/Physics", "text": "In quantum mechanics, Heisenberg's uncertainty principle asserts a fundamental limit to the precision with which certain pairs of physical properties of a particle, such as position and momentum, can be known."},
    {"domain": "Math/Physics", "text": "The Fourier transform decomposes functions depending on space or time into functions depending on spatial or temporal frequency, representing complex signals as superpositions of sinusoidal basis waves."},
    {"domain": "Math/Physics", "text": "General relativity explains gravitational interaction as the curvature of spacetime caused by the uneven distribution of mass-energy, mathematically modeled via Einstein's field equations."},
]


def evaluate_single_sequence(
    params: Dict[str, jnp.ndarray],
    prompt_ids: jnp.ndarray,
    target_ids: jnp.ndarray,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: ModelConfigV5,
    mode: str = "A",
    chunk_stride: int = 8,
    ctx_stride: int = 32,
    adaptive_threshold: float = 3.2,
) -> Tuple[float, float, float, float, int, int]:
    """
    Evaluates one unpadded sequence under the designated update schedule.
    Returns:
        (mean_token_nll, d_core, d_mid, d_outer, chunk_triggers, ctx_triggers)
    """
    K = config.num_shells
    D = config.surface_dim
    total_L = target_ids.shape[1]
    prompt_len = prompt_ids.shape[1]

    # Prompt deliberation
    p_flux, p_kv = encode_sensory(params, prompt_ids, config=config, pos_offset=0)
    r_prof = (jnp.arange(1, K + 1, dtype=jnp.float32) / float(K))[None, :, None]
    init_state = (jnp.broadcast_to(p_flux, (1, K, D)).astype(jnp.float32) * r_prof).astype(jnp.float32)
    state, _ = deliberate_cycles_jax(params, init_state, buffers, config=config, is_training=False, sensory_kv=p_kv)

    # Preceding target tokens as auditory feedback
    prev_toks = jnp.concatenate([prompt_ids[:, -1:], target_ids[:, :-1]], axis=1)
    tf_flux, (tf_k, tf_v) = encode_sensory(params, prev_toks, config=config, pos_offset=prompt_len, is_causal=True)
    causal_k = jnp.concatenate([p_kv[0], tf_k], axis=2)
    causal_v = jnp.concatenate([p_kv[1], tf_v], axis=2)

    cur_state = state
    cur_plan = jnp.zeros((1, D), dtype=jnp.float32)
    cur_trace = None

    state_history = [cur_state]
    nlls = []
    chunk_triggers = 0
    ctx_triggers = 0
    curr_chunk_len = 0

    for t in range(total_L):
        curr_ctx_len = prompt_len + t + 1
        sub_k = causal_k[:, :, :curr_ctx_len, :]
        sub_v = causal_v[:, :, :curr_ctx_len, :]

        logits, nxt_state, cur_trace, cur_plan = emit_step(
            params,
            cur_state,
            (sub_k, sub_v),
            cur_plan,
            cur_trace,
            t,
            buffers,
            config=config,
            auditory_feedback=tf_flux[:, t, :],
            query_position=prompt_len + t,
        )

        tgt = target_ids[0, t]
        log_p = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        step_nll = float(-log_p[0, tgt])
        nlls.append(step_nll)

        curr_chunk_len += 1

        # Multi-timescale state routing
        if mode == "A":
            cur_state = nxt_state

        elif mode.startswith("B"):
            cur_state = nxt_state
            if (t > 0) and (t % chunk_stride == 0):
                delib = deliberate_step(params, cur_state, buffers, config=config)
                cur_state = cur_state.at[:, 1:12, :].set(0.85 * cur_state[:, 1:12, :] + 0.15 * delib[:, 1:12, :])
                chunk_triggers += 1
                curr_chunk_len = 0
            if (t > 0) and (t % ctx_stride == 0):
                delib = deliberate_step(params, cur_state, buffers, config=config)
                cur_state = cur_state.at[:, 12:, :].set(0.90 * cur_state[:, 12:, :] + 0.10 * delib[:, 12:, :])
                ctx_triggers += 1

        elif mode.startswith("C"):
            probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
            entropy = float(-jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1)[0])
            g_act = 1.0 / (1.0 + math.exp(-max(min(entropy - adaptive_threshold, 10.0), -10.0)))

            cur_state = nxt_state
            is_trigger = (g_act > 0.5) or (curr_chunk_len >= chunk_stride * 2)
            if is_trigger and (t > 0):
                chunk_triggers += 1
                curr_chunk_len = 0
                delib = deliberate_step(params, cur_state, buffers, config=config)
                cur_state = cur_state.at[:, 1:12, :].set(
                    cur_state[:, 1:12, :] + 0.15 * g_act * (delib[:, 1:12, :] - cur_state[:, 1:12, :])
                )
                if chunk_triggers % 4 == 0:
                    ctx_triggers += 1
                    cur_state = cur_state.at[:, 12:, :].set(
                        0.90 * cur_state[:, 12:, :] + 0.10 * delib[:, 12:, :]
                    )

        elif mode == "D":
            delib = deliberate_step(params, nxt_state, buffers, config=config)
            cur_state = 0.85 * nxt_state + 0.15 * delib

        state_history.append(cur_state)

    state_traj = jnp.stack(state_history, axis=1)  # [1, T+1, K, D]
    step_diffs = jnp.linalg.norm(state_traj[:, 1:, :, :] - state_traj[:, :-1, :, :], axis=-1)
    mean_d = np.array(jnp.mean(step_diffs, axis=(0, 1)))
    d_c, d_m, d_o = float(mean_d[0]), float(np.mean(mean_d[1:12])), float(np.mean(mean_d[12:]))

    return float(np.mean(nlls)), d_c, d_m, d_o, chunk_triggers, ctx_triggers


def compute_bootstrap_ci(data: List[float], num_samples: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Computes empirical bootstrap confidence interval for mean delta."""
    if len(data) <= 1:
        return float(data[0] if data else 0.0), float(data[0] if data else 0.0)
    arr = np.array(data)
    boot_means = []
    rng = np.random.RandomState(42)
    for _ in range(num_samples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(float(np.mean(sample)))
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1.0 - alpha) * 100))


def run_sweep(checkpoint_path: str = "checkpoints/v5_small_latest.pt", seq_len: int = 32):
    print("=" * 95)
    print("🧠 CEREBRO-SPHEROID V5: PHASE 3 MULTI-SEQUENCE ADAPTIVE FREQUENCY SWEEP")
    print(f"📁 Target Checkpoint: {checkpoint_path} | Sequence Length: {seq_len} tokens")
    print("=" * 95)

    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    cfg = infer_v5_config_from_state_dict(state, default_preset="v5_small")
    cfg.use_sensory_self_attn = True
    cfg.use_rope = False
    cfg.top_k_resonance = 0

    params, buffers, meta = load_torch_pt_to_jax(checkpoint_path, cfg, target_dtype=jnp.float32)
    print(f"✓ Checkpoint loaded successfully | Total Evaluation Sequences: {len(CORPUS)}")

    variants = [
        ("A",        "A: Frozen Baseline",      {"mode": "A", "chunk_stride": 8, "ctx_stride": 32, "adaptive_threshold": 3.2}),
        ("B_4",      "B4: Fixed /4 tokens",     {"mode": "B", "chunk_stride": 4, "ctx_stride": 16, "adaptive_threshold": 3.2}),
        ("B_8",      "B8: Fixed /8 tokens",     {"mode": "B", "chunk_stride": 8, "ctx_stride": 32, "adaptive_threshold": 3.2}),
        ("B_16",     "B16: Fixed /16 tokens",   {"mode": "B", "chunk_stride": 16, "ctx_stride": 64, "adaptive_threshold": 3.2}),
        ("C_loose",  "C-Loose (tau=2.8)",       {"mode": "C", "chunk_stride": 8, "ctx_stride": 32, "adaptive_threshold": 2.8}),
        ("C_curr",   "C-Current (tau=3.2)",     {"mode": "C", "chunk_stride": 8, "ctx_stride": 32, "adaptive_threshold": 3.2}),
        ("C_strict", "C-Strict (tau=3.8)",      {"mode": "C", "chunk_stride": 8, "ctx_stride": 32, "adaptive_threshold": 3.8}),
        ("D",        "D: Full Dense Recurrent", {"mode": "D", "chunk_stride": 1, "ctx_stride": 1, "adaptive_threshold": 3.2}),
    ]

    # Pre-tokenize all sequences
    tokenized_corpus = []
    for item in CORPUS:
        toks = tokenizer.encode(item["text"])
        target_ids = jnp.array([toks[:seq_len]], dtype=jnp.int32)
        prompt_ids = jnp.array([[1]], dtype=jnp.int32)
        tokenized_corpus.append({"domain": item["domain"], "prompt": prompt_ids, "target": target_ids})

    # Warmup JIT
    print("⏳ Warming up JAX compilation...")
    _ = evaluate_single_sequence(
        params, tokenized_corpus[0]["prompt"], tokenized_corpus[0]["target"], buffers, cfg, mode="A"
    )

    # Results dictionary
    results_per_var = {v_id: [] for v_id, _, _ in variants}
    metrics_per_var = {v_id: {"d_c": [], "d_m": [], "d_o": [], "chunks": [], "ctx": []} for v_id, _, _ in variants}

    print(f"\n🚀 Evaluating {len(CORPUS)} sequences across 8 update schedules...")
    t_start = time.time()

    for idx, sample in enumerate(tokenized_corpus):
        p_ids, t_ids = sample["prompt"], sample["target"]
        dom = sample["domain"]

        # Run all variants on this sequence
        seq_nlls = {}
        for v_id, _, kwargs in variants:
            nll, dc, dm, do, c_trig, ctx_trig = evaluate_single_sequence(
                params, p_ids, t_ids, buffers, cfg, **kwargs
            )
            seq_nlls[v_id] = nll
            metrics_per_var[v_id]["d_c"].append(dc)
            metrics_per_var[v_id]["d_m"].append(dm)
            metrics_per_var[v_id]["d_o"].append(do)
            metrics_per_var[v_id]["chunks"].append(c_trig)
            metrics_per_var[v_id]["ctx"].append(ctx_trig)

        base_nll = seq_nlls["A"]
        for v_id in seq_nlls:
            delta = seq_nlls[v_id] - base_nll
            results_per_var[v_id].append({
                "seq_idx": idx,
                "domain": dom,
                "nll": seq_nlls[v_id],
                "delta": delta,
            })

    total_time = time.time() - t_start
    print(f"✓ Sweep completed in {total_time:.2f}s ({total_time / (len(CORPUS) * len(variants)):.3f}s per eval)")

    # --------------------------------------------------------------------------
    # STATISTICAL SYNTHESIS ACROSS VARIANTS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("🏆 PHASE 3 ADAPTIVE-UPDATE SWEEP: STATISTICAL PERFORMANCE SYNTHESIS")
    print("=" * 105)
    print(f"{'Variant':<25} | {'Mean NLL':<9} | {'Mean Delta':<11} | {'Median Delta':<13} | {'95% Bootstrap CI':<20} | {'Win Rate':<9} | {'D_mid':<7} | {'D_outer':<7}")
    print("-" * 105)

    summary_table = []
    for v_id, label, _ in variants:
        nlls = [r["nll"] for r in results_per_var[v_id]]
        deltas = [r["delta"] for r in results_per_var[v_id]]
        mean_nll = float(np.mean(nlls))
        mean_delta = float(np.mean(deltas))
        median_delta = float(np.median(deltas))
        ci_low, ci_high = compute_bootstrap_ci(deltas)
        wins = sum(1 for d in deltas if d < -1e-6)
        win_rate = (wins / len(deltas)) * 100.0 if v_id != "A" else 0.0

        mean_dm = float(np.mean(metrics_per_var[v_id]["d_m"]))
        mean_do = float(np.mean(metrics_per_var[v_id]["d_o"]))

        delta_str = f"{mean_delta:+.4f}" if v_id != "A" else "0.0000 (Base)"
        med_delta_str = f"{median_delta:+.4f}" if v_id != "A" else "0.0000"
        ci_str = f"[{ci_low:+.4f}, {ci_high:+.4f}]" if v_id != "A" else "[0.0000, 0.0000]"
        win_str = f"{win_rate:5.1f}%" if v_id != "A" else "Base"

        print(f"{label:<25} | {mean_nll:<9.4f} | {delta_str:<11} | {med_delta_str:<13} | {ci_str:<20} | {win_str:<9} | {mean_dm:<7.4f} | {mean_do:<7.4f}")
        summary_table.append({
            "id": v_id, "label": label, "mean_nll": mean_nll, "mean_delta": mean_delta,
            "median_delta": median_delta, "ci": (ci_low, ci_high), "win_rate": win_rate,
            "d_mid": mean_dm, "d_outer": mean_do
        })

    # --------------------------------------------------------------------------
    # DOMAIN-BY-DOMAIN BREAKDOWN
    # --------------------------------------------------------------------------
    domains = ["Code", "Language", "Science", "Math/Physics"]
    print("\n" + "=" * 105)
    print("🔬 DOMAIN-SPECIFIC PAIRED DELTA NLL BREAKDOWN (Mean Delta vs Baseline A)")
    print("=" * 105)
    print(f"{'Variant':<25} | " + " | ".join([f"{d:<15}" for d in domains]))
    print("-" * 105)

    for v_id, label, _ in variants:
        if v_id == "A":
            dom_strs = [f"{0.0:+15.4f}" for _ in domains]
        else:
            dom_strs = []
            for d in domains:
                d_deltas = [r["delta"] for r in results_per_var[v_id] if r["domain"] == d]
                m_del = float(np.mean(d_deltas)) if d_deltas else 0.0
                dom_strs.append(f"{m_del:+15.4f}")
        print(f"{label:<25} | " + " | ".join(dom_strs))

    print("\n" + "=" * 105)
    print("✅ SWEEP COMPLETE: EMPIRICAL EVIDENCE READY FOR STRATEGIC VERIFICATION")
    print("=" * 105)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Sequence Adaptive Frequency Sweep")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_small_latest.pt")
    parser.add_argument("--seq_len", type=int, default=32)
    args = parser.parse_args()

    run_sweep(checkpoint_path=args.checkpoint, seq_len=args.seq_len)
