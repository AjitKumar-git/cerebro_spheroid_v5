#!/usr/bin/env python3
"""
🔬 CerebroSpheroid V5: Mechanistic Token-Class Recurrence Analyzer

Audits why Code benefits from recurrence while Natural Language degrades:
1. Classifies tokens into syntactic/structural categories:
   - Code: Indentation/Newlines, Brackets, Keywords, Operators, Identifiers.
   - Language: Punctuation/Sentence End, Capitalized Entities, Function Words, Content Words.
2. Measures Gate Activation Probability P(trigger | Token Class).
3. Measures Paired Delta NLL (Recurrence vs Frozen Baseline) by Token Class.
4. Directly tests the Syntactic Hierarchy Hypothesis vs Token Distribution Hypothesis.
"""

import os
import sys
import math
import argparse
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
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
from experiments.adaptive_frequency_sweep import CORPUS


PYTHON_KEYWORDS = {
    "def", "class", "if", "else", "elif", "return", "for", "while",
    "in", "import", "from", "with", "as", "try", "except", "lambda",
    "yield", "break", "continue", "pass", "None", "True", "False", "and", "or", "not", "is"
}

BRACKETS = {"(", ")", "[", "]", "{", "}", "):", "]:", "}:", "()"}
PUNCTUATION = {".", ",", ";", ":", "!", "?", "--", "---", "\"", "'"}


def classify_token(piece: str, domain: str) -> str:
    """Classifies a decoded token piece into a syntactic/semantic category."""
    p_strip = piece.strip()
    if domain == "Code":
        if ("\n" in piece) or (piece.startswith("  ") or piece == "\t"):
            return "Indentation/Newline"
        if any(b in piece for b in BRACKETS):
            return "Bracket/Delimiter"
        if p_strip in PYTHON_KEYWORDS:
            return "Keyword"
        if any(op in piece for op in ["=", "+", "-", "*", "/", "<", ">", "%", "&", "|", "^"]):
            return "Operator/Comparison"
        if p_strip.isidentifier():
            return "Identifier/Variable"
        return "Other Code Token"
    else:
        # Natural Language / Science / Math
        if any(punct in piece for punct in PUNCTUATION):
            return "Punctuation/Boundary"
        if piece.strip() and piece.strip()[0].isupper():
            return "Capitalized / Entity"
        if len(p_strip) <= 3 and p_strip.lower() in {"the", "a", "an", "in", "on", "at", "of", "to", "for", "is", "are", "was", "and", "or", "by"}:
            return "Function Word"
        return "Content Word / Term"


def audit_token_class_mechanisms(
    checkpoint_path: str = "checkpoints/v5_small_latest.pt",
    seq_len: int = 32,
    adaptive_threshold: float = 3.2,
):
    print("=" * 95)
    print("🔬 CEREBRO-SPHEROID V5: TOKEN-CLASS RECURRENCE & SYNTACTIC MECHANISM AUDIT")
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
    K = cfg.num_shells
    D = cfg.surface_dim

    # Accumulators by domain and token class
    # domain -> class -> list of dicts: {"piece": str, "nll_base": float, "nll_recurr": float, "delta": float, "triggered": bool}
    class_stats = defaultdict(lambda: defaultdict(list))

    print(f"⏳ Analyzing token-level mechanics across {len(CORPUS)} sequences...")

    for sample in CORPUS:
        dom = sample["domain"]
        toks = tokenizer.encode(sample["text"])[:seq_len]
        if len(toks) < 8:
            continue

        target_ids = jnp.array([toks], dtype=jnp.int32)
        prompt_ids = jnp.array([[1]], dtype=jnp.int32)
        total_L = target_ids.shape[1]
        prompt_len = prompt_ids.shape[1]

        # 1. Prompt deliberation
        p_flux, p_kv = encode_sensory(params, prompt_ids, config=cfg, pos_offset=0)
        r_prof = (jnp.arange(1, K + 1, dtype=jnp.float32) / float(K))[None, :, None]
        init_state = (jnp.broadcast_to(p_flux, (1, K, D)).astype(jnp.float32) * r_prof).astype(jnp.float32)
        state, _ = deliberate_cycles_jax(params, init_state, buffers, config=cfg, is_training=False, sensory_kv=p_kv)

        # 2. Causal sensory stream
        prev_toks = jnp.concatenate([prompt_ids[:, -1:], target_ids[:, :-1]], axis=1)
        tf_flux, (tf_k, tf_v) = encode_sensory(params, prev_toks, config=cfg, pos_offset=prompt_len, is_causal=True)
        causal_k = jnp.concatenate([p_kv[0], tf_k], axis=2)
        causal_v = jnp.concatenate([p_kv[1], tf_v], axis=2)

        # Evaluate both Frozen Baseline (A) and Adaptive Recurrent (C) step-by-step
        s_base = state
        s_recurr = state
        plan_base = jnp.zeros((1, D), dtype=jnp.float32)
        plan_recurr = jnp.zeros((1, D), dtype=jnp.float32)
        tr_base = None
        tr_recurr = None

        chunk_counter = 0
        curr_chunk_len = 0

        for t in range(total_L):
            curr_ctx_len = prompt_len + t + 1
            sub_k = causal_k[:, :, :curr_ctx_len, :]
            sub_v = causal_v[:, :, :curr_ctx_len, :]
            feed = tf_flux[:, t, :]
            tgt_token = int(target_ids[0, t])
            token_piece = tokenizer.decode([tgt_token])
            t_class = classify_token(token_piece, dom)

            # --- Base Forward (A: Frozen) ---
            log_base, nxt_s_base, tr_base, plan_base = emit_step(
                params, s_base, (sub_k, sub_v), plan_base, tr_base, t, buffers, config=cfg,
                auditory_feedback=feed, query_position=prompt_len + t
            )
            lp_b = jax.nn.log_softmax(log_base.astype(jnp.float32), axis=-1)
            nll_b = float(-lp_b[0, tgt_token])
            s_base = nxt_s_base  # Only shell 0 evolves

            # --- Adaptive Forward (C: Adaptive) ---
            log_rec, nxt_s_rec, tr_recurr, plan_recurr = emit_step(
                params, s_recurr, (sub_k, sub_v), plan_recurr, tr_recurr, t, buffers, config=cfg,
                auditory_feedback=feed, query_position=prompt_len + t
            )
            lp_r = jax.nn.log_softmax(log_rec.astype(jnp.float32), axis=-1)
            nll_r = float(-lp_r[0, tgt_token])

            # Adaptive Gate Decision
            probs = jax.nn.softmax(log_rec.astype(jnp.float32), axis=-1)
            entropy = float(-jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1)[0])
            g_act = 1.0 / (1.0 + math.exp(-max(min(entropy - adaptive_threshold, 10.0), -10.0)))
            curr_chunk_len += 1
            is_trigger = (g_act > 0.5) or (curr_chunk_len >= 16)

            if is_trigger and (t > 0):
                chunk_counter += 1
                curr_chunk_len = 0
                delib = deliberate_step(params, nxt_s_rec, buffers, config=cfg)
                s_recurr = nxt_s_rec.at[:, 1:12, :].set(
                    nxt_s_rec[:, 1:12, :] + 0.15 * g_act * (delib[:, 1:12, :] - nxt_s_rec[:, 1:12, :])
                )
            else:
                s_recurr = nxt_s_rec

            delta = nll_r - nll_b
            class_stats[dom][t_class].append({
                "piece": token_piece,
                "nll_base": nll_b,
                "nll_recurr": nll_r,
                "delta": delta,
                "triggered": is_trigger,
            })

    # --------------------------------------------------------------------------
    # REPORTING: CODE SYNTACTIC BREAKDOWN
    # --------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("💻 PYTHON CODE: SYNTACTIC TOKEN-CLASS RECURRENCE BREAKDOWN")
    print("=" * 105)
    print(f"{'Token Class':<24} | {'Count':<7} | {'Gate Fire %':<12} | {'Base NLL':<10} | {'Recurr NLL':<10} | {'Mean Delta (Delta NLL)':<20} | {'Mechanistic Impact'}")
    print("-" * 105)

    code_classes = sorted(class_stats["Code"].keys(), key=lambda c: len(class_stats["Code"][c]), reverse=True)
    for c in code_classes:
        records = class_stats["Code"][c]
        count = len(records)
        fire_pct = (sum(1 for r in records if r["triggered"]) / count) * 100.0
        m_base = float(np.mean([r["nll_base"] for r in records]))
        m_rec = float(np.mean([r["nll_recurr"] for r in records]))
        m_delta = float(np.mean([r["delta"] for r in records]))

        impact = "STRONG IMPROVEMENT" if m_delta < -0.05 else ("IMPROVEMENT" if m_delta < 0.0 else "DEGRADATION")
        print(f"{c:<24} | {count:<7} | {fire_pct:<11.1f}% | {m_base:<10.4f} | {m_rec:<10.4f} | {m_delta:<+20.4f} | {impact}")

    # --------------------------------------------------------------------------
    # REPORTING: NATURAL LANGUAGE & PROSE BREAKDOWN
    # --------------------------------------------------------------------------
    for dom in ["Language", "Science", "Math/Physics"]:
        print("\n" + "=" * 105)
        print(f"📖 {dom.upper()}: TOKEN-CLASS RECURRENCE BREAKDOWN")
        print("=" * 105)
        print(f"{'Token Class':<24} | {'Count':<7} | {'Gate Fire %':<12} | {'Base NLL':<10} | {'Recurr NLL':<10} | {'Mean Delta (Delta NLL)':<20} | {'Mechanistic Impact'}")
        print("-" * 105)

        dom_classes = sorted(class_stats[dom].keys(), key=lambda c: len(class_stats[dom][c]), reverse=True)
        for c in dom_classes:
            records = class_stats[dom][c]
            count = len(records)
            fire_pct = (sum(1 for r in records if r["triggered"]) / count) * 100.0
            m_base = float(np.mean([r["nll_base"] for r in records]))
            m_rec = float(np.mean([r["nll_recurr"] for r in records]))
            m_delta = float(np.mean([r["delta"] for r in records]))

            impact = "STRONG IMPROVEMENT" if m_delta < -0.05 else ("IMPROVEMENT" if m_delta < 0.0 else "DEGRADATION")
            print(f"{c:<24} | {count:<7} | {fire_pct:<11.1f}% | {m_base:<10.4f} | {m_rec:<10.4f} | {m_delta:<+20.4f} | {impact}")

    print("\n" + "=" * 105)
    print("✅ MECHANISTIC ANALYSIS COMPLETE: HARD TOKEN-LEVEL EVIDENCE COMPILED")
    print("=" * 105)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mechanistic Token-Class Analyzer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_small_latest.pt")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=3.2)
    args = parser.parse_args()

    audit_token_class_mechanisms(
        checkpoint_path=args.checkpoint, seq_len=args.seq_len, adaptive_threshold=args.threshold
    )
