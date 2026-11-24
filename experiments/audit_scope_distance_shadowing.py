#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Scope Distance and Shadowing Attribution Probe
Tests recurrence benefit as a function of:
1. Reference distance (Short: 7 tokens vs Long: 30 tokens with intervening vars).
2. Lexical identity vs intervening variable binding (a, b, c, d).
3. Variable shadowing and nested scope resolution (inner x vs outer x).
"""

import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import jax
import jax.numpy as jnp
from src.spheroid_v5 import infer_v5_config_from_state_dict
from src.jax.v5_model import ModelConfigV5, encode_sensory, emit_step, deliberate_cycles_jax, deliberate_step
from src.jax.checkpoint import load_torch_pt_to_jax

def run_probe(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_latest.pt")
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

    cases = [
        ("Short Distance", "x = 10\nreturn x\n"),
        ("Long Distance (Intervening)", "x = 10\na = 2\nb = 3\nc = a + b\nd = c * 2\nreturn x\n"),
        ("Nested Scope Shadowing", "x = 10\ndef inner():\n    x = 20\n    return x\nreturn x\n")
    ]

    def eval_sequence(text, recurrent=False):
        tok_ids = tokenizer.encode(text)
        if tok_ids[0] == 1:
            # prompt is BOS (id=1), targets are tokens[1:]
            prompt = jnp.array([[1]], dtype=jnp.int32)
            targets = jnp.array([tok_ids[1:]], dtype=jnp.int32)
        else:
            prompt = jnp.array([[1]], dtype=jnp.int32)
            targets = jnp.array([tok_ids], dtype=jnp.int32)

        B, total_L = targets.shape
        p_flux, p_kv = encode_sensory(params, prompt, config=cfg, pos_offset=0)
        r_prof = (jnp.arange(1, K + 1, dtype=jnp.float32) / float(K))[None, :, None]
        init_state = (jnp.broadcast_to(p_flux, (B, K, D)).astype(jnp.float32) * r_prof).astype(jnp.float32)
        state, _ = deliberate_cycles_jax(params, init_state, buffers, config=cfg, is_training=False, sensory_kv=p_kv)

        prev_toks = jnp.concatenate([prompt[:, -1:], targets[:, :-1]], axis=1)
        tf_flux, (tf_k, tf_v) = encode_sensory(params, prev_toks, config=cfg, pos_offset=prompt.shape[1], is_causal=True)
        causal_k = jnp.concatenate([p_kv[0], tf_k], axis=2)
        causal_v = jnp.concatenate([p_kv[1], tf_v], axis=2)

        cur_state = state
        cur_plan = jnp.zeros((B, D), dtype=jnp.float32)
        cur_trace = None

        step_nlls = []

        for t in range(total_L):
            curr_ctx_len = prompt.shape[1] + t + 1
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
                config=cfg,
                auditory_feedback=tf_flux[:, t, :],
                query_position=prompt.shape[1] + t,
            )

            tgt = targets[:, t]
            log_p = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            token_nll = -float(log_p[0, tgt[0]])
            step_nlls.append(token_nll)

            if recurrent:
                # Surprisal-gated intermediate recurrence (Mode C)
                probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
                entropy = float(-jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1)[0])
                g_act = 1.0 / (1.0 + np.exp(-(entropy - 3.2)))
                delib = deliberate_step(params, nxt_state, buffers, config=cfg)
                cur_state = nxt_state.at[:, 1:12, :].set(
                    nxt_state[:, 1:12, :] + 0.15 * g_act * (delib[:, 1:12, :] - nxt_state[:, 1:12, :])
                )
            else:
                cur_state = nxt_state

        target_tokens = [tokenizer.decode([tid]) for tid in targets[0]]
        return target_tokens, step_nlls

    print("=" * 80)
    print("🔬 SCOPE DISTANCE & SHADOWING RECURRENCE AUDIT")
    print("=" * 80)

    for name, text in cases:
        tokens, base_nlls = eval_sequence(text, recurrent=False)
        _, recurr_nlls = eval_sequence(text, recurrent=True)

        print(f"\n📂 Case: {name}")
        print(f"{'Idx':<4} | {'Token':<12} | {'Base NLL':<10} | {'Recurr NLL':<10} | {'Delta NLL':<10}")
        print("-" * 55)
        for i, (tok, b, r) in enumerate(zip(tokens, base_nlls, recurr_nlls)):
            delta = r - b
            print(f"{i:<4d} | {repr(tok):<12} | {b:<10.4f} | {r:<10.4f} | {delta:<+10.4f}")

if __name__ == "__main__":
    run_probe()
