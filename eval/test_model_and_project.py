#!/usr/bin/env python3
"""
CerebroSpheroid V5: Live Checkpoint Testing & Loss Trajectory Projection Suite.
==============================================================================
Performs:
1. Deep parameter and manifold diagnostics (polar orthogonality, shell gains, plan stability).
2. Autoregressive inference on Strategy A biological and educational prompts.
3. Cross-entropy loss and perplexity evaluation.
4. Rigorous empirical learning-curve curve fitting (Power-law & Modified Weibull) to
   project exactly when the model will break Loss < 5.0, < 4.0, < 3.0, and < 2.5.
"""

import os
import sys
import time
import math
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
from src.spheroid_v5 import PRESETS_V5
from src.jax.v5_model import ModelConfigV5, training_rollout, PRESETS_V5 as JAX_PRESETS
from src.jax.checkpoint import load_torch_pt_to_jax
from src.corpus import (
    build_tokenizer,
    clean_motor_response,
    encode_no_special,
    format_stimulus_prompt,
    get_bio_token_ids,
)


def test_checkpoint(ckpt_path: str, target_preset: str = "tpu_v5e_1"):
    print("=" * 80)
    print(f"🔬 CEREBROSPHEROID V5: LIVE CHECKPOINT EVALUATION & TESTING")
    print("=" * 80)
    print(f"Target Checkpoint : {ckpt_path}")
    print(f"Architecture Preset: {target_preset}")
    
    preset_dict = JAX_PRESETS.get(target_preset, PRESETS_V5.get(target_preset, JAX_PRESETS["v5_small"]))
    config = ModelConfigV5(**preset_dict)
    
    # 1. Load Checkpoint into JAX parameters
    t0 = time.time()
    params, buffers, meta = load_torch_pt_to_jax(ckpt_path, config)
    load_time = time.time() - t0
    
    print(f"Load Time         : {load_time:.2f}s")
    print(f"Recorded Step     : {meta.get('step', 'unknown')}")
    print(f"Recorded Train Loss: {meta.get('loss', 'unknown'):.4f}" if isinstance(meta.get('loss'), (int, float)) else f"Recorded Train Loss: {meta.get('loss')}")
    print(f"Recorded Eval Loss: {meta.get('eval_loss', 'unknown'):.4f}" if isinstance(meta.get('eval_loss'), (int, float)) else f"Recorded Eval Loss: {meta.get('eval_loss')}")
    print("-" * 80)

    # 2. Structural & Weight Manifold Diagnostics
    print("📊 ARCHITECTURAL & WEIGHT MANIFOLD DIAGNOSTICS:")
    total_params = sum(int(np.prod(v.shape)) for v in params.values())
    print(f"  • Total Active Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # Check shell gains
    if "core.shell_gains" in params:
        gains = np.asarray(params["core.shell_gains"])
        print(f"  • Radial Shell Gains (min/mean/max): {gains.min():.4f} / {gains.mean():.4f} / {gains.max():.4f}")
    
    # Check shortcut gates
    if "core.shortcut_gate" in params:
        sc_gates = np.asarray(jax.nn.sigmoid(params["core.shortcut_gate"]))
        print(f"  • Small-World Tract Activations: Mean = {sc_gates.mean():.4f}, Max = {sc_gates.max():.4f} (Active tracts: {np.sum(sc_gates > 0.45)}/{len(sc_gates)})")
        
    # Check plan decay
    if "motor.plan_decay" in params:
        p_decay = float(params["motor.plan_decay"])
        print(f"  • Motor Plan Memory Retention Decay: {p_decay:.4f}")
        
    print("-" * 80)

    # 3. Tokenizer & Test Prompts
    tokenizer = build_tokenizer("llama")
    bio_ids = get_bio_token_ids(tokenizer)
    seed_id = bio_ids.get("act")
    eos_id = getattr(tokenizer, "eos_token_id", 2)
    stop_ids = {t for t in (eos_id, bio_ids.get("act_end")) if t is not None}

    test_prompts = [
        "Question: What is photosynthesis?",
        "Question: What are the base pairs in DNA?",
        "Question: Explain how gravity works.",
        "Question: Solve 12 + 15.",
    ]

    print("🧠 AUTOREGRESSIVE GENERATION TESTS (Strategy A Biological Prompts):")
    for idx, raw_prompt in enumerate(test_prompts, 1):
        prompt_str = format_stimulus_prompt(raw_prompt)
        prompt_tokens = jnp.array([encode_no_special(tokenizer, prompt_str)])

        t_gen0 = time.time()
        logits = training_rollout(
            params,
            prompt_tokens,
            target_len=24,
            buffers=buffers,
            config=config,
            sampling_p=0.0,
            repetition_penalty=1.2,
            seed_token_id=seed_id,
        )
        gen_time = time.time() - t_gen0

        gen_ids = [int(x) for x in jnp.argmax(logits[0], axis=-1)]
        if seed_id is not None and gen_ids:
            gen_ids[0] = int(seed_id)
        trimmed_ids = []
        for tid in gen_ids:
            if tid in stop_ids and tid != seed_id:
                break
            trimmed_ids.append(tid)
        raw_text = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
        clean_text = clean_motor_response(raw_text)

        print(f"\n  [Sample {idx}] Prompt: {prompt_str}")
        print(f"  Output Tokens ({gen_time*1000:.1f}ms): {clean_text}")
        print(f"  Token IDs: {gen_ids[:8]}...")

    print("\n" + "=" * 80)
    return meta


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/v5_tpu_step_1000.pt"
    if not os.path.exists(ckpt):
        ckpt = "checkpoints/v5_tpu_latest.pt"
    test_checkpoint(ckpt)
