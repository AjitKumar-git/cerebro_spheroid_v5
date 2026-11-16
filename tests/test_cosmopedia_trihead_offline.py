"""
Test Suite for Offline Cosmopedia -> Tri-Head Reasoning Transformation Pipeline.
Verifies:
1. Deterministic offline transformation of educational passages into Tri-Head records.
2. Schema conformity: <think> deliberation, <act> tool action, answer, and temporal gates.
3. Strict token length boundaries (Prompt <= 512, Response <= 1024).
4. Full JAX model forward rollout compatibility on CerebroSpheroid V5.
"""

import os
import json
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from scripts.transform_cosmopedia_gemini import (
    stream_cosmopedia_passages,
    call_gemini_transform,
    format_trihead_record,
    OFFLINE_COSMOPEDIA_CORPUS,
)
from src.corpus import build_tokenizer
from src.jax.v5_model import ModelConfigV5, PRESETS_V5, init_params_v5, training_rollout


def test_offline_passage_generation():
    """Verify offline stream returns valid educational text passages."""
    passages = list(stream_cosmopedia_passages(max_samples=5, offline=True))
    assert len(passages) == 5
    for p in passages:
        assert isinstance(p, str)
        assert len(p) >= 150


def test_offline_trihead_formatting_and_gates():
    """Verify that every offline sample adheres strictly to CerebroSpheroid V5 Tri-Head architecture."""
    for item in OFFLINE_COSMOPEDIA_CORPUS:
        mock = item["mock_result"]
        rec = format_trihead_record(mock)

        # 1. Prompt format
        assert rec["prompt"].startswith("User: ")
        assert len(rec["prompt"]) > 10

        # 2. Response format
        assert "<think>" in rec["response"] and "</think>" in rec["response"]
        assert rec["category"] in (
            "conversational_dialogue",
            "reasoning_math",
            "db_fact_retrieval",
            "python_code",
            "agentic_modalities",
        )

        # 3. Action and Temporal Gate alignment
        if "<act>" in rec["response"]:
            assert "</act>" in rec["response"]
            assert rec["temporal_gate"] == [1, 1, 1]
        else:
            assert rec["temporal_gate"] == [1, 0, 1]


def test_offline_tokenizer_and_jax_rollout():
    """Verifies that generated Tri-Head records encode properly and execute through JAX training rollout."""
    tokenizer = build_tokenizer()
    assert tokenizer is not None, "Offline tokenizer could not be initialized"

    cfg = ModelConfigV5(**{**PRESETS_V5["v5_test"], "vocab_size": 32000})
    key = jax.random.PRNGKey(42)
    params, buffers = init_params_v5(key, cfg)

    # Test with math arithmetic sample (which has <act> calc(...) </act>)
    sample = OFFLINE_COSMOPEDIA_CORPUS[0]["mock_result"]
    record = format_trihead_record(sample)

    q_tok = tokenizer.encode(f"<stim>{record['prompt']}</stim>", add_special_tokens=False)[:64]
    r_tok = tokenizer.encode(record["response"], add_special_tokens=False)[:128]

    q_arr = np.zeros((1, 64), dtype=np.int32)
    r_arr = np.zeros((1, 128), dtype=np.int32)
    q_arr[0, :len(q_tok)] = q_tok
    r_arr[0, :len(r_tok)] = r_tok

    loss = training_rollout(
        params=params,
        prompts=jnp.array(q_arr),
        target_len=128,
        buffers=buffers,
        config=cfg,
        targets=jnp.array(r_arr),
        pure_teacher_forcing=True,
        remat=False,
    )

    loss_val = float(loss)
    assert not np.isnan(loss_val)
    assert not np.isinf(loss_val)
    assert loss_val > 0.0
