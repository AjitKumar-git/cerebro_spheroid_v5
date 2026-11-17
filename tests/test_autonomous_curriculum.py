#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Refined Autonomous Training Curriculum & Dataset Streamer
===================================================================
Verifies document quality filters, prompt/response extraction,
and autonomous curriculum iteration.
"""

import pytest
from src.elastic_window import (
    extract_prompt_response_pair,
    get_default_elastic_buckets,
    select_best_bucket,
    build_online_dataset_iterator,
)


def test_elastic_buckets_configuration():
    buckets = get_default_elastic_buckets(1024)
    assert len(buckets) >= 3
    for name, bp, bt in buckets:
        assert bp + bt <= 1024 + 256
        assert bp >= 128
        assert bt >= 256


def test_select_best_bucket():
    buckets = [
        ("standard_cot", 256, 512),
        ("deep_reasoning", 256, 768),
        ("balanced", 384, 512),
    ]
    # Short prompt, long target -> deep reasoning bucket
    b_idx = select_best_bucket(150, 600, buckets)
    assert buckets[b_idx][0] == "deep_reasoning"

    # Standard prompt & target -> standard_cot
    b_idx2 = select_best_bucket(200, 400, buckets)
    assert buckets[b_idx2][0] == "standard_cot"


def test_extract_prompt_response_pair_3tier():
    # Record with thinking tag
    row = {
        "problem": "What is 2 + 2?",
        "thought": "2 + 2 is an elementary addition problem equal to 4.",
        "solution": "4",
    }
    prompt, resp = extract_prompt_response_pair(row)
    assert "2 + 2" in prompt
    assert "<think>" in resp
    assert "elementary addition" in resp
    assert "4" in resp


def test_extract_prompt_response_pair_conversations():
    row = {
        "conversations": [
            {"from": "human", "value": "Explain quantum entanglement simply."},
            {"from": "gpt", "value": "Quantum entanglement is a phenomenon where particles remain connected."},
        ]
    }
    prompt, resp = extract_prompt_response_pair(row)
    assert "quantum entanglement" in prompt.lower()
    assert "particles remain connected" in resp.lower()


def test_offline_custom_dataset_schema():
    import pyarrow.parquet as pq
    import os

    path = "data/cerebro_v5_custom_dataset/train.parquet"
    if os.path.exists(path):
        table = pq.read_table(path)
        assert table.num_rows > 25000
        cols = table.column_names
        for expected in ["id", "category", "thinking_level", "prompt", "thinking", "action", "answer", "formatted_input", "formatted_output"]:
            assert expected in cols


def test_canonical_eval_split_5pillars_alignment(monkeypatch):
    import numpy as np
    import src.elastic_window
    from src.elastic_window import build_canonical_eval_split

    # Fast offline execution via zero-latency fallback
    monkeypatch.setattr(src.elastic_window, "_open_hf_stream", lambda *args, **kwargs: iter([]))

    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2

        def encode(self, text, max_length=None):
            words = text.split()
            toks = [ord(w[0]) % 500 + 10 for w in words]
            return toks[:max_length] if max_length else toks

    tok = MockTokenizer()
    eval_batches, eval_domains = build_canonical_eval_split(
        tokenizer=tok,
        batch_size=4,
        prompt_len=128,
        target_len=256,
        target_pad_id=-100,
        pad_id=0,
    )

    # Must contain exactly 5 aligned curriculum domains
    assert len(eval_domains) == 5
    assert "Educational Prose (FineWeb-Edu)" in eval_domains[0]
    assert "Textbook Concepts (Cosmopedia v2)" in eval_domains[1]
    assert "Mathematical Proofs (FineMath & GSM8K)" in eval_domains[2]
    assert "Algorithmic Python (Code Large & Self-OSS)" in eval_domains[3]
    assert "Commonsense Dialogue (SmolTalk)" in eval_domains[4]

    assert len(eval_batches) == 5
    for (q_arr, r_arr, fb_arr), d_name in zip(eval_batches, eval_domains):
        assert q_arr.shape == (4, 128)
        assert r_arr.shape == (4, 256)
        assert fb_arr.shape == (4, 256)
        assert q_arr.dtype == np.int32
        assert r_arr.dtype == np.int32
        assert fb_arr.dtype == np.int32

        # Causal teacher-forcing feedback checks:
        # Step 0 feedback token is always target_pad_id (-100)
        assert np.all(fb_arr[:, 0] == -100)
        # Shifted alignment: fb[:, 1:] equals target[:, :-1] wherever target is unpadded
        unpadded_mask = (r_arr[:, 1:] != -100) & (r_arr[:, :-1] != -100)
        assert np.array_equal(fb_arr[:, 1:][unpadded_mask], r_arr[:, :-1][unpadded_mask])
        # Padded target positions are strictly masked with -100
        assert np.all(fb_arr[r_arr == -100] == -100)


def test_canonical_eval_split_streaming_records(monkeypatch):
    import numpy as np
    import src.elastic_window
    from src.elastic_window import build_canonical_eval_split

    # Mock _open_hf_stream yielding synthetic mock records
    def mock_stream(repo, config_name=None, split="train", **kwargs):
        return iter([
            {"text": "Topic: General relativity and gravitational waves.\n\nGeneral relativity models gravitation as Riemannian curvature of spacetime.", "score": 4.5},
            {"prompt": "Explain thermodynamics.", "text": "Thermodynamics governs heat, work, and entropy in physical systems.", "score": 4.2},
            {"problem": "What is 5 * 6?", "solution": "<think>\nCompute 5 * 6 = 30.\n</think>\n30", "score": 4.8},
            {"instruction": "Write a python function to add two numbers.", "code": "def add(a, b):\n    return a + b"},
            {"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I am doing well, thank you!"}]},
        ])

    monkeypatch.setattr(src.elastic_window, "_open_hf_stream", mock_stream)

    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2

        def encode(self, text, max_length=None):
            words = text.split()
            toks = [ord(w[0]) % 500 + 10 for w in words]
            return toks[:max_length] if max_length else toks

    tok = MockTokenizer()
    eval_batches, eval_domains = build_canonical_eval_split(
        tokenizer=tok,
        batch_size=2,
        prompt_len=64,
        target_len=128,
        target_pad_id=-100,
        pad_id=0,
    )

    assert len(eval_domains) == 5
    assert len(eval_batches) == 5
    for q_arr, r_arr, fb_arr in eval_batches:
        assert q_arr.shape == (2, 64)
        assert r_arr.shape == (2, 128)
        assert fb_arr.shape == (2, 128)
        assert np.all(fb_arr[:, 0] == -100)
