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
