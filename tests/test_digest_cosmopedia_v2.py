#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Cosmopedia v2 Teacher Digestion Pipeline
===================================================
Verifies shard partitioning, schema construction, argument parsing,
and dataset card formatting.
"""

import pytest
import pyarrow as pa
import numpy as np
from scripts.digest_cosmopedia_v2 import COSMOPEDIA_DATASET_CARD_TEMPLATE


def test_cosmopedia_dataset_card_metadata():
    assert "cosmopedia-v2" in COSMOPEDIA_DATASET_CARD_TEMPLATE
    assert "target_weights" in COSMOPEDIA_DATASET_CARD_TEMPLATE
    assert "Qwen2.5-1.5B-Instruct" in COSMOPEDIA_DATASET_CARD_TEMPLATE
    assert "CerebroSpheroid" in COSMOPEDIA_DATASET_CARD_TEMPLATE


def test_fleet_worker_partitioning():
    """Ensure worker sharding perfectly covers all 104 shards with 0 overlap."""
    total_shards = 104
    num_workers = 4
    covered_shards = []

    for worker_id in range(num_workers):
        worker_shards = [s for s in range(total_shards) if s % num_workers == worker_id]
        covered_shards.extend(worker_shards)
        assert len(worker_shards) == total_shards // num_workers

    assert len(covered_shards) == total_shards
    assert sorted(covered_shards) == list(range(total_shards))


def test_digested_record_schema():
    """Verify PyArrow schema and data types for digested Cosmopedia shards."""
    schema = pa.schema([
        ("id", pa.string()),
        ("prompt", pa.string()),
        ("audience", pa.string()),
        ("format", pa.string()),
        ("tokens", pa.list_(pa.int32())),
        ("target_weights", pa.list_(pa.float16())),
        ("seq_len", pa.int32()),
        ("mean_weight", pa.float32()),
        ("max_weight", pa.float32()),
        ("min_weight", pa.float32()),
    ])

    test_tokens = [101, 2045, 1996, 102]
    test_weights = np.array([1.2, 0.4, 1.8, 0.6], dtype=np.float16)

    record = {
        "id": "cosmo2_00000_0000001",
        "prompt": "Explain calculus",
        "audience": "college_students",
        "format": "textbook",
        "tokens": test_tokens,
        "target_weights": [float(w) for w in test_weights],
        "seq_len": len(test_tokens),
        "mean_weight": float(test_weights.mean()),
        "max_weight": float(test_weights.max()),
        "min_weight": float(test_weights.min()),
    }

    table = pa.Table.from_pylist([record], schema=schema)
    assert table.num_rows == 1
    assert table.num_columns == 10
    assert table["tokens"][0].as_py() == test_tokens
