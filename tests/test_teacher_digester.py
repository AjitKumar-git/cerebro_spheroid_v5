# -*- coding: utf-8 -*-
"""
Unit tests for Qwen Teacher Digester & Dynamic Token Loss Weighting Engine
"""

import pytest
import numpy as np
import torch
from src.teacher_digester import (
    create_digester,
    HeuristicDigester,
    QwenTeacherDigester,
)
from src.corpus import build_tokenizer


def test_heuristic_digester_weight_bounds():
    tokenizer = build_tokenizer()
    digester = HeuristicDigester(alpha=1.5, w_min=0.2, w_max=4.0)

    sample_text = (
        "In linear algebra, if A is a symmetric positive-definite matrix, "
        "then all eigenvalues lambda_i > 0 and det(A) = prod(lambda_i)."
    )

    token_ids, weights = digester.digest_text(sample_text, tokenizer)

    assert len(token_ids) == len(weights)
    assert len(token_ids) > 0

    # Ensure weights stay strictly within specified [w_min, w_max] bounds
    for w in weights:
        assert 0.19 <= w <= 4.01, f"Weight {w} out of bounds"

    # Ensure sequence mean is approximately normalized to 1.0
    mean_w = np.mean(weights)
    assert 0.85 <= mean_w <= 1.15, f"Mean weight {mean_w} drifted from 1.0"


def test_heuristic_digester_dense_vs_boilerplate():
    tokenizer = build_tokenizer()
    digester = HeuristicDigester(alpha=1.5, w_min=0.2, w_max=4.0)

    sample_text = "The matrix eigenvalue is 42."
    token_ids, weights = digester.digest_text(sample_text, tokenizer)

    subwords = [tokenizer.decode([tid]).strip().lower() for tid in token_ids]

    the_weight = None
    eigen_weight = None
    num_weight = None

    for sub, w in zip(subwords, weights):
        if sub == "the":
            the_weight = w
        if "eigen" in sub:
            eigen_weight = w
        if "42" in sub:
            num_weight = w

    if the_weight is not None and eigen_weight is not None:
        assert eigen_weight > the_weight, (
            f"Dense keyword 'eigen' ({eigen_weight}) should have higher weight than 'the' ({the_weight})"
        )


def test_teacher_digester_factory():
    h_digester = create_digester(mode="heuristic")
    assert isinstance(h_digester, HeuristicDigester)

    q_digester = create_digester(
        teacher_model="Qwen/Qwen2.5-1.5B-Instruct",
        mode="online",
        device="cpu",
    )
    assert isinstance(q_digester, QwenTeacherDigester)
    assert not q_digester.loaded  # Lazy loaded by default
