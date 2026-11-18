"""
Unit tests verifying functional context stage transitions and non-fragmenting long reasoning chunking.
"""

import collections
import pytest
import numpy as np

from src.elastic_window import (
    ElasticBucketBatcher,
    get_default_elastic_buckets,
    select_best_bucket,
)


class MockTokenizer:
    def __init__(self, vocab_size=32000):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.vocab_size = vocab_size

    def encode(self, text, max_length=None, **kwargs):
        # Deterministic dummy tokenization: 1 token per word
        words = text.split()
        ids = [abs(hash(w)) % (self.vocab_size - 10) + 10 for w in words]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids)


def test_elastic_batcher_dynamic_update_buckets():
    tok = MockTokenizer()
    initial_buckets = get_default_elastic_buckets(512)
    rows = [
        {"stimulus": "What is 2 + 2?", "articulation": "The answer is 4."},
        {"stimulus": "What is 3 + 3?", "articulation": "The answer is 6."},
    ]
    data_iter = iter(rows * 50)
    batcher = ElasticBucketBatcher(
        tokenizer=tok,
        data_iterator=data_iter,
        batch_size=2,
        total_window=512,
        buckets=initial_buckets,
        framework="numpy",
    )

    assert len(batcher.buckets) == len(initial_buckets)
    assert batcher.total_window == 512

    # Transition to Stage 1: 1024 tokens
    new_buckets = get_default_elastic_buckets(1024)
    batcher.update_buckets(new_buckets, new_total_window=1024)

    assert batcher.total_window == 1024
    assert len(batcher.buckets) == len(new_buckets)
    assert len(batcher.queues) == len(new_buckets)


def test_reasoning_prompt_header_preservation_across_continuation_slices():
    tok = MockTokenizer()
    # Create an example with a long reasoning trace that exceeds bt=64
    question = "Solve the step-by-step calculus integral problem for f(x)."
    long_reasoning = "Step 1: Expand terms. " * 30 + "Therefore the final result is 42."
    row = {"problem": question, "solution": long_reasoning, "_dataset": "open-r1/OpenR1-Math-220k"}

    custom_buckets = [("test_bucket", 32, 64)]
    batcher = ElasticBucketBatcher(
        tokenizer=tok,
        data_iterator=iter([row] * 20),
        batch_size=1,
        total_window=96,
        buckets=custom_buckets,
        framework="numpy",
    )

    # Pull batch
    b_idx, name, p_batch, t_batch = batcher.next_batch(bucket_idx=0)
    assert p_batch.shape == (1, 32)
    assert t_batch.shape == (1, 64)

    # The prompt batch in continuation slices preserves the question header tokens!
    q_tokens = tok.encode(f"<stim> {question} </stim>")
    expected_header = q_tokens[:min(len(q_tokens), 32 // 3)]
    for h_tok in expected_header[:3]:
        assert h_tok in p_batch[0], "Prompt header tokens must be preserved in sliced reasoning slices!"
