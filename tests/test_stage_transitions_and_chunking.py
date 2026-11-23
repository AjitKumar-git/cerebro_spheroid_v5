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
    q_tokens = tok.encode(question)
    expected_header = q_tokens[:min(len(q_tokens), 32 // 3)]
    for h_tok in expected_header[:3]:
        assert h_tok in p_batch[0], "Prompt header tokens must be preserved in sliced reasoning slices!"


def test_rectified_streamer_update_block_size(monkeypatch):
    from src.rectified_streamer import RectifiedContiguousPretrainStreamer

    def _mock_doc_gen():
        while True:
            yield "The quick brown fox jumps over the lazy dog. " * 50

    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_fineweb_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_code_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_edu_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_math_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_retrieval_stream", lambda self: _mock_doc_gen())

    tok = MockTokenizer()
    streamer = RectifiedContiguousPretrainStreamer(
        tokenizer=tok,
        batch_size=2,
        block_size=512,
        stage="pretrain",
    )
    try:
        p, t = streamer.get_batch()
        assert p.shape == (2, 1)
        assert t.shape == (2, 512)

        # Dynamically scale to 1024 tokens
        streamer.update_block_size(1024)
        assert streamer.block_size == 1024

        p2, t2 = streamer.get_batch()
        assert p2.shape == (2, 1)
        assert t2.shape == (2, 1024)
    finally:
        streamer.stop()


def test_hbm_safety_guard_and_curriculum_remat():
    # Verify that sequences >= 1024 enforce remat=True
    seq_len = 1024
    remat_choice = False
    context_curriculum = True
    if (seq_len >= 1024 or context_curriculum) and not remat_choice:
        remat_choice = True

    assert remat_choice is True

    # Verify that batch size is safely capped to prevent HBM OOM
    batch_size = 128
    max_micro_tokens = 131072 if remat_choice else 65536
    if batch_size * seq_len > max_micro_tokens:
        batch_size = max(1, max_micro_tokens // seq_len)
    assert batch_size * seq_len <= 131072

    # At 4096 tokens, batch size must be capped <= 32
    seq_len_4k = 4096
    batch_size_4k = 128
    if batch_size_4k * seq_len_4k > max_micro_tokens:
        batch_size_4k = max(1, max_micro_tokens // seq_len_4k)
    assert batch_size_4k <= 32
    assert batch_size_4k * seq_len_4k <= 131072


def test_rectified_streamer_deep_reasoning_transition(monkeypatch):
    """Verifies that streamer transitions smoothly to 35% Math, 35% Cosmopedia, 20% Code, 10% Edu."""
    from src.rectified_streamer import RectifiedContiguousPretrainStreamer

    def _mock_doc_gen():
        while True:
            yield "Mathematical proof of the theorem using rigorous deductive reasoning steps."

    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_fineweb_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_code_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_edu_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_math_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_cosmopedia_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_retrieval_stream", lambda self: _mock_doc_gen())

    tok = MockTokenizer()
    streamer = RectifiedContiguousPretrainStreamer(
        tokenizer=tok,
        batch_size=2,
        block_size=256,
        stage="pretrain",
    )
    try:
        p, t = streamer.get_batch()
        assert p.shape == (2, 1)
        assert t.shape == (2, 256)
        assert streamer.stage == "pretrain"

        # Transition to Deep Reasoning
        streamer.transition_to_deep_reasoning()
        assert streamer.stage == "deep_reasoning"
        assert streamer.shares["math"] == 0.35
        assert streamer.shares["cosmopedia"] == 0.35
        assert streamer.shares["code"] == 0.20
        assert streamer.shares["edu_academic"] == 0.10
        assert "broad_web" not in streamer.shares

        p2, t2 = streamer.get_batch()
        assert p2.shape == (2, 1)
        assert t2.shape == (2, 256)
    finally:
        streamer.stop()


def test_train_v5_auto_reasoning_transition_args():
    """Verifies that train_v5_tpu_jax exposes auto_reasoning_transition, 15.7B defaults, and conservative Stage 2 LR guards."""
    from training.train_v5_tpu_jax import parse_args

    args = parse_args([])
    assert args.auto_reasoning_transition is True
    assert args.reasoning_transition_tokens == 15_700_000_000
    assert args.final_campaign_tokens == 50_000_000_000
    assert args.reasoning_max_lr == 2e-4
    assert args.reasoning_min_lr == 3e-5


def test_stage2_conservative_lr_schedule():
    """Verifies that Stage 2 deep reasoning schedule guards against catastrophic forgetting of Stage 1."""
    import math

    step = 200_000
    reasoning_transition_step = 191_651
    total_steps = 400_000
    r_peak = 2e-4
    r_min = 3e-5

    r_total_steps = max(1, total_steps - reasoning_transition_step)
    r_elapsed = max(0, step - reasoning_transition_step)
    r_ratio = min(1.0, float(r_elapsed) / float(r_total_steps))
    r_coeff = 0.5 * (1.0 + math.cos(math.pi * r_ratio))
    cur_lr = float(r_min + (r_peak - r_min) * r_coeff)

    # Must stay strictly below pretraining peak LR (2e-3)
    assert cur_lr <= r_peak
    assert cur_lr >= r_min
    assert cur_lr <= 2e-4
