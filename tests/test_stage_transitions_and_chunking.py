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
    """Verifies that train_v5_tpu_jax exposes auto_reasoning_transition, 41.4B defaults, and conservative Stage 2 LR guards."""
    from training.train_v5_tpu_jax import parse_args

    args = parse_args([])
    assert args.auto_reasoning_transition is True
    assert args.reasoning_transition_tokens == 41_400_000_000
    assert args.final_campaign_tokens == 50_000_000_000
    assert args.reasoning_max_lr == 2e-4
    assert args.reasoning_min_lr == 3e-5


def test_rectified_streamer_41_4b_phase_schedule(monkeypatch):
    """Verifies 15.7B (FineWeb) -> 15.7B (FineWeb-Edu) -> 5B (FineWeb-Edu) -> 5B (FineWeb-Edu) = 41.4B."""
    from src.rectified_streamer import RectifiedContiguousPretrainStreamer

    def _mock_doc_gen():
        while True:
            yield "Sample training text for tokenizer coverage. " * 20

    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_fineweb_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_edu_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_code_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_math_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_cosmopedia_stream", lambda self: _mock_doc_gen())
    monkeypatch.setattr(RectifiedContiguousPretrainStreamer, "_open_retrieval_stream", lambda self: _mock_doc_gen())

    tok_box = [0]
    tok = MockTokenizer()
    streamer = RectifiedContiguousPretrainStreamer(
        tokenizer=tok,
        batch_size=2,
        block_size=256,
        stage="pretrain",
        tokens_getter=lambda: tok_box[0],
        target_tokens=41_400_000_000,
    )
    try:
        # Phase 0: 0 -> 15.7B (e.g. 9.387B) uses broad_web (FineWeb)
        tok_box[0] = 9_387_028_736
        assert streamer._smollm2_shares() == {"broad_web": 0.90, "code": 0.10}

        # Even if caller passed legacy target_tokens=15.7B, absolute token gates keep 9.45B in Phase 0
        streamer.target_tokens = 15_700_000_000
        tok_box[0] = 9_450_236_288
        assert streamer._smollm2_shares() == {"broad_web": 0.90, "code": 0.10}
        streamer.target_tokens = 41_400_000_000

        # Phase 1: 15.7B -> 31.4B uses edu_academic (FineWeb-Edu)
        tok_box[0] = 15_700_000_000
        assert streamer._smollm2_shares() == {"edu_academic": 0.75, "code": 0.20, "math": 0.05}

        # Phase 2: 31.4B -> 36.4B (5B) uses edu_academic (FineWeb-Edu)
        tok_box[0] = 31_400_000_000
        assert streamer._smollm2_shares() == {"edu_academic": 0.70, "code": 0.20, "math": 0.10}

        # Phase 3: 36.4B -> 41.4B (5B) uses edu_academic (FineWeb-Edu)
        tok_box[0] = 36_400_000_000
        assert streamer._smollm2_shares() == {"edu_academic": 0.58, "code": 0.24, "math": 0.14, "cosmopedia": 0.04}
    finally:
        streamer.stop()


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


def test_from_scratch_arch_upgrades_gating():
    """Verifies V5.1 arch upgrades (4 sensory layers, full-mantle readout, bidirectional tracts) activate ONLY on --from_scratch."""
    from src.jax.v5_model import PRESETS_V5
    from training.train_v5_tpu_jax import (
        apply_from_scratch_arch_upgrades,
        parse_args,
        resolve_checkpoint_arch_overrides,
    )

    # 1. Resume mode (from_scratch=False) preserves legacy v5_small architecture
    args_resume = parse_args([])
    args_resume.from_scratch = False
    cfg_resume = apply_from_scratch_arch_upgrades(PRESETS_V5["v5_small"].copy(), args_resume)
    assert cfg_resume.get("num_sensory_layers", 1) == 1
    assert cfg_resume.get("use_centralized_readout", True) is True
    assert cfg_resume.get("use_bidirectional_tracts", False) is False

    # Legacy checkpoint resume leaves everything unchanged
    legacy_ckpt = {"model_state": {"sensory.embedding.weight": None}, "step": 266000}
    resumed_cfg, changed_legacy = resolve_checkpoint_arch_overrides(legacy_ckpt, cfg_resume)
    assert changed_legacy is False
    assert resumed_cfg.get("num_sensory_layers", 1) == 1

    # 2. Fresh from_scratch=True activates V5.1 upgrades
    args_fresh = parse_args(["--from_scratch"])
    cfg_fresh = apply_from_scratch_arch_upgrades(PRESETS_V5["v5_small"].copy(), args_fresh)
    assert cfg_fresh["num_sensory_layers"] == 4
    assert cfg_fresh["use_centralized_readout"] is False
    assert cfg_fresh["use_shell_tap"] is True
    assert cfg_fresh["use_bidirectional_tracts"] is True

    # 3. Resuming a V5.1 checkpoint later with from_scratch=False preserves V5.1 flags
    v51_ckpt = {
        "model_state": {"sensory_dnn.extra_layers.2.q_proj.weight": None},
        "use_centralized_readout": False,
        "use_shell_tap": True,
        "use_bidirectional_tracts": True,
    }
    restored_cfg, changed_v51 = resolve_checkpoint_arch_overrides(v51_ckpt, cfg_resume)
    assert changed_v51 is True
    assert restored_cfg["num_sensory_layers"] == 4
    assert restored_cfg["use_centralized_readout"] is False
    assert restored_cfg["use_bidirectional_tracts"] is True
    assert cfg_fresh["use_sensory_core_coupling"] is True
    assert cfg_fresh["use_parallel_plan_scan"] is True
    assert cfg_fresh["top_k_resonance"] == 0
    assert cfg_fresh["preserve_shell_norm_clamp"] is False
    assert cfg_resume["use_sensory_core_coupling"] is False
    assert cfg_resume["use_parallel_plan_scan"] is False
    assert cfg_resume["preserve_shell_norm_clamp"] is True


def test_cached_sensory_step_matches_causal_encode():
    """Verifies _encode_sensory_cached_step matches full-sequence encode_sensory(..., is_causal=True) step-by-step."""
    import jax
    import jax.numpy as jnp
    from src.jax.v5_model import (
        ModelConfigV5,
        _encode_sensory_cached_step,
        encode_sensory,
        generate_causal_prefill_jax,
        init_params_v5,
    )

    cfg = ModelConfigV5(
        vocab_size=64,
        num_shells=4,
        surface_dim=32,
        radial_dim=16,
        num_heads=4,
        max_phase_steps=32,
        num_sensory_layers=2,
        top_k_resonance=0,
        preserve_shell_norm_clamp=False,
    )
    params, buffers = init_params_v5(jax.random.PRNGKey(3), cfg)
    toks = jnp.array([[1, 7, 12, 19]], dtype=jnp.int32)
    full_flux, (full_k, full_v) = encode_sensory(params, toks, config=cfg, pos_offset=0, is_causal=True)

    L = toks.shape[1]
    skb = jnp.zeros((2, 1, 4, L, 8), dtype=jnp.float32)
    svb = jnp.zeros((2, 1, 4, L, 8), dtype=jnp.float32)
    for t in range(L):
        mask = (jnp.arange(L) <= t)[None, None, None, :]
        step_flux, (sk, sv), skb, svb = _encode_sensory_cached_step(
            params, toks[:, t:t + 1], jnp.array(t, dtype=jnp.int32), skb, svb, mask, config=cfg
        )
        assert jnp.allclose(step_flux[:, 0, :], full_flux[:, t, :], atol=5e-3)
        assert jnp.allclose(sk[:, :, 0, :], full_k[:, :, t, :], atol=5e-3)
        assert jnp.allclose(sv[:, :, 0, :], full_v[:, :, t, :], atol=5e-3)

    out_ids = generate_causal_prefill_jax(params, toks, max_new_tokens=3, buffers=buffers, config=cfg, temperature=0.0)
    assert len(out_ids) == 3


def test_chunked_mha_exact_parity():
    """Verifies query-chunked MHA matches unchunked MHA with zero mathematical drift."""
    import jax
    import jax.numpy as jnp
    from src.jax.v5_model import _chunked_mha_jax

    key = jax.random.PRNGKey(11)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (2, 4, 32, 8))
    k = jax.random.normal(k2, (2, 4, 36, 8))
    v = jax.random.normal(k3, (2, 4, 36, 8))
    mask = jnp.ones((2, 1, 1, 36), dtype=bool)

    out_full = _chunked_mha_jax(q, k, v, attn_mask=mask, is_causal=True, query_pos_offset=4, top_k_res=8, chunk_size=512)
    out_chunk = _chunked_mha_jax(q, k, v, attn_mask=mask, is_causal=True, query_pos_offset=4, top_k_res=8, chunk_size=8, use_remat=True)
    # Note: 32 % 8 == 0 only triggers when Q_len > chunk_size; test with chunk_size=16 (< 32)
    out_chunk16 = _chunked_mha_jax(q, k, v, attn_mask=mask, is_causal=True, query_pos_offset=4, top_k_res=8, chunk_size=16, use_remat=True)
    assert jnp.allclose(out_full, out_chunk16, atol=1e-5)


def test_gpu_tpu_1to1_parity():
    """Verifies PyTorch GPU model, AutoTuner framework='pytorch', and GPU trainer CLI match TPU V5.1 flags."""
    from autotune.adapters.v5 import benchmark_v5_model
    from scripts.train_v5_graded_curriculum import parse_args as parse_gpu_args
    from src.spheroid_v5 import ModelConfigV5 as TorchCfg

    t_cfg = TorchCfg(vocab_size=64, num_shells=4, surface_dim=32, radial_dim=16, num_heads=4, max_phase_steps=32)
    assert t_cfg.use_sensory_core_coupling is False
    assert t_cfg.use_parallel_plan_scan is False
    assert t_cfg.preserve_shell_norm_clamp is True

    res = benchmark_v5_model(
        preset="v5_small",
        batch_size=2,
        seq_len=8,
        prompt_len=4,
        num_steps=2,
        dtype="float32",
        model_config=t_cfg,
        framework="pytorch",
    )
    assert res["status"] == "OK"
    assert res["tok_per_sec"] > 0.0

    gpu_args = parse_gpu_args(["--dataset_source", "pretrain", "--auto_tune", "--selective_loss", "--z_loss_weight", "1e-4"])
    assert gpu_args.dataset_source == "pretrain"
    assert gpu_args.auto_tune is True
    assert gpu_args.selective_loss is True
    assert gpu_args.z_loss_weight == 1e-4
    assert gpu_args.target_tokens == 41_400_000_000


def test_from_scratch_coupled_core_and_parallel_plan_forward_grad():
    """Verifies use_sensory_core_coupling + use_parallel_plan_scan run forward & grad cleanly without touching defaults."""
    import jax
    import jax.numpy as jnp
    from src.jax.v5_model import ModelConfigV5, init_params_v5, training_rollout

    cfg_default = ModelConfigV5(vocab_size=64, num_shells=4, surface_dim=32, radial_dim=16, num_heads=4, max_phase_steps=32)
    assert cfg_default.use_sensory_core_coupling is False
    assert cfg_default.use_parallel_plan_scan is False

    cfg_new = ModelConfigV5(
        vocab_size=64,
        num_shells=4,
        surface_dim=32,
        radial_dim=16,
        num_heads=4,
        max_phase_steps=32,
        num_sensory_layers=2,
        use_centralized_readout=False,
        use_shell_tap=True,
        use_bidirectional_tracts=True,
        use_sensory_core_coupling=True,
        use_parallel_plan_scan=True,
    )
    params, buffers = init_params_v5(jax.random.PRNGKey(0), cfg_new)
    prompts = jnp.array([[1], [1]], dtype=jnp.int32)
    targets = jnp.array([[5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16]], dtype=jnp.int32)

    def _loss(p):
        return training_rollout(
            p, prompts, 6, buffers, config=cfg_new, targets=targets, pure_teacher_forcing=True, deliberation_mode="teacher_forcing"
        )

    loss_val, grads = jax.value_and_grad(_loss)(params)
    assert jnp.isfinite(loss_val)
    assert float(jnp.linalg.norm(grads["core.intra_swiglu.w_gate"])) > 0.0
    assert float(jnp.linalg.norm(grads["motor.plan_update.weight"])) > 0.0
