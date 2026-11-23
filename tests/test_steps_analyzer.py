"""
Tests for TPU JAX Steps Analyzer & Token Budget Scheduling.
============================================================

Verifies honest token accounting in pretraining (BOS seed vs SFT prompt context),
curriculum-aware multi-stage step budgeting, and resumption token deficit recalibration.
"""

import math
import pytest
from training.train_v5_tpu_jax import (
    calculate_curriculum_aware_target_steps,
    parse_args,
    run_training,
)


def test_honest_pretrain_token_accounting_vs_legacy_mismatch():
    """
    Verifies the fix for the 15.7B -> 8B token exhaustion bug.
    
    Legacy bug:
        tokens_per_opt_step = eff_batch_size * (prompt_len + seq_len)
        With prompt_len=256, seq_len=256, eff_batch_size=160:
        81,920 tok/step -> 15.7B / 81,920 = 191,650 steps.
        In reality, pretraining has prompt_len=1 (BOS seed).
        Real tokens per step = 160 * (1 + 256) = 41,120 tok/step.
        191,650 steps * 41,120 = 7.88 Billion tokens (~8B), cutting training in half!

    Fix:
        Pretraining sets real_prompt_slots=1, real_target_slots=256.
        calculate_curriculum_aware_target_steps schedules ~381,809 steps,
        ensuring all 15.7 Billion tokens are ingested.
    """
    target_budget = 15_700_000_000
    eff_batch_size = 160
    seq_len = 256

    # Legacy flawed calculation:
    legacy_tok_per_step = eff_batch_size * (256 + seq_len)  # 81,920
    legacy_steps = int(math.ceil(target_budget / legacy_tok_per_step))  # 191,651
    legacy_real_tokens_ingested = legacy_steps * (eff_batch_size * (1 + seq_len))
    assert 7.8e9 <= legacy_real_tokens_ingested <= 7.9e9  # Exactly ~7.88B!

    # Fixed honest calculation:
    fixed_steps = calculate_curriculum_aware_target_steps(
        target_budget=target_budget,
        eff_batch_size=eff_batch_size,
        base_prompt_len=1,  # BOS token seed
        base_seq_len=seq_len,
        context_curriculum=False,
    )
    assert fixed_steps == 381810
    fixed_real_tokens_ingested = fixed_steps * (eff_batch_size * (1 + seq_len))
    assert fixed_real_tokens_ingested >= target_budget


def test_resumption_from_halfway_checkpoint():
    """
    Verifies that resuming from an existing checkpoint (e.g. at step 191,650 with 7.88B tokens)
    schedules the remaining ~190k steps to finish the 15.7B campaign, instead of exiting immediately.
    """
    target_budget = 15_700_000_000
    resumed_cum_tokens = 7_880_000_000
    eff_batch_size = 160
    seq_len = 256
    start_step = 191_650

    remaining_budget = target_budget - resumed_cum_tokens
    assert remaining_budget == 7_820_000_000

    remaining_steps = calculate_curriculum_aware_target_steps(
        target_budget=remaining_budget,
        eff_batch_size=eff_batch_size,
        base_prompt_len=1,
        base_seq_len=seq_len,
        context_curriculum=False,
        start_step=start_step,
    )

    total_campaign_steps = start_step + remaining_steps
    assert 380_000 <= total_campaign_steps <= 385_000

    # Total tokens ingested across both sessions reaches the full 15.7B target
    total_tokens = resumed_cum_tokens + remaining_steps * (eff_batch_size * (1 + seq_len))
    assert total_tokens >= target_budget


def test_context_curriculum_step_budgeting():
    """
    Verifies curriculum-aware step budgeting across expanding context stages
    (512 -> 1024 -> 2048 -> 4096) with 100k steps/stage.
    """
    target_budget = 15_700_000_000
    eff_batch_size = 64
    context_stages = [512, 1024, 2048, 4096]
    context_stage_steps = 100_000

    steps = calculate_curriculum_aware_target_steps(
        target_budget=target_budget,
        eff_batch_size=eff_batch_size,
        base_prompt_len=1,
        base_seq_len=512,
        context_curriculum=True,
        context_stages=context_stages,
        context_stage_steps=context_stage_steps,
        start_step=0,
    )

    # Stage 0: 100,000 steps * 64 * 513 = 3,283,200,000 tokens
    # Stage 1: 100,000 steps * 64 * 1025 = 6,560,000,000 tokens (cum: 9.84B)
    # Stage 2: (15.7B - 9.8432B) / (64 * 2049) = 5.8568B / 131,136 = 44,663 steps
    # Total steps = 100,000 + 100,000 + 44,663 = 244,663 steps
    assert 244_000 <= steps <= 245_000


def test_explicit_total_steps_flag_recognition():
    """Verifies that all aliases for explicit total steps are recognized."""
    for flag in ["--total-steps", "--total_steps", "--steps", "--max-steps", "--max_steps"]:
        args = parse_args([flag, "50000"])
        assert args.total_steps == 50000
        assert args.total_steps_explicit is True

    # When not passed explicitly, total_steps_explicit is False
    args_default = parse_args([])
    assert args_default.total_steps_explicit is False


def test_stage_1_pretrain_held_out_domains(monkeypatch):
    """
    Verifies that Stage 1 (Pretrain / Step 1) only evaluates FineWeb (broad_web)
    and The Stack (code), with 85% / 15% canonical weights.
    """
    from src.corpus import build_tokenizer
    import src.elastic_window as ew

    # Mock _open_hf_stream to fail fast and use authentic local embedded domain fallbacks
    monkeypatch.setattr(ew, "_open_hf_stream", lambda *args, **kwargs: iter([]))

    tokenizer = build_tokenizer("llama")
    eval_batches, domain_labels, domain_weights = ew.build_pretrain_eval_split(
        tokenizer=tokenizer,
        batch_size=2,
        prompt_len=1,
        target_len=64,
        stage="pretrain",
        return_weights=True,
    )

    assert len(eval_batches) == 2
    assert len(domain_labels) == 2
    assert len(domain_weights) == 2

    # Verify only broad_web and code domains are present
    assert any("Broad Web" in lbl or "FineWeb" in lbl for lbl in domain_labels)
    assert any("Code" in lbl or "Stack" in lbl for lbl in domain_labels)
    assert not any("Math" in lbl for lbl in domain_labels)
    assert not any("Cosmopedia" in lbl for lbl in domain_labels)
    assert not any("Edu" in lbl for lbl in domain_labels)

    # Verify normalized weights 85% / 15%
    assert math.isclose(domain_weights[0], 0.85, abs_tol=1e-4)
    assert math.isclose(domain_weights[1], 0.15, abs_tol=1e-4)
    assert math.isclose(sum(domain_weights), 1.0, abs_tol=1e-4)


def test_stage_2_deep_reasoning_held_out_domains(monkeypatch):
    """
    Verifies that Stage 2 (Deep Reasoning) evaluates Math, Cosmopedia, Code, and Edu Academic,
    with 35% / 35% / 20% / 10% weights.
    """
    from src.corpus import build_tokenizer
    import src.elastic_window as ew

    monkeypatch.setattr(ew, "_open_hf_stream", lambda *args, **kwargs: iter([]))

    tokenizer = build_tokenizer("llama")
    eval_batches, domain_labels, domain_weights = ew.build_pretrain_eval_split(
        tokenizer=tokenizer,
        batch_size=2,
        prompt_len=1,
        target_len=64,
        stage="deep_reasoning",
        return_weights=True,
    )

    assert len(eval_batches) == 4
    assert len(domain_labels) == 4
    assert len(domain_weights) == 4

    # Verify reasoning domains are present
    assert any("Math" in lbl for lbl in domain_labels)
    assert any("Cosmopedia" in lbl for lbl in domain_labels)
    assert any("Code" in lbl for lbl in domain_labels)
    assert any("Edu" in lbl for lbl in domain_labels)

    # Verify normalized weights: Math 35%, Cosmopedia 35%, Code 20%, Edu 10%
    w_map = {lbl: wt for lbl, wt in zip(domain_labels, domain_weights)}
    math_key = [k for k in w_map if "Math" in k][0]
    cosmo_key = [k for k in w_map if "Cosmopedia" in k][0]
    code_key = [k for k in w_map if "Code" in k][0]
    edu_key = [k for k in w_map if "Edu" in k][0]

    assert math.isclose(w_map[math_key], 0.35, abs_tol=1e-4)
    assert math.isclose(w_map[cosmo_key], 0.35, abs_tol=1e-4)
    assert math.isclose(w_map[code_key], 0.20, abs_tol=1e-4)
    assert math.isclose(w_map[edu_key], 0.10, abs_tol=1e-4)
    assert math.isclose(sum(domain_weights), 1.0, abs_tol=1e-4)


def test_held_out_explicit_weights_override(monkeypatch):
    """Verifies that explicit domain_weights override stage defaults and normalize to 1.0."""
    from src.corpus import build_tokenizer
    import src.elastic_window as ew

    monkeypatch.setattr(ew, "_open_hf_stream", lambda *args, **kwargs: iter([]))

    tokenizer = build_tokenizer("llama")
    eval_batches, domain_labels, domain_weights = ew.build_pretrain_eval_split(
        tokenizer=tokenizer,
        batch_size=2,
        prompt_len=1,
        target_len=64,
        domains=["broad_web", "code"],
        domain_weights={"broad_web": 0.50, "code": 0.50},
        return_weights=True,
    )

    assert len(eval_batches) == 2
    assert math.isclose(domain_weights[0], 0.50, abs_tol=1e-4)
    assert math.isclose(domain_weights[1], 0.50, abs_tol=1e-4)
    assert math.isclose(sum(domain_weights), 1.0, abs_tol=1e-4)

