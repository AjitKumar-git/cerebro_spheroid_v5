"""
Tests for CerebroSpheroid V5 Exploration Reward Model
====================================================
Verifies:
1. RunningMeanStd (Welford's online normalization)
2. CerebroCuriosityRND (Representation novelty & familiarity decay via predictor updates)
3. ConcentricManifoldExplorer (Shell effective rank, radial flow, and entropy bonuses)
4. TrajectoryDiversityEvaluator (Intra-group leave-one-out n-gram and embedding dispersion)
5. ProcessVerificationEvaluator (Hypothesis exploration, step verification, ground truth extraction)
6. AntiDegenerationGuardrails (Repetition loops and length bounds)
7. CerebroExplorationRewardModel (End-to-end composite rollout scoring, group scoring, save/load)
"""

import os
import tempfile
import torch
import pytest

from src.cerebro_reward_model import (
    RunningMeanStd,
    RNDTargetNetwork,
    RNDPredictorNetwork,
    CerebroCuriosityRND,
    ConcentricManifoldExplorer,
    TrajectoryDiversityEvaluator,
    ProcessVerificationEvaluator,
    ToolExecutionEvaluator,
    ProtocolAndFormattingEvaluator,
    AntiDegenerationGuardrails,
    CerebroExplorationRewardModel,
)


def test_running_mean_std():
    rms = RunningMeanStd(shape=())
    x1 = torch.tensor([10.0, 20.0, 30.0])
    rms.update(x1)
    assert rms.count > 3.0
    assert torch.isclose(rms.mean, torch.tensor(20.0), atol=1e-2)

    # Normalize a batch
    x_test = torch.tensor([20.0, 50.0])
    normed = rms.normalize(x_test, clamp_min=-3.0, clamp_max=3.0)
    assert normed.shape == (2,)
    assert torch.isclose(normed[0], torch.tensor(0.0), atol=0.2)
    assert normed[1] > 0.0
    assert normed[1] <= 3.0


def test_curiosity_rnd_novelty_and_decay():
    torch.manual_seed(42)
    in_dim = 128
    out_dim = 64
    rnd = CerebroCuriosityRND(in_dim=in_dim, out_dim=out_dim, lr=1e-3)

    # Generate familiar vs novel representations
    familiar_proto = torch.randn(1, in_dim)
    familiar_states = familiar_proto + 0.05 * torch.randn(32, in_dim)
    novel_states = torch.randn(8, in_dim) + 5.0  # distinct region

    # Initial novelty
    initial_familiar_novelty = rnd.compute_novelty(familiar_states).mean().item()
    initial_novel_novelty = rnd.compute_novelty(novel_states).mean().item()
    assert initial_familiar_novelty > 0.0
    assert initial_novel_novelty > 0.0

    # Train predictor network on familiar states
    for _ in range(30):
        rnd.update_predictor(familiar_states)

    # After training, familiar states should have lower novelty error than novel states
    post_familiar_novelty = rnd.compute_novelty(familiar_states).mean().item()
    post_novel_novelty = rnd.compute_novelty(novel_states).mean().item()

    assert post_familiar_novelty < post_novel_novelty


def test_concentric_manifold_explorer_collapse_vs_rich():
    K = 16
    D = 64
    explorer = ConcentricManifoldExplorer()

    # Case 1: Degenerate collapsed state (rank 1 across all shells, no radial flow)
    base_vec = torch.randn(1, D)
    collapsed_state = base_vec.repeat(K, 1)  # [K, D] all shells identical

    metrics_collapsed = explorer.evaluate_manifold_exploration(collapsed_state)
    assert metrics_collapsed["radial_flow"] == pytest.approx(0.0, abs=1e-5)
    assert metrics_collapsed["effective_rank"] == pytest.approx(1.0, abs=0.2)

    # Case 2: Rich orthogonal multi-shell state with active radial flow
    rich_state = torch.randn(K, D)
    # Add radial wave propagation gradient
    for k in range(K):
        rich_state[k] = rich_state[k] * (1.0 + 0.1 * k)

    metrics_rich = explorer.evaluate_manifold_exploration(rich_state)
    assert metrics_rich["radial_flow"] > 0.5
    assert metrics_rich["effective_rank"] > metrics_collapsed["effective_rank"]
    assert metrics_rich["manifold_reward"] > metrics_collapsed["manifold_reward"]
    assert 0.0 <= metrics_rich["shell_entropy"] <= 1.0


def test_trajectory_diversity_evaluator():
    evaluator = TrajectoryDiversityEvaluator()

    # Identical tokens
    toks_a = [10, 20, 30, 40, 50, 60]
    toks_b = [10, 20, 30, 40, 50, 60]
    dist_same = evaluator.compute_jaccard_distance(toks_a, toks_b, n=3)
    assert dist_same == 0.0

    # Completely disjoint tokens
    toks_c = [100, 200, 300, 400, 500, 600]
    dist_diff = evaluator.compute_jaccard_distance(toks_a, toks_c, n=3)
    assert dist_diff == 1.0

    # Group diversity evaluation
    candidates_tokens = [
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],  # duplicate
        [10, 20, 30, 40, 50, 60],  # novel
        [100, 200, 300, 400, 500],  # novel
    ]
    rewards = evaluator.evaluate_group_diversity(candidates_tokens)
    assert len(rewards) == 4
    # The duplicate pair (indices 0 and 1) should have lower average distance than index 2 or 3
    assert rewards[0] < rewards[2]
    assert rewards[1] < rewards[3]


def test_process_verification_evaluator():
    evaluator = ProcessVerificationEvaluator()

    # Text with hypothesis exploration and self-correction
    rich_reasoning_text = (
        "Let us consider the problem. Alternatively, another approach is to assume x = 5. "
        "Checking if substituting back into equation 1 holds: 5*2 = 10. "
        "Wait, that's incorrect because the right side is 12. Let's recompute. "
        "So x must be 6. This confirms the solution. Therefore, the answer is 6."
    )

    res = evaluator.evaluate_process_and_verification(
        text=rich_reasoning_text,
        ground_truth="6"
    )

    assert res["hypothesis_exploration"] > 0.0
    assert res["step_verification"] > 0.0
    assert res["self_correction"] > 0.0
    assert res["process_reward"] > 0.4
    assert res["outcome_reward"] == 1.0  # Extracted answer 6 matches ground truth

    # Flat rote answer without reasoning or verification
    rote_text = "The answer is 6."
    res_rote = evaluator.evaluate_process_and_verification(
        text=rote_text,
        ground_truth="6"
    )
    assert res_rote["hypothesis_exploration"] == 0.0
    assert res_rote["step_verification"] == 0.0
    assert res_rote["self_correction"] == 0.0
    assert res_rote["process_reward"] == 0.0
    assert res_rote["outcome_reward"] == 1.0


def test_anti_degeneration_guardrails():
    guard = AntiDegenerationGuardrails()

    # Repetition loop
    stutter_tokens = [42, 99, 105] * 20  # high n-gram repetition
    rep_penalty = guard.check_repetition_penalty(stutter_tokens, n=3, threshold=0.25)
    assert rep_penalty > 0.0

    # Diverse non-repeating tokens
    diverse_tokens = list(range(100))
    no_penalty = guard.check_repetition_penalty(diverse_tokens, n=3, threshold=0.25)
    assert no_penalty == 0.0

    # Length bounds
    too_short = guard.check_length_penalty(length=5, min_len=16)
    assert too_short > 0.0

    too_long = guard.check_length_penalty(length=2000, max_len=1024)
    assert too_long > 0.0

    good_len = guard.check_length_penalty(length=128, min_len=16, max_len=1024)
    assert good_len == 0.0


def test_cerebro_exploration_reward_model_end_to_end():
    surface_dim = 128
    K = 16
    reward_model = CerebroExplorationRewardModel(
        surface_dim=surface_dim,
        rnd_out_dim=64,
        rnd_lr=1e-3,
    )

    # 1. Single trajectory score
    prompt = "What is the square root of 144?"
    response = "Let's verify: 12 * 12 = 144. This confirms the answer is 12."
    token_ids = list(range(1, 35))
    hidden_repr = torch.randn(surface_dim)
    concentric_state = torch.randn(K, surface_dim)

    score_dict = reward_model.score_single_trajectory(
        prompt=prompt,
        response_text=response,
        token_ids=token_ids,
        hidden_representation=hidden_repr,
        concentric_state=concentric_state,
        ground_truth="12",
    )

    assert "total_reward" in score_dict
    assert score_dict["outcome_reward"] == 1.0
    assert score_dict["process_reward"] > 0.0
    assert score_dict["curiosity_novelty"] > 0.0
    assert score_dict["manifold_exploration"] > 0.0
    assert score_dict["penalty"] == 0.0

    # 2. Group rollouts scoring (GRPO style)
    candidates = [
        {
            "text": "The answer is 12.",
            "tokens": [1, 2, 3, 4],  # too short
            "hidden": torch.randn(surface_dim),
            "state": torch.randn(K, surface_dim),
        },
        {
            "text": "Alternatively, suppose 12 * 12 = 144. Checking if true. The answer is 12.",
            "tokens": list(range(10, 45)),
            "hidden": torch.randn(surface_dim),
            "state": torch.randn(K, surface_dim),
        },
        {
            "text": "Suppose x = 11, 11*11=121. Wait, that contradicts 144. Let's recompute: 12*12=144. The answer is 12.",
            "tokens": list(range(100, 150)),
            "hidden": torch.randn(surface_dim),
            "state": torch.randn(K, surface_dim),
        },
        {
            "text": "Gibberish repeating word word word word word word word word word.",
            "tokens": [42, 99, 105] * 15,
            "hidden": torch.randn(surface_dim),
            "state": torch.randn(K, surface_dim),
        },
    ]

    scored_group = reward_model.score_group_rollouts(
        prompt=prompt,
        candidates=candidates,
        ground_truth="12",
    )
    assert len(scored_group) == 4
    for cand_score in scored_group:
        assert "diversity_reward" in cand_score
        assert "total_reward" in cand_score

    # Candidate 2 (with exploration and verification) should score higher than candidate 0 (too short)
    # and candidate 3 (repeating gibberish)
    assert scored_group[2]["total_reward"] > scored_group[0]["total_reward"]
    assert scored_group[2]["total_reward"] > scored_group[3]["total_reward"]

    # 3. Curiosity Predictor State Updates
    loss = reward_model.update_curiosity_with_visited_states(torch.randn(16, surface_dim))
    assert isinstance(loss, float)
    assert loss >= 0.0

    # 4. Checkpoint Save and Load
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        ckpt_path = tmp.name

    try:
        reward_model.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)

        new_model = CerebroExplorationRewardModel(surface_dim=surface_dim, rnd_out_dim=64)
        new_model.load_checkpoint(ckpt_path)

        # Check parameter equality
        for p1, p2 in zip(reward_model.curiosity_rnd.predictor.parameters(), new_model.curiosity_rnd.predictor.parameters()):
            assert torch.allclose(p1, p2)
        assert torch.allclose(reward_model.curiosity_rnd.running_stats.mean, new_model.curiosity_rnd.running_stats.mean)
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def test_grpo_loss_computation_and_policy_gradient():
    from scripts.train_cerebro_grpo_exploration import compute_grpo_loss
    from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5

    cfg = ModelConfigV5(**PRESETS_V5['v5_small'])
    cfg.use_rope = False
    cfg.top_k_resonance = 0

    policy = CerebroSpheroidV5(cfg)
    ref = CerebroSpheroidV5(cfg)
    ref.eval()

    G = 4
    L_p = 8
    L_c = 12

    prompt_ids = torch.randint(1, 1000, (G, L_p))
    completion_ids = torch.randint(1, 1000, (G, L_c))
    advantages = torch.tensor([1.5, -0.5, 0.2, -1.2])

    loss, metrics = compute_grpo_loss(
        policy_model=policy,
        ref_model=ref,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        advantages=advantages,
        clip_ratio=0.20,
        kl_coef=0.04,
        pad_id=0,
    )

    assert torch.is_tensor(loss)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert "policy_loss" in metrics
    assert "kl_div" in metrics
    assert "mean_ratio" in metrics

    # Verify backward pass computes valid finite gradients
    loss.backward()
    has_grad = False
    for p in policy.parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad))
            has_grad = True
    assert has_grad


def test_tool_execution_evaluator():
    evaluator = ToolExecutionEvaluator()

    # 1. Matching calculator tool execution
    cand_match = (
        "Calling the calculator to compute 17 * 24 + 85...\n"
        "<tool_call>calc(17 * 24 + 85)</tool_call>\n"
        "Waiting for tool execution..."
    )
    res_match = evaluator.evaluate_tool_execution(cand_match, ground_truth="493")
    assert res_match["has_tool_call"] is True
    assert res_match["tool_success"] is True
    assert res_match["tool_matches_ground_truth"] is True
    assert res_match["tool_reward"] == 1.0

    # 2. Non-crashing factual tool execution
    cand_factual = "<tool_call>factual('capital of Germany')</tool_call>"
    res_factual = evaluator.evaluate_tool_execution(cand_factual, ground_truth="Berlin")
    assert res_factual["has_tool_call"] is True
    assert res_factual["tool_success"] is True
    assert res_factual["tool_matches_ground_truth"] is True

    # 3. Errored / unparseable tool call
    cand_err = "<tool_call>calc(///invalid syntax///)</tool_call>"
    res_err = evaluator.evaluate_tool_execution(cand_err, ground_truth="42")
    assert res_err["has_tool_call"] is True
    assert res_err["tool_success"] is False
    assert res_err["tool_reward"] < 0.0

    # 4. No tool call
    cand_none = "The answer is simply 42 without any external tools."
    res_none = evaluator.evaluate_tool_execution(cand_none, ground_truth="42")
    assert res_none["has_tool_call"] is False
    assert res_none["tool_reward"] == 0.0


def test_protocol_and_formatting_evaluator():
    evaluator = ProtocolAndFormattingEvaluator()

    # 1. Compliant 3-tier formatting with temporal, waiting, and grounded phrasing
    good_text = (
        "<think>Let's evaluate the prompt and prepare the tool.</think>\n"
        "Calling the calculator to compute this. Waiting for tool response...\n"
        "<tool_call>calc(400 * 0.35)</tool_call>\n"
        "<tool_response>140</tool_response>\n"
        "The tool has responded. Calling the calculator yielded 140, so the answer is 140."
    )
    res_good = evaluator.evaluate_protocol_and_formatting(good_text)
    assert res_good["format_reward"] > 0.5
    assert res_good["balanced_tags"] > 0.0
    assert res_good["temporal_cues"] > 0.0
    assert res_good["waiting_cues"] > 0.0
    assert res_good["grounded_phrasing"] > 0.0
    assert res_good["unbalanced_penalty"] == 0.0
    assert res_good["legacy_tag_penalty"] == 0.0

    # 2. Unbalanced / legacy tags
    bad_text = (
        "<think>Unclosed thinking block\n"
        "<tool_use>calc(10 + 5)</tool_use>\n"
        "<observation>15</observation>"
    )
    res_bad = evaluator.evaluate_protocol_and_formatting(bad_text)
    assert res_bad["unbalanced_penalty"] > 0.0
    assert res_bad["legacy_tag_penalty"] > 0.0
    assert res_bad["format_reward"] < 0.0


