import pytest
from training.continuous_trainer import parse_args, build_trainer_command


def test_continuous_trainer_defaults():
    args, unknown = parse_args()
    assert args.preset == "v5_small"
    assert args.target_tokens == 50_000_000_000
    assert args.reasoning_transition_tokens == 15_700_000_000
    assert args.final_campaign_tokens == 50_000_000_000
    assert args.reasoning_max_lr == 2e-4
    assert args.reasoning_min_lr == 3e-5
    assert args.cloud_sync is True
    assert args.max_restarts is None


def test_continuous_trainer_builds_command_correctly():
    args, _ = parse_args()
    args.preset = "v5_small"
    args.effective_batch_size = 64
    args.target_tokens = 50_000_000_000
    args.save_interval = 250
    args.eval_interval = 1000
    args.log_every = 10

    extra_args = ["--auto-tune", "--tune-objective", "balanced"]
    cmd = build_trainer_command(args, extra_args)

    assert any("train_v5_tpu_jax.py" in part for part in cmd)
    assert "--preset" in cmd and "v5_small" in cmd
    assert "--target-tokens" in cmd and "50000000000" in cmd
    assert "--reasoning-max-lr" in cmd and "0.0002" in cmd
    assert "--auto-tune" in cmd
    assert "--tune-objective" in cmd and "balanced" in cmd
