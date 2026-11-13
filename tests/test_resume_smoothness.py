import math
import os
import tempfile
import torch
import pytest

from scripts.train_v5_graded_curriculum import parse_args as parse_curriculum_args, _save_and_push_checkpoints
from training.train_v5_tpu_jax import parse_args as parse_tpu_args
from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from src.agentic_harness import ToolActionHead, ActionType


def test_resume_warmup_args_defaults():
    cur_args = parse_curriculum_args(["--model-preset", "v5_small"])
    assert hasattr(cur_args, "resume_warmup_steps")
    assert cur_args.resume_warmup_steps == 100

    tpu_args = parse_tpu_args(["--preset", "tpu_v5e_1"])
    assert hasattr(tpu_args, "resume_warmup_steps")
    assert tpu_args.resume_warmup_steps == 100


def test_checkpoint_payload_persists_ema_loss():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = dict(PRESETS_V5["v5_small"])
        cfg["vocab_size"] = 100
        cfg["num_shells"] = 4
        config = ModelConfigV5(**cfg)
        model = CerebroSpheroidV5(config).cpu()
        action_head = ToolActionHead(surface_dim=config.surface_dim, num_actions=len(ActionType)).cpu()

        best_loss = _save_and_push_checkpoints(
            checkpoint_dir=tmpdir,
            model_preset="v5_small",
            raw_model=model,
            raw_action_head=action_head,
            config=config,
            step=1000,
            loss_val=2.75,
            best_loss=3.0,
            need_local_save=True,
            need_cloud_sync=False,
            eval_loss=2.70,
            ema_loss=2.85,
            final_save=False,
        )

        latest_path = os.path.join(tmpdir, "v5_small_latest.pt")
        assert os.path.exists(latest_path)
        loaded = torch.load(latest_path, map_location="cpu", weights_only=False)
        assert "ema_loss" in loaded
        assert abs(loaded["ema_loss"] - 2.85) < 1e-4
        assert abs(loaded["loss"] - 2.75) < 1e-4
        assert abs(loaded["eval_loss"] - 2.70) < 1e-4


def test_resumption_warmup_lr_schedule():
    """Verifies that resumption warmup smoothly scales LR from 10% to 100% over resume_warmup_steps."""
    start_step = 85400
    resume_warmup_steps = 100
    target_lr = 3.4e-4

    # Step immediately after resume
    step = start_step + 1
    elapsed = step - start_step
    factor_step1 = 0.10 + 0.90 * (float(elapsed) / float(resume_warmup_steps))
    lr_step1 = target_lr * factor_step1
    assert abs(factor_step1 - 0.109) < 1e-4
    assert lr_step1 < 0.15 * target_lr  # Gently starts at ~10.9%

    # Halfway through warmup
    step = start_step + 50
    elapsed = step - start_step
    factor_mid = 0.10 + 0.90 * (float(elapsed) / float(resume_warmup_steps))
    lr_mid = target_lr * factor_mid
    assert abs(factor_mid - 0.55) < 1e-4

    # End of warmup
    step = start_step + 100
    elapsed = step - start_step
    factor_end = 0.10 + 0.90 * (float(elapsed) / float(resume_warmup_steps))
    lr_end = target_lr * factor_end
    assert abs(factor_end - 1.0) < 1e-4
    assert abs(lr_end - target_lr) < 1e-6
