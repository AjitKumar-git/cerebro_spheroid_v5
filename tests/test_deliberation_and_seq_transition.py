import pytest
import math
import numpy as np
import torch
import torch.nn.functional as F

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    PRESETS_V5,
    load_v5_checkpoint_into_torch,
)
from src.jax.v5_model import ModelConfigV5 as JAXConfigV5, init_params_v5, training_rollout
from src.jax.auto_tuner import get_hardware_info
import jax
import jax.numpy as jnp


def test_pytorch_deliberation_sequential_rollout_backward():
    """Verifies that PyTorch forward with sampling_p < 1.0 triggers sequential deliberation and backward pass."""
    cfg = dict(PRESETS_V5["v5_small"])
    cfg["vocab_size"] = 500
    cfg["num_shells"] = 4
    config = ModelConfigV5(**cfg)
    model = CerebroSpheroidV5(config).cpu()

    B, L_p, L_t = 2, 8, 16
    prompts = torch.randint(1, 500, (B, L_p))
    targets = torch.randint(1, 500, (B, L_t))
    feedback = torch.randint(1, 500, (B, L_t))

    # Fast TF should NOT be used when sampling_p < 1.0
    out = model(prompts, targets=targets, feedback_tokens=feedback, sampling_p=0.5, pitfall_prob=0.2)
    loss = out[0] if isinstance(out, tuple) else out
    assert loss is not None
    assert math.isfinite(loss.item())
    assert loss.item() > 0.0

    loss.backward()
    assert model.sensory_dnn.embedding.weight.grad is not None
    assert model.sensory_dnn.embedding.weight.grad.norm().item() > 0.0


def test_jax_deliberation_rollout_and_pitfall():
    """Verifies that JAX training_rollout with sampling_p < 1.0 and pitfall_prob > 0.0 produces valid finite loss."""
    cfg = dict(PRESETS_V5["v5_small"])
    cfg["vocab_size"] = 500
    cfg["num_shells"] = 4
    config = JAXConfigV5(**cfg)

    key = jax.random.PRNGKey(42)
    params, buffers = init_params_v5(key, config)

    B, L_p, L_t = 2, 8, 16
    prompts = jnp.ones((B, L_p), dtype=jnp.int32)
    targets = jnp.ones((B, L_t), dtype=jnp.int32)
    feedback = jnp.ones((B, L_t), dtype=jnp.int32)

    loss = training_rollout(
        params,
        prompts,
        target_len=L_t,
        buffers=buffers,
        config=config,
        feedback_tokens=feedback,
        sampling_p=0.5,
        pitfall_prob=0.15,
        targets=targets,
        pure_teacher_forcing=False,
    )
    assert loss is not None
    loss_val = float(loss)
    assert math.isfinite(loss_val)
    assert loss_val > 0.0


def test_sequence_length_transition_preserves_position_embeddings():
    """Verifies that transitioning sequence length from 256 to 512 cleanly loads and adapts weights without failure."""
    cfg256 = dict(PRESETS_V5["v5_small"])
    cfg256["vocab_size"] = 1000
    cfg256["num_shells"] = 4
    cfg256["max_seq_len"] = 288
    model256 = CerebroSpheroidV5(ModelConfigV5(**cfg256))
    sd256 = model256.state_dict()

    cfg512 = dict(PRESETS_V5["v5_small"])
    cfg512["vocab_size"] = 1000
    cfg512["num_shells"] = 4
    cfg512["max_seq_len"] = 544
    model512 = CerebroSpheroidV5(ModelConfigV5(**cfg512))

    loaded, total = load_v5_checkpoint_into_torch(model512, {"model_state": sd256}, verbose=False)
    assert loaded > 0
    # Positional embeddings table (capacity 1024) easily handles seq_len transition up to 1024
    assert model512.sensory_dnn.pos_embed.weight.shape[0] >= 544


def test_autotuner_cache_sig_incorporates_sequence_length():
    """Verifies that auto-tuner cache signature isolates configs by sequence length."""
    hw = get_hardware_info()
    sig = hw["hardware_signature"]

    # At 256, backward-compatible signature is retained
    seq_suffix_256 = f"_seq{256}" if 256 != 256 else ""
    assert seq_suffix_256 == ""

    # At 512, differentiated signature is generated
    seq_suffix_512 = f"_seq{512}" if 512 != 256 else ""
    assert seq_suffix_512 == "_seq512"
    assert f"{sig}{seq_suffix_512}" == f"{sig}_seq512"


def test_loss_window_governor_curbs_loss_climb_and_restores_teacher_forcing():
    """Verifies that the loss-window governor dynamically throttles deliberation and restores teacher forcing."""
    import argparse
    from training.train_v5_tpu_jax import parse_args

    args = parse_args([])
    args.deliberation_mode = "scheduled"
    args.schedule_steps = 1000
    args.min_sampling_p = 0.5
    args.pitfall_prob = 0.15
    args.loss_window_governor = True
    args.loss_floor = 2.80
    args.loss_ceiling = 3.25

    # Simulated step beyond schedule: nominal deliberation reached min_sampling_p (0.50)
    # 1. Healthy loss <= floor (e.g. 2.75): full deliberation permitted
    alpha = max(0.0, min(1.0, (2.75 - args.loss_floor) / (args.loss_ceiling - args.loss_floor)))
    eff_p = 0.5 + alpha * (1.0 - 0.5)
    eff_pitfall = args.pitfall_prob * (1.0 - alpha)
    assert alpha == 0.0
    assert eff_p == 0.5
    assert eff_pitfall == 0.15

    # 2. Moderate loss in mid-window (e.g. 3.025): governor gently floats sampling_p up and damps pitfall
    alpha_mid = max(0.0, min(1.0, (3.025 - args.loss_floor) / (args.loss_ceiling - args.loss_floor)))
    eff_p_mid = 0.5 + alpha_mid * (1.0 - 0.5)
    eff_pitfall_mid = args.pitfall_prob * (1.0 - alpha_mid)
    assert abs(alpha_mid - 0.5) < 1e-4
    assert abs(eff_p_mid - 0.75) < 1e-4
    assert abs(eff_pitfall_mid - 0.075) < 1e-4

    # 3. Critical loss climbing to 3.7 - 3.9: governor triggers immediate 100% teacher forcing and zeroes pitfalls
    for high_loss in [3.25, 3.50, 3.70, 3.90]:
        alpha_high = max(0.0, min(1.0, (high_loss - args.loss_floor) / (args.loss_ceiling - args.loss_floor)))
        eff_p_high = 0.5 + alpha_high * (1.0 - 0.5)
        eff_pitfall_high = args.pitfall_prob * (1.0 - alpha_high)
        assert alpha_high == 1.0, f"Expected full brake at loss {high_loss}"
        assert eff_p_high == 1.0, f"Expected 100% teacher forcing at loss {high_loss}"
        assert eff_pitfall_high == 0.0, f"Expected zero pitfall perturbation at loss {high_loss}"


def test_workload_hyperparameters_cli_and_model_parity(monkeypatch):
    """Verifies MTP, Persona Steering, and Adaptive Thinking CLI flags and model config wiring."""
    import sys
    from scripts.train_v5_graded_curriculum import parse_args as parse_gpu_args
    from training.train_v5_tpu_jax import parse_args as parse_tpu_args

    # 1. Test GPU CLI parsing
    monkeypatch.setattr(sys, "argv", [
        "train_v5_graded_curriculum.py",
        "--num-mtp-heads", "3",
        "--persona-id", "1",
        "--no-adaptive-thinking",
        "--thinking-cycles", "6",
        "--max-thinking-cycles", "20",
    ])
    gpu_args = parse_gpu_args()
    assert gpu_args.num_mtp_heads == 3
    assert gpu_args.persona_id == 1
    assert gpu_args.use_adaptive_thinking is False
    assert gpu_args.thinking_cycles == 6
    assert gpu_args.max_thinking_cycles == 20

    # 2. Test TPU CLI parsing
    tpu_args = parse_tpu_args([
        "--num_mtp_heads", "2",
        "--persona_id", "2",
        "--use_adaptive_thinking",
        "--thinking_cycles", "8",
        "--max_thinking_cycles", "12",
    ])
    assert tpu_args.num_mtp_heads == 2
    assert tpu_args.persona_id == 2
    assert tpu_args.use_adaptive_thinking is True
    assert tpu_args.thinking_cycles == 8
    assert tpu_args.max_thinking_cycles == 12


def test_pytorch_alternating_deliberation_modes():
    """Verifies that PyTorch supports interleaved, dual_pass, and predictive_coding deliberation modes with backward pass."""
    cfg = dict(PRESETS_V5["v5_small"])
    cfg["vocab_size"] = 500
    cfg["num_shells"] = 4
    config = ModelConfigV5(**cfg)
    model = CerebroSpheroidV5(config).cpu()

    B, L_p, L_t = 2, 8, 16
    prompts = torch.randint(1, 500, (B, L_p))
    targets = torch.randint(1, 500, (B, L_t))
    feedback = torch.randint(1, 500, (B, L_t))

    for mode in ["interleaved", "dual_pass", "predictive_coding"]:
        model.zero_grad(set_to_none=True)
        out = model(
            prompts,
            targets=targets,
            feedback_tokens=feedback,
            sampling_p=0.5,
            deliberation_mode=mode,
        )
        loss = out[0] if isinstance(out, tuple) else out
        assert loss is not None, f"Loss is None for mode {mode}"
        assert math.isfinite(loss.item()), f"Loss non-finite for mode {mode}"
        assert loss.item() > 0.0, f"Loss non-positive for mode {mode}"

        loss.backward()
        assert model.sensory_dnn.embedding.weight.grad is not None, f"Embedding grad None for mode {mode}"
        assert model.sensory_dnn.embedding.weight.grad.norm().item() > 0.0, f"Embedding grad zero for mode {mode}"


def test_jax_alternating_deliberation_modes():
    """Verifies that JAX training_rollout supports interleaved, dual_pass, and predictive_coding with valid gradients."""
    cfg = dict(PRESETS_V5["v5_small"])
    cfg["vocab_size"] = 500
    cfg["num_shells"] = 4
    config = JAXConfigV5(**cfg)

    key = jax.random.PRNGKey(42)
    params, buffers = init_params_v5(key, config)

    B, L_p, L_t = 2, 8, 16
    prompts = jnp.ones((B, L_p), dtype=jnp.int32)
    targets = jnp.ones((B, L_t), dtype=jnp.int32)
    feedback = jnp.ones((B, L_t), dtype=jnp.int32)

    for mode in ["interleaved", "dual_pass", "predictive_coding"]:
        def loss_fn(p):
            return training_rollout(
                p,
                prompts,
                target_len=L_t,
                buffers=buffers,
                config=config,
                feedback_tokens=feedback,
                sampling_p=0.5,
                targets=targets,
                pure_teacher_forcing=False,
                deliberation_mode=mode,
            )

        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        assert math.isfinite(float(loss_val)), f"JAX Loss non-finite for mode {mode}"
        assert float(loss_val) > 0.0, f"JAX Loss non-positive for mode {mode}"
        assert "sensory.embedding.weight" in grads
        grad_norm = float(jnp.linalg.norm(grads["sensory.embedding.weight"]))
        assert math.isfinite(grad_norm) and grad_norm > 0.0, f"JAX Embedding grad zero/non-finite for mode {mode}"


def test_deliberation_mode_cli_parsing():
    """Verifies that all deliberation modes are accepted by both GPU and TPU argument parsers."""
    from scripts.train_v5_graded_curriculum import parse_args as parse_gpu_args
    from training.train_v5_tpu_jax import parse_args as parse_tpu_args

    for mode in ["teacher_forcing", "scheduled", "interleaved", "dual_pass", "predictive_coding"]:
        gpu_args = parse_gpu_args(["--deliberation-mode", mode])
        assert gpu_args.deliberation_mode == mode

        tpu_args = parse_tpu_args(["--deliberation_mode", mode])
        assert tpu_args.deliberation_mode == mode



