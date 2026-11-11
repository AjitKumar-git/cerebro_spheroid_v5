import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax.numpy as jnp
import numpy as np
import pytest

from src.jax.v5_model import (
    ModelConfigV5,
    PRESETS_V5,
    init_params_v5,
    deliberate_step,
    encode_sensory,
    training_rollout,
)
from src.jax.spheroid_optimizer import (
    init_spheroid_opt_state,
    build_train_step_spheroid,
)
from src.jax.data_streamer import LinguisticDataStreamer
from training.train_v5_tpu_jax import adapt_and_load_checkpoint_weights


def test_v5_test_config_initialization():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    key = jax.random.PRNGKey(42)
    params, buffers = init_params_v5(key, cfg)

    assert "sensory.embedding.weight" in params
    assert "core.intra_swiglu.w_gate" in params
    assert "core.intra_swiglu.w_val" in params
    assert "core.intra_swiglu.w_down" in params
    assert "core.intra_norms.weight" in params
    assert params["core.intra_swiglu.w_gate"].shape == (cfg.num_shells, cfg.surface_dim, cfg.surface_dim * 2)
    assert params["core.intra_norms.weight"].shape == (cfg.num_shells, cfg.surface_dim)
    assert len(buffers) == 2


def test_v5_deliberation_step():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    key = jax.random.PRNGKey(42)
    params, buffers = init_params_v5(key, cfg)

    B = 2
    state = jnp.zeros((B, cfg.num_shells, cfg.surface_dim), dtype=jnp.float32)
    next_state = deliberate_step(params, state, buffers)

    assert next_state.shape == state.shape
    assert not jnp.isnan(next_state).any()


def test_v5_training_rollout_dtypes():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    key = jax.random.PRNGKey(42)
    params, buffers = init_params_v5(key, cfg)

    B, S_prompt, S_target = 2, 4, 8
    prompts = jnp.ones((B, S_prompt), dtype=jnp.int32)
    targets = jnp.ones((B, S_target), dtype=jnp.int32)

    # Float32 execution
    loss_f32 = training_rollout(params, prompts, S_target, buffers, config=cfg, targets=targets)
    assert not jnp.isnan(loss_f32)
    assert loss_f32 > 0

    # Bfloat16 execution (TPU native MXU precision)
    params_bf16 = {k: v.astype(jnp.bfloat16) if jnp.issubdtype(v.dtype, jnp.floating) else v for k, v in params.items()}
    loss_bf16 = training_rollout(params_bf16, prompts, S_target, buffers, config=cfg, targets=targets)
    assert not jnp.isnan(loss_bf16)
    assert loss_bf16 > 0


def test_spheroid_optimizer_train_step():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    key = jax.random.PRNGKey(42)
    params, buffers = init_params_v5(key, cfg)

    params_bf16 = {k: v.astype(jnp.bfloat16) if jnp.issubdtype(v.dtype, jnp.floating) else v for k, v in params.items()}
    opt_state = init_spheroid_opt_state(params_bf16)
    train_step = build_train_step_spheroid(cfg, buffers, remat=True)

    B, S_prompt, S_target = 2, 4, 8
    prompts = jnp.ones((B, S_prompt), dtype=jnp.int32)
    targets = jnp.ones((B, S_target), dtype=jnp.int32)

    new_params, new_opt_s, loss = train_step(params_bf16, opt_state, prompts, targets, 1e-4)
    assert not jnp.isnan(loss)
    # Ensure parameter update occurred
    assert not jnp.array_equal(new_params["core.intra_swiglu.w_gate"], params_bf16["core.intra_swiglu.w_gate"])


def test_checkpoint_adapter_2d_to_3d_fusion():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    key = jax.random.PRNGKey(42)
    target_params, buffers = init_params_v5(key, cfg)

    # Construct mock legacy 2D checkpoint
    s_dim = cfg.surface_dim
    K = cfg.num_shells
    mock_ckpt = {
        "step": 5000,
        "loss": 3.65,
        "model_state": {
            "sensory_dnn.embedding.weight": np.zeros((cfg.vocab_size, s_dim), dtype=np.float32),
        }
    }
    for k in range(K):
        mock_ckpt["model_state"][f"volumetric_core.intra_shell_swiglu.{k}.fc_gate.weight"] = np.ones((s_dim * 2, s_dim), dtype=np.float32)
        mock_ckpt["model_state"][f"volumetric_core.intra_shell_swiglu.{k}.fc_val.weight"] = np.ones((s_dim * 2, s_dim), dtype=np.float32)
        mock_ckpt["model_state"][f"volumetric_core.intra_shell_swiglu.{k}.fc_down.weight"] = np.ones((s_dim, s_dim * 2), dtype=np.float32)
        mock_ckpt["model_state"][f"volumetric_core.intra_norms.{k}.weight"] = np.ones(s_dim, dtype=np.float32)

    restored_cnt = adapt_and_load_checkpoint_weights(mock_ckpt, target_params, jnp.float32, cfg)
    assert restored_cnt >= 4
    # Check 3D tensor fusion
    assert target_params["core.intra_swiglu.w_gate"].shape == (K, s_dim, s_dim * 2)
    assert jnp.all(target_params["core.intra_swiglu.w_gate"] == 1.0)
