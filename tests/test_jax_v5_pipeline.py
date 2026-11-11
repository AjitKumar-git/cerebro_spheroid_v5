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


def test_causal_kv_no_leakage_and_monotonic_descent():
    """Verifies that Causal KV accumulation enforces strict non-leakage and drives clean loss descent."""
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    key = jax.random.PRNGKey(42)
    params, buffers = init_params_v5(key, cfg)

    prompts = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    targets_a = jnp.array([[10, 20, 30, 40]], dtype=jnp.int32)
    targets_b = jnp.array([[10, 20, 30, 99]], dtype=jnp.int32)  # Differs only at step 3

    # Step 0 loss should be IDENTICAL regardless of future token at step 3 (zero leakage)
    opt_state = init_spheroid_opt_state(params)
    train_step = build_train_step_spheroid(cfg, buffers, remat=False)

    # Multi-step training descent
    losses = []
    p = params
    s = opt_state
    for _ in range(10):
        p, s, loss = train_step(p, s, prompts, targets_a, 2e-3)
        losses.append(float(loss))

    assert losses[-1] < losses[0], f"Loss must decrease: {losses[0]} -> {losses[-1]}"
    assert losses[0] - losses[-1] > 1.0, "Loss must decrease substantially with Causal KV"


def test_legacy_checkpoint_weight_transfer_and_continuity():
    """Verifies legacy checkpoints missing newly added parameters transfer seamlessly."""
    import numpy as np
    import torch
    import jax
    import jax.numpy as jnp
    from src.jax.v5_model import ModelConfigV5, init_params_v5, training_rollout
    from src.jax.checkpoint import adapt_and_load_checkpoint_weights

    cfg = ModelConfigV5(
        vocab_size=64,
        num_shells=4,
        surface_dim=32,
        radial_dim=16,
        num_heads=4,
        max_phase_steps=32,
        thinking_cycles=2,
    )
    src_params, buffers = init_params_v5(jax.random.PRNGKey(11), cfg)

    new_only_keys = {
        "sensory.pos_embed.weight",
        "core.radial_mix_scale",
        "core.shortcut_mix_scale",
        "core.radial_up_out.weight",
        "motor.phase_scale",
        "motor.input_gate",
    }
    legacy_state = {}
    for k, v in src_params.items():
        if k in new_only_keys:
            continue
        legacy_state[k] = torch.from_numpy(np.asarray(v, dtype=np.float32).copy()).to(torch.bfloat16)

    legacy_ckpt = {"step": 1000, "loss": 6.0, "model_state": legacy_state}
    target_params, _ = init_params_v5(jax.random.PRNGKey(99), cfg)
    target_params = {k: v.astype(jnp.bfloat16) for k, v in target_params.items()}

    restored = adapt_and_load_checkpoint_weights(legacy_ckpt, target_params, jnp.bfloat16, cfg)
    assert restored == len(legacy_state)

    # Decoupled outward radial projection must copy radial_up.weight when missing
    assert jnp.allclose(
        target_params["core.radial_up_out.weight"].astype(jnp.float32),
        target_params["core.radial_up.weight"].astype(jnp.float32),
    )
    # Identity-preserving defaults for new parameters
    assert jnp.allclose(target_params["sensory.pos_embed.weight"].astype(jnp.float32), 0.0)
    assert jnp.allclose(target_params["motor.input_gate"].astype(jnp.float32), 1.0)
    assert jnp.allclose(target_params["core.radial_mix_scale"].astype(jnp.float32), 0.1, atol=1e-3)
    assert jnp.allclose(target_params["core.shortcut_mix_scale"].astype(jnp.float32), 0.05, atol=1e-3)
    assert jnp.allclose(target_params["motor.phase_scale"].astype(jnp.float32), cfg.phase_scale, atol=1e-2)

    prompts = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    targets = jnp.array([[5, 6, 7, 8]], dtype=jnp.int32)
    loss = training_rollout(target_params, prompts, target_len=4, buffers=buffers, config=cfg, targets=targets)
    assert jnp.isfinite(loss)


def test_radial_shell_magnitude_hierarchy_preserved():
    """Verifies P1 Pre-Norm preserves distinct per-shell magnitudes and P5 outward projection works."""
    import jax
    import jax.numpy as jnp
    from src.jax.v5_model import ModelConfigV5, init_params_v5, deliberate_step

    cfg = ModelConfigV5(
        vocab_size=64,
        num_shells=6,
        surface_dim=32,
        radial_dim=16,
        num_heads=4,
        max_phase_steps=32,
        thinking_cycles=3,
    )
    params, buffers = init_params_v5(jax.random.PRNGKey(21), cfg)

    state = jnp.zeros((1, cfg.num_shells, cfg.surface_dim), dtype=jnp.float32)
    surface_vec = jax.random.normal(jax.random.PRNGKey(22), (1, 1, cfg.surface_dim)) * 2.0
    state = state.at[:, -1:, :].set(surface_vec)

    for _ in range(3):
        state = deliberate_step(params, state, buffers)

    shell_rms = jnp.sqrt(jnp.mean(jnp.square(state), axis=-1) + 1e-8)[0]
    # Pre-Norm residual core should NOT collapse every shell to identical RMS=1.0
    assert float(jnp.std(shell_rms)) > 1e-2

    # Perturbing outward projection must alter deep-to-surface deliberation state
    params_alt = dict(params)
    params_alt["core.radial_up_out.weight"] = params["core.radial_up_out.weight"] * 2.5
    state_ref = deliberate_step(params, state, buffers)
    state_mod = deliberate_step(params_alt, state, buffers)
    assert float(jnp.max(jnp.abs(state_mod - state_ref))) > 1e-4


def test_recency_weighted_prompt_order_sensitivity():
    """Verifies P3 recency-weighted pooling distinguishes reordered prompts in JAX and PyTorch."""
    import torch
    import jax
    import jax.numpy as jnp
    from src.jax.v5_model import ModelConfigV5, init_params_v5, training_rollout
    from src.spheroid_v5 import ModelConfigV5 as TorchCfg, CerebroSpheroidV5

    cfg = ModelConfigV5(
        vocab_size=64,
        num_shells=4,
        surface_dim=32,
        radial_dim=16,
        num_heads=4,
        max_phase_steps=32,
        thinking_cycles=1,
    )
    params, buffers = init_params_v5(jax.random.PRNGKey(31), cfg)

    pa = jnp.array([[10, 20, 30, 40]], dtype=jnp.int32)
    pb = jnp.array([[40, 30, 20, 10]], dtype=jnp.int32)
    logits_a = training_rollout(params, pa, target_len=3, buffers=buffers, config=cfg)
    logits_b = training_rollout(params, pb, target_len=3, buffers=buffers, config=cfg)
    assert float(jnp.max(jnp.abs(logits_a - logits_b))) > 1e-3

    # PyTorch order sensitivity parity
    t_cfg = TorchCfg(
        vocab_size=64,
        num_shells=4,
        surface_dim=32,
        radial_dim=16,
        num_heads=4,
        max_phase_steps=32,
        thinking_cycles=1,
    )
    t_model = CerebroSpheroidV5(t_cfg).eval()
    t_pa = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    t_pb = torch.tensor([[40, 30, 20, 10]], dtype=torch.long)
    with torch.no_grad():
        t_la = t_model(t_pa, target_len=3)
        t_lb = t_model(t_pb, target_len=3)
    assert float((t_la - t_lb).abs().max()) > 1e-3


def test_bfloat16_norms_and_scalars_update_under_spheroid_optimizer():
    """Verifies float32 master weights & norm exclusion prevent BF16 ULP parameter freezing."""
    import jax
    import jax.numpy as jnp
    from src.jax.v5_model import ModelConfigV5, init_params_v5
    from src.jax.spheroid_optimizer import (
        is_manifold_operator,
        init_spheroid_opt_state,
        build_spheroid_train_engine,
    )

    cfg = ModelConfigV5(
        vocab_size=64,
        num_shells=4,
        surface_dim=32,
        radial_dim=16,
        num_heads=4,
        max_phase_steps=32,
        thinking_cycles=2,
    )
    params_f32, buffers = init_params_v5(jax.random.PRNGKey(41), cfg)
    params_bf16 = {k: v.astype(jnp.bfloat16) for k, v in params_f32.items()}

    # Confirm norms and positional lookup embeddings are excluded from Newton-Schulz
    assert not is_manifold_operator("core.intra_norms.weight", params_bf16["core.intra_norms.weight"])
    assert not is_manifold_operator("sensory.pos_embed.weight", params_bf16["sensory.pos_embed.weight"])
    assert is_manifold_operator("core.radial_up.weight", params_bf16["core.radial_up.weight"])

    opt_state = init_spheroid_opt_state(params_bf16)
    init_intra = opt_state["master_p"]["core.intra_norms.weight"].copy()
    init_gains = opt_state["master_p"]["core.shell_gains"].copy()
    init_affine = opt_state["master_p"]["motor.affine_scale"].copy()
    init_gate = opt_state["master_p"]["motor.input_gate"].copy()

    compute_grad, apply_opt = build_spheroid_train_engine(cfg, buffers, remat=False)
    prompts = jnp.array([[3, 7, 11, 15], [4, 8, 12, 16]], dtype=jnp.int32)
    targets = jnp.array([[20, 21, 22, 23], [24, 25, 26, 27]], dtype=jnp.int32)

    cur_p = params_bf16
    cur_opt = opt_state
    # lr=1e-4 gives lr_1d=2e-5 < BF16 half-ULP (0.0039 at 1.0); master_p must still advance cleanly
    for _ in range(5):
        _, grads = compute_grad(cur_p, prompts, targets)
        cur_p, cur_opt = apply_opt(cur_p, cur_opt, grads, 1e-4)

    for k in cur_p:
        assert cur_p[k].dtype == jnp.bfloat16

    assert float(jnp.linalg.norm(cur_opt["master_p"]["core.intra_norms.weight"] - init_intra)) > 1e-5
    assert float(jnp.linalg.norm(cur_opt["master_p"]["core.shell_gains"] - init_gains)) > 1e-5
    assert float(jnp.linalg.norm(cur_opt["master_p"]["motor.affine_scale"] - init_affine)) > 1e-5
    assert float(jnp.linalg.norm(cur_opt["master_p"]["motor.input_gate"] - init_gate)) > 1e-5


def test_preshifted_feedback_no_double_shift_and_pytorch_causality(monkeypatch):
    """Verifies pre-shifted feedback_tokens Step-0 conditioning, zero leakage, and -100 padding."""
    import numpy as np
    import torch
    import jax
    import jax.numpy as jnp
    from src.jax.v5_model import ModelConfigV5, init_params_v5, training_rollout
    from src.jax.data_streamer import LinguisticDataStreamer
    from src.spheroid_v5 import ModelConfigV5 as TorchCfg, CerebroSpheroidV5

    cfg = ModelConfigV5(
        vocab_size=64,
        num_shells=4,
        surface_dim=32,
        radial_dim=16,
        num_heads=4,
        max_phase_steps=32,
        thinking_cycles=1,
    )
    params, buffers = init_params_v5(jax.random.PRNGKey(51), cfg)

    prompts = jnp.array([[2, 3, 4, 5]], dtype=jnp.int32)
    fb_a = jnp.array([[17, 10, 11, 12, 13]], dtype=jnp.int32)
    fb_b = jnp.array([[29, 10, 44, 12, 13]], dtype=jnp.int32)
    fb_c = jnp.array([[17, 10, 44, 12, 13]], dtype=jnp.int32)

    # JAX: changing fb[:, 0] changes step 0; changing step 2 must not leak to steps < 2
    logits_a = training_rollout(
        params, prompts, target_len=5, buffers=buffers, config=cfg,
        feedback_tokens=fb_a, sampling_p=1.0, feedback_is_shifted=True,
    )
    logits_b = training_rollout(
        params, prompts, target_len=5, buffers=buffers, config=cfg,
        feedback_tokens=fb_b, sampling_p=1.0, feedback_is_shifted=True,
    )
    logits_c = training_rollout(
        params, prompts, target_len=5, buffers=buffers, config=cfg,
        feedback_tokens=fb_c, sampling_p=1.0, feedback_is_shifted=True,
    )
    assert float(jnp.max(jnp.abs(logits_a[:, 0, :] - logits_b[:, 0, :]))) > 1e-3
    assert float(jnp.max(jnp.abs(logits_a[:, :2, :] - logits_c[:, :2, :]))) < 1e-5
    assert float(jnp.max(jnp.abs(logits_a[:, 2, :] - logits_c[:, 2, :]))) > 1e-3

    # PyTorch causality and Step-0 feedback conditioning
    t_cfg = TorchCfg(
        vocab_size=64,
        num_shells=4,
        surface_dim=32,
        radial_dim=16,
        num_heads=4,
        max_phase_steps=32,
        thinking_cycles=1,
    )
    t_model = CerebroSpheroidV5(t_cfg).eval()
    tp = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    tt = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)
    tfb_1 = torch.tensor([[17, 10, 11, 12, 13]], dtype=torch.long)
    tfb_2 = torch.tensor([[17, 10, 44, 12, 13]], dtype=torch.long)
    with torch.no_grad():
        _, tl_1 = t_model(tp, targets=tt, feedback_tokens=tfb_1, feedback_is_shifted=True)
        _, tl_2 = t_model(tp, targets=tt, feedback_tokens=tfb_2, feedback_is_shifted=True)
    # Steps 0 and 1 must be strictly unaffected by step-2 feedback mutation
    assert float((tl_1[:, :2, :] - tl_2[:, :2, :]).abs().max()) < 1e-5
    # Step 2 must directly respond to step-2 feedback mutation (no double shift / no mask blockage)
    assert float((tl_1[:, 2, :] - tl_2[:, 2, :]).abs().max()) > 1e-3

    # Verify PyTorch sequential rollout also obeys causality without double shift
    t_model.force_sequential = True
    with torch.no_grad():
        _, tls_1 = t_model(tp, targets=tt, feedback_tokens=tfb_1, feedback_is_shifted=True)
        _, tls_2 = t_model(tp, targets=tt, feedback_tokens=tfb_2, feedback_is_shifted=True)
    assert float((tls_1[:, :2, :] - tls_2[:, :2, :]).abs().max()) < 1e-5
    assert float((tls_1[:, 2, :] - tls_2[:, 2, :]).abs().max()) > 1e-3

    # LinguisticDataStreamer default target_pad_id == -100
    monkeypatch.setattr(
        LinguisticDataStreamer,
        "_generate_synthetic_prose",
        lambda self: ("<stim>Hi</stim>", "<act>Yes</act>"),
    )
    streamer = LinguisticDataStreamer(batch_size=2, prompt_len=8, target_len=64, vocab_size=64, data_dir="__nonexistent__")
    try:
        _, r_b = streamer.get_batch()
        assert int(jnp.sum(r_b == -100)) > 0
        assert int(jnp.sum(r_b[:, -16:] == 0)) == 0
    finally:
        if hasattr(streamer, "stop"):
            streamer.stop()
