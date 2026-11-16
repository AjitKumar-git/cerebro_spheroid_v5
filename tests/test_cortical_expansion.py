"""
🧠 Tests for Progressive Cortical Expansion & Frozen Core Dynamics
===================================================================
Verifies:
  1. Mask generation across concentric shells, shortcut tracts, and headers.
  2. Parameter count breakdown (61.3% frozen core, 38.7% trainable outer mantle).
  3. Weight transferability from 16-shell source into 32-shell target.
  4. Bit-for-bit parameter invariance on frozen core under Spheroid optimizer updates.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.cortical_expansion import build_cortical_expansion_masks, count_cortical_expansion_params
from src.jax.checkpoint import adapt_and_load_checkpoint_weights
from src.jax.spheroid_optimizer import build_spheroid_train_engine, init_spheroid_opt_state
from src.jax.v5_model import ModelConfigV5, PRESETS_V5, init_params_v5


def test_cortical_expansion_masks_structure():
    cfg = ModelConfigV5(**PRESETS_V5["v5_compact_32shells"])
    params, buffers = init_params_v5(jax.random.PRNGKey(0), cfg)
    masks = build_cortical_expansion_masks(params, num_frozen_shells=16, num_frozen_tracts=40)

    assert len(masks) == len(params)

    # Core SwiGLU gates: 0..15 frozen (0.0), 16..31 trainable (1.0)
    w_gate_mask = np.asarray(masks["core.intra_swiglu.w_gate"])
    assert np.all(w_gate_mask[:16] == 0.0)
    assert np.all(w_gate_mask[16:] == 1.0)

    # Core shortcut tracts: 0..39 frozen (0.0), 40..95 trainable (1.0)
    sc_mask = np.asarray(masks["core.shortcut_weight"])
    assert np.all(sc_mask[:40] == 0.0)
    assert np.all(sc_mask[40:] == 1.0)

    # Radial transitions: 0..14 frozen (0.0), 15..30 trainable (1.0)
    rad_in_mask = np.asarray(masks["core.w_radial_in"])
    assert np.all(rad_in_mask[:15] == 0.0)
    assert np.all(rad_in_mask[15:] == 1.0)

    # Sensory and motor headers: strictly 0.0 (frozen)
    assert np.all(np.asarray(masks["sensory.embedding.weight"]) == 0.0)
    assert np.all(np.asarray(masks["motor.swiglu.fc_gate.weight"]) == 0.0)


def test_cortical_expansion_parameter_breakdown():
    cfg = ModelConfigV5(**PRESETS_V5["v5_compact_32shells"])
    params, buffers = init_params_v5(jax.random.PRNGKey(0), cfg)
    masks = build_cortical_expansion_masks(params, num_frozen_shells=16, num_frozen_tracts=40)
    stats = count_cortical_expansion_params(params, masks)

    assert stats["total_params"] > 160_000_000
    assert stats["total_params"] < 170_000_000
    assert 60.0 <= stats["frozen_pct"] <= 63.0
    assert 37.0 <= stats["trainable_pct"] <= 40.0


def test_frozen_weights_invariance_under_spheroid_optimizer():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    params, buffers = init_params_v5(jax.random.PRNGKey(42), cfg)
    masks = build_cortical_expansion_masks(params, num_frozen_shells=4, num_frozen_tracts=4)

    compute_step, apply_opt = build_spheroid_train_engine(cfg, buffers, param_masks=masks)
    opt_state = init_spheroid_opt_state(params)

    grads = {k: jnp.ones_like(v) * 0.05 for k, v in params.items()}

    orig_embed = params["sensory.embedding.weight"].copy()
    orig_core0 = params["core.intra_swiglu.w_gate"][0].copy()
    orig_outer6 = params["core.intra_swiglu.w_gate"][6].copy()

    # Step twice with optimizer
    for _ in range(2):
        params, opt_state = apply_opt(params, opt_state, grads, 1e-3)

    diff_embed = float(jnp.max(jnp.abs(params["sensory.embedding.weight"] - orig_embed)))
    diff_core0 = float(jnp.max(jnp.abs(params["core.intra_swiglu.w_gate"][0] - orig_core0)))
    diff_outer6 = float(jnp.max(jnp.abs(params["core.intra_swiglu.w_gate"][6] - orig_outer6)))

    # Frozen parameters must have EXACT zero drift
    assert diff_embed == 0.0, f"Sensory embedding drifted by {diff_embed}"
    assert diff_core0 == 0.0, f"Core shell 0 drifted by {diff_core0}"
    # Active outer shell must have learned / updated
    assert diff_outer6 > 0.0, f"Outer shell 6 did not update: {diff_outer6}"


def test_preserved_tract_wiring():
    from src.cortical_expansion import generate_expanded_tract_wiring
    from src.jax.v5_model import generate_small_world_indices

    base_s, base_d = generate_small_world_indices(16, 40)
    exp_s, exp_d = generate_expanded_tract_wiring(
        num_shells=32,
        num_tracts=96,
        base_src=base_s,
        base_dst=base_d,
        num_frozen_shells=16,
        framework="jax",
    )

    assert len(exp_s) == 96
    assert len(exp_d) == 96

    # Legacy tracts 0..39 must match bit-for-bit
    assert np.all(np.asarray(exp_s[:40]) == np.asarray(base_s))
    assert np.all(np.asarray(exp_d[:40]) == np.asarray(base_d))

    # Newly expanded tracts 40..95 must be within valid range [0, 31]
    assert np.all((np.asarray(exp_s[40:]) >= 0) & (np.asarray(exp_s[40:]) < 32))
    assert np.all((np.asarray(exp_d[40:]) >= 0) & (np.asarray(exp_d[40:]) < 32))


def test_net2net_initialization_properties():
    from src.cortical_expansion import apply_net2net_cortical_init

    cfg = ModelConfigV5(**PRESETS_V5["v5_compact_32shells"])
    params, buffers = init_params_v5(jax.random.PRNGKey(0), cfg)
    apply_net2net_cortical_init(params, num_frozen_shells=16, num_frozen_tracts=40)

    # Net2Net invariant properties
    assert np.all(np.asarray(params["core.intra_swiglu.w_down"][16:]) == 0.0)
    assert np.all(np.asarray(params["core.shortcut_weight"][40:]) == 0.0)
    assert np.all(np.asarray(params["core.shortcut_gate"][40:]) <= -10.0)
    assert np.all(np.asarray(params["core.w_radial_in"][15:]) == 0.0)
    assert np.all(np.asarray(params["core.w_radial_out"][15:]) == 0.0)
    assert np.all(np.asarray(params["motor.shell_score"][16:]) <= -1e4)


def test_zero_regression_step_0_equivalence():
    import os
    import torch
    from src.jax.v5_model import deliberate_step, emit_step, encode_sensory

    ckpt_path = "checkpoints/v5_small_best.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint {ckpt_path} not available for zero-regression integration test.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg16 = ModelConfigV5(**PRESETS_V5["v5_small"])
    params16, bufs16 = init_params_v5(jax.random.PRNGKey(42), cfg16)
    adapt_and_load_checkpoint_weights(ckpt, params16, jnp.float32, cfg16)

    cfg32 = ModelConfigV5(**PRESETS_V5["v5_compact_32shells"])
    params32, bufs32 = init_params_v5(jax.random.PRNGKey(42), cfg32)
    restored, bufs32 = adapt_and_load_checkpoint_weights(ckpt, params32, jnp.float32, cfg32, buffers=bufs32)

    prompt = jnp.array([[31, 1724, 338, 278, 7483, 310, 3444, 29973, 32]], dtype=jnp.int32)
    sensory_flux16, kv16 = encode_sensory(params16, prompt, config=cfg16)
    p_flux16 = sensory_flux16[:, -1, :]

    s16 = jnp.zeros((1, 16, cfg16.surface_dim)).at[:, 15, :].set(p_flux16)
    for _ in range(4):
        s16 = deliberate_step(params16, s16, bufs16)

    s32 = jnp.zeros((1, 32, cfg32.surface_dim)).at[:, 15, :].set(p_flux16).at[:, 31, :].set(p_flux16)
    for _ in range(4):
        s32 = deliberate_step(params32, s32, bufs32)

    plan16 = jnp.zeros((1, cfg16.surface_dim))
    plan32 = jnp.zeros((1, cfg32.surface_dim))

    logits16, _, _, _ = emit_step(params16, s16, kv16, plan16, None, 0, bufs16, config=cfg16)
    logits32, _, _, _ = emit_step(params32, s32, kv16, plan32, None, 0, bufs32, config=cfg32, encephalization_alpha=0.0)

    diff_logits = float(jnp.max(jnp.abs(logits16 - logits32)))
    assert diff_logits == 0.0, f"Step 0 logit difference must be exactly 0.0, got: {diff_logits}"

    tok16 = int(jnp.argmax(logits16, axis=-1)[0])
    tok32 = int(jnp.argmax(logits32, axis=-1)[0])
    assert tok16 == tok32, f"Token mismatch at step 0: 16-shell={tok16}, 32-shell={tok32}"

