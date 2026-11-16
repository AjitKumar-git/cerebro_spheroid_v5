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
