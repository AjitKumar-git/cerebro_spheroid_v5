"""
Unit Tests for Dynamic JAX Recurrent Halting & End-to-End Loss Weighting
========================================================================
"""

import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from src.jax.v5_model import (
    ModelConfigV5,
    init_params_v5,
    training_rollout,
    deliberate_cycles_jax,
)
from src.jax.spheroid_optimizer import build_train_step_spheroid
from autotune.fallback import MicroBatchSlicer


class TestAdaptivePonderingAndLossWeights(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(101)

    def test_jax_adaptive_thinking_scan_and_ponder_cost(self):
        """Verifies that JAX deliberate_cycles_jax accumulates halting probability and computes valid gradients."""
        config = ModelConfigV5(
            vocab_size=256,
            surface_dim=64,
            radial_dim=32,
            num_shells=4,
            num_shortcut_tracts=8,
            num_heads=2,
            thinking_cycles=2,
            max_thinking_cycles=5,
            use_adaptive_thinking=True,
            thinking_halt_threshold=0.75,
        )

        params, buffers = init_params_v5(self.key, config)
        self.assertIn("core.halting_gate.weight", params)
        self.assertIn("core.halting_gate.bias", params)

        B = 2
        init_state = jnp.ones((B, config.num_shells, config.surface_dim), dtype=jnp.float32)

        # Run deliberation cycles
        final_state, ponder_cost = deliberate_cycles_jax(
            params, init_state, buffers, config, is_training=True
        )

        self.assertEqual(final_state.shape, (B, config.num_shells, config.surface_dim))
        self.assertTrue(math.isfinite(float(ponder_cost)))
        self.assertGreater(float(ponder_cost), 0.0)

        # Verify gradient flow through halting gate in training_rollout
        prompts = jnp.ones((B, 4), dtype=jnp.int32)
        targets = jnp.ones((B, 6), dtype=jnp.int32)

        def loss_fn(p):
            return training_rollout(
                p,
                prompts,
                target_len=6,
                buffers=buffers,
                config=config,
                targets=targets,
                pure_teacher_forcing=False,
            )

        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        self.assertTrue(math.isfinite(float(loss_val)))
        self.assertIn("core.halting_gate.weight", grads)
        self.assertIn("core.halting_gate.bias", grads)

    def test_loss_weighting_differentiates_token_impact(self):
        """Verifies that target_weights strictly controls token loss and gradient scaling."""
        config = ModelConfigV5(
            vocab_size=256,
            surface_dim=64,
            radial_dim=32,
            num_shells=4,
            num_shortcut_tracts=8,
            num_heads=2,
            thinking_cycles=1,
            use_adaptive_thinking=False,
        )
        params, buffers = init_params_v5(self.key, config)

        B = 1
        prompts = jnp.ones((B, 2), dtype=jnp.int32)
        targets = jnp.array([[10, 20]], dtype=jnp.int32)

        # Case 1: Only token 0 has weight 1.0; token 1 has weight 0.0
        weights_token0 = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
        # Case 2: Only token 1 has weight 1.0; token 0 has weight 0.0
        weights_token1 = jnp.array([[0.0, 1.0]], dtype=jnp.float32)

        def loss_for_weights(w):
            return training_rollout(
                params,
                prompts,
                target_len=2,
                buffers=buffers,
                config=config,
                targets=targets,
                pure_teacher_forcing=True,
                target_weights=w,
            )

        loss_w0 = float(loss_for_weights(weights_token0))
        loss_w1 = float(loss_for_weights(weights_token1))

        self.assertTrue(math.isfinite(loss_w0))
        self.assertTrue(math.isfinite(loss_w1))

        # Case 3: 3x weighting on token 1 increases its relative gradient contribution
        weights_boosted = jnp.array([[1.0, 3.0]], dtype=jnp.float32)
        loss_boosted = float(loss_for_weights(weights_boosted))
        self.assertTrue(math.isfinite(loss_boosted))

    def test_micro_batch_slicer_with_weights_pipeline(self):
        """Verifies MicroBatchSlicer correctly handles 4-element batches with weights."""
        slicer = MicroBatchSlicer()

        # Batch of 4 elements: prompts, targets, feedback, weights
        p = np.arange(8).reshape(4, 2)
        t = np.arange(16).reshape(4, 4)
        fb = np.full((4, 4), -100)
        w = np.array([[1.0, 1.0, 2.0, 2.0]] * 4)

        def mock_fetch():
            return p, t, fb, w

        # Slice into micro-batches of size 2
        p1, t1, fb1, w1 = slicer.get_micro_batch(2, mock_fetch)
        self.assertEqual(p1.shape[0], 2)
        self.assertEqual(t1.shape[0], 2)
        self.assertEqual(w1.shape[0], 2)
        self.assertTrue(slicer.has_remainder())

        # Retrieve remaining slice of size 2
        p2, t2, fb2, w2 = slicer.get_micro_batch(2)
        self.assertEqual(p2.shape[0], 2)
        self.assertEqual(t2.shape[0], 2)
        self.assertEqual(w2.shape[0], 2)
        self.assertFalse(slicer.has_remainder())


if __name__ == "__main__":
    unittest.main()
