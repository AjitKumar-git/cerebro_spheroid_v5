# -*- coding: utf-8 -*-
"""
Unit tests for CerebroSpheroid V5 Smart Dynamic Mixture Engine
=============================================================
Tests:
  1. Bounded simplex projection and safety clamp enforcement [alpha_min, alpha_max].
  2. Multi-domain loss tracking and gradient learning velocity computation.
  3. Overfitting detection and automatic allocation dampening.
  4. Saturation detection and compute reallocation away from flat-loss domains.
  5. Checkpoint serialization and lossless state resumption.
  6. Integration with TokenQuotaManager in-flight target updates.
  7. Label-to-tag normalization mapping.
"""

import unittest
from typing import Dict

from src.rectified_streamer import TokenQuotaManager
from src.smart_mixture import DomainMetrics, SmartMixtureController, map_domain_label_to_tag


class TestSmartMixture(unittest.TestCase):
    def setUp(self):
        self.initial_shares = {
            "broad_web": 0.58,
            "code": 0.24,
            "math": 0.14,
            "cosmopedia": 0.04,
        }
        self.controller = SmartMixtureController(
            initial_shares=self.initial_shares,
            eta=0.10,
            prior_weight=0.15,
            overfit_threshold=0.03,
        )

    def test_bounded_simplex_projection(self):
        """Ensures that weights always sum to 1.0 and respect domain safety clamps."""
        raw_weights = {
            "broad_web": 0.95,  # Above max (0.75)
            "code": 0.01,       # Below min (0.05)
            "math": 0.02,       # Below min (0.05)
            "cosmopedia": 0.02, # Below min (0.05)
        }
        projected = self.controller.project_to_bounded_simplex(raw_weights)

        # Sum must equal 1.0
        self.assertAlmostEqual(sum(projected.values()), 1.0, places=4)

        # Bounds must be strictly respected
        self.assertLessEqual(projected["broad_web"], 0.75 + 1e-4)
        self.assertGreaterEqual(projected["broad_web"], 0.10 - 1e-4)
        self.assertGreaterEqual(projected["code"], 0.05 - 1e-4)
        self.assertGreaterEqual(projected["math"], 0.05 - 1e-4)
        self.assertGreaterEqual(projected["cosmopedia"], 0.05 - 1e-4)

    def test_learning_velocity_and_reward(self):
        """High-velocity domains learning rapidly should receive increased compute allocation."""
        # Step 1: Initial baseline eval
        step1_losses = {"broad_web": 2.80, "code": 1.40, "math": 2.50, "cosmopedia": 2.30}
        step1_toks = {"broad_web": 1_000_000, "code": 500_000, "math": 200_000, "cosmopedia": 100_000}
        shares1, metrics1 = self.controller.step(100, step1_losses, step1_toks)

        # Step 2: Math and cosmopedia drop dramatically (high velocity), broad web is slow
        step2_losses = {"broad_web": 2.79, "code": 1.38, "math": 2.10, "cosmopedia": 1.90}
        step2_toks = {"broad_web": 2_000_000, "code": 1_000_000, "math": 400_000, "cosmopedia": 200_000}
        shares2, metrics2 = self.controller.step(200, step2_losses, step2_toks)

        # Math and Cosmopedia should have strong positive velocity
        self.assertGreater(metrics2["math"].smoothed_velocity, metrics2["broad_web"].smoothed_velocity)
        self.assertGreater(metrics2["cosmopedia"].smoothed_velocity, metrics2["broad_web"].smoothed_velocity)

        # Math share should increase relative to prior or initial
        self.assertGreater(shares2["math"], self.initial_shares["math"])
        self.assertGreater(shares2["cosmopedia"], self.initial_shares["cosmopedia"])

    def test_overfitting_penalty_and_dampening(self):
        """A domain whose validation loss climbs should be flagged as overfitting and dampened."""
        # Step 1: Baseline
        self.controller.step(100, {"broad_web": 2.80, "code": 1.20, "math": 2.20}, {"broad_web": 1e6, "code": 1e6, "math": 1e6})

        # Step 2: Code loss climbs significantly (overfitting/memorization)
        shares, metrics = self.controller.step(200, {"broad_web": 2.75, "code": 1.35, "math": 2.10}, {"broad_web": 2e6, "code": 2e6, "math": 2e6})

        self.assertTrue(metrics["code"].is_overfitting)
        self.assertGreaterEqual(metrics["code"].overfit_streak, 1)
        self.assertLess(metrics["code"].overfit_penalty, 1.0)

        # Step 3: Continued overfit streak further dampens allocation
        shares3, metrics3 = self.controller.step(300, {"broad_web": 2.70, "code": 1.45, "math": 2.00}, {"broad_web": 3e6, "code": 3e6, "math": 3e6})
        self.assertEqual(metrics3["code"].overfit_streak, 2)
        # Code share should be pushed toward its min_shares clamp
        self.assertLessEqual(shares3["code"], 0.15)
        self.assertGreaterEqual(shares3["code"], 0.05)

    def test_saturation_detection(self):
        """A domain with zero loss change across multiple evals is flagged as saturated."""
        # 4 consecutive steps where broad_web loss does not budge
        losses = {"broad_web": 2.7500, "code": 1.20, "math": 2.20}
        toks = {"broad_web": 1e6, "code": 1e6, "math": 1e6}
        self.controller.step(100, losses, toks)

        for s in [200, 300, 400]:
            toks = {k: v + 1e6 for k, v in toks.items()}
            # broad_web completely flat
            losses["code"] -= 0.05
            losses["math"] -= 0.05
            shares, metrics = self.controller.step(s, losses, toks)

        self.assertTrue(metrics["broad_web"].is_saturated)
        self.assertGreaterEqual(metrics["broad_web"].saturation_streak, 1)

    def test_state_serialization_and_restoration(self):
        """Controller state must be fully serializable to dict and recoverable without loss."""
        # Feed some data
        self.controller.step(100, {"broad_web": 2.80, "code": 1.30}, {"broad_web": 1e6, "code": 5e5})
        self.controller.step(200, {"broad_web": 2.75, "code": 1.25}, {"broad_web": 2e6, "code": 1e6})

        saved_state = self.controller.to_dict()

        # Create new controller and restore
        new_controller = SmartMixtureController(initial_shares={"broad_web": 0.5, "code": 0.5})
        new_controller.load_dict(saved_state)

        self.assertDictEqual(self.controller.best_val_loss, new_controller.best_val_loss)
        self.assertDictEqual(self.controller.overfit_streak, new_controller.overfit_streak)
        self.assertDictEqual(self.controller.smoothed_velocity, new_controller.smoothed_velocity)
        for d in self.controller.current_shares:
            self.assertAlmostEqual(self.controller.current_shares[d], new_controller.current_shares[d], places=5)

    def test_token_quota_manager_dynamic_update(self):
        """TokenQuotaManager must allow dynamic target updating without resetting historical counts."""
        quota_mgr = TokenQuotaManager({"broad_web": 0.80, "code": 0.20})
        quota_mgr.record_tokens("broad_web", 8000)
        quota_mgr.record_tokens("code", 2000)

        snap1 = quota_mgr.get_snapshot()
        self.assertEqual(snap1["broad_web"]["tokens"], 8000)
        self.assertEqual(snap1["code"]["tokens"], 2000)

        # Update targets to 50/50
        quota_mgr.update_target_shares({"broad_web": 0.50, "code": 0.50})
        snap2 = quota_mgr.get_snapshot()

        # Historical tokens must be intact
        self.assertEqual(snap2["broad_web"]["tokens"], 8000)
        self.assertEqual(snap2["code"]["tokens"], 2000)
        self.assertEqual(snap2["code"]["target_pct"], 50.0)

        # Next domain should now choose 'code' because it has a deficit under the 50/50 target!
        self.assertEqual(quota_mgr.get_next_domain(), "code")

    def test_domain_label_normalization(self):
        """Display names and verbose labels must map to clean canonical curriculum tags."""
        self.assertEqual(map_domain_label_to_tag("Broad Web (FineWeb sample-10BT)"), "broad_web")
        self.assertEqual(map_domain_label_to_tag("Code Repositories (The Stack v2)"), "code")
        self.assertEqual(map_domain_label_to_tag("Mathematics & Proofs (FineMath)"), "math")
        self.assertEqual(map_domain_label_to_tag("Synthetic Textbooks (Cosmopedia v2)"), "cosmopedia")
        self.assertEqual(map_domain_label_to_tag("Educational & Academic (FineWeb-Edu)"), "edu_academic")
        self.assertEqual(map_domain_label_to_tag("Associative Retrieval"), "retrieval")


if __name__ == "__main__":
    unittest.main()
