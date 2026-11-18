"""
Unit Test Suite for Offline Reasoning, State Rollback & Latent Convergence
==========================================================================
"""

import math
import unittest
import torch
import torch.nn.functional as F

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from scripts.offline_reasoning_verification import RiverCrossingEnv


class TestOfflineReasoningEngine(unittest.TestCase):
    def test_river_crossing_environment_invariants(self):
        env = RiverCrossingEnv()
        self.assertEqual(env.state, [0, 0, 0, 0])

        # Illegal action: goat is not with farmer
        env.state = [1, 0, 0, 0]
        ok, msg, _ = env.apply_action("take_goat")
        self.assertFalse(ok)
        self.assertIn("ILLEGAL", msg)

        # Fatal violation: wolf and goat left alone
        is_val, reason = env.is_valid_state([0, 1, 1, 0])
        self.assertFalse(is_val)
        self.assertIn("Wolf eats goat", reason)

        # Fatal violation: goat and cabbage left alone
        is_val, reason = env.is_valid_state([0, 0, 1, 1])
        self.assertFalse(is_val)
        self.assertIn("Goat eats cabbage", reason)

    def test_system2_state_rollback_solver(self):
        env = RiverCrossingEnv()

        def solve(state, visited, path):
            if state == [1, 1, 1, 1]:
                return path
            moves = ["take_goat", "take_wolf", "take_cabbage", "take_none"]
            for m in moves:
                f, w, g, c = state
                new_s = list(state)
                new_f = 1 - f
                new_s[0] = new_f
                if m == "take_goat" and g == f:
                    new_s[2] = new_f
                elif m == "take_wolf" and w == f:
                    new_s[1] = new_f
                elif m == "take_cabbage" and c == f:
                    new_s[3] = new_f
                elif m == "take_none":
                    pass
                else:
                    continue

                valid, _ = env.is_valid_state(new_s)
                key = tuple(new_s)
                if valid and key not in visited:
                    visited.add(key)
                    res = solve(new_s, visited, path + [m])
                    if res is not None:
                        return res
                    visited.remove(key)
            return None

        plan = solve(env.state, {tuple(env.state)}, [])
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan), 7)
        for act in plan:
            ok, _, _ = env.apply_action(act)
            self.assertTrue(ok)
        self.assertTrue(env.is_solved())

    def test_latent_thinking_cycles_contraction(self):
        cfg = dict(PRESETS_V5["v5_small"])
        cfg["vocab_size"] = 500
        cfg["num_shells"] = 6
        cfg["surface_dim"] = 128
        cfg["radial_dim"] = 64
        model = CerebroSpheroidV5(ModelConfigV5(**cfg)).eval()

        prompt = torch.randint(1, 500, (1, 8))
        flux, _ = model.sensory_dnn(prompt)
        centroid = model._recency_weighted_pool(flux)
        sub = torch.zeros(1, 5, 128)
        state = torch.cat([sub, centroid.unsqueeze(1)], dim=1)

        deltas = []
        for _ in range(6):
            prev = state.clone()
            state = model.volumetric_core.deliberate_step(state)
            u_curr = F.normalize(state, dim=-1)
            u_prev = F.normalize(prev, dim=-1)
            deltas.append((u_curr - u_prev).norm().item())

        # Contraction check: velocity in later cycles must drop below peak exploration velocity
        self.assertLess(deltas[-1], max(deltas[:3]))


if __name__ == "__main__":
    unittest.main()
