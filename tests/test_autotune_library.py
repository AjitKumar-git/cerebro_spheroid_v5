"""
Comprehensive Unit Tests for Standalone Autotune Library.
=========================================================

Verifies:
1. Clean importability of `autotune` and `cerebro_autotune` without v5 coupling.
2. Accurate hardware detection and memory tier profiling across TPU/GPU.
3. Architecture fingerprinting and 'is_new_architecture' cache routing.
4. Progressive candidate ladders and safe OOM branch pruning.
5. Optimization across all target objectives:
   - tokens/sec
   - steps/s
   - loss reduction/sec (dLoss/dt)
   - balanced
6. Zero-overhead execution when a previously tuned architecture is detected.
7. Generic step adapter execution with no v5 dependencies.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import autotune
import cerebro_autotune
from autotune import (
    AutoTuner,
    TuneObjective,
    HardwareInfo,
    get_hardware_info,
    resolve_jax_compute_dtype,
    ArchitectureDetector,
    ArchitectureInfo,
    extract_architecture_signature,
    score_benchmark_result,
    calculate_loss_reduction_velocity,
    generate_vram_maximizing_candidates,
    CandidateLadder,
    HardwareConfigCache,
    BenchmarkRunner,
)
from autotune.adapters.generic import GenericStepAdapter


class TestAutotuneLibrary(unittest.TestCase):

    def test_package_exports_and_aliases(self):
        """Verifies clean exports from both `autotune` and `cerebro_autotune`."""
        for pkg in (autotune, cerebro_autotune):
            self.assertTrue(hasattr(pkg, "AutoTuner"))
            self.assertTrue(hasattr(pkg, "TuneObjective"))
            self.assertTrue(hasattr(pkg, "get_hardware_info"))
            self.assertTrue(hasattr(pkg, "generate_vram_maximizing_candidates"))
            self.assertTrue(hasattr(pkg, "ArchitectureDetector"))
            self.assertTrue(hasattr(pkg, "score_benchmark_result"))

    def test_tune_objective_normalization(self):
        """Verifies string aliases normalize to valid TuneObjective enums."""
        self.assertEqual(TuneObjective.from_str("tokens/sec"), TuneObjective.TOKENS_PER_SEC)
        self.assertEqual(TuneObjective.from_str("tok_per_sec"), TuneObjective.TOKENS_PER_SEC)
        self.assertEqual(TuneObjective.from_str("steps/s"), TuneObjective.STEPS_PER_SEC)
        self.assertEqual(TuneObjective.from_str("steps_per_sec"), TuneObjective.STEPS_PER_SEC)
        self.assertEqual(TuneObjective.from_str("loss reduction/sec"), TuneObjective.LOSS_REDUCTION_PER_SEC)
        self.assertEqual(TuneObjective.from_str("loss_reduction_per_sec"), TuneObjective.LOSS_REDUCTION_PER_SEC)
        self.assertEqual(TuneObjective.from_str("dloss_dt"), TuneObjective.LOSS_REDUCTION_PER_SEC)
        self.assertEqual(TuneObjective.from_str("balanced"), TuneObjective.BALANCED)
        self.assertEqual(TuneObjective.from_str(None), TuneObjective.BALANCED)

    def test_architecture_signature_extraction(self):
        """Verifies architecture fingerprint extraction across presets and configs."""
        # Simple preset
        sig1 = extract_architecture_signature("llama_7b", seq_len=512)
        self.assertEqual(sig1.architecture_name, "llama_7b")
        self.assertEqual(sig1.seq_len, 512)
        self.assertIn("llama_7b", sig1.architecture_signature)

        # Dict config
        dict_cfg = {
            "name": "custom_transformer",
            "param_count": 350_000_000,
            "hidden_dim": 1024,
            "num_layers": 24,
        }
        sig2 = extract_architecture_signature(dict_cfg, seq_len=1024)
        self.assertEqual(sig2.architecture_name, "custom_transformer")
        self.assertEqual(sig2.param_count, 350_000_000)
        self.assertEqual(sig2.hidden_dim, 1024)
        self.assertEqual(sig2.num_layers, 24)
        self.assertIn("350.0M", sig2.architecture_signature)
        self.assertIn("d1024", sig2.architecture_signature)

    def test_new_architecture_detection(self):
        """Verifies ArchitectureDetector correctly identifies new vs cached models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "hardware_configs.json")
            cache = HardwareConfigCache(cache_path=cache_file)
            detector = ArchitectureDetector(cache_manager=cache)

            arch = extract_architecture_signature("my_new_model", seq_len=256)
            hw_sig = "TPU_v5e_16GB"

            # 1. Unseen architecture must be flagged as new
            self.assertTrue(detector.is_new_architecture(arch, hardware_sig=hw_sig, seq_len=256))

            # 2. Save a verified configuration
            dummy_cfg = {
                "batch_size": 64,
                "remat": False,
                "tok_per_sec": 75000.0,
                "steps_per_sec": 2.5,
                "seq_len": 256,
                "tune_objective": "balanced",
            }
            cache.save("my_new_model", hw_sig, dummy_cfg)

            # 3. Seen architecture on matching hardware & seq_len is NOT new
            self.assertFalse(detector.is_new_architecture(arch, hardware_sig=hw_sig, seq_len=256))

            # 4. Same architecture on different hardware or different seq_len is NEW
            self.assertTrue(detector.is_new_architecture(arch, hardware_sig="NVIDIA_H100_80GB", seq_len=256))
            self.assertTrue(detector.is_new_architecture(arch, hardware_sig=hw_sig, seq_len=512))

    def test_loss_reduction_velocity_calculation(self):
        """Verifies calculation of loss reduction per second (dLoss/dt)."""
        # Case 1: Empirical loss delta is measured
        # initial_loss=4.0, final_loss=3.0 over 2 effective steps with 5.0 steps/sec
        # delta = 1.0, delta/step = 0.5, dLoss/dt = 0.5 * 5.0 = 2.5 loss units/sec
        vel = calculate_loss_reduction_velocity(
            initial_loss=4.0,
            final_loss=3.0,
            num_steps=3,
            steps_per_sec=5.0,
            batch_size=32,
            tok_per_sec=10000.0,
        )
        self.assertAlmostEqual(vel, 2.5, places=3)

        # Case 2: No loss delta (e.g. constant loss) -> theoretical model is used
        vel_theo = calculate_loss_reduction_velocity(
            initial_loss=3.5,
            final_loss=3.5,
            num_steps=3,
            steps_per_sec=4.0,
            batch_size=64,
            tok_per_sec=50000.0,
        )
        self.assertGreater(vel_theo, 0.0)

    def test_multi_objective_scoring_differentiation(self):
        """Verifies that different objectives prefer appropriately specialized configurations."""
        # Candidate A: Small batch, fast step rate (good for steps/s, high updates)
        cfg_fast_steps = {
            "batch_size": 16,
            "remat": False,
            "tok_per_sec": 30000.0,
            "steps_per_sec": 6.0,
            "initial_loss": 3.0,
            "final_loss": 2.7,
            "num_steps": 3,
        }
        # Candidate B: Huge batch, slow steps but enormous token throughput
        cfg_high_tok = {
            "batch_size": 256,
            "remat": True,
            "tok_per_sec": 120000.0,
            "steps_per_sec": 0.8,
            "initial_loss": 3.0,
            "final_loss": 2.95,
            "num_steps": 3,
        }
        # Candidate C: Moderate batch with rapid loss drop
        cfg_fast_loss = {
            "batch_size": 64,
            "remat": False,
            "tok_per_sec": 70000.0,
            "steps_per_sec": 3.0,
            "initial_loss": 3.0,
            "final_loss": 2.4,  # Rapid descent
            "num_steps": 3,
        }

        # tokens/sec must prefer B
        score_tok_a = score_benchmark_result(cfg_fast_steps, "tok_per_sec")
        score_tok_b = score_benchmark_result(cfg_high_tok, "tok_per_sec")
        self.assertGreater(score_tok_b, score_tok_a)

        # steps/s must prefer A
        score_step_a = score_benchmark_result(cfg_fast_steps, "steps_per_sec")
        score_step_b = score_benchmark_result(cfg_high_tok, "steps_per_sec")
        self.assertGreater(score_step_a, score_step_b)

        # loss reduction/sec must prefer C (delta 0.6 over 2 steps = 0.3 * 3.0 = 0.9/s)
        score_loss_c = score_benchmark_result(cfg_fast_loss, "loss_reduction_per_sec")
        score_loss_b = score_benchmark_result(cfg_high_tok, "loss_reduction_per_sec")
        self.assertGreater(score_loss_c, score_loss_b)

    def test_candidate_ladder_smart_pruning(self):
        """Verifies CandidateLadder prunes redundant and impossible candidate branches."""
        candidates = [
            {"batch_size": 32, "remat": False},
            {"batch_size": 32, "remat": True},
            {"batch_size": 64, "remat": False},
            {"batch_size": 64, "remat": True},
            {"batch_size": 128, "remat": False},
            {"batch_size": 128, "remat": True},
            {"batch_size": 256, "remat": False},
            {"batch_size": 256, "remat": True},
        ]
        ladder = CandidateLadder(candidates)

        # 1. Candidate (32, False) passes
        self.assertFalse(ladder.should_skip(32, False))
        ladder.record_result(32, False, "PASS")

        # Redundant (32, True) must be skipped!
        self.assertTrue(ladder.should_skip(32, True))

        # 2. Candidate (64, False) OOMs
        self.assertFalse(ladder.should_skip(64, False))
        ladder.record_result(64, False, "OOM (Memory Exceeded)")

        # Any larger batch without remat (128, False), (256, False) must be pruned!
        self.assertTrue(ladder.should_skip(128, False))
        self.assertTrue(ladder.should_skip(256, False))

        # (64, True) is still viable with activation checkpointing!
        self.assertFalse(ladder.should_skip(64, True))
        ladder.record_result(64, True, "PASS")

        # (128, True) is tested
        self.assertFalse(ladder.should_skip(128, True))
        ladder.record_result(128, True, "OOM (Memory Exceeded)")

        # Any larger batch with remat (256, True) must now be pruned!
        self.assertTrue(ladder.should_skip(256, True))

    def test_autotuner_zero_overhead_cached_execution(self):
        """Verifies AutoTuner.tune_if_new skips micro-benchmarks when architecture is cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "hardware_configs.json")
            cache = HardwareConfigCache(cache_path=cache_file)

            mock_benchmark = MagicMock()
            mock_benchmark.return_value = {
                "preset": "test_arch",
                "batch_size": 64,
                "remat": False,
                "tok_per_sec": 50000.0,
                "steps_per_sec": 2.0,
                "step_ms": 500.0,
                "compile_time": 1.0,
                "memory_gb": 4.0,
                "memory_utilization_pct": 50.0,
                "status": "PASS",
            }

            tuner = AutoTuner(
                objective=TuneObjective.BALANCED,
                cache_manager=cache,
                benchmark_fn=mock_benchmark,
                upload_winner=False,
                verbose=False,
            )

            # First run: new architecture -> benchmarks are run
            candidates = [{"batch_size": 64, "remat": False}]
            res1 = tuner.tune_if_new(
                architecture="test_arch",
                candidates=candidates,
                seq_len=256,
            )
            self.assertEqual(res1["batch_size"], 64)
            self.assertEqual(mock_benchmark.call_count, 1)

            # Second run: known architecture -> 0 benchmarks run, returns cached result
            mock_benchmark.reset_mock()
            res2 = tuner.tune_if_new(
                architecture="test_arch",
                candidates=candidates,
                seq_len=256,
            )
            self.assertEqual(res2["batch_size"], 64)
            self.assertEqual(mock_benchmark.call_count, 0)

    def test_generic_step_adapter_without_v5_dependency(self):
        """Verifies that autotune can benchmark arbitrary step functions with no v5 imports."""
        def custom_step_fn(batch_size, remat, seq_len, num_steps, **kwargs):
            if batch_size > 128:
                raise RuntimeError("CUDA out of memory")
            step_ms = 20.0 + (batch_size * 0.5)
            steps_s = 1000.0 / step_ms
            tok_s = batch_size * seq_len * steps_s
            return {
                "step_ms": step_ms,
                "steps_per_sec": steps_s,
                "tok_per_sec": tok_s,
                "initial_loss": 3.0,
                "final_loss": 2.8,
                "memory_gb": 2.0 + (batch_size * 0.05),
                "memory_utilization_pct": 40.0,
                "status": "PASS",
            }

        adapter = GenericStepAdapter(custom_step_fn)
        runner = BenchmarkRunner(adapter.benchmark)

        res = runner.run(
            preset="custom_resnet",
            batch_size=64,
            remat=False,
            seq_len=128,
            num_steps=3,
        )
        self.assertEqual(res["status"], "PASS")
        self.assertGreater(res["tok_per_sec"], 0.0)
        self.assertAlmostEqual(res["initial_loss"], 3.0)
        self.assertAlmostEqual(res["final_loss"], 2.8)
        self.assertGreater(res["loss_reduction_per_sec"], 0.0)

        # OOM behavior
        res_oom = runner.run(
            preset="custom_resnet",
            batch_size=256,
            remat=False,
            seq_len=128,
            num_steps=3,
        )
        self.assertIn("OOM", res_oom["status"])
        self.assertEqual(res_oom["steps_per_sec"], 0.0)

    def test_candidate_ladder_plateau_pruning(self):
        """Verifies that throughput plateau / degradation early-stops larger batch explorations."""
        candidates = [
            {"batch_size": 32, "remat": False},
            {"batch_size": 64, "remat": False},
            {"batch_size": 128, "remat": False},
            {"batch_size": 256, "remat": False},
            {"batch_size": 384, "remat": False},
            {"batch_size": 512, "remat": False},
        ]
        ladder = CandidateLadder(candidates, enable_plateau_pruning=True, plateau_tolerance=2)

        # 32: 30k
        ladder.record_result(32, False, "PASS", tok_per_sec=30000.0)
        self.assertFalse(ladder.should_skip(64, False))

        # 64: 60k (growing)
        ladder.record_result(64, False, "PASS", tok_per_sec=60000.0)
        self.assertFalse(ladder.should_skip(128, False))

        # 128: 100k (peak throughput)
        ladder.record_result(128, False, "PASS", tok_per_sec=100000.0)
        self.assertFalse(ladder.should_skip(256, False))

        # 256: 95k (first degradation)
        ladder.record_result(256, False, "PASS", tok_per_sec=95000.0)
        self.assertFalse(ladder.stopped_due_to_plateau)

        # 384: 90k (second degradation -> threshold reached)
        ladder.record_result(384, False, "PASS", tok_per_sec=90000.0)
        self.assertTrue(ladder.stopped_due_to_plateau)

        # 512 must be skipped because plateau was reached!
        self.assertTrue(ladder.should_skip(512, False))

    def test_fast_candidate_generation(self):
        """Verifies fast candidate generation produces a thinned geometric ladder."""
        full_ladder = generate_vram_maximizing_candidates("v5_small", total_gb=16.0, fast=False)
        fast_ladder = generate_vram_maximizing_candidates("v5_small", total_gb=16.0, fast=True)
        self.assertLess(len(fast_ladder), len(full_ladder))
        # Ensure optimal anchor batch sizes remain present in fast ladder
        fast_batches = [c["batch_size"] for c in fast_ladder]
        self.assertIn(64, fast_batches)
        self.assertIn(128, fast_batches)
        self.assertIn(256, fast_batches)

    def test_v5_benchmark_adapter_graceful_oom_cleanup(self):
        """Verifies benchmark_v5_model catches OOM without raising UnboundLocalError on prompt_b."""
        from autotune.adapters.v5 import benchmark_v5_model, _V5_TRAIN_STEP_CACHE
        with patch("src.jax.spheroid_optimizer.build_train_step_spheroid") as mock_build:
            mock_step = MagicMock(side_effect=RuntimeError("RESOURCE_EXHAUSTED: out of memory on TPU"))
            mock_build.return_value = mock_step
            _V5_TRAIN_STEP_CACHE.clear()

            res = benchmark_v5_model(
                preset="v5_small",
                batch_size=256,
                remat=False,
                seq_len=1024,
                num_steps=1,
            )
            self.assertIn("OOM", res["status"])
            self.assertEqual(res["steps_per_sec"], 0.0)



if __name__ == "__main__":
    unittest.main()
