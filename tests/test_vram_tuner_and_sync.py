"""
Tests for CerebroSpheroid V5: Hardware VRAM Auto-Tuning & bioflo.in Remote Sync Engine.
========================================================================================

Verifies that:
1. get_hardware_info correctly detects device topology and available memory tiers.
2. generate_vram_maximizing_candidates dynamically scales batch sizes and remat exploration
   across 4GB, 8GB, 16GB, and 32GB+ memory tiers.
3. benchmark_config calculates peak VRAM utilization and token throughput without crashing.
4. Local caching and remote bioflo.in sync correctly round-trip winning hyperparameters.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp

from src.jax.auto_tuner import (
    get_hardware_info,
    generate_vram_maximizing_candidates,
    benchmark_config,
    auto_tune_environment,
    load_local_hardware_config,
    save_local_hardware_config,
    _score_benchmark_result,
)
from src.cloud_sync import (
    detect_source_environment,
    fetch_remote_hardware_config,
    push_remote_hardware_config,
    select_highest_step_slot,
)


class TestVRAMTunerAndSync(unittest.TestCase):

    def test_hardware_profiling(self):
        """Validates hardware detection and signature synthesis."""
        hw = get_hardware_info()
        self.assertIn("device_kind", hw)
        self.assertIn("platform", hw)
        self.assertIn("total_memory_bytes", hw)
        self.assertIn("total_gb", hw)
        self.assertIn("hardware_signature", hw)
        self.assertGreater(hw["total_gb"], 0.0)
        self.assertIsInstance(hw["hardware_signature"], str)

    def test_candidate_generation_memory_tiers(self):
        """Verifies candidate search ladders push VRAM utilization across tiers."""
        # Ultra tier (64GB+, e.g. TPU v5p 95GB, H100 80GB)
        c_64 = generate_vram_maximizing_candidates("v5_small", total_gb=64.0, seq_len=256)
        batch_sizes_64 = [c["batch_size"] for c in c_64]
        self.assertIn(1024, batch_sizes_64)
        self.assertIn(2048, batch_sizes_64)
        self.assertIn(4096, batch_sizes_64)

        # 32GB tier (e.g. TPU v6e Trillium 32GB, TPU v4 32GB, A100-40GB)
        c_32 = generate_vram_maximizing_candidates("tpu_v6e_1", total_gb=32.0, seq_len=256)
        batch_sizes_32 = [c["batch_size"] for c in c_32]
        self.assertIn(256, batch_sizes_32)
        self.assertIn(512, batch_sizes_32)
        self.assertIn(1024, batch_sizes_32)
        self.assertIn(2048, batch_sizes_32)
        # Verify remat=False is also evaluated for higher throughput when memory permits
        remat_false_32 = [c for c in c_32 if not c["remat"]]
        self.assertGreaterEqual(len(remat_false_32), 1)

        # 16GB tier (e.g. TPU v5e-1, T4 16GB) - expanded beyond 384
        c_16 = generate_vram_maximizing_candidates("tpu_v5e_1", total_gb=16.0, seq_len=256)
        batch_sizes_16 = [c["batch_size"] for c in c_16]
        self.assertIn(128, batch_sizes_16)
        self.assertIn(256, batch_sizes_16)
        self.assertIn(384, batch_sizes_16)
        self.assertIn(512, batch_sizes_16)
        self.assertIn(768, batch_sizes_16)
        self.assertIn(1024, batch_sizes_16)

        # 8GB tier (e.g. desktop RTX 3070 8GB)
        c_8 = generate_vram_maximizing_candidates("v5_small", total_gb=8.0, seq_len=256)
        batch_sizes_8 = [c["batch_size"] for c in c_8]
        self.assertIn(64, batch_sizes_8)
        self.assertIn(96, batch_sizes_8)

    def test_benchmark_config_vram_tracking(self):
        """Executes a micro-benchmark step and verifies VRAM and tok/s metrics."""
        res = benchmark_config(
            preset="v5_test",
            batch_size=2,
            remat=True,
            seq_len=64,
            num_steps=2,
            dtype="fp32",
        )
        self.assertEqual(res["status"], "PASS")
        self.assertGreater(res["tok_per_sec"], 0.0)
        self.assertGreater(res["compile_time"], 0.0)
        self.assertGreater(res["vram_gb"], 0.0)
        self.assertGreater(res["vram_utilization_pct"], 0.0)

    def test_local_config_caching_roundtrip(self):
        """Verifies storing and retrieving cached hardware configurations."""
        dummy_sig = "Mock_Test_Accelerator_16GB"
        dummy_cfg = {
            "batch_size": 128,
            "remat": True,
            "seq_len": 256,
            "tok_per_sec": 75000.0,
            "step_ms": 435.2,
            "vram_gb": 12.8,
            "vram_utilization_pct": 80.0,
        }
        save_local_hardware_config("v5_test", dummy_sig, dummy_cfg)
        loaded = load_local_hardware_config("v5_test", dummy_sig)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["batch_size"], 128)
        self.assertEqual(loaded["tok_per_sec"], 75000.0)

    @patch("urllib.request.urlopen")
    def test_remote_cloud_sync_get_config(self, mock_urlopen):
        """Tests fetch_remote_hardware_config parsing from bioflo.in endpoint."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "status": "found",
            "model": "v5_small",
            "device": "TPU_v5e_16GB",
            "config": {
                "batch_size": 256,
                "remat": True,
                "tok_per_sec": 84200.0,
                "vram_gb": 13.5,
                "vram_utilization_pct": 84.4,
            }
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = fetch_remote_hardware_config("v5_small", "TPU_v5e_16GB")
        self.assertIsNotNone(res)
        self.assertEqual(res["batch_size"], 256)
        self.assertEqual(res["tok_per_sec"], 84200.0)

    @patch("urllib.request.urlopen")
    def test_remote_cloud_sync_push_config(self, mock_urlopen):
        """Tests push_remote_hardware_config payload delivery to bioflo.in endpoint."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "success": True,
            "status": "saved",
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        ok = push_remote_hardware_config(
            model="v5_small",
            device_signature="TPU_v5e_16GB",
            config={
                "batch_size": 256,
                "remat": True,
                "tok_per_sec": 84200.0,
                "step_ms": 776.0,
                "vram_gb": 13.5,
                "vram_utilization_pct": 84.4,
            }
        )
        self.assertTrue(ok)

    def test_auto_tune_with_local_cached_skip(self):
        """Verifies auto_tune_environment bypasses micro-benchmarks if cached config exists."""
        dummy_sig = get_hardware_info()["hardware_signature"]
        cached_winner = {
            "batch_size": 64,
            "remat": True,
            "seq_len": 64,
            "tok_per_sec": 99999.0,
            "step_ms": 100.0,
            "vram_gb": 4.0,
            "vram_utilization_pct": 50.0,
        }
        save_local_hardware_config("v5_test", dummy_sig, cached_winner)

        # Run auto_tune_environment with cached config enabled
        winner = auto_tune_environment(
            preset="v5_test",
            seq_len=64,
            use_cached_config=True,
            force_tune=False,
            upload_winner=False,
            verbose=False,
        )
        self.assertEqual(winner["tok_per_sec"], 99999.0)
        self.assertEqual(winner["batch_size"], 64)

    def test_tune_objective_selection_and_remat_ladder(self):
        """Verifies remat=False comes before remat=True per batch and tune_objective distinguishes speed profiles."""
        c_16 = generate_vram_maximizing_candidates("v5_small", total_gb=16.0, seq_len=256)
        self.assertGreaterEqual(len(c_16), 2)
        for i in range(0, len(c_16), 2):
            self.assertEqual(c_16[i]["batch_size"], c_16[i + 1]["batch_size"])
            self.assertFalse(c_16[i]["remat"])
            self.assertTrue(c_16[i + 1]["remat"])

        fast_step_cfg = {"batch_size": 64, "remat": False, "tok_per_sec": 65566.0, "steps_per_sec": 4.00}
        balanced_cfg = {"batch_size": 128, "remat": False, "tok_per_sec": 95000.0, "steps_per_sec": 2.30}
        heavy_batch_cfg = {"batch_size": 384, "remat": True, "tok_per_sec": 82696.0, "steps_per_sec": 0.84}

        # tok_per_sec prefers highest token rate regardless of slow steps/s
        self.assertGreater(
            _score_benchmark_result(balanced_cfg, "tok_per_sec"),
            _score_benchmark_result(heavy_batch_cfg, "tok_per_sec"),
        )
        # steps_per_sec prioritizes highest update rate
        self.assertGreater(
            _score_benchmark_result(fast_step_cfg, "steps_per_sec"),
            _score_benchmark_result(balanced_cfg, "steps_per_sec"),
        )
        self.assertGreater(
            _score_benchmark_result(fast_step_cfg, "steps_per_sec"),
            _score_benchmark_result(heavy_batch_cfg, "steps_per_sec"),
        )
        # balanced penalizes high-latency large batches in favor of Pareto efficiency
        self.assertGreater(
            _score_benchmark_result(balanced_cfg, "balanced"),
            _score_benchmark_result(heavy_batch_cfg, "balanced"),
        )
        self.assertGreater(
            _score_benchmark_result(fast_step_cfg, "balanced"),
            _score_benchmark_result(heavy_batch_cfg, "balanced"),
        )

    @patch("src.jax.auto_tuner.save_local_hardware_config")
    @patch("src.jax.auto_tuner.benchmark_config")
    def test_auto_tune_remat_oom_fallback(self, mock_bench, _mock_save):
        """Verifies remat=False passes skip redundant remat=True and OOM stops larger remat=False batches."""
        seen_calls = []

        def _fake_bench(preset="v5_small", batch_size=32, remat=True, **kwargs):
            seen_calls.append((batch_size, remat))
            if not remat and batch_size >= 256:
                return {
                    "preset": preset,
                    "batch_size": batch_size,
                    "remat": remat,
                    "tok_per_sec": 0.0,
                    "steps_per_sec": 0.0,
                    "step_ms": None,
                    "compile_time": None,
                    "status": "OOM",
                }
            steps_s = 256.0 / float(batch_size)
            tok_s = 50000.0 + batch_size * 100.0 + (2000.0 if not remat else 0.0)
            return {
                "preset": preset,
                "batch_size": batch_size,
                "remat": remat,
                "tok_per_sec": tok_s,
                "steps_per_sec": steps_s,
                "step_ms": 1000.0 / steps_s,
                "compile_time": 1.0,
                "memory_gb": 8.0,
                "memory_utilization_pct": 50.0,
                "memory_type": "HBM",
                "vram_gb": 8.0,
                "vram_utilization_pct": 50.0,
                "status": "PASS",
            }

        mock_bench.side_effect = _fake_bench
        candidates = [
            {"batch_size": 64, "remat": False},
            {"batch_size": 64, "remat": True},
            {"batch_size": 256, "remat": False},
            {"batch_size": 256, "remat": True},
            {"batch_size": 384, "remat": False},
            {"batch_size": 384, "remat": True},
        ]
        winner = auto_tune_environment(
            preset="v5_test",
            candidates=candidates,
            seq_len=64,
            use_cached_config=False,
            upload_winner=False,
            verbose=False,
            tune_objective="steps_per_sec",
        )
        # (64, True) skipped since (64, False) passed; (384, False) skipped after (256, False) OOMed
        self.assertEqual(seen_calls, [(64, False), (256, False), (256, True), (384, True)])
        self.assertEqual(winner["batch_size"], 64)
        self.assertFalse(winner["remat"])

    def test_select_highest_step_slot_prefers_newest_across_model_and_root(self):
        """Ensure checkpoint slot selection picks highest recorded step across model & root slots."""
        manifest_root_newer = {
            "manifest": {
                "v5_small_latest": {"step": 4750, "loss": 3.4084},
                "latest": {"step": 9750, "loss": 2.9815},
            }
        }
        slot, step, loss = select_highest_step_slot(manifest_root_newer, model="tpu_v5e_1", target_type="latest")
        self.assertIsNone(slot)
        self.assertEqual(step, 9750)
        self.assertAlmostEqual(loss, 2.9815, places=4)

        manifest_model_newer = {
            "manifest": {
                "v5_small_latest": {"step": 12000, "loss": 2.75},
                "latest": {"step": 9750, "loss": 2.9815},
            }
        }
        slot2, step2, loss2 = select_highest_step_slot(manifest_model_newer, model="tpu_v5e_1", target_type="latest")
        self.assertEqual(slot2, "v5_small")
        self.assertEqual(step2, 12000)
        self.assertAlmostEqual(loss2, 2.75, places=4)

        slot3, step3, loss3 = select_highest_step_slot({}, model="tpu_v5e_1", target_type="best")
        self.assertEqual(slot3, "v5_small")
        self.assertEqual(step3, 0)
        self.assertEqual(loss3, float("inf"))

    def test_detect_source_environment_kaggle(self):
        """Detect Kaggle runtime cleanly without misclassifying as Colab."""
        with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, clear=False):
            with patch("src.cloud_sync.os.path.exists", side_effect=lambda p: p == "/kaggle"):
                self.assertEqual(detect_source_environment(), "kaggle")

    @patch("src.cloud_sync.fetch_remote_hardware_config", return_value=None)
    @patch("src.jax.auto_tuner.save_local_hardware_config")
    @patch("src.jax.auto_tuner.load_local_hardware_config", return_value=None)
    @patch("src.jax.auto_tuner.benchmark_config")
    @patch("src.jax.auto_tuner.get_hardware_info")
    def test_auto_tune_caps_max_batch_size_and_scopes_cache(
        self,
        mock_hw,
        mock_bench,
        mock_load_cache,
        mock_save_cache,
        _mock_fetch_remote,
    ):
        """Verify max_batch_size=64 constrains candidates, winner, and cache signature."""
        candidates = generate_vram_maximizing_candidates("tpu_v5e_1", total_gb=16.0, seq_len=256, max_batch_size=64)
        self.assertTrue(len(candidates) >= 2)
        self.assertTrue(all(c["batch_size"] <= 64 for c in candidates))
        self.assertIn({"batch_size": 64, "remat": False}, candidates)
        self.assertIn({"batch_size": 64, "remat": True}, candidates)

        mock_hw.return_value = {
            "device_kind": "TPU_v5_lite",
            "platform": "tpu",
            "is_tpu": True,
            "is_gpu": False,
            "total_memory_bytes": 16 * 1024**3,
            "total_gb": 16.0,
            "hardware_signature": "TPU_v5_lite_16GB",
        }
        seen_batches = []

        def _fake_bench(preset, batch_size, remat, seq_len=256, **kwargs):
            seen_batches.append((batch_size, remat))
            sps = 4.0 if (batch_size == 64 and not remat) else 3.0
            return {
                "preset": preset,
                "batch_size": batch_size,
                "remat": remat,
                "tok_per_sec": sps * batch_size * seq_len,
                "steps_per_sec": sps,
                "step_ms": 1000.0 / sps,
                "compile_time": 1.0,
                "memory_gb": 3.5,
                "memory_utilization_pct": 22.0,
                "memory_type": "HBM",
                "vram_gb": 3.5,
                "vram_utilization_pct": 22.0,
                "status": "PASS",
            }

        mock_bench.side_effect = _fake_bench
        winner = auto_tune_environment(
            preset="tpu_v5e_1",
            seq_len=256,
            use_cached_config=True,
            upload_winner=False,
            verbose=False,
            tune_objective="steps_per_sec",
            max_batch_size=64,
        )
        self.assertLessEqual(winner["batch_size"], 64)
        self.assertEqual(winner["batch_size"], 64)
        self.assertFalse(winner["remat"])
        self.assertTrue(all(b <= 64 for b, _ in seen_batches))
        cache_sig_used = mock_load_cache.call_args[0][1]
        self.assertTrue(cache_sig_used.endswith("_cap64"), f"Expected _cap64 cache suffix, got {cache_sig_used}")
        save_sig_used = mock_save_cache.call_args[0][1]
        self.assertTrue(save_sig_used.endswith("_cap64"), f"Expected _cap64 save suffix, got {save_sig_used}")

    @patch("jax.devices")
    def test_tpu_v6e_and_v5p_hardware_profiling(self, mock_devices):
        """Verifies TPU v6e (Trillium 32GB) and TPU v5p (95GB) profiling and memory mapping."""
        # TPU v6e (Trillium)
        dev_v6e = MagicMock()
        dev_v6e.device_kind = "TPU v6e"
        dev_v6e.platform = "tpu"
        del dev_v6e.memory_stats
        mock_devices.return_value = [dev_v6e]

        hw_v6e = get_hardware_info()
        self.assertEqual(hw_v6e["total_gb"], 32.0)
        self.assertTrue(hw_v6e["is_tpu"])
        self.assertIn("32GB", hw_v6e["hardware_signature"])

        # Trillium alias
        dev_trillium = MagicMock()
        dev_trillium.device_kind = "Trillium"
        dev_trillium.platform = "tpu"
        del dev_trillium.memory_stats
        mock_devices.return_value = [dev_trillium]

        hw_trillium = get_hardware_info()
        self.assertEqual(hw_trillium["total_gb"], 32.0)

        # TPU v5p (95GB)
        dev_v5p = MagicMock()
        dev_v5p.device_kind = "TPU v5p"
        dev_v5p.platform = "tpu"
        del dev_v5p.memory_stats
        mock_devices.return_value = [dev_v5p]

        hw_v5p = get_hardware_info()
        self.assertEqual(hw_v5p["total_gb"], 95.0)
        self.assertTrue(hw_v5p["is_tpu"])
        self.assertIn("95GB", hw_v5p["hardware_signature"])

    def test_tpu_v6e_presets_and_aliases(self):
        """Verifies tpu_v6e_1 and tpu_v6e presets and cloud sync alias normalization."""
        from src.jax.v5_model import PRESETS_V5 as JAX_PRESETS
        from src.spheroid_v5 import PRESETS_V5 as PT_PRESETS, infer_v5_config_from_state_dict

        self.assertIn("tpu_v6e_1", JAX_PRESETS)
        self.assertIn("tpu_v6e", JAX_PRESETS)
        self.assertIn("tpu_v6e_1", PT_PRESETS)
        self.assertIn("tpu_v6e", PT_PRESETS)
        self.assertEqual(JAX_PRESETS["tpu_v6e_1"]["surface_dim"], 640)
        self.assertEqual(JAX_PRESETS["tpu_v6e_1"]["num_shells"], 16)

        # Inferred config normalization
        dummy_state = {"preset": "tpu_v6e_1"}
        inferred = infer_v5_config_from_state_dict(dummy_state)
        self.assertEqual(inferred.surface_dim, 640)
        self.assertEqual(inferred.num_shells, 16)


if __name__ == "__main__":
    unittest.main()
