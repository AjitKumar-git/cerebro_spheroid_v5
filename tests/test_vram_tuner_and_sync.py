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
)
from src.cloud_sync import (
    fetch_remote_hardware_config,
    push_remote_hardware_config,
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
        # 32GB+ tier (e.g. TPU v4, A100-40/80GB)
        c_32 = generate_vram_maximizing_candidates("v5_small", total_gb=32.0, seq_len=256)
        batch_sizes_32 = [c["batch_size"] for c in c_32]
        self.assertIn(256, batch_sizes_32)
        self.assertIn(512, batch_sizes_32)
        # Verify remat=False is also evaluated for higher throughput when memory permits
        remat_false_32 = [c for c in c_32 if not c["remat"]]
        self.assertGreaterEqual(len(remat_false_32), 1)

        # 16GB tier (e.g. TPU v5e-1, T4 16GB)
        c_16 = generate_vram_maximizing_candidates("v5_small", total_gb=16.0, seq_len=256)
        batch_sizes_16 = [c["batch_size"] for c in c_16]
        self.assertIn(128, batch_sizes_16)
        self.assertIn(256, batch_sizes_16)
        self.assertIn(384, batch_sizes_16)

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


if __name__ == "__main__":
    unittest.main()
