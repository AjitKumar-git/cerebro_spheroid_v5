"""
Tests for CerebroSpheroid V5 Live Token Tracker, Dataset Attribution, and Batch Sizing.
"""

import collections
import os
import unittest
from unittest.mock import MagicMock, patch

from src.elastic_window import ElasticBucketBatcher, PrefetchedBatcher
from src.cloud_sync import push_remote_token_tracker, fetch_remote_token_tracker


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    is_fast = False

    def encode(self, text, max_length=512):
        return [1] * min(len(text.split()), max_length)


class TestTokenTrackerAndBatcher(unittest.TestCase):
    def test_dataset_token_accumulation_in_elastic_batcher(self):
        datasets = ["fineweb_edu", "cosmopedia", "finemath", "python_code", "smoltalk"]

        def dummy_stream():
            i = 0
            while True:
                ds = datasets[i % len(datasets)]
                i += 1
                yield {
                    "stimulus": f"Question {i} test",
                    "articulation": f"Answer {i} reasoning step",
                    "_dataset": ds,
                }

        tok = DummyTokenizer()
        batch_size = 8
        window = 512
        buckets = [("static", 256, 256)]

        batcher = ElasticBucketBatcher(
            tokenizer=tok,
            data_iterator=dummy_stream(),
            batch_size=batch_size,
            total_window=window,
            buckets=buckets,
            framework="numpy",
            return_feedback=True,
        )

        # Pull 5 batches
        for _ in range(5):
            batcher.next_batch()

        stats = batcher.get_tokens_by_dataset()
        self.assertIsInstance(stats, dict)
        total_ingested = sum(stats.values())
        expected_tokens = 5 * batch_size * window  # 5 * 8 * 512 = 20,480
        self.assertEqual(total_ingested, expected_tokens)

        for ds in datasets:
            self.assertIn(ds, stats)
            self.assertGreater(stats[ds], 0)

        # Test set_tokens_by_dataset restoration
        resumed_stats = {"fineweb_edu": 100_000_000, "cosmopedia": 50_000_000}
        batcher.set_tokens_by_dataset(resumed_stats)
        self.assertEqual(batcher.get_tokens_by_dataset(), resumed_stats)

    def test_prefetched_batcher_forwarding(self):
        tok = DummyTokenizer()
        data = [{"stimulus": "q", "articulation": "a", "_dataset": "finemath"} for _ in range(20)]
        batcher = ElasticBucketBatcher(
            tokenizer=tok,
            data_iterator=iter(data),
            batch_size=2,
            total_window=512,
            buckets=[("static", 256, 256)],
            framework="numpy",
        )
        prefetched = PrefetchedBatcher(batcher, queue_size=2)
        try:
            # Check get and set methods exist and forward cleanly
            batcher.set_tokens_by_dataset({"finemath": 5000})
            self.assertEqual(prefetched.get_tokens_by_dataset(), {"finemath": 5000})
        finally:
            prefetched.shutdown()

    def test_effective_batch_size_scaling_logic(self):
        # 1. Base default: 256
        eff_target = 256
        batch_size = 32
        grad_accum = max(1, eff_target // max(1, batch_size))
        self.assertEqual(grad_accum, 8)
        self.assertEqual(batch_size * grad_accum, 256)

        # 2. Alternative: 128
        eff_target = 128
        batch_size = 32
        grad_accum = max(1, eff_target // max(1, batch_size))
        self.assertEqual(grad_accum, 4)
        self.assertEqual(batch_size * grad_accum, 128)

        # 3. Base support for 126
        eff_target = 126
        batch_size = 128
        if eff_target == 126:
            if batch_size > 126:
                batch_size = 126
                grad_accum = 1
        self.assertEqual(batch_size * grad_accum, 126)

    @patch("urllib.request.urlopen")
    def test_push_remote_token_tracker_payload(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"success": true}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        success = push_remote_token_tracker(
            model="tpu_v5e_1",
            cumulative_tokens=15_000_000_000,
            target_tokens=200_000_000_000,
            tokens_by_dataset={"fineweb_edu": 8_000_000_000, "cosmopedia": 2_000_000_000},
            effective_batch_size=256,
            step=114440,
            loss=1.845,
            tok_per_sec=204500.0,
            url="https://bioflo.in/api/cerebro/sync.php",
            token="test_token",
        )
        self.assertTrue(success)
        self.assertTrue(mock_urlopen.called)

    def test_convergence_early_stopping_logic(self):
        target_eval_loss = 1.60
        threshold = 3

        # Simulate fluctuating evaluation loss sequence
        eval_losses = [2.10, 1.85, 1.58, 1.62, 1.59, 1.57, 1.55]
        consistent_count = 0
        converged = False
        converged_at_idx = None

        for idx, el in enumerate(eval_losses):
            if el <= target_eval_loss:
                consistent_count += 1
                if consistent_count >= threshold:
                    converged = True
                    converged_at_idx = idx
                    break
            else:
                consistent_count = 0

        self.assertTrue(converged)
        # Should converge on index 6 (1.59, 1.57, 1.55) after reset at index 3 (1.62)
        self.assertEqual(converged_at_idx, 6)

    def test_step900k_checkpoint_discovery(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ckpt_path = os.path.join(root_dir, "checkpoints", "v5_small_latest_step900k.pt")
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint not found at: {ckpt_path}")


if __name__ == "__main__":
    unittest.main()
