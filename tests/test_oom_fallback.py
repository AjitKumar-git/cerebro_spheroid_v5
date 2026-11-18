"""
Unit Tests for Training OOM Detection & Fallback to Smaller Batch Size.
======================================================================
"""

import unittest
import numpy as np

from autotune.fallback import (
    is_oom_error,
    compute_fallback_batch_size,
    adjust_gradient_accumulation,
    MicroBatchSlicer,
    TrainingOOMHandler,
)


class TestOOMFallback(unittest.TestCase):

    def test_is_oom_error_detection(self):
        """Verifies accurate detection of TPU HBM, GPU VRAM, and RAM OOM exceptions."""
        oom_exceptions = [
            RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1610612736 bytes on TPU"),
            Exception("jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: HbmOom"),
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
            MemoryError("Out of memory"),
            Exception("failed to allocate 4194304 bytes on device"),
            RuntimeError("allocation_failed in buffer allocator"),
            Exception("std::bad_alloc"),
        ]
        for exc in oom_exceptions:
            self.assertTrue(is_oom_error(exc), f"Failed to identify OOM for: {exc}")

        non_oom_exceptions = [
            ValueError("Invalid learning rate"),
            KeyError("missing_key"),
            ZeroDivisionError("division by zero"),
            KeyboardInterrupt("interrupted"),
            RuntimeError("Connection reset by peer"),
        ]
        for exc in non_oom_exceptions:
            self.assertFalse(is_oom_error(exc), f"False positive OOM for: {exc}")

    def test_compute_fallback_batch_size(self):
        """Tests decreasing batch size calculations down to minimum allowable floor."""
        self.assertEqual(compute_fallback_batch_size(128), 64)
        self.assertEqual(compute_fallback_batch_size(64), 32)
        self.assertEqual(compute_fallback_batch_size(32), 16)
        self.assertEqual(compute_fallback_batch_size(16), 8)
        self.assertEqual(compute_fallback_batch_size(8), 4)
        self.assertEqual(compute_fallback_batch_size(4), 2)
        self.assertEqual(compute_fallback_batch_size(2), 1)

        # Non-power-of-two (e.g. 126 sequences)
        self.assertEqual(compute_fallback_batch_size(126), 62)

        # At floor
        with self.assertRaises(RuntimeError):
            compute_fallback_batch_size(1, min_batch_size=1)

    def test_adjust_gradient_accumulation(self):
        """Verifies effective batch size conservation via gradient accumulation scaling."""
        eff_bs = 256
        # 128 micro-batch -> grad_accum = 2
        self.assertEqual(adjust_gradient_accumulation(eff_bs, 128), 2)
        # OOM fallback to 64 -> grad_accum = 4 (64 * 4 = 256)
        self.assertEqual(adjust_gradient_accumulation(eff_bs, 64), 4)
        # OOM fallback to 32 -> grad_accum = 8 (32 * 8 = 256)
        self.assertEqual(adjust_gradient_accumulation(eff_bs, 32), 8)
        # OOM fallback to 16 -> grad_accum = 16 (16 * 16 = 256)
        self.assertEqual(adjust_gradient_accumulation(eff_bs, 16), 16)

    def test_micro_batch_slicer_exact_chunks(self):
        """Verifies that large batches are sliced into exact micro-chunks without losing elements."""
        p_data = np.arange(64).reshape(64, 1)
        t_data = np.arange(64).reshape(64, 1) * 10
        fb_data = np.arange(64).reshape(64, 1) * 100

        called = [0]
        def _fake_next_batch():
            called[0] += 1
            return p_data, t_data, fb_data

        slicer = MicroBatchSlicer(_fake_next_batch)

        # Slice 64 items into two 32-item micro-batches
        p1, t1, fb1 = slicer.get_micro_batch(32)
        self.assertEqual(p1.shape[0], 32)
        self.assertEqual(p1[0, 0], 0)
        self.assertEqual(p1[-1, 0], 31)
        self.assertEqual(called[0], 1)
        self.assertTrue(slicer.has_remainder())

        # Second micro-batch comes from stashed remainder without fetching new batch
        p2, t2, fb2 = slicer.get_micro_batch(32)
        self.assertEqual(p2.shape[0], 32)
        self.assertEqual(p2[0, 0], 32)
        self.assertEqual(p2[-1, 0], 63)
        self.assertEqual(called[0], 1)  # No new fetch!
        self.assertFalse(slicer.has_remainder())

    def test_training_oom_handler_workflow(self):
        """Tests complete OOM handler lifecycle during simulated training failure."""
        handler = TrainingOOMHandler(
            initial_batch_size=64,
            effective_batch_size=128,
            initial_grad_accum=2,
            min_batch_size=1,
            verbose=False,
        )

        p_batch = np.zeros((64, 128))
        t_batch = np.zeros((64, 128))

        # Simulate OOM at step 42
        oom_exc = RuntimeError("RESOURCE_EXHAUSTED: HbmOom on TPU v5e")
        new_bs, new_ga = handler.handle_oom(
            exc=oom_exc,
            step=42,
            failed_batch=(p_batch, t_batch),
        )

        self.assertEqual(new_bs, 32)
        self.assertEqual(new_ga, 4)
        self.assertEqual(new_bs * new_ga, 128)  # Effective batch size conserved!
        self.assertEqual(handler.fallback_count, 1)
        self.assertTrue(handler.slicer.has_remainder())

        # First slice of failed batch
        p_micro1, t_micro1, _ = handler.slicer.get_micro_batch(new_bs)
        self.assertEqual(p_micro1.shape[0], 32)

        # Second slice of failed batch
        p_micro2, t_micro2, _ = handler.slicer.get_micro_batch(new_bs)
        self.assertEqual(p_micro2.shape[0], 32)
        self.assertFalse(handler.slicer.has_remainder())


if __name__ == "__main__":
    unittest.main()
