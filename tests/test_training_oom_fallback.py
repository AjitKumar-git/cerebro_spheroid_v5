"""
End-to-End Test for Training Loop OOM Fallback to Smaller Batch Size.
=====================================================================
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from autotune.fallback import TrainingOOMHandler, is_oom_error


class TestTrainingOOMFallbackIntegration(unittest.TestCase):

    def test_training_loop_dynamic_fallback(self):
        """Simulates an active training loop where batch_size=64 OOMs and falls back to 32."""
        initial_bs = 64
        effective_bs = 64
        initial_ga = 1

        handler = TrainingOOMHandler(
            initial_batch_size=initial_bs,
            effective_batch_size=effective_bs,
            initial_grad_accum=initial_ga,
            min_batch_size=1,
            verbose=False,
        )

        mock_streamer = MagicMock()
        mock_streamer.batch_size = 64

        # Dummy dataset
        p_data = np.zeros((64, 128), dtype=np.int32)
        t_data = np.zeros((64, 128), dtype=np.int32)

        def _next_batch():
            return p_data, t_data, None

        # Step function that OOMs when batch_size >= 64, but succeeds when batch_size <= 32
        executed_batches = []
        def _mock_train_step(p, opt_s, q_batch, r_batch, *args, **kwargs):
            if q_batch.shape[0] >= 64:
                raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate on TPU")
            executed_batches.append(q_batch.shape[0])
            return p, opt_s, 2.50

        step = 10
        cur_bs = handler.batch_size
        cur_ga = handler.grad_accum
        step_executed = False

        while not step_executed:
            prompt_b, target_b, feedback_b = None, None, None
            try:
                for _ in range(cur_ga):
                    prompt_b, target_b, feedback_b = handler.slicer.get_micro_batch(cur_bs, _next_batch)
                    _mock_train_step(None, None, prompt_b, target_b)
                step_executed = True
            except Exception as e_step:
                if not is_oom_error(e_step) or cur_bs <= 1:
                    raise
                failed_tup = (prompt_b, target_b, feedback_b) if prompt_b is not None else None
                cur_bs, cur_ga = handler.handle_oom(e_step, step=step, failed_batch=failed_tup)
                handler.update_streamer(mock_streamer)

        # Assertions
        self.assertTrue(step_executed)
        self.assertEqual(cur_bs, 32)
        self.assertEqual(cur_ga, 2)
        self.assertEqual(cur_bs * cur_ga, effective_bs)
        self.assertEqual(mock_streamer.batch_size, 32)
        # Executed two micro-steps of size 32
        self.assertEqual(executed_batches, [32, 32])
        self.assertEqual(sum(executed_batches), 64)  # Zero elements dropped!

    def test_training_loop_remat_then_batch_cascade(self):
        """Simulates cascade where remat=False first toggles to remat=True, then halves batch if still OOM."""
        handler = TrainingOOMHandler(
            initial_batch_size=64,
            effective_batch_size=64,
            initial_grad_accum=1,
            remat=False,
            auto_engage_remat=True,
            min_batch_size=1,
            verbose=False,
        )

        p_data = np.zeros((64, 128), dtype=np.int32)
        t_data = np.zeros((64, 128), dtype=np.int32)

        def _next_batch():
            return p_data, t_data, None

        # OOM condition:
        # If remat is False, fails for any batch >= 32
        # If remat is True, fails for batch >= 64, but succeeds for batch <= 32
        cur_remat = handler.remat
        executed_batches = []

        def _mock_step(p_batch, r_batch, remat_flag):
            if not remat_flag and p_batch.shape[0] >= 32:
                raise RuntimeError("RESOURCE_EXHAUSTED: HbmOom without remat")
            if remat_flag and p_batch.shape[0] >= 64:
                raise RuntimeError("RESOURCE_EXHAUSTED: HbmOom with remat at 64")
            executed_batches.append((p_batch.shape[0], remat_flag))
            return 2.45

        step = 10
        cur_bs = handler.batch_size
        cur_ga = handler.grad_accum
        step_executed = False

        while not step_executed:
            prompt_b, target_b, feedback_b = None, None, None
            try:
                for _ in range(cur_ga):
                    prompt_b, target_b, feedback_b = handler.slicer.get_micro_batch(cur_bs, _next_batch)
                    _mock_step(prompt_b, target_b, cur_remat)
                step_executed = True
            except Exception as e_step:
                if not is_oom_error(e_step) or (cur_bs <= 1 and cur_remat):
                    raise
                failed_tup = (prompt_b, target_b, feedback_b) if prompt_b is not None else None
                cur_bs, cur_ga = handler.handle_oom(e_step, step=step, failed_batch=failed_tup, current_remat=cur_remat)
                cur_remat = handler.remat

        self.assertTrue(step_executed)
        self.assertTrue(cur_remat)
        self.assertEqual(cur_bs, 32)
        self.assertEqual(cur_ga, 2)
        self.assertEqual(executed_batches, [(32, True), (32, True)])


if __name__ == "__main__":
    unittest.main()
