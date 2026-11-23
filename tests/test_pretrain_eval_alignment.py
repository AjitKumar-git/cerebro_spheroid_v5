import unittest
import numpy as np
from src.corpus import build_tokenizer
from src.elastic_window import build_pretrain_eval_split

class TestPretrainEvalAlignment(unittest.TestCase):
    def setUp(self):
        self.tokenizer = build_tokenizer("llama")

    def test_pretrain_eval_split_prompt_len_1(self):
        batch_size = 4
        target_len = 512
        eval_batches, domain_labels, domain_weights = build_pretrain_eval_split(
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            prompt_len=1,
            target_len=target_len,
            return_weights=True,
            domains=["broad_web", "code"],
        )

        self.assertEqual(len(eval_batches), 2)
        for idx, (qp, qt, qfb) in enumerate(eval_batches):
            # Prompt must be shape [batch_size, 1] containing BOS token
            self.assertEqual(qp.shape, (batch_size, 1), f"Domain {domain_labels[idx]} prompt shape mismatch")
            bos_id = getattr(self.tokenizer, "bos_token_id", 1) or 1
            np.testing.assert_array_equal(qp, np.full((batch_size, 1), bos_id, dtype=np.int32))

            # Target must be shape [batch_size, target_len]
            self.assertEqual(qt.shape, (batch_size, target_len), f"Domain {domain_labels[idx]} target shape mismatch")

            # Feedback must be shape [batch_size, target_len] with fb[:, 0] == -100 and fb[:, 1:] == qt[:, :-1]
            self.assertEqual(qfb.shape, (batch_size, target_len))
            self.assertTrue(np.all(qfb[:, 0] == -100))
            np.testing.assert_array_equal(qfb[:, 1:], qt[:, :-1])

    def test_pretrain_eval_split_prompt_len_256(self):
        batch_size = 4
        prompt_len = 256
        target_len = 512
        eval_batches, domain_labels, domain_weights = build_pretrain_eval_split(
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            prompt_len=prompt_len,
            target_len=target_len,
            return_weights=True,
            domains=["broad_web", "code"],
        )

        self.assertEqual(len(eval_batches), 2)
        for idx, (qp, qt, qfb) in enumerate(eval_batches):
            self.assertEqual(qp.shape, (batch_size, prompt_len))
            self.assertEqual(qt.shape, (batch_size, target_len))
            self.assertEqual(qfb.shape, (batch_size, target_len))

if __name__ == "__main__":
    unittest.main()
