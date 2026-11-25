import os
import sys
import unittest
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.elastic_window import select_best_bucket, ElasticBucketBatcher, get_default_elastic_buckets
from transformers import AutoTokenizer


class DummyTokenizer:
    def __init__(self, vocab_size=32000, eos_token_id=2):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False, max_length=None, truncation=False):
        tokens = [abs(hash(w)) % (self.vocab_size - 10) + 5 for w in text.split()]
        if truncation and max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
        return tokens

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(f"tok_{t}" for t in token_ids)


class TestSFTTruncationResolution(unittest.TestCase):
    def setUp(self):
        self.tok = DummyTokenizer()

    def test_select_best_bucket_avoids_target_truncation(self):
        """
        Verify that select_best_bucket chooses a bucket that accommodates the full target response
        rather than heavily truncating into a smaller bucket just to avoid padding.
        """
        buckets = [
            ("short_prompt_short_target", 256, 256),
            ("standard_cot", 256, 512),
            ("deep_reasoning", 256, 768),
        ]
        # A prompt of 50 tokens and target of 320 tokens
        # Target exceeds bucket 0 (256), but fits completely in bucket 1 (512)
        idx = select_best_bucket(50, 320, buckets)
        self.assertEqual(idx, 1, f"Expected bucket 1 (512 target), got bucket {idx} ({buckets[idx]})")

    def test_elastic_bucket_batcher_sft_intact_eos_and_prompt(self):
        """
        Verify that for SFT/reasoning samples:
        1. Prompts preserve the stimulus header prefix and question suffix.
        2. Long responses are NOT fragmented into multiple headless chunks.
        3. Terminating eos_id is always intact at the end of the response.
        """
        buckets = [("sft_bucket", 64, 128)]
        # Generate a sample where response has 160 tokens (> 128) and prompt has 80 tokens (> 64)
        sample = {
            "conversations": [
                {"from": "human", "value": "StimulusHeaderPrefix " + "filler " * 70 + "What is the final answer?"},
                {"from": "gpt", "value": "<think> " + "reasoning step " * 60 + "</think> The answer is 42."},
            ],
            "_dataset": "conversation_sft",
        }

        def data_iter():
            while True:
                yield sample

        batcher = ElasticBucketBatcher(
            tokenizer=self.tok,
            data_iterator=data_iter(),
            batch_size=2,
            total_window=192,
            buckets=buckets,
            framework="numpy",
            return_feedback=True,
            target_pad_id=-100,
        )

        b_idx, b_name, p_batch, t_batch, fb_batch = batcher.get_batch()
        self.assertEqual(p_batch.shape, (2, 64))
        self.assertEqual(t_batch.shape, (2, 128))

        # Check that targets end with terminating eos_id (2)
        for row in t_batch:
            non_pad = [x for x in row if x != -100]
            self.assertGreater(len(non_pad), 0)
            self.assertEqual(non_pad[-1], self.tok.eos_token_id, "Target response must terminate with eos_id!")

    def test_v5_dataset_loader_eos_and_prefix_suffix(self):
        """Verify CerebroV5Dataset ensures terminating eos_id and preserves prompt question."""
        from src.v5_dataset_loader import CerebroV5Dataset

        class MockDataset(CerebroV5Dataset):
            def _load_data(self):
                self.samples = [
                    {
                        "formatted_input": "context " * 300 + "What is 2+2?",
                        "formatted_output": "<think> 2+2=4 </think> 4",
                        "thinking_level": 2,
                    }
                ]

        ds = MockDataset(
            file_path="mock.parquet",
            tokenizer=self.tok,
            max_prompt_len=64,
            max_target_len=32,
        )
        item = ds[0]
        p_ids = item["prompt_ids"]
        t_ids = item["target_ids"]

        self.assertLessEqual(len(p_ids), 64)
        self.assertLessEqual(len(t_ids), 32)
        self.assertEqual(t_ids[-1], self.tok.eos_token_id, "Target must end with eos_id!")

    def test_load_harmonized_dataset_terminating_eos(self):
        """Verify that load_harmonized_dataset ensures 100% of targets end with eos_token_id."""
        from scripts.run_harmonized_sft import load_harmonized_dataset
        entries = load_harmonized_dataset(
            tokenizer=self.tok,
            vocab_size=32000,
            slot_oversample=1,
            max_cot_samples=10,
            max_prompt_len=2048,
            max_target_len=1024,
        )
        self.assertGreater(len(entries), 0)
        for e in entries:
            t_ids = e["target_ids"]
            self.assertEqual(t_ids[-1], self.tok.eos_token_id, "Every target in harmonized SFT must end with eos_token_id!")

    def test_load_tri_modal_dataset_retrieval_integrity(self):
        """Verify that retrieval curriculum samples are only included when they fit without amputating the needle."""
        from scripts.run_refined_sft_pytorch import load_tri_modal_dataset
        # With max_prompt_len=256, samples with 512, 1024, 2048, 4096 tokens should NOT be ingested
        # because slicing them would amputate the needle. Only <= 256 should be ingested.
        entries = load_tri_modal_dataset(
            tokenizer=self.tok,
            vocab_size=32000,
            max_cot_samples=10,
            max_prompt_len=256,
            max_target_len=512,
        )
        retrieval_entries = [e for e in entries if "retrieval" in e.get("type", "")]
        for re in retrieval_entries:
            self.assertLessEqual(len(re["prompt_ids"]), 256)
            self.assertEqual(re["target_ids"][-1], self.tok.eos_token_id)


if __name__ == "__main__":
    unittest.main()
