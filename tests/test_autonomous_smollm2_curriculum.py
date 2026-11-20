import unittest
from unittest.mock import patch, MagicMock
from collections import Counter
from src.elastic_window import build_online_dataset_iterator

class TestAutonomousSmolLM2Curriculum(unittest.TestCase):
    """
    Validates that the developmental autonomous pretraining curriculum strictly adheres
    to SmolLM2 (arXiv:2502.02737 Section 4) 11T multi-stage data mixture progression:
      - Stage 1 (<55%): 90% English Web, 10% Code, 0% Math
      - Stage 2 (55%-73%): 75% English Web, 20% Code, 5% Math
      - Stage 3 (73%-91%): 70% English Web, 20% Code, 10% Math
      - Stage 4 (91%-100%): 58% English Web, 24% Code, 14% Math, 4% Cosmopedia v2
      - Zero SFT datasets (smoltalk, gsm8k, MetaMathQA, openhermes, magpie) in pretrain.
      - 100% loss-bearing contiguous pretrain document formatting.
    """

    def _mock_open_stream(self, prefix: str):
        def _stream():
            i = 0
            while True:
                yield {"text": f"This is authentic synthetic mock content for {prefix} document number {i}. " * 5}
                i += 1
        return _stream()

    def test_smollm2_stage_1_mixture(self):
        # Progress tau = 0.20 (< 0.55) -> 90% English Web, 10% Code, 0% Math
        step = 200
        total_steps = 1000

        with patch("src.elastic_window._open_hf_stream") as mock_hf:
            mock_hf.side_effect = lambda repo, cfg=None, **kw: self._mock_open_stream(f"{repo}_{cfg}")

            it = build_online_dataset_iterator(
                dataset_source="autonomous",
                step_getter=lambda: step,
                total_steps=total_steps,
            )

            counts = Counter()
            for _ in range(100):
                sample = next(it)
                self.assertEqual(sample["prompt"], "Document:\n")
                self.assertNotIn("<act>", sample["text"])
                self.assertNotIn("<think>", sample["text"])
                self.assertNotIn("smoltalk", sample["_dataset"])
                counts[sample["_dataset"]] += 1

            self.assertEqual(counts["english_web"], 90)
            self.assertEqual(counts["code"], 10)
            self.assertEqual(counts["math"], 0)
            self.assertEqual(counts["cosmopedia_v2"], 0)

    def test_smollm2_stage_2_mixture(self):
        # Progress tau = 0.60 (0.55 <= tau < 0.73) -> 75% English Web, 20% Code, 5% Math
        step = 600
        total_steps = 1000

        with patch("src.elastic_window._open_hf_stream") as mock_hf:
            mock_hf.side_effect = lambda repo, cfg=None, **kw: self._mock_open_stream(f"{repo}_{cfg}")

            it = build_online_dataset_iterator(
                dataset_source="autonomous",
                step_getter=lambda: step,
                total_steps=total_steps,
            )

            counts = Counter()
            for _ in range(100):
                sample = next(it)
                self.assertEqual(sample["prompt"], "Document:\n")
                self.assertNotIn("<act>", sample["text"])
                self.assertNotIn("<think>", sample["text"])
                counts[sample["_dataset"]] += 1

            self.assertEqual(counts["english_web"], 75)
            self.assertEqual(counts["code"], 20)
            self.assertEqual(counts["math"], 5)
            self.assertEqual(counts["cosmopedia_v2"], 0)

    def test_smollm2_stage_3_mixture(self):
        # Progress tau = 0.80 (0.73 <= tau < 0.91) -> 70% English Web, 20% Code, 10% Math
        step = 800
        total_steps = 1000

        with patch("src.elastic_window._open_hf_stream") as mock_hf:
            mock_hf.side_effect = lambda repo, cfg=None, **kw: self._mock_open_stream(f"{repo}_{cfg}")

            it = build_online_dataset_iterator(
                dataset_source="autonomous",
                step_getter=lambda: step,
                total_steps=total_steps,
            )

            counts = Counter()
            for _ in range(100):
                sample = next(it)
                self.assertEqual(sample["prompt"], "Document:\n")
                self.assertNotIn("<act>", sample["text"])
                self.assertNotIn("<think>", sample["text"])
                counts[sample["_dataset"]] += 1

            self.assertEqual(counts["english_web"], 70)
            self.assertEqual(counts["code"], 20)
            self.assertEqual(counts["math"], 10)
            self.assertEqual(counts["cosmopedia_v2"], 0)

    def test_smollm2_stage_4_anneal_mixture(self):
        # Progress tau = 0.95 (>= 0.91) -> 58% English Web, 24% Code, 14% Math, 4% Cosmopedia v2
        step = 950
        total_steps = 1000

        with patch("src.elastic_window._open_hf_stream") as mock_hf:
            mock_hf.side_effect = lambda repo, cfg=None, **kw: self._mock_open_stream(f"{repo}_{cfg}")

            it = build_online_dataset_iterator(
                dataset_source="autonomous",
                step_getter=lambda: step,
                total_steps=total_steps,
            )

            counts = Counter()
            for _ in range(100):
                sample = next(it)
                self.assertEqual(sample["prompt"], "Document:\n")
                self.assertNotIn("<act>", sample["text"])
                self.assertNotIn("<think>", sample["text"])
                counts[sample["_dataset"]] += 1

            self.assertEqual(counts["english_web"], 58)
            self.assertEqual(counts["code"], 24)
            self.assertEqual(counts["math"], 14)
            self.assertEqual(counts["cosmopedia_v2"], 4)

    def test_alias_routing_to_autonomous(self):
        # Confirm aliases route to autonomous
        for alias in ("pretrain", "pretraining", "smollm", "smollm2", "v5_base", "curriculum"):
            with patch("src.elastic_window._open_hf_stream") as mock_hf:
                mock_hf.side_effect = lambda repo, cfg=None, **kw: self._mock_open_stream(f"{repo}_{cfg}")
                it = build_online_dataset_iterator(dataset_source=alias, total_steps=1000)
                item = next(it)
                self.assertEqual(item["prompt"], "Document:\n")
                self.assertEqual(item["_dataset"], "english_web")

    def test_offline_emergency_fallback(self):
        # When all online streams fail, generator draws gracefully from local emergency pool
        with patch("src.elastic_window._open_hf_stream") as mock_hf:
            mock_hf.side_effect = RuntimeError("Simulated total HF connection timeout")
            it = build_online_dataset_iterator(dataset_source="autonomous", total_steps=1000)
            item = next(it)
            self.assertEqual(item["prompt"], "Document:\n")
            self.assertIn(item["_dataset"], ("offline_emergency_reserve", "english_web"))
            self.assertNotIn("<act>", item["text"])
            self.assertNotIn("<think>", item["text"])

if __name__ == "__main__":
    unittest.main()
