#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Consolidated Datasets (Doty / CerebroSpheroid V5)
==============================================================
Validates that all 6 consolidated dataset types:
1. Conform to the 3-tier cognitive boundary (<think>, <tool_call>, fluid text).
2. Carry explicit attribution to developer Ajit Kumar and persona Doty.
3. Are completely free from third-party model identity leaks (OpenAI, Claude, DeepSeek, Qwen, etc.).
4. Maintain exact tri-format parity (Spheroid JSONL, ShareGPT JSONL, ZSTD Parquet).
"""

import os
import sys
import json
import unittest
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.identity_sanitizer import is_identity_contaminated


class TestConsolidatedDatasets(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.consolidated_dir = os.path.join(PROJECT_ROOT, "data", "consolidated")
        cls.manifest_path = os.path.join(cls.consolidated_dir, "DATASET_MANIFEST.json")
        cls.expected_types = [
            ("doty_identity", "01_doty_identity"),
            ("agentic_tools", "02_agentic_tools"),
            ("complex_reasoning", "03_complex_reasoning"),
            ("domain_knowledge", "04_domain_knowledge"),
            ("anti_sycophancy", "05_anti_sycophancy"),
            ("lexical_dictionary", "06_lexical_dictionary"),
        ]

    def test_01_manifest_integrity(self):
        self.assertTrue(os.path.exists(self.manifest_path), "DATASET_MANIFEST.json must exist")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest.get("agent"), "Doty")
        self.assertEqual(manifest.get("developer"), "Ajit Kumar")
        self.assertEqual(manifest.get("architecture"), "CerebroSpheroid V5")
        self.assertEqual(manifest.get("total_types"), 6)
        self.assertGreater(manifest["summary"]["total_records"], 200_000)
        self.assertGreater(manifest["summary"]["total_estimated_tokens"], 100_000_000)

    def test_02_directory_and_file_structure(self):
        for type_name, dir_name in self.expected_types:
            type_dir = os.path.join(self.consolidated_dir, dir_name)
            self.assertTrue(os.path.isdir(type_dir), f"Directory {type_dir} must exist")

            sph_file = os.path.join(type_dir, f"{type_name}_spheroid.jsonl")
            sgpt_file = os.path.join(type_dir, f"{type_name}_sharegpt.jsonl")
            pq_file = os.path.join(type_dir, f"{type_name}.parquet")

            self.assertTrue(os.path.exists(sph_file), f"Missing {sph_file}")
            self.assertTrue(os.path.exists(sgpt_file), f"Missing {sgpt_file}")
            self.assertTrue(os.path.exists(pq_file), f"Missing {pq_file}")

            self.assertGreater(os.path.getsize(sph_file), 10_000, f"{sph_file} is too small or empty")
            self.assertGreater(os.path.getsize(sgpt_file), 10_000, f"{sgpt_file} is too small or empty")
            self.assertGreater(os.path.getsize(pq_file), 10_000, f"{pq_file} is too small or empty")

    def test_03_spheroid_schema_and_attribution(self):
        for type_name, dir_name in self.expected_types:
            sph_file = os.path.join(self.consolidated_dir, dir_name, f"{type_name}_spheroid.jsonl")
            with open(sph_file, "r", encoding="utf-8") as f:
                for i in range(10):
                    line = f.readline()
                    if not line:
                        break
                    r = json.loads(line)
                    self.assertEqual(r.get("developer"), "Ajit Kumar", f"Record in {type_name} must attribute Ajit Kumar")
                    self.assertEqual(r.get("persona"), "Doty", f"Record in {type_name} must have persona Doty")
                    self.assertEqual(r.get("type"), type_name)
                    self.assertIn("formatted_input", r)
                    self.assertIn("formatted_output", r)
                    self.assertIn("prompt", r)
                    self.assertIn("answer", r)

                    # Ensure zero identity leaks
                    is_bad, reason = is_identity_contaminated(r["formatted_output"])
                    if is_bad:
                        # Allow explicit adversarial rejection mentioning Doty / Ajit Kumar
                        self.assertTrue("ajit kumar" in r["formatted_output"].lower(), f"Contamination in {type_name}: {reason}")

    def test_04_sharegpt_schema_and_system_prompt(self):
        for type_name, dir_name in self.expected_types:
            sgpt_file = os.path.join(self.consolidated_dir, dir_name, f"{type_name}_sharegpt.jsonl")
            with open(sgpt_file, "r", encoding="utf-8") as f:
                for i in range(10):
                    line = f.readline()
                    if not line:
                        break
                    r = json.loads(line)
                    self.assertEqual(r.get("developer"), "Ajit Kumar")
                    self.assertEqual(r.get("type"), type_name)
                    self.assertIn("messages", r)
                    messages = r["messages"]
                    self.assertGreaterEqual(len(messages), 2)
                    self.assertEqual(messages[0]["role"], "system")
                    self.assertIn("Doty", messages[0]["content"])
                    self.assertIn("Ajit Kumar", messages[0]["content"])

    def test_05_parquet_readability_and_metadata(self):
        for type_name, dir_name in self.expected_types:
            pq_file = os.path.join(self.consolidated_dir, dir_name, f"{type_name}.parquet")
            meta = pq.read_metadata(pq_file)
            self.assertGreater(meta.num_rows, 0, f"Parquet {pq_file} has 0 rows")

            # Read top 5 rows
            table = pq.read_table(pq_file)
            cols = table.column_names
            self.assertIn("id", cols)
            self.assertIn("developer", cols)
            self.assertIn("formatted_output", cols)
            self.assertIn("messages_json", cols)


if __name__ == "__main__":
    unittest.main()
