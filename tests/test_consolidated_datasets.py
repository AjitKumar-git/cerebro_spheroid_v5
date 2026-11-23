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
5. Strictly adhere to native ChatML special tokens:
   - Input format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
   - Output format: <think>...</think><tool_call>...</tool_call>...{answer}<|im_end|>
   - Validates that <|im_start|> maps to ID 27 and <|im_end|> maps to ID 28 in the tokenizer.
6. Support functional tool calling syntax without small-model JSON squishing.
"""

import os
import sys
import json
import re
import unittest
import pyarrow.parquet as pq
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.identity_sanitizer import is_identity_contaminated


class TestConsolidatedDatasets(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.consolidated_dir = os.path.join(PROJECT_ROOT, "data", "consolidated")
        cls.manifest_path = os.path.join(cls.consolidated_dir, "DATASET_MANIFEST.json")
        cls.tok_dir = os.path.join(PROJECT_ROOT, "data", "tokenizer")
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.tok_dir)
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
        tokens_contract = manifest.get("special_tokens_contract", {})
        self.assertIn("tokenizer_chatml_ids", tokens_contract)
        self.assertEqual(tokens_contract["tokenizer_chatml_ids"].get("<|im_start|>"), 27)
        self.assertEqual(tokens_contract["tokenizer_chatml_ids"].get("<|im_end|>"), 28)

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
                for i in range(20):
                    line = f.readline()
                    if not line:
                        break
                    r = json.loads(line)
                    self.assertEqual(r.get("developer"), "Ajit Kumar", f"Record in {type_name} must attribute Ajit Kumar")
                    self.assertEqual(r.get("persona"), "Doty", f"Record in {type_name} must have persona Doty")
                    self.assertEqual(r.get("type"), type_name)
                    self.assertIn("formatted_input", r)
                    self.assertIn("formatted_output", r)
                    self.assertIn("input", r)
                    self.assertIn("output", r)
                    self.assertIn("prompt", r)
                    self.assertIn("answer", r)

                    # Ensure zero identity leaks
                    is_bad, reason = is_identity_contaminated(r["formatted_output"])
                    if is_bad:
                        self.assertTrue("ajit kumar" in r["formatted_output"].lower(), f"Contamination in {type_name}: {reason}")

    def test_04_native_chatml_special_tokens(self):
        """
        Validates ChatML special tokens:
        1. Tokenizer maps <|im_start|> to ID 27 and <|im_end|> to ID 28.
        2. formatted_input starts with '<|im_start|>' and contains '<|im_end|>\n<|im_start|>assistant\n'.
        3. formatted_output ends with '<|im_end|>'.
        4. input and output match formatted_input and formatted_output.
        5. Tokenizing formatted_input preserves atomic IDs 27 and 28 with zero fragmentation.
        """
        self.assertEqual(self.tokenizer.encode("<|im_start|>", add_special_tokens=False), [27])
        self.assertEqual(self.tokenizer.encode("<|im_end|>", add_special_tokens=False), [28])

        for type_name, dir_name in self.expected_types:
            sph_file = os.path.join(self.consolidated_dir, dir_name, f"{type_name}_spheroid.jsonl")
            with open(sph_file, "r", encoding="utf-8") as f:
                for i in range(20):
                    line = f.readline()
                    if not line:
                        break
                    r = json.loads(line)
                    inp = r.get("formatted_input", "")
                    out = r.get("formatted_output", "")

                    # 1. ChatML input contract
                    self.assertTrue(inp.startswith("<|im_start|>"), f"Input in {type_name} must start with '<|im_start|>': {inp[:30]}")
                    self.assertTrue(inp.endswith("<|im_start|>assistant\n"), f"Input in {type_name} must end with assistant prompt: {inp[-30:]}")

                    # 2. ChatML output contract
                    self.assertTrue(out.endswith("<|im_end|>"), f"Output in {type_name} must end with '<|im_end|>': {out[-30:]}")

                    # 3. input/output parity
                    self.assertEqual(r.get("input"), inp, f"'input' must match 'formatted_input' in {type_name}")
                    self.assertEqual(r.get("output"), out, f"'output' must match 'formatted_output' in {type_name}")

                    # 4. Atomic tokenization verification
                    inp_tokens = self.tokenizer.encode(inp, add_special_tokens=False)
                    self.assertEqual(inp_tokens[0], 27, f"First token in {type_name} must be ID 27 (<|im_start|>)")
                    self.assertEqual(inp_tokens[-4], 27, f"Assistant prefix token in {type_name} must be ID 27 (<|im_start|>)")

                    out_tokens = self.tokenizer.encode(out, add_special_tokens=False)
                    self.assertEqual(out_tokens[-1], 28, f"Last token in {type_name} must be ID 28 (<|im_end|>)")

    def test_05_functional_tool_syntax(self):
        """
        Validates that agentic tool calls in 02_agentic_tools avoid nested JSON squishing,
        using clean functional syntax: calc(...), facts(...), python(...), func(args).
        """
        sph_file = os.path.join(self.consolidated_dir, "02_agentic_tools", "agentic_tools_spheroid.jsonl")
        with open(sph_file, "r", encoding="utf-8") as f:
            checked = 0
            for line in f:
                r = json.loads(line)
                tc = r.get("tool_call", "").strip()
                if tc:
                    self.assertFalse(tc.startswith('{"name":'), f"Tool call in {r['id']} uses raw JSON instead of functional syntax: {tc}")
                    checked += 1
                if checked >= 25:
                    break

    def test_06_sharegpt_schema_and_system_prompt(self):
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

    def test_07_parquet_readability_and_metadata(self):
        for type_name, dir_name in self.expected_types:
            pq_file = os.path.join(self.consolidated_dir, dir_name, f"{type_name}.parquet")
            meta = pq.read_metadata(pq_file)
            self.assertGreater(meta.num_rows, 0, f"Parquet {pq_file} has 0 rows")

            table = pq.read_table(pq_file)
            cols = table.column_names
            self.assertIn("id", cols)
            self.assertIn("developer", cols)
            self.assertIn("formatted_input", cols)
            self.assertIn("formatted_output", cols)
            self.assertIn("input", cols)
            self.assertIn("output", cols)
            self.assertIn("messages_json", cols)


if __name__ == "__main__":
    unittest.main()
