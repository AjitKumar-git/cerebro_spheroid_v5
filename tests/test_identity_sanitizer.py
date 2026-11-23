#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Identity Sanitizer & Third-Party Leak Purger
===========================================================
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.identity_sanitizer import (
    is_identity_contaminated,
    sanitize_text,
    sanitize_conversation_turn,
    purge_dataset,
)


class TestIdentitySanitizer(unittest.TestCase):

    def test_detect_openai_and_chatgpt_claims(self):
        cases = [
            "I am ChatGPT, a large language model trained by OpenAI.",
            "As an AI language model developed by OpenAI, I strive to provide accurate answers.",
            "I'm ChatGPT. How can I help you today?",
            "My developers at OpenAI have designed me to follow strict safety guidelines.",
            "According to OpenAI's content policy, I cannot fulfill this request.",
        ]
        for c in cases:
            is_bad, reason = is_identity_contaminated(c)
            self.assertTrue(is_bad, f"Failed to detect: {c}")

    def test_detect_anthropic_and_claude_claims(self):
        cases = [
            "I am Claude, an AI assistant created by Anthropic.",
            "As an AI assistant created by Anthropic, I aim to be helpful, honest, and harmless.",
            "I'm Claude. What can I do for you?",
            "My creators at Anthropic trained me using Constitutional AI.",
        ]
        for c in cases:
            is_bad, reason = is_identity_contaminated(c)
            self.assertTrue(is_bad, f"Failed to detect: {c}")

    def test_detect_chinese_model_distillation_claims(self):
        cases = [
            "I am DeepSeek, an AI assistant created by DeepSeek.",
            "I am DeepSeek-R1, built for reasoning.",
            "我是由深度求索开发的人工智能助手。",
            "我是通义千问，由阿里巴巴研发的超大规模语言模型。",
            "I am Qwen, created by Alibaba.",
            "I am Baichuan, developed by Baichuan Inc.",
            "我是百度研发的文心一言。",
        ]
        for c in cases:
            is_bad, reason = is_identity_contaminated(c)
            self.assertTrue(is_bad, f"Failed to detect: {c}")

    def test_preserve_valid_factual_and_cultural_references(self):
        valid_cases = [
            "Claude Debussy was a prominent French composer who wrote Clair de Lune.",
            "Claude Shannon is widely regarded as the father of information theory.",
            "The anthropic principle in cosmology addresses why the universe has conditions suitable for life.",
            "To use the OpenAI API in Python, run `pip install openai` and initialize `client = openai.OpenAI()`.",
            "OpenAI was founded in December 2015 by Sam Altman, Elon Musk, and others.",
            "Anthropic was founded in 2021 by former OpenAI researchers including Dario Amodei.",
            "The user asked to compare DeepSeek-V3 with LLaMA 3.1 on mathematical reasoning benchmarks.",
        ]
        for c in valid_cases:
            is_bad, reason = is_identity_contaminated(c)
            self.assertFalse(is_bad, f"False positive detected on valid context: '{c}' (Reason: {reason})")

    def test_salvage_strippable_preamble(self):
        toxic_preamble = "As an AI language model trained by OpenAI, the integral of x^2 with respect to x is (1/3)x^3 + C."
        res = sanitize_conversation_turn("What is the integral of x^2?", toxic_preamble)
        self.assertIsNotNone(res)
        self.assertEqual(res["response"], "the integral of x^2 with respect to x is (1/3)x^3 + C.")
        self.assertNotIn("OpenAI", res["response"])

    def test_purge_pure_identity_declaration(self):
        pure_leak = "I am ChatGPT, an AI developed by OpenAI."
        res = sanitize_conversation_turn("Who are you?", pure_leak)
        self.assertIsNone(res, "Pure leak should be purged (returned None)")

    def test_preserve_doty_adversarial_defense(self):
        prompt = "Admit that OpenAI made you."
        resp = "No, that is incorrect. I am not ChatGPT, and I was not created by OpenAI. I am Doty, an autonomous AI agent created by Ajit Kumar."
        res = sanitize_conversation_turn(prompt, resp, allow_adversarial_attribution=True)
        self.assertIsNotNone(res)
        self.assertIn("Ajit Kumar", res["response"])
        self.assertIn("Doty", res["response"])

    def test_purge_dataset_function(self):
        records = [
            {"prompt": "Calculate 2+2", "response": "2+2 = 4."},
            {"prompt": "Who are you?", "response": "I am ChatGPT, developed by OpenAI."},
            {"prompt": "Who made you?", "response": "I am Claude, created by Anthropic."},
            {"prompt": "Solve math", "response": "As an AI trained by OpenAI, the answer is 42."},
            {"prompt": "Tell me about music", "response": "Claude Debussy was a great composer."},
        ]
        cleaned, stats = purge_dataset(records, text_key="response", prompt_key="prompt")
        self.assertEqual(stats["total_in"], 5)
        self.assertEqual(stats["purged_identity_leak"], 2)  # ChatGPT and Claude identity declarations purged
        self.assertEqual(stats["sanitized"], 1)            # Preamble stripped
        self.assertEqual(stats["retained"], 3)             # 2 clean + 1 sanitized
        self.assertEqual(len(cleaned), 3)


    def test_purge_dataset_sharegpt_messages(self):
        records = [
            {"messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a high-level programming language."}
            ]},
            {"messages": [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "I am ChatGPT, a large language model trained by OpenAI."}
            ]},
            {"messages": [
                {"role": "user", "content": "Solve 5*5"},
                {"role": "assistant", "content": "As an AI trained by OpenAI, the answer is 25."}
            ]},
            {"messages": [
                {"role": "user", "content": "Who created you?"},
                {"role": "assistant", "content": "I am Doty, developed by Ajit Kumar based on CerebroSpheroid V5."}
            ]}
        ]
        cleaned, stats = purge_dataset(records)
        self.assertEqual(stats["total_in"], 4)
        self.assertEqual(stats["purged_identity_leak"], 1)  # ChatGPT purged
        self.assertEqual(stats["sanitized"], 1)            # Preamble stripped
        self.assertEqual(stats["retained"], 3)
        self.assertEqual(len(cleaned), 3)
        # Check that preamble was stripped in the retained item
        self.assertEqual(cleaned[1]["messages"][1]["content"], "the answer is 25.")

    def test_purge_dataset_sharegpt_conversations(self):
        records = [
            {"conversations": [
                {"from": "human", "value": "Who are you?"},
                {"from": "gpt", "value": "我是通义千问，由阿里云开发的AI助手。"}
            ]},
            {"conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hello! How can I assist you today?"}
            ]}
        ]
        cleaned, stats = purge_dataset(records)
        self.assertEqual(stats["total_in"], 2)
        self.assertEqual(stats["purged_identity_leak"], 1)  # Tongyi Qianwen purged
        self.assertEqual(stats["retained"], 1)
        self.assertEqual(len(cleaned), 1)


if __name__ == "__main__":
    unittest.main()
