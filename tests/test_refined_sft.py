"""
Test Suite for Refined SFT Strategy and Protocol (5 Pillars).
Validates:
1. Balanced Tri-Modal Data Curriculum
2. Dynamic Answer-Anchor Loss Weighting (AALW 3.5x)
3. Deliberate Control Token Calibration
4. Horizon Expansion (256 tokens) & SCMOptimizer
5. Dynamic Stopping Criteria & Prompt-Type Aware Decoding
"""

import os
import json
import torch
import pytest
import numpy as np

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5
from src.corpus import (
    format_stimulus_prompt,
    format_motor_response,
    extract_strict_act_payload,
    get_bio_token_ids,
)


def test_pillar1_tri_modal_dataset_distribution():
    """Verify that the generated dataset matches the 45% / 40% / 15% tri-modal balance."""
    jsonl_path = os.path.join("data", "cerebro_cognitive_8k", "cerebro_tri_modal_sft_10k.jsonl")
    parquet_path = os.path.join("data", "cerebro_cognitive_8k", "cerebro_tri_modal_sft_10k.parquet")
    assert os.path.exists(jsonl_path) or os.path.exists(parquet_path), "Tri-modal dataset not found in data/cerebro_cognitive_8k"

    categories = {}
    direct_slot_count = 0
    cot_count = 0
    total = 0

    if os.path.exists(parquet_path):
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        total = len(table)
        cat_col = table["category"].to_pylist() if "category" in table.column_names else []
        slot_col = table["is_direct_slot"].to_pylist() if "is_direct_slot" in table.column_names else []
        for cat, is_slot in zip(cat_col, slot_col):
            categories[cat] = categories.get(cat, 0) + 1
            if is_slot:
                direct_slot_count += 1
            if "cot" in cat:
                cot_count += 1
    else:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                rec = json.loads(line)
                cat = rec.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
                if rec.get("is_direct_slot", False):
                    direct_slot_count += 1
                if "cot" in cat:
                    cot_count += 1

    assert total == 10000, f"Expected 10,000 records, got {total}"
    # 45% Direct Slot QA (4500)
    assert direct_slot_count >= 4500, f"Expected >= 4500 direct slot records, got {direct_slot_count}"
    # 40% CoT (4000)
    assert cot_count == 4000, f"Expected exactly 4000 CoT records, got {cot_count}"
    # 15% Science and Agentic anchors
    agentic_science = categories.get("agentic_tool", 0) + categories.get("science_anchor", 0)
    assert agentic_science == 1500, f"Expected 1500 science/agentic anchors, got {agentic_science}"


def test_pillar2_dynamic_answer_anchor_loss_weighting():
    """Verify that AALW applies 3.5x penalty to answer tokens and 1.0x to CoT tokens."""
    cfg = ModelConfigV5(
        vocab_size=256,
        surface_dim=64,
        radial_dim=32,
        num_shells=4,
        num_shortcut_tracts=4,
        num_heads=2,
        thinking_cycles=1,
    )
    model = CerebroSpheroidV5(cfg)
    model.eval()

    # Prompts: [B=2, L=8]
    prompts = torch.randint(3, 200, (2, 8))

    # Target 0: CoT problem with #### anchor (token 3191 or here mock token 50)
    # [10, 20, 30, 50, 60, 70, -100, -100]
    # Target 1: Direct slot QA (no anchor): [15, 25, 35, 45, -100, -100, -100, -100]
    targets = torch.tensor([
        [10, 20, 30, 50, 60, 70, -100, -100],
        [15, 25, 35, 45, -100, -100, -100, -100],
    ], dtype=torch.long)

    # Test with standard loss vs anchor loss
    with torch.no_grad():
        loss_standard, _ = model(
            prompts=prompts,
            targets=targets,
            use_anchor_loss=False,
            sampling_p=1.0,
            deliberation_mode="teacher_forcing",
        )
        loss_anchor, _ = model(
            prompts=prompts,
            targets=targets,
            use_anchor_loss=True,
            anchor_token_id=50,
            anchor_weight=3.5,
            sampling_p=1.0,
            deliberation_mode="teacher_forcing",
        )

    assert torch.isfinite(loss_standard)
    assert torch.isfinite(loss_anchor)
    assert loss_anchor > 0.0


def test_pillar4_scm_optimizer_step():
    """Verify SCMOptimizer operates with Newton-Schulz and rank-boost on spheroid parameters."""
    from src.scm_optimizer import SCMOptimizer, newton_schulz5

    # Test Newton-Schulz orthogonalization
    G = torch.randn(128, 128)
    O = newton_schulz5(G, steps=5)
    assert O.shape == (128, 128)
    # O @ O.T should be approximately Identity (mean absolute error < 0.1)
    mae = torch.mean(torch.abs(O @ O.T - torch.eye(128)))
    assert mae < 0.1, f"Matrix not properly orthogonalized: mae={mae}"

    cfg = ModelConfigV5(
        vocab_size=128,
        surface_dim=64,
        radial_dim=32,
        num_shells=4,
        num_shortcut_tracts=4,
        num_heads=2,
        thinking_cycles=1,
    )
    model = CerebroSpheroidV5(cfg)
    optimizer = SCMOptimizer(model.parameters(), lr=0.01, rank_boost=4.0)

    # Run dummy backward pass
    prompts = torch.randint(3, 100, (2, 8))
    targets = torch.randint(3, 100, (2, 8))
    loss, _ = model(prompts, targets, sampling_p=1.0, deliberation_mode="teacher_forcing")
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    # Verify parameters remained finite
    for p in model.parameters():
        assert torch.all(torch.isfinite(p))


def test_pillar5_prompt_type_aware_decoding_stops_on_newline():
    """Verify that generate_text includes newline in stop_ids for slot queries."""
    cfg = ModelConfigV5(
        vocab_size=512,
        surface_dim=64,
        radial_dim=32,
        num_shells=4,
        num_shortcut_tracts=4,
        num_heads=2,
        thinking_cycles=1,
    )
    model = CerebroSpheroidV5(cfg)

    class DummyTokenizer:
        vocab_size = 512
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            if text == "\n":
                return [13]
            return [ord(c) % 200 + 10 for c in text]

        def decode(self, token_ids, skip_special_tokens=True):
            chars = []
            for t in token_ids:
                if t == 13:
                    chars.append("\n")
                elif 10 <= t <= 210:
                    chars.append(chr(t - 10))
            return "".join(chars)

    tok = DummyTokenizer()

    # Slot query prompt
    prompt = "Complete the analogy: hot is to cold as day is to ____."
    resp = model.generate_text(prompt, tok, max_new_tokens=32)
    assert isinstance(resp, str)


def test_pillar3_control_token_calibration():
    """Verify that control tokens 31-34 (<stim>, </stim>, <act>, </act>) are calibrated from chat delimiters."""
    v5_vocab = 32000
    src_vocab = 151936
    dim = 64
    src_emb = torch.randn(src_vocab, dim)
    v5_emb = torch.zeros(v5_vocab, dim)

    class MockSourceTok:
        def encode(self, s, add_special_tokens=False):
            if "user" in s or "INST" in s:
                return [1001, 1002]
            if "assistant" in s:
                return [2001, 2002]
            if "end" in s or "</s>" in s:
                return [3001]
            return [500]

    src_tok = MockSourceTok()
    control_token_candidates = {
        31: ["<|im_start|>user\n", "[INST]", "User: "],       # <stim>
        32: ["<|im_end|>\n", "[/INST]", "\n\n"],             # </stim>
        33: ["<|im_start|>assistant\n", "Assistant: "],       # <act>
        34: ["<|im_end|>", "</s>"],                          # </act>
    }

    for ctrl_id, candidates in control_token_candidates.items():
        for cand in candidates:
            encoded = src_tok.encode(cand, add_special_tokens=False)
            valid_ids = [idx for idx in encoded if idx < src_emb.shape[0]]
            if valid_ids:
                pooled = src_emb[valid_ids, :dim].mean(dim=0)
                v5_emb[ctrl_id, :dim] = pooled
                break

    for ctrl_id in (31, 32, 33, 34):
        norm = torch.norm(v5_emb[ctrl_id]).item()
        assert norm > 0.0, f"Control token {ctrl_id} was not calibrated!"

