"""
Unit test for CerebroSpheroid V5 Graded Curriculum Engine
Tests:
1. Phase step boundary transitions (0 to 1,000,000)
2. Continuous Learning Rate piecewise decay
3. On-the-fly streaming batch generator (Space-saving verification)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.graded_curriculum import (
    CurriculumPhase,
    CurriculumScheduleConfig,
    GradedCurriculumStreamer,
    compute_curriculum_lr,
    get_curriculum_phase,
)


def test_phase_boundaries():
    print("Testing Phase Step Boundaries...")
    cfg = CurriculumScheduleConfig(
        total_steps=1_000_000,
        phase_1_end=50_000,
        phase_2_end=500_000,
        phase_3_end=900_000,
    )

    # Phase 1: 0 to 50k
    assert get_curriculum_phase(0, cfg) == CurriculumPhase.PHASE_1_BASIC
    assert get_curriculum_phase(25_000, cfg) == CurriculumPhase.PHASE_1_BASIC
    assert get_curriculum_phase(49_999, cfg) == CurriculumPhase.PHASE_1_BASIC

    # Phase 2: 50k to 500k
    assert get_curriculum_phase(50_000, cfg) == CurriculumPhase.PHASE_2_REASONING
    assert get_curriculum_phase(250_000, cfg) == CurriculumPhase.PHASE_2_REASONING
    assert get_curriculum_phase(499_999, cfg) == CurriculumPhase.PHASE_2_REASONING

    # Phase 3: 500k to 900k
    assert get_curriculum_phase(500_000, cfg) == CurriculumPhase.PHASE_3_WORKFLOWS
    assert get_curriculum_phase(700_000, cfg) == CurriculumPhase.PHASE_3_WORKFLOWS
    assert get_curriculum_phase(899_999, cfg) == CurriculumPhase.PHASE_3_WORKFLOWS

    # Phase 4: 900k to 1M
    assert get_curriculum_phase(900_000, cfg) == CurriculumPhase.PHASE_4_CONSOLIDATION
    assert get_curriculum_phase(950_000, cfg) == CurriculumPhase.PHASE_4_CONSOLIDATION
    assert get_curriculum_phase(1_000_000, cfg) == CurriculumPhase.PHASE_4_CONSOLIDATION

    print("✓ Phase boundaries test passed.")


def test_lr_schedule_continuity():
    print("Testing Learning Rate Schedule Continuity...")
    cfg = CurriculumScheduleConfig(
        total_steps=1_000_000,
        phase_1_end=50_000,
        phase_2_end=500_000,
        phase_3_end=900_000,
        warmup_steps=2_000,
        max_lr=3e-4,
        phase_2_min_lr=1.5e-4,
        phase_3_min_lr=5e-5,
        min_lr=0.0,
    )

    # Warmup
    lr_0 = compute_curriculum_lr(0, cfg)
    lr_warm = compute_curriculum_lr(1_999, cfg)
    assert lr_0 > 0.0 and lr_warm <= 3e-4

    # Phase 1 plateau
    assert compute_curriculum_lr(10_000, cfg) == 3e-4
    assert compute_curriculum_lr(50_000, cfg) == 3e-4

    # Boundary 2 transition (at 500k)
    lr_500k = compute_curriculum_lr(500_000, cfg)
    assert abs(lr_500k - 1.5e-4) < 1e-6, f"Expected 1.5e-4, got {lr_500k}"

    # Boundary 3 transition (at 900k)
    lr_900k = compute_curriculum_lr(900_000, cfg)
    assert abs(lr_900k - 5e-5) < 1e-6, f"Expected 5e-5, got {lr_900k}"

    # End of run (at 1M)
    lr_1m = compute_curriculum_lr(1_000_000, cfg)
    assert abs(lr_1m - 0.0) < 1e-6, f"Expected 0.0, got {lr_1m}"

    print("✓ LR continuity test passed.")


def test_streaming_batch_generation():
    print("Testing GradedCurriculumStreamer On-The-Fly Batching...")
    cfg = CurriculumScheduleConfig(seq_len=64, batch_size=2)

    class MockTokenizer:
        vocab_size = 1000
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 1000 for c in text]

    streamer = GradedCurriculumStreamer(
        tokenizer=MockTokenizer(),
        config=cfg,
        search_dirs=["data", "../reasoning_agent/data"],
    )

    device = torch.device("cpu")

    # Test sampling across each phase
    for test_step in [1_000, 100_000, 600_000, 950_000]:
        inp, tgt, act, vis, ph = streamer.get_batch(step=test_step, device=device)
        assert inp.shape == (2, 64), f"Expected (2, 64), got {inp.shape}"
        assert tgt.shape == (2, 64), f"Expected (2, 64), got {tgt.shape}"
        assert act.shape == (2,), f"Expected (2,), got {act.shape}"
        if vis is not None:
            assert vis.shape == (2, cfg.num_visual_patches, cfg.visual_dim), f"Expected (2, 16, 512), got {vis.shape}"
        assert isinstance(ph, CurriculumPhase)

    print("✓ GradedCurriculumStreamer on-the-fly multimodal batching passed.")


if __name__ == "__main__":
    test_phase_boundaries()
    test_lr_schedule_continuity()
    test_streaming_batch_generation()
    print("\nAll Graded Curriculum tests PASSED successfully!")
