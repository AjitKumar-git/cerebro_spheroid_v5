"""
Unit test for CerebroSpheroid V5 Agentic Harness
Tests:
1. ActionType enum & ToolActionHead forward pass
2. SandboxedToolExecutor (bash, python, calculator, file_read)
3. AgenticSpheroidHarness initialization & sensory re-afference
4. Multi-turn persistent working memory continuity
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5
from src.agentic_harness import (
    ActionType,
    ToolActionHead,
    SandboxedToolExecutor,
    AgenticSpheroidHarness,
)


def test_tool_action_head():
    print("Testing ToolActionHead...")
    D = 128
    head = ToolActionHead(surface_dim=D, num_actions=len(ActionType))
    x = torch.randn(2, D)
    logits, probs = head(x)
    assert logits.shape == (2, len(ActionType)), f"Expected (2, {len(ActionType)}), got {logits.shape}"
    assert probs.shape == (2, len(ActionType)), f"Expected (2, {len(ActionType)}), got {probs.shape}"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5), "Probs must sum to 1.0"
    print("✓ ToolActionHead test passed.")


def test_sandboxed_executor():
    print("Testing SandboxedToolExecutor...")
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = SandboxedToolExecutor(workspace_root=tmpdir)

        # Calculator test
        calc_out = executor.execute_calculator("2 * (3 + 4)")
        assert calc_out == "14", f"Expected '14', got '{calc_out}'"

        # Python test
        py_out = executor.execute_python("print(7 * 6)")
        assert py_out == "42", f"Expected '42', got '{py_out}'"

        # Bash test
        bash_out = executor.execute_bash("echo 'cerebro'")
        assert bash_out == "cerebro", f"Expected 'cerebro', got '{bash_out}'"

        # File write and read test
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("cortical mantle active")
        read_out = executor.execute_file_read("test.txt")
        assert read_out == "cortical mantle active", f"Expected 'cortical mantle active', got '{read_out}'"

    print("✓ SandboxedToolExecutor test passed.")


def test_agentic_harness_integration():
    print("Testing AgenticSpheroidHarness with Mini Config...")
    config = ModelConfigV5(
        vocab_size=256,
        surface_dim=64,
        radial_dim=32,
        num_shells=8,
        thinking_cycles=2,
        max_seq_len=128,
        num_shortcut_tracts=8,
        top_k_tracts=4,
    )
    model = CerebroSpheroidV5(config)
    harness = AgenticSpheroidHarness(model)

    device = torch.device("cpu")
    harness.to(device)

    # Mock tokenizer
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 256 for c in text[:16]]
        def decode(self, token_ids, skip_special_tokens=True):
            return "".join([chr(t) for t in token_ids])

    tokenizer = MockTokenizer()

    # Step 1: conversational turn
    res1 = harness.step_agentic_turn(
        user_prompt="Hello Cerebro",
        tokenizer=tokenizer,
        device=device,
        max_tokens=8,
        force_action=ActionType.CHAT,
    )
    assert res1["action_id"] == ActionType.CHAT
    assert harness.persistent_state is not None
    assert harness.persistent_state.shape == (1, 8, 64)

    # Step 2: calculator tool turn
    res2 = harness.step_agentic_turn(
        user_prompt="Compute 12 * 12",
        tokenizer=tokenizer,
        device=device,
        max_tokens=8,
        force_action=ActionType.CALCULATOR,
    )
    assert res2["action_id"] == ActionType.CALCULATOR
    assert "tool_output" in res2
    # Check that state persists across turns
    assert harness.persistent_state is not None

    print("✓ AgenticSpheroidHarness integration test passed.")


if __name__ == "__main__":
    test_tool_action_head()
    test_sandboxed_executor()
    test_agentic_harness_integration()
    print("\nAll Agentic Harness tests PASSED successfully!")
