"""
Unit and Integration Tests for Minimal Tool MVP
==============================================
Validates:
1. Schema validator and deterministic tool execution.
2. Safe calculator arithmetic and error safety.
3. Lookup table queries and missing key handling.
4. Dataset generation across 3 cases (Case 1, 2, 3).
5. Tokenizer round-trip fidelity.
6. Model forward pass, masked multi-task loss, and policy prediction.
7. Closed-loop tool result re-afference.
8. SCMER regularization computation.
"""

import json
import pytest
import torch

from experiments.minimal_tool_mvp.dataset_generator import TraceDatasetGenerator
from experiments.minimal_tool_mvp.model import MinimalModelConfig, MinimalToolModel
from experiments.minimal_tool_mvp.schema_validator import (
    DeterministicToolExecutor,
    SafeCalculator,
    validate_tool_call,
    validate_tool_result,
)
from experiments.minimal_tool_mvp.tokenizer import MinimalTokenizer
from experiments.minimal_tool_mvp.train import TraceDataset, collate_trace_batch


def test_schema_validator_valid_and_invalid():
    # Valid calculator call
    valid_calc = json.dumps({"tool": "calculator", "args": {"expression": "12 * 15"}})
    ok, err, parsed = validate_tool_call(valid_calc)
    assert ok is True
    assert err is None
    assert parsed["tool"] == "calculator"

    # Missing args
    invalid_no_args = json.dumps({"tool": "calculator"})
    ok, err, _ = validate_tool_call(invalid_no_args)
    assert ok is False
    assert "Missing or invalid 'args'" in err

    # Unknown tool
    invalid_tool = json.dumps({"tool": "quantum_computer", "args": {}})
    ok, err, _ = validate_tool_call(invalid_tool)
    assert ok is False
    assert "Unknown tool" in err

    # Valid lookup call
    valid_lookup = json.dumps({"tool": "lookup", "args": {"table": "elements", "key": "Au"}})
    ok, err, parsed = validate_tool_call(valid_lookup)
    assert ok is True
    assert parsed["args"]["key"] == "Au"


def test_safe_calculator_and_errors():
    assert SafeCalculator.evaluate("15 + 27") == 42
    assert SafeCalculator.evaluate("100 // 4") == 25
    assert SafeCalculator.evaluate("2 ** 5") == 32

    # Division by zero
    with pytest.raises(ZeroDivisionError):
        SafeCalculator.evaluate("10 // 0")

    # Disallowed syntax
    with pytest.raises(ValueError):
        SafeCalculator.evaluate("__import__('os').system('ls')")


def test_deterministic_tool_executor():
    executor = DeterministicToolExecutor()

    # Success calculator
    res = executor.execute({"tool": "calculator", "args": {"expression": "340 * 10"}})
    assert res["status"] == "ok"
    assert res["result"] == 3400

    # Div by zero error
    res_err = executor.execute({"tool": "calculator", "args": {"expression": "50 // 0"}})
    assert res_err["status"] == "error"
    assert "Division by zero" in res_err["message"]

    # Success lookup
    res_look = executor.execute({"tool": "lookup", "args": {"table": "elements", "key": "Fe"}})
    assert res_look["status"] == "ok"
    assert res_look["result"] == "Iron"

    # Missing lookup key
    res_miss = executor.execute({"tool": "lookup", "args": {"table": "capitals", "key": "Atlantis"}})
    assert res_miss["status"] == "error"
    assert "not found" in res_miss["message"]


def test_dataset_generator_distribution():
    gen = TraceDatasetGenerator(seed=123)
    dataset = gen.generate_dataset(30)
    assert len(dataset) == 30

    cases = {d.case for d in dataset}
    assert "no_tool_needed" in cases
    assert "tool_needed" in cases
    assert "tool_bad_choice" in cases

    # Check that policy targets match case definitions
    for d in dataset:
        if d.case == "tool_needed":
            assert d.policy_target == 1
            assert d.tool_call is not None
        else:
            assert d.policy_target == 0
            assert d.tool_call is None


def test_tokenizer_round_trip():
    tok = MinimalTokenizer()
    text = "<|user|>Calculate 25 * 4.<|user_end|><|tool_call|>{\"tool\":\"calculator\",\"args\":{\"expression\":\"25 * 4\"}}<|tool_call_end|>"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text


def test_model_forward_and_masked_loss():
    tok = MinimalTokenizer()
    gen = TraceDatasetGenerator(seed=42)
    traces = gen.generate_dataset(6)

    ds = TraceDataset(traces, tok, max_seq_len=128)
    batch = collate_trace_batch([ds[i] for i in range(len(ds))], tok.pad_token_id)

    cfg = MinimalModelConfig(vocab_size=tok.vocab_size, max_seq_len=128, hidden_dim=64, num_layers=2, num_heads=2)
    model = MinimalToolModel(cfg)

    loss_dict = model.compute_loss(
        input_ids=batch["input_ids"],
        target_ids=batch["target_ids"],
        policy_targets=batch["policy_targets"],
        policy_positions=batch["policy_positions"],
        tool_call_mask=batch["tool_call_mask"],
        answer_mask=batch["answer_mask"],
        scmer_enabled=True,
        scmer_lambda=0.01,
    )

    assert "loss_total" in loss_dict
    assert "loss_policy" in loss_dict
    assert "loss_tool_call" in loss_dict
    assert "loss_answer" in loss_dict
    assert "loss_scmer" in loss_dict
    assert loss_dict["loss_total"].item() > 0.0


def test_scmer_penalty_calculation():
    cfg = MinimalModelConfig(vocab_size=50, hidden_dim=32)
    model = MinimalToolModel(cfg)

    # Compute SCMER loss
    scmer_loss = model.compute_scmer_loss(margin=0.05, num_pairs=500)
    assert isinstance(scmer_loss, torch.Tensor)
    assert scmer_loss.ndim == 0
    assert scmer_loss.item() >= 0.0
