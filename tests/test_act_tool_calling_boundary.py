import pytest
from src.corpus import (
    format_motor_response,
    format_tool_action,
    clean_motor_response,
    parse_model_output,
    is_valid_tool_call,
)
from src.elastic_window import _normalize_3tier_response


def test_format_motor_response_normal_output_outside_act():
    # Normal factual answers and conversational prose must stay outside <act>
    assert format_motor_response("Paris") == "Paris"
    assert format_motor_response("The capital of France is Paris.") == "The capital of France is Paris."
    assert format_motor_response("42") == "42"
    assert "<act>" not in format_motor_response("def foo(x):\n    return x + 1")


def test_format_tool_action_encloses_tool_invocations():
    # Explicit tool action format
    assert format_tool_action("calc(15 + 27)") == "<act> calc(15 + 27) </act>"
    assert format_tool_action("python(code='max(9.11, 9.9)')") == "<act> python(code='max(9.11, 9.9)') </act>"
    # format_motor_response should wrap explicit tool prefixes
    assert format_motor_response("calc: 15 + 27") == "<act> calc: 15 + 27 </act>"
    assert format_motor_response("calc(2 + 2)") == "<act> calc(2 + 2) </act>"


def test_clean_motor_response_prioritizes_answer_outside_act():
    # When both tool call and final answer exist, answer is outside <act>
    text = "<act> calc(15 + 27) </act>\nAnswer: 42"
    assert clean_motor_response(text) == "Answer: 42"

    # Direct factual answers without tags
    assert clean_motor_response("Berlin") == "Berlin"

    # Thinking + tool call + answer
    complex_text = "<think>\nComputing sum\n</think>\n<act> calc(10 + 20) </act>\nAnswer: 30"
    assert clean_motor_response(complex_text) == "Answer: 30"

    # Pure tool action fallback when no answer text is produced
    assert clean_motor_response("<act> calc(15 + 27) </act>") == "calc(15 + 27)"


def test_is_valid_tool_call():
    assert is_valid_tool_call("calc(15 + 27)") is True
    assert is_valid_tool_call("calc: 2 + 2") is True
    assert is_valid_tool_call("calculator: 42") is True
    assert is_valid_tool_call("python(code='2**16')") is True
    assert is_valid_tool_call("sql: SELECT * FROM table;") is True
    assert is_valid_tool_call("search('weather in paris')") is True

    # Prose or text should NOT be treated as tool calls
    assert is_valid_tool_call("The capital of France is Paris.") is False
    assert is_valid_tool_call("Pumping Systematic oxygenation.") is False
    assert is_valid_tool_call("Hello world!") is False


def test_parse_model_output_strictly_separates_tiers():
    raw = "<think>\nThinking step\n</think>\n<act> calc(100 / 4) </act>\nAnswer: 25"
    parsed = parse_model_output(raw)
    assert parsed["thinking"] == "Thinking step"
    assert parsed["tool_calls"] == ["calc(100 / 4)"]
    assert parsed["answer"] == "Answer: 25"

    # Factual QA without tool calls
    raw_fact = "The capital of France is Paris."
    parsed_fact = parse_model_output(raw_fact)
    assert parsed_fact["thinking"] == ""
    assert parsed_fact["tool_calls"] == []
    assert parsed_fact["answer"] == "The capital of France is Paris."


def test_normalize_3tier_response_code_remains_outside_act():
    code = "def add(a: int, b: int) -> int:\n    return a + b"
    norm = _normalize_3tier_response(code)
    assert "<act>" not in norm
    assert "def add" in norm

    fenced = "```python\ndef solve():\n    return 42\n```"
    norm_fenced = _normalize_3tier_response(fenced)
    assert "<act>" not in norm_fenced
    assert "def solve" in norm_fenced


def test_normalize_3tier_response_gsm8k_derivation():
    gsm = "John has 2 apples and buys 3 more. #### 5"
    norm_gsm = _normalize_3tier_response(gsm)
    assert "<act> calc(5) </act>" in norm_gsm
    assert "Answer: 5" in norm_gsm


def test_suppress_premature_act_penalty():
    import torch
    from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5

    cfg = ModelConfigV5(vocab_size=1000, surface_dim=64, radial_dim=32, num_shells=4, num_shortcut_tracts=4, num_heads=2)
    model = CerebroSpheroidV5(cfg)
    
    prompts = torch.tensor([[10, 20, 30]])
    targets = torch.tensor([[40, 50, 60]]) # No act (33) in target
    
    loss_standard, _ = model(prompts=prompts, targets=targets, suppress_premature_act=False)
    loss_penalized, _ = model(prompts=prompts, targets=targets, suppress_premature_act=True, act_penalty_weight=1.0)
    
    assert loss_penalized >= loss_standard

