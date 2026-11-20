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
    # Normal factual answers and conversational prose must stay outside tool tags
    assert format_motor_response("Paris") == "Paris"
    assert format_motor_response("The capital of France is Paris.") == "The capital of France is Paris."
    assert format_motor_response("42") == "42"
    assert "<tool_call>" not in format_motor_response("def foo(x):\n    return x + 1")
    assert "<act>" not in format_motor_response("def foo(x):\n    return x + 1")


def test_format_tool_action_encloses_tool_invocations():
    # Explicit tool action format uses industry standard <tool_call>
    assert format_tool_action("calc(15 + 27)") == "<tool_call> calc(15 + 27) </tool_call>"
    assert format_tool_action("python(code='max(9.11, 9.9)')") == "<tool_call> python(code='max(9.11, 9.9)') </tool_call>"
    # format_motor_response should wrap explicit tool prefixes
    assert format_motor_response("calc: 15 + 27") == "<tool_call> calc: 15 + 27 </tool_call>"
    assert format_motor_response("calc(2 + 2)") == "<tool_call> calc(2 + 2) </tool_call>"
    # Legacy <act> strings are migrated to <tool_call>
    assert format_motor_response("<act> calc(5) </act>") == "<tool_call> calc(5) </tool_call>"


def test_clean_motor_response_prioritizes_answer_outside_act():
    # When both tool call and final answer exist, answer is outside tool tags
    text = "<tool_call> calc(15 + 27) </tool_call>\nAnswer: 42"
    assert clean_motor_response(text) == "Answer: 42"

    # Direct factual answers without tags
    assert clean_motor_response("Berlin") == "Berlin"

    # Thinking + tool call + answer
    complex_text = "<think>\nComputing sum\n</think>\n<tool_call> calc(10 + 20) </tool_call>\nAnswer: 30"
    assert clean_motor_response(complex_text) == "Answer: 30"

    # Pure tool action fallback when no answer text is produced
    assert clean_motor_response("<tool_call> calc(15 + 27) </tool_call>") == "calc(15 + 27)"

    # Backward compatibility with legacy <act>
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
    raw = "<think>\nThinking step\n</think>\n<tool_call> calc(100 / 4) </tool_call>\nAnswer: 25"
    parsed = parse_model_output(raw)
    assert parsed["thinking"] == "Thinking step"
    assert parsed["tool_calls"] == ["calc(100 / 4)"]
    assert parsed["answer"] == "Answer: 25"

    # Backward compatibility with legacy <act>
    raw_legacy = "<think>\nThinking step\n</think>\n<act> calc(100 / 4) </act>\nAnswer: 25"
    parsed_legacy = parse_model_output(raw_legacy)
    assert parsed_legacy["tool_calls"] == ["calc(100 / 4)"]

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
    assert "<tool_call>" not in norm
    assert "def add" in norm

    fenced = "```python\ndef solve():\n    return 42\n```"
    norm_fenced = _normalize_3tier_response(fenced)
    assert "<act>" not in norm_fenced
    assert "<tool_call>" not in norm_fenced
    assert "def solve" in norm_fenced


def test_normalize_3tier_response_gsm8k_derivation():
    gsm = "John has 2 apples and buys 3 more. #### 5"
    norm_gsm = _normalize_3tier_response(gsm)
    assert "<tool_call> calc(5) </tool_call>" in norm_gsm
    assert "<act>" not in norm_gsm
    assert "Answer: 5" in norm_gsm


def test_suppress_premature_act_penalty():
    import torch
    from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5

    cfg = ModelConfigV5(vocab_size=1000, surface_dim=64, radial_dim=32, num_shells=4, num_shortcut_tracts=4, num_heads=2, tool_token_id=33)
    model = CerebroSpheroidV5(cfg)
    
    prompts = torch.tensor([[10, 20, 30]])
    targets = torch.tensor([[40, 50, 60]]) # No tool (33) in target
    
    loss_standard, _ = model(prompts=prompts, targets=targets, suppress_premature_act=False)
    loss_penalized, _ = model(prompts=prompts, targets=targets, suppress_premature_act=True, act_penalty_weight=1.0)
    
    assert loss_penalized >= loss_standard


def test_tokenizer_tool_delimiters_are_atomic_tokens():
    """Verifies that <tool_call> and <tool_response> are registered as atomic single tokens."""
    from src.corpus import build_tokenizer, get_special_token_ids

    tok = build_tokenizer()
    spec = get_special_token_ids(tok)

    assert spec.get("tool_call") == 31
    assert spec.get("tool_call_end") == 32
    assert spec.get("tool_response") == 33
    assert spec.get("tool_response_end") == 34

    # Each tag must encode to exactly one token ID, not multiple split subwords
    for tag, expected_id in [
        ("<tool_call>", 31),
        ("</tool_call>", 32),
        ("<tool_response>", 33),
        ("</tool_response>", 34),
    ]:
        enc = tok.encode(tag, add_special_tokens=False)
        assert enc == [expected_id], f"{tag} expected [{expected_id}], but got {enc}"
        assert tok.decode([expected_id]) == tag


def test_collate_cerebro_v5_batch_target_padding_minus_100():
    """Verifies that collate_cerebro_v5_batch strictly pads targets with -100 (never 0 / <unk>)."""
    import torch
    from src.v5_dataset_loader import collate_cerebro_v5_batch

    batch = [
        {
            "prompt_ids": torch.tensor([10, 11]),
            "target_ids": torch.tensor([20, 21, 22, 23]),
            "weights": torch.tensor([1.0, 1.0, 1.0, 1.0]),
            "thinking_level": 1,
        },
        {
            "prompt_ids": torch.tensor([10, 11, 12, 13]),
            "target_ids": torch.tensor([20, 21]),
            "weights": torch.tensor([1.0, 1.0]),
            "thinking_level": 0,
        },
    ]

    collated = collate_cerebro_v5_batch(batch)
    targets = collated["targets"]

    assert targets.shape == (2, 4)
    # Sample 0 has all 4 target tokens
    assert targets[0].tolist() == [20, 21, 22, 23]
    # Sample 1 has 2 target tokens and 2 padded positions
    assert targets[1, :2].tolist() == [20, 21]
    assert targets[1, 2:].tolist() == [-100, -100], "Target padding MUST be -100 to avoid training on <unk> (0)!"


def test_cross_entropy_ignores_minus_100_padding():
    """Verifies that F.cross_entropy with ignore_index=-100 ignores all -100 padding."""
    import torch
    import torch.nn.functional as F

    logits = torch.randn(2, 6, 100)
    # Both samples have identical first 3 tokens
    targets = torch.tensor([
        [15, 25, 35, -100, -100, -100],
        [15, 25, 35, -100, -100, -100],
    ])

    loss1 = F.cross_entropy(logits.view(-1, 100), targets.view(-1), ignore_index=-100)

    # If targets had 0 (<unk>), loss would be totally different and wrong
    targets_with_zeros = torch.tensor([
        [15, 25, 35, 0, 0, 0],
        [15, 25, 35, 0, 0, 0],
    ])
    loss_zeros = F.cross_entropy(logits.view(-1, 100), targets_with_zeros.view(-1), ignore_index=-100)

    assert not torch.isclose(loss1, loss_zeros), "Padding with 0 (<unk>) pollutes loss vs ignoring -100!"

