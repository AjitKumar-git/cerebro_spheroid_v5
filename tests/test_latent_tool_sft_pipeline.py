import pytest
from src.corpus import (
    format_latent_tool_response,
    build_tool_closed_loop_prompt,
    format_tool_closed_loop_turns,
    clean_motor_response,
    parse_model_output,
    is_valid_tool_call,
)
from src.elastic_window import _normalize_3tier_response


def test_normalize_strips_english_scratchpads():
    # Model should NOT be trained to decode latent deliberation into English scratchpads
    raw = "<think>\nLet's calculate step by step:\n1. 15 + 27 = 42\n</think>\nAnswer: 42"
    normalized = _normalize_3tier_response(raw, strip_scratchpad=True)
    assert "<think>" not in normalized
    assert "</think>" not in normalized
    assert "42" in normalized

    # When scratchpad and tool call co-exist, scratchpad is stripped and tool call is retained
    raw_tool = "<think>\nCalling calculator tool\n</think>\n<tool_call> calc(15 + 27) </tool_call>\nAnswer: 42"
    normalized_tool = _normalize_3tier_response(raw_tool, strip_scratchpad=True)
    assert "<think>" not in normalized_tool
    assert "<tool_call> calc(15 + 27) </tool_call>" in normalized_tool
    assert "Answer: 42" in normalized_tool


def test_normalize_gsm8k_derivation_without_forced_think():
    # GSM8K derivations ending in #### <ans> should format tool call and answer without <think> tags
    raw_gsm8k = "Tom has 15 apples and buys 27 more. Total is 15 + 27 = 42. #### 42"
    normalized = _normalize_3tier_response(raw_gsm8k, strip_scratchpad=True)
    assert "<think>" not in normalized
    assert "<tool_call> calc(42) </tool_call>" in normalized
    assert "Answer: 42" in normalized


def test_format_latent_tool_response():
    # Strips multiple variant thinking tags while preserving tool call and direct answer
    response = "<|begin_of_thought|>\nInternal reasoning\n<|end_of_thought|>\n<tool_call> python(code='2**10') </tool_call>\nAnswer: 1024"
    cleaned = format_latent_tool_response(response, strip_scratchpad=True)
    assert "<|begin_of_thought|>" not in cleaned
    assert "<tool_call> python(code='2**10') </tool_call>" in cleaned
    assert "1024" in cleaned

    # Direct factual answers are untouched
    assert format_latent_tool_response("The capital of France is Paris.") == "The capital of France is Paris."


def test_build_tool_closed_loop_prompt():
    prompt = "What is 123 * 456?"
    tool_call = "calc(123 * 456)"
    tool_response = "56088"
    loop_prompt = build_tool_closed_loop_prompt(prompt, tool_call, tool_response)
    assert prompt in loop_prompt
    assert "<tool_call> calc(123 * 456) </tool_call>" in loop_prompt
    assert "<tool_response> 56088 </tool_response>" in loop_prompt
    assert loop_prompt.endswith("Answer:")


def test_format_tool_closed_loop_turns():
    prompt = "Compute square root of 65536"
    tool_call = "calc(sqrt(65536))"
    tool_response = "256.0"
    final_answer = "The square root of 65536 is 256."

    turns = format_tool_closed_loop_turns(prompt, tool_call, tool_response, final_answer)
    assert len(turns) == 2

    turn1_prompt, turn1_target = turns[0]
    assert turn1_prompt == prompt
    assert turn1_target == "<tool_call> calc(sqrt(65536)) </tool_call>"

    turn2_prompt, turn2_target = turns[1]
    assert "<tool_response> 256.0 </tool_response>" in turn2_prompt
    assert turn2_target == "The square root of 65536 is 256."


def test_closed_loop_tool_execution_mock():
    # Verify the host tool execution contract
    executed_tools = []
    def mock_executor(cmd: str) -> str:
        executed_tools.append(cmd)
        if "15 + 27" in cmd:
            return "42"
        return "OK"

    tool_call_str = "calc(15 + 27)"
    result = mock_executor(tool_call_str)
    assert result == "42"
    assert executed_tools == ["calc(15 + 27)"]

    feedback = build_tool_closed_loop_prompt("What is 15 + 27?", tool_call_str, result)
    assert "<tool_response> 42 </tool_response>" in feedback
