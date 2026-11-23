#!/usr/bin/env python3
"""
Unit Tests: Unified Agentic Tool Dispatcher
============================================
Verifies safety, execution accuracy, syntax parsing, and sandboxing
for the UnifiedToolDispatcher.
"""

import json
import pytest
from src.tools.unified_tool_dispatcher import UnifiedToolDispatcher


@pytest.fixture
def dispatcher():
    return UnifiedToolDispatcher()


class TestUnifiedToolDispatcher:
    def test_calculator_basic_arithmetic(self, dispatcher):
        assert dispatcher.execute_calculator("15 + 27") == "42"
        assert dispatcher.execute_calculator("100 / 4") == "25"
        assert dispatcher.execute_calculator("2 ** 10") == "1024"
        assert dispatcher.execute_calculator("2 ^ 8") == "256"

    def test_calculator_math_functions(self, dispatcher):
        assert dispatcher.execute_calculator("sqrt(144)") == "12"
        assert dispatcher.execute_calculator("abs(-42)") == "42"
        assert float(dispatcher.execute_calculator("sin(0)")) == 0.0

    def test_calculator_security_sandbox(self, dispatcher):
        # Prevent arbitrary imports or os calls
        res = dispatcher.execute_calculator("__import__('os').system('ls')")
        assert "Error" in res or "Unauthorized" in res

        res_open = dispatcher.execute_calculator("open('/etc/passwd').read()")
        assert "Error" in res_open or "Unauthorized" in res_open

    def test_python_repl_execution(self, dispatcher):
        code = "print(sum([1, 2, 3, 4, 5]))"
        out = dispatcher.execute_python(code)
        assert out == "15"

    def test_python_repl_timeout(self, dispatcher):
        code = "import time; time.sleep(5)"
        out = dispatcher.execute_python(code, timeout=1)
        assert "timed out" in out

    def test_factual_query_capitals(self, dispatcher):
        res_france = dispatcher.execute_factual_query("capital of France")
        assert "Paris" in res_france

        res_japan = dispatcher.execute_factual_query("capital of Japan")
        assert "Tokyo" in res_japan

    def test_factual_query_constants_and_chemicals(self, dispatcher):
        res_water = dispatcher.execute_factual_query("chemical formula for water")
        assert "H2O" in res_water

    def test_raw_call_string_parsing_functional(self, dispatcher):
        # Functional style
        assert dispatcher.execute_raw_call_string("calc(45 * 2)") == "90"
        assert dispatcher.execute_raw_call_string("calculator(100 - 35)") == "65"

    def test_raw_call_string_parsing_json(self, dispatcher):
        # Standard tool calling JSON format
        call_json = json.dumps({"name": "calc", "arguments": {"expression": "8 * 9"}})
        assert dispatcher.execute_raw_call_string(call_json) == "72"

    def test_raw_call_string_with_act_tags(self, dispatcher):
        call_with_tags = "<act> calc(50 + 50) </act>"
        assert dispatcher.execute_raw_call_string(call_with_tags) == "100"

        call_with_tool_call = "<tool_call> calc(7 * 7) </tool_call>"
        assert dispatcher.execute_raw_call_string(call_with_tool_call) == "49"
