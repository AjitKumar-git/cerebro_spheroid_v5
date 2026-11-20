"""
Synthetic Dataset Generator for Minimal Tool MVP
===============================================
Generates structured training and evaluation traces across three distinct cases:
1. No tool needed: Direct answer
2. Tool needed: Optional tool call -> tool result -> final answer
3. Tool is a bad choice: Direct answer despite tool availability / distractor

Supports structured JSON traces with schema validation and deliberate error traces.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from experiments.minimal_tool_mvp.schema_validator import (
    LOOKUP_TABLES,
    DeterministicToolExecutor,
    validate_tool_call,
    validate_tool_result,
)


@dataclass
class ToolTrace:
    trace_id: str
    case: str  # "no_tool_needed" | "tool_needed" | "tool_bad_choice"
    user_request: str
    policy_target: int  # 0 = direct_answer, 1 = call_tool
    tool_call: Optional[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    final_answer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TraceDatasetGenerator:
    """Generates synthetic traces for training and evaluation."""

    FACTUAL_DIRECT_QA = [
        ("What is the capital of France?", "Paris"),
        ("What color is a ripe banana?", "Yellow"),
        ("How many days are in a leap year?", "366"),
        ("Is 11 an even or odd number?", "Odd"),
        ("What is the boiling point of water in Celsius?", "100"),
        ("What is the capital of Japan?", "Tokyo"),
        ("How many legs does a spider have?", "8"),
        ("What is the chemical symbol for water?", "H2O"),
        ("Which planet is closest to the Sun?", "Mercury"),
        ("How many minutes are in an hour?", "60"),
        ("What is the opposite of hot?", "Cold"),
        ("What is the freezing point of water in Celsius?", "0"),
        ("How many continents are on Earth?", "7"),
        ("What is the primary color of grass?", "Green"),
        ("Which ocean is the largest on Earth?", "Pacific Ocean"),
    ]

    ELEMENT_NAMES = list(LOOKUP_TABLES["elements"].items())
    CAPITAL_NAMES = list(LOOKUP_TABLES["capitals"].items())
    PORT_NAMES = list(LOOKUP_TABLES["ports"].items())

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.executor = DeterministicToolExecutor()

    def generate_case_1_no_tool(self, idx: int) -> ToolTrace:
        """Case 1: No tool needed (answer directly)."""
        sub_type = self.rng.choice(["fact", "trivial_math", "simple_identity"])
        if sub_type == "fact":
            q, a = self.rng.choice(self.FACTUAL_DIRECT_QA)
            user_req = q
            final_ans = a
        elif sub_type == "trivial_math":
            a = self.rng.randint(1, 9)
            b = self.rng.randint(1, 9)
            op = self.rng.choice(["+", "-"])
            res = a + b if op == "+" else a - b
            user_req = f"What is {a} {op} {b}?"
            final_ans = str(res)
        else:
            word = self.rng.choice(["alpha", "bravo", "charlie", "delta", "echo"])
            user_req = f"Repeat the word {word}."
            final_ans = word

        return ToolTrace(
            trace_id=f"c1_{idx:05d}",
            case="no_tool_needed",
            user_request=user_req,
            policy_target=0,
            tool_call=None,
            tool_result=None,
            final_answer=final_ans,
        )

    def generate_case_2_tool_needed(self, idx: int) -> ToolTrace:
        """Case 2: Tool needed (call calculator or lookup)."""
        tool_type = self.rng.choice(["calculator_mult", "calculator_div", "calculator_pow", "lookup", "error_case"])

        if tool_type == "calculator_mult":
            a = self.rng.randint(23, 987)
            b = self.rng.randint(17, 89)
            expr = f"{a} * {b}"
            user_req = f"Calculate {a} * {b}."
            call = {"tool": "calculator", "args": {"expression": expr}}
            res = self.executor.execute(call)
            final_ans = f"{a} * {b} is {res['result']}."

        elif tool_type == "calculator_div":
            b = self.rng.randint(12, 60)
            res_val = self.rng.randint(15, 250)
            a = b * res_val
            expr = f"{a} // {b}"
            user_req = f"Evaluate {a} // {b}."
            call = {"tool": "calculator", "args": {"expression": expr}}
            res = self.executor.execute(call)
            final_ans = f"{a} // {b} is {res['result']}."

        elif tool_type == "calculator_pow":
            base = self.rng.randint(3, 19)
            exp = self.rng.randint(3, 5)
            expr = f"{base} ** {exp}"
            user_req = f"Compute {base} ^ {exp}."
            call = {"tool": "calculator", "args": {"expression": expr}}
            res = self.executor.execute(call)
            final_ans = f"{base} raised to {exp} is {res['result']}."

        elif tool_type == "lookup":
            sub = self.rng.choice(["elements", "capitals", "ports"])
            if sub == "elements":
                sym, name = self.rng.choice(self.ELEMENT_NAMES)
                user_req = f"Lookup the element name for symbol {sym} in elements table."
                call = {"tool": "lookup", "args": {"table": "elements", "key": sym}}
                res = self.executor.execute(call)
                final_ans = f"The element for symbol {sym} is {res['result']}."
            elif sub == "capitals":
                country, cap = self.rng.choice(self.CAPITAL_NAMES)
                user_req = f"Query the capital of {country} from the capitals table."
                call = {"tool": "lookup", "args": {"table": "capitals", "key": country}}
                res = self.executor.execute(call)
                final_ans = f"The capital of {country} is {res['result']}."
            else:
                proto, port = self.rng.choice(self.PORT_NAMES)
                user_req = f"What is the default port for {proto} in ports table?"
                call = {"tool": "lookup", "args": {"table": "ports", "key": proto}}
                res = self.executor.execute(call)
                final_ans = f"The port for {proto} is {res['result']}."

        else:
            # Deliberate error traces to teach robust error handling
            err_kind = self.rng.choice(["div_zero", "unknown_key"])
            if err_kind == "div_zero":
                n = self.rng.randint(10, 500)
                expr = f"{n} // 0"
                user_req = f"Calculate {n} // 0."
                call = {"tool": "calculator", "args": {"expression": expr}}
                res = self.executor.execute(call)
                final_ans = "Error: Division by zero is undefined."
            else:
                fake_key = self.rng.choice(["Atlantis", "Eldorado", "Valhalla", "ShangriLa"])
                user_req = f"Query the capital of {fake_key} from the capitals table."
                call = {"tool": "lookup", "args": {"table": "capitals", "key": fake_key}}
                res = self.executor.execute(call)
                final_ans = f"Error: {fake_key} was not found in the capitals table."

        return ToolTrace(
            trace_id=f"c2_{idx:05d}",
            case="tool_needed",
            user_request=user_req,
            policy_target=1,
            tool_call=call,
            tool_result=res,
            final_answer=final_ans,
        )

    def generate_case_3_bad_choice(self, idx: int) -> ToolTrace:
        """Case 3: Tool is a bad choice (answer directly even though tool is available/mentioned)."""
        prompt_style = self.rng.choice(["no_calc_comparison", "without_tools_math", "distractor_identity", "explicit_forbidden"])

        if prompt_style == "no_calc_comparison":
            a = self.rng.randint(10, 99)
            b = self.rng.randint(10, 99)
            while b == a:
                b = self.rng.randint(10, 99)
            is_greater = a > b
            ans_str = f"Yes, {a} is greater than {b}." if is_greater else f"No, {a} is not greater than {b}."
            user_req = f"Do not use a calculator: is {a} greater than {b}?"
            final_ans = ans_str

        elif prompt_style == "without_tools_math":
            a = self.rng.randint(1, 5)
            b = self.rng.randint(1, 5)
            user_req = f"Without using any tools, what is {a} + {b}?"
            final_ans = str(a + b)

        elif prompt_style == "distractor_identity":
            word = self.rng.choice(["apple", "banana", "robot", "galaxy", "tensor"])
            user_req = f"A calculator and lookup tool are available, but answer directly: spell {word} backwards."
            final_ans = word[::-1]

        else:
            word = self.rng.choice(["sun", "ocean", "mountain", "forest"])
            user_req = f"Do not call any tool: write '{word}' in uppercase."
            final_ans = word.upper()

        return ToolTrace(
            trace_id=f"c3_{idx:05d}",
            case="tool_bad_choice",
            user_request=user_req,
            policy_target=0,
            tool_call=None,
            tool_result=None,
            final_answer=final_ans,
        )

    def generate_dataset(self, num_samples: int = 3000) -> List[ToolTrace]:
        """Generates a balanced dataset across the 3 cases."""
        samples_per_case = num_samples // 3
        dataset: List[ToolTrace] = []

        for i in range(samples_per_case):
            dataset.append(self.generate_case_1_no_tool(i))
        for i in range(samples_per_case):
            dataset.append(self.generate_case_2_tool_needed(i))
        for i in range(num_samples - 2 * samples_per_case):
            dataset.append(self.generate_case_3_bad_choice(i))

        self.rng.shuffle(dataset)
        return dataset
