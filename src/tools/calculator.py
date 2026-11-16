#!/usr/bin/env python3
"""
🧮 CerebroSpheroid V5: Sandboxed Mathematical Calculator Tool
============================================================
A fast, safe, AST-based mathematical expression evaluator designed for
the model to invoke during agentic tool-use (<act> calc(...) </act>).

Features:
  - Safe Abstract Syntax Tree (AST) traversal (zero eval() security risks)
  - Standard arithmetic: +, -, *, /, //, %, **, ^
  - Advanced math: sqrt, cbrt, sin, cos, tan, log, log2, log10, exp, factorial, gcd, lcm, abs, round
  - Constants: pi, e, tau, phi (golden ratio)
  - Unit percentages: 25% of 800, 50% * 120
  - Precision cleanup (e.g. 0.1 + 0.2 -> 0.3, 42.0 -> 42)
"""

import ast
import math
import operator
import re
import sys
from typing import Any, Callable, Dict, Union

# Allowed binary operators
SAFE_OPERATORS: Dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitXor: operator.pow,  # Support '^' as exponentiation (common in natural language math)
}

# Allowed unary operators
SAFE_UNARY_OPERATORS: Dict[type, Callable[..., Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed math functions
SAFE_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "cbrt": math.cbrt if hasattr(math, "cbrt") else lambda x: x ** (1.0 / 3.0),
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "log": math.log,
    "ln": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "factorial": math.factorial,
    "gamma": math.gamma,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "gcd": math.gcd,
    "lcm": math.lcm if hasattr(math, "lcm") else lambda a, b: abs(a * b) // math.gcd(a, b),
    "hypot": math.hypot,
    "degrees": math.degrees,
    "radians": math.radians,
}

# Mathematical constants
SAFE_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "PI": math.pi,
    "e": math.e,
    "E": math.e,
    "tau": math.tau,
    "phi": (1.0 + math.sqrt(5.0)) / 2.0,  # Golden ratio
    "inf": math.inf,
}


class CalculatorEvaluator(ast.NodeVisitor):
    """Safely evaluates an AST representation of a mathematical expression."""

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Union[int, float]:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    def visit_Num(self, node: Any) -> Union[int, float]:  # Backwards compatibility Python <= 3.7
        return node.n

    def visit_Name(self, node: ast.Name) -> float:
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        raise ValueError(f"Unknown mathematical symbol or variable: '{node.id}'")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_fn = SAFE_OPERATORS[op_type]
        try:
            return op_fn(left, right)
        except ZeroDivisionError:
            raise ZeroDivisionError("Division by zero")
        except OverflowError:
            raise OverflowError("Arithmetic overflow: numerical result too large")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_type = type(node.op)
        if op_type not in SAFE_UNARY_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = self.visit(node.operand)
        return SAFE_UNARY_OPERATORS[op_type](operand)

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct mathematical functions are supported")
        fn_name = node.func.id
        if fn_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Unsupported function: '{fn_name}'")
        fn = SAFE_FUNCTIONS[fn_name]
        args = [self.visit(arg) for arg in node.args]
        try:
            return fn(*args)
        except (ValueError, TypeError, OverflowError) as e:
            raise ValueError(f"Math error in {fn_name}(): {e}")

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Unsupported syntax expression element: {type(node).__name__}")


def clean_math_expression(raw_expr: str) -> str:
    """Cleans tool envelopes and prepares raw mathematical expression."""
    expr = raw_expr.strip()

    # Strip tool wrapper envelopes if passed by model
    # e.g. <act> calc(15 + 27) </act> -> 15 + 27
    expr = re.sub(r"^<act>\s*", "", expr)
    expr = re.sub(r"\s*</act>$", "", expr)
    expr = re.sub(r"^(?:calc|calculate|calculator)\s*\(\s*", "", expr, flags=re.IGNORECASE)
    expr = re.sub(r"^(?:calc|calculate|calculator)\s*:\s*", "", expr, flags=re.IGNORECASE)
    if expr.endswith(")") and "(" not in expr:
        expr = expr[:-1].strip()

    # Strip trailing equals sign or answer suffix (e.g. "15 + 27 = " or "15 + 27 = 42")
    if "=" in expr:
        expr = expr.split("=")[0].strip()

    # Handle percentage idioms: "25% of 800" -> "(25/100) * 800"
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"(\1 / 100.0) * ", expr, flags=re.IGNORECASE)
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1 / 100.0)", expr)

    # Convert natural language exponentiation '^' to Python '**' for proper precedence
    expr = re.sub(r"\^", "**", expr)

    # Standardize comma decimals if European format used without spaces (e.g. 3,14 -> 3.14)
    # Only if comma is between digits and not followed by a space
    expr = re.sub(r"(?<=\d),(?=\d{1,2}(?!\d))", ".", expr)
    # Remove thousand-separator commas (e.g. 1,000,000 -> 1000000)
    expr = re.sub(r"(?<=\d),(?=\d{3})", "", expr)

    return expr.strip()


def calculate(raw_expression: str) -> str:
    """
    Safely calculates mathematical expression and returns formatted string result.
    Safe for autonomous agent execution.
    """
    cleaned = clean_math_expression(raw_expression)
    if not cleaned:
        return "[Calculator Error: Empty expression]"

    try:
        parsed_ast = ast.parse(cleaned, mode="eval")
        evaluator = CalculatorEvaluator()
        result = evaluator.visit(parsed_ast)

        # Clean numerical presentation
        if isinstance(result, bool):
            return str(result)
        if isinstance(result, int):
            return str(result)
        if isinstance(result, float):
            if math.isinf(result):
                return "Infinity" if result > 0 else "-Infinity"
            if math.isnan(result):
                return "NaN"
            # If float is effectively an integer, return clean integer
            if result.is_integer() and abs(result) < 1e15:
                return str(int(result))
            # Round off tiny floating point artifacts (e.g. 0.30000000000000004 -> 0.3)
            rounded = round(result, 10)
            if rounded.is_integer() and abs(rounded) < 1e15:
                return str(int(rounded))
            # Strip trailing zeros
            res_str = f"{rounded:.10f}".rstrip("0").rstrip(".")
            return res_str
        return str(result)

    except ZeroDivisionError:
        return "[Calculator Error: Division by zero]"
    except OverflowError:
        return "[Calculator Error: Number out of range / overflow]"
    except (ValueError, SyntaxError) as e:
        return f"[Calculator Error: {e}]"
    except Exception as e:
        return f"[Calculator Error: {e}]"


# Alias for model calling
calc = calculate


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(calculate(query))
    else:
        print("🧮 CerebroSpheroid V5: Interactive Sandboxed Calculator")
        print("Type mathematical expressions or 'exit' to quit.\n")
        while True:
            try:
                line = input("calc > ").strip()
                if not line or line.lower() in ("exit", "quit", "q"):
                    break
                print(f"= {calculate(line)}")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
