"""
Minimal Viable Project: Schema Validator & Deterministic Tool Executor
=====================================================================
Provides JSON schema validation, safe execution, and deterministic handling
for minimal tool interaction (calculator and key-value lookup).
"""

from __future__ import annotations

import ast
import json
import operator
from typing import Any, Dict, Optional, Tuple, Union


# Registered tool schemas
TOOL_SCHEMAS = {
    "calculator": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
        },
        "required": ["expression"],
    },
    "lookup": {
        "type": "object",
        "properties": {
            "table": {"type": "string", "enum": ["capitals", "elements", "ports"]},
            "key": {"type": "string", "description": "Key to query in table"},
        },
        "required": ["table", "key"],
    },
}

# Deterministic lookup knowledge tables
LOOKUP_TABLES = {
    "capitals": {
        "France": "Paris",
        "Japan": "Tokyo",
        "Germany": "Berlin",
        "Kenya": "Nairobi",
        "Canada": "Ottawa",
        "Brazil": "Brasilia",
        "Australia": "Canberra",
        "India": "New Delhi",
        "Egypt": "Cairo",
        "South Korea": "Seoul",
        "Italy": "Rome",
        "Spain": "Madrid",
        "United Kingdom": "London",
        "Argentina": "Buenos Aires",
        "Norway": "Oslo",
    },
    "elements": {
        "H": "Hydrogen",
        "He": "Helium",
        "Li": "Lithium",
        "C": "Carbon",
        "N": "Nitrogen",
        "O": "Oxygen",
        "Na": "Sodium",
        "Al": "Aluminum",
        "Si": "Silicon",
        "Fe": "Iron",
        "Cu": "Copper",
        "Au": "Gold",
        "Ag": "Silver",
        "Pt": "Platinum",
        "U": "Uranium",
    },
    "ports": {
        "http": 80,
        "https": 443,
        "ssh": 22,
        "ftp": 21,
        "dns": 53,
        "smtp": 25,
        "mysql": 3306,
        "postgres": 5432,
        "redis": 6379,
        "mongodb": 27017,
    },
}


class SchemaValidationError(Exception):
    """Raised when a tool call does not conform to the schema."""
    pass


def validate_tool_call(call_record: Union[str, Dict[str, Any]]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validates a tool call record against registered schemas.
    
    Returns:
        (is_valid, error_message_if_any, parsed_dict_if_valid)
    """
    if isinstance(call_record, str):
        try:
            record = json.loads(call_record)
        except Exception as e:
            return False, f"Invalid JSON string: {e}", None
    elif isinstance(call_record, dict):
        record = call_record
    else:
        return False, f"Expected dict or JSON string, got {type(call_record).__name__}", None

    if not isinstance(record, dict):
        return False, "Tool call root must be a JSON object", None

    if "tool" not in record:
        return False, "Missing required key 'tool'", None

    tool_name = record["tool"]
    if tool_name not in TOOL_SCHEMAS:
        return False, f"Unknown tool '{tool_name}'. Available: {list(TOOL_SCHEMAS.keys())}", None

    if "args" not in record or not isinstance(record["args"], dict):
        return False, "Missing or invalid 'args' (must be a dictionary)", None

    schema = TOOL_SCHEMAS[tool_name]
    args = record["args"]

    # Check required fields
    for req in schema.get("required", []):
        if req not in args:
            return False, f"Tool '{tool_name}' missing required argument '{req}'", None

    # Check properties
    for prop_name, prop_val in args.items():
        if prop_name not in schema["properties"]:
            return False, f"Unexpected argument '{prop_name}' for tool '{tool_name}'", None
        expected_type = schema["properties"][prop_name].get("type")
        if expected_type == "string" and not isinstance(prop_val, str):
            return False, f"Argument '{prop_name}' must be a string", None
        if "enum" in schema["properties"][prop_name]:
            allowed = schema["properties"][prop_name]["enum"]
            if prop_val not in allowed:
                return False, f"Argument '{prop_name}' must be one of {allowed}, got '{prop_val}'", None

    return True, None, record


def validate_tool_result(result_record: Union[str, Dict[str, Any]]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Validates that a tool execution result conforms to standard schema."""
    if isinstance(result_record, str):
        try:
            record = json.loads(result_record)
        except Exception as e:
            return False, f"Invalid JSON string: {e}", None
    elif isinstance(result_record, dict):
        record = result_record
    else:
        return False, f"Expected dict or JSON string, got {type(result_record).__name__}", None

    if not isinstance(record, dict) or "status" not in record:
        return False, "Result must be a JSON object with 'status' key", None

    if record["status"] not in ("ok", "error"):
        return False, f"Status must be 'ok' or 'error', got '{record['status']}'", None

    if record["status"] == "ok" and "result" not in record:
        return False, "Success status requires 'result' field", None

    if record["status"] == "error" and "message" not in record:
        return False, "Error status requires 'message' field", None

    return True, None, record


class SafeCalculator:
    """Safe AST-based arithmetic expression evaluator."""

    ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @classmethod
    def evaluate(cls, expression: str) -> Union[int, float]:
        clean_expr = expression.strip().replace("^", "**")
        if not clean_expr:
            raise ValueError("Empty expression")
        if len(clean_expr) > 200:
            raise ValueError("Expression too long")

        try:
            node = ast.parse(clean_expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Syntax error in expression: {e}")

        def _eval_node(n: ast.AST) -> Union[int, float]:
            if isinstance(n, ast.Expression):
                return _eval_node(n.body)
            elif isinstance(n, ast.Constant):
                if isinstance(n.value, (int, float)):
                    return n.value
                raise ValueError(f"Unsupported constant type: {type(n.value)}")
            elif isinstance(n, ast.UnaryOp):
                op_func = cls.ALLOWED_OPS.get(type(n.op))
                if op_func is None:
                    raise ValueError(f"Unsupported unary operator: {type(n.op)}")
                val = _eval_node(n.operand)
                return op_func(val)
            elif isinstance(n, ast.BinOp):
                op_func = cls.ALLOWED_OPS.get(type(n.op))
                if op_func is None:
                    raise ValueError(f"Unsupported binary operator: {type(n.op)}")
                left = _eval_node(n.left)
                right = _eval_node(n.right)
                if isinstance(n.op, ast.Pow) and right > 20:
                    raise ValueError("Exponent too large (max 20)")
                if isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                    raise ZeroDivisionError("Division by zero")
                return op_func(left, right)
            else:
                raise ValueError(f"Unsupported AST node: {type(n).__name__}")

        result = _eval_node(node)
        # Format clean integer if exact
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result


class DeterministicToolExecutor:
    """Executes tool calls deterministically with structured responses."""

    def __init__(self, custom_tables: Optional[Dict[str, Dict[str, Any]]] = None):
        self.tables = custom_tables or LOOKUP_TABLES

    def execute(self, call: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes a tool call. Returns structured result dict:
        {"status": "ok", "result": ...} or {"status": "error", "message": ...}
        """
        is_valid, err, parsed = validate_tool_call(call)
        if not is_valid:
            return {"status": "error", "message": f"Schema validation error: {err}"}

        tool_name = parsed["tool"]
        args = parsed["args"]

        if tool_name == "calculator":
            expr = args.get("expression", "")
            try:
                calc_val = SafeCalculator.evaluate(expr)
                return {"status": "ok", "result": calc_val}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        elif tool_name == "lookup":
            table_name = args.get("table", "")
            key = args.get("key", "")
            if table_name not in self.tables:
                return {"status": "error", "message": f"Table '{table_name}' does not exist"}
            table = self.tables[table_name]
            if key in table:
                return {"status": "ok", "result": table[key]}
            # Case-insensitive fallback
            for k, v in table.items():
                if k.lower() == key.lower():
                    return {"status": "ok", "result": v}
            return {"status": "error", "message": f"Key '{key}' not found in table '{table_name}'"}

        return {"status": "error", "message": f"Unhandled tool '{tool_name}'"}
