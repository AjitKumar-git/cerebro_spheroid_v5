#!/usr/bin/env python3
"""
⚡ CerebroSpheroid V5: Unified Tool Dispatcher & Agent Runtime
============================================================
Interprets and executes tool calls emitted by the model in Level 3 (<act> ... </act>).
Connects the model directly to:
  1. Sandboxed Mathematical Calculator (calc)
  2. World Facts Database Manager (db_lookup, db_search, db_query)
"""

import ast
import json
import re
from typing import Any, Dict, Optional, Tuple, Union

from src.tools.calculator import calculate
from src.tools.world_facts_manager import db_lookup, db_search, db_query, facts_db


def extract_tool_call(action_text: str) -> Tuple[Optional[str], Optional[Union[str, Tuple[Any, ...]]]]:
    """
    Parses an action string into (tool_name, arguments).
    Supported formats:
      - calc(15 + 27)
      - db_lookup("France", "capital")
      - db_search("Everest")
      - db_query("SELECT ...")
      - calculator: 15 + 27
    """
    text = action_text.strip()
    # Strip <act> and </act> tags if present
    text = re.sub(r"^<act>\s*", "", text)
    text = re.sub(r"\s*</act>$", "", text)
    text = text.strip()

    if not text:
        return None, None

    # 1. Functional format: tool_name(arg1, arg2, ...)
    match_fn = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$", text, re.DOTALL)
    if match_fn:
        tool_name = match_fn.group(1).lower()
        raw_args = match_fn.group(2).strip()

        # Parse arguments safely via AST
        if not raw_args:
            return tool_name, ()

        # Calculator special case: keep math expression as raw string
        if tool_name in ("calc", "calculate", "calculator"):
            return "calc", raw_args

        try:
            # Wrap in tuple syntax to parse positional arguments
            parsed_args = ast.literal_eval(f"({raw_args})")
            if not isinstance(parsed_args, tuple):
                parsed_args = (parsed_args,)
            return tool_name, parsed_args
        except Exception:
            # Fallback: simple comma split
            parts = [p.strip().strip("\"'") for p in raw_args.split(",")]
            return tool_name, tuple(parts)

    # 2. Key-value format: "calc: 15 + 27" or "db_lookup: France, capital" or "sql: SELECT ..."
    match_kv = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)$", text, re.DOTALL)
    if match_kv:
        tool_name = match_kv.group(1).lower()
        payload = match_kv.group(2).strip()
        if tool_name in ("calc", "calculate", "calculator", "sql", "db_query", "query_db"):
            return tool_name, payload
        if "," in payload:
            parts = [p.strip().strip("\"'") for p in payload.split(",")]
            return tool_name, tuple(parts)
        return tool_name, payload

    # 3. Direct expression heuristic (pure arithmetic without tool name)
    if any(c in text for c in "+-*/^%") and re.search(r"\d", text):
        return "calc", text

    return "unknown", text


def dispatch_tool(tool_name: str, args: Any) -> str:
    """Executes the tool with the given arguments and returns string observation."""
    name = tool_name.lower().strip()

    # 1. Calculator
    if name in ("calc", "calculate", "calculator", "math"):
        expr = args if isinstance(args, str) else (args[0] if isinstance(args, tuple) and args else "")
        return calculate(str(expr))

    # 2. Database Lookup
    elif name in ("db_lookup", "lookup_fact", "fact_lookup", "lookup"):
        if isinstance(args, tuple) and len(args) >= 2:
            entity, attr = str(args[0]), str(args[1])
            return db_lookup(entity, attr)
        elif isinstance(args, tuple) and len(args) == 1:
            return db_search(str(args[0]))
        elif isinstance(args, str):
            return db_search(args)
        return "[Tool Error: db_lookup requires (entity, attribute)]"

    # 3. Database Keyword Search
    elif name in ("db_search", "search_facts", "search"):
        query = args if isinstance(args, str) else (args[0] if isinstance(args, tuple) and args else "")
        return db_search(str(query))

    # 4. Database SQL Query
    elif name in ("db_query", "query_db", "sql"):
        query = args if isinstance(args, str) else (args[0] if isinstance(args, tuple) and args else "")
        return db_query(str(query))

    return f"[Tool Error: Unknown tool '{tool_name}']"


def execute_action(action_str: str) -> str:
    """End-to-end execution of a model <act> action payload."""
    tool_name, args = extract_tool_call(action_str)
    if not tool_name:
        return "[Empty Action]"
    return dispatch_tool(tool_name, args)
