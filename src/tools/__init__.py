"""
🛠️ CerebroSpheroid V5: Tool & Database Suite
============================================
Exported tools for autonomous agent execution and environment grounding:
  - calculate, calc: Safe AST-based mathematical evaluator
  - WorldFactsManager, facts_db: Relational database manager for world knowledge
  - db_lookup, db_search, db_query: High-speed fact retrieval hooks
  - execute_action, dispatch_tool: Universal agent action dispatcher
"""

from src.tools.calculator import calculate, calc
from src.tools.world_facts_manager import (
    WorldFactsManager,
    facts_db,
    db_lookup,
    db_search,
    db_query,
)
from src.tools.tool_dispatcher import (
    execute_action,
    dispatch_tool,
    extract_tool_call,
)

__all__ = [
    "calculate",
    "calc",
    "WorldFactsManager",
    "facts_db",
    "db_lookup",
    "db_search",
    "db_query",
    "execute_action",
    "dispatch_tool",
    "extract_tool_call",
]
