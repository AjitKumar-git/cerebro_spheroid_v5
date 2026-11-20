#!/usr/bin/env python3
"""
Unit tests for CerebroSpheroid V5 FactualKnowledgeEngine and relational tables.
Verifies all 17 tables, SQL querying, and 3-Tier Cognitive QA synthesis.
"""

import os
import sys
import tempfile
import pytest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.factual_knowledge_engine import FactualKnowledgeEngine

EXPECTED_TABLES = [
    "available_tools",
    "biochemical_amino_acids",
    "bioprocess_cell_lines",
    "countries_and_capitals",
    "http_status_codes",
    "irregular_verbs",
    "lexical_antonyms_synonyms",
    "linux_commands",
    "mathematical_constants",
    "mime_types",
    "periodic_table",
    "physical_constants",
    "planetary_astronomy",
    "prime_numbers",
    "si_unit_conversions",
    "standard_regex_patterns",
    "world_history_milestones",
]


@pytest.fixture
def engine():
    return FactualKnowledgeEngine()


def test_database_initialization(engine):
    """Verifies that all 17 standard tables exist and contain seeded data."""
    stats = engine.list_tables()
    for table in EXPECTED_TABLES:
        assert table in stats, f"Missing table: {table}"
        assert stats[table] > 0, f"Table {table} is empty!"
    assert len(stats) >= 17


def test_sql_queries_across_new_tables(engine):
    """Tests executing SQL queries on the new factual tables."""
    # 1. HTTP status codes
    http_res = engine.query("SELECT code, name FROM http_status_codes WHERE code=404;")
    assert len(http_res) == 1
    assert http_res[0]["name"] == "Not Found"

    # 2. Linux commands
    ls_res = engine.query("SELECT flag, description FROM linux_commands WHERE command='ls' AND flag='-l';")
    assert len(ls_res) == 1
    assert "long listing" in ls_res[0]["description"].lower()

    # 3. Prime numbers
    prime_res = engine.query("SELECT prime_value FROM prime_numbers WHERE n_index=1;")
    assert len(prime_res) == 1
    assert prime_res[0]["prime_value"] == 2

    prime_500 = engine.query("SELECT prime_value FROM prime_numbers WHERE n_index=500;")
    assert len(prime_500) == 1
    assert prime_500[0]["prime_value"] == 3571

    # 4. Bioprocess cell lines
    cho_res = engine.query("SELECT organism, doubling_time_hr FROM bioprocess_cell_lines WHERE cell_line='CHO-K1';")
    assert len(cho_res) == 1
    assert cho_res[0]["doubling_time_hr"] == 18.0

    # 5. Available tools
    sql_tool = engine.query("SELECT category, schema_json FROM available_tools WHERE tool_name='sql_query';")
    assert len(sql_tool) == 1
    assert sql_tool[0]["category"] == "Database"
    assert "function" in sql_tool[0]["schema_json"]


def test_qa_pair_synthesis_levels(engine):
    """Tests synthesizing training samples conforming to Levels 0, 1, and 3."""
    samples = engine.generate_qa_pairs(num_samples=60, thinking_levels=[0, 1, 3])
    assert len(samples) == 60

    # Check level 0
    l0 = [s for s in samples if s["thinking_level"] == 0]
    assert len(l0) > 0
    for s in l0:
        assert "<think>" not in s["formatted_output"]
        assert "<act>" not in s["formatted_output"]
        assert s["action"] is None

    # Check level 1
    l1 = [s for s in samples if s["thinking_level"] == 1]
    assert len(l1) > 0
    for s in l1:
        assert "<think>" in s["formatted_output"]
        assert "</think>" in s["formatted_output"]
        assert "<act>" not in s["formatted_output"]
        assert s["action"] is None

    # Check level 3 (agentic SQL tool)
    l3 = [s for s in samples if s["thinking_level"] == 3]
    assert len(l3) > 0
    for s in l3:
        assert "<think>" in s["formatted_output"]
        assert "</think>" in s["formatted_output"]
        assert "<tool_call>" in s["formatted_output"]
        assert "</tool_call>" in s["formatted_output"]
        assert "<act>" not in s["formatted_output"]
        assert s["action"] is not None
        assert "sql_query(" in s["action"]


def test_csv_export(engine):
    """Verifies that all tables export properly to CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = engine.dump_tables_to_csv(output_dir=tmpdir)
        csv_files = os.listdir(out)
        for table in EXPECTED_TABLES:
            assert f"{table}.csv" in csv_files
