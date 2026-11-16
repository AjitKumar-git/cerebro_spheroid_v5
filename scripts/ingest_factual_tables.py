#!/usr/bin/env python3
"""
CLI Tool: Ingest, Query, and Compile Factual Knowledge for CerebroSpheroid V5
=============================================================================
Usage Examples:
  # 1. Initialize and view tables
  python3 scripts/ingest_factual_tables.py --init
  python3 scripts/ingest_factual_tables.py --list

  # 2. Run SQL queries on command
  python3 scripts/ingest_factual_tables.py --query "SELECT symbol, name, atomic_mass FROM periodic_table WHERE period=2"
  python3 scripts/ingest_factual_tables.py --query "SELECT country_name, capital, currency FROM countries_and_capitals WHERE continent='Europe' LIMIT 5"

  # 3. Ingest custom table from CSV or JSON on command
  python3 scripts/ingest_factual_tables.py --ingest my_custom_data.csv --table custom_knowledge --pk id

  # 4. Generate custom V5 training dataset on command
  python3 scripts/ingest_factual_tables.py --generate-dataset --samples 5000 --output data/cerebro_v5_custom_dataset/factual_generated_5k.parquet
"""

import os
import sys
import argparse
import json

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.factual_knowledge_engine import FactualKnowledgeEngine


def main():
    parser = argparse.ArgumentParser(description="CerebroSpheroid V5 Factual Table Ingestion & Dataset Compiler")
    parser.add_argument("--init", action="store_true", help="Initialize and seed standard factual knowledge tables")
    parser.add_argument("--list", action="store_true", help="List all registered tables and row counts")
    parser.add_argument("--query", type=str, help="Execute arbitrary SQL query against the factual database")
    parser.add_argument("--ingest", type=str, help="Path to CSV or JSON file to ingest into database")
    parser.add_argument("--table", type=str, help="Target table name for ingestion")
    parser.add_argument("--pk", type=str, default=None, help="Primary key column for ingested table")
    parser.add_argument("--if-exists", type=str, choices=["append", "replace"], default="append", help="Action if table exists")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate ready-to-train V5 dataset on command")
    parser.add_argument("--samples", type=int, default=5000, help="Number of training samples to synthesize")
    parser.add_argument("--output", type=str, default="data/cerebro_v5_custom_dataset/factual_generated_5k.jsonl", help="Output dataset path (.jsonl or .parquet)")
    parser.add_argument("--dump-csv", action="store_true", help="Dump all database tables to CSV format")
    parser.add_argument("--db-path", type=str, default=None, help="Custom SQLite database path")

    args = parser.parse_args()
    engine = FactualKnowledgeEngine(db_path=args.db_path)

    # 1. Initialize
    if args.init:
        print("🏛️ Initializing CerebroSpheroid V5 Factual Knowledge Tables...")
        stats = engine.list_tables()
        print(f"✓ Database location: {engine.db_path}")
        print(f"✓ Total active tables: {len(stats)}")
        for t, count in stats.items():
            print(f"   - {t:<30}: {count:,} rows")
        csv_dir = engine.dump_tables_to_csv()
        print(f"✓ CSV snapshot exported to: {csv_dir}")
        return

    # 2. List Tables
    if args.list:
        stats = engine.list_tables()
        print("\n" + "=" * 60)
        print(f"📋 CEREBROSPHEROID V5: ACTIVE FACTUAL KNOWLEDGE TABLES")
        print("=" * 60)
        print(f"{'Table Name':<35} | {'Row Count':<12}")
        print("-" * 60)
        total_rows = 0
        for t, count in stats.items():
            print(f"{t:<35} | {count:<12,}")
            total_rows += count
        print("-" * 60)
        print(f"{'TOTAL ROWS ACROSS ALL TABLES':<35} | {total_rows:<12,}")
        print("=" * 60 + "\n")
        return

    # 3. Query
    if args.query:
        print(f"\n🔍 Executing SQL: {args.query}")
        try:
            rows = engine.query(args.query)
            if not rows:
                print("Result: 0 rows returned.")
                return
            headers = list(rows[0].keys())
            print("\n" + "-" * 85)
            header_str = " | ".join([f"{h:<16}" for h in headers[:5]])
            print(header_str)
            print("-" * 85)
            for r in rows[:15]:
                vals = [str(r[h])[:16] for h in headers[:5]]
                print(" | ".join([f"{v:<16}" for v in vals]))
            if len(rows) > 15:
                print(f"... ({len(rows) - 15} additional rows truncated)")
            print("-" * 85)
            print(f"✓ Total matching records: {len(rows):,}\n")
        except Exception as e:
            print(f"❌ SQL Execution Error: {e}")
        return

    # 4. Ingest File
    if args.ingest:
        if not args.table:
            print("❌ Error: --table <name> is required when ingesting a file.")
            sys.exit(1)
        file_path = os.path.abspath(args.ingest)
        print(f"\n📥 Ingesting file: {file_path} into table '{args.table}'...")
        if file_path.endswith(".csv"):
            n = engine.ingest_from_csv(args.table, file_path, primary_key=args.pk, if_exists=args.if_exists)
        elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
            n = engine.ingest_from_json(args.table, file_path, primary_key=args.pk, if_exists=args.if_exists)
        else:
            print(f"❌ Unsupported format: {file_path} (must be .csv, .json, or .jsonl)")
            sys.exit(1)
        print(f"✅ Ingestion successful: {n:,} rows stored into '{args.table}'.")
        return

    # 5. Generate Dataset on Command
    if args.generate_dataset:
        print(f"\n⚙️ Generating V5 Training Dataset on Command...")
        print(f"   Samples: {args.samples:,} | Output: {args.output}")
        out = engine.export_dataset_on_command(
            output_path=args.output,
            num_samples=args.samples,
            thinking_levels=[0, 1, 3],
        )
        file_sz = os.path.getsize(out) / (1024 * 1024)
        print(f"✅ Dataset Generated Successfully!")
        print(f"   Output file: {out} ({file_sz:.2f} MB)")
        return

    # 6. Dump CSV
    if args.dump_csv:
        csv_dir = engine.dump_tables_to_csv()
        print(f"✅ All database tables dumped to CSV at: {csv_dir}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
