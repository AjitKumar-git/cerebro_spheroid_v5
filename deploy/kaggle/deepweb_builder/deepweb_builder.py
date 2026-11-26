#!/usr/bin/env python3
"""
DeepWeb Dataset Builder (FineWeb-Edu Ingestion Engine).
Streams HuggingFaceFW/fineweb-edu, applies the Discrete Evidence Engine (DEE)
for epistemic, factual, temporal, and mathematical verification, and exports
the clean 'DeepWeb' dataset partitioned into SmolLM2 developmental stages.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import urllib.request
from pathlib import Path

def fetch_hf_token_from_website() -> str:
    """Fetches HF authentication token dynamically from centralized website vault."""
    url = "https://bioflo.in/api/cerebro/sync.php?action=get_config&model=v5_small&device=backup_hf_token"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DeepWeb-Dataset-Builder/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            token = data.get("token") or data.get("config", {}).get("token") or data.get("value")
            if token and token.startswith("hf_"):
                return token.strip()
    except Exception as e:
        print(f"Warning: Failed fetching token from website endpoint: {e}")
    return os.environ.get("HF_TOKEN", "")

def run_dee_pipeline(dee_bin: Path, input_csv: Path, output_dir: Path) -> dict:
    """Executes multi-threaded DEE pipeline on a chunk CSV."""
    cmd = [str(dee_bin), "pipeline", str(input_csv), "-o", str(output_dir)]
    start = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    if res.returncode != 0:
        print(f"DEE Pipeline error: {res.stderr}")
        raise RuntimeError(f"DEE execution failed: {res.stderr}")
    print(res.stdout)
    return {"elapsed": elapsed, "stdout": res.stdout}

def main():
    parser = argparse.ArgumentParser(description="DeepWeb Dataset Builder - FineWeb-Edu Curation")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu", help="Source Hugging Face dataset")
    parser.add_argument("--subset", type=str, default="sample-10BT", help="Dataset subset or config name")
    parser.add_argument("--max-documents", type=int, default=50000, help="Maximum documents to ingest")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Chunk size for DEE batch processing")
    parser.add_argument("--output-dir", type=Path, default=Path("deepweb_output"), help="Output directory")
    parser.add_argument("--dee-binary", type=Path, default=Path("./dee"), help="Path to compiled DEE binary")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_dir / "temp_shards"
    temp_dir.mkdir(parents=True, exist_ok=True)
    staged_dir = args.output_dir / "staged_curriculum"
    staged_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve DEE binary
    dee_bin = args.dee_binary
    if not dee_bin.exists():
        if Path("/kaggle/input/deepweb-builder-tool/dee").exists():
            dee_bin = Path("/kaggle/input/deepweb-builder-tool/dee")
        elif Path("./tools/dee/target/release/dee").exists():
            dee_bin = Path("./tools/dee/target/release/dee")
        elif Path("/kaggle/working/dee").exists():
            dee_bin = Path("/kaggle/working/dee")
        else:
            print("DEE binary not found locally. Attempting build...")
            subprocess.run(["cargo", "build", "--release", "--manifest-path", "tools/dee/Cargo.toml"], check=True)
            dee_bin = Path("tools/dee/target/release/dee")

    os.chmod(dee_bin, 0o755)
    print(f"Using DEE Binary: {dee_bin.resolve()}")

    # 2. Authenticate Hugging Face
    hf_token = fetch_hf_token_from_website()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("Successfully authenticated Hugging Face token dynamically from website vault.")
    else:
        print("Proceeding with public unauthenticated streaming.")

    # 3. Stream from Hugging Face
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets and pyarrow...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "pyarrow", "pandas"], check=True)
        from datasets import load_dataset

    print(f"Connecting to {args.dataset} ({args.subset}) in streaming mode...")
    ds = load_dataset(args.dataset, name=args.subset, split="train", streaming=True, token=hf_token or None)

    import csv
    doc_count = 0
    chunk_index = 0
    current_chunk = []

    print(f"Ingesting up to {args.max_documents} documents in chunks of {args.chunk_size}...")
    start_all = time.time()

    for item in ds:
        text = item.get("text", "")
        if text and len(text.split()) >= 15:
            current_chunk.append((item.get("id", f"doc_{doc_count}"), text))
            doc_count += 1

            if len(current_chunk) >= args.chunk_size:
                chunk_index += 1
                chunk_file = temp_dir / f"chunk_{chunk_index:04d}.csv"
                with open(chunk_file, "w", newline="", encoding="utf-8") as f:
                    wtr = csv.writer(f)
                    wtr.writerow(["id", "text"])
                    for cid, ctext in current_chunk:
                        wtr.writerow([cid, ctext])
                
                print(f"\n--- Processing Chunk {chunk_index} ({len(current_chunk)} docs) through DEE ---")
                run_dee_pipeline(dee_bin, chunk_file, staged_dir)
                current_chunk.clear()

        if doc_count >= args.max_documents:
            break

    # Process remaining documents
    if current_chunk:
        chunk_index += 1
        chunk_file = temp_dir / f"chunk_{chunk_index:04d}.csv"
        with open(chunk_file, "w", newline="", encoding="utf-8") as f:
            wtr = csv.writer(f)
            wtr.writerow(["id", "text"])
            for cid, ctext in current_chunk:
                wtr.writerow([cid, ctext])
        print(f"\n--- Processing Final Chunk {chunk_index} ({len(current_chunk)} docs) through DEE ---")
        run_dee_pipeline(dee_bin, chunk_file, staged_dir)

    total_time = time.time() - start_all
    print("\n================================================================================")
    print("DEEPWEB DATASET GENERATION COMPLETE")
    print(f"Total Streamed Documents:  {doc_count}")
    print(f"Total Wall Clock Time:     {total_time:.2f} seconds ({doc_count / max(total_time, 0.001):.1f} docs/sec)")
    print(f"Output Directory:          {staged_dir.resolve()}")
    print("Partitioned SmolLM2 Curriculum Files:")
    for p in sorted(staged_dir.glob("*.csv")):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  - {p.name:<30}: {size_mb:.2f} MB")
    print("================================================================================")

if __name__ == "__main__":
    main()
