#!/usr/bin/env python3
"""
DeepCode Dataset Builder (The Stack v2 & StarCoder Ingestion Engine).
Streams high-quality educational source code (bigcode/the-stack-v2-train-smollm,
starcoderdata), applies the Discrete Evidence Engine (DEE) for Code AST validation,
syntax integrity, balanced brackets, and comment density filtering, and exports
compressed Parquet shards partitioned into SmolLM2 developmental stages.
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import urllib.request
from pathlib import Path
import pandas as pd

def fetch_hf_token_from_website() -> str:
    """Fetches HF authentication token dynamically from centralized website vault."""
    url = "https://bioflo.in/api/cerebro/sync.php?action=get_config&model=v5_small&device=backup_hf_token"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DeepCode-Dataset-Builder/1.0"})
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
    return {"elapsed": elapsed, "stdout": res.stdout}

def get_disk_status_gb(path: Path) -> tuple[float, float, float]:
    """Returns (total_gb, used_gb, free_gb) for the filesystem holding path."""
    stat = shutil.disk_usage(path)
    return stat.total / (1024 ** 3), stat.used / (1024 ** 3), stat.free / (1024 ** 3)

def has_substantive_code(text: str) -> bool:
    """Heuristic filter rejecting minified assets or non-code trivialities."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 5:
        return False
    # Reject single-line minified files
    if any(len(line) > 1500 for line in lines):
        return False
    code_keywords = ["def ", "class ", "import ", "fn ", "struct ", "impl ", "return", "for ", "while ", "let ", "const ", "var "]
    return any(kw in text for kw in code_keywords)

def main():
    parser = argparse.ArgumentParser(description="DeepCode Dataset Builder - Code & Logic Curation Engine")
    parser.add_argument("--dataset", type=str, default="bigcode/the-stack-v2-train-smollm", help="Source Hugging Face code dataset")
    parser.add_argument("--subset", type=str, default="default", help="Dataset subset or language config (e.g. Python, Rust, default)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max-documents", type=int, default=0, help="Maximum documents to ingest (0 = stream until disk threshold or end)")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for DEE batch processing")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/deepcode" if Path("/kaggle/working").exists() else "deepcode_output"), help="Output directory")
    parser.add_argument("--disk-limit-gb", type=float, default=18.5, help="Maximum used disk space in GB before graceful halt")
    parser.add_argument("--min-free-gb", type=float, default=1.5, help="Minimum free disk space in GB before graceful halt")
    parser.add_argument("--dee-binary", type=Path, default=Path("./dee"), help="Path to compiled DEE binary")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_dir / "temp_scratch"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_root = args.output_dir / "parquet_shards"
    stage_dirs = {
        "stage1_fundamental": parquet_root / "stage1_fundamental",
        "stage2_intermediate": parquet_root / "stage2_intermediate",
        "stage3_advanced": parquet_root / "stage3_advanced",
        "stage4_research": parquet_root / "stage4_research",
    }
    for sdir in stage_dirs.values():
        sdir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve DEE binary
    dee_bin = args.dee_binary
    if not dee_bin.exists():
        for p in [Path("/kaggle/input/deepweb-builder-tool/dee"), Path("./dee"), Path("./tools/dee/target/release/dee"), Path("/kaggle/working/dee")]:
            if p.exists():
                dee_bin = p
                break
        else:
            print("DEE binary not found locally. Attempting build...")
            subprocess.run(["cargo", "build", "--release", "--manifest-path", "tools/dee/Cargo.toml"], check=True)
            dee_bin = Path("tools/dee/target/release/dee")

    os.chmod(dee_bin, 0o755)
    print(f"DEE Code AST Engine Binary: {dee_bin.resolve()}")

    # 2. Authenticate Hugging Face
    hf_token = fetch_hf_token_from_website()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("Authenticated Hugging Face token dynamically from centralized vault.")

    # 3. Stream from Hugging Face
    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "pyarrow", "pandas"], check=True)
        from datasets import load_dataset

    print(f"Connecting to {args.dataset} ({args.subset}, split={args.split}) in streaming mode...")
    ds = load_dataset(args.dataset, name=args.subset if args.subset != "default" else None, split=args.split, streaming=True, token=hf_token or None)

    import csv
    doc_count = 0
    chunk_index = 0
    current_chunk = []
    
    stage_doc_totals = {k: 0 for k in stage_dirs.keys()}
    total_clean_docs = 0

    target_desc = f"{args.max_documents} docs" if args.max_documents > 0 else "Continuous until disk limit"
    print(f"Beginning DeepCode Ingestion: Target={target_desc}, Chunk Size={args.chunk_size}...")
    start_all = time.time()

    for item in ds:
        text = item.get("content", "") or item.get("text", "")
        if text and len(text.split()) >= 15 and has_substantive_code(text):
            current_chunk.append((item.get("id", f"code_{doc_count}"), text))
            doc_count += 1

            if len(current_chunk) >= args.chunk_size:
                chunk_index += 1
                chunk_csv = temp_dir / f"code_chunk_{chunk_index:05d}.csv"
                chunk_out_dir = temp_dir / f"out_chunk_{chunk_index:05d}"
                chunk_out_dir.mkdir(parents=True, exist_ok=True)

                with open(chunk_csv, "w", newline="", encoding="utf-8") as f:
                    wtr = csv.writer(f)
                    wtr.writerow(["id", "text"])
                    for cid, ctext in current_chunk:
                        wtr.writerow([cid, ctext])

                # Process chunk through multi-threaded DEE with Code AST verification
                run_dee_pipeline(dee_bin, chunk_csv, chunk_out_dir)
                
                # Convert each generated stage CSV to compressed Parquet shard immediately
                for stage_key, sdir in stage_dirs.items():
                    stage_csv = chunk_out_dir / f"{stage_key}.csv"
                    if stage_csv.exists() and stage_csv.stat().st_size > 0:
                        try:
                            df = pd.read_csv(stage_csv, header=None, names=["id", "text"], on_bad_lines="skip")
                            if not df.empty:
                                parquet_path = sdir / f"code_shard_{chunk_index:05d}.parquet"
                                df.to_parquet(parquet_path, compression="snappy", index=False)
                                stage_doc_totals[stage_key] += len(df)
                                total_clean_docs += len(df)
                        except Exception as e:
                            print(f"Warning reading {stage_csv.name}: {e}")

                shutil.rmtree(chunk_out_dir, ignore_errors=True)
                chunk_csv.unlink(missing_ok=True)
                current_chunk.clear()

                total_gb, used_gb, free_gb = get_disk_status_gb(args.output_dir)
                elapsed = time.time() - start_all
                print(f"DeepCode Chunk {chunk_index:04d} Done | Code Docs: {doc_count:,d} | Clean Kept: {total_clean_docs:,d} | Speed: {doc_count / max(elapsed, 0.001):.1f} docs/sec | Disk: {used_gb:.2f}GB used, {free_gb:.2f}GB free")

                if free_gb <= args.min_free_gb or used_gb >= args.disk_limit_gb:
                    print(f"\n[ALERT] Disk space safety limit reached! (Used: {used_gb:.2f}GB, Free: {free_gb:.2f}GB). Gracefully halting...")
                    break

        if args.max_documents > 0 and doc_count >= args.max_documents:
            print(f"\nTarget document limit ({args.max_documents:,d}) reached.")
            break

    # Process remaining documents
    if current_chunk:
        chunk_index += 1
        chunk_csv = temp_dir / f"code_chunk_{chunk_index:05d}.csv"
        chunk_out_dir = temp_dir / f"out_chunk_{chunk_index:05d}"
        chunk_out_dir.mkdir(parents=True, exist_ok=True)

        with open(chunk_csv, "w", newline="", encoding="utf-8") as f:
            wtr = csv.writer(f)
            wtr.writerow(["id", "text"])
            for cid, ctext in current_chunk:
                wtr.writerow([cid, ctext])

        run_dee_pipeline(dee_bin, chunk_csv, chunk_out_dir)

        for stage_key, sdir in stage_dirs.items():
            stage_csv = chunk_out_dir / f"{stage_key}.csv"
            if stage_csv.exists() and stage_csv.stat().st_size > 0:
                try:
                    df = pd.read_csv(stage_csv, header=None, names=["id", "text"], on_bad_lines="skip")
                    if not df.empty:
                        parquet_path = sdir / f"code_shard_{chunk_index:05d}.parquet"
                        df.to_parquet(parquet_path, compression="snappy", index=False)
                        stage_doc_totals[stage_key] += len(df)
                        total_clean_docs += len(df)
                except Exception as e:
                    print(f"Warning reading {stage_csv.name}: {e}")

        shutil.rmtree(chunk_out_dir, ignore_errors=True)
        chunk_csv.unlink(missing_ok=True)

    shutil.rmtree(temp_dir, ignore_errors=True)

    total_time = time.time() - start_all
    print("\n" + "=" * 80)
    print("DEEPCODE DATASET GENERATION COMPLETE")
    print(f"Total Evaluated Code Documents:  {doc_count:,d}")
    print(f"Total Clean Kept:                {total_clean_docs:,d} ({total_clean_docs/max(doc_count,1)*100:.1f}%)")
    print(f"Total Execution Time:            {total_time:.2f} seconds ({doc_count / max(total_time, 0.001):.1f} docs/sec)")
    print(f"Parquet Output Directory:        {parquet_root.resolve()}")
    print("\nSmolLM2 Curriculum Code Distribution (Snappy Parquet Shards):")
    for stage_key, sdir in stage_dirs.items():
        shards = list(sdir.glob("*.parquet"))
        total_size_mb = sum(f.stat().st_size for f in shards) / (1024 * 1024)
        print(f"  - {stage_key:<24}: {len(shards):>4} shards | {stage_doc_totals[stage_key]:>8,d} docs | {total_size_mb:>7.2f} MB")
    print("=" * 80)

if __name__ == "__main__":
    main()
