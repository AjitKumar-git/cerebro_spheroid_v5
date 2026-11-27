#!/usr/bin/env python3
"""
DeepSynth Dataset Builder (Cosmopedia-v2 Ingestion & Synthetic Textbook Sharding Engine).
Streams HuggingFaceTB/cosmopedia-v2, applies the Discrete Evidence Engine (DEE)
with Gopher/FineWeb RepetitionAuditor for epistemic, factual, and structural verification,
and streams compressed Parquet shards DIRECTLY to Hugging Face (AjitKumar-Bio/DeepSynth)
partitioned into the SmolLM2 4-stage developmental curriculum.
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
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token and env_token.strip().startswith("hf_"):
        return env_token.strip()

    sync_token = os.environ.get("BIOFLO_SYNC_TOKEN") or os.environ.get("CEREBRO_SYNC_TOKEN") or "CerebroSpheroid_Sync_Secure_2026"
    url = "https://bioflo.in/api/cerebro/sync.php?action=get_config&model=v5_small&device=backup_hf_token"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {sync_token}",
                "User-Agent": "DeepSynth-Dataset-Builder/1.0"
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tok = data.get("config", {}).get("dtype", "") or data.get("token") or data.get("value")
            if tok and tok.strip().startswith("hf_"):
                return tok.strip()
    except Exception as e:
        print(f"Warning: Website token fetch error: {e}")
    return ""

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

def main():
    parser = argparse.ArgumentParser(description="DeepSynth Dataset Builder - Cosmopedia-v2 Synthetic Textbook Ingestion")
    parser.add_argument("--dataset", type=str, default="HuggingFaceTB/cosmopedia-v2", help="Source Hugging Face dataset")
    parser.add_argument("--subset", type=str, default="cosmopedia-v2", help="Dataset subset or config name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max-documents", type=int, default=0, help="Maximum documents to ingest (0 = stream until disk threshold or end)")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for DEE batch processing")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/deepsynth" if Path("/kaggle/working").exists() else "deepsynth_output"), help="Output directory")
    parser.add_argument("--hf-repo", type=str, default="", help="Destination Hugging Face repository (e.g. AjitKumar-Bio/DeepSynth)")
    parser.add_argument("--upload-to-hf", action="store_true", default=True, help="Directly upload shards to Hugging Face")
    parser.add_argument("--delete-uploaded-local", action="store_true", default=True, help="Delete local Parquet after upload to preserve disk space")
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
            print("DEE binary not found. Compiling...")
            subprocess.run(["cargo", "build", "--release", "--manifest-path", "tools/dee/Cargo.toml"], check=True)
            dee_bin = Path("tools/dee/target/release/dee")

    os.chmod(dee_bin, 0o755)
    print(f"DEE Engine Binary: {dee_bin.resolve()}")

    # 2. Authenticate Hugging Face & Setup Target Repo
    hf_token = fetch_hf_token_from_website()
    hf_api = None
    hf_repo_id = args.hf_repo

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        try:
            from huggingface_hub import HfApi
            hf_api = HfApi(token=hf_token)
            user_info = hf_api.whoami()
            username = user_info.get("name", "AjitKumar-Bio")
            if not hf_repo_id:
                hf_repo_id = f"{username}/DeepSynth"
            hf_api.create_repo(repo_id=hf_repo_id, repo_type="dataset", exist_ok=True, private=False)
            print(f"Connected to Hugging Face: {username} -> Repo: https://huggingface.co/datasets/{hf_repo_id}")
        except Exception as e:
            print(f"Warning: Hugging Face setup issue: {e}")
            hf_api = None

    # 3. Stream from Hugging Face
    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "pyarrow", "pandas", "huggingface_hub"], check=True)
        from datasets import load_dataset

    print(f"Connecting to {args.dataset} ({args.subset}, split={args.split}) in streaming mode...")
    ds = load_dataset(args.dataset, name=args.subset if args.subset != "default" else None, split=args.split, streaming=True, token=hf_token or None)

    import csv
    doc_count = 0
    chunk_index = 0
    current_chunk = []
    
    stage_doc_totals = {k: 0 for k in stage_dirs.keys()}
    total_clean_docs = 0
    uploaded_shards_count = 0

    target_desc = f"{args.max_documents} docs" if args.max_documents > 0 else "Continuous until disk limit"
    print(f"Beginning DeepSynth Ingestion: Target={target_desc}, Chunk Size={args.chunk_size}...")
    start_all = time.time()

    for item in ds:
        text = item.get("text", "") or item.get("content", "")
        if text and len(text.split()) >= 30:
            current_chunk.append((item.get("id", f"synth_{doc_count}"), text))
            doc_count += 1

            if len(current_chunk) >= args.chunk_size:
                chunk_index += 1
                chunk_csv = temp_dir / f"synth_chunk_{chunk_index:05d}.csv"
                chunk_out_dir = temp_dir / f"out_chunk_{chunk_index:05d}"
                chunk_out_dir.mkdir(parents=True, exist_ok=True)

                with open(chunk_csv, "w", newline="", encoding="utf-8") as f:
                    wtr = csv.writer(f)
                    wtr.writerow(["id", "text"])
                    for cid, ctext in current_chunk:
                        wtr.writerow([cid, ctext])

                # Process chunk through multi-threaded DEE with RepetitionAuditor
                run_dee_pipeline(dee_bin, chunk_csv, chunk_out_dir)
                
                # Convert each stage CSV to compressed Parquet and upload to HF directly
                for stage_key, sdir in stage_dirs.items():
                    stage_csv = chunk_out_dir / f"{stage_key}.csv"
                    if stage_csv.exists() and stage_csv.stat().st_size > 0:
                        try:
                            df = pd.read_csv(stage_csv, header=None, names=["id", "text"], on_bad_lines="skip")
                            if not df.empty:
                                parquet_path = sdir / f"synth_shard_{chunk_index:05d}.parquet"
                                df.to_parquet(parquet_path, compression="snappy", index=False)
                                stage_doc_totals[stage_key] += len(df)
                                total_clean_docs += len(df)

                                if args.upload_to_hf and hf_api and hf_repo_id:
                                    repo_path = f"data/{stage_key}/synth_shard_{chunk_index:05d}.parquet"
                                    hf_api.upload_file(
                                        path_or_fileobj=str(parquet_path),
                                        path_in_repo=repo_path,
                                        repo_id=hf_repo_id,
                                        repo_type="dataset",
                                        commit_message=f"Add {stage_key} synthetic textbook shard {chunk_index:05d} ({len(df)} docs)"
                                    )
                                    uploaded_shards_count += 1
                                    
                                    if args.delete_uploaded_local:
                                        parquet_path.unlink(missing_ok=True)
                        except Exception as e:
                            print(f"Warning: Failed processing {stage_csv}: {e}")

                # Clean up scratch chunk files
                shutil.rmtree(chunk_out_dir, ignore_errors=True)
                chunk_csv.unlink(missing_ok=True)
                current_chunk = []

                # Report Progress & Disk Safety
                tot_gb, used_gb, free_gb = get_disk_status_gb(args.output_dir)
                elapsed = time.time() - start_all
                rate = doc_count / elapsed if elapsed > 0 else 0.0
                print(
                    f"Chunk {chunk_index:04d} Complete | Raw Docs: {doc_count:,} | Clean Kept: {total_clean_docs:,} "
                    f"({total_clean_docs/doc_count*100:.1f}%) | HF Shards Uploaded: {uploaded_shards_count} | "
                    f"Disk Used: {used_gb:.2f}GB / Free: {free_gb:.2f}GB | Rate: {rate:.1f} docs/sec"
                )

                if used_gb >= args.disk_limit_gb or free_gb <= args.min_free_gb:
                    print(f"Reached safety disk limit (Used: {used_gb:.2f}GB >= {args.disk_limit_gb}GB or Free: {free_gb:.2f}GB <= {args.min_free_gb}GB). Gracefully halting.")
                    break

                if args.max_documents > 0 and doc_count >= args.max_documents:
                    print(f"Reached target max documents ({args.max_documents}). Gracefully halting.")
                    break

    # Process remaining documents if any
    if current_chunk:
        chunk_index += 1
        chunk_csv = temp_dir / f"synth_chunk_{chunk_index:05d}.csv"
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
                        parquet_path = sdir / f"synth_shard_{chunk_index:05d}.parquet"
                        df.to_parquet(parquet_path, compression="snappy", index=False)
                        stage_doc_totals[stage_key] += len(df)
                        total_clean_docs += len(df)

                        if args.upload_to_hf and hf_api and hf_repo_id:
                            repo_path = f"data/{stage_key}/synth_shard_{chunk_index:05d}.parquet"
                            hf_api.upload_file(
                                path_or_fileobj=str(parquet_path),
                                path_in_repo=repo_path,
                                repo_id=hf_repo_id,
                                repo_type="dataset",
                                commit_message=f"Add {stage_key} synthetic textbook shard {chunk_index:05d} ({len(df)} docs)"
                            )
                            uploaded_shards_count += 1
                            if args.delete_uploaded_local:
                                parquet_path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"Warning: Failed processing {stage_csv}: {e}")

        shutil.rmtree(chunk_out_dir, ignore_errors=True)
        chunk_csv.unlink(missing_ok=True)

    # Clean up scratch directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Final Summary Report
    print("=" * 80)
    print("DEEPSYNTH INGESTION & CURRICULUM PARTITIONING SUMMARY")
    print("=" * 80)
    print(f"Total Raw Documents Ingested   : {doc_count:,}")
    print(f"Total Clean Documents Kept     : {total_clean_docs:,} ({total_clean_docs/doc_count*100:.1f}%)" if doc_count else "0")
    print(f"Total Shards Uploaded to HF    : {uploaded_shards_count}")
    print(f"Target Hugging Face Repository : https://huggingface.co/datasets/{hf_repo_id}")
    print("-" * 80)
    for stage, count in stage_doc_totals.items():
        pct = count / total_clean_docs * 100 if total_clean_docs else 0.0
        print(f"  * {stage:<24} : {count:>8,} documents ({pct:>5.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
