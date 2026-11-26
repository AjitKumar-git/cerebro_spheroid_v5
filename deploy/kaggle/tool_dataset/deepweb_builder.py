#!/usr/bin/env python3
"""
DeepWeb Dataset Builder (FineWeb-Edu Ingestion & Parquet Sharding Engine).
Streams HuggingFaceFW/fineweb-edu, applies the Discrete Evidence Engine (DEE)
for epistemic, factual, temporal, and mathematical verification, and streams
compressed Parquet shards DIRECTLY to Hugging Face (AjitKumar-Bio/DeepWeb)
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
                "User-Agent": "DeepWeb-Dataset-Builder/1.0"
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
    parser = argparse.ArgumentParser(description="DeepWeb Dataset Builder - FineWeb-Edu Ingestion Engine")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu", help="Source Hugging Face dataset")
    parser.add_argument("--subset", type=str, default="sample-10BT", help="Dataset subset or config name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max-documents", type=int, default=0, help="Maximum documents to ingest (0 = stream until disk threshold or end)")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Chunk size for DEE batch processing")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/deepweb" if Path("/kaggle/working").exists() else "deepweb_output"), help="Output directory")
    parser.add_argument("--hf-repo", type=str, default="", help="Destination Hugging Face repository (e.g. AjitKumar-Bio/DeepWeb)")
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
                hf_repo_id = f"{username}/DeepWeb"
            hf_api.create_repo(repo_id=hf_repo_id, repo_type="dataset", exist_ok=True, private=False)
            print(f"Connected to Hugging Face: {username} -> Repo: https://huggingface.co/datasets/{hf_repo_id}")
        except Exception as e:
            print(f"Warning: Hugging Face setup issue: {e}")
            hf_api = None
    else:
        print("Proceeding without Hugging Face direct upload.")

    # 3. Stream from Hugging Face
    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "pyarrow", "pandas", "huggingface_hub"], check=True)
        from datasets import load_dataset

    print(f"Connecting to {args.dataset} ({args.subset}, split={args.split}) in streaming mode...")
    ds = load_dataset(args.dataset, name=args.subset, split=args.split, streaming=True, token=hf_token or None)

    import csv
    doc_count = 0
    chunk_index = 0
    current_chunk = []
    
    stage_doc_totals = {k: 0 for k in stage_dirs.keys()}
    total_clean_docs = 0
    uploaded_shards_count = 0

    target_desc = f"{args.max_documents} docs" if args.max_documents > 0 else "Continuous until disk limit"
    print(f"Beginning Ingestion: Target={target_desc}, Chunk Size={args.chunk_size}...")
    start_all = time.time()

    for item in ds:
        text = item.get("text", "") or item.get("content", "")
        if text and len(text.split()) >= 15:
            current_chunk.append((item.get("id", f"doc_{doc_count}"), text))
            doc_count += 1

            if len(current_chunk) >= args.chunk_size:
                chunk_index += 1
                chunk_csv = temp_dir / f"raw_chunk_{chunk_index:05d}.csv"
                chunk_out_dir = temp_dir / f"out_chunk_{chunk_index:05d}"
                chunk_out_dir.mkdir(parents=True, exist_ok=True)

                with open(chunk_csv, "w", newline="", encoding="utf-8") as f:
                    wtr = csv.writer(f)
                    wtr.writerow(["id", "text"])
                    for cid, ctext in current_chunk:
                        wtr.writerow([cid, ctext])

                # Process chunk through multi-threaded DEE
                run_dee_pipeline(dee_bin, chunk_csv, chunk_out_dir)
                
                # Convert each stage CSV to compressed Parquet and upload to HF directly
                for stage_key, sdir in stage_dirs.items():
                    stage_csv = chunk_out_dir / f"{stage_key}.csv"
                    if stage_csv.exists() and stage_csv.stat().st_size > 0:
                        try:
                            df = pd.read_csv(stage_csv, header=None, names=["id", "text"], on_bad_lines="skip")
                            if not df.empty:
                                parquet_path = sdir / f"shard_{chunk_index:05d}.parquet"
                                df.to_parquet(parquet_path, compression="snappy", index=False)
                                stage_doc_totals[stage_key] += len(df)
                                total_clean_docs += len(df)

                                # Direct upload to Hugging Face
                                if args.upload_to_hf and hf_api and hf_repo_id:
                                    repo_path = f"data/{stage_key}/shard_{chunk_index:05d}.parquet"
                                    hf_api.upload_file(
                                        path_or_fileobj=str(parquet_path),
                                        path_in_repo=repo_path,
                                        repo_id=hf_repo_id,
                                        repo_type="dataset",
                                        commit_message=f"Upload {stage_key} shard {chunk_index:05d} ({len(df)} docs)"
                                    )
                                    uploaded_shards_count += 1
                                    if args.delete_uploaded_local:
                                        parquet_path.unlink(missing_ok=True)
                        except Exception as e:
                            print(f"Warning processing {stage_csv.name}: {e}")

                shutil.rmtree(chunk_out_dir, ignore_errors=True)
                chunk_csv.unlink(missing_ok=True)
                current_chunk.clear()

                total_gb, used_gb, free_gb = get_disk_status_gb(args.output_dir)
                elapsed = time.time() - start_all
                upload_status = f" | HF Uploads: {uploaded_shards_count} shards" if hf_api else ""
                print(f"Chunk {chunk_index:04d} Done | Docs: {doc_count:,d} | Clean: {total_clean_docs:,d} | Speed: {doc_count / max(elapsed, 0.001):.1f} d/s{upload_status} | Disk: {used_gb:.2f}GB used, {free_gb:.2f}GB free")

                if free_gb <= args.min_free_gb or used_gb >= args.disk_limit_gb:
                    print(f"\n[ALERT] Disk space safety limit reached! (Used: {used_gb:.2f}GB, Free: {free_gb:.2f}GB). Gracefully halting...")
                    break

        if args.max_documents > 0 and doc_count >= args.max_documents:
            print(f"\nTarget document limit ({args.max_documents:,d}) reached.")
            break

    # Process final trailing chunk
    if current_chunk:
        chunk_index += 1
        chunk_csv = temp_dir / f"raw_chunk_{chunk_index:05d}.csv"
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
                        parquet_path = sdir / f"shard_{chunk_index:05d}.parquet"
                        df.to_parquet(parquet_path, compression="snappy", index=False)
                        stage_doc_totals[stage_key] += len(df)
                        total_clean_docs += len(df)

                        if args.upload_to_hf and hf_api and hf_repo_id:
                            repo_path = f"data/{stage_key}/shard_{chunk_index:05d}.parquet"
                            hf_api.upload_file(
                                path_or_fileobj=str(parquet_path),
                                path_in_repo=repo_path,
                                repo_id=hf_repo_id,
                                repo_type="dataset",
                                commit_message=f"Upload {stage_key} shard {chunk_index:05d} ({len(df)} docs)"
                            )
                            uploaded_shards_count += 1
                            if args.delete_uploaded_local:
                                parquet_path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"Warning processing {stage_csv.name}: {e}")

        shutil.rmtree(chunk_out_dir, ignore_errors=True)
        chunk_csv.unlink(missing_ok=True)

    shutil.rmtree(temp_dir, ignore_errors=True)

    total_time = time.time() - start_all
    print("\n" + "=" * 80)
    print("DEEPWEB PARQUET DIRECT HUGGING FACE STREAMING COMPLETE")
    print(f"Total Ingested Documents:  {doc_count:,d}")
    print(f"Total Clean Kept:          {total_clean_docs:,d} ({total_clean_docs/max(doc_count,1)*100:.1f}%)")
    print(f"Total Execution Time:      {total_time:.2f} seconds ({doc_count / max(total_time, 0.001):.1f} docs/sec)")
    if hf_api and hf_repo_id:
        print(f"Live Hugging Face Dataset: https://huggingface.co/datasets/{hf_repo_id}")
        print(f"Total Uploaded Shards:     {uploaded_shards_count} parquet shards")
    print("=" * 80)

if __name__ == "__main__":
    main()
