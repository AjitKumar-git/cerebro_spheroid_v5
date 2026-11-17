#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cosmopedia v2 Sharded Teacher Digestion Pipeline & HuggingFace Hub Publisher
=============================================================================
Streams and digests the complete Cosmopedia v2 dataset (39.13M examples across 104 shards)
using Qwen2.5-1.5B-Instruct as a teacher digester to compute exact dynamic next-token
information-density weights mapped to Spheroid-32k tokens.

Key Capabilities:
-----------------
1. Shard-by-Shard Resumability:
   - Processes each of the 104 shards independently.
   - Automatically checks target Hugging Face Hub repository; skips already completed shards.
2. Distributed Parallel Workers:
   - Supports `--shard-start` / `--shard-end` or `--worker-id` / `--num-workers` so multiple
     GPUs, Colab instances, or cloud VMs can digest shards in parallel without overlapping.
3. Low-Footprint Streaming:
   - Downloads 1 raw shard at a time (~1.17 GB).
   - Writes digested batches using pyarrow streaming without storing full shards in RAM.
   - Immediately uploads completed shard to Hugging Face Hub and flushes local temporary caches.
4. Cross-Tokenizer Projection:
   - Losslessly projects Qwen2.5 (152k vocab) predictive surprisals & entropies onto
     CerebroSpheroid (32k vocab) tokens via exact character-span offsets.

Usage Examples:
---------------
# 1. Digest Shard 0 and push to Hugging Face:
python scripts/digest_cosmopedia_v2.py --shard-indices 0 --device cuda --push-to-hub

# 2. Run Shards 0 through 5 on Colab GPU:
python scripts/digest_cosmopedia_v2.py --shard-start 0 --shard-end 5 --device cuda --push-to-hub

# 3. Distributed Worker 0 of 4 (processes shards 0, 4, 8, 12, ...):
python scripts/digest_cosmopedia_v2.py --worker-id 0 --num-workers 4 --device cuda --push-to-hub
"""

import os
import sys
import time
import argparse
import shutil
from typing import List, Optional, Set, Dict, Any
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

os.environ["HF_HOME"] = os.path.expanduser("~/.hf_cache")

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.corpus import build_tokenizer
from src.teacher_digester import create_digester, QwenTeacherDigester, HeuristicDigester


COSMOPEDIA_DATASET_CARD_TEMPLATE = r"""---
license: odc-by
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- cosmopedia-v2
- cerebro-spheroid
- teacher-digestion
- qwen2.5-1.5b
- dynamic-loss-weighting
- information-density
- pretraining
size_categories:
- 10M<n<100M
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/train-*.parquet"
---

# Cosmopedia v2 Teacher-Digested Pretraining Corpus

This dataset is a teacher-digested transformation of [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (Cosmopedia v2 split, 39.13M documents, ~30 Billion tokens).
Each document is digested by **Qwen2.5-1.5B-Instruct** to compute dynamic per-token information-density loss weights, losslessly projected onto **CerebroSpheroid's 32,000 vocabulary**.

## Dataset Schema

Each row contains:
- `id` (*string*): Unique sample identifier (`cosmo2_{shard:05d}_{row:07d}`).
- `prompt` (*string*): Original stimulus/topic prompt that generated the document.
- `audience` (*string*): Target pedagogical audience (e.g. `college_students`, `general`).
- `format` (*string*): Pedagogical format (e.g. `textbook`, `webpage`, `educational_article`).
- `tokens` (*list<int32>*): CerebroSpheroid SentencePiece token IDs (vocab size 32,000).
- `target_weights` (*list<float16>*): Dynamic next-token loss weights $w_t \in [0.2, 4.0]$ ($E[w_t] = 1.0$).
- `seq_len` (*int32*): Number of tokens in the sequence.
- `mean_weight` (*float32*): Sequence average loss weight.
- `max_weight` (*float32*): Peak information-density weight in the document.
- `min_weight` (*float32*): Minimum floor weight in the document.

## Information-Density Mathematical Formulation

For each sequence with teacher causal language model $Q$:
$$\\mathcal{L}_{Q, t} = -\\log P_Q(x_t \\mid x_{<t})$$
$$\\mathcal{H}_{Q, t} = -\\sum_{v \\in \\mathcal{V}_Q} P_Q(v \\mid x_{<t}) \\log P_Q(v \\mid x_{<t})$$
$$D_t = \\mathcal{L}_{Q, t} + \\beta \\mathcal{H}_{Q, t}$$
$$z_t = \\frac{D_t - \\mu_D}{\\sigma_D}$$
$$w_t = \\text{clip}\\left(1 + \\alpha \\tanh(z_t), w_{\\min}, w_{\\max}\\right)$$

Normalized so that $\\frac{1}{T}\\sum_{t=1}^T w_t = 1.0$.

## Streaming Usage with PyTorch & CerebroSpheroid

```python
from datasets import load_dataset

# Stream directly with zero Qwen runtime overhead:
dataset = load_dataset(
    "AjitKumar-Bio/cosmopedia-v2-digested",
    split="train",
    streaming=True
)

for sample in dataset:
    tokens = sample["tokens"]
    weights = sample["target_weights"]
    # Feed directly into CerebroSpheroid pretraining loss
```
"""


def get_existing_hub_files(api, repo_id: str) -> Set[str]:
    """Retrieves existing file paths in the target HuggingFace repository."""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        return set(files)
    except Exception as e:
        print(f"  ℹ️ Target repository {repo_id} not yet indexed or new: {e}")
        return set()


def digest_single_shard(
    shard_idx: int,
    total_shards: int,
    source_repo: str,
    target_repo_id: str,
    output_dir: str,
    digester,
    spheroid_tokenizer,
    api,
    max_samples: Optional[int] = None,
    max_length: int = 1024,
    batch_flush_size: int = 1000,
    push_to_hub: bool = True,
    keep_local: bool = False,
    clean_download_cache: bool = True,
) -> Dict[str, Any]:
    """
    Downloads, digests, writes, and uploads a single Cosmopedia v2 shard.
    """
    from huggingface_hub import hf_hub_download

    shard_name = f"train-{shard_idx:05d}-of-{total_shards:05d}.parquet"
    source_filename = f"cosmopedia-v2/{shard_name}"
    target_repo_path = f"data/{shard_name}"
    local_output_path = os.path.join(output_dir, shard_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"📦 [SHARD {shard_idx + 1}/{total_shards}] Processing {source_filename}")
    print(f"{'=' * 80}")

    # 1. Download raw shard from source dataset
    t0_dl = time.time()
    print(f"  📥 Downloading raw source shard from {source_repo}...")
    try:
        raw_shard_path = hf_hub_download(
            repo_id=source_repo,
            filename=source_filename,
            repo_type="dataset",
            cache_dir=os.environ.get("HF_HOME"),
        )
        dl_time = time.time() - t0_dl
        file_size_mb = os.path.getsize(raw_shard_path) / (1024 * 1024)
        print(f"  ✓ Downloaded raw shard ({file_size_mb:.1f} MB in {dl_time:.1f}s)")
    except Exception as e:
        print(f"  ❌ Error downloading {source_filename}: {e}")
        return {"status": "error", "shard": shard_idx, "error": str(e)}

    # 2. Setup pyarrow streaming writer
    schema = pa.schema([
        ("id", pa.string()),
        ("prompt", pa.string()),
        ("audience", pa.string()),
        ("format", pa.string()),
        ("tokens", pa.list_(pa.int32())),
        ("target_weights", pa.list_(pa.float16())),
        ("seq_len", pa.int32()),
        ("mean_weight", pa.float32()),
        ("max_weight", pa.float32()),
        ("min_weight", pa.float32()),
    ])

    writer = pq.ParquetWriter(local_output_path, schema, compression="snappy")

    # 3. Read and Digest Records
    parquet_file = pq.ParquetFile(raw_shard_path)
    total_rows = parquet_file.metadata.num_rows
    if max_samples is not None:
        target_rows = min(total_rows, max_samples)
    else:
        target_rows = total_rows

    print(f"  🧠 Digesting {target_rows:,} samples (out of {total_rows:,} rows in shard)...")

    processed_count = 0
    total_tokens_digested = 0
    batch_buffer: List[Dict[str, Any]] = []
    t_start = time.time()
    last_log_time = t_start

    def flush_buffer():
        nonlocal batch_buffer
        if not batch_buffer:
            return
        batch_table = pa.Table.from_pylist(batch_buffer, schema=schema)
        writer.write_table(batch_table)
        batch_buffer = []

    # Stream batch by batch
    for batch in parquet_file.iter_batches(batch_size=500):
        b_dict = batch.to_pydict()
        prompts = b_dict.get("prompt", [])
        texts = b_dict.get("text", [])
        audiences = b_dict.get("audience", [])
        formats = b_dict.get("format", [])

        num_in_batch = len(texts)
        for i in range(num_in_batch):
            if processed_count >= target_rows:
                break

            doc_text = texts[i]
            if not doc_text or len(doc_text.strip()) == 0:
                continue

            # Teacher digestion
            tokens, weights = digester.digest_text(
                doc_text,
                spheroid_tokenizer,
                max_length=max_length,
            )

            if len(tokens) == 0:
                continue

            weights_arr = np.array(weights, dtype=np.float32)
            total_tokens_digested += len(tokens)

            record = {
                "id": f"cosmo2_{shard_idx:05d}_{processed_count:07d}",
                "prompt": str(prompts[i]) if i < len(prompts) and prompts[i] is not None else "",
                "audience": str(audiences[i]) if i < len(audiences) and audiences[i] is not None else "",
                "format": str(formats[i]) if i < len(formats) and formats[i] is not None else "",
                "tokens": [int(t) for t in tokens],
                "target_weights": [float(w) for w in weights_arr.astype(np.float16)],
                "seq_len": int(len(tokens)),
                "mean_weight": float(weights_arr.mean()),
                "max_weight": float(weights_arr.max()),
                "min_weight": float(weights_arr.min()),
            }
            batch_buffer.append(record)
            processed_count += 1

            if len(batch_buffer) >= batch_flush_size:
                flush_buffer()

            now = time.time()
            if now - last_log_time >= 10.0 or processed_count == target_rows:
                elapsed = now - t_start
                rate = processed_count / max(1e-3, elapsed)
                tok_rate = total_tokens_digested / max(1e-3, elapsed)
                eta = (target_rows - processed_count) / max(1e-3, rate)
                print(
                    f"  [{shard_name}] {processed_count:,}/{target_rows:,} samples "
                    f"({processed_count / target_rows * 100:.1f}%) | "
                    f"{rate:.1f} doc/s | {tok_rate:.0f} tok/s | ETA: {eta / 60:.1f}m"
                )
                last_log_time = now

        if processed_count >= target_rows:
            break

    # Flush remainder and close writer
    flush_buffer()
    writer.close()

    total_time = time.time() - t_start
    digested_size_mb = os.path.getsize(local_output_path) / (1024 * 1024)
    print(
        f"  ✓ Digestion complete: {processed_count:,} samples ({total_tokens_digested:,} tokens) "
        f"in {total_time / 60:.1f}m ({digested_size_mb:.1f} MB)"
    )

    # 4. Upload Shard to Hugging Face Hub
    if push_to_hub:
        print(f"  🌐 Uploading digested shard to HuggingFace Hub: {target_repo_id}/{target_repo_path}...")
        t_up = time.time()
        api.upload_file(
            path_or_fileobj=local_output_path,
            path_in_repo=target_repo_path,
            repo_id=target_repo_id,
            repo_type="dataset",
            commit_message=f"Upload digested Cosmopedia v2 shard {shard_idx + 1}/{total_shards} ({processed_count:,} docs)",
        )
        print(f"  ✓ Uploaded shard {shard_idx} to Hub in {time.time() - t_up:.1f}s")

    # 5. Local Space Cleanup
    if not keep_local and os.path.exists(local_output_path):
        os.remove(local_output_path)
        print(f"  🧹 Cleaned local digested parquet: {local_output_path}")

    if clean_download_cache and os.path.exists(raw_shard_path):
        try:
            os.remove(raw_shard_path)
            print(f"  🧹 Cleaned downloaded raw source shard: {raw_shard_path}")
        except Exception:
            pass

    return {
        "status": "success",
        "shard": shard_idx,
        "samples": processed_count,
        "tokens": total_tokens_digested,
        "duration_sec": total_time,
    }


def run_pipeline(
    repo_id: str = "AjitKumar-Bio/cosmopedia-v2-digested",
    source_repo: str = "HuggingFaceTB/smollm-corpus",
    total_shards: int = 104,
    shard_indices: Optional[List[int]] = None,
    shard_start: int = 0,
    shard_end: Optional[int] = None,
    worker_id: int = 0,
    num_workers: int = 1,
    max_samples_per_shard: Optional[int] = None,
    max_length: int = 1024,
    teacher_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    digest_mode: str = "online",
    device_str: str = "cuda",
    teacher_scale: float = 1.5,
    min_weight: float = 0.2,
    max_weight: float = 4.0,
    output_dir: str = "data/cosmopedia_v2_digested_shards",
    push_to_hub: bool = True,
    skip_existing: bool = True,
    keep_local: bool = False,
    private: bool = False,
):
    print("=" * 80)
    print("🧠 CEREBROSPHEROID V5: COSMOPEDIA V2 TEACHER DIGESTION PIPELINE")
    print("=" * 80)
    print(f"  Target HF Dataset   : {repo_id}")
    print(f"  Source Dataset      : {source_repo} (Cosmopedia v2)")
    print(f"  Total Shards        : {total_shards}")
    print(f"  Teacher Model       : {teacher_model} (mode: {digest_mode})")
    print(f"  Dynamic Loss Scale  : alpha={teacher_scale}, bounds=[{min_weight}, {max_weight}]")
    print(f"  Max Sequence Length : {max_length}")
    print(f"  Device              : {device_str}")
    print(f"  Worker Config       : Worker {worker_id + 1} of {num_workers}")
    print(f"  Push to Hub         : {push_to_hub}")
    print(f"  Skip Existing       : {skip_existing}")
    print("=" * 80 + "\n")

    from huggingface_hub import HfApi, get_token

    hf_token = get_token()
    if not hf_token:
        raise ValueError("No Hugging Face auth token found. Run `huggingface-cli login` first.")

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)

    # Ensure dataset card exists
    try:
        api.upload_file(
            path_or_fileobj=COSMOPEDIA_DATASET_CARD_TEMPLATE.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Initialize Cosmopedia v2 digested dataset card",
        )
    except Exception as e:
        print(f"  ℹ️ Dataset card notice: {e}")

    # Determine Shards to Process
    if shard_indices is not None and len(shard_indices) > 0:
        candidate_shards = sorted(shard_indices)
    else:
        end_idx = total_shards if shard_end is None else min(shard_end, total_shards)
        candidate_shards = list(range(shard_start, end_idx))

    # Partition across workers if num_workers > 1
    assigned_shards = [s for s in candidate_shards if s % num_workers == worker_id]
    print(f"📋 Total candidate shards: {len(candidate_shards)} | Assigned to this worker: {len(assigned_shards)}")
    print(f"   Shard list: {assigned_shards[:10]}{'...' if len(assigned_shards) > 10 else ''}\n")

    # Check already completed shards on Hub
    existing_files = get_existing_hub_files(api, repo_id) if skip_existing else set()

    shards_to_process = []
    for s in assigned_shards:
        hub_path = f"data/train-{s:05d}-of-{total_shards:05d}.parquet"
        if skip_existing and hub_path in existing_files:
            print(f"  ⏭️ Shard {s:05d} already exists on Hub ({hub_path}). Skipping.")
        else:
            shards_to_process.append(s)

    if not shards_to_process:
        print("✨ All assigned shards have already been digested and uploaded!")
        return

    print(f"🚀 Ready to process {len(shards_to_process)} shards.\n")

    # Initialize Tokenizer and Teacher Digester
    print("📖 Loading Spheroid tokenizer and Teacher Digester...")
    spheroid_tokenizer = build_tokenizer("data/tokenizer")
    digester = create_digester(
        teacher_model=teacher_model,
        device=device_str,
        mode=digest_mode,
        alpha=teacher_scale,
        w_min=min_weight,
        w_max=max_weight,
    )

    total_samples_all = 0
    total_tokens_all = 0
    t_pipeline_start = time.time()

    for idx, shard_id in enumerate(shards_to_process):
        print(f"\n[{idx + 1}/{len(shards_to_process)}] Starting Shard {shard_id}...")
        result = digest_single_shard(
            shard_idx=shard_id,
            total_shards=total_shards,
            source_repo=source_repo,
            target_repo_id=repo_id,
            output_dir=output_dir,
            digester=digester,
            spheroid_tokenizer=spheroid_tokenizer,
            api=api,
            max_samples=max_samples_per_shard,
            max_length=max_length,
            push_to_hub=push_to_hub,
            keep_local=keep_local,
            clean_download_cache=True,
        )

        if result.get("status") == "success":
            total_samples_all += result.get("samples", 0)
            total_tokens_all += result.get("tokens", 0)

    elapsed_all = time.time() - t_pipeline_start
    print("\n" + "=" * 80)
    print("✨ COSMOPEDIA V2 TEACHER DIGESTION PIPELINE RUN COMPLETE")
    print(f"  Total Processed Shards : {len(shards_to_process)}")
    print(f"  Total Samples Digested : {total_samples_all:,}")
    print(f"  Total Tokens Digested  : {total_tokens_all:,}")
    print(f"  Total Wall Clock Time  : {elapsed_all / 3600:.2f} hours")
    print(f"  Target HuggingFace Repo: https://huggingface.co/datasets/{repo_id}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Digest Cosmopedia v2 and publish to HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, default="AjitKumar-Bio/cosmopedia-v2-digested", help="Target HuggingFace repo ID")
    parser.add_argument("--source-repo", type=str, default="HuggingFaceTB/smollm-corpus", help="Source HuggingFace repo ID")
    parser.add_argument("--total-shards", type=int, default=104, help="Total number of shards in Cosmopedia v2")
    parser.add_argument("--shard-indices", type=str, default=None, help="Comma-separated shard indices (e.g. '0,1,2')")
    parser.add_argument("--shard-start", type=int, default=0, help="Start shard index")
    parser.add_argument("--shard-end", type=int, default=None, help="End shard index (exclusive)")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID for parallel sharding")
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of parallel workers")
    parser.add_argument("--max-samples-per-shard", type=int, default=None, help="Max samples per shard (for testing/smoke runs)")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum token sequence length")
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="HuggingFace teacher model ID")
    parser.add_argument("--digest-mode", type=str, default="online", choices=["online", "heuristic"], help="Teacher digestion mode")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device ('cuda' or 'cpu')")
    parser.add_argument("--teacher-scale", type=float, default=1.5, help="Teacher dynamic weight scale alpha")
    parser.add_argument("--min-weight", type=float, default=0.2, help="Floor token weight")
    parser.add_argument("--max-weight", type=float, default=4.0, help="Ceiling token weight")
    parser.add_argument("--output-dir", type=str, default="data/cosmopedia_v2_digested_shards", help="Local temporary shard directory")
    parser.add_argument("--no-push", action="store_true", help="Do not upload to HuggingFace Hub")
    parser.add_argument("--no-skip", action="store_true", help="Do not skip existing shards on Hub")
    parser.add_argument("--keep-local", action="store_true", help="Keep local parquet files after upload")
    parser.add_argument("--private", action="store_true", help="Make HuggingFace dataset private")

    args = parser.parse_args()

    parsed_shard_indices = None
    if args.shard_indices is not None and args.shard_indices.strip():
        parsed_shard_indices = [int(x.strip()) for x in args.shard_indices.split(",") if x.strip()]

    run_pipeline(
        repo_id=args.repo_id,
        source_repo=args.source_repo,
        total_shards=args.total_shards,
        shard_indices=parsed_shard_indices,
        shard_start=args.shard_start,
        shard_end=args.shard_end,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        max_samples_per_shard=args.max_samples_per_shard,
        max_length=args.max_length,
        teacher_model=args.teacher_model,
        digest_mode=args.digest_mode,
        device_str=args.device,
        teacher_scale=args.teacher_scale,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        output_dir=args.output_dir,
        push_to_hub=not args.no_push,
        skip_existing=not args.no_skip,
        keep_local=args.keep_local,
        private=args.private,
    )


if __name__ == "__main__":
    main()
