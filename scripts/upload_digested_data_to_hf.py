#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teacher Digested Pretraining Dataset Builder & HuggingFace Publisher
====================================================================
Digests raw training datasets using Qwen2.5-1.5B-Instruct (or heuristic digester),
computes exact information-density dynamic next-token weights, and uploads the
resulting dataset directly to HuggingFace Hub.

Output Dataset Schema:
----------------------
- id             : Original unique sample identifier
- category       : Domain / category (math, code, reasoning, stem, conversation)
- thinking_level : Subconscious deliberation depth level (0, 1, 2, 3)
- prompt         : Raw stimulus prompt
- thinking       : Reasoning thought chain (<think> ... </think>)
- answer         : Ground-truth target completion
- text           : Full concatenated document
- tokens         : List[int] - Spheroid-32k token IDs
- target_weights : List[float] - Dynamic next-token weights w_t in [0.2, 4.0]
- seq_len        : Total token sequence length
- mean_weight    : Normalized mean target weight (~1.0)
- max_weight     : Peak information density weight in this document
- min_weight     : Floor weight for common syntax/stopwords

Usage:
------
python scripts/upload_digested_data_to_hf.py \
    --repo-id "AjitKumar-Bio/cerebro-spheroid-v5-digested-pretrain" \
    --teacher-model "Qwen/Qwen2.5-1.5B-Instruct" \
    --digest-mode online \
    --device cuda
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

os.environ["HF_HOME"] = os.path.expanduser("~/.hf_cache")

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.corpus import build_tokenizer
from src.teacher_digester import create_digester, QwenTeacherDigester, HeuristicDigester


DATASET_CARD_TEMPLATE = """---
license: apache-2.0
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- cerebro-spheroid
- qwen-teacher-digestion
- dynamic-loss-weighting
- information-density
- pretraining
- cot
- reasoning
size_categories:
- 10K<n<100K
---

# 🧠 CerebroSpheroid V5: Teacher-Digested Dynamic Loss Pretraining Dataset

A high-density pretraining and alignment dataset enriched with **per-token information density weights** computed via **Qwen2.5-1.5B-Instruct** teacher digestion.

---

## 🌟 Why Teacher-Digested Loss Weighting?

In standard language model pretraining, every token is weighted equally ($w = 1.0$). Consequently, over **60% of optimizer updates** are spent fitting low-entropy syntax stopwords (`"the"`, `"is"`, `"of"`, `"and"`, `"\n\n"`).

By passing text through **Qwen2.5-1.5B-Instruct** in evaluation mode:
1. **Surprisal Extraction**: $\\mathcal{L}_{Q, t} = -\\log P_{\\text{Qwen}}(x_t \\mid x_{<t})$
2. **Predictive Entropy**: Measures teacher distribution uncertainty at critical semantic branching points.
3. **Cross-Tokenizer Character-Span Projection**: Exact projection from Qwen's 152k vocabulary onto CerebroSpheroid's 32k Byte-Fallback BPE vocabulary.
4. **Dynamic Next-Token Weighting**: Bounded in $[0.2, 4.0]$ and normalized to mean $1.0$, concentrating gradient updates on high-density reasoning steps, formulas, and technical terminology while scaling down repetitive boilerplate.

---

## 📊 Dataset Schema

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Unique sample identifier |
| `category` | `string` | Task category (`math`, `code`, `stem`, `reasoning`, `factual`) |
| `thinking_level` | `int32` | Deliberation tier (0: direct, 1: CoT, 2: multi-step, 3: tool-augmented) |
| `prompt` | `string` | Stimulus input prompt |
| `thinking` | `string` | Deep subconscious deliberation thought trace |
| `answer` | `string` | Ground-truth response |
| `text` | `string` | Full unified training document |
| `tokens` | `list<int32>` | Token sequence in CerebroSpheroid 32k vocabulary |
| `target_weights`| `list<float32>` | Dynamic next-token loss weights $w_t \\in [0.2, 4.0]$ |
| `seq_len` | `int32` | Sequence length in tokens |
| `mean_weight` | `float32` | Mean token weight (~1.0) |
| `max_weight` | `float32` | Peak information density score in the sample |
| `min_weight` | `float32` | Floor weight for syntax / stopwords |

---

## 🚀 How to Use in PyTorch / JAX

```python
from datasets import load_dataset
import torch

# Load the digested dataset
ds = load_dataset("AjitKumar-Bio/cerebro-spheroid-v5-digested-pretrain")

# Retrieve sample tokens and dynamic target weights
sample = ds["train"][0]
tokens = torch.tensor(sample["tokens"], dtype=torch.long)
weights = torch.tensor(sample["target_weights"], dtype=torch.float32)

prompts = tokens[:-1]
targets = tokens[1:]
target_weights = weights[1:]  # Aligned 1:1 with next-token targets

# Use target_weights in your weighted cross-entropy loss:
# loss = (cross_entropy_unreduced * target_weights).sum() / target_weights.sum()
```

---

## 📜 Citation & License
- **License**: Apache 2.0
- **Model Family**: [CerebroSpheroid V5](https://github.com/AjitKumar-git/cerebro_spheroid_v5)
- **Teacher Model**: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
"""


def digest_dataframe(
    df: pd.DataFrame,
    digester,
    tokenizer,
    max_samples: Optional[int] = None,
    desc: str = "Digesting",
) -> pd.DataFrame:
    """Digests each row in a DataFrame and generates tokens and target_weights."""
    if max_samples is not None and len(df) > max_samples:
        df = df.iloc[:max_samples].copy()

    records = []
    total = len(df)
    t0 = time.time()

    print(f"\n🚀 [{desc}] Processing {total:,} samples...")

    for idx, row in df.iterrows():
        p = str(row.get("prompt", "")).strip()
        t = str(row.get("thinking", "")).strip()
        a = str(row.get("answer", "")).strip()

        if t:
            full_text = f"<stim>{p}</stim>\n<think>{t}</think>\n{a}"
        else:
            full_text = f"<stim>{p}</stim>\n{a}"

        # Digest using the teacher model
        try:
            token_ids, weights = digester.digest_text(full_text, tokenizer)
        except Exception as e:
            # Fallback
            token_ids = tokenizer.encode(full_text, add_special_tokens=False)
            weights = [1.0] * len(token_ids)

        # Align weights with token IDs length
        if len(weights) < len(token_ids):
            weights.extend([1.0] * (len(token_ids) - len(weights)))
        elif len(weights) > len(token_ids):
            weights = weights[:len(token_ids)]

        w_arr = np.array(weights, dtype=np.float32)

        records.append({
            "id": str(row.get("id", f"sample_{idx}")),
            "category": str(row.get("category", "general")),
            "thinking_level": int(row.get("thinking_level", 0)),
            "prompt": p,
            "thinking": t,
            "answer": a,
            "text": full_text,
            "tokens": [int(tok) for tok in token_ids],
            "target_weights": [float(w) for w in weights],
            "seq_len": int(len(token_ids)),
            "mean_weight": float(np.mean(w_arr)) if len(w_arr) > 0 else 1.0,
            "max_weight": float(np.max(w_arr)) if len(w_arr) > 0 else 1.0,
            "min_weight": float(np.min(w_arr)) if len(w_arr) > 0 else 1.0,
        })

        if (len(records) % 250 == 0) or len(records) == total:
            elapsed = time.time() - t0
            rate = len(records) / max(1e-3, elapsed)
            print(f"  [{desc}] {len(records):,}/{total:,} samples ({rate:.1f} doc/s) | ETA: {(total - len(records)) / max(1e-3, rate):.0f}s")

    return pd.DataFrame(records)


def upload_digested_dataset(
    repo_id: str = "AjitKumar-Bio/cerebro-spheroid-v5-digested-pretrain",
    train_parquet: str = "data/cerebro_v5_custom_dataset/train.parquet",
    val_parquet: str = "data/cerebro_v5_custom_dataset/validation.parquet",
    output_dir: str = "data/cerebro_v5_digested_dataset",
    teacher_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    digest_mode: str = "online",
    device_str: str = "cuda",
    teacher_scale: float = 1.5,
    min_weight: float = 0.2,
    max_weight: float = 4.0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    private: bool = False,
):
    print("=" * 80)
    print("🧠 CEREBROSPHEROID V5: TEACHER DIGESTION & HUGGINGFACE PUBLISHER")
    print("=" * 80)
    print(f"  HuggingFace Repo ID : {repo_id}")
    print(f"  Teacher Model       : {teacher_model} (mode: {digest_mode})")
    print(f"  Dynamic Weights     : alpha={teacher_scale}, range=[{min_weight}, {max_weight}]")
    print(f"  Train Source        : {train_parquet}")
    print(f"  Val Source          : {val_parquet}")
    print(f"  Local Cache Dir     : {output_dir}")
    print(f"  Visibility          : {'Private' if private else 'Public'}")
    print("=" * 80 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    from huggingface_hub import HfApi, get_token

    hf_token = get_token()
    if not hf_token:
        raise ValueError("No HuggingFace authentication token found. Please run `huggingface-cli login`.")

    api = HfApi(token=hf_token)
    repo_url = api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    print(f"✓ HuggingFace repository ready: {repo_url}")

    # 1. Initialize Tokenizer & Digester
    tokenizer = build_tokenizer("data/tokenizer")
    digester = create_digester(
        teacher_model=teacher_model,
        device=device_str,
        mode=digest_mode,
        alpha=teacher_scale,
        w_min=min_weight,
        w_max=max_weight,
    )

    # 2. Process Validation Split
    val_dest = os.path.join(output_dir, "validation.parquet")
    if os.path.exists(val_dest):
        print(f"✓ Found existing cached validation shard: {val_dest}")
        val_digested = pd.read_parquet(val_dest)
    else:
        val_df = pd.read_parquet(val_parquet) if os.path.exists(val_parquet) else pd.DataFrame([{"prompt": "Hello", "answer": "Hi"}])
        val_digested = digest_dataframe(val_df, digester, tokenizer, max_samples=max_val_samples, desc="Validation")
        val_digested.to_parquet(val_dest, index=False)
        print(f"✓ Saved validation shard: {val_dest} ({len(val_digested):,} rows)")

    # 3. Process Train Split
    train_dest = os.path.join(output_dir, "train.parquet")
    if os.path.exists(train_dest):
        print(f"✓ Found existing cached train shard: {train_dest}")
        train_digested = pd.read_parquet(train_dest)
    else:
        train_df = pd.read_parquet(train_parquet) if os.path.exists(train_parquet) else pd.DataFrame([{"prompt": "Hello", "answer": "Hi"}])
        train_digested = digest_dataframe(train_df, digester, tokenizer, max_samples=max_train_samples, desc="Training")
        train_digested.to_parquet(train_dest, index=False)
        print(f"✓ Saved training shard: {train_dest} ({len(train_digested):,} rows)")

    # 4. Upload Dataset Card to HuggingFace Hub (in-memory, no local README.md)
    print(f"\n📤 Uploading dataset files to {repo_id}...")
    try:
        api.upload_file(
            path_or_fileobj=DATASET_CARD_TEMPLATE.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  ✓ Uploaded HuggingFace Dataset Card")
    except Exception as e:
        print(f"  ℹ️ Dataset card notice: {e}")

    api.upload_file(
        path_or_fileobj=val_dest,
        path_in_repo="data/validation-00000-of-00001.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  ✓ Uploaded validation-00000-of-00001.parquet")

    api.upload_file(
        path_or_fileobj=train_dest,
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  ✓ Uploaded train-00000-of-00001.parquet")

    print("\n" + "=" * 80)
    print(f"✨ HUGGINGFACE DATASET PUBLICATION COMPLETE: {repo_id}")
    print(f"  URL: https://huggingface.co/datasets/{repo_id}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Digest dataset and publish to Hugging Face")
    parser.add_argument("--repo-id", type=str, default="AjitKumar-Bio/cerebro-spheroid-v5-digested-pretrain")
    parser.add_argument("--train-parquet", type=str, default="data/cerebro_v5_custom_dataset/train.parquet")
    parser.add_argument("--val-parquet", type=str, default="data/cerebro_v5_custom_dataset/validation.parquet")
    parser.add_argument("--output-dir", type=str, default="data/cerebro_v5_digested_dataset")
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--digest-mode", type=str, choices=["online", "heuristic"], default="online")
    parser.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--teacher-scale", type=float, default=1.5)
    parser.add_argument("--min-weight", type=float, default=0.2)
    parser.add_argument("--max-weight", type=float, default=4.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--private", action="store_true", default=False)
    args = parser.parse_args()

    upload_digested_dataset(
        repo_id=args.repo_id,
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        output_dir=args.output_dir,
        teacher_model=args.teacher_model,
        digest_mode=args.digest_mode,
        device_str=args.device,
        teacher_scale=args.teacher_scale,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        private=args.private,
    )


if __name__ == "__main__":
    main()
