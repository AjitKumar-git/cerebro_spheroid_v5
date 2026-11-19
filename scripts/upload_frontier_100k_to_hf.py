#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload CerebroSpheroid V5 Frontier 100k Reasoning & Tool Dataset to Hugging Face
================================================================================
Publishes the premier 100,000-prompt reasoning and tool-calling dataset to
Hugging Face Hub under `AjitKumar-Bio/cerebro-spheroid-v5-frontier-reasoning-tools-100k`.
"""

import os
import sys
import time
from huggingface_hub import HfApi

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hf_utils import get_hf_token

REPO_ID = "AjitKumar-Bio/cerebro-spheroid-v5-frontier-reasoning-tools-100k"
LOCAL_DIR = os.path.join(PROJECT_ROOT, "data", "frontier_100k")

DATASET_CARD = """---
license: apache-2.0
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- reasoning
- chain-of-thought
- deepseek-r1
- tool-calling
- function-calling
- agentic
- open-thoughts
- sky-t1
- numinamath
- cerebro-spheroid
- sft
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/train.parquet"
---

# CerebroSpheroid V5: Frontier 100,000 Reasoning & Industry Tool-Calling Dataset

This dataset contains **exactly 100,000 high-density, verified reasoning and tool-calling examples** synthesized from the world's premier open-source reasoning models and verified function-calling pipelines.

## 🎯 Dataset Composition (100,000 Prompts)

| Source | Domain | Count | Primary Focus |
| :--- | :--- | :--- | :--- |
| **`open-thoughts/OpenThoughts-114k`** | DeepSeek-R1 Distill | **57,495** | Deep deliberation scratchpad across STEM, Code, and Logic |
| **`NovaSky-AI/Sky-T1_data_17k`** | Olympiad Math & Proofs | **14,392** | Competition mathematics (AIME, Olympiad, TACO/APPs) |
| **`AI-MO/NuminaMath-CoT`** | Formal Derivations | **15,000** | Rigorous step-by-step mathematical proofs |
| **`argilla/apigen-function-calling` & `Hermes`** | Industry Tool Calling | **13,113** | Multi-turn verified JSON API function calling |
| **Total** | **Frontier Reasoning & Tools** | **100,000** | **100% Unique, Deduplicated Prompts** |

---

## 🧠 Dual-Parity Token Format

Every sample in this dataset supports **both** native cognitive scratchpads and industry-standard agentic tool use:

### 1. Spheroid Cognitive Tokens
*   **Thinking Scratchpad:** `<think> ... </think>` (internal deliberation and plan)
*   **External Tool Execution:** `<act> calc(...) </act>` or `<act> python(code='...') </act>`
*   **Environment Feedback:** `<obs> ... </obs>` (closed-loop tool observation)
*   **Final Answer:** Natural text strictly outside `<act>`.

### 2. Industry Standard Format (OpenAI / ShareGPT / Hermes)
*   Standard function calling schema:
    ```xml
    <tool_call>
    {"name": "calculator", "arguments": {"expression": "25 * 4"}}
    </tool_call>
    ```
*   Standard chat messages with `"role": "tool"`, `"tool_calls"`, and `"tool_call_id"`.

---

## 🚀 Quick Usage

### Python (`datasets`)

```python
from datasets import load_dataset

# Load ZSTD-compressed Parquet (fastest, zero-copy)
dataset = load_dataset("AjitKumar-Bio/cerebro-spheroid-v5-frontier-reasoning-tools-100k", split="train")

print("Total samples:", len(dataset))
sample = dataset[0]
print("Prompt:", sample["prompt"])
print("Response:", sample["response"])
```

### Inspect Available Files

- `data/train.parquet` (337 MB): Formatted in high-throughput columnar Apache Parquet.
- `data/frontier_100k_spheroid_format.jsonl` (1.17 GB): Native Spheroid format.
- `data/frontier_100k_industry_sharegpt.jsonl` (1.18 GB): ShareGPT / OpenAI chat format.

---

## 📜 License
Apache-2.0. Open for research, fine-tuning, and commercial deployment.
"""


def upload_dataset():
    print("=" * 80)
    print(f"🚀 Uploading Frontier 100k Dataset to Hugging Face: {REPO_ID}")
    print("=" * 80)

    token = get_hf_token()
    if not token:
        raise RuntimeError("No Hugging Face token found. Set HF_TOKEN in ~/.env or environment.")

    api = HfApi(token=token)
    user_info = api.whoami()
    print(f"✓ Authenticated as: {user_info.get('name')}")

    # 1. Ensure Repository Exists
    print(f"\n📦 Ensuring repository '{REPO_ID}' exists on Hugging Face...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=False
    )
    print(f"✓ Repository ready: https://huggingface.co/datasets/{REPO_ID}")

    # 2. Upload README.md (Dataset Card)
    print("\n📝 Uploading dataset card (README.md)...")
    readme_path = os.path.join(LOCAL_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(DATASET_CARD.strip() + "\n")

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="docs: add rich dataset card and architecture documentation"
    )
    print("✓ Uploaded README.md")

    # 3. Upload Primary Train Parquet (named data/train.parquet for automatic HF Viewer)
    local_parquet = os.path.join(LOCAL_DIR, "frontier_100k_reasoning_tool_sft.parquet")
    if os.path.exists(local_parquet):
        size_mb = os.path.getsize(local_parquet) / 1e6
        print(f"\n📦 Uploading primary train parquet ({size_mb:.1f} MB) -> data/train.parquet...")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=local_parquet,
            path_in_repo="data/train.parquet",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="feat: upload 100k verified reasoning & tool calling train.parquet"
        )
        print(f"✓ Uploaded data/train.parquet in {time.time() - t0:.1f}s")

    # 4. Upload Spheroid JSONL
    local_sph = os.path.join(LOCAL_DIR, "frontier_100k_spheroid_format.jsonl")
    if os.path.exists(local_sph):
        size_mb = os.path.getsize(local_sph) / 1e6
        print(f"\n📦 Uploading native Spheroid format JSONL ({size_mb:.1f} MB)...")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=local_sph,
            path_in_repo="data/frontier_100k_spheroid_format.jsonl",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="feat: upload frontier 100k spheroid cognitive format"
        )
        print(f"✓ Uploaded frontier_100k_spheroid_format.jsonl in {time.time() - t0:.1f}s")

    # 5. Upload Industry ShareGPT JSONL
    local_ind = os.path.join(LOCAL_DIR, "frontier_100k_industry_sharegpt.jsonl")
    if os.path.exists(local_ind):
        size_mb = os.path.getsize(local_ind) / 1e6
        print(f"\n📦 Uploading industry ShareGPT format JSONL ({size_mb:.1f} MB)...")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=local_ind,
            path_in_repo="data/frontier_100k_industry_sharegpt.jsonl",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="feat: upload frontier 100k industry sharegpt messages format"
        )
        print(f"✓ Uploaded frontier_100k_industry_sharegpt.jsonl in {time.time() - t0:.1f}s")

    print("\n" + "=" * 80)
    print(f"🎉 Dataset Successfully Published to Hugging Face!")
    print(f"🔗 URL: https://huggingface.co/datasets/{REPO_ID}")
    print("=" * 80)


if __name__ == "__main__":
    upload_dataset()
