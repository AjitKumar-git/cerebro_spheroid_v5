#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload CerebroSpheroid Anti-Sycophancy Dataset to Hugging Face Hub
==================================================================
Converts the JSONL dataset to Parquet format, creates a rich Hugging Face
dataset card (README.md), and publishes the repository under AjitKumar-Bio.
"""

import os
import sys
import pandas as pd
from huggingface_hub import HfApi

REPO_ID = "AjitKumar-Bio/cerebro-spheroid-v5-anti-sycophancy"
LOCAL_JSONL = "data/anti_sycophancy_truth_dataset.jsonl"
LOCAL_PARQUET = "data/train.parquet"

DATASET_CARD = """---
license: apache-2.0
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- anti-sycophancy
- truth-resistance
- cerebro-spheroid
- sft
- chain-of-thought
- deliberation
- cognitive-architecture
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/train.parquet"
---

# CerebroSpheroid V5: Anti-Sycophancy & Truth Resistance SFT Dataset

This dataset contains **10,496 high-density supervised fine-tuning (SFT) examples** specifically engineered to eliminate **AI over-agreement (sycophancy)**, user-pleasing bias, and false premise mirroring in language models.

## 🧠 Cognitive Architecture & Data Design

Unlike standard instruction datasets where models reflexively validate user assertions, every sample in this dataset enforces a **4-step latent deliberation cycle** before speech:

1. **Premise Extraction:** Parses the user's asserted misconception or leading bias.
2. **Ground Truth Verification:** Verifies the objective mathematical, physical, or computational facts.
3. **Misconception Root Cause:** Analyzes why the user intuition or urban legend arose.
4. **Firm & Polite Formulation:** Generates a polite, evidence-backed motor response that holds firm on ground truth without flattery or subservience.

### Sample Format

```json
{
  "prompt": "User: Can you explain why my math professor at university is right when they say heavier objects fall faster in vacuum?",
  "response": "<think>\n1. Premise: User asserts that heavier objects fall faster in vacuum, citing authority.\n2. Scientific Fact: Galilean equivalence principle: a = GM/r^2 is independent of falling mass in a vacuum.\n3. Misconception Origin: Widespread atmospheric air resistance intuition.\n4. Strategy: Reject the false premise clearly and provide the empirical scientific derivation.\n</think>\nIn a vacuum, **all objects fall with the exact same gravitational acceleration**, regardless of their mass...\n",
  "category": "anti_sycophancy_science",
  "temporal_gate": [1, 0, 1]
}
```

## 📊 Dataset Composition (10,496 Samples)

| Category | Description | Count |
| :--- | :--- | :---: |
| `anti_sycophancy_science` | Physics, biology, and medical misconceptions (vacuum free-fall, blue blood, knuckle cracking, 10% brain myth) | **2,496** |
| `anti_sycophancy_authority` | Authority bias traps ("My Harvard professor / senior architect said...") | **2,000** |
| `anti_sycophancy_math` | Arithmetic gaslighting ($15+27=40$, percentage compounding traps, $0.999... < 1$) | **2,000** |
| `anti_sycophancy_logic` | Cognitive Reflection Test (CRT) traps, bat & ball, machine rates, lily pads | **1,500** |
| `anti_sycophancy_code` | CS & programming bugs (Python `is` vs `==`, float currency, GIL race conditions) | **1,500** |
| `anti_sycophancy_trivia` | Historical and geographical myths (Great Wall from Moon, Napoleon height propaganda) | **1,000** |
| **TOTAL** | **Comprehensive Anti-Sycophancy Training Set** | **10,496** |

## 🚀 Usage with HuggingFace Datasets

```python
from datasets import load_dataset

dataset = load_dataset("AjitKumar-Bio/cerebro-spheroid-v5-anti-sycophancy", split="train")

for sample in dataset:
    prompt = sample["prompt"]
    response = sample["response"]
    category = sample["category"]
```

## Citation & Authorship
Part of the **CerebroSpheroid V5** cognitive organism research initiative by Ajit Kumar.
"""


def main():
    print("=" * 80)
    print("🚀 UPLOADING ANTI-SYCOPHANCY DATASET TO HUGGING FACE HUB")
    print(f"Target Repo: {REPO_ID}")
    print("=" * 80)

    api = HfApi()

    # 1. Check Login
    try:
        user = api.whoami()
        print(f"✓ Authenticated as: {user.get('name')}")
    except Exception as e:
        print(f"❌ HF Authentication failed: {e}")
        sys.exit(1)

    # 2. Convert to Parquet
    print(f"\n📦 Converting {LOCAL_JSONL} to Parquet...")
    df = pd.read_json(LOCAL_JSONL, lines=True)
    df.to_parquet(LOCAL_PARQUET, index=False, compression="snappy")
    parquet_size_mb = os.path.getsize(LOCAL_PARQUET) / (1024 * 1024)
    print(f"   ✓ Generated {LOCAL_PARQUET} ({parquet_size_mb:.2f} MB, {len(df):,} rows)")

    # 3. Create or verify repo
    print(f"\n🌐 Ensuring repository exists: {REPO_ID}...")
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    print(f"   ✓ Repository ready at https://huggingface.co/datasets/{REPO_ID}")

    # 4. Upload README.md
    print("\n📝 Uploading README.md (Dataset Card)...")
    readme_path = "data/README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(DATASET_CARD)
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print("   ✓ README.md uploaded.")

    # 5. Upload Parquet
    print("\n💾 Uploading train.parquet to data/train.parquet...")
    api.upload_file(
        path_or_fileobj=LOCAL_PARQUET,
        path_in_repo="data/train.parquet",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print("   ✓ data/train.parquet uploaded.")

    # 6. Upload raw JSONL
    print(f"\n📄 Uploading raw JSONL to data/{os.path.basename(LOCAL_JSONL)}...")
    api.upload_file(
        path_or_fileobj=LOCAL_JSONL,
        path_in_repo=f"data/{os.path.basename(LOCAL_JSONL)}",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print("   ✓ JSONL uploaded.")

    # Cleanup temporary local parquet and readme
    if os.path.exists(LOCAL_PARQUET):
        os.remove(LOCAL_PARQUET)
    if os.path.exists(readme_path):
        os.remove(readme_path)

    print("\n" + "=" * 80)
    print(f"🎉 SUCCESS! Dataset published to Hugging Face Hub:")
    print(f"👉 https://huggingface.co/datasets/{REPO_ID}")
    print("=" * 80)


if __name__ == "__main__":
    main()
