#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload CerebroSpheroid V5 Full Dictionary Lexicon to Hugging Face
================================================================
Updates `AjitKumar-Bio/cerebro-spheroid-v5-frontier-reasoning-tools-100k`
with the full dictionary configuration (`dictionary.parquet`), containing
111,651 complete dictionary definitions, WordNet examples, and affix anchors
to prevent rare token decay across the SentencePiece 32k vocabulary.
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
DICT_DIR = os.path.join(PROJECT_ROOT, "data", "dictionary")

UPDATED_DATASET_CARD = """---
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
- dictionary
- lexicon
- vocabulary-preservation
- rare-token-decay-prevention
- cerebro-spheroid
- sft
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/train.parquet"
- config_name: dictionary
  data_files:
  - split: train
    path: "data/dictionary.parquet"
- config_name: all
  data_files:
  - split: train
    path: "data/*.parquet"
---

# CerebroSpheroid V5: Frontier 100k Reasoning, Tools & Full Dictionary Lexicon

This repository provides **over 211,000 total high-density training records** across two synergistic configurations:
1. **`default` (100,000 Prompts):** Frontier step-by-step reasoning (DeepSeek-R1, Sky-T1, NuminaMath) and industry API tool calling (Apigen, Hermes).
2. **`dictionary` (111,651 Prompts):** Full English Unabridged Dictionary (Webster's + WordNet multi-sense definitions, parts of speech, illustrative examples, and affix morpheme anchors) designed specifically to **prevent rare token decay and vocabulary collapse**.
3. **`all` (211,651 Prompts):** Complete blended corpus for simultaneous reasoning capability and lexical grounding.

---

## 🛡️ Why a Full Dictionary? (Preventing Rare Token Decay)

In LLM fine-tuning, midtraining, and reasoning adaptation, training heavily concentrates on math, code, and common conversational tokens. Tokens that appear rarely in specialized reasoning benchmarks receive near-zero gradients. Under standard optimizer weight decay (e.g. AdamW with $\lambda = 0.01$ or SCMER), inactive token embeddings suffer from:
*   **Norm Shrinkage & Anisotropy Drift:** Inactive embedding rows shrink towards zero or collapse into narrow cones.
*   **Catastrophic Forgetting of Rare Words:** The model unlearns obscure vocabulary, scientific terms, archaic lemmas, and specialized prefixes/suffixes.
*   **Perplexity Spikes:** Inference perplexity explodes whenever rare or low-frequency words appear in context.

**The Solution:**
By mixing in the `dictionary` subset during training (or pre-training / SFT), **every single word, lemma, and subword piece in the vocabulary receives explicit, loss-bearing gradient updates**. Each entry includes linguistic derivation in `<think>` and authoritative definitions, keeping the entire embedding manifold active, isotropic, and semantically grounded.

---

## 🎯 Dataset Composition

| Subset | Domain | Count | Primary Focus |
| :--- | :--- | :--- | :--- |
| **`default`** | DeepSeek-R1 Distill | **57,495** | Deep deliberation scratchpad across STEM, Code, and Logic |
| **`default`** | Olympiad Math & Proofs | **14,392** | Competition mathematics (AIME, Olympiad, TACO/APPs) |
| **`default`** | Formal Derivations | **15,000** | Rigorous step-by-step mathematical proofs |
| **`default`** | Industry Tool Calling | **13,113** | Multi-turn verified JSON API function calling |
| **`dictionary`** | Webster's Unabridged | **111,566** | Comprehensive word definitions and parts of speech |
| **`dictionary`** | Affix Morpheme Anchors | **85** | Greek/Latin roots, prefixes, and suffixes |
| **Total** | **Unified Frontier Corpus** | **211,651** | **Reasoning + Function Calling + Lexical Grounding** |

---

## 🧠 Dual-Parity Token Format

Every sample in both subsets supports **both** native cognitive scratchpads and industry-standard chat:

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

# 1. Load the 100,000 reasoning and tool prompts (Default)
reasoning_ds = load_dataset(
    "AjitKumar-Bio/cerebro-spheroid-v5-frontier-reasoning-tools-100k",
    split="train",
)
print("Reasoning & Tools:", len(reasoning_ds))

# 2. Load the 111,651 Full Dictionary Lexicon (Rare Token Preservation)
dictionary_ds = load_dataset(
    "AjitKumar-Bio/cerebro-spheroid-v5-frontier-reasoning-tools-100k",
    "dictionary",
    split="train",
)
print("Dictionary Lexicon:", len(dictionary_ds))

# 3. Load all 211,651 unified records
unified_ds = load_dataset(
    "AjitKumar-Bio/cerebro-spheroid-v5-frontier-reasoning-tools-100k",
    "all",
    split="train",
)
print("Unified Total:", len(unified_ds))
```

### Files in Repository

- `data/train.parquet` (337 MB): 100,000 reasoning & tool calling records.
- `data/dictionary.parquet` (15.1 MB): 111,651 dictionary definitions & morpheme anchors.
- `data/frontier_100k_spheroid_format.jsonl` (1.17 GB): Native Spheroid format (reasoning + tools).
- `data/frontier_100k_industry_sharegpt.jsonl` (1.18 GB): ShareGPT format (reasoning + tools).
- `data/dictionary_spheroid_format.jsonl` (62.7 MB): Native Spheroid format (full dictionary).
- `data/dictionary_industry_sharegpt.jsonl` (66.1 MB): ShareGPT format (full dictionary).

---

## 📜 License
Apache-2.0. Open for research, fine-tuning, and commercial deployment.
"""


def upload_dictionary():
    print("=" * 80)
    print(f"🚀 Uploading Full Dictionary Lexicon to Hugging Face: {REPO_ID}")
    print("=" * 80)

    token = get_hf_token()
    if not token:
        raise RuntimeError("No Hugging Face token found. Set HF_TOKEN in ~/.env or environment.")

    api = HfApi(token=token)
    user_info = api.whoami()
    print(f"✓ Authenticated as: {user_info.get('name')}")

    # 1. Upload Dictionary Parquet (named data/dictionary.parquet)
    local_pq = os.path.join(DICT_DIR, "dictionary_lexicon_full.parquet")
    if os.path.exists(local_pq):
        size_mb = os.path.getsize(local_pq) / 1e6
        print(f"\n📦 Uploading full dictionary parquet ({size_mb:.1f} MB) -> data/dictionary.parquet...")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=local_pq,
            path_in_repo="data/dictionary.parquet",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="feat(dataset): add full dictionary parquet to prevent rare token decay"
        )
        print(f"✓ Uploaded data/dictionary.parquet in {time.time() - t0:.1f}s")

    # 2. Upload Dictionary Spheroid JSONL
    local_sph = os.path.join(DICT_DIR, "dictionary_spheroid_format.jsonl")
    if os.path.exists(local_sph):
        size_mb = os.path.getsize(local_sph) / 1e6
        print(f"\n📦 Uploading dictionary Spheroid JSONL ({size_mb:.1f} MB)...")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=local_sph,
            path_in_repo="data/dictionary_spheroid_format.jsonl",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="feat(dataset): add full dictionary spheroid format jsonl"
        )
        print(f"✓ Uploaded data/dictionary_spheroid_format.jsonl in {time.time() - t0:.1f}s")

    # 3. Upload Dictionary Industry ShareGPT JSONL
    local_ind = os.path.join(DICT_DIR, "dictionary_industry_sharegpt.jsonl")
    if os.path.exists(local_ind):
        size_mb = os.path.getsize(local_ind) / 1e6
        print(f"\n📦 Uploading dictionary ShareGPT JSONL ({size_mb:.1f} MB)...")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=local_ind,
            path_in_repo="data/dictionary_industry_sharegpt.jsonl",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="feat(dataset): add full dictionary industry sharegpt format jsonl"
        )
        print(f"✓ Uploaded data/dictionary_industry_sharegpt.jsonl in {time.time() - t0:.1f}s")

    # 4. Upload Updated README.md (Dataset Card with multi-subset configs)
    print("\n📝 Updating dataset card (README.md) with multi-subset configurations...")
    readme_path = os.path.join(DICT_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(UPDATED_DATASET_CARD.strip() + "\n")

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="docs(dataset): configure multi-subset configs for default and dictionary"
    )
    print("✓ Uploaded updated README.md")

    print("\n" + "=" * 80)
    print(f"🎉 Full Dictionary Dataset Successfully Uploaded to Hugging Face!")
    print(f"🔗 URL: https://huggingface.co/datasets/{REPO_ID}")
    print("=" * 80)


if __name__ == "__main__":
    upload_dictionary()
