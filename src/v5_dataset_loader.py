#!/usr/bin/env python3
"""
📦 CerebroSpheroid V5: High-Throughput Tri-Modal Custom Dataset Loader
=====================================================================
Streams and batches samples from `data/cerebro_v5_custom_dataset/` with
automated Token-Level Answer-Anchor Loss Weighting (AALW).

Weights:
  - Thinking scratchpad tokens (<think> ... </think>): 1.0x
  - Deliberation boundary closing tag (</think>):      2.0x
  - Tool action payload tokens (<act> ... </act>):     3.5x
  - Plain-text answer & anchor tokens:                 4.5x
"""

import os
import json
from typing import Dict, Any, List, Optional, Iterator
import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq

from transformers import AutoTokenizer
from src.corpus import encode_no_special


class CerebroV5Dataset(Dataset):
    """PyTorch Dataset for CerebroSpheroid V5 Custom Tri-Modal Corpus."""

    def __init__(
        self,
        file_path: str,
        tokenizer: Any,
        vocab_size: int = 32000,
        max_prompt_len: int = 256,
        max_target_len: int = 512,
        anchor_weight: float = 4.5,
        action_weight: float = 3.5,
        think_close_weight: float = 2.0,
        think_weight: float = 1.0,
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_prompt_len = max_prompt_len
        self.max_target_len = max_target_len
        self.anchor_weight = anchor_weight
        self.action_weight = action_weight
        self.think_close_weight = think_close_weight
        self.think_weight = think_weight

        self.samples: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self):
        if self.file_path.endswith(".parquet"):
            table = pq.read_table(self.file_path)
            pydict = table.to_pydict()
            n_rows = len(pydict["id"])
            for i in range(n_rows):
                self.samples.append({
                    "id": pydict["id"][i],
                    "category": pydict["category"][i],
                    "thinking_level": pydict["thinking_level"][i],
                    "prompt": pydict["prompt"][i],
                    "thinking": pydict["thinking"][i],
                    "action": pydict["action"][i],
                    "answer": pydict["answer"][i],
                    "formatted_input": pydict["formatted_input"][i],
                    "formatted_output": pydict["formatted_output"][i],
                })
        else:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        p_raw = item["formatted_input"]
        o_raw = item["formatted_output"]
        lvl = item.get("thinking_level", 0)

        # 1. Tokenize Input (<stim> ... </stim>)
        p_ids = [t % self.vocab_size for t in encode_no_special(self.tokenizer, p_raw)][:self.max_prompt_len]
        if not p_ids:
            p_ids = [1]

        # 2. Tokenize Output
        o_ids = [t % self.vocab_size for t in encode_no_special(self.tokenizer, o_raw)][:self.max_target_len]
        if not o_ids:
            o_ids = [2]

        # 3. Compute Token-Level Loss Weights (AALW)
        weights = [self.think_weight] * len(o_ids)

        if lvl == 0:
            # Entire plain text answer gets high anchor weight
            weights = [self.anchor_weight] * len(o_ids)
        else:
            # Mark boundaries in output string
            # Level 1/2: <think> ... </think> {answer}
            # Level 3:   <think> ... </think> <act> ... </act> {answer}
            # Fast substring token locator
            out_str = o_raw
            think_end_str = "</think>"
            act_start_str = "<act>"
            act_end_str = "</act>"

            pos_think_end = out_str.find(think_end_str)
            pos_act_end = out_str.find(act_end_str)

            if pos_think_end != -1:
                prefix_think = out_str[:pos_think_end + len(think_end_str)]
                n_think_toks = len(encode_no_special(self.tokenizer, prefix_think))

                # Set answer weights after </think>
                for i in range(min(n_think_toks, len(weights)), len(weights)):
                    weights[i] = self.anchor_weight

                # Weight </think> boundary higher
                if 0 < n_think_toks <= len(weights):
                    weights[max(0, n_think_toks - 2):n_think_toks] = [self.think_close_weight] * min(2, n_think_toks)

            if pos_act_end != -1 and pos_think_end != -1:
                prefix_act = out_str[:pos_act_end + len(act_end_str)]
                n_act_toks = len(encode_no_special(self.tokenizer, prefix_act))
                # Set action tokens to action_weight
                for i in range(min(n_think_toks, len(weights)), min(n_act_toks, len(weights))):
                    weights[i] = self.action_weight
                # Tokens after </act> are final answer
                for i in range(min(n_act_toks, len(weights)), len(weights)):
                    weights[i] = self.anchor_weight

        return {
            "prompt_ids": torch.tensor(p_ids, dtype=torch.long),
            "target_ids": torch.tensor(o_ids, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
            "thinking_level": lvl,
            "category": item.get("category", "general"),
        }


def collate_cerebro_v5_batch(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collates variable-length tokenized pairs into right-padded batch tensors."""
    max_p = max(len(b["prompt_ids"]) for b in batch)
    max_t = max(len(b["target_ids"]) for b in batch)
    bsz = len(batch)

    prompts = torch.full((bsz, max_p), pad_id, dtype=torch.long)
    targets = torch.full((bsz, max_t), pad_id, dtype=torch.long)
    weights = torch.zeros((bsz, max_t), dtype=torch.float32)
    levels = torch.tensor([b["thinking_level"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        p_len = len(b["prompt_ids"])
        t_len = len(b["target_ids"])
        prompts[i, :p_len] = b["prompt_ids"]
        targets[i, :t_len] = b["target_ids"]
        weights[i, :t_len] = b["weights"]

    return {
        "prompts": prompts,
        "targets": targets,
        "weights": weights,
        "thinking_levels": levels,
    }


def get_v5_dataloader(
    file_path: str,
    tokenizer: Any,
    batch_size: int = 16,
    shuffle: bool = True,
    max_prompt_len: int = 256,
    max_target_len: int = 512,
    num_workers: int = 0,
) -> DataLoader:
    """Creates a configured PyTorch DataLoader ready for SFT training."""
    dataset = CerebroV5Dataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_prompt_len=max_prompt_len,
        max_target_len=max_target_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_cerebro_v5_batch,
        num_workers=num_workers,
    )
