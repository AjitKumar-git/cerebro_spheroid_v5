"""
Training Loop & Collation for Minimal Tool MVP
=============================================
Implements supervised training on complete multi-turn traces:
user request -> optional call -> tool result -> final answer.
Supports masked multi-task objective and SCMER controlled ablation.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from experiments.minimal_tool_mvp.dataset_generator import ToolTrace, TraceDatasetGenerator
from experiments.minimal_tool_mvp.model import MinimalModelConfig, MinimalToolModel
from experiments.minimal_tool_mvp.tokenizer import MinimalTokenizer


class TraceDataset(Dataset):
    """Pytorch Dataset wrapping list of ToolTrace records."""

    def __init__(self, traces: List[ToolTrace], tokenizer: MinimalTokenizer, max_seq_len: int = 384):
        self.traces = traces
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trace = self.traces[idx]
        tok = self.tokenizer

        u_start = tok.encode("<|user|>")
        u_text = tok.encode(trace.user_request)
        u_end = tok.encode("<|user_end|>")
        prefix = u_start + u_text + u_end
        policy_pos = len(prefix) - 1  # position of <|user_end|>

        tokens: List[int] = list(prefix)
        tool_call_mask: List[float] = [0.0] * len(prefix)
        answer_mask: List[float] = [0.0] * len(prefix)

        if trace.policy_target == 1:
            # Case 2: Tool Needed
            call_json = json.dumps(trace.tool_call, separators=(",", ":"))
            tc_start = tok.encode("<|tool_call|>")
            tc_body = tok.encode(call_json)
            tc_end = tok.encode("<|tool_call_end|>")
            tc_all = tc_start + tc_body + tc_end

            tokens.extend(tc_all)
            # Mask tool call tokens (the target for the model to generate)
            tool_call_mask.extend([1.0] * len(tc_all))
            answer_mask.extend([0.0] * len(tc_all))

            # Sensory feed-in: tool result (input to model, no loss)
            res_json = json.dumps(trace.tool_result, separators=(",", ":"))
            tr_start = tok.encode("<|tool_result|>")
            tr_body = tok.encode(res_json)
            tr_end = tok.encode("<|tool_result_end|>")
            tr_all = tr_start + tr_body + tr_end

            tokens.extend(tr_all)
            tool_call_mask.extend([0.0] * len(tr_all))
            answer_mask.extend([0.0] * len(tr_all))

        # Final Answer tokens
        ans_start = tok.encode("<|answer|>")
        ans_body = tok.encode(trace.final_answer)
        ans_end = tok.encode("<|answer_end|>")
        ans_all = ans_start + ans_body + ans_end

        tokens.extend(ans_all)
        tool_call_mask.extend([0.0] * len(ans_all))
        # Mask answer tokens (target for the model to generate)
        answer_mask.extend([1.0] * len(ans_all))

        # Truncate if necessary
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]
            tool_call_mask = tool_call_mask[: self.max_seq_len]
            answer_mask = answer_mask[: self.max_seq_len]

        # Targets are shifted by 1 position for next-token prediction
        target_ids = tokens[1:] + [tok.pad_token_id]
        # Shift masks accordingly
        tool_call_mask_shifted = tool_call_mask[1:] + [0.0]
        answer_mask_shifted = answer_mask[1:] + [0.0]

        return {
            "tokens": tokens,
            "targets": target_ids,
            "policy_target": trace.policy_target,
            "policy_pos": policy_pos,
            "tool_call_mask": tool_call_mask_shifted,
            "answer_mask": answer_mask_shifted,
            "length": len(tokens),
        }


def collate_trace_batch(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collates and dynamically pads batches."""
    max_len = max(item["length"] for item in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    target_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    policy_targets = torch.zeros(B, dtype=torch.long)
    policy_positions = torch.zeros(B, dtype=torch.long)
    tool_call_mask = torch.zeros((B, max_len), dtype=torch.float32)
    answer_mask = torch.zeros((B, max_len), dtype=torch.float32)

    for i, item in enumerate(batch):
        L = item["length"]
        input_ids[i, :L] = torch.tensor(item["tokens"], dtype=torch.long)
        target_ids[i, :L] = torch.tensor(item["targets"], dtype=torch.long)
        policy_targets[i] = item["policy_target"]
        policy_positions[i] = item["policy_pos"]
        tool_call_mask[i, :L] = torch.tensor(item["tool_call_mask"], dtype=torch.float32)
        answer_mask[i, :L] = torch.tensor(item["answer_mask"], dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "policy_targets": policy_targets,
        "policy_positions": policy_positions,
        "tool_call_mask": tool_call_mask,
        "answer_mask": answer_mask,
    }


def train_model(
    model: MinimalToolModel,
    train_dataset: TraceDataset,
    num_epochs: int = 15,
    batch_size: int = 32,
    lr: float = 3e-3,
    scmer_enabled: bool = False,
    scmer_lambda: float = 0.005,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Trains the MinimalToolModel on the trace dataset."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_trace_batch(b, train_dataset.tokenizer.pad_token_id),
    )

    history: Dict[str, List[float]] = {
        "loss_total": [],
        "loss_answer": [],
        "loss_policy": [],
        "loss_tool_call": [],
    }
    if scmer_enabled:
        history["loss_scmer"] = []

    model.train()
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_losses: Dict[str, float] = {k: 0.0 for k in history.keys()}
        steps = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            policy_targets = batch["policy_targets"].to(device)
            policy_positions = batch["policy_positions"].to(device)
            tool_call_mask = batch["tool_call_mask"].to(device)
            answer_mask = batch["answer_mask"].to(device)

            optimizer.zero_grad()
            loss_dict = model.compute_loss(
                input_ids=input_ids,
                target_ids=target_ids,
                policy_targets=policy_targets,
                policy_positions=policy_positions,
                tool_call_mask=tool_call_mask,
                answer_mask=answer_mask,
                scmer_enabled=scmer_enabled,
                scmer_lambda=scmer_lambda,
            )

            loss_total = loss_dict["loss_total"]
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in epoch_losses.keys():
                epoch_losses[k] += loss_dict[k].item()
            steps += 1

        for k in history.keys():
            history[k].append(epoch_losses[k] / max(1, steps))

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == num_epochs):
            scmer_str = f" | SCMER: {epoch_losses.get('loss_scmer', 0.0)/steps:.4f}" if scmer_enabled else ""
            print(
                f"[Epoch {epoch:02d}/{num_epochs:02d}] "
                f"Total: {history['loss_total'][-1]:.4f} | "
                f"Policy: {history['loss_policy'][-1]:.4f} | "
                f"Tool: {history['loss_tool_call'][-1]:.4f} | "
                f"Answer: {history['loss_answer'][-1]:.4f}"
                f"{scmer_str}"
            )

    elapsed = time.time() - start_time
    if verbose:
        print(f"Training finished in {elapsed:.2f}s.")

    return history
