"""
Minimal Autoregressive Model with Policy & Tool-Call Decoders
============================================================
Implements:
1. Shared language-model backbone (Causal Transformer).
2. Policy classification head (answer vs. call_tool).
3. Structured tool-call and answer generation via shared causal LM head.
4. SCMER (Selective Cosine-Margin Embedding Regularization) module for controlled ablation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MinimalModelConfig:
    vocab_size: int = 120
    max_seq_len: int = 384
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.05
    policy_classes: int = 2  # 0 = direct_answer, 1 = call_tool
    alpha_policy: float = 1.0
    beta_tool_call: float = 1.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MinimalModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_dim // cfg.num_heads
        self.hidden_dim = cfg.hidden_dim

        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch 2.x fast scaled_dot_product_attention with causal mask
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: MinimalModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(cfg.ffn_dim, cfg.hidden_dim, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MinimalToolModel(nn.Module):
    """
    Minimal viable autoregressive agentic model.
    Contains:
    - Shared language-model backbone.
    - Policy classification head (answer vs. call_tool).
    - Causal language modeling head for tool calls and answers.
    """

    def __init__(self, cfg: MinimalModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_embeddings = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_dim)

        # 1. Policy head: predicts 0 (direct answer) or 1 (call tool)
        self.policy_head = nn.Linear(cfg.hidden_dim, cfg.policy_classes)

        # 2. Token prediction head: shared for tool calls and answers
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight  # Weight tying

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Bias policy head slightly towards 0 (direct answer) initial prior
        nn.init.constant_(self.policy_head.bias, 0.0)
        self.policy_head.bias.data[0] = 0.5

    def forward_backbone(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        h = self.tok_embeddings(input_ids) + self.pos_embeddings(positions)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return h

    def compute_scmer_loss(
        self,
        margin: float = 0.10,
        subsample: int = 64,
        num_pairs: int = 1024,
    ) -> torch.Tensor:
        """
        Selective Cosine-Margin Embedding Regularization (SCMER).
        Penalizes cosine similarities above margin m between distinct embedding vectors.
        """
        w = self.tok_embeddings.weight  # [V, d]
        # Normalize embeddings
        w_norm = F.normalize(w, p=2, dim=-1)

        # Sample random pairs of distinct embeddings
        v_size = w.shape[0]
        idx_i = torch.randint(0, v_size, (num_pairs,), device=w.device)
        idx_j = torch.randint(0, v_size, (num_pairs,), device=w.device)
        mask_distinct = (idx_i != idx_j)
        if not mask_distinct.any():
            return torch.tensor(0.0, device=w.device)

        idx_i = idx_i[mask_distinct]
        idx_j = idx_j[mask_distinct]

        cos_sim = (w_norm[idx_i] * w_norm[idx_j]).sum(dim=-1)
        violations = F.relu(cos_sim - margin)
        scmer_penalty = (violations ** 2).mean()
        return scmer_penalty

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        policy_targets: torch.Tensor,
        policy_positions: torch.Tensor,
        tool_call_mask: torch.Tensor,
        answer_mask: torch.Tensor,
        scmer_enabled: bool = False,
        scmer_lambda: float = 0.005,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes masked multi-task training loss:
        L = L_answer + alpha * L_policy + beta * L_tool_call (+ L_scmer)
        """
        B, T = input_ids.shape
        h = self.forward_backbone(input_ids)  # [B, T, d]

        # 1. Policy Loss at policy_positions (<|user_end|>)
        # Gather representations at the policy position for each batch element
        batch_idx = torch.arange(B, device=input_ids.device)
        user_end_h = h[batch_idx, policy_positions]  # [B, d]
        policy_logits = self.policy_head(user_end_h)  # [B, 2]
        loss_policy = F.cross_entropy(policy_logits, policy_targets)

        # 2. LM Logits for sequence
        lm_logits = self.lm_head(h)  # [B, T, V]
        token_loss_all = F.cross_entropy(
            lm_logits.view(-1, self.cfg.vocab_size),
            target_ids.view(-1),
            reduction="none",
        ).view(B, T)

        # 3. Masked Tool-Call Loss
        tool_call_tokens = tool_call_mask.sum()
        if tool_call_tokens > 0:
            loss_tool_call = (token_loss_all * tool_call_mask).sum() / tool_call_tokens
        else:
            loss_tool_call = torch.tensor(0.0, device=input_ids.device)

        # 4. Masked Answer Loss
        answer_tokens = answer_mask.sum()
        if answer_tokens > 0:
            loss_answer = (token_loss_all * answer_mask).sum() / answer_tokens
        else:
            loss_answer = torch.tensor(0.0, device=input_ids.device)

        # Total combined loss
        total_loss = (
            loss_answer
            + self.cfg.alpha_policy * loss_policy
            + self.cfg.beta_tool_call * loss_tool_call
        )

        loss_dict = {
            "loss_total": total_loss,
            "loss_answer": loss_answer.detach(),
            "loss_policy": loss_policy.detach(),
            "loss_tool_call": loss_tool_call.detach(),
        }

        # 5. Optional SCMER Regularization
        if scmer_enabled and scmer_lambda > 0.0:
            loss_scmer = self.compute_scmer_loss()
            total_loss = total_loss + scmer_lambda * loss_scmer
            loss_dict["loss_scmer"] = loss_scmer.detach()
            loss_dict["loss_total"] = total_loss

        return loss_dict

    @torch.no_grad()
    def predict_policy(self, input_ids: torch.Tensor, user_end_pos: int) -> Tuple[int, float]:
        """Predicts policy action (0 = direct_answer, 1 = call_tool) and probability."""
        self.eval()
        h = self.forward_backbone(input_ids)
        user_end_h = h[:, user_end_pos]
        logits = self.policy_head(user_end_h)
        probs = F.softmax(logits, dim=-1)
        pred_class = int(torch.argmax(probs, dim=-1).item())
        prob = float(probs[0, pred_class].item())
        return pred_class, prob

    @torch.no_grad()
    def generate_autoregressive(
        self,
        prompt_ids: torch.Tensor,
        stop_token_id: int,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressive greedy or sampled decoding until stop token."""
        self.eval()
        curr_ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            if curr_ids.shape[1] >= self.cfg.max_seq_len:
                break
            h = self.forward_backbone(curr_ids)
            next_logits = self.lm_head(h[:, -1, :])  # [1, V]

            if temperature <= 1e-4:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            if next_token.item() == stop_token_id:
                break

        return curr_ids
