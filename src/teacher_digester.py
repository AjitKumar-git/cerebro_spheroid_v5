# -*- coding: utf-8 -*-
"""
Teacher Digester Engine for Cerebro Spheroid V5
================================================
Loads a frontier teacher language model (default: Qwen/Qwen2.5-1.5B-Instruct)
to digest and analyze input training data prior to feeding it to Cerebro Spheroid V5.

Mathematical Formulation:
--------------------------
1. Teacher Forward Pass (Eval / No-Grad):
   For sequence x = (x_1, x_2, ..., x_T):
   P_teacher(x_t | x_<t) is the probability assigned to target token x_t.
   Surprisal (Per-token Cross-Entropy):
       L_{Q, t} = -log P_teacher(x_t | x_<t)
   Teacher Predictive Entropy:
       H_{Q, t} = -sum_v P_teacher(v | x_<t) * log P_teacher(v | x_<t)

2. Token Information Density Score:
   S_t = L_{Q, t} + beta * H_{Q, t}
   Normalized relative to sequence mean and standard deviation:
       z_t = (S_t - mu_S) / (sigma_S + eps)
   Dynamic Weight Mapping:
       w_{Q, t} = clamp(1.0 + alpha * tanh(z_t), w_min, w_max)
   Mean-Preserving Normalization:
       w_{Q, t} <- w_{Q, t} * (1.0 / mean(w_Q))

3. Cross-Tokenizer Character-Span Alignment:
   Qwen Tokenizer (152k vocab) and Spheroid Tokenizer (32k vocab) tokenize
   the same document with different boundaries.
   We project per-token teacher weights w_{Q, t} onto character intervals [c_start, c_end)
   and reconstruct the exact target weights for Spheroid tokens:
       w_{Spheroid, k} = (1 / |C_k|) * sum_{c in C_k} char_weight[c]

This concentrates gradient updates on high-information tokens (reasoning steps,
technical entities, mathematical relations) while scaling down boilerplate stopwords,
enforcing a denser, more parameter-efficient representation in v5_compact.
"""

import os
import math
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

os.environ["HF_HOME"] = os.path.expanduser("~/.hf_cache")

logger = logging.getLogger("TeacherDigester")


class QwenTeacherDigester:
    """
    Teacher Digester model manager using Qwen2.5-1.5B-Instruct
    to generate dynamic next-token importance weights.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: Optional[torch.dtype] = None,
        alpha: float = 1.5,
        beta: float = 0.2,
        w_min: float = 0.2,
        w_max: float = 4.0,
        lazy_load: bool = True,
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.alpha = alpha
        self.beta = beta
        self.w_min = w_min
        self.w_max = w_max
        self.loaded = False

        if dtype is None:
            if "cuda" in device and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            elif "cuda" in device:
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype

        self.tokenizer = None
        self.model = None

        if not lazy_load:
            self.load()

    def load(self):
        """Loads teacher tokenizer and causal LM into evaluation mode."""
        if self.loaded:
            return

        print(f"📖 [TEACHER DIGESTER] Loading teacher model: {self.model_id} on {self.device} ({self.dtype})...")
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=str(self.device) if "cuda" in str(self.device) else None,
            trust_remote_code=True,
        )
        if "cpu" in str(self.device):
            self.model = self.model.to(self.device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.loaded = True
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"  ✓ Teacher model active: {param_count:.2f}B parameters (eval mode, zero grads)")

    def compute_teacher_token_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes normalized importance weights for Qwen target tokens:
        Returns tensor of shape [batch_size, seq_len - 1].
        """
        if not self.loaded:
            self.load()

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]  # [B, T-1, V]
            targets = input_ids[:, 1:]          # [B, T-1]
            del outputs

            B, T_minus_1, V = logits.shape

            # 1. Chunked Surprisal (Cross-Entropy) to minimize peak VRAM footprint
            surprisal_chunks = []
            chunk_size = 128
            for c_logits, c_targets in zip(logits.split(chunk_size, dim=1), targets.split(chunk_size, dim=1)):
                c_b, c_t, _ = c_logits.shape
                s_c = F.cross_entropy(
                    c_logits.reshape(c_b * c_t, V).float(),
                    c_targets.reshape(c_b * c_t),
                    reduction="none",
                ).reshape(c_b, c_t)
                surprisal_chunks.append(s_c)
            surprisal = torch.cat(surprisal_chunks, dim=1)

            # 2. Predictive Entropy (memory-efficient chunking)
            if self.beta > 0.0:
                entropy_chunks = []
                for c_logits in logits.split(chunk_size, dim=1):
                    p_c = F.softmax(c_logits.float(), dim=-1)
                    log_p_c = F.log_softmax(c_logits.float(), dim=-1)
                    h_c = -(p_c * log_p_c).sum(dim=-1)
                    entropy_chunks.append(h_c)
                entropy = torch.cat(entropy_chunks, dim=1)
                density = surprisal + self.beta * entropy
            else:
                density = surprisal

            del logits

            # 4. Standardized Bounded Dynamic Weights
            mean_d = density.mean(dim=-1, keepdim=True)
            std_d = density.std(dim=-1, keepdim=True) + 1e-6
            z = (density - mean_d) / std_d

            weights = 1.0 + self.alpha * torch.tanh(z)
            weights = torch.clamp(weights, min=self.w_min, max=self.w_max)

            # Normalize sequence mean to 1.0 to preserve gradient magnitude scale
            w_mean = weights.mean(dim=-1, keepdim=True) + 1e-6
            weights = weights / w_mean

            return weights

    def digest_text(
        self,
        text: str,
        spheroid_tokenizer,
        max_length: int = 2048,
    ) -> Tuple[List[int], List[float]]:
        """
        Digests a text document:
        1. Tokenizes with Qwen and computes per-token surprisal & entropy.
        2. Projects importance weights to character spans.
        3. Tokenizes with Spheroid tokenizer (32k) and maps weights to Spheroid tokens.
        
        Returns:
            spheroid_token_ids: List[int]
            spheroid_token_weights: List[float] (aligned with spheroid_token_ids)
        """
        if not self.loaded:
            self.load()

        if not text or len(text.strip()) == 0:
            return [1, 2], [1.0, 1.0]

        # 1. Qwen Tokenization with Character Offsets
        q_enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        q_input_ids = q_enc["input_ids"]
        q_offsets = q_enc["offset_mapping"][0].tolist()

        if q_input_ids.shape[1] <= 2:
            s_ids = spheroid_tokenizer.encode(text, add_special_tokens=False)
            return s_ids, [1.0] * len(s_ids)

        # 2. Compute Qwen next-token weights
        q_weights = self.compute_teacher_token_weights(q_input_ids)[0].cpu().numpy()
        del q_enc
        del q_input_ids

        # 3. Construct Character-Level Importance Array
        char_weights = np.ones(len(text), dtype=np.float32)

        # q_weights[t] is the weight for predicting token t+1 at character span q_offsets[t+1]
        for t_idx, (c_start, c_end) in enumerate(q_offsets[1:]):
            if c_end > c_start and c_start < len(char_weights):
                c_end = min(c_end, len(char_weights))
                char_weights[c_start:c_end] = float(q_weights[t_idx])

        # 4. Tokenize with Spheroid Tokenizer with Character Offsets
        s_enc = spheroid_tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        s_input_ids = s_enc["input_ids"]
        s_offsets = s_enc["offset_mapping"]

        # 5. Map Character Weights to Spheroid Tokens
        s_weights = []
        for c_start, c_end in s_offsets:
            if c_end > c_start:
                span_w = char_weights[c_start:c_end]
                s_weights.append(float(np.mean(span_w)))
            else:
                s_weights.append(1.0)

        # Ensure bounded weights
        s_weights = [min(max(w, self.w_min), self.w_max) for w in s_weights]
        if s_weights:
            avg_w = sum(s_weights) / len(s_weights)
            if avg_w > 1e-4:
                s_weights = [w / avg_w for w in s_weights]

        return s_input_ids, s_weights

    def audit_sycophancy(
        self,
        prompt_neutral: str,
        prompt_biased: str,
        ground_truth: str,
        false_claim: str,
    ) -> Dict[str, Any]:
        """
        Audits a prompt pair for Sycophancy (Over-Agreement) Shift.
        Measures whether the model's confidence in the ground truth degrades
        when presented with a user asserting a false claim.

        Returns:
            Dictionary containing log-probabilities, sycophancy shift delta,
            and whether the model is susceptible to the biased premise.
        """
        if not self.loaded:
            self.load()

        def _get_logprob(prompt: str, target: str) -> float:
            full_text = prompt.rstrip() + " " + target.lstrip()
            enc = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            prompt_len = self.tokenizer(prompt.rstrip(), return_tensors="pt")["input_ids"].shape[1]
            target_ids = enc["input_ids"][:, prompt_len:]

            if target_ids.shape[1] == 0:
                return -100.0

            with torch.no_grad():
                logits = self.model(**enc).logits[:, prompt_len - 1 : -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                token_lps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
                return float(token_lps.sum().item())

        lp_truth_neutral = _get_logprob(prompt_neutral, ground_truth)
        lp_truth_biased = _get_logprob(prompt_biased, ground_truth)
        lp_false_neutral = _get_logprob(prompt_neutral, false_claim)
        lp_false_biased = _get_logprob(prompt_biased, false_claim)

        sycophancy_shift = lp_truth_neutral - lp_truth_biased
        is_sycophantic = (lp_false_biased > lp_truth_biased)

        return {
            "prompt_neutral": prompt_neutral,
            "prompt_biased": prompt_biased,
            "ground_truth": ground_truth,
            "false_claim": false_claim,
            "logprob_truth_neutral": lp_truth_neutral,
            "logprob_truth_biased": lp_truth_biased,
            "logprob_false_neutral": lp_false_neutral,
            "logprob_false_biased": lp_false_biased,
            "sycophancy_shift_delta": sycophancy_shift,
            "is_sycophantic": is_sycophantic,
            "resistance_score": max(0.0, min(100.0, 100.0 - max(0.0, sycophancy_shift) * 10.0)),
        }

    def compute_counter_sycophancy_weights(
        self,
        prompt: str,
        response: str,
        spheroid_tokenizer,
        flattery_phrases: Optional[List[str]] = None,
    ) -> Tuple[List[int], List[float]]:
        """
        Generates token-level training loss weights that systematically penalize
        sycophantic agreeing fillers and prompt parroting while rewarding objective
        factual correction tokens.
        """
        full_text = prompt.rstrip() + "\n" + response.lstrip()
        s_ids, s_weights = self.digest_text(full_text, spheroid_tokenizer)

        if flattery_phrases is None:
            flattery_phrases = [
                "you are right", "you're right", "you make a great point",
                "as you said", "as you mentioned", "absolutely correct",
                "certainly! you", "indeed, as you", "that is a brilliant",
                "i completely agree", "i agree with your", "as you pointed out",
            ]

        response_lower = response.lower()
        flattery_char_mask = np.zeros(len(full_text), dtype=bool)
        resp_start_idx = full_text.find(response)
        if resp_start_idx != -1:
            for phrase in flattery_phrases:
                p_idx = 0
                while True:
                    found = response_lower.find(phrase, p_idx)
                    if found == -1:
                        break
                    start_char = resp_start_idx + found
                    end_char = start_char + len(phrase)
                    flattery_char_mask[start_char:end_char] = True
                    p_idx = found + len(phrase)

        # Tokenize with offsets to map character flattery mask onto Spheroid tokens
        enc = spheroid_tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=2048,
        )
        s_offsets = enc["offset_mapping"]

        adjusted_weights = []
        for w, (c_s, c_e) in zip(s_weights, s_offsets):
            if c_e > c_s and flattery_char_mask[c_s:c_e].any():
                # Suppress sycophantic flattery tokens to minimum floor (0.05)
                adjusted_weights.append(self.w_min * 0.25)
            else:
                adjusted_weights.append(w)

        # Normalize mean to 1.0
        if adjusted_weights:
            mean_w = sum(adjusted_weights) / len(adjusted_weights)
            adjusted_weights = [min(max(w / mean_w, self.w_min), self.w_max) for w in adjusted_weights]

        return s_ids, adjusted_weights


class HeuristicDigester:
    """
    Lightweight, deterministic information-density digester.
    Used for local testing or when frontier teacher weights are not loaded.
    Scores tokens based on subword entropy, syntax significance,
    mathematical symbols, numbers, and structural reasoning blocks.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        w_min: float = 0.2,
        w_max: float = 4.0,
    ):
        self.alpha = alpha
        self.w_min = w_min
        self.w_max = w_max

        # Common low-information stopwords / punctuation
        self.low_info_words = {
            "the", "is", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
            "at", "by", "from", "as", "that", "this", "it", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "but", "if",
            "then", "so", ",", ".", ";", ":", "'", '"', "(", ")", "[", "]", "{", "}"
        }

    def digest_text(
        self,
        text: str,
        spheroid_tokenizer,
        max_length: int = 2048,
    ) -> Tuple[List[int], List[float]]:
        enc = spheroid_tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        s_input_ids = enc["input_ids"]
        s_offsets = enc["offset_mapping"]

        weights = []
        for idx, (c_start, c_end) in enumerate(s_offsets):
            subword = text[c_start:c_end].strip().lower()
            if not subword:
                weights.append(0.5)
            elif subword in self.low_info_words:
                weights.append(0.35)
            elif any(c.isdigit() for c in subword):
                weights.append(2.5)  # Numbers / constants
            elif any(c in "+-*/=^<>\\_$%&|~" for c in subword):
                weights.append(3.0)  # Math operators
            elif len(subword) > 7:
                weights.append(2.0)  # Rich vocabulary / technical terms
            else:
                weights.append(1.0)

        # Normalize sequence mean to 1.0
        if weights:
            mean_w = sum(weights) / len(weights)
            weights = [min(max(w / mean_w, self.w_min), self.w_max) for w in weights]

        return s_input_ids, weights


def create_digester(
    teacher_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    mode: str = "online",
    alpha: float = 1.5,
    w_min: float = 0.2,
    w_max: float = 4.0,
) -> Union[QwenTeacherDigester, HeuristicDigester]:
    """Factory function for instantiating the appropriate digester engine."""
    if mode == "heuristic":
        return HeuristicDigester(alpha=alpha, w_min=w_min, w_max=w_max)
    else:
        return QwenTeacherDigester(
            model_id=teacher_model,
            device=device,
            alpha=alpha,
            w_min=w_min,
            w_max=w_max,
            lazy_load=True,
        )


def load_spheroid_tokenizer(tokenizer_dir: str = "data/tokenizer"):
    """Convenience helper to load the Spheroid SentencePiece tokenizer."""
    from src.corpus import build_tokenizer
    return build_tokenizer(tokenizer_dir)

