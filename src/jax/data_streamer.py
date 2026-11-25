"""
CerebroSpheroid V5: High-Throughput Linguistic Data Streamer & Queue
====================================================================

Streams real structured linguistic sequences into TPU MXU buffers:
1. First Priority: Local cognitive datasets (e.g. data/cerebro_cognitive_8k/*.parquet).
2. Second Priority: Hugging Face streaming datasets (e.g. TinyStories / FineWeb-Edu / wikitext).
3. Third Priority: Deterministic structured linguistic Markov-syntactic generator.

Maintains an asynchronous background prefetch queue so TPU systolic units
operate at maximum sustained tok/s without I/O or tokenization stalls.
"""

from __future__ import annotations

import glob
import json
import os
import queue
import random
import threading
import time
from typing import Generator, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from src.corpus import format_stimulus_prompt, format_motor_response


class LinguisticDataStreamer:
    """Multi-source continuous text token streamer for pure JAX training."""

    def __init__(
        self,
        batch_size: int = 32,
        prompt_len: int = 256,
        target_len: int = 512,
        vocab_size: int = 32000,
        data_dir: Optional[str] = None,
        queue_size: int = 8,
        target_pad_id: int = -100,
        allow_synthetic_fallback: Optional[bool] = None,
    ):
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.target_len = target_len
        self.total_seq_len = prompt_len + target_len
        self.vocab_size = vocab_size
        self.target_pad_id = target_pad_id
        self.allow_synthetic_fallback = (
            (data_dir == "__nonexistent__") if allow_synthetic_fallback is None else bool(allow_synthetic_fallback)
        )
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.running = True
        self.fatal_error: Optional[Exception] = None

        # Locate data directories
        if data_dir is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.abspath(os.path.join(curr_dir, "..", "..", "data"))
        else:
            self.data_dir = data_dir

        self.tokenizer = self._init_tokenizer()
        self.pairs_corpus: List[Tuple[str, str]] = []
        self._load_local_data()

        # Launch background prefetch worker
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _validate_tokenizer_vocab(self, tok: Any) -> None:
        """Validates tokenizer vocabulary size and special token IDs against model vocab_size."""
        if hasattr(tok, "__len__") and len(tok) > self.vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary ({len(tok)}) exceeds model vocab_size ({self.vocab_size})."
            )
        for attr in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
            tid = getattr(tok, attr, None)
            if tid is not None and not (0 <= int(tid) < self.vocab_size):
                raise ValueError(
                    f"Tokenizer {attr}={tid} is out of bounds for model vocab_size={self.vocab_size}."
                )

    def _init_tokenizer(self):
        """Loads 32,000-vocabulary LLaMA tokenizer matching pretrained embedding matrix."""
        if self.vocab_size < 256 and self.allow_synthetic_fallback:
            return None

        # 1. First priority: Local bundled LLaMA 32k tokenizer in data/tokenizer
        local_tok_dir = os.path.join(self.data_dir, "tokenizer")
        if os.path.exists(local_tok_dir) and HAS_TRANSFORMERS:
            tok = AutoTokenizer.from_pretrained(local_tok_dir, local_files_only=True)
            if tok.pad_token_id is None:
                tok.pad_token_id = 0
            self._validate_tokenizer_vocab(tok)
            print(f"🔤 [DataStreamer] Loaded local LLaMA-32k tokenizer from {local_tok_dir} (vocab: {len(tok)})")
            return tok

        # 2. Second priority: Remote LLaMA candidates (strictly 32k vocab, never 50k GPT-2)
        if HAS_TRANSFORMERS:
            for cand in ["huggyllama/llama-7b", "openlm-research/open_llama_7b", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]:
                try:
                    tok = AutoTokenizer.from_pretrained(cand, local_files_only=False)
                    if tok.pad_token_id is None:
                        tok.pad_token_id = 0
                    self._validate_tokenizer_vocab(tok)
                    print(f"🔤 [DataStreamer] Loaded remote LLaMA-32k tokenizer: {cand} (vocab: {len(tok)})")
                    return tok
                except ValueError:
                    raise
                except Exception:
                    continue
        return None

    def _encode_text(self, text: str) -> List[int]:
        """Encodes text to vocabulary IDs strictly within [0, vocab_size - 1]."""
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            for t in tokens:
                if t < 0 or t >= self.vocab_size:
                    raise ValueError(
                        f"Tokenizer emitted out-of-range token ID {t} for vocab_size={self.vocab_size}. "
                        "Refusing to silently clamp IDs."
                    )
            return list(tokens)
        # Fallback to UTF-8 byte tokens with offset (only when no tokenizer is configured)
        return [(int(b) + 3) % self.vocab_size for b in text.encode("utf-8", errors="ignore")]

    def _load_local_data(self):
        """Discovers and caches local parquet / jsonl text files formatted with Strategy A bio-tokens."""
        if not os.path.exists(self.data_dir):
            if not self.allow_synthetic_fallback:
                raise FileNotFoundError(
                    f"Data directory '{self.data_dir}' does not exist and synthetic fallback is disabled."
                )
            return

        # 1. Parquet files
        parquet_files = glob.glob(os.path.join(self.data_dir, "**", "*.parquet"), recursive=True)
        for pf in parquet_files:
            if "benchmark" in pf.lower() or "cache" in pf.lower() or "tokenizer" in pf.lower():
                continue
            if HAS_PYARROW:
                try:
                    table = pq.read_table(pf)
                    cols = table.column_names
                    if "question" in cols and "answer" in cols:
                        questions = table["question"].to_pylist()
                        answers = table["answer"].to_pylist()
                        for q, a in zip(questions, answers):
                            q_str, a_str = str(q).strip(), str(a).strip()
                            if len(q_str) > 5 and len(a_str) > 5:
                                p = format_stimulus_prompt(q_str)
                                r = format_motor_response(a_str)
                                self.pairs_corpus.append((p, r))
                    elif "text" in cols:
                        texts = table["text"].to_pylist()
                        for t in texts:
                            t_str = str(t).strip()
                            if len(t_str) > 20:
                                if "\n\n" in t_str:
                                    parts = t_str.split("\n\n", 1)
                                    p = format_stimulus_prompt(parts[0])
                                    r = format_motor_response(parts[1])
                                else:
                                    p = format_stimulus_prompt("Explain and analyze the following text:")
                                    r = format_motor_response(t_str)
                                self.pairs_corpus.append((p, r))
                    else:
                        for col in cols:
                            if any(k in col.lower() for k in ("text", "answer", "prompt", "content")):
                                for x in table[col].to_pylist():
                                    s_val = str(x).strip()
                                    if len(s_val) > 20:
                                        p = format_stimulus_prompt("Analyze the following subject:")
                                        r = format_motor_response(s_val)
                                        self.pairs_corpus.append((p, r))
                                break
                except Exception as e:
                    raise RuntimeError(f"Failed to load local Parquet file '{pf}': {e}") from e

        # 2. JSONL files (e.g. diverse_sft_corpus_25k.jsonl, reasoning, science, math)
        jsonl_files = glob.glob(os.path.join(self.data_dir, "**", "*.jsonl"), recursive=True)
        for jf in jsonl_files:
            if "benchmark" in jf.lower() or "cache" in jf.lower() or "tokenizer" in jf.lower():
                continue
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        p_val = obj.get("prompt") or obj.get("question") or obj.get("input") or obj.get("stimulus")
                        r_val = obj.get("response") or obj.get("answer") or obj.get("output") or obj.get("articulation")
                        if p_val and r_val:
                            p_str = str(p_val).strip()
                            r_str = str(r_val).strip()
                            if len(p_str) > 2 and len(r_str) > 1:
                                p = format_stimulus_prompt(p_str)
                                r = format_motor_response(r_str)
                                self.pairs_corpus.append((p, r))
            except Exception as e:
                raise RuntimeError(f"Failed to load local JSONL file '{jf}': {e}") from e

        # 3. JSON files (e.g. cerebro_cognitive_8k/*.json)
        json_files = glob.glob(os.path.join(self.data_dir, "**", "*.json"), recursive=True)
        for jf in json_files:
            if "benchmark" in jf or "config" in jf or "tokenizer" in jf:
                continue
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
                for item in items:
                    if isinstance(item, dict):
                        p_val = item.get("prompt") or item.get("question") or item.get("input") or item.get("stimulus")
                        r_val = item.get("response") or item.get("answer") or item.get("output") or item.get("articulation")
                        if p_val and r_val:
                            p_str = str(p_val).strip()
                            r_str = str(r_val).strip()
                            if len(p_str) > 2 and len(r_str) > 1:
                                p = format_stimulus_prompt(p_str)
                                r = format_motor_response(r_str)
                                self.pairs_corpus.append((p, r))
            except Exception as e:
                raise RuntimeError(f"Failed to load local JSON file '{jf}': {e}") from e

        if self.pairs_corpus:
            random.shuffle(self.pairs_corpus)
            print(f"📚 [DataStreamer] Loaded {len(self.pairs_corpus):,} local multi-domain Strategy A (<stim>/<act>) pairs for pretraining!")
        elif not self.allow_synthetic_fallback:
            raise RuntimeError(
                f"No valid training pairs found in '{self.data_dir}' and synthetic fallback is disabled."
            )

    def _generate_synthetic_prose(self) -> Tuple[str, str]:
        """Emergency fallback: diverse cognitive reasoning questions without procedural arithmetic or static world trivia traps."""
        topics = [
            ("Explain the concept of entropy in thermodynamics and its relationship to microstates.",
             "Entropy is a measure of the molecular disorder or randomness within a system. In statistical thermodynamics, Boltzmann formulated entropy as S = k * ln(W), where k is the Boltzmann constant and W represents the number of microscopic configurations (microstates) corresponding to a thermodynamic macrostate. As a system naturally evolves toward equilibrium, spontaneous processes tend to maximize microstates, causing net entropy to increase in accordance with the Second Law of Thermodynamics."),
            ("Describe the mechanism of action of DNA polymerase during semi-conservative replication.",
             "DNA polymerase synthesizes new DNA strands by catalyzing the formation of phosphodiester bonds between the 3'-hydroxyl group of the growing strand and the 5'-phosphate group of an incoming deoxynucleotide triphosphate (dNTP). It operates strictly in the 5' to 3' direction, requiring a pre-existing RNA primer synthesized by primase. On the leading strand, synthesis is continuous, while on the lagging strand, discontinuous Okazaki fragments are formed, later joined by DNA ligase."),
            ("Derive the time complexity of the merge sort algorithm using the master theorem.",
             "Merge sort divides an array of size n into two halves of size n/2, recursively sorts both halves, and merges them in linear time O(n). The recurrence relation is T(n) = 2T(n/2) + O(n). Using the Master Theorem: a = 2, b = 2, and f(n) = n. Since log_b(a) = log_2(2) = 1, and f(n) = Theta(n^1), we fall under Case 2 of the Master Theorem. Therefore, the overall time complexity of merge sort is Theta(n * log(n)) in best, average, and worst cases."),
            ("Analyze how transformer self-attention computes contextual representations.",
             "Self-attention transforms an input sequence of vectors into queries (Q), keys (K), and values (V) via linear projections. The attention weights are calculated via scaled dot-product: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V. This allows every token in the sequence to dynamically attend to every other token based on pairwise relevance, enabling rich parallelized contextual representations without sequential recurrence bottlenecks."),
            ("What does `bool([0])` evaluate to in Python?", "In Python, `bool([0])` evaluates to `True` because any non-empty list has truth value True regardless of its element values."),
            ("Write a Python function to check if a number is even.", "def is_even(n: int) -> bool:\n    return n % 2 == 0")
        ]
        q, a = random.choice(topics)
        return format_stimulus_prompt(q), format_motor_response(a)

    def _worker_loop(self):
        """Continuous background prefetch loop packing Strategy A batches into queue."""
        doc_idx = 0

        try:
            while self.running:
                p_batch = []
                t_batch = []

                for _ in range(self.batch_size):
                    if self.pairs_corpus:
                        if doc_idx > 0 and (doc_idx % len(self.pairs_corpus) == 0):
                            random.shuffle(self.pairs_corpus)
                        p_txt, r_txt = self.pairs_corpus[doc_idx % len(self.pairs_corpus)]
                        doc_idx += 1
                    else:
                        p_txt, r_txt = self._generate_synthetic_prose()

                    # Strategy A: Encode <stim> ... </stim> prompt and <act> ... </act> response
                    p_ids = self._encode_text(p_txt)[:self.prompt_len]
                    r_ids = self._encode_text(r_txt)
                    eos_id = getattr(getattr(self, "tokenizer", None), "eos_token_id", None)
                    if eos_id is None:
                        eos_id = 2
                    if self.target_len > 0 and (not r_ids or r_ids[-1] != eos_id):
                        r_ids = r_ids[: max(0, self.target_len - 1)] + [eos_id]
                    r_ids = r_ids[:self.target_len]

                    # Right-pad prompts with 0 (pad_id) and targets with target_pad_id (-100 ignore index)
                    if len(p_ids) < self.prompt_len:
                        p_ids = p_ids + [0] * (self.prompt_len - len(p_ids))
                    if len(r_ids) < self.target_len:
                        r_ids = r_ids + [self.target_pad_id] * (self.target_len - len(r_ids))

                    p_batch.append(p_ids)
                    t_batch.append(r_ids)

                prompts = np.array(p_batch, dtype=np.int32)
                targets = np.array(t_batch, dtype=np.int32)
                self.queue.put((prompts, targets))
        except Exception as e:
            self.fatal_error = e
            self.running = False

    def get_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Retrieves next pre-tokenized batch ready for TPU MXU."""
        if self.fatal_error is not None:
            raise RuntimeError(f"LinguisticDataStreamer worker failed: {self.fatal_error}") from self.fatal_error
        prompts_np, targets_np = self.queue.get(timeout=30)
        return jnp.asarray(prompts_np), jnp.asarray(targets_np)

    def stop(self):
        self.running = False
