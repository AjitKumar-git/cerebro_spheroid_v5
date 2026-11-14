"""
Elastic Sliding Context Window & Complementary Bucket Harmonizer.
Framework-agnostic engine supporting PyTorch (GPU) and JAX / XLA (TPU).

Core Principles:
1. Fixed Total Sequence Budget: W_total = L_prompt + L_target is strictly invariant.
2. Complementary Inverse Length Buckets:
   - Deep Reasoning (CoT) : Short Question (64)     -> Massive Deliberation Output (W - 64)
   - Standard CoT         : Typical Question (256)   -> Long Derivation Output (W - 256)
   - Balanced             : Moderate Context (W//2)  -> Moderate Generation (W//2)
   - Long Context         : Long Input (W - 128)     -> Concise Answer (128)
3. Zero Hardware Recompilation:
   Fixed static shapes compile once under XLA / CUDA graphs, eliminating recompilation freezes.
4. Zero Padding Waste:
   Maximizes useful token throughput and compute density across heterogeneous datasets.
"""

from typing import Dict, List, Tuple, Optional, Any, Iterator
import collections
import queue
import random
import threading
import time
import numpy as np

from src.corpus import encode_no_special


def extract_prompt_response_pair(row: Any) -> Tuple[str, str]:
    """Extract `(prompt, response)` strings from HF/Parquet/tuple dataset records."""
    if isinstance(row, (tuple, list)):
        if len(row) == 3:
            return str(row[1]), str(row[2])
        elif len(row) >= 2:
            return str(row[0]), str(row[1])
        return "", ""

    if isinstance(row, dict):
        # 0. CerebroCognitive schema
        if "stimulus" in row and "articulation" in row:
            q_txt = str(row.get("stimulus") or "").strip()
            r_txt = str(row.get("articulation") or "").strip()
            if q_txt and r_txt:
                return q_txt, r_txt

        # 1. OpenHermes / ShareGPT format
        convs = row.get("conversations", [])
        if convs:
            q_txt, r_txt = "", ""
            sys_txt = ""
            for turn in convs:
                sender = str(turn.get("from", "")).lower()
                val = turn.get("value", "")
                if "system" in sender and not sys_txt:
                    sys_txt = str(val).strip()
                elif ("human" in sender or "user" in sender) and not q_txt:
                    q_txt = str(val).strip()
                elif ("gpt" in sender or "assistant" in sender or "model" in sender) and not r_txt:
                    r_txt = str(val).strip()
            if sys_txt and q_txt:
                q_txt = f"{sys_txt}\n\n{q_txt}"
            if q_txt and r_txt:
                return str(q_txt), str(r_txt)

        # 2. SmolTalk / UltraChat / ChatML format
        msgs = row.get("messages", [])
        if msgs:
            q_txt, r_txt = "", ""
            sys_txt = ""
            for msg in msgs:
                role = str(msg.get("role", "")).lower()
                val = msg.get("content", "")
                if "system" in role and not sys_txt:
                    sys_txt = str(val).strip()
                elif ("user" in role or "human" in role) and not q_txt:
                    q_txt = str(val).strip()
                elif ("assistant" in role or "model" in role or "gpt" in role) and not r_txt:
                    r_txt = str(val).strip()
            if sys_txt and q_txt:
                q_txt = f"{sys_txt}\n\n{q_txt}"
            if q_txt and r_txt:
                return str(q_txt), str(r_txt)

        # 3. GSM8K / MetaMathQA / Q&A / Cosmopedia format
        q_txt = row.get("question", row.get("prompt", row.get("input", row.get("query", ""))))
        r_txt = row.get("answer", row.get("response", row.get("output", row.get("text", ""))))
        if q_txt and r_txt:
            return str(q_txt), str(r_txt)
        if not q_txt and "text" in row:
            raw_t = str(row["text"]).strip()
            if "\n\n" in raw_t:
                p, r = raw_t.split("\n\n", 1)
            else:
                p, r = "Explain and analyze the following topic:", raw_t
            return p.strip(), r.strip()
        return str(q_txt), str(r_txt)
    return "", ""


_DEFAULT_DATASET_REVISIONS: Dict[str, str] = {
    # OpenHermes-2.5's default branch holds a single 1.94 GB JSON array file,
    # causing datasets' JSON builder to buffer the full file + ujson AST in RAM
    # (~15+ GB per rank) and trigger host OOM SIGKILL on 2xT4 containers. The
    # auto-converted Parquet branch streams row-groups in <250 MB RSS per rank.
    "teknium/OpenHermes-2.5": "refs/convert/parquet",
}


def _get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face API token from standard environment variables."""
    import os
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _open_hf_stream(
    repo: str,
    config_name: Optional[str] = None,
    split: str = "train",
    rank: int = 0,
    world_size: int = 1,
    revision: Optional[str] = None,
) -> Iterator[Any]:
    """Open a HuggingFace streaming dataset with optional multi-rank sharding and auth token."""
    import os

    # Redirect legacy script-based gsm8k repo to official Parquet dataset to prevent 429 rate limit stalls
    if repo == "gsm8k":
        repo = "openai/gsm8k"

    if "HF_HOME" not in os.environ:
        default_cache = os.path.join(os.path.expanduser("~"), ".cache")
        if not os.access(default_cache, os.W_OK):
            os.environ["HF_HOME"] = "/tmp/hf_cache"

    from datasets import load_dataset

    effective_revision = revision if revision is not None else _DEFAULT_DATASET_REVISIONS.get(repo)
    kwargs: Dict[str, Any] = {"split": split, "streaming": True}
    if effective_revision is not None:
        kwargs["revision"] = effective_revision

    token = _get_hf_token()
    if token:
        kwargs["token"] = token

    try:
        ds = load_dataset(repo, config_name, **kwargs) if config_name else load_dataset(repo, **kwargs)
    except TypeError:
        # Backward compatibility for older datasets library versions lacking token or revision kwargs
        if "revision" in kwargs:
            kwargs.pop("revision", None)
        if "token" in kwargs and TypeError:
            kwargs.pop("token", None)
        ds = load_dataset(repo, config_name, **kwargs) if config_name else load_dataset(repo, **kwargs)
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
            print(f"⚠️ [DataStreamer] HuggingFace Rate Limit (429) on {repo}. Recommend setting HF_TOKEN.")
        raise

    if world_size > 1:
        try:
            from datasets.distributed import split_dataset_by_node
            ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
        except Exception:
            if rank > 0 and hasattr(ds, "skip"):
                ds = ds.skip(rank * 256)
    return iter(ds)


def build_online_dataset_iterator(
    dataset_source: str = "dense_edu",
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[Dict[str, Any]]:
    """
    Infinite, resilient HuggingFace multi-corpus streaming iterator shared by TPU and GPU trainers.
    Supports 'dense_edu'/'foundation', 'reasoning'/'reasoning_anneal', 'conversation_sft'/'agentic',
    'code', and 'consolidation'/'balanced_mix'.
    """
    source = (dataset_source or "dense_edu").lower()
    if source in ("foundation", "auto", "basic", "fineweb_edu"):
        source = "dense_edu"
    elif source in ("reasoning",):
        source = "reasoning_anneal"
    elif source in ("agentic", "workflows", "chat_sft"):
        source = "conversation_sft"
    elif source in ("balanced_mix", "balanced", "gold"):
        source = "consolidation"

    def _next_or_reopen(it: Optional[Iterator[Any]], opener):
        if it is None:
            it = opener()
        try:
            return next(it), it
        except StopIteration:
            it = opener()
            return next(it), it

    if source == "dense_edu":
        cosmo_it: Optional[Iterator[Any]] = None
        fw_it: Optional[Iterator[Any]] = None
        oh_it: Optional[Iterator[Any]] = None

        def _dense_gen():
            nonlocal cosmo_it, fw_it, oh_it
            turn = 0
            while True:
                mod5 = turn % 5
                try:
                    if mod5 in (0, 2):
                        row, cosmo_it = _next_or_reopen(
                            cosmo_it, lambda: _open_hf_stream("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", rank=rank, world_size=world_size)
                        )
                        p = (row.get("prompt") or "Explain the educational context of this topic:").strip()
                        r = (row.get("text") or "").strip()
                        if len(r) >= 30:
                            yield {"conversations": [{"from": "human", "value": p}, {"from": "gpt", "value": r}]}
                    elif mod5 in (1, 4):
                        for _ in range(20):
                            row, fw_it = _next_or_reopen(
                                fw_it, lambda: _open_hf_stream("HuggingFaceFW/fineweb-edu", "sample-10BT", rank=rank, world_size=world_size)
                            )
                            if float(row.get("score", 0.0) or row.get("educational_score", 0.0)) >= 4.0:
                                text = (row.get("text") or "").strip()
                                if len(text) >= 50:
                                    p, r = text.split("\n\n", 1) if "\n\n" in text else ("Explain the concept:", text)
                                    yield {"conversations": [{"from": "human", "value": p.strip()}, {"from": "gpt", "value": r.strip()}]}
                                    break
                    else:
                        row, oh_it = _next_or_reopen(
                            oh_it, lambda: _open_hf_stream("teknium/OpenHermes-2.5", rank=rank, world_size=world_size)
                        )
                        yield row
                except Exception:
                    time.sleep(0.5)
                turn += 1

        return _dense_gen()

    if source in ("conversation_sft", "code"):
        smol_it: Optional[Iterator[Any]] = None
        cosmo_it: Optional[Iterator[Any]] = None
        oh_it: Optional[Iterator[Any]] = None

        def _conv_gen():
            nonlocal smol_it, cosmo_it, oh_it
            turn = 0
            while True:
                mod10 = turn % 10
                try:
                    if mod10 < 5:
                        row, smol_it = _next_or_reopen(
                            smol_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "all", rank=rank, world_size=world_size)
                        )
                        yield row
                    elif mod10 < 8 and source != "code":
                        row, cosmo_it = _next_or_reopen(
                            cosmo_it, lambda: _open_hf_stream("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", rank=rank, world_size=world_size)
                        )
                        p = (row.get("prompt") or "Explain the concept:").strip()
                        r = (row.get("text") or "").strip()
                        yield {"conversations": [{"from": "human", "value": p}, {"from": "gpt", "value": r}]}
                    else:
                        row, oh_it = _next_or_reopen(
                            oh_it, lambda: _open_hf_stream("teknium/OpenHermes-2.5", rank=rank, world_size=world_size)
                        )
                        yield row
                except Exception:
                    time.sleep(0.5)
                turn += 1

        return _conv_gen()

    if source == "reasoning_anneal":
        math_it: Optional[Iterator[Any]] = None
        tool_it: Optional[Iterator[Any]] = None
        code_it: Optional[Iterator[Any]] = None
        logic_it: Optional[Iterator[Any]] = None
        know_it: Optional[Iterator[Any]] = None

        def _reason_gen():
            nonlocal math_it, tool_it, code_it, logic_it, know_it
            turn = 0
            while True:
                mod5 = turn % 5
                try:
                    if mod5 == 0:
                        # 1. Pure Mathematical Calculation & Chain-of-Thought (MetaMathQA - 395k clean proofs)
                        row, math_it = _next_or_reopen(
                            math_it, lambda: _open_hf_stream("meta-math/MetaMathQA", rank=rank, world_size=world_size)
                        )
                        yield row
                    elif mod5 == 1:
                        # 2. Advanced Tool Calling & Function Dispatch (SmolTalk apigen-80k)
                        row, tool_it = _next_or_reopen(
                            tool_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "apigen-80k", rank=rank, world_size=world_size)
                        )
                        yield row
                    elif mod5 == 2:
                        # 3. Python-Edu, Software Algorithms & Problem Solving (SmolTalk self-oss-instruct)
                        row, code_it = _next_or_reopen(
                            code_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "self-oss-instruct", rank=rank, world_size=world_size)
                        )
                        yield row
                    elif mod5 == 3:
                        # 4. Multi-Turn Deductive Reasoning & Logic (SmolTalk smol-magpie-ultra)
                        row, logic_it = _next_or_reopen(
                            logic_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "smol-magpie-ultra", rank=rank, world_size=world_size)
                        )
                        yield row
                    else:
                        # 5. World Knowledge Replay Anchor (FineWeb-Edu) - Prevents Catastrophic Forgetting!
                        row, know_it = _next_or_reopen(
                            know_it, lambda: _open_hf_stream("HuggingFaceFW/fineweb-edu", "sample-10BT", rank=rank, world_size=world_size)
                        )
                        yield row
                except Exception:
                    try:
                        import json
                        base_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cerebro_cognitive_8k", "cerebro_cognitive_10k.jsonl")
                        if os.path.exists(base_data):
                            with open(base_data, "r", encoding="utf-8") as f:
                                for line in f:
                                    if line.strip():
                                        d = json.loads(line)
                                        yield {"conversations": [{"from": "human", "value": d.get("question", "")}, {"from": "gpt", "value": d.get("answer", "")}]}
                    except Exception:
                        pass
                    time.sleep(0.5)
                turn += 1

        return _reason_gen()

    if source == "consolidation":
        dense_it: Optional[Iterator[Any]] = None
        reason_it: Optional[Iterator[Any]] = None
        conv_it: Optional[Iterator[Any]] = None

        def _consol_gen():
            nonlocal dense_it, reason_it, conv_it
            turn = 0
            while True:
                mod4 = turn % 4
                try:
                    if mod4 in (0, 1):
                        if dense_it is None:
                            dense_it = build_online_dataset_iterator("dense_edu", rank=rank, world_size=world_size)
                        yield next(dense_it)
                    elif mod4 == 2:
                        if reason_it is None:
                            reason_it = build_online_dataset_iterator("reasoning_anneal", rank=rank, world_size=world_size)
                        yield next(reason_it)
                    else:
                        if conv_it is None:
                            conv_it = build_online_dataset_iterator("conversation_sft", rank=rank, world_size=world_size)
                        yield next(conv_it)
                except Exception:
                    time.sleep(0.5)
                turn += 1

        return _consol_gen()

    # Fallback: OpenHermes-2.5
    oh_it: Optional[Iterator[Any]] = None

    def _oh_gen():
        nonlocal oh_it
        while True:
            try:
                row, oh_it = _next_or_reopen(
                    oh_it, lambda: _open_hf_stream("teknium/OpenHermes-2.5", rank=rank, world_size=world_size)
                )
                yield row
            except Exception:
                time.sleep(0.5)

    return _oh_gen()


def sample_window(
    answer_ids: List[int],
    window: int,
    rng: Optional[random.Random] = None,
    tail_probability: float = 0.25,
) -> Tuple[Optional[int], List[int]]:
    """Picks the scored slice of one answer and the token heard before it (V4 window algorithm).

    Returns `(context_token, scored_tokens)`. The context token is fed to the auditory channel
    at step 0 and never scored, so a mid-answer window is conditioned on real text instead of
    on silence. `tail_probability` of windows are the final `window` tokens, which keeps
    end-of-text and EOS in the objective even though most answers are longer than a window.
    """
    if rng is None:
        rng = random.Random()
    if len(answer_ids) <= window:
        return None, list(answer_ids)
    max_start = len(answer_ids) - window - 1
    if max_start <= 0:
        start = 0
    elif rng.random() < tail_probability:
        start = max_start
    else:
        start = rng.randint(0, max_start)
    return answer_ids[start], answer_ids[start + 1 : start + 1 + window]


def get_default_elastic_buckets(total_window: int = 1024) -> List[Tuple[str, int, int]]:
    """Returns high-density complementary sliding window buckets avoiding extreme padding traps."""
    assert total_window >= 256, "total_window must be at least 256 tokens"
    if total_window >= 1024:
        return [
            ("short_cot", 256, 256),
            ("standard_cot", 256, 384),
            ("balanced", 384, 384),
            ("deep_reasoning", 256, 512),
            ("long_context", 768, 256),
        ]
    elif total_window >= 768:
        return [
            ("short_cot", 128, 256),
            ("standard_cot", 256, 256),
            ("deep_reasoning", 256, 512),
            ("long_context", 512, 256),
        ]
    elif total_window >= 512:
        return [
            ("short_cot", 128, 128),
            ("standard_cot", 128, 256),
            ("balanced", 256, 256),
            ("long_context", 384, 128),
        ]
    else:
        half = total_window // 2
        return [("balanced", half, half)]


def select_best_bucket(p_len: int, t_len: int, buckets: List[Tuple[str, int, int]]) -> int:
    """Selects the bucket that minimizes wasted padding while preserving target reasoning tokens."""
    best_idx = min(2, len(buckets) - 1)
    min_penalty = float("inf")

    for idx, (_, bp, bt) in enumerate(buckets):
        if p_len <= bp and t_len <= bt:
            # Fits completely: penalty is wasted padding
            penalty = (bp - p_len) + (bt - t_len)
        elif p_len <= bp:
            # Fits prompt, mild target truncation
            penalty = (bp - p_len) + (t_len - bt) * 3
        else:
            # Prompt exceeds bucket: penalize both prompt and target truncation
            penalty = (p_len - bp) * 5 + max(0, t_len - bt) * 3

        if penalty < min_penalty:
            min_penalty = penalty
            best_idx = idx

    return best_idx


class ElasticBucketBatcher:
    """
    Asynchronous / Streaming Bucket Batcher for Elastic Context Windows.
    Buffers incoming dataset records into complementary (L_p, L_t) queues
    and yields batches of exact fixed shapes for GPU and TPU execution.
    """

    def __init__(
        self,
        tokenizer: Any,
        data_iterator: Iterator[Any],
        batch_size: int = 16,
        total_window: int = 1024,
        buckets: Optional[List[Tuple[str, int, int]]] = None,
        framework: str = "jax",  # 'jax' or 'torch'
        target_pad_id: int = -100,
        return_feedback: bool = False,
        rng: Optional[random.Random] = None,
        tail_probability: float = 0.25,
    ):
        self.tokenizer = tokenizer
        self.data_iter = data_iterator
        self.batch_size = batch_size
        self.total_window = total_window
        self.buckets = buckets if buckets is not None else get_default_elastic_buckets(total_window)
        self.framework = framework.lower()
        self.target_pad_id = target_pad_id
        self.return_feedback = return_feedback
        self.rng = rng or random.Random(42)
        self.tail_probability = tail_probability
        self.max_queue_size = max(batch_size * 4, 128)

        self.pad_id = tokenizer.pad_token_id if (hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None) else 0
        # Queues per bucket index with bounded capacity
        self.queues: Dict[int, collections.deque] = {i: collections.deque(maxlen=self.max_queue_size) for i in range(len(self.buckets))}

    def _extract_text_pair(self, row: Any) -> Tuple[str, str]:
        """Extracts (prompt, response) from diverse dataset formats via shared extractor."""
        return extract_prompt_response_pair(row)

    def _format_batch(self, b_idx: int, name: str, p_list: list, t_list: list, c_list: Optional[list] = None):
        prompts_np = np.asarray(p_list, dtype=np.int32)
        targets_np = np.asarray(t_list, dtype=np.int32)
        feedback_np = None
        if self.return_feedback:
            feedback_np = np.full(targets_np.shape, self.target_pad_id, dtype=np.int32)
            if targets_np.shape[1] > 1:
                feedback_np[:, 1:] = targets_np[:, :-1]
            if c_list is not None:
                for i, c in enumerate(c_list):
                    if c is not None:
                        feedback_np[i, 0] = c

        if self.framework == "torch":
            import torch
            prompts = torch.from_numpy(prompts_np).long()
            targets = torch.from_numpy(targets_np).long()
            if feedback_np is not None:
                return b_idx, name, prompts, targets, torch.from_numpy(feedback_np).long()
            return b_idx, name, prompts, targets
        if self.framework == "jax":
            import jax.numpy as jnp
            prompts = jnp.asarray(prompts_np)
            targets = jnp.asarray(targets_np)
            if feedback_np is not None:
                return b_idx, name, prompts, targets, jnp.asarray(feedback_np)
            return b_idx, name, prompts, targets
        if feedback_np is not None:
            return b_idx, name, prompts_np, targets_np, feedback_np
        return b_idx, name, prompts_np, targets_np

    def _pull_record(self) -> Any:
        if hasattr(self.data_iter, "__next__"):
            return next(self.data_iter)
        elif hasattr(self.data_iter, "next"):
            return self.data_iter.next()
        elif callable(self.data_iter):
            return self.data_iter()
        else:
            raise TypeError("data_iterator is not iterable or callable")

    def get_batch(self) -> Any:
        return self.next_batch()

    def next_batch(self) -> Any:
        """
        Pulls from dataset stream until any bucket has batch_size examples.
        Returns: (bucket_idx, bucket_name, prompt_batch, target_batch[, feedback_batch])
        """
        while True:
            # Check if any bucket queue is ready to pop a full batch
            for b_idx in range(len(self.buckets)):
                if len(self.queues[b_idx]) >= self.batch_size:
                    name, p_len, t_len = self.buckets[b_idx]
                    batch_items = [self.queues[b_idx].popleft() for _ in range(self.batch_size)]
                    p_list, t_list, c_list = zip(*batch_items)
                    return self._format_batch(b_idx, name, p_list, t_list, c_list)

            # Pull raw records (batch-tokenize when a fast Rust tokenizer is available)
            use_fast_batch = bool(getattr(self.tokenizer, "is_fast", False) and callable(self.tokenizer))
            raw_pull_target = min(32, max(8, self.batch_size // 2)) if use_fast_batch else 1
            raw_pairs = []
            stream_ended = False

            while len(raw_pairs) < raw_pull_target:
                try:
                    row = self._pull_record()
                except (StopIteration, RuntimeError):
                    stream_ended = True
                    break

                q_txt, r_txt = self._extract_text_pair(row)
                if not q_txt or not r_txt:
                    continue

                # Format with enclosed biological delimiters: <stim> ... </stim> and <act> ... </act>
                try:
                    from src.corpus import format_stimulus_prompt, format_motor_response
                    q_txt = format_stimulus_prompt(q_txt)
                    r_txt = format_motor_response(r_txt)
                except Exception:
                    if not q_txt.startswith("<stim>"):
                        q_txt = f"<stim> {q_txt} </stim>"
                    if not r_txt.startswith("<act>"):
                        r_txt = f"<act> {r_txt} </act>"
                raw_pairs.append((q_txt, r_txt))

            if not raw_pairs and stream_ended:
                # If stream exhausted, flush whatever bucket has the most items
                best_b = max(self.queues.keys(), key=lambda k: len(self.queues[k]))
                if len(self.queues[best_b]) == 0:
                    raise StopIteration("Data stream exhausted.")
                items = list(self.queues[best_b])
                self.queues[best_b].clear()
                while len(items) < self.batch_size:
                    items.append(items[0])
                name, p_len, t_len = self.buckets[best_b]
                p_list, t_list, c_list = zip(*items)
                return self._format_batch(best_b, name, p_list, t_list, c_list)

            # Tokenize bounded to prevent processing gigantic out-of-distribution text
            max_enc = self.total_window + 256
            q_texts = [pair[0] for pair in raw_pairs]
            r_texts = [pair[1] for pair in raw_pairs]
            tokenized_pairs = None
            if use_fast_batch and len(raw_pairs) > 1:
                try:
                    q_ids_batch = self.tokenizer(
                        q_texts, max_length=max_enc, truncation=True, add_special_tokens=False
                    )["input_ids"]
                    r_ids_batch = self.tokenizer(
                        r_texts, max_length=max_enc, truncation=True, add_special_tokens=False
                    )["input_ids"]
                    tokenized_pairs = list(zip(q_ids_batch, r_ids_batch))
                except Exception:
                    tokenized_pairs = None

            if tokenized_pairs is None:
                tokenized_pairs = []
                for q_txt, r_txt in raw_pairs:
                    if hasattr(self.tokenizer, "encode"):
                        q_ids = encode_no_special(self.tokenizer, q_txt, max_length=max_enc)
                        r_ids = encode_no_special(self.tokenizer, r_txt, max_length=max_enc)
                    else:
                        q_ids = [1] * len(q_txt.split())
                        r_ids = [1] * len(r_txt.split())
                    tokenized_pairs.append((q_ids, r_ids))

            eos_id = self.tokenizer.eos_token_id if (hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None) else 2
            for q_ids, r_ids in tokenized_pairs:
                if len(r_ids) == 0 or r_ids[-1] != eos_id:
                    r_ids = r_ids + [eos_id]

                q_len, r_len = len(q_ids), len(r_ids)

                # Route to best complementary bucket
                b_idx = select_best_bucket(q_len, r_len, self.buckets)
                name, bp, bt = self.buckets[b_idx]

                # Pad / truncate to the exact bucket dimensions
                if len(q_ids) > bp:
                    q_padded = q_ids[:bp]
                else:
                    q_padded = q_ids + [self.pad_id] * (bp - len(q_ids))

                if len(r_ids) > bt:
                    context_token, scored = sample_window(r_ids, bt, self.rng, self.tail_probability)
                    r_padded = scored
                else:
                    context_token = None
                    r_padded = r_ids + [self.target_pad_id] * (bt - len(r_ids))

                self.queues[b_idx].append((q_padded, r_padded, context_token))


class PrefetchedBatcher:
    """
    Wraps an ElasticBucketBatcher with a background worker thread.
    Pre-tokenizes and buffers batches into a queue so the GPU / TPU
    never waits for Python CPU tokenization or network download latency.
    """

    def __init__(self, batcher: ElasticBucketBatcher, queue_size: int = 8):
        self.batcher = batcher
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.worker_thread.start()

    def _producer_loop(self):
        while not self.stop_event.is_set():
            try:
                batch = self.batcher.next_batch()
                self.queue.put(batch)
            except Exception:
                break

    def next_batch(self):
        return self.queue.get()

    def get_batch(self):
        return self.next_batch()

    def stop(self):
        self.shutdown()

    def shutdown(self):
        self.stop_event.set()

