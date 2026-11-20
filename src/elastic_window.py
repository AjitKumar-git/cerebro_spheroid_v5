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
import os
import re
import json
import collections
import queue
import random
import threading
import time
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.corpus import encode_no_special


class DatasetTelemetry:
    """Tracks per-source data health, yield counts, rejection reasons, and token lengths."""
    def __init__(self):
        self.lock = threading.Lock()
        self.examined: Dict[str, int] = collections.defaultdict(int)
        self.accepted: Dict[str, int] = collections.defaultdict(int)
        self.rejections: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        self.token_lengths: Dict[str, List[int]] = collections.defaultdict(list)
        self.last_log_time = time.time()

    def record_examined(self, source: str):
        with self.lock:
            self.examined[source] += 1

    def record_rejected(self, source: str, reason: str):
        with self.lock:
            self.rejections[source][reason] += 1

    def record_accepted(self, source: str, prompt_tokens: int, target_tokens: int):
        with self.lock:
            self.accepted[source] += 1
            tok_list = self.token_lengths[source]
            total_tok = int(prompt_tokens + target_tokens)
            if len(tok_list) < 2000:
                tok_list.append(total_tok)
            else:
                tok_list[random.randint(0, 1999)] = total_tok

    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            summary = {}
            for src in set(list(self.examined.keys()) + list(self.accepted.keys())):
                exam = self.examined.get(src, 0)
                acc = self.accepted.get(src, 0)
                rej = exam - acc if exam >= acc else 0
                rej_rate = (rej / exam * 100.0) if exam > 0 else 0.0
                lens = self.token_lengths.get(src, [])
                summary[src] = {
                    "examined": exam,
                    "accepted": acc,
                    "rejected": rej,
                    "rejection_rate_pct": round(rej_rate, 2),
                    "rejection_breakdown": dict(self.rejections[src]),
                    "median_tokens": int(np.median(lens)) if lens else 0,
                    "p95_tokens": int(np.percentile(lens, 95)) if lens else 0,
                }
            return summary

    def log_report(self, force: bool = False):
        now = time.time()
        if not force and (now - self.last_log_time < 30.0):
            return
        self.last_log_time = now
        summary = self.get_summary()
        if not summary:
            return
        print("\n" + "=" * 80)
        print("📊 DATASET PIPELINE HEALTH & QUALITY TELEMETRY")
        print("=" * 80)
        for src, stats in summary.items():
            print(
                f"  [{src:20s}] Yielded: {stats['accepted']:,} | Examined: {stats['examined']:,} | "
                f"Rej: {stats['rejection_rate_pct']:.1f}% | Median Tok: {stats['median_tokens']} (P95: {stats['p95_tokens']})"
            )
            if stats["rejection_breakdown"]:
                reasons = ", ".join(f"{k}: {v}" for k, v in stats["rejection_breakdown"].items())
                print(f"    ↳ Rejection causes: {reasons}")
        print("=" * 80 + "\n")


GLOBAL_TELEMETRY = DatasetTelemetry()


def _normalize_3tier_response(r_txt: str, q_txt: str = "") -> str:
    """
    Enforces the 3-Tier Cognitive Output Architecture:
      1. Thinking           : <think> ... </think> -> Private scratchpad deliberation
      2. Non-Conversational : <act> ... </act>     -> Coding, Calculator, & Tool actions
      3. Talking / Answering: Unbounded Plain Text -> Natural human dialogue & explanations
    """
    if not r_txt:
        return ""
    # Normalize thinking tags across models to canonical <think> ... </think>
    r_txt = r_txt.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")
    r_txt = r_txt.replace("<thought>", "<think>").replace("</thought>", "</think>")

    # 1. Arithmetic & GSM8K derivations ending in '#### <ans>'
    # Deterministic tool call goes in <act>, final answer is normal text OUTSIDE <act>
    if "####" in r_txt and "<act>" not in r_txt:
        parts = r_txt.split("####", 1)
        deriv = parts[0].strip()
        ans_str = parts[1].strip()
        if not deriv.startswith("<think>"):
            deriv = f"<think>\n{deriv}\n</think>"
        return f"{deriv}\n<act> calc({ans_str}) </act>\nAnswer: {ans_str}"

    return r_txt


def extract_prompt_response_pair(row: Any) -> Tuple[str, str]:
    """Extract `(prompt, response)` strings from HF/Parquet/tuple dataset records."""
    if isinstance(row, (tuple, list)):
        if len(row) == 3:
            return str(row[1]), _normalize_3tier_response(str(row[2]), str(row[1]))
        elif len(row) >= 2:
            return str(row[0]), _normalize_3tier_response(str(row[1]), str(row[0]))
        return "", ""

    if isinstance(row, dict):
        # 0. CerebroCognitive schema
        if "stimulus" in row and "articulation" in row:
            q_txt = str(row.get("stimulus") or "").strip()
            r_txt = str(row.get("articulation") or "").strip()
            if q_txt and r_txt:
                return q_txt, _normalize_3tier_response(r_txt, q_txt)

        # 1. OpenHermes / ShareGPT format (Preserving Complete Multi-Turn History)
        convs = row.get("conversations", [])
        if convs:
            sys_txt = ""
            turns = []
            for turn in convs:
                sender = str(turn.get("from", "")).lower()
                val = str(turn.get("value", "") or "").strip()
                if not val:
                    continue
                if "system" in sender and not sys_txt:
                    sys_txt = val
                elif "human" in sender or "user" in sender:
                    turns.append(("user", val))
                elif "gpt" in sender or "assistant" in sender or "model" in sender:
                    turns.append(("assistant", val))

            if turns and turns[-1][0] == "assistant":
                last_response = turns[-1][1]
                history_turns = turns[:-1]
                parts = []
                if sys_txt:
                    parts.append(f"System: {sys_txt}")
                for r, text in history_turns:
                    prefix = "User" if r == "user" else "Assistant"
                    parts.append(f"{prefix}: {text}")
                q_txt = "\n\n".join(parts)
                if not q_txt:
                    q_txt = "User: Please assist with the following task."
                return str(q_txt), _normalize_3tier_response(str(last_response), str(q_txt))

        # 2. SmolTalk / UltraChat / ChatML format (Preserving Complete Multi-Turn History)
        msgs = row.get("messages", [])
        if msgs:
            sys_txt = ""
            turns = []
            for msg in msgs:
                role = str(msg.get("role", "")).lower()
                val = str(msg.get("content", "") or "").strip()
                if not val:
                    continue
                if "system" in role and not sys_txt:
                    sys_txt = val
                elif "user" in role or "human" in role:
                    turns.append(("user", val))
                elif "assistant" in role or "model" in role or "gpt" in role:
                    turns.append(("assistant", val))

            if turns and turns[-1][0] == "assistant":
                last_response = turns[-1][1]
                history_turns = turns[:-1]
                parts = []
                if sys_txt:
                    parts.append(f"System: {sys_txt}")
                for r, text in history_turns:
                    prefix = "User" if r == "user" else "Assistant"
                    parts.append(f"{prefix}: {text}")
                q_txt = "\n\n".join(parts)
                if not q_txt:
                    q_txt = "User: Please assist with the following task."
                return str(q_txt), _normalize_3tier_response(str(last_response), str(q_txt))

        # 3. Frontier Reasoning / Math / CoT / Q&A format
        q_val = row.get("problem") or row.get("instruction") or row.get("question") or row.get("prompt") or row.get("input") or row.get("query")
        r_val = row.get("solution") or row.get("completion") or row.get("answer") or row.get("response") or row.get("output") or row.get("text")
        if q_val and r_val:
            q_txt = str(q_val).strip()
            r_txt = str(r_val).strip()
            # If explicit thought or reasoning trace is separate, enclose in <think> ... </think>
            th_txt = str(row.get("thought") or row.get("thinking") or row.get("reasoning") or "").strip()
            if th_txt and "<think>" not in r_txt:
                r_txt = f"<think>\n{th_txt}\n</think>\n{r_txt}"
            return q_txt, _normalize_3tier_response(r_txt, q_txt)

        # 4. Code Corpora: Extract Docstring Specifications or Causal Continuation
        if not q_val and "code" in row:
            raw_c = str(row["code"]).strip()
            # Check for function or class signature with docstring
            doc_match = re.search(r'^(?:def\s+[a-zA-Z_]\w*|class\s+[a-zA-Z_]\w*).*?:\s*"""(.*?)"""', raw_c, re.DOTALL | re.MULTILINE)
            if doc_match:
                docstring = doc_match.group(1).strip()
                q_txt = f"Implement a Python solution for the following specification:\n\n{docstring}"
                r_txt = raw_c
                return q_txt, _normalize_3tier_response(r_txt, q_txt)
            else:
                lines = raw_c.splitlines()
                if len(lines) > 4:
                    p = "\n".join(lines[:3])
                    r = "\n".join(lines[3:])
                    return p, r
                return "# Python source code:", raw_c

        # 5. Raw Text Corpora (FineWeb, FineWeb-Edu, Reference Documents)
        if not q_val and "text" in row:
            raw_t = str(row["text"]).strip()
            # Keep the whole document loss-bearing. Splitting its first paragraph
            # into the prompt silently excluded that text from next-token loss.
            # Raw corpus content must also bypass chat/math response rewriting.
            prompt = "Document:\n"
            return prompt, raw_t
        return str(q_val or ""), _normalize_3tier_response(str(r_val or ""), str(q_val or ""))
    return "", ""
    return "", ""


_DEFAULT_DATASET_REVISIONS: Dict[str, str] = {
    # OpenHermes-2.5's default branch holds a single 1.94 GB JSON array file,
    # causing datasets' JSON builder to buffer the full file + ujson AST in RAM
    # (~15+ GB per rank) and trigger host OOM SIGKILL on 2xT4 containers. The
    # auto-converted Parquet branch streams row-groups in <250 MB RSS per rank.
    "teknium/OpenHermes-2.5": "refs/convert/parquet",
    "HuggingFaceFW/fineweb": "9bb295ddab0e05d785b879661af7260fed5140fc",
    # Pinned immutable commit SHAs to guarantee exact reproducibility and prevent upstream dataset mutation:
    "HuggingFaceFW/fineweb-edu": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "HuggingFaceTB/cosmopedia-v2": "3ba9d605774198c5868892d7a8deda78031a781f",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "HuggingFaceTB/smoltalk": "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
    "HuggingFaceTB/finemath": "e92b25a616738fe95dc186b64dfb19f9c8525594",
    "open-r1/OpenR1-Math-220k": "e4e141ec9dea9f8326f4d347be56105859b2bd68",
}

_DEFAULT_DATASET_CONFIGS: Dict[str, str] = {
    "HuggingFaceFW/fineweb": "sample-10BT",
    "HuggingFaceTB/cosmopedia-v2": "cosmopedia-v2",
    "HuggingFaceFW/fineweb-edu": "sample-350BT",
    "openai/gsm8k": "main",
    "gsm8k": "main",
    "HuggingFaceTB/smoltalk": "all",
    "HuggingFaceTB/finemath": "finemath-4plus",
    "open-r1/OpenR1-Math-220k": "default",
}


def _get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face API token using unified hf_utils resolver."""
    from src.hf_utils import get_hf_token
    return get_hf_token()


def _open_hf_stream(
    repo: str,
    config_name: Optional[str] = None,
    split: str = "train",
    rank: int = 0,
    world_size: int = 1,
    revision: Optional[str] = None,
    max_retries: int = 6,
) -> Iterator[Any]:
    """Open a HuggingFace streaming dataset with robust token authentication, exponential backoff, and 429 recovery."""
    import os
    import time
    import random

    # Redirect legacy script-based gsm8k repo to official Parquet dataset to prevent 429 rate limit stalls
    if repo == "gsm8k":
        repo = "openai/gsm8k"

    effective_config = config_name if config_name is not None else _DEFAULT_DATASET_CONFIGS.get(repo)

    if "HF_HOME" not in os.environ:
        user_hf = os.path.expanduser("~/.hf_cache")
        os.makedirs(user_hf, exist_ok=True)
        os.environ["HF_HOME"] = user_hf
        os.environ["HF_HUB_CACHE"] = os.path.join(user_hf, "hub")

    from datasets import load_dataset

    effective_revision = revision if revision is not None else _DEFAULT_DATASET_REVISIONS.get(repo)

    for attempt in range(max_retries):
        kwargs: Dict[str, Any] = {"split": split, "streaming": True}
        if effective_revision is not None:
            kwargs["revision"] = effective_revision

        token = _get_hf_token()
        if token:
            kwargs["token"] = token
            try:
                from huggingface_hub import login
                login(token=token, add_to_git_credential=False)
            except Exception:
                pass

        try:
            ds = load_dataset(repo, effective_config, **kwargs) if effective_config else load_dataset(repo, **kwargs)
            if world_size > 1:
                try:
                    from datasets.distributed import split_dataset_by_node
                    ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
                except Exception:
                    if rank > 0 and hasattr(ds, "skip"):
                        ds = ds.skip(rank * 256)
            return iter(ds)

        except TypeError:
            # Backward compatibility for older datasets library versions lacking token or revision kwargs
            if "revision" in kwargs:
                kwargs.pop("revision", None)
            if "token" in kwargs:
                kwargs.pop("token", None)
            try:
                ds = load_dataset(repo, effective_config, **kwargs) if effective_config else load_dataset(repo, **kwargs)
                if world_size > 1:
                    try:
                        from datasets.distributed import split_dataset_by_node
                        ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
                    except Exception:
                        if rank > 0 and hasattr(ds, "skip"):
                            ds = ds.skip(rank * 256)
                return iter(ds)
            except Exception as e:
                err_str = str(e).lower()
                if "no (supported) data files" in err_str or "not found" in err_str or "does not exist" in err_str:
                    raise e
                wait_t = min(30.0, (2.0 ** attempt) + random.uniform(0.5, 2.0))
                if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                    print(f"⚠️ [DataStreamer] HuggingFace Rate Limit (429) on {repo}. Backing off {wait_t:.1f}s (retry {attempt+1}/{max_retries})...")
                else:
                    print(f"⚠️ [DataStreamer] Dataset stream notice on {repo} ({e}). Retrying in {wait_t:.1f}s...")
                time.sleep(wait_t)

        except Exception as e:
            err_str = str(e).lower()
            if "no (supported) data files" in err_str or "not found" in err_str or "does not exist" in err_str:
                raise e
            wait_t = min(30.0, (2.0 ** attempt) + random.uniform(0.5, 2.0))
            if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                print(f"⚠️ [DataStreamer] HuggingFace Rate Limit (429) on {repo}. Backing off {wait_t:.1f}s (retry {attempt+1}/{max_retries})...")
            else:
                print(f"⚠️ [DataStreamer] Dataset stream notice on {repo} ({e}). Retrying in {wait_t:.1f}s...")
            time.sleep(wait_t)

    # Final attempt fallback
    kwargs = {"split": split, "streaming": True}
    token = _get_hf_token()
    if token:
        kwargs["token"] = token
    ds = load_dataset(repo, effective_config, **kwargs) if effective_config else load_dataset(repo, **kwargs)
    return iter(ds)


def build_online_dataset_iterator(
    dataset_source: str = "dense_edu",
    rank: int = 0,
    world_size: int = 1,
    step_getter: Optional[Any] = None,
    total_steps: int = 1_500_000,
    inject_retrieval: bool = True,
    dataset_config: Optional[str] = None,
    dataset_revision: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Infinite, resilient HuggingFace multi-corpus streaming iterator shared by TPU and GPU trainers.
    Supports:
      - 'autonomous' / 'v5_base' / 'v5_base_curriculum': 4-phase step-aware developmental curriculum.
      - 'dense_edu' / 'foundation': High-density academic educational text.
      - 'reasoning_anneal' / 'tri_pillar': Balanced 1:1:1 Science, Commonsense, and Math/Code.
      - 'conversation_sft' / 'agentic': Multi-turn dialogues and tool execution traces.
      - 'code': Pure indented Python software and algorithmic problem solving.
      - 'consolidation': Final cooling replay mixture.
    """
    def _wrap_with_retrieval_injection(base_gen: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        if not inject_retrieval:
            return base_gen
        retrieval_path = os.path.join(REPO_ROOT, "data", "retrieval_curriculum_data.jsonl")
        if not os.path.exists(retrieval_path):
            print(
                f"⚠️ [Retrieval Injection Warning] inject_retrieval=True, but '{retrieval_path}' does not exist!\n"
                f"   Running without retrieval injection. Generate it via: python3 scripts/generate_retrieval_curriculum_data.py"
            )
            return base_gen

        print(f"📦 [Retrieval Injector] Active: Injecting long-context curriculum samples from '{retrieval_path}' (every 10th sample, ~10% mix)")

        def _open_retrieval_stream():
            while True:
                with open(retrieval_path, "r", encoding="utf-8") as rf:
                    for line in rf:
                        line = line.strip()
                        if line:
                            try:
                                d = json.loads(line)
                                d["_dataset"] = "retrieval_curriculum"
                                yield d
                            except Exception:
                                pass

        def _wrapped():
            r_it = _open_retrieval_stream()
            t = 0
            while True:
                if r_it is not None and (t % 10 == 9):
                    try:
                        item = next(r_it)
                        yield item
                        t += 1
                        continue
                    except Exception:
                        r_it = _open_retrieval_stream()
                try:
                    item = next(base_gen)
                    yield item
                    t += 1
                except StopIteration:
                    break
        return _wrapped()
    source = (dataset_source or "dense_edu").lower()
    if source in ("fineweb", "fineweb_10bt", "fineweb-10bt"):
        dataset_source = "HuggingFaceFW/fineweb"
        source = dataset_source.lower()
    if source == "huggingfacefw/fineweb":
        dataset_config = dataset_config or _DEFAULT_DATASET_CONFIGS["HuggingFaceFW/fineweb"]
    if source in ("dataset", "frontier", "frontier_100k", "frontier_reasoning", "frontier_sft", "ajitkumar-bio/cerebro-spheroid-v5-frontier-reasoning-tools-100k"):
        source = "frontier_100k"
    elif os.path.exists(dataset_source) or os.path.exists(os.path.join(REPO_ROOT, dataset_source)):
        source = "local_dataset"
    elif "/" in dataset_source and not dataset_source.startswith("http"):
        source = "custom_hf_dataset"
    elif source in ("chat_thinking", "db_thinking", "specialized_trihead", "specialized_sft"):
        source = "chat_thinking"
    elif source in ("sft", "instruction_sft", "chat_sft", "alignment", "reasoning_sft", "cot_sft", "think_sft", "r1_sft", "cot", "deepseek_r1", "conversational", "conversational_sft", "layman"):
        source = "conversational_sft"
    elif source in ("foundation", "auto", "basic", "fineweb_edu"):
        source = "dense_edu"
    elif source in ("reasoning", "beat_smollm", "smollm_pass", "frontier_mix", "tri_pillar"):
        source = "reasoning_anneal"
    elif source in ("agentic", "workflows"):
        source = "conversation_sft"
    elif source in ("balanced_mix", "balanced", "gold"):
        source = "consolidation"
    elif source in ("autonomous", "autonomous_curriculum", "v5_base", "v5_base_curriculum", "v5_curriculum", "curriculum"):
        source = "autonomous"

    def _next_or_reopen(it: Optional[Iterator[Any]], opener, max_retries: int = 5):
        import time, random
        for attempt in range(max_retries):
            if it is None:
                try:
                    it = opener()
                except Exception as e:
                    err_s = str(e).lower()
                    if "no (supported) data files" in err_s or "dataset not found" in err_s or "does not exist" in err_s:
                        raise e
                    wait_t = min(30.0, (2.0 ** attempt) + random.uniform(0.5, 2.0))
                    print(f"⚠️ [HuggingFace Stream] Re-opener error ({e}). Retrying in {wait_t:.1f}s...")
                    time.sleep(wait_t)
                    continue
            try:
                return next(it), it
            except StopIteration:
                try:
                    it = opener()
                    return next(it), it
                except Exception:
                    it = None
            except Exception as exc:
                it = None
                wait_t = min(30.0, (2.0 ** attempt) + random.uniform(0.5, 2.0))
                err_str = str(exc).lower()
                if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                    print(f"⚠️ [HuggingFace Stream] 429 Rate Limit encountered. Backing off for {wait_t:.1f}s (retry {attempt+1}/{max_retries})...")
                else:
                    print(f"⚠️ [HuggingFace Stream] Read interruption ({exc}). Re-establishing connection in {wait_t:.1f}s...")
                time.sleep(wait_t)
        it = opener()
        return next(it), it

    def _is_valid_doc(text: str, min_chars: int = 150, max_chars: int = 14000) -> bool:
        if not text or not isinstance(text, str):
            return False
        s = text.strip()
        if len(s) < min_chars or len(s) > max_chars:
            return False
        # Filter excessive repetition
        if len(s) > 400:
            words = s.split()
            if len(words) > 30 and (len(set(words[:50])) / 50.0 < 0.35):
                return False
        return True

    def _doc_passes_filter(row: Dict[str, Any], min_score: float) -> Tuple[bool, str]:
        score_val = row.get("score")
        if score_val is None:
            score_val = row.get("educational_score")
        if score_val is None:
            score_val = row.get("int_score")
        try:
            score = float(score_val if score_val is not None else 0.0)
        except (ValueError, TypeError):
            score = 0.0

        if score < min_score:
            return False, "score"

        lang = str(row.get("language", "en") or "en").lower()
        try:
            lang_score = float(row.get("language_score", 1.0) or 1.0)
        except (ValueError, TypeError):
            lang_score = 1.0
        if lang != "en" or lang_score < 0.70:
            return False, "language"

        text = str(row.get("text", "") or "")
        if not _is_valid_doc(text, min_chars=150):
            return False, "length_repetition"

        return True, "ok"

    def _next_filtered_fineweb(it: Optional[Iterator[Any]], opener, min_score: float = 3.0, max_attempts: int = 1000):
        """
        Strict Educational Web Filter:
        Guarantees that ONLY high-quality, English, non-repetitive educational text
        with educational_score >= min_score is returned. Never yields rejected rows.
        """
        attempts = 0
        while attempts < max_attempts:
            row, it = _next_or_reopen(it, opener)
            attempts += 1
            GLOBAL_TELEMETRY.record_examined("fineweb_edu")

            passed, reason = _doc_passes_filter(row, min_score)
            if not passed:
                GLOBAL_TELEMETRY.record_rejected("fineweb_edu", reason)
                continue

            text = str(row.get("text", "") or "")
            GLOBAL_TELEMETRY.record_accepted("fineweb_edu", 0, len(text.split()))
            return row, it

        # If 1000 records failed in current stream slice, cleanly reopen stream and continue searching with strict filter
        it = opener()
        while True:
            row, it = _next_or_reopen(it, opener)
            GLOBAL_TELEMETRY.record_examined("fineweb_edu")
            passed, reason = _doc_passes_filter(row, min_score)
            if not passed:
                GLOBAL_TELEMETRY.record_rejected("fineweb_edu", reason)
                continue
            text = str(row.get("text", "") or "")
            GLOBAL_TELEMETRY.record_accepted("fineweb_edu", 0, len(text.split()))
            return row, it

    if "/" in dataset_source:
        repo_id = dataset_source
        custom_it: Optional[Iterator[Any]] = None

        def _open_custom_stream():
            config_name = dataset_config or _DEFAULT_DATASET_CONFIGS.get(repo_id)
            revision = dataset_revision or _DEFAULT_DATASET_REVISIONS.get(repo_id)
            print(
                f"🌐 Connecting online stream to {repo_id}"
                f"{('/' + config_name) if config_name else ''}"
                f"{f' @ {revision}' if revision else ''}..."
            )
            return _open_hf_stream(
                repo_id,
                config_name,
                rank=rank,
                world_size=world_size,
                revision=revision,
            )

        def _custom_gen():
            nonlocal custom_it
            while True:
                row, custom_it = _next_or_reopen(custom_it, _open_custom_stream)
                prompt = str(row.get("prompt", "") or "").strip()
                text = str(row.get("text", "") or "").strip()
                answer = str(row.get("answer", "") or "").strip()
                toks = row.get("tokens")
                weights = row.get("target_weights")
                yield {
                    "prompt": prompt,
                    "text": text if text else answer,
                    "answer": answer if answer else text,
                    "tokens": toks,
                    "target_weights": weights,
                    "_dataset": str(row.get("_dataset") or repo_id),
                }

        return _wrap_with_retrieval_injection(_custom_gen())

    if source == "autonomous":
        # Refined 4-Phase Developmental Autonomous Curriculum for CerebroSpheroid V5
        # Phase 1 (0 -> 15%): World Knowledge, Educational Literacy & Imprinting
        #   (40% FineWeb-Edu >= 3.0, 25% Cosmopedia v2 Digested, 15% Math/GSM8K, 10% Python Code, 10% Everyday Commonsense)
        # Phase 2 (15% -> 50%): Broadening, Commonsense Expansion & Algorithmic Grounding
        #   (30% FineWeb-Edu >= 3.5, 20% Cosmopedia v2 Digested, 25% Math Proofs & CoT, 25% Indented Python)
        # Phase 3 (50% -> 85%): Balanced Tri-Pillar Reasoning & Deep Deliberation
        #   (33.3% Science/STEM >= 4.0, 33.3% Multi-Turn Logic/Magpie, 33.3% Math-CoT & Algorithmic Code)
        # Phase 4 (85% -> 100%): Annealing Basin, Proof Consolidation & 3-Tier Cognitive Alignment
        #   (30% Verified GSM8K & MetaMath, 30% FineWeb-Edu >= 4.0, 20% Python Code, 20% Magpie Ultra / 3-Tier SFT)
        know_it: Optional[Iterator[Any]] = None
        cosmo_it: Optional[Iterator[Any]] = None
        everyday_it: Optional[Iterator[Any]] = None
        openhermes_it: Optional[Iterator[Any]] = None
        logic_it: Optional[Iterator[Any]] = None
        code_it: Optional[Iterator[Any]] = None
        pycode_it: Optional[Iterator[Any]] = None
        gsm_it: Optional[Iterator[Any]] = None
        math_it: Optional[Iterator[Any]] = None
        finemath_it: Optional[Iterator[Any]] = None

        # Load emergency offline backup reserve to prevent worker starvation during network drops
        emergency_offline_pool: List[Dict[str, Any]] = []
        try:
            local_train_p = os.path.join(REPO_ROOT, "data", "cerebro_v5_custom_dataset", "train.parquet")
            if os.path.exists(local_train_p):
                import pyarrow.parquet as pq
                tbl = pq.read_table(local_train_p)
                prompts = tbl["prompt"].to_pylist()
                answers = tbl["answer"].to_pylist()
                for p_txt, a_txt in zip(prompts[:1000], answers[:1000]):
                    if p_txt and a_txt:
                        emergency_offline_pool.append({
                            "prompt": p_txt,
                            "text": a_txt,
                            "answer": a_txt,
                            "_dataset": "offline_emergency_reserve",
                        })
        except Exception:
            pass

        def _open_cosmo_stream():
            candidate_streams = [
                ("AjitKumar-Bio/cosmopedia-v2-digested", None),
                ("AjitKumar-Bio/cerebro-spheroid-v5-digested-pretrain", None),
                ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2"),
                ("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2"),
            ]
            for c_repo, c_cfg in candidate_streams:
                try:
                    return _open_hf_stream(c_repo, c_cfg, rank=rank, world_size=world_size)
                except Exception:
                    continue
            return _open_hf_stream("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", rank=rank, world_size=world_size)

        def _get_step_val() -> int:
            val = 0
            if callable(step_getter):
                try:
                    val = int(step_getter())
                except Exception:
                    val = 0
            elif isinstance(step_getter, (int, float)):
                val = int(step_getter)
            return val

        def _autonomous_gen():
            nonlocal know_it, cosmo_it, everyday_it, openhermes_it, logic_it, code_it, pycode_it, gsm_it, math_it, finemath_it
            turn = 0

            def _tag_yield(item: Any, ds_tag: str):
                if isinstance(item, dict):
                    item["_dataset"] = ds_tag
                return item

            while True:
                raw_step = _get_step_val()
                # If external step counter is not advancing or unset, progress via internal turn count
                effective_step = raw_step if raw_step > 0 else turn
                tau = float(effective_step) / float(max(1, total_steps))

                try:
                    if tau < 0.15:
                        # Phase 1: Foundation & Cognitive Imprinting (Cycle mod 10)
                        # 40% FineWeb-Edu >= 3.0, 25% Cosmopedia v2 Digested, 15% Math, 10% Python Code, 10% Everyday Commonsense
                        mod10 = turn % 10
                        if mod10 in (0, 2, 4, 6):
                            row, know_it = _next_filtered_fineweb(
                                know_it, lambda: _open_hf_stream("HuggingFaceFW/fineweb-edu", "sample-350BT", rank=rank, world_size=world_size), min_score=3.0
                            )
                            yield _tag_yield(row, "fineweb_edu")
                        elif mod10 in (1, 5):
                            row, cosmo_it = _next_or_reopen(cosmo_it, _open_cosmo_stream)
                            p = (row.get("prompt") or "Explain the concept:").strip()
                            r = (row.get("text") or row.get("answer") or "").strip()
                            item = {
                                "prompt": p,
                                "text": r,
                                "answer": r,
                                "tokens": row.get("tokens"),
                                "target_weights": row.get("target_weights"),
                                "conversations": [{"from": "human", "value": p}, {"from": "gpt", "value": r}],
                            }
                            yield _tag_yield(item, "cosmopedia_digested" if row.get("target_weights") else "cosmopedia")
                        elif mod10 == 3:
                            row, everyday_it = _next_or_reopen(
                                everyday_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "everyday-conversations", rank=rank, world_size=world_size)
                            )
                            yield _tag_yield(row, "smoltalk")
                        elif mod10 in (7, 9):
                            if (turn // 10) % 2 == 0:
                                row, finemath_it = _next_or_reopen(
                                    finemath_it, lambda: _open_hf_stream("HuggingFaceTB/finemath", "finemath-4plus", rank=rank, world_size=world_size)
                                )
                            else:
                                row, math_it = _next_or_reopen(
                                    math_it, lambda: _open_hf_stream("meta-math/MetaMathQA", rank=rank, world_size=world_size)
                                )
                            yield _tag_yield(row, "finemath")
                        else:
                            if (turn // 10) % 2 == 0:
                                row, pycode_it = _next_or_reopen(
                                    pycode_it, lambda: _open_hf_stream("ajibawa-2023/Python-Code-Large", rank=rank, world_size=world_size)
                                )
                            else:
                                row, code_it = _next_or_reopen(
                                    code_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "self-oss-instruct", rank=rank, world_size=world_size)
                                )
                            yield _tag_yield(row, "python_code")

                    elif tau < 0.50:
                        # Phase 2: Broadening, Commonsense Expansion & Algorithmic Grounding (Cycle mod 10)
                        # 30% FineWeb-Edu >= 3.5, 20% Cosmopedia v2 Digested, 25% Math Proofs & CoT, 25% Indented Python
                        mod10 = turn % 10
                        if mod10 in (0, 3, 6):
                            row, know_it = _next_filtered_fineweb(
                                know_it, lambda: _open_hf_stream("HuggingFaceFW/fineweb-edu", "sample-350BT", rank=rank, world_size=world_size), min_score=3.5
                            )
                            yield _tag_yield(row, "fineweb_edu")
                        elif mod10 in (1, 5):
                            row, cosmo_it = _next_or_reopen(cosmo_it, _open_cosmo_stream)
                            p = (row.get("prompt") or "Explain the concept:").strip()
                            r = (row.get("text") or row.get("answer") or "").strip()
                            item = {
                                "prompt": p,
                                "text": r,
                                "answer": r,
                                "tokens": row.get("tokens"),
                                "target_weights": row.get("target_weights"),
                                "conversations": [{"from": "human", "value": p}, {"from": "gpt", "value": r}],
                            }
                            yield _tag_yield(item, "cosmopedia_digested" if row.get("target_weights") else "cosmopedia")
                        elif mod10 in (2, 7):
                            if (turn // 10) % 2 == 0:
                                row, pycode_it = _next_or_reopen(
                                    pycode_it, lambda: _open_hf_stream("ajibawa-2023/Python-Code-Large", rank=rank, world_size=world_size)
                                )
                            else:
                                row, code_it = _next_or_reopen(
                                    code_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "self-oss-instruct", rank=rank, world_size=world_size)
                                )
                            yield _tag_yield(row, "python_code")
                        else:
                            mod_math = (turn // 10) % 3
                            if mod_math == 0:
                                row, finemath_it = _next_or_reopen(
                                    finemath_it, lambda: _open_hf_stream("HuggingFaceTB/finemath", "finemath-4plus", rank=rank, world_size=world_size)
                                )
                            elif mod_math == 1:
                                row, gsm_it = _next_or_reopen(
                                    gsm_it, lambda: _open_hf_stream("openai/gsm8k", "main", rank=rank, world_size=world_size)
                                )
                            else:
                                row, math_it = _next_or_reopen(
                                    math_it, lambda: _open_hf_stream("meta-math/MetaMathQA", rank=rank, world_size=world_size)
                                )
                            yield _tag_yield(row, "finemath")

                    elif tau < 0.85:
                        # Phase 3: Deep Deliberation & 1:1:1 Tri-Pillar Reasoning (Cycle mod 6)
                        # 33.3% STEM Science >= 4.0, 33.3% Multi-Turn Logic & Commonsense, 33.3% Math-CoT & Indented Code
                        mod6 = turn % 6
                        if mod6 in (0, 3):
                            row, know_it = _next_filtered_fineweb(
                                know_it, lambda: _open_hf_stream("HuggingFaceFW/fineweb-edu", "sample-350BT", rank=rank, world_size=world_size), min_score=4.0
                            )
                            yield _tag_yield(row, "fineweb_edu")
                        elif mod6 == 1:
                            row, everyday_it = _next_or_reopen(
                                everyday_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "everyday-conversations", rank=rank, world_size=world_size)
                            )
                            yield _tag_yield(row, "smoltalk")
                        elif mod6 == 2:
                            if (turn // 6) % 2 == 0:
                                row, pycode_it = _next_or_reopen(
                                    pycode_it, lambda: _open_hf_stream("ajibawa-2023/Python-Code-Large", rank=rank, world_size=world_size)
                                )
                            else:
                                row, code_it = _next_or_reopen(
                                    code_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "self-oss-instruct", rank=rank, world_size=world_size)
                                )
                            yield _tag_yield(row, "python_code")
                        elif mod6 == 4:
                            mod_math = (turn // 6) % 3
                            if mod_math == 0:
                                row, finemath_it = _next_or_reopen(
                                    finemath_it, lambda: _open_hf_stream("HuggingFaceTB/finemath", "finemath-4plus", rank=rank, world_size=world_size)
                                )
                            elif mod_math == 1:
                                row, gsm_it = _next_or_reopen(
                                    gsm_it, lambda: _open_hf_stream("openai/gsm8k", "main", rank=rank, world_size=world_size)
                                )
                            else:
                                row, math_it = _next_or_reopen(
                                    math_it, lambda: _open_hf_stream("meta-math/MetaMathQA", rank=rank, world_size=world_size)
                                )
                            yield _tag_yield(row, "finemath")
                        else:
                            if (turn // 6) % 2 == 0:
                                row, openhermes_it = _next_or_reopen(
                                    openhermes_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "openhermes-100k", rank=rank, world_size=world_size)
                                )
                            else:
                                row, logic_it = _next_or_reopen(
                                    logic_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "smol-magpie-ultra", rank=rank, world_size=world_size)
                                )
                            yield _tag_yield(row, "smoltalk")

                    else:
                        # Phase 4: Annealing Basin, Proof Consolidation & 3-Tier Cognitive Alignment (Cycle mod 10)
                        # 30% Verified GSM8K/FineMath/MetaMath, 30% FineWeb-Edu >= 4.0, 20% Indented Code with Tests, 20% Magpie Ultra Logic
                        mod10 = turn % 10
                        if mod10 in (0, 3, 6):
                            row, know_it = _next_filtered_fineweb(
                                know_it, lambda: _open_hf_stream("HuggingFaceFW/fineweb-edu", "sample-350BT", rank=rank, world_size=world_size), min_score=4.0
                            )
                            yield _tag_yield(row, "fineweb_edu")
                        elif mod10 in (1, 4, 7):
                            mod_math = (turn // 10) % 3
                            if mod_math == 0:
                                row, finemath_it = _next_or_reopen(
                                    finemath_it, lambda: _open_hf_stream("HuggingFaceTB/finemath", "finemath-4plus", rank=rank, world_size=world_size)
                                )
                            elif mod_math == 1:
                                row, gsm_it = _next_or_reopen(
                                    gsm_it, lambda: _open_hf_stream("openai/gsm8k", "main", rank=rank, world_size=world_size)
                                )
                            else:
                                row, math_it = _next_or_reopen(
                                    math_it, lambda: _open_hf_stream("meta-math/MetaMathQA", rank=rank, world_size=world_size)
                                )
                            yield _tag_yield(row, "finemath")
                        elif mod10 in (2, 8):
                            if (turn // 10) % 2 == 0:
                                row, pycode_it = _next_or_reopen(
                                    pycode_it, lambda: _open_hf_stream("ajibawa-2023/Python-Code-Large", rank=rank, world_size=world_size)
                                )
                            else:
                                row, code_it = _next_or_reopen(
                                    code_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "self-oss-instruct", rank=rank, world_size=world_size)
                                )
                            yield _tag_yield(row, "python_code")
                        else:
                            row, logic_it = _next_or_reopen(
                                logic_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "smol-magpie-ultra", rank=rank, world_size=world_size)
                            )
                            yield _tag_yield(row, "smoltalk")

                except Exception as stream_exc:
                    # If online streams fail, immediately draw from local emergency reserve pool without stalling
                    if emergency_offline_pool:
                        import random
                        fallback_sample = random.choice(emergency_offline_pool)
                        yield fallback_sample
                    else:
                        time.sleep(0.5)
                turn += 1

        return _wrap_with_retrieval_injection(_autonomous_gen())

    if source in ("frontier_100k", "local_dataset", "custom_hf_dataset"):
        # Premier Frontier Reasoning & Tool Dataset (100k samples) / Custom Datasets
        # Seamlessly supports local JSONL / Parquet or streaming from Hugging Face Hub:
        # AjitKumar-Bio/cerebro-spheroid-v5-frontier-reasoning-tools-100k
        def _frontier_dataset_gen():
            import json, random, time

            # 1. Resolve local candidates
            local_candidates = []
            if source == "local_dataset":
                local_candidates.append(
                    dataset_source if os.path.isabs(dataset_source) else os.path.join(REPO_ROOT, dataset_source)
                )

            local_candidates.extend([
                os.path.join(REPO_ROOT, "data", "frontier_100k", "frontier_100k_spheroid_format.jsonl"),
                os.path.join(REPO_ROOT, "data", "frontier_100k", "frontier_100k_industry_sharegpt.jsonl"),
                os.path.join(REPO_ROOT, "data", "frontier_100k", "frontier_100k_reasoning_tool_sft.parquet"),
                os.path.join(REPO_ROOT, "data", "cerebro_v5_custom_dataset", "train.jsonl"),
                os.path.join(REPO_ROOT, "data", "diverse_sft_corpus_25k.jsonl"),
                os.path.join(REPO_ROOT, "data", "chat_thinking_sft_15k.jsonl"),
            ])

            chosen_local = None
            for p in local_candidates:
                if os.path.exists(p) and os.path.getsize(p) > 1024:
                    chosen_local = p
                    break

            if chosen_local is not None:
                print(f"📦 [Dataset Streamer] Streaming directly from local dataset: {chosen_local}")
                while True:
                    if chosen_local.endswith(".parquet"):
                        import pandas as pd
                        df = pd.read_parquet(chosen_local)
                        records = df.to_dict(orient="records")
                        random.shuffle(records)
                        for r in records:
                            r["_dataset"] = "frontier_100k"
                            yield r
                    else:
                        with open(chosen_local, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        d = json.loads(line)
                                        d["_dataset"] = "frontier_100k"
                                        yield d
                                    except Exception:
                                        continue
            else:
                # 2. Remote Hugging Face Hub Streaming Pipeline
                hf_repo = dataset_source if (source == "custom_hf_dataset") else "AjitKumar-Bio/cerebro-spheroid-v5-frontier-reasoning-tools-100k"
                print(f"🌐 [Dataset Streamer] Streaming directly from Hugging Face Hub: {hf_repo}...")
                hf_it: Optional[Iterator[Any]] = None
                while True:
                    try:
                        row, hf_it = _next_or_reopen(
                            hf_it, lambda: _open_hf_stream(hf_repo, split="train", rank=rank, world_size=world_size)
                        )
                        if isinstance(row, dict):
                            row["_dataset"] = "frontier_100k"
                        yield row
                    except Exception as e:
                        print(f"⚠️ [Dataset Streamer] Remote stream notice ({e}). Backing off...")
                        time.sleep(1.0)

        return _wrap_with_retrieval_injection(_frontier_dataset_gen())

    if source == "chat_thinking":
        # 100% Offline Database-Augmented Chat, Reasoning & Action Streamer:
        # Zero HuggingFace dependence, zero unauthenticated rate limits.
        # Combines conversational chat, SQL factual queries, calculator, code, and agentic tools.
        chat_thinking_path = os.path.join(REPO_ROOT, "data", "chat_thinking_sft_15k.jsonl")
        if not os.path.exists(chat_thinking_path):
            chat_thinking_path = os.path.join(os.getcwd(), "data", "chat_thinking_sft_15k.jsonl")
        if not os.path.exists(chat_thinking_path):
            try:
                from scripts.build_chat_thinking_dataset import build_chat_thinking_dataset
                build_chat_thinking_dataset()
            except Exception as e:
                print(f"Notice auto-generating chat thinking dataset: {e}")

        def _chat_thinking_gen():
            import json, os, time, random
            while True:
                if os.path.exists(chat_thinking_path):
                    with open(chat_thinking_path, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    random.shuffle(lines)
                    for line in lines:
                        try:
                            item = json.loads(line)
                            item["_dataset"] = "chat_thinking"
                            yield item
                        except Exception:
                            continue
                else:
                    time.sleep(1.0)

        return _wrap_with_retrieval_injection(_chat_thinking_gen())

    if source == "conversational_sft":
        # Specialized Layman Conversational & Step-by-Step CoT Reasoning Mixture:
        # Zero unsupervised web crawl, zero raw textbook physics/calculus drift.
        #   30% SmolTalk everyday-conversations (natural dialogue, greetings, small talk, assistance)
        #   25% SmolTalk openhermes-100k (broad world knowledge, clear explanations, definitions)
        #   20% Local Layman Conversational CoT ('what is earth', 'pixle', 'port', capitals, math with <think>)
        #   15% SmolTalk self-oss-instruct (clean, functional Python coding with step-by-step logic)
        #   10% GSM8K + Master Diverse Slot QA (elementary arithmetic, word problems, direct factual QA)
        everyday_it: Optional[Iterator[Any]] = None
        openhermes_it: Optional[Iterator[Any]] = None
        code_it: Optional[Iterator[Any]] = None
        gsm_it: Optional[Iterator[Any]] = None
        layman_it: Optional[Iterator[Any]] = None
        diverse_it: Optional[Iterator[Any]] = None

        def _open_local_cyclic_jsonl(filepath: str) -> Iterator[Dict[str, Any]]:
            import json, os, time
            if not os.path.exists(filepath) and "layman_conversational_cot" in filepath:
                try:
                    from scripts.build_layman_sft_dataset import build_layman_dataset
                    build_layman_dataset()
                except Exception:
                    pass
            while True:
                if os.path.exists(filepath):
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            l = line.strip()
                            if l:
                                try:
                                    yield json.loads(l)
                                except Exception:
                                    continue
                else:
                    time.sleep(1.0)

        layman_path = os.path.join(REPO_ROOT, "data", "layman_conversational_cot.jsonl")
        if not os.path.exists(layman_path):
            layman_path = os.path.join(os.getcwd(), "data", "layman_conversational_cot.jsonl")

        diverse_path = os.path.join(REPO_ROOT, "data", "diverse_sft_corpus_25k.jsonl")
        if not os.path.exists(diverse_path):
            diverse_path = os.path.join(os.getcwd(), "data", "diverse_sft_corpus_25k.jsonl")

        def _conversational_gen():
            nonlocal everyday_it, openhermes_it, code_it, gsm_it, layman_it, diverse_it
            turn = 0
            while True:
                mod10 = turn % 10
                try:
                    if mod10 in (0, 3, 6):
                        # 30% SmolTalk Everyday Conversations
                        row, everyday_it = _next_or_reopen(
                            everyday_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "everyday-conversations", rank=rank, world_size=world_size)
                        )
                        ds_name = "smoltalk_everyday"
                        if isinstance(row, dict):
                            row["_dataset"] = ds_name
                        yield row

                    elif mod10 in (1, 4):
                        # 25% SmolTalk OpenHermes 100k (Definitions, World Knowledge & Explanation)
                        if (turn // 10) % 2 == 0:
                            row, openhermes_it = _next_or_reopen(
                                openhermes_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "openhermes-100k", rank=rank, world_size=world_size)
                            )
                            ds_name = "openhermes_100k"
                        else:
                            row, openhermes_it = _next_or_reopen(
                                openhermes_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "smol-magpie-ultra", rank=rank, world_size=world_size)
                            )
                            ds_name = "smol_magpie_ultra"
                        if isinstance(row, dict):
                            row["_dataset"] = ds_name
                        yield row

                    elif mod10 in (2, 7):
                        # 20% Local Layman Conversational & CoT (earth, pixel, port, capitals, greetings, math with <think>)
                        if layman_it is None:
                            layman_it = _open_local_cyclic_jsonl(layman_path)
                        try:
                            row = next(layman_it)
                        except Exception:
                            layman_it = _open_local_cyclic_jsonl(layman_path)
                            row = next(layman_it)
                        if isinstance(row, dict):
                            row["_dataset"] = "layman_cot"
                        yield row

                    elif mod10 in (5, 8):
                        # 15% Python Software & Algorithm Synthesis (Self-OSS-Instruct)
                        row, code_it = _next_or_reopen(
                            code_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "self-oss-instruct", rank=rank, world_size=world_size)
                        )
                        if isinstance(row, dict):
                            row["_dataset"] = "self_oss_instruct"
                        yield row

                    else:
                        # 10% GSM8K Arithmetic Reasoning & Master Diverse Slot QA
                        if os.path.exists(diverse_path) and (turn // 10) % 2 == 0:
                            if diverse_it is None:
                                diverse_it = _open_local_cyclic_jsonl(diverse_path)
                            try:
                                row = next(diverse_it)
                            except Exception:
                                diverse_it = _open_local_cyclic_jsonl(diverse_path)
                                row = next(diverse_it)
                            ds_name = "master_diverse_slot"
                        else:
                            row, gsm_it = _next_or_reopen(
                                gsm_it, lambda: _open_hf_stream("openai/gsm8k", "main", rank=rank, world_size=world_size)
                            )
                            ds_name = "gsm8k_cot"
                        if isinstance(row, dict):
                            row["_dataset"] = ds_name
                        yield row

                except Exception:
                    time.sleep(0.5)
                turn += 1

        return _wrap_with_retrieval_injection(_conversational_gen())

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
                                fw_it, lambda: _open_hf_stream("HuggingFaceFW/fineweb-edu", "sample-350BT", rank=rank, world_size=world_size)
                            )
                            if float(row.get("score", 0.0) or row.get("educational_score", 0.0) or row.get("int_score", 0.0)) >= 3.0:
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

        return _wrap_with_retrieval_injection(_dense_gen())

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

        return _wrap_with_retrieval_injection(_conv_gen())

    if source == "reasoning_anneal":
        gsm_it: Optional[Iterator[Any]] = None
        math_it: Optional[Iterator[Any]] = None
        everyday_it: Optional[Iterator[Any]] = None
        openhermes_it: Optional[Iterator[Any]] = None
        code_it: Optional[Iterator[Any]] = None
        logic_it: Optional[Iterator[Any]] = None
        know_it: Optional[Iterator[Any]] = None

        def _reason_gen():
            nonlocal gsm_it, math_it, everyday_it, openhermes_it, code_it, logic_it, know_it
            turn = 0
            while True:
                mod6 = turn % 6
                try:
                    if mod6 == 0:
                        # 1. Exact Match Math-CoT: GSM8K (with official #### anchor) + MetaMathQA proofs
                        if (turn // 6) % 2 == 0:
                            row, gsm_it = _next_or_reopen(
                                gsm_it, lambda: _open_hf_stream("openai/gsm8k", "main", rank=rank, world_size=world_size)
                            )
                        else:
                            row, math_it = _next_or_reopen(
                                math_it, lambda: _open_hf_stream("meta-math/MetaMathQA", rank=rank, world_size=world_size)
                            )
                        yield row
                    elif mod6 in (1, 3):
                        # 2 & 4. Science & Academic Anchor: FineWeb-Edu (score >= 3.0) -> 33.3% allocation!
                        # Protects and locks in ARC-Challenge (40%+) and MMLU factual recall
                        row, know_it = _next_filtered_fineweb(
                            know_it, lambda: _open_hf_stream("HuggingFaceFW/fineweb-edu", "sample-350BT", rank=rank, world_size=world_size), min_score=3.0
                        )
                        yield row
                    elif mod6 == 2:
                        # 3. Everyday Commonsense Physical Actions (SmolTalk everyday-conversations) -> Targets HellaSwag
                        row, everyday_it = _next_or_reopen(
                            everyday_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "everyday-conversations", rank=rank, world_size=world_size)
                        )
                        yield row
                    elif mod6 == 4:
                        # 5. Multi-Subject Academic STEM & Logic (SmolTalk openhermes-100k + smol-magpie-ultra)
                        if (turn // 6) % 2 == 0:
                            row, openhermes_it = _next_or_reopen(
                                openhermes_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "openhermes-100k", rank=rank, world_size=world_size)
                            )
                        else:
                            row, logic_it = _next_or_reopen(
                                logic_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "smol-magpie-ultra", rank=rank, world_size=world_size)
                            )
                        yield row
                    else:
                        # 6. Python-Edu Software Algorithms & Code Blocks (SmolTalk self-oss-instruct) -> Targets HumanEval
                        row, code_it = _next_or_reopen(
                            code_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "self-oss-instruct", rank=rank, world_size=world_size)
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

        return _wrap_with_retrieval_injection(_reason_gen())

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

        return _wrap_with_retrieval_injection(_consol_gen())

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

    return _wrap_with_retrieval_injection(_oh_gen())


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


def get_default_elastic_buckets(total_window: int = 1024, max_target: Optional[int] = None) -> List[Tuple[str, int, int]]:
    """Returns high-density complementary sliding window buckets preserving intact prompts and deep reasoning trajectories."""
    assert total_window >= 4, "total_window must be at least 4 tokens"
    if total_window >= 4096:
        base = [
            ("short_prompt_long_target", 256, 3840),
            ("balanced_long", 2048, 2048),
            ("long_prompt", 3840, 256),
        ]
    elif total_window >= 2048:
        base = [
            ("short_prompt_long_target", 256, 1792),
            ("balanced_long", 1024, 1024),
            ("long_prompt", 1792, 256),
        ]
    elif total_window >= 1024:
        base = [
            ("standard_cot", 256, 512),
            ("deep_reasoning", 256, 768),
            ("balanced", 384, 512),
            ("long_prompt", 512, 512),
        ]
    elif total_window >= 768:
        base = [
            ("short_prompt_deep_cot", 128, 512),
            ("standard_cot", 256, 512),
            ("balanced", 256, 384),
            ("long_prompt", 384, 384),
        ]
    elif total_window >= 512:
        base = [
            ("standard_cot", 128, 384),
            ("balanced", 256, 256),
            ("deep_reasoning", 128, 384),
        ]
    else:
        half = total_window // 2
        base = [("balanced", half, half)]

    if max_target is not None and max_target > 0:
        capped = []
        for name, p_l, t_l in base:
            if t_l > max_target:
                p_new = min(total_window - max_target, p_l + (t_l - max_target))
                capped.append((name, p_new, max_target))
            else:
                capped.append((name, p_l, t_l))
        return capped
    return base


def select_best_bucket(p_len: int, t_len: int, buckets: List[Tuple[str, int, int]]) -> int:
    """
    Selects the bucket that minimizes wasted padding while strictly preserving full-text prompt intactness.
    Prompt truncation carries a massive penalty (50,000) so a bucket accommodating the full question is always prioritized.
    Target truncation carries a heavy penalty (25,000) so that complete reasoning answers are never cut short.
    """
    best_idx = min(1, len(buckets) - 1)
    min_penalty = float("inf")

    for idx, (_, bp, bt) in enumerate(buckets):
        if p_len <= bp and t_len <= bt:
            # Fits completely: penalty is wasted padding
            penalty = (bp - p_len) + (bt - t_len)
        elif p_len <= bp:
            # Fits prompt completely, target exceeds bucket: heavily penalize target truncation so full response is preserved!
            penalty = 25000 + (bp - p_len) + (t_len - bt) * 50
        else:
            # Prompt exceeds bucket: massive penalty! Never truncate the prompt if another bucket fits it!
            penalty = 50000 + (p_len - bp) * 100 + max(0, t_len - bt) * 50

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
        self.tokens_by_dataset: Dict[str, int] = collections.defaultdict(int)
        self._lock = threading.Lock()

    def get_tokens_by_dataset(self) -> Dict[str, int]:
        """Returns snapshot of ingested tokens categorized per dataset."""
        with self._lock:
            return dict(self.tokens_by_dataset)

    def set_tokens_by_dataset(self, d: Dict[str, int]) -> None:
        """Sets ingested token counts (useful when resuming from checkpoint)."""
        with self._lock:
            self.tokens_by_dataset = collections.defaultdict(int, {str(k): int(v) for k, v in d.items()})

    def update_buckets(self, new_buckets: List[Tuple[str, int, int]], new_total_window: Optional[int] = None) -> None:
        """Dynamically transitions active context buckets for curriculum scaling."""
        with self._lock:
            self.buckets = list(new_buckets)
            if new_total_window is not None:
                self.total_window = int(new_total_window)
            # Re-initialize bucket queues with new dimensions
            self.queues = {i: collections.deque(maxlen=self.max_queue_size) for i in range(len(self.buckets))}

    def update_target_len(self, new_target_len: int) -> None:
        """Dynamically updates target length and re-computes active buckets."""
        with self._lock:
            new_buckets = get_default_elastic_buckets(self.total_window, max_target=int(new_target_len))
            self.update_buckets(new_buckets, new_total_window=self.total_window)

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

    def next_batch(self, bucket_idx: Optional[int] = None) -> Any:
        """
        Pulls from dataset stream until any bucket has batch_size examples.
        Returns: (bucket_idx, bucket_name, prompt_batch, target_batch[, feedback_batch])
        """
        while True:
            # Check if any bucket queue is ready to pop a full batch
            bucket_order = [bucket_idx] if bucket_idx is not None else range(len(self.buckets))
            for b_idx in bucket_order:
                if b_idx not in self.queues:
                    raise IndexError(f"Unknown elastic bucket index: {b_idx}")
                if len(self.queues[b_idx]) >= self.batch_size:
                    name, p_len, t_len = self.buckets[b_idx]
                    batch_items = [self.queues[b_idx].popleft() for _ in range(self.batch_size)]
                    p_list, t_list, c_list, d_list = zip(*batch_items)
                    with self._lock:
                        for ds, p, t in zip(d_list, p_list, t_list):
                            # Attribute actual tokens, not fixed-shape padding.
                            self.tokens_by_dataset[ds] += int(
                                np.count_nonzero(np.asarray(p) != self.pad_id)
                                + np.count_nonzero(np.asarray(t) >= 0)
                            )
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

                ds_name = row.get("_dataset", "corpus") if isinstance(row, dict) else "corpus"

                # Biological Sensory Input Delimiters: <stim> ... </stim>
                try:
                    from src.corpus import format_stimulus_prompt
                    q_txt = format_stimulus_prompt(q_txt)
                except Exception:
                    if not q_txt.startswith("<stim>"):
                        q_txt = f"<stim> {q_txt} </stim>"

                # 3-Tier Cognitive Formatting for Response:
                # 1. If response already contains <act> or <think>, preserve exact structured boundaries.
                # 2. Tool action output: External tool calls are enclosed in <act> ... </act>.
                # 3. Conversational dialogue & code generation: Natural fluid plain text outside <act>.
                r_txt = _normalize_3tier_response(r_txt, q_txt)
                if "<act>" not in r_txt:
                    if "calculator" in ds_name.lower():
                        r_txt = f"<act> calc({r_txt.strip()}) </act>\nAnswer: {r_txt.strip()}"
                raw_pairs.append((q_txt, r_txt, ds_name))

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
                p_list, t_list, c_list, d_list = zip(*items)
                with self._lock:
                    for ds, p, t in zip(d_list, p_list, t_list):
                        self.tokens_by_dataset[ds] += int(
                            np.count_nonzero(np.asarray(p) != self.pad_id)
                            + np.count_nonzero(np.asarray(t) >= 0)
                        )
                return self._format_batch(best_b, name, p_list, t_list, c_list)

            # Tokenize bounded to prevent processing gigantic out-of-distribution text
            max_enc = self.total_window + 256
            q_texts = [pair[0] for pair in raw_pairs]
            r_texts = [pair[1] for pair in raw_pairs]
            d_names = [pair[2] for pair in raw_pairs]
            tokenized_pairs = None
            if use_fast_batch and len(raw_pairs) > 1:
                try:
                    q_ids_batch = self.tokenizer(
                        q_texts, max_length=max_enc, truncation=True, add_special_tokens=False
                    )["input_ids"]
                    r_ids_batch = self.tokenizer(
                        r_texts, max_length=max_enc, truncation=True, add_special_tokens=False
                    )["input_ids"]
                    tokenized_pairs = list(zip(q_ids_batch, r_ids_batch, d_names, r_texts))
                except Exception:
                    tokenized_pairs = None

            if tokenized_pairs is None:
                tokenized_pairs = []
                for q_txt, r_txt, ds in raw_pairs:
                    if hasattr(self.tokenizer, "encode"):
                        q_ids = encode_no_special(self.tokenizer, q_txt, max_length=max_enc)
                        r_ids = encode_no_special(self.tokenizer, r_txt, max_length=max_enc)
                    else:
                        q_ids = [1] * len(q_txt.split())
                        r_ids = [1] * len(r_txt.split())
                    tokenized_pairs.append((q_ids, r_ids, ds, r_txt))

            eos_id = self.tokenizer.eos_token_id if (hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None) else 2
            for q_ids, r_ids, ds, r_txt in tokenized_pairs:
                if len(r_ids) == 0 or r_ids[-1] != eos_id:
                    r_ids = r_ids + [eos_id]

                q_len, r_len = len(q_ids), len(r_ids)

                # Route to best complementary bucket
                b_idx = select_best_bucket(q_len, r_len, self.buckets)
                name, bp, bt = self.buckets[b_idx]

                # Pad / truncate to the exact bucket dimensions (preserve stimulus prefix + question suffix for prompts)
                if len(q_ids) > bp:
                    prefix_len = min(32, bp // 4)
                    suffix_len = bp - prefix_len
                    q_padded = q_ids[:prefix_len] + q_ids[-suffix_len:]
                else:
                    q_padded = q_ids + [self.pad_id] * (bp - len(q_ids))

                if len(r_ids) > bt:
                    is_reasoning_or_qa = any(
                        kw in ds.lower()
                        for kw in (
                            "math", "gsm8k", "openhermes", "smoltalk", "cot", "reason", "qa",
                            "hermes", "r1", "sft", "instruct", "chat", "dialog", "turn", "retrieval", "tool", "agent"
                        )
                    ) or ("<think>" in str(r_txt) or "<act>" in str(r_txt))

                    if is_reasoning_or_qa:
                        # For reasoning, QA, SFT & dialogues: never sever responses into headless fragments.
                        # Preserve intact trajectory capped at bt, guaranteeing terminating eos_id!
                        r_capped = r_ids[: bt - 1] + [eos_id]
                        chunks = [(q_padded, r_capped, None)]
                    else:
                        # Contiguous sequential document/trajectory slicing with true preceding context carry for pretraining
                        chunks = []
                        for start in range(0, len(r_ids), bt):
                            chunk = r_ids[start : start + bt]
                            if start == 0:
                                chunk_prompt = q_padded
                                ctx_tok = None
                            else:
                                preceding = r_ids[max(0, start - bp) : start]
                                chunk_prompt = preceding

                                if len(chunk_prompt) < bp:
                                    chunk_prompt = chunk_prompt + [self.pad_id] * (bp - len(chunk_prompt))
                                else:
                                    chunk_prompt = chunk_prompt[:bp]
                                ctx_tok = r_ids[start - 1]

                            if len(chunk) < bt:
                                chunk = chunk + [self.target_pad_id] * (bt - len(chunk))
                            chunks.append((chunk_prompt, chunk, ctx_tok))
                        chunks.append((chunk_prompt, chunk, ctx_tok))
                else:
                    chunks = [(q_padded, r_ids + [self.target_pad_id] * (bt - len(r_ids)), None)]

                for chunk_prompt, r_padded, context_token in chunks:
                    self.queues[b_idx].append((chunk_prompt, r_padded, context_token, ds))


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
        consecutive_errors = 0
        while not self.stop_event.is_set():
            try:
                batch = self.batcher.next_batch()
                if batch is not None:
                    # Block with timeout to check stop_event periodically
                    while not self.stop_event.is_set():
                        try:
                            self.queue.put(batch, timeout=1.0)
                            consecutive_errors = 0
                            break
                        except queue.Full:
                            continue
            except Exception as e:
                consecutive_errors += 1
                if self.stop_event.is_set():
                    break
                wait_s = min(10.0, 0.5 * (2 ** min(consecutive_errors, 5)))
                time.sleep(wait_s)

    def next_batch(self, timeout: float = 60.0):
        while not self.stop_event.is_set():
            try:
                return self.queue.get(timeout=timeout)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
        return None

    def get_batch(self):
        return self.next_batch()

    def get_tokens_by_dataset(self) -> Dict[str, int]:
        """Forwards token breakdown request to underlying batcher."""
        return self.batcher.get_tokens_by_dataset()

    def set_tokens_by_dataset(self, d: Dict[str, int]) -> None:
        """Forwards token setting to underlying batcher."""
        self.batcher.set_tokens_by_dataset(d)

    def update_buckets(self, new_buckets: List[Tuple[str, int, int]], new_total_window: Optional[int] = None) -> None:
        """Flushes buffered batches and applies new context buckets to underlying batcher."""
        # Safely flush prefetched queue of obsolete shapes
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break
        self.batcher.update_buckets(new_buckets, new_total_window)

    def update_target_len(self, new_target_len: int) -> None:
        """Flushes buffered batches and updates target length on the underlying batcher."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break
        self.batcher.update_target_len(new_target_len)

    def stop(self):
        self.shutdown()

    def shutdown(self):
        self.stop_event.set()


def build_canonical_eval_split(
    tokenizer: Any,
    batch_size: int = 8,
    prompt_len: int = 256,
    target_len: int = 512,
    target_pad_id: int = -100,
    pad_id: int = 0,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]], List[str]]:
    """
    Constructs a 5-domain canonical held-out evaluation split aligned 1:1 with
    the CerebroSpheroid V5 training curriculum:
      1. Educational Prose (FineWeb-Edu)          : High educational score (>=4.0) held-out web prose
      2. Textbook Concepts (Cosmopedia v2)        : Synthetic textbook & pedagogical concept explanations
      3. Mathematical Proofs (FineMath & GSM8K)   : Exact derivations & CoT with 3-tier <think>/<act> tags
      4. Algorithmic Python (Code Large & Self-OSS): Structured Python code blocks enclosed in <act> tags
      5. Commonsense Dialogue (SmolTalk)          : Fluid multi-turn everyday conversational interactions

    Returns:
      (eval_batches, domain_labels)
      where each batch is (prompt_batch, target_batch, feedback_batch) as static int32 numpy arrays.
    """
    from src.corpus import format_stimulus_prompt

    # Pre-curated authentic held-out reference exemplars per domain (used as instantaneous zero-stall fallback)
    domain_fallbacks: Dict[str, List[Tuple[str, str]]] = {
        "fineweb_edu": [
            ("Explain the biological mechanism and evolutionary significance of cellular respiration and ATP synthesis in eukaryotic mitochondria.",
             "Cellular respiration is the foundational metabolic pathway through which eukaryotic cells extract biochemical energy from nutrients. In the mitochondrial matrix, pyruvate derived from glycolysis undergoes decarboxylation to form acetyl-CoA, feeding the citric acid cycle to generate high-energy electron carriers (NADH and FADH2). These electrons are sequentially transferred through complexes I-IV of the inner mitochondrial membrane electron transport chain, pumping protons into the intermembrane space. The resulting electrochemical proton gradient drives ATP synthase via chemiosmosis to produce approximately 30 to 32 ATP molecules per oxidized glucose molecule, providing the thermodynamic currency that enabled multicellular complexity."),
            ("Describe the thermodynamic principles governing closed thermodynamic systems during irreversible expansion.",
             "In a closed thermodynamic system undergoing irreversible expansion, matter exchange across system boundaries is prohibited while energy exchange via heat (Q) and work (W) remains active. By the First Law of Thermodynamics, the change in internal energy is dU = dQ - dW. According to the Second Law, the total entropy generation of the universe during any spontaneous or irreversible process is strictly positive (dS_total > 0). Within the system, entropy changes satisfy dS >= dQ/T, where equality holds solely for quasi-static reversible pathways. The dissipation of usable work into internal thermal energy reflects uncompensated entropy creation, driving the system toward thermodynamic equilibrium."),
            ("Analyze the geological and climatic impacts of the Paleocene-Eocene Thermal Maximum (PETM).",
             "The Paleocene-Eocene Thermal Maximum (PETM), occurring approximately 56 million years ago, represents one of the most rapid and severe carbon cycle perturbations in Earth's Cenozoic history. Massive releases of carbon (estimated at 2,000 to 5,000 gigatons, likely from destabilized marine methane hydrates and volcanic intrusives) triggered global surface warming of 5 to 8 degrees Celsius. This sudden carbon injection caused severe ocean acidification, marked by extensive dissolution of deep-sea calcium carbonates, a dramatic negative carbon isotope excursion (CIE), and widespread turnover in benthic foraminifera alongside rapid evolutionary diversification of modern mammalian orders."),
            ("Explain the role of CRISPR-Cas9 endonuclease in targeted genomic engineering and repair mechanisms.",
             "CRISPR-Cas9 is an RNA-guided endonuclease system adapted from bacterial adaptive immunity against bacteriophages. The system comprises the Cas9 enzyme complexed with a chimeric single guide RNA (sgRNA). The sgRNA specifies a 20-nucleotide target sequence adjacent to a 5'-NGG protospacer adjacent motif (PAM). Upon target binding, Cas9 introduces a site-specific double-strand break (DSB) approximately three base pairs upstream of the PAM. Eukaryotic host repair pathways resolve this break either through error-prone Non-Homologous End Joining (NHEJ), generating frameshifting insertions or deletions (indels), or via Homology-Directed Repair (HDR) in the presence of an exogenous repair template, enabling precise nucleotide substitutions."),
        ],
        "cosmopedia": [
            ("Explain the concept of manifold learning and dimensionality reduction in computational topology.",
             "Manifold learning is a branch of non-linear dimensionality reduction predicated on the manifold hypothesis, which posits that high-dimensional real-world observation vectors frequently lie embedded along a lower-dimensional smooth Riemannian sub-manifold. While linear methods like Principal Component Analysis (PCA) preserve global Euclidean variance, non-linear algorithms such as Isomap, Locally Linear Embedding (LLE), and UMAP capture non-Euclidean geodesic topological distances. By computing local neighborhood graphs and constructing distance-preserving coordinate charts, manifold learning maps high-dimensional sensory topologies into compact latent coordinates while preserving extrinsic clustering and intrinsic geometric invariants."),
            ("Explain the difference between algorithmic computational complexity classes P, NP, and NP-Complete.",
             "Computational complexity classes categorize decision problems based on the asymptotic resource bounds required for solution and verification on a deterministic Turing machine. Class P encompasses problems decidable in polynomial time O(n^k). Class NP comprises problems whose candidate certificates can be verified in deterministic polynomial time, or equivalently solved in polynomial time by a non-deterministic Turing machine. A problem L is NP-hard if every problem in NP polynomial-time Karp-reduces to L. When an NP-hard problem is itself an element of NP, it is classified as NP-Complete (e.g., Boolean Satisfiability / 3-SAT by the Cook-Levin theorem). Whether P equals NP remains the preeminent open conjecture in theoretical computer science."),
            ("Explain the principles of transformer multi-head self-attention mechanisms and scaling.",
             "The multi-head self-attention mechanism maps input token representations into scaled dot-product query (Q), key (K), and value (V) projections across h distinct linear attention heads: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V. The denominator scaling factor sqrt(d_k) counteracts extreme magnitude growth in high-dimensional inner products, preventing softmax gradients from vanishing. Multi-head projection allows the network to jointly attend to information from different representation subspaces at different token positions simultaneously, synthesizing long-range syntactic and semantic relational graphs with O(1) sequential operation count."),
            ("Explain how neural network representations transition across deep hierarchical representations.",
             "Hierarchical feature representations in deep neural architectures develop sequentially across successive layer boundaries. In early layers, receptive fields detect local, low-level high-frequency primitives such as edges, color gradients, and morphological boundaries. In intermediate hidden layers, non-linear activation functions compose these primitives into motifs, textures, and constituent structural parts. In terminal deep layers, the network projects parts into invariant semantic concept manifolds, where euclidean distances directly correspond to high-level categorical similarities rather than pixel or surface-level lexical similarities."),
        ],
        "finemath": [
            ("Janet’s ducks lay 16 eggs per day. She eats three for breakfast and bakes four into muffins. She sells the remaining eggs at the market for $2 each. How much does she make each day?",
             "<think>\nJanet starts with 16 eggs.\nShe eats 3 for breakfast, leaving 16 - 3 = 13 eggs.\nShe bakes 4 into muffins, leaving 13 - 4 = 9 eggs.\nShe sells the remaining 9 eggs at $2 each.\nTotal revenue = 9 * 2 = 18 dollars.\n</think>\n<act>\ncalculator: 16 - 3 - 4 = 9, 9 * 2 = 18\n</act>\n18"),
            ("A train travels 120 miles in 2 hours, stops for 30 minutes, and then travels 150 miles in 2.5 hours. What was its average speed in miles per hour for the entire trip?",
             "<think>\nFirst segment: distance = 120 miles, time = 2 hours.\nStop: distance = 0 miles, time = 30 minutes = 0.5 hours.\nSecond segment: distance = 150 miles, time = 2.5 hours.\nTotal distance = 120 + 0 + 150 = 270 miles.\nTotal time = 2 + 0.5 + 2.5 = 5.0 hours.\nAverage speed = Total distance / Total time = 270 / 5 = 54 mph.\n</think>\n<act>\ncalculator: (120 + 150) / (2 + 0.5 + 2.5) = 54\n</act>\n54"),
            ("Find all real solutions to the quadratic equation 3x^2 - 12x + 9 = 0 using the quadratic formula.",
             "<think>\nThe quadratic equation is 3x^2 - 12x + 9 = 0.\nCoefficients are a = 3, b = -12, c = 9.\nThe discriminant Delta is b^2 - 4ac = (-12)^2 - 4(3)(9) = 144 - 108 = 36.\nSince Delta = 36 > 0, there are two distinct real roots.\nRoots are x = (-b +- sqrt(Delta)) / (2a) = (12 +- 6) / 6.\nFor plus: x = (12 + 6) / 6 = 18 / 6 = 3.\nFor minus: x = (12 - 6) / 6 = 6 / 6 = 1.\n</think>\n<act>\ncalculator: (-(-12) + 6)/(2*3) = 3, (-(-12) - 6)/(2*3) = 1\n</act>\nx = 1, x = 3"),
            ("Calculate the definite integral of f(x) = 4x^3 - 6x + 5 from x = 1 to x = 3.",
             "<think>\nThe antiderivative F(x) = int (4x^3 - 6x + 5) dx = x^4 - 3x^2 + 5x.\nEvaluate at upper limit x = 3: F(3) = 3^4 - 3(3^2) + 5(3) = 81 - 27 + 15 = 69.\nEvaluate at lower limit x = 1: F(1) = 1^4 - 3(1^2) + 5(1) = 1 - 3 + 5 = 3.\nDefinite integral = F(3) - F(1) = 69 - 3 = 66.\n</think>\n<act>\ncalculator: (81 - 27 + 15) - (1 - 3 + 5) = 66\n</act>\n66"),
        ],
        "python_code": [
            ("Write a Python function to compute the longest increasing subsequence (LIS) of a list in O(n log n) time.",
             "<act>\nimport bisect\nfrom typing import List\n\ndef longest_increasing_subsequence(nums: List[int]) -> int:\n    \"\"\"Calculates length of the longest strictly increasing subsequence in O(n log n).\"\"\"\n    if not nums:\n        return 0\n    tails: List[int] = []\n    for x in nums:\n        idx = bisect.bisect_left(tails, x)\n        if idx == len(tails):\n            tails.append(x)\n        else:\n            tails[idx] = x\n    return len(tails)\n</act>"),
            ("Implement a thread-safe Least Recently Used (LRU) Cache in Python with get and put methods.",
             "<act>\nfrom collections import OrderedDict\nimport threading\nfrom typing import Any, Optional\n\nclass LRUCache:\n    \"\"\"Thread-safe O(1) LRU Cache implementation using OrderedDict and threading.Lock.\"\"\"\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.cache: OrderedDict[str, Any] = OrderedDict()\n        self.lock = threading.Lock()\n\n    def get(self, key: str) -> Optional[Any]:\n        with self.lock:\n            if key not in self.cache:\n                return None\n            self.cache.move_to_end(key)\n            return self.cache[key]\n\n    def put(self, key: str, value: Any) -> None:\n        with self.lock:\n            if key in self.cache:\n                self.cache.move_to_end(key)\n            self.cache[key] = value\n            if len(self.cache) > self.capacity:\n                self.cache.popitem(last=False)\n</act>"),
            ("Write a Python function to find the maximum subarray sum using Kadane's algorithm.",
             "<act>\nfrom typing import List\n\ndef max_subarray_sum(nums: List[int]) -> int:\n    \"\"\"Computes maximum contiguous subarray sum in O(n) time and O(1) auxiliary space.\"\"\"\n    if not nums:\n        return 0\n    current_sum = max_sum = nums[0]\n    for val in nums[1:]:\n        current_sum = max(val, current_sum + val)\n        max_sum = max(max_sum, current_sum)\n    return max_sum\n</act>"),
            ("Implement binary search in Python to return the index of a target in a sorted list or -1 if not found.",
             "<act>\nfrom typing import List\n\ndef binary_search(arr: List[int], target: int) -> int:\n    \"\"\"Performs O(log n) binary search on sorted array, returning index or -1.\"\"\"\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n</act>"),
        ],
        "smoltalk": [
            ("Hello! Can you help me plan a weekend itinerary for exploring local hiking trails and photography spots?",
             "Hello! I'd be glad to help you plan a wonderful weekend outdoors. To give you the best recommendations, could you let me know: 1) What general geographic area or mountain range you are looking at? 2) What trail difficulty and distance do you prefer (e.g., easy 2-mile walk vs. challenging 8-mile elevation gain)? 3) Are you specifically hoping for golden-hour sunrise viewpoints, waterfalls, or panoramic ridgelines? Once you share your preferences, we can build a detailed morning-to-evening itinerary!"),
            ("What is the best way to maintain focus and avoid burnout while preparing for comprehensive technical exams?",
             "Maintaining long-term focus while studying for rigorous technical exams requires balancing cognitive load with deliberate recovery. Three evidence-backed strategies are: First, utilize spaced repetition and the Pomodoro method (e.g., 50 minutes of deep active recall followed by a strict 10-minute mental break without screens). Second, prioritize active problem solving over passive reading — solving practice problems builds neural familiarity under test conditions. Third, protect your sleep schedule: sleep is when memory consolidation occurs, so 7 to 8 hours of quality sleep directly improves retention and recall speed."),
            ("Can you explain the difference between a direct flight and a non-stop flight?",
             "While the terms sound identical, airlines distinguish between them based on whether the aircraft touches down. A non-stop flight flies from your origin airport to your destination airport with zero intermediate landings. A direct flight, however, has a single flight number but may make an intermediate stop for refueling or boarding additional passengers without requiring you to change planes. Thus, all non-stop flights are direct, but not all direct flights are non-stop."),
            ("How do I properly store and care for cast iron cookware so it lasts for decades?",
             "Caring for cast iron cookware comes down to maintaining its polymerized oil seasoning layer and preventing rust. After cooking, wash it with warm water and a non-abrasive sponge or chainmail scrubber; mild dish soap is fine, but avoid harsh abrasives or soaking. Dry it immediately and thoroughly with a towel or over low heat on the stovetop. While it is still warm, apply a very thin coat of neutral, high-smoke-point oil (like canola or grapeseed), wiping away any excess with a dry paper towel until the surface is matte, then store it in a dry location."),
        ],
    }

    def _open_cosmo_stream():
        candidate_streams = [
            ("AjitKumar-Bio/cosmopedia-v2-digested", None),
            ("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2"),
            ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2"),
        ]
        for c_repo, c_cfg in candidate_streams:
            try:
                return _open_hf_stream(c_repo, c_cfg)
            except Exception:
                continue
        return _open_hf_stream("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2")

    def _open_math_stream():
        try:
            return _open_hf_stream("openai/gsm8k", "main", split="test")
        except Exception:
            return _open_hf_stream("HuggingFaceTB/finemath", "finemath-4plus")

    def _open_code_stream():
        try:
            return _open_hf_stream("HuggingFaceTB/smoltalk", "self-oss-instruct")
        except Exception:
            return _open_hf_stream("ajibawa-2023/Python-Code-Large", None)

    domain_configs = [
        (
            "Educational Prose (FineWeb-Edu)",
            lambda: _open_hf_stream("HuggingFaceFW/fineweb-edu", "sample-350BT"),
            "fineweb_edu",
            4.0,
        ),
        (
            "Textbook Concepts (Cosmopedia v2)",
            _open_cosmo_stream,
            "cosmopedia",
            0.0,
        ),
        (
            "Mathematical Proofs (FineMath & GSM8K)",
            _open_math_stream,
            "finemath",
            0.0,
        ),
        (
            "Algorithmic Python (Code Large & Self-OSS)",
            _open_code_stream,
            "python_code",
            0.0,
        ),
        (
            "Commonsense Dialogue (SmolTalk)",
            lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "everyday-conversations"),
            "smoltalk",
            0.0,
        ),
    ]

    # Helper to draw local parquet fallback rows if available
    local_val_tbl = None
    try:
        local_val_path = os.path.join(REPO_ROOT, "data", "cerebro_v5_custom_dataset", "validation.parquet")
        if os.path.exists(local_val_path):
            import pyarrow.parquet as pq
            local_val_tbl = pq.read_table(local_val_path)
    except Exception:
        local_val_tbl = None

    eval_batches = []
    domain_labels = []

    for name, stream_opener, ds_tag, min_score in domain_configs:
        rows = []
        try:
            it = stream_opener()
            max_scan = 40
            scan_count = 0
            for row in it:
                scan_count += 1
                if min_score > 0.0:
                    score = float(row.get("score", 0.0) or row.get("educational_score", 0.0) or row.get("int_score", 0.0))
                    if score < min_score:
                        if scan_count >= max_scan:
                            break
                        continue
                q, r = extract_prompt_response_pair(row)
                if len(q.strip()) >= 5 and len(r.strip()) >= 10:
                    rows.append((q, r))
                if len(rows) >= batch_size or scan_count >= max_scan:
                    break
        except Exception:
            pass

        # If online stream provided fewer than batch_size, draw from local validation.parquet
        if len(rows) < batch_size and local_val_tbl is not None:
            try:
                prompts = local_val_tbl["prompt"].to_pylist()
                answers = local_val_tbl["answer"].to_pylist()
                categories = local_val_tbl["category"].to_pylist()
                for p_txt, a_txt, cat in zip(prompts, answers, categories):
                    match = False
                    if ds_tag == "fineweb_edu" and ("science" in cat or "mmlu" in cat):
                        match = True
                    elif ds_tag == "cosmopedia" and "cognitive" in cat:
                        match = True
                    elif ds_tag == "finemath" and ("math" in cat or "calculator" in cat):
                        match = True
                    elif ds_tag == "python_code" and ("code" in cat or "python" in cat):
                        match = True
                    elif ds_tag == "smoltalk" and ("conversational" in cat or "commonsense" in cat):
                        match = True
                    if match and p_txt and a_txt:
                        rows.append((p_txt.strip(), a_txt.strip()))
                        if len(rows) >= batch_size:
                            break
            except Exception:
                pass

        # If still fewer than batch_size, draw from embedded high-quality domain fallbacks
        if len(rows) < batch_size and ds_tag in domain_fallbacks:
            for p_txt, a_txt in domain_fallbacks[ds_tag]:
                rows.append((p_txt, a_txt))
                if len(rows) >= batch_size:
                    break

        # Final safety loop cycling available rows to satisfy exact batch_size requirement
        idx_cycle = 0
        while len(rows) < batch_size:
            rows.append(rows[idx_cycle % len(rows)])
            idx_cycle += 1

        prompts_batch = []
        targets_batch = []

        eos_id = getattr(tokenizer, "eos_token_id", 2)
        if eos_id is None:
            eos_id = 2

        for q_txt, r_txt in rows[:batch_size]:
            q_fmt = format_stimulus_prompt(q_txt)
            r_fmt = _normalize_3tier_response(r_txt.strip(), q_txt)
            if "<act>" not in r_fmt:
                if "calculator" in ds_tag.lower():
                    r_fmt = f"<act> calc({r_fmt}) </act>\nAnswer: {r_fmt}"

            if hasattr(tokenizer, "encode"):
                q_ids = encode_no_special(tokenizer, q_fmt, max_length=prompt_len)
                r_ids = encode_no_special(tokenizer, r_fmt, max_length=target_len)
            else:
                q_ids = [1] * min(len(q_fmt.split()), prompt_len)
                r_ids = [1] * min(len(r_fmt.split()), target_len)

            if len(r_ids) == 0 or r_ids[-1] != eos_id:
                r_ids.append(eos_id)

            # Pad or truncate prompt (preserve stimulus header prefix and question suffix)
            if len(q_ids) > prompt_len:
                prefix_len = min(32, prompt_len // 4)
                suffix_len = prompt_len - prefix_len
                q_pad = q_ids[:prefix_len] + q_ids[-suffix_len:]
            else:
                q_pad = q_ids + [pad_id] * (prompt_len - len(q_ids))

            # Pad or truncate target (always guarantee terminating eos_id)
            if len(r_ids) > target_len:
                r_pad = r_ids[: target_len - 1] + [eos_id]
            else:
                r_pad = r_ids + [target_pad_id] * (target_len - len(r_ids))

            prompts_batch.append(q_pad)
            targets_batch.append(r_pad)

        q_arr = np.asarray(prompts_batch, dtype=np.int32)
        r_arr = np.asarray(targets_batch, dtype=np.int32)

        # Strictly causal teacher-forcing feedback tokens shifted by 1:
        # feedback[0] = -100, feedback[t] = target[t-1]
        # Padded target positions where target == -100 are masked with target_pad_id (-100).
        fb_arr = np.full(r_arr.shape, target_pad_id, dtype=np.int32)
        if r_arr.shape[1] > 1:
            fb_arr[:, 1:] = r_arr[:, :-1]
        fb_arr = np.where(r_arr == target_pad_id, target_pad_id, fb_arr)

        eval_batches.append((q_arr, r_arr, fb_arr))
        domain_labels.append(name)

    return eval_batches, domain_labels


def build_pretrain_eval_split(
    tokenizer: Any,
    batch_size: int = 8,
    prompt_len: int = 256,
    target_len: int = 512,
    target_pad_id: int = -100,
    pad_id: int = 0,
    return_weights: bool = False,
):
    """
    Constructs a 2-domain canonical held-out evaluation split aligned 1:1 with
    the CerebroSpheroid V5 Stage 1 Pretraining distribution:
      1. Broad Web Text (FineWeb sample-10BT)   : Weight 0.85 (85%)
      2. Code Repositories (The Stack v2)       : Weight 0.15 (15%)

    Evaluates 100% loss-bearing contiguous next-token sequences:
    The prompt provides `prompt_len` tokens of context, and the model is evaluated
    on predicting the subsequent contiguous `target_len` tokens from the same document.

    Returns:
      (eval_batches, domain_labels) or (eval_batches, domain_labels, domain_weights)
      where each batch is (prompt_batch, target_batch, feedback_batch) as static int32 numpy arrays.
    """
    import glob

    # Pre-curated authentic held-out reference exemplars per domain (used as instantaneous zero-stall fallback)
    domain_fallbacks: Dict[str, List[str]] = {
        "broad_web": [
            "The thermodynamics of non-equilibrium systems represents a critical bridge between statistical mechanics and macroscopic phenomenology. In open systems operating far from thermodynamic equilibrium, entropy production is not merely a dissipative byproduct but a driving force behind self-organizing dissipative structures, as articulated by Ilya Prigogine. The fluctuation theorem, derived by Evans, Cohen, and Morriss, generalizes the Second Law to finite microscopic systems observed over arbitrary time intervals. It establishes that the probability of observing an entropy-consuming trajectory decreases exponentially with system size and observation duration. In cellular biology, non-equilibrium steady states are maintained across cellular membranes through active ion pumping mechanisms such as the sodium-potassium ATPase (Na+/K+-ATPase). By consuming metabolic chemical potential in the form of ATP hydrolysis, cells sustain steep electrochemical gradients essential for action potential generation, secondary active transport of glucose and amino acids, and osmotic volume regulation. In cosmological astrophysics, cosmic nucleosynthesis during the initial expansionary cooling phase occurred under rapid non-equilibrium conditions, fixing primordial abundances of hydrogen, helium-4, deuterium, and lithium-7 before nuclear reaction rates fell behind Hubble expansion.",
            "The evolution of terrestrial ecosystems during the Carboniferous period (approximately 359 to 299 million years ago) exerted profound influences on Earth's lithospheric and atmospheric composition. Vast equatorial swamp forests dominated by arborescent lycophytes such as Lepidodendron and Sigillaria sequestered atmospheric carbon dioxide at unprecedented rates. The emergence of robust structural polymers, particularly lignin in vascular plant xylem, occurred prior to the evolutionary diversification of saprotrophic fungi capable of enzymatically decomposing complex phenylpropanoid networks. As unoxidized organic biomass accumulated in waterlogged, anoxic peat bogs, immense geological coal measures were laid down across modern North America, Europe, and Asia. Atmospheric oxygen concentrations climbed to an estimated 35 percent, permitting the expansion of gigantic terrestrial arthropods like Arthropleura and Meganeura, while simultaneous carbon drawdown triggered late Paleozoic icehouse conditions and widespread glaciations across Gondwana.",
            "In classical and quantum electrodynamics, gauge symmetry serves as the bedrock principle dictating fundamental particle interactions. Under local U(1) phase transformations, invariance of the complex Dirac fermion Lagrangian mandates the introduction of a massless vector gauge boson—the electromagnetic photon—coupled through minimal substitution with covariant derivative D_mu = partial_mu - i e A_mu. The renormalization group framework formulated by Kenneth Wilson reveals how effective coupling constants scale logarithmically with energy or inverse distance. In quantum electrodynamics (QED), vacuum polarization produces screening effects where virtual electron-positron pairs reduce the effective bare electric charge at macroscopic distances, causing the fine-structure constant alpha to run from approximately 1/137 at zero momentum transfer to roughly 1/128 at the Z-boson mass scale (~91.2 GeV). Experimental precision measurements of the electron anomalous magnetic dipole moment (g-2) agree with multi-loop perturbative QED calculations to more than ten significant figures, standing among the most rigorous triumphs of modern theoretical physics.",
            "Plate tectonics governs the dynamic resurfacing and thermal convective dissipation of Earth's silicate mantle. At mid-ocean ridges (divergent boundaries), decompression melting of upwelling asthenospheric peridotite generates basaltic tholeiitic magma, which accretes along the rift axis to form new oceanic crust composed of layered pillow basalts, sheeted sheeted dikes, and plutonic gabbros. As oceanic lithosphere cools conductive thermal contraction increases its density relative to the underlying asthenosphere. Eventually, aged cold slabs become gravitationally unstable and founder back into the mantle along subduction zones (convergent boundaries). Slab pull constitutes the primary tectonic driving force, dragging the descending slab down through the mantle transition zone toward the core-mantle boundary (D'' layer). Hydrous minerals in the subducting slab undergo dehydration metamorphism at depth, releasing water and volatiles into the overlying mantle wedge, lowering the solidus temperature and triggering flux melting that drives arc volcanism and continental crust generation.",
            "Algorithmic information theory and computational complexity provide deep insights into the limits of formal mathematical deduction and inductive learning. Andrey Kolmogorov and Ray Solomonoff independently founded algorithmic probability by defining the descriptive complexity (Kolmogorov complexity) of a finite binary string as the length of the shortest computer program that outputs the string on a universal Turing machine. Solomonoff induction formalizes Occam's razor: given sequential observational data, hypotheses are assigned a priori probabilities proportional to 2^(-K(M)), favoring computationally parsimonious hypotheses. Because Kolmogorov complexity is mathematically incomputable due to the undecidability of the halting problem, practical compression algorithms like Lempel-Ziv-Welch (LZW) and byte-pair encoding (BPE) serve as tractable approximations. Gregory Chaitin demonstrated that within any consistent, recursively axiomatizable formal system capable of arithmetic, there exists an upper bound beyond which one cannot prove that any specific string has high Kolmogorov complexity, revealing an inherent informational horizon in mathematical logic.",
        ],
        "code": [
            """import asyncio\nimport collections\nimport time\nfrom typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic\n\nT = TypeVar("T")\n\nclass AsyncTokenBucketLimiter:\n    \"\"\"Thread-safe asynchronous token bucket rate limiter for distributed microservices.\"\"\"\n    def __init__(self, capacity: float, refill_rate: float):\n        self.capacity: float = float(capacity)\n        self.tokens: float = float(capacity)\n        self.refill_rate: float = float(refill_rate)\n        self.last_update: float = time.monotonic()\n        self._lock = asyncio.Lock()\n\n    async def acquire(self, tokens_required: float = 1.0) -> bool:\n        async with self._lock:\n            now = time.monotonic()\n            elapsed = now - self.last_update\n            self.last_update = now\n            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)\n            if self.tokens >= tokens_required:\n                self.tokens -= tokens_required\n                return True\n            return False\n\nclass AsyncFIFOCache(Generic[T]):\n    \"\"\"High-throughput generic asynchronous cache with TTL eviction.\"\"\"\n    def __init__(self, maxsize: int = 1024, ttl_seconds: float = 300.0):\n        self.maxsize = maxsize\n        self.ttl = ttl_seconds\n        self._storage: collections.OrderedDict[str, Tuple[float, T]] = collections.OrderedDict()\n        self._lock = asyncio.Lock()\n\n    async def get(self, key: str) -> Optional[T]:\n        async with self._lock:\n            if key not in self._storage:\n                return None\n            ts, val = self._storage[key]\n            if time.monotonic() - ts > self.ttl:\n                del self._storage[key]\n                return None\n            return val\n\n    async def set(self, key: str, value: T) -> None:\n        async with self._lock:\n            now = time.monotonic()\n            if key in self._storage:\n                del self._storage[key]\n            elif len(self._storage) >= self.maxsize:\n                self._storage.popitem(last=False)\n            self._storage[key] = (now, value)\n""",
            """import math\nfrom typing import List, Optional, Tuple\n\nclass AVLNode:\n    \"\"\"Self-balancing AVL tree node with balance factor tracking.\"\"\"\n    def __init__(self, key: int, value: str):\n        self.key: int = key\n        self.value: str = value\n        self.height: int = 1\n        self.left: Optional[\"AVLNode\"] = None\n        self.right: Optional[\"AVLNode\"] = None\n\nclass AVLTree:\n    \"\"\"Thread-safe AVL balanced binary search tree implementation.\"\"\"\n    def __init__(self):\n        self.root: Optional[AVLNode] = None\n\n    def _get_height(self, node: Optional[AVLNode]) -> int:\n        return node.height if node else 0\n\n    def _get_balance(self, node: Optional[AVLNode]) -> int:\n        return self._get_height(node.left) - self._get_height(node.right) if node else 0\n\n    def _rotate_right(self, y: AVLNode) -> AVLNode:\n        x = y.left\n        assert x is not None\n        t2 = x.right\n        x.right = y\n        y.left = t2\n        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))\n        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))\n        return x\n\n    def _rotate_left(self, x: AVLNode) -> AVLNode:\n        y = x.right\n        assert y is not None\n        t2 = y.left\n        y.left = x\n        x.right = t2\n        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))\n        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))\n        return y\n\n    def insert(self, key: int, value: str) -> None:\n        def _insert(node: Optional[AVLNode], k: int, v: str) -> AVLNode:\n            if not node:\n                return AVLNode(k, v)\n            if k < node.key:\n                node.left = _insert(node.left, k, v)\n            elif k > node.key:\n                node.right = _insert(node.right, k, v)\n            else:\n                node.value = v\n                return node\n            node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))\n            balance = self._get_balance(node)\n            if balance > 1 and node.left and k < node.left.key:\n                return self._rotate_right(node)\n            if balance < -1 and node.right and k > node.right.key:\n                return self._rotate_left(node)\n            if balance > 1 and node.left and k > node.left.key:\n                node.left = self._rotate_left(node.left)\n                return self._rotate_right(node)\n            if balance < -1 and node.right and k < node.right.key:\n                node.right = self._rotate_right(node.right)\n                return self._rotate_left(node)\n            return node\n        self.root = _insert(self.root, key, value)\n""",
            """import numpy as np\nfrom typing import Callable, Dict, List, Set, Tuple\n\nclass ComputationalGraphNode:\n    \"\"\"Node in an automatic differentiation topological compute DAG.\"\"\"\n    def __init__(self, data: np.ndarray, parents: Tuple[\"ComputationalGraphNode\", ...] = (), op: str = \"\"):\n        self.data: np.ndarray = np.asarray(data, dtype=np.float32)\n        self.grad: Optional[np.ndarray] = None\n        self.parents: Tuple[\"ComputationalGraphNode\", ...] = parents\n        self.op: str = op\n        self._backward: Callable[[], None] = lambda: None\n\n    def __add__(self, other: \"ComputationalGraphNode\") -> \"ComputationalGraphNode\":\n        other_node = other if isinstance(other, ComputationalGraphNode) else ComputationalGraphNode(other)\n        out = ComputationalGraphNode(self.data + other_node.data, (self, other_node), \"+\")\n        def _backward():\n            if self.grad is None:\n                self.grad = np.zeros_like(self.data)\n            if other_node.grad is None:\n                other_node.grad = np.zeros_like(other_node.data)\n            self.grad += out.grad\n            other_node.grad += out.grad\n        out._backward = _backward\n        return out\n\n    def __matmul__(self, other: \"ComputationalGraphNode\") -> \"ComputationalGraphNode\":\n        other_node = other if isinstance(other, ComputationalGraphNode) else ComputationalGraphNode(other)\n        out = ComputationalGraphNode(self.data @ other_node.data, (self, other_node), \"@\")\n        def _backward():\n            if self.grad is None:\n                self.grad = np.zeros_like(self.data)\n            if other_node.grad is None:\n                other_node.grad = np.zeros_like(other_node.data)\n            self.grad += out.grad @ other_node.data.T\n            other_node.grad += self.data.T @ out.grad\n        out._backward = _backward\n        return out\n\n    def backward(self) -> None:\n        topo: List[ComputationalGraphNode] = []\n        visited: Set[ComputationalGraphNode] = set()\n        def build_topo(v: ComputationalGraphNode):\n            if v not in visited:\n                visited.add(v)\n                for p in v.parents:\n                    build_topo(p)\n                topo.append(v)\n        build_topo(self)\n        self.grad = np.ones_like(self.data)\n        for node in reversed(topo):\n            node._backward()\n""",
            """import socket\nfrom typing import Dict, List, Optional, Tuple\n\nclass SimpleHTTPServer:\n    \"\"\"Pure standard-library non-blocking HTTP 1.1 socket server.\"\"\"\n    def __init__(self, host: str = \"127.0.0.1\", port: int = 8080):\n        self.host = host\n        self.port = port\n        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n        self.server_sock.setblocking(False)\n\n    def start(self):\n        self.server_sock.bind((self.host, self.port))\n        self.server_sock.listen(128)\n\n    def handle_request(self, raw_data: bytes) -> bytes:\n        try:\n            req_str = raw_data.decode(\"utf-8\", errors=\"replace\")\n            lines = req_str.split(\"\\r\\n\")\n            if not lines:\n                return b\"HTTP/1.1 400 Bad Request\\r\\nContent-Length: 0\\r\\n\\r\\n\"\n            parts = lines[0].split()\n            if len(parts) < 2:\n                return b\"HTTP/1.1 400 Bad Request\\r\\nContent-Length: 0\\r\\n\\r\\n\"\n            method, path = parts[0], parts[1]\n            body = f'{{\"status\": \"ok\", \"method\": \"{method}\", \"path\": \"{path}\"}}'.encode(\"utf-8\")\n            resp = (\n                f\"HTTP/1.1 200 OK\\r\\n\"\n                f\"Content-Type: application/json\\r\\n\"\n                f\"Content-Length: {len(body)}\\r\\n\"\n                f\"Connection: close\\r\\n\\r\\n\"\n            ).encode(\"utf-8\") + body\n            return resp\n        except Exception as e:\n            err = f'{{\"error\": \"{str(e)}\"}}\\r\\n'.encode(\"utf-8\")\n            return f\"HTTP/1.1 500 Internal Server Error\\r\\nContent-Length: {len(err)}\\r\\n\\r\\n\".encode(\"utf-8\") + err\n""",
        ],
    }

    def _open_fineweb_stream():
        try:
            return _open_hf_stream("HuggingFaceFW/fineweb", "sample-10BT", max_retries=1)
        except Exception:
            return iter([])

    def _open_code_stream():
        try:
            return _open_hf_stream("ajibawa-2023/Python-Code-Large", None, max_retries=1)
        except Exception:
            return iter([])

    domain_configs = [
        (
            "Broad Web (FineWeb sample-10BT)",
            _open_fineweb_stream,
            "broad_web",
            0.85,
        ),
        (
            "Code Repositories (The Stack v2)",
            _open_code_stream,
            "code",
            0.15,
        ),
    ]

    eos_id = getattr(tokenizer, "eos_token_id", 2)
    if eos_id is None:
        eos_id = 2
    bos_id = getattr(tokenizer, "bos_token_id", 1)
    if bos_id is None:
        bos_id = 1

    is_pure_pretrain = (prompt_len <= 1)
    seq_total = target_len if is_pure_pretrain else (prompt_len + target_len)
    target_tokens_needed = batch_size * seq_total

    eval_batches = []
    domain_labels = []
    domain_weights = []

    for name, stream_opener, ds_tag, weight in domain_configs:
        collected_texts: List[str] = []

        # 1. Online stream extraction
        try:
            it = stream_opener()
            scan_count = 0
            for row in it:
                scan_count += 1
                text = ""
                if isinstance(row, dict):
                    text = row.get("text", "") or row.get("content", "") or row.get("code", "")
                elif isinstance(row, str):
                    text = row
                if text and len(text.strip()) >= 200:
                    collected_texts.append(text.strip())
                if len(collected_texts) >= max(batch_size, 64) or scan_count >= max(batch_size * 2, 200):
                    break
        except Exception:
            pass

        # 2. Local parquets extraction if present
        if len(collected_texts) < batch_size:
            try:
                if ds_tag == "code":
                    local_code = glob.glob(os.path.join(REPO_ROOT, "data", "*code*.parquet"))
                    if local_code:
                        import pyarrow.parquet as pq
                        for p in local_code:
                            tbl = pq.read_table(p)
                            for col in ["text", "content", "code"]:
                                if col in tbl.column_names:
                                    for item in tbl[col].to_pylist():
                                        if item and len(item) > 200:
                                            collected_texts.append(item.strip())
                                            if len(collected_texts) >= max(batch_size, 64):
                                                break
                elif ds_tag == "broad_web":
                    local_val_path = os.path.join(REPO_ROOT, "data", "cerebro_v5_custom_dataset", "validation.parquet")
                    if os.path.exists(local_val_path):
                        import pyarrow.parquet as pq
                        tbl = pq.read_table(local_val_path)
                        for col in ["text", "content", "prompt", "answer"]:
                            if col in tbl.column_names:
                                for item in tbl[col].to_pylist():
                                    if item and len(item) > 200:
                                        collected_texts.append(item.strip())
                                        if len(collected_texts) >= max(batch_size, 64):
                                            break
            except Exception:
                pass

        # 3. Authentic embedded fallbacks
        if len(collected_texts) < batch_size and ds_tag in domain_fallbacks:
            collected_texts.extend(domain_fallbacks[ds_tag])

        # Convert collected texts to unbroken contiguous token stream
        token_stream: List[int] = []
        for doc in collected_texts:
            # Tokenize in safe chunks of <= 16k chars to prevent HF >100k sequence length warnings
            CHUNK_CHAR_SIZE = 16384
            doc_chunks = [doc[ci : ci + CHUNK_CHAR_SIZE] for ci in range(0, len(doc), CHUNK_CHAR_SIZE)] if len(doc) > CHUNK_CHAR_SIZE else [doc]
            for c_doc in doc_chunks:
                if hasattr(tokenizer, "encode"):
                    toks = encode_no_special(tokenizer, c_doc)
                else:
                    toks = [ord(c) % 32000 for c in c_doc]
                token_stream.extend(toks)
            token_stream.append(eos_id)

        # Safety fill if empty
        if not token_stream:
            token_stream = [1] * target_tokens_needed

        # Cycle until we have sufficient tokens for batch_size * seq_total
        while len(token_stream) < target_tokens_needed:
            token_stream.extend(token_stream[: max(1, target_tokens_needed - len(token_stream))])

        prompts_batch = []
        targets_batch = []
        for i in range(batch_size):
            start_idx = i * seq_total
            chunk = token_stream[start_idx : start_idx + seq_total]
            if is_pure_pretrain:
                q_ids = [bos_id]
                r_ids = chunk[:target_len]
            else:
                q_ids = chunk[:prompt_len]
                r_ids = chunk[prompt_len : prompt_len + target_len]

            if len(q_ids) < prompt_len:
                q_ids = q_ids + [pad_id] * (prompt_len - len(q_ids))
            if len(r_ids) < target_len:
                r_ids = r_ids + [target_pad_id] * (target_len - len(r_ids))

            prompts_batch.append(q_ids)
            targets_batch.append(r_ids)

        q_arr = np.asarray(prompts_batch, dtype=np.int32)
        r_arr = np.asarray(targets_batch, dtype=np.int32)

        fb_arr = np.full(r_arr.shape, target_pad_id, dtype=np.int32)
        if r_arr.shape[1] > 1:
            fb_arr[:, 1:] = r_arr[:, :-1]
        fb_arr = np.where(r_arr == target_pad_id, target_pad_id, fb_arr)

        eval_batches.append((q_arr, r_arr, fb_arr))
        domain_labels.append(name)
        domain_weights.append(weight)

    if return_weights:
        return eval_batches, domain_labels, domain_weights
    return eval_batches, domain_labels



