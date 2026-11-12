"""
Deterministic, resumable corpus iteration for full-corpus training.

The loaders in `src/data_loaders.py` build correctly shaped batches for short runs, but
they re-shuffle with module-level `random` and re-tokenise per record, so a run cannot say
where it is in the corpus or continue from there. This module answers both questions:
every source is a pure function of (seed, position in the stream), and a caller can skip
exactly N already-consumed records on resume.

Holdout is by parquet row group. Row groups are how parquet is physically stored, so
excluding them costs nothing, and a held-out slice drawn this way cannot appear in
training no matter how the within-group order is shuffled.

Every iterator yields raw `(prompt_text, answer_text)` pairs. Tokenisation is separate
because training slices a window out of the answer and the eval path does not.
"""

import hashlib
import json
import os
import random
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pyarrow as pa
if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
    pa.PyExtensionType = pa.ExtensionType
import pyarrow.parquet as pq

Pair = Tuple[str, str]

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

SOURCES = (
    "openhermes",
    "lmsys",
    "knowledge",
    "cognitive",
    "gsm8k",
    "cosmopedia",
    "fineweb_edu",
    "finemath",
    "stack_edu",
)

STREAMING_SOURCES = ("cosmopedia", "fineweb_edu", "finemath", "stack_edu")


# --------------------------------------------------------------------------------------------
# row selection
# --------------------------------------------------------------------------------------------

def group_plan(num_groups: int, seed: int, holdout_every: int, want_heldout: bool) -> List[int]:
    """Ordered row-group indices, split deterministically into train and held-out halves.

    `holdout_every=50` reserves groups 0, 50, 100, ... for evaluation. Passing 0 disables
    the split and every group is a train group.
    """
    order = list(range(num_groups))
    random.Random(seed).shuffle(order)
    if holdout_every <= 0:
        return [] if want_heldout else order
    if want_heldout:
        return [g for g in order if g % holdout_every == 0]
    return [g for g in order if g % holdout_every != 0]


def row_plan(num_rows: int, seed: int, holdout_every: int, want_heldout: bool) -> List[int]:
    """Same split for single-group files, keyed on row index instead of row group."""
    order = list(range(num_rows))
    random.Random(seed).shuffle(order)
    if holdout_every <= 0:
        return [] if want_heldout else order
    if want_heldout:
        return [r for r in order if r % holdout_every == 0]
    return [r for r in order if r % holdout_every != 0]


def _shuffled(rows: List[Pair], seed: int, salt: int) -> List[Pair]:
    order = list(range(len(rows)))
    random.Random(seed * 1000003 + salt).shuffle(order)
    return [rows[i] for i in order]


# --------------------------------------------------------------------------------------------
# per-source readers
# --------------------------------------------------------------------------------------------

def _conversation_pairs(conv: Sequence) -> List[Pair]:
    """Extracts every user->assistant turn from a chat-like conversation list."""
    if not conv or not isinstance(conv, list):
        return []
    out = []
    for i in range(len(conv) - 1):
        u, a = conv[i], conv[i + 1]
        if not isinstance(u, dict) or not isinstance(a, dict):
            continue
        u_role = u.get("from") or u.get("role")
        a_role = a.get("from") or a.get("role")
        if u_role in ("human", "user") and a_role in ("gpt", "assistant", "model"):
            u_text = (u.get("value") or u.get("content") or "").strip()
            a_text = (a.get("value") or a.get("content") or "").strip()
            if u_text and a_text:
                out.append((u_text, a_text))
    return out


def openhermes_pairs(path: str, seed: int = 0, holdout_every: int = 0,
                     want_heldout: bool = False, english_only: bool = True) -> Iterator[Pair]:
    parquet_file = pq.ParquetFile(path)
    for rg_idx in group_plan(parquet_file.num_row_groups, seed, holdout_every, want_heldout):
        table = parquet_file.read_row_group(rg_idx)
        columns = table.column_names
        rows: List[Pair] = []
        if "language" in columns and english_only:
            langs = table["language"].to_pylist()
            convs = table["conversations"].to_pylist()
            for lang, conv in zip(langs, convs):
                if lang and lang.lower() not in ("english", "en"):
                    continue
                rows += _conversation_pairs(conv)
        elif "conversations" in columns:
            for conv in table["conversations"].to_pylist():
                rows += _conversation_pairs(conv)
        elif "instruction" in columns and "output" in columns:
            out_col = table["output"].to_pylist()
            in_col = table["input"].to_pylist() if "input" in columns else [""] * len(out_col)
            for inst, inp, out in zip(table["instruction"].to_pylist(), in_col, out_col):
                prompt = f"{inst.strip()} {inp.strip()}".strip() if inp else inst.strip()
                if prompt and out and out.strip():
                    rows.append((prompt, out.strip()))
        yield from _shuffled(rows, seed, rg_idx)


def lmsys_pairs(data_dir: str, seed: int = 0, holdout_every: int = 0,
                want_heldout: bool = False) -> Iterator[Pair]:
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    order = list(range(len(files)))
    random.Random(seed).shuffle(order)
    for file_idx in order:
        path = os.path.join(data_dir, files[file_idx])
        parquet_file = pq.ParquetFile(path)
        for rg_idx in group_plan(parquet_file.num_row_groups, seed + file_idx, holdout_every, want_heldout):
            table = parquet_file.read_row_group(rg_idx)
            langs = table["language"].to_pylist()
            convs = table["conversation"].to_pylist()
            rows: List[Pair] = []
            for lang, conv in zip(langs, convs):
                if lang != "English":
                    continue
                rows += _conversation_pairs(conv)
            yield from _shuffled(rows, seed, 7919 * (file_idx + 1) + rg_idx)


def small_parquet_pairs(path: str, seed: int = 0, holdout_every: int = 0,
                        want_heldout: bool = False) -> Iterator[Pair]:
    """Reader for curated files (knowledge, gsm8k, etc.), split by row index."""
    table = pq.read_table(path)
    cols = table.column_names
    if "stimulus" in cols and "articulation" in cols:
        stimuli = table["stimulus"].to_pylist()
        answers = table["articulation"].to_pylist()
    elif "question" in cols and "answer" in cols:
        stimuli = table["question"].to_pylist()
        answers = table["answer"].to_pylist()
    elif "prompt" in cols and "response" in cols:
        stimuli = table["prompt"].to_pylist()
        answers = table["response"].to_pylist()
    else:
        raise ValueError(f"Unrecognized columns in {path}: {cols}")

    for idx in row_plan(len(stimuli), seed, holdout_every, want_heldout):
        q, a = stimuli[idx], answers[idx]
        if q and a and str(q).strip() and str(a).strip():
            yield (str(q).strip(), str(a).strip())


# --------------------------------------------------------------------------------------------
# streaming fluff-free educational corpora (zero disk download, zero setup wait)
# --------------------------------------------------------------------------------------------

def _get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face API token from environment if present."""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def cosmopedia_stream(seed: int = 0) -> Iterator[Pair]:
    """Pure online streaming iterator for Cosmopedia v2 (100% synthetic fluff-free textbooks/articles)."""
    import time
    from datasets import load_dataset
    hf_tok = _get_hf_token()
    kw = {"split": "train", "streaming": True}
    if hf_tok:
        kw["token"] = hf_tok
    while True:
        try:
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", **kw)
            for item in ds:
                prompt = (item.get("prompt") or "").strip()
                text = (item.get("text") or item.get("content") or "").strip()
                if not prompt and "\n\n" in text:
                    parts = text.split("\n\n", 1)
                    prompt, text = parts[0].strip(), parts[1].strip()
                elif not prompt:
                    prompt = "Explain the following educational topic in depth:"
                if prompt and text:
                    yield (prompt, text)
        except Exception as e:
            err_msg = str(e).lower()
            wait_s = 10 if ("429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg) else 2
            print(f"[corpus stream] Cosmopedia transient stream reconnect: {e} (backing off {wait_s}s)")
            time.sleep(wait_s)


def fineweb_edu_stream(seed: int = 0, min_score: float = 3.0) -> Iterator[Pair]:
    """Pure online streaming iterator for FineWeb-Edu (AI-scored educational web text >= 3.0)."""
    import time
    from datasets import load_dataset
    hf_tok = _get_hf_token()
    kw = {"split": "train", "streaming": True}
    if hf_tok:
        kw["token"] = hf_tok
    while True:
        try:
            try:
                ds = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", **kw)
            except Exception:
                ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", **kw)
            for item in ds:
                score = float(item.get("score", 3.5) or item.get("educational_score", 3.5) or 3.5)
                if score < min_score:
                    continue
                text = (item.get("text") or item.get("content") or "").strip()
                if len(text) < 60:
                    continue
                if "\n\n" in text:
                    parts = text.split("\n\n", 1)
                    prompt, response = parts[0].strip(), parts[1].strip()
                else:
                    lines = text.split(". ", 1)
                    prompt, response = (lines[0] + ".").strip(), lines[1].strip() if len(lines) > 1 else text
                if len(prompt) < 15:
                    prompt = "Explain the educational context of the following subject:\n" + prompt
                if prompt and response:
                    yield (prompt, response)
        except Exception as e:
            err_msg = str(e).lower()
            wait_s = 10 if ("429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg) else 2
            print(f"[corpus stream] FineWeb-Edu transient stream reconnect: {e} (backing off {wait_s}s)")
            time.sleep(wait_s)


def finemath_stream(seed: int = 0) -> Iterator[Pair]:
    """Pure online streaming iterator for FineMath (high-density mathematical equations and proofs)."""
    import time
    from datasets import load_dataset
    hf_tok = _get_hf_token()
    kw = {"split": "train", "streaming": True}
    if hf_tok:
        kw["token"] = hf_tok
    while True:
        try:
            ds = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", **kw)
            for item in ds:
                text = (item.get("text") or "").strip()
                if len(text) < 60:
                    continue
                if "\n\n" in text:
                    parts = text.split("\n\n", 1)
                    prompt = f"Solve and explain the mathematical problem:\n{parts[0].strip()}"
                    response = parts[1].strip()
                else:
                    prompt = "Explain the mathematical derivation and proof for the following:"
                    response = text
                if prompt and response:
                    yield (prompt, response)
        except Exception as e:
            err_msg = str(e).lower()
            wait_s = 10 if ("429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg) else 2
            print(f"[corpus stream] FineMath transient stream reconnect: {e} (backing off {wait_s}s)")
            time.sleep(wait_s)


def stack_edu_stream(seed: int = 0) -> Iterator[Pair]:
    """Pure online streaming iterator for Stack-Edu (curated production-grade code)."""
    import time
    from datasets import load_dataset
    hf_tok = _get_hf_token()
    kw = {"split": "train", "streaming": True}
    if hf_tok:
        kw["token"] = hf_tok
    while True:
        try:
            ds = load_dataset("devngho/the-stack-mini-edu", "scored_over_4", **kw)
            for item in ds:
                text = (item.get("content") or item.get("text") or "").strip()
                if len(text) < 60:
                    continue
                ext = item.get("ext") or "code"
                parts = text.split("\n\n", 1)
                header = parts[0].strip()[:200]
                prompt = f"Write production-grade, well-documented {ext} implementation for:\n{header}"
                response = text
                if prompt and response:
                    yield (prompt, response)
        except Exception as e:
            err_msg = str(e).lower()
            wait_s = 10 if ("429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg) else 2
            print(f"[corpus stream] Stack-Edu transient stream reconnect: {e} (backing off {wait_s}s)")
            time.sleep(wait_s)


def cosmopedia_pairs(path: str, seed: int = 0, holdout_every: int = 0,
                     want_heldout: bool = False, max_records: int = 50000) -> Iterator[Pair]:
    """Cosmopedia-v2 reader: uses local parquet if present, otherwise streams directly online."""
    if not os.path.exists(path):
        yield from cosmopedia_stream(seed)
        return

    table = pq.read_table(path)
    cols = table.column_names
    if "prompt" in cols and "response" in cols:
        prompts = table["prompt"].to_pylist()
        responses = table["response"].to_pylist()
    elif "text" in cols:
        texts = table["text"].to_pylist()
        prompts, responses = [], []
        for t in texts:
            if "\n\n" in str(t):
                p, r = str(t).split("\n\n", 1)
            else:
                p, r = "Explain the following concept:", str(t)
            prompts.append(p)
            responses.append(r)
    else:
        prompts = table[cols[0]].to_pylist()
        responses = table[cols[1]].to_pylist() if len(cols) > 1 else prompts

    n = min(len(prompts), max_records)
    for idx in row_plan(n, seed, holdout_every, want_heldout):
        q, a = prompts[idx], responses[idx]
        if q and a and str(q).strip() and str(a).strip():
            yield (str(q).strip(), str(a).strip())


def fineweb_edu_pairs(path: str, seed: int = 0, holdout_every: int = 0,
                      want_heldout: bool = False, max_records: int = 50000,
                      min_score: float = 3.0) -> Iterator[Pair]:
    """FineWeb-Edu reader: uses local parquet if present, otherwise streams directly online."""
    if not os.path.exists(path):
        yield from fineweb_edu_stream(seed, min_score=min_score)
        return

    table = pq.read_table(path)
    cols = table.column_names
    if "prompt" in cols and "response" in cols:
        prompts = table["prompt"].to_pylist()
        responses = table["response"].to_pylist()
    elif "text" in cols:
        texts = table["text"].to_pylist()
        prompts, responses = [], []
        for t in texts:
            if "\n\n" in str(t):
                p, r = str(t).split("\n\n", 1)
            else:
                p, r = "Explain the following educational concept:", str(t)
            prompts.append(p)
            responses.append(r)
    else:
        prompts = table[cols[0]].to_pylist()
        responses = table[cols[1]].to_pylist() if len(cols) > 1 else prompts

    n = min(len(prompts), max_records)
    for idx in row_plan(n, seed, holdout_every, want_heldout):
        q, a = prompts[idx], responses[idx]
        if q and a and str(q).strip() and str(a).strip():
            yield (str(q).strip(), str(a).strip())


# --------------------------------------------------------------------------------------------
# mixture
# --------------------------------------------------------------------------------------------

def default_paths() -> Dict[str, str]:
    return {
        "openhermes": os.path.join(DATA_DIR, "openhermes", "openhermes_2_5.parquet"),
        "lmsys": os.path.join(DATA_DIR, "lmsys_chat_1m", "data"),
        "knowledge": os.path.join(DATA_DIR, "corpus", "world_knowledge_canonical_train.parquet"),
        "cognitive": os.path.join(DATA_DIR, "cerebro_cognitive", "cerebro_cognitive_train.parquet"),
        "gsm8k": os.path.join(DATA_DIR, "gsm8k", "gsm8k_train.parquet"),
        "cosmopedia": os.path.join(DATA_DIR, "cosmopedia", "cosmopedia_v2_train.parquet"),
        "fineweb_edu": os.path.join(DATA_DIR, "fineweb_edu", "fineweb_edu_sample.parquet"),
        "finemath": os.path.join(DATA_DIR, "finemath", "finemath_sample.parquet"),
        "stack_edu": os.path.join(DATA_DIR, "stack_edu", "stack_edu_sample.parquet"),
    }


def parse_mix(spec: str) -> List[Tuple[str, float]]:
    """Parses 'openhermes=0.45,lmsys=0.35,knowledge=0.15,cognitive=0.05'."""
    parts = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"bad --mix entry {chunk!r}; expected name=weight")
        name, weight = chunk.split("=", 1)
        name = name.strip()
        if name not in SOURCES:
            raise ValueError(f"unknown source {name!r}; known: {', '.join(SOURCES)}")
        parts.append((name, float(weight)))
    total = sum(w for _, w in parts)
    if total <= 0:
        raise ValueError("--mix weights must sum to something positive")
    return [(n, w / total) for n, w in parts]


class Mixture:
    """Weighted interleave of sources with exact resume by per-source record count.

    Supports both local parquet files and high-density online streaming datasets
    (Cosmopedia v2, FineWeb-Edu, FineMath, Stack-Edu) without requiring upfront disk downloads.
    """

    def __init__(self, specs: List[Tuple[str, float]], paths: Dict[str, str], seed: int = 0,
                 holdout_every: int = 0, want_heldout: bool = False,
                 consumed: Dict[str, int] = None,
                 holdout_sources: Iterable[str] = ("openhermes", "lmsys", "gsm8k")):
        self.seed = seed
        self.paths = paths
        self.holdout_every = holdout_every
        self.want_heldout = want_heldout
        self.holdout_sources = holdout_sources
        self.names = [n for n, _ in specs]
        self.weights = [w for _, w in specs]
        self.consumed = {n: int((consumed or {}).get(n, 0)) for n in self.names}
        self._iterators: Dict[str, Iterator[Pair]] = {}
        self._factories = {}

        for name in self.names:
            factory = self._make_factory(name, paths.get(name, ""))
            self._factories[name] = factory
            self._iterators[name] = factory()

        for name in self.names:
            count = self.consumed[name]
            if name in STREAMING_SOURCES:
                for _ in range(min(count, 50000)):
                    try:
                        next(self._iterators[name])
                    except StopIteration:
                        break
            else:
                for _ in range(count):
                    try:
                        next(self._iterators[name])
                    except StopIteration:
                        self._iterators[name] = self._factories[name]()
                        try:
                            next(self._iterators[name])
                        except StopIteration:
                            raise RuntimeError(f"source {name!r} is completely empty")

        self._rng = random.Random(seed * 31 + 17)
        for _ in range(sum(self.consumed.values())):
            self._rng.choices(self.names, weights=self.weights, k=1)

    def _make_factory(self, name: str, path: str):
        if name in STREAMING_SOURCES:
            if name == "cosmopedia":
                return lambda s=self.seed: cosmopedia_stream(s)
            elif name == "fineweb_edu":
                return lambda s=self.seed: fineweb_edu_stream(s)
            elif name == "finemath":
                return lambda s=self.seed: finemath_stream(s)
            elif name == "stack_edu":
                return lambda s=self.seed: stack_edu_stream(s)

        if not os.path.exists(path):
            raise FileNotFoundError(f"source {name!r} not found at {path}")
        every = self.holdout_every if name in self.holdout_sources else 0
        if name == "openhermes":
            return lambda p=path, e=every: openhermes_pairs(p, self.seed, e, self.want_heldout)
        elif name == "lmsys":
            return lambda p=path, e=every: lmsys_pairs(p, self.seed, e, self.want_heldout)
        else:
            return lambda p=path, e=every: small_parquet_pairs(p, self.seed, e, self.want_heldout)

    def update_weights(self, specs: List[Tuple[str, float]], paths: Optional[Dict[str, str]] = None):
        """Dynamically adjusts mixture weights and registers any new curriculum sources on the fly."""
        paths = paths or self.paths
        new_names = [n for n, _ in specs]
        new_weights = [w for _, w in specs]
        for name in new_names:
            if name not in self._iterators:
                self.consumed[name] = 0
                factory = self._make_factory(name, paths.get(name, ""))
                self._factories[name] = factory
                self._iterators[name] = factory()
        self.names = new_names
        self.weights = new_weights

    def next(self) -> Tuple[str, str, str]:
        name = self._rng.choices(self.names, weights=self.weights, k=1)[0]
        try:
            prompt, answer = next(self._iterators[name])
        except StopIteration:
            self._iterators[name] = self._factories[name]()
            prompt, answer = next(self._iterators[name])
        self.consumed[name] += 1
        return name, prompt, answer

    def state(self) -> Dict[str, int]:
        return dict(self.consumed)


def total_weights(specs: List[Tuple[str, float]]) -> Dict[str, float]:
    return {n: w for n, w in specs}


# --------------------------------------------------------------------------------------------
# tokenisation & provenance & leakage checks
# --------------------------------------------------------------------------------------------

def build_tokenizer(name: str = "llama"):
    """
    Builds and configures a tokenizer for CerebroSpheroid models:
    - 'llama': Modern 32,000 Byte-Fallback BPE (prefers local bundled data/tokenizer, then HuggingFace).
      Features byte-level UTF-8 fallback (zero <unk>), single-digit number tokenization,
      pad_token_id=0, bos_token_id=1, eos_token_id=2. Strictly 32,000 vocabulary.
    """
    if name == "gpt2":
        raise ValueError(
            "GPT-2 tokenizer (50,257 vocab) has been deprecated and completely removed from CerebroSpheroid V5. "
            "V5 models have an architectural vocabulary size of exactly 32,000 tokens. "
            "Please use the bundled LLaMA-32k tokenizer (data/tokenizer)."
        )

    os.environ["HF_HOME"] = os.path.expanduser("~/.hf_cache")
    from transformers import AutoTokenizer

    # 1. First priority: Check local bundled LLaMA-32k tokenizer in data/tokenizer
    local_tok = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tokenizer")
    if os.path.exists(os.path.join(local_tok, "tokenizer.json")):
        try:
            tok = AutoTokenizer.from_pretrained(local_tok, local_files_only=True)
            if tok.pad_token_id is None:
                tok.pad_token_id = 0
            tok.model_max_length = 100_000
            return tok
        except Exception:
            pass

    # 2. Remote LLaMA 32k fallback
    for cand in ["huggyllama/llama-7b", "openlm-research/open_llama_7b", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]:
        try:
            tok = AutoTokenizer.from_pretrained(cand)
            if tok.pad_token_id is None:
                tok.pad_token_id = 0
            tok.model_max_length = 100_000
            return tok
        except Exception:
            continue

    if name and name not in ("llama", "default"):
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token_id is None:
            tok.pad_token_id = 0
        return tok

    raise RuntimeError("Failed to load LLaMA-32k tokenizer. Ensure data/tokenizer is present.")


# Biology-derived enclosed delimiters: <stim> ... </stim> and <act> ... </act>
BIO_STIM_PREFIX = "<stim> "
BIO_STIM_SUFFIX = " </stim>"
BIO_ACT_PREFIX = "<act> "
BIO_ACT_SUFFIX = " </act>"


def format_stimulus_prompt(prompt: str) -> str:
    """Encloses prompt in <stim> ... </stim> delimiters."""
    p = prompt.strip()
    if p.startswith("stim "):
        p = p[5:].strip()
    if not p.startswith("<stim>"):
        p = f"{BIO_STIM_PREFIX}{p}"
    if not p.endswith("</stim>"):
        p = f"{p}{BIO_STIM_SUFFIX}"
    return p


def format_motor_response(answer: str) -> str:
    """Encloses answer in <act> ... </act> delimiters."""
    a = answer.strip()
    if a.startswith("act "):
        a = a[4:].strip()
    if not a.startswith("<act>"):
        a = f"{BIO_ACT_PREFIX}{a}"
    if not a.endswith("</act>"):
        a = f"{a}{BIO_ACT_SUFFIX}"
    return a


def clean_motor_response(text: str) -> str:
    """Strips <act> and </act> delimiters if present, with fallback for legacy prefixes."""
    t = text.strip()
    m = re.search(r"<act>\s*(.*?)(?:\s*</act>|$)", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    if t.startswith("<act>"):
        t = t[5:].strip()
    if t.endswith("</act>"):
        t = t[:-6].strip()
    if t.startswith("act "):
        t = t[4:].strip()
    if t.startswith("act:"):
        t = t[4:].strip()
    return t


def encode_no_special(tokenizer, text: str, max_length: Optional[int] = None) -> List[int]:
    """Encode text without prepending/appending BOS/EOS tokens."""
    if hasattr(tokenizer, "encode"):
        kwargs: Dict[str, Any] = {"add_special_tokens": False}
        if max_length is not None:
            kwargs["max_length"] = max_length
            kwargs["truncation"] = True
        try:
            return list(tokenizer.encode(text, **kwargs))
        except TypeError:
            try:
                ids = list(tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                ids = list(tokenizer.encode(text))
            return ids[:max_length] if max_length is not None else ids
    if callable(tokenizer):
        try:
            kwargs = {"add_special_tokens": False}
            if max_length is not None:
                kwargs["max_length"] = max_length
                kwargs["truncation"] = True
            out = tokenizer(text, **kwargs)
        except TypeError:
            out = tokenizer(text)
        ids = list(out["input_ids"] if isinstance(out, dict) else out)
        return ids[:max_length] if max_length is not None else ids
    raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)}")


def get_bio_token_ids(tokenizer) -> Dict[str, Optional[int]]:
    """Resolve single-token IDs for <stim>, </stim>, <act>, </act> if defined."""
    tags = {
        "stim": "<stim>",
        "stim_end": "</stim>",
        "act": "<act>",
        "act_end": "</act>",
    }
    result: Dict[str, Optional[int]] = {k: None for k in tags}
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return result
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for key, marker in tags.items():
        try:
            ids = encode_no_special(tokenizer, marker)
        except Exception:
            continue
        if len(ids) == 1 and int(ids[0]) != unk_id:
            result[key] = int(ids[0])
    return result


def tokenize_pair(tokenizer, prompt: str, answer: str, max_prompt_len: int,
                  max_answer_tokens: int) -> Tuple[List[int], List[int]]:
    """Tokenises one record using enclosed biological delimiters ('<stim> ... </stim>' / '<act> ... </act>'), appending EOS."""
    formatted_prompt = format_stimulus_prompt(prompt)
    prompt_ids = encode_no_special(tokenizer, formatted_prompt, max_length=max_prompt_len)

    formatted_answer = format_motor_response(answer)
    answer_ids = encode_no_special(tokenizer, formatted_answer, max_length=max_answer_tokens)
    if len(answer_ids) < max_answer_tokens:
        answer_ids = answer_ids + [tokenizer.eos_token_id]
    return prompt_ids, answer_ids


def get_file_hash(path: str) -> str:
    """Computes deterministic short md5 hash of file or directory contents."""
    if not os.path.exists(path):
        return "missing"
    if os.path.isdir(path):
        files = sorted(os.listdir(path))
        return hashlib.md5("".join(files).encode()).hexdigest()[:12]
    h = hashlib.md5()
    with open(path, "rb") as f:
        chunk = f.read(65536)
        while chunk:
            h.update(chunk)
            chunk = f.read(65536)
    return h.hexdigest()[:12]


def get_dataset_provenance(paths: Dict[str, str], mix_spec: str) -> Dict[str, Any]:
    """Records dataset file hashes and the mix string for auditability in the state sidecar."""
    prov = {"mix": mix_spec, "hashes": {}}
    for name, p in paths.items():
        prov["hashes"][name] = get_file_hash(p)
    return prov


def assert_no_data_leakage(
    train_prompts: Sequence[str],
    val_prompts: Sequence[str],
    withheld_entities: Sequence[str] = (),
) -> Tuple[int, int]:
    """
    Acceptance criterion 3:
    Asserts 0 duplicated questions across train and val, and 0 withheld entities in train.
    """
    train_set = set(p.strip().lower() for p in train_prompts)
    val_set = set(p.strip().lower() for p in val_prompts)
    duplicates = train_set.intersection(val_set)
    assert len(duplicates) == 0, f"Leakage detected: {len(duplicates)} duplicated prompts across train and val!"

    entity_leaks = 0
    if withheld_entities:
        for p in train_prompts:
            p_lower = p.lower()
            for entity in withheld_entities:
                if entity.lower() in p_lower:
                    entity_leaks += 1
                    break
        assert entity_leaks == 0, f"Leakage detected: {entity_leaks} train prompts contain withheld entities!"

    return len(duplicates), entity_leaks

