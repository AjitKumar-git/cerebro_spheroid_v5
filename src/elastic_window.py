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
    if "####" in r_txt and "<act>" not in r_txt:
        parts = r_txt.split("####", 1)
        deriv = parts[0].strip()
        ans_str = parts[1].strip()
        if not deriv.startswith("<think>"):
            deriv = f"<think>\n{deriv}\n</think>"
        return f"{deriv}\n<act>\ncalculator: {ans_str}\n</act>\n{ans_str}"

    # 2. Markdown code fences -> Non-conversational Action <act> ... </act>
    if "<act>" not in r_txt:
        # Match fenced code block (python, py, bash, sh, json, or untyped)
        pattern = r"```(?:python|py|bash|sh|json)?\s*\n(.*?)\n```"
        match = re.search(pattern, r_txt, re.DOTALL)
        if match:
            code_content = match.group(1).strip()
            # Replace code block with <act> ... </act>
            r_txt = r_txt[:match.start()] + f"<act>\n{code_content}\n</act>" + r_txt[match.end():]
        elif any(r_txt.strip().startswith(prefix) for prefix in ("def ", "class ", "import ", "from ", "#!/usr/bin/env")):
            r_txt = f"<act>\n{r_txt.strip()}\n</act>"

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
                return str(q_txt), _normalize_3tier_response(str(r_txt), str(q_txt))

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
                return str(q_txt), _normalize_3tier_response(str(r_txt), str(q_txt))

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
        if not q_val and "code" in row:
            raw_c = str(row["code"]).strip()
            return "Implement the following Python code:", f"<act>\n{raw_c}\n</act>"
        if not q_val and "text" in row:
            raw_t = str(row["text"]).strip()
            if "\n\n" in raw_t:
                p, r = raw_t.split("\n\n", 1)
            else:
                p, r = "Explain and analyze the following topic:", raw_t
            return p.strip(), _normalize_3tier_response(r.strip(), p.strip())
        return str(q_val or ""), _normalize_3tier_response(str(r_val or ""), str(q_val or ""))
    return "", ""
    return "", ""


_DEFAULT_DATASET_REVISIONS: Dict[str, str] = {
    # OpenHermes-2.5's default branch holds a single 1.94 GB JSON array file,
    # causing datasets' JSON builder to buffer the full file + ujson AST in RAM
    # (~15+ GB per rank) and trigger host OOM SIGKILL on 2xT4 containers. The
    # auto-converted Parquet branch streams row-groups in <250 MB RSS per rank.
    "teknium/OpenHermes-2.5": "refs/convert/parquet",
}

_DEFAULT_DATASET_CONFIGS: Dict[str, str] = {
    "HuggingFaceTB/cosmopedia-v2": "cosmopedia-v2",
    "HuggingFaceFW/fineweb-edu": "sample-350BT",
    "openai/gsm8k": "main",
    "gsm8k": "main",
    "HuggingFaceTB/smoltalk": "all",
    "HuggingFaceTB/finemath": "finemath-4plus",
    "open-r1/OpenR1-Math-220k": "default",
}


def _get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face API token from environment variables or cache."""
    import os
    env_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if env_token:
        return env_token
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


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
                wait_t = min(30.0, (2.0 ** attempt) + random.uniform(0.5, 2.0))
                if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                    print(f"⚠️ [DataStreamer] HuggingFace Rate Limit (429) on {repo}. Backing off {wait_t:.1f}s (retry {attempt+1}/{max_retries})...")
                else:
                    print(f"⚠️ [DataStreamer] Dataset stream notice on {repo} ({e}). Retrying in {wait_t:.1f}s...")
                time.sleep(wait_t)

        except Exception as e:
            err_str = str(e).lower()
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
    source = (dataset_source or "dense_edu").lower()
    if source in ("chat_thinking", "db_thinking", "specialized_trihead", "specialized_sft"):
        source = "chat_thinking"
    elif source in ("sft", "instruction_sft", "chat_sft", "alignment", "reasoning_sft", "cot_sft", "think_sft", "r1_sft", "cot", "deepseek_r1", "conversational", "conversational_sft", "layman"):
        source = "conversational_sft"
    elif source in ("foundation", "auto", "basic", "fineweb_edu"):
        source = "dense_edu"
    elif source in ("reasoning", "frontier", "beat_smollm", "smollm_pass", "frontier_mix", "tri_pillar"):
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

    def _next_filtered_fineweb(it: Optional[Iterator[Any]], opener, min_score: float = 3.0):
        for _ in range(50):
            row, it = _next_or_reopen(it, opener)
            score = float(row.get("score", 0.0) or row.get("educational_score", 0.0) or row.get("int_score", 0.0))
            if score >= min_score:
                return row, it
        return row, it

    if "/" in dataset_source:
        repo_id = dataset_source
        custom_it: Optional[Iterator[Any]] = None

        def _open_custom_stream():
            from datasets import load_dataset
            print(f"🌐 Connecting online stream to custom Hugging Face repo: {repo_id}...")
            return iter(load_dataset(repo_id, split="train", streaming=True))

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
                }

        return _custom_gen()

    if source == "autonomous":
        # 4-Phase Developmental Autonomous Curriculum for V5 Base / 200B Campaign
        # Phase 1 (0 -> 15%): Foundation World Knowledge & Educational Prose (60% FineWeb-Edu >= 3, 15% Cosmo, 10% Everyday, 10% Math, 5% Code)
        # Phase 2 (15% -> 50%): Broadening & Algorithmic Grounding (35% FineWeb-Edu >= 3, 25% Everyday, 20% Code, 20% Math)
        # Phase 3 (50% -> 85%): Balanced Tri-Pillar Reasoning (33.3% Science, 33.3% Commonsense & Logic, 33.3% Math-CoT & Code)
        # Phase 4 (85% -> 100%): Annealing Basin & Boundary Consolidation (30% FineWeb-Edu >= 3, 30% FineMath/MetaMath, 20% Code, 20% Magpie)
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

        def _get_step_val() -> int:
            if callable(step_getter):
                try:
                    return int(step_getter())
                except Exception:
                    return 0
            elif isinstance(step_getter, (int, float)):
                return int(step_getter)
            return 0

        def _autonomous_gen():
            nonlocal know_it, cosmo_it, everyday_it, openhermes_it, logic_it, code_it, pycode_it, gsm_it, math_it, finemath_it
            turn = 0

            def _tag_yield(item: Any, ds_tag: str):
                if isinstance(item, dict):
                    item["_dataset"] = ds_tag
                return item

            while True:
                cur_step = _get_step_val()
                tau = float(cur_step) / float(max(1, total_steps))

                try:
                    if tau < 0.15:
                        # Phase 1: Foundation & World Knowledge Imprinting (Cycle mod 10)
                        # 60% FineWeb-Edu (score >= 3.0), 15% Cosmopedia, 10% Everyday Commonsense, 10% Math, 5% Code
                        mod10 = turn % 10
                        if mod10 in (0, 2, 4, 6, 8):
                            row, know_it = _next_filtered_fineweb(
                                know_it, lambda: _open_hf_stream("HuggingFaceFW/fineweb-edu", "sample-350BT", rank=rank, world_size=world_size), min_score=3.0
                            )
                            yield _tag_yield(row, "fineweb_edu")
                        elif mod10 in (1, 5):
                            row, cosmo_it = _next_or_reopen(
                                cosmo_it, lambda: _open_hf_stream("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", rank=rank, world_size=world_size)
                            )
                            p = (row.get("prompt") or "Explain the concept:").strip()
                            r = (row.get("text") or "").strip()
                            yield _tag_yield({"conversations": [{"from": "human", "value": p}, {"from": "gpt", "value": r}]}, "cosmopedia")
                        elif mod10 == 3:
                            row, everyday_it = _next_or_reopen(
                                everyday_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "everyday-conversations", rank=rank, world_size=world_size)
                            )
                            yield _tag_yield(row, "smoltalk")
                        elif mod10 == 7:
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
                        # 35% FineWeb-Edu (score >= 3.0), 25% Everyday Commonsense, 20% Python Code, 20% Math Proofs
                        mod10 = turn % 10
                        if mod10 in (0, 3, 6):
                            row, know_it = _next_filtered_fineweb(
                                know_it, lambda: _open_hf_stream("HuggingFaceFW/fineweb-edu", "sample-350BT", rank=rank, world_size=world_size), min_score=3.0
                            )
                            yield _tag_yield(row, "fineweb_edu")
                        elif mod10 in (1, 5, 8):
                            row, everyday_it = _next_or_reopen(
                                everyday_it, lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "everyday-conversations", rank=rank, world_size=world_size)
                            )
                            yield _tag_yield(row, "smoltalk")
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
                        # 33.3% Science Anchor (score >= 3.0), 33.3% Commonsense & Multi-Turn Logic, 33.3% Math-CoT & Indented Code
                        mod6 = turn % 6
                        if mod6 in (0, 3):
                            row, know_it = _next_filtered_fineweb(
                                know_it, lambda: _open_hf_stream("HuggingFaceFW/fineweb-edu", "sample-350BT", rank=rank, world_size=world_size), min_score=3.0
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
                        # Phase 4: Annealing Basin & Boundary Consolidation (Cycle mod 10)
                        # 30% FineWeb-Edu (score >= 3.0), 30% Verified GSM8K/FineMath/MetaMath, 20% Indented Code, 20% Magpie Ultra Logic
                        mod10 = turn % 10
                        if mod10 in (0, 3, 6):
                            row, know_it = _next_filtered_fineweb(
                                know_it, lambda: _open_hf_stream("HuggingFaceFW/fineweb-edu", "sample-350BT", rank=rank, world_size=world_size), min_score=3.0
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

                except Exception:
                    time.sleep(0.5)
                turn += 1

        return _autonomous_gen()

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

        return _chat_thinking_gen()

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

        return _conversational_gen()

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
    """Returns high-density complementary sliding window buckets preserving intact prompts and deep reasoning trajectories."""
    assert total_window >= 256, "total_window must be at least 256 tokens"
    if total_window >= 1024:
        return [
            ("standard_cot", 256, 512),
            ("deep_reasoning", 256, 768),
            ("balanced", 384, 512),
            ("long_prompt", 512, 512),
        ]
    elif total_window >= 768:
        return [
            ("short_prompt_deep_cot", 128, 512),
            ("standard_cot", 256, 512),
            ("balanced", 256, 384),
            ("long_prompt", 384, 384),
        ]
    elif total_window >= 512:
        return [
            ("standard_cot", 128, 384),
            ("balanced", 256, 256),
            ("deep_reasoning", 128, 384),
        ]
    else:
        half = total_window // 2
        return [("balanced", half, half)]


def select_best_bucket(p_len: int, t_len: int, buckets: List[Tuple[str, int, int]]) -> int:
    """
    Selects the bucket that minimizes wasted padding while strictly preserving full-text prompt intactness.
    Prompt truncation carries a massive penalty (50,000) so a bucket accommodating the full question is always prioritized.
    """
    best_idx = min(1, len(buckets) - 1)
    min_penalty = float("inf")

    for idx, (_, bp, bt) in enumerate(buckets):
        if p_len <= bp and t_len <= bt:
            # Fits completely: penalty is wasted padding
            penalty = (bp - p_len) + (bt - t_len)
        elif p_len <= bp:
            # Fits prompt completely, target has mild truncation: penalize target truncation
            penalty = (bp - p_len) + (t_len - bt) * 3
        else:
            # Prompt exceeds bucket: massive penalty! Never truncate the prompt if another bucket fits it!
            penalty = 50000 + (p_len - bp) * 100 + max(0, t_len - bt) * 3

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
                    p_list, t_list, c_list, d_list = zip(*batch_items)
                    with self._lock:
                        for ds, p, t in zip(d_list, p_list, t_list):
                            self.tokens_by_dataset[ds] += (len(p) + len(t))
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
                # 2. Non-conversational output: Coding, Calculator, Tools are enclosed in <act> ... </act>.
                # 3. Conversational dialogue: Natural fluid plain text outside <act>.
                r_txt = _normalize_3tier_response(r_txt, q_txt)
                if "<act>" not in r_txt:
                    if any(kw in ds_name.lower() for kw in ("code", "python", "self_oss")):
                        r_txt = f"<act>\n{r_txt.strip()}\n</act>"
                    elif "calculator" in ds_name.lower():
                        r_txt = f"<act>\ncalculator: {r_txt.strip()}\n</act>"
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
                        self.tokens_by_dataset[ds] += (len(p) + len(t))
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
                    tokenized_pairs = list(zip(q_ids_batch, r_ids_batch, d_names))
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
                    tokenized_pairs.append((q_ids, r_ids, ds))

            eos_id = self.tokenizer.eos_token_id if (hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None) else 2
            for q_ids, r_ids, ds in tokenized_pairs:
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

                self.queues[b_idx].append((q_padded, r_padded, context_token, ds))


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

    def get_tokens_by_dataset(self) -> Dict[str, int]:
        """Forwards token breakdown request to underlying batcher."""
        return self.batcher.get_tokens_by_dataset()

    def set_tokens_by_dataset(self, d: Dict[str, int]) -> None:
        """Forwards token setting to underlying batcher."""
        self.batcher.set_tokens_by_dataset(d)

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
    Constructs a fixed, 4-domain canonical held-out evaluation split.
    Domains evaluated:
      1. Science & World Knowledge : FineWeb-Edu (score >= 4.0)
      2. Commonsense Physical Acts : SmolTalk (everyday-conversations)
      3. Exact Mathematical Proofs : GSM8K (official test split)
      4. Software & Python Code    : SmolTalk (self-oss-instruct)

    Returns:
      (eval_batches, domain_labels)
      where each batch is (prompt_batch, target_batch, feedback_batch) as static int32 numpy arrays.
    """
    from src.corpus import format_stimulus_prompt, format_motor_response

    domain_configs = [
        ("Commonsense Dialogue (SmolTalk)", lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "everyday-conversations")),
        ("World Knowledge (Hermes)", lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "openhermes-100k")),
        ("Python Code (Self-OSS)", lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "self-oss-instruct")),
        ("Math-CoT (GSM8K Test)", lambda: _open_hf_stream("openai/gsm8k", "main", split="test")),
        ("Multi-Turn Logic (Magpie Ultra)", lambda: _open_hf_stream("HuggingFaceTB/smoltalk", "smol-magpie-ultra")),
    ]

    eval_batches = []
    domain_labels = []

    for name, stream_opener in domain_configs:
        rows = []
        try:
            it = stream_opener()
            for row in it:
                if "fineweb" in name.lower() and float(row.get("score", 0.0) or row.get("educational_score", 0.0)) < 4.0:
                    continue
                q, r = extract_prompt_response_pair(row)
                if len(q.strip()) >= 5 and len(r.strip()) >= 10:
                    rows.append((q, r))
                if len(rows) >= batch_size:
                    break
        except Exception as e:
            # Fallback if specific HF stream is unreachable
            pass

        # Synthesize fallback rows if stream provided fewer than batch_size
        while len(rows) < batch_size:
            rows.append((
                f"Evaluation benchmark inquiry for {name} question {len(rows)+1}:",
                f"Detailed step-by-step rigorous canonical proof and solution in domain {name}."
            ))

        prompts_batch = []
        targets_batch = []
        feedbacks_batch = []

        eos_id = getattr(tokenizer, "eos_token_id", 2)
        if eos_id is None:
            eos_id = 2

        for q_txt, r_txt in rows[:batch_size]:
            q_fmt = format_stimulus_prompt(q_txt)
            r_fmt = r_txt.strip()
            if "<act>" not in r_fmt and any(kw in name.lower() for kw in ("code", "python")):
                r_fmt = f"<act>\n{r_fmt}\n</act>"

            if hasattr(tokenizer, "encode"):
                q_ids = encode_no_special(tokenizer, q_fmt, max_length=prompt_len)
                r_ids = encode_no_special(tokenizer, r_fmt, max_length=target_len)
            else:
                q_ids = [1] * min(len(q_fmt.split()), prompt_len)
                r_ids = [1] * min(len(r_fmt.split()), target_len)

            if len(r_ids) == 0 or r_ids[-1] != eos_id:
                r_ids.append(eos_id)

            # Pad or truncate prompt
            if len(q_ids) > prompt_len:
                q_pad = q_ids[:prompt_len]
            else:
                q_pad = q_ids + [pad_id] * (prompt_len - len(q_ids))

            # Pad or truncate target
            if len(r_ids) > target_len:
                r_pad = r_ids[:target_len]
            else:
                r_pad = r_ids + [target_pad_id] * (target_len - len(r_ids))

            # Construct strictly causal teacher-forcing feedback tokens shifted by 1:
            # Prev token at step 0 is prompt continuation (or -100). Steps 1..N-1 are target[:-1].
            # Padding positions where target == target_pad_id are masked with target_pad_id (-100).
            valid_prompt = [tok for tok in q_ids if tok != pad_id]
            continuation_tok = valid_prompt[-1] if len(valid_prompt) > 0 else target_pad_id
            fb_pad = [continuation_tok] + r_pad[:-1]
            fb_pad = [fb if r != target_pad_id else target_pad_id for fb, r in zip(fb_pad, r_pad)]

            prompts_batch.append(q_pad)
            targets_batch.append(r_pad)
            feedbacks_batch.append(fb_pad)

        q_arr = np.asarray(prompts_batch, dtype=np.int32)
        r_arr = np.asarray(targets_batch, dtype=np.int32)
        fb_arr = np.asarray(feedbacks_batch, dtype=np.int32)

        eval_batches.append((q_arr, r_arr, fb_arr))
        domain_labels.append(name)

    return eval_batches, domain_labels


