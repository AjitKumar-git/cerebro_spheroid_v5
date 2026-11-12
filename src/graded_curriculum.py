"""
CerebroSpheroid V5: Graded Curriculum Engine & Space-Saving Dataset Streamer
=============================================================================

Implements the 4-Phase Developmental Curriculum for 1,000,000 Training Steps:

Phase 1 (0 – 50,000 steps | 5%):
    • Objective: Foundation Syntax, Grammar, Morphological Semantics & Low-Entropy Structure.
    • Data: World knowledge facts, Wikipedia subsets, clean syntactic prose.
    • LR: Linear Warmup (0 -> 3e-4) over 2,000 steps, plateau at 3e-4.
    • Dynamics: Shells 0-15 establish base orthographic manifold anchors.

Phase 2 (50,000 – 500,000 steps | 45%):
    • Objective: Core Knowledge, Multi-Step Chain-of-Thought (CoT) Reasoning & Calculation.
    • Data: OpenHermes-2.5 (1M rows), GSM8k mathematical deductions, CerebroCognitive logic.
    • LR: Gradual cosine plateau (3e-4 -> 1.5e-4).
    • Dynamics: Subcortical core & Arcuate Fasciculus Top-16 sparse resonance heavily engaged.

Phase 3 (500,000 – 900,000 steps | 40%):
    • Objective: Modern Workflows, Agentic Execution, Tool Invocations & Multi-Turn Dialogues.
    • Data: UltraChat multi-turn dialogues, Glaive/Hermes tool calling traces, synthetic
            sandboxed agentic execution trajectories (BASH, PYTHON, CALCULATOR, FILE).
    • LR: Intermediate cosine decay (1.5e-4 -> 5e-5).
    • Dynamics: Cortical mantle (Shell 31) and ToolActionHead discrete dispatch calibration.

Phase 4 (900,000 – 1,000,000 steps | 10%):
    • Objective: Model Consolidation, Fixed-Point Energy Minimization & Annealing.
    • Data: Replay mixture (5% Phase 1, 15% Phase 2, 20% Phase 3, 60% Gold-Standard Reasoning).
    • LR: Strict cosine cooldown to 0.0.
    • Dynamics: High-energy regularization (+0.05) to crystallize fixed-point attractors.

Storage Optimization ("Save Training Space"):
    • Zero upfront flat token disk dumps (no 100+ GB binary files).
    • Lazy streaming row batches directly from compressed Parquet tables via PyArrow iterators.
    • Tokenization performed on-the-fly in dynamic micro-buffers.
    • Seamless offline-fallback architecture across local paths.
"""

from __future__ import annotations

import glob
import math
import os
import queue
import random
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from src.corpus import format_motor_response, format_stimulus_prompt
from src.elastic_window import (
    build_online_dataset_iterator,
    extract_prompt_response_pair,
    get_default_elastic_buckets,
    sample_window,
    select_best_bucket,
)


class CurriculumPhase(IntEnum):
    PHASE_1_BASIC = 0          # 0 to 50k steps
    PHASE_2_REASONING = 1      # 50k to 500k steps
    PHASE_3_WORKFLOWS = 2      # 500k to 900k steps
    PHASE_4_CONSOLIDATION = 3  # 900k to 1000k steps


PHASE_NAMES = {
    CurriculumPhase.PHASE_1_BASIC: "Phase 1: Basic Grammar & Syntax Foundations",
    CurriculumPhase.PHASE_2_REASONING: "Phase 2: Knowledge, Reasoning & Calculation",
    CurriculumPhase.PHASE_3_WORKFLOWS: "Phase 3: Modern Workflows & Agentic Tool-Use",
    CurriculumPhase.PHASE_4_CONSOLIDATION: "Phase 4: Consolidation & Annealing",
}


@dataclass
class CurriculumScheduleConfig:
    """Hyperparameters and step boundaries for the 1,000,000-step curriculum."""
    total_steps: int = 1_000_000
    phase_1_end: int = 50_000
    phase_2_end: int = 500_000
    phase_3_end: int = 900_000
    warmup_steps: int = 2_000
    max_lr: float = 3e-4
    phase_2_min_lr: float = 1.5e-4
    phase_3_min_lr: float = 5e-5
    min_lr: float = 0.0
    energy_reg_phase_1_3: float = 0.01
    energy_reg_phase_4: float = 0.05
    prompt_len: int = 32
    seq_len: int = 256
    batch_size: int = 8
    visual_dim: int = 512
    num_visual_patches: int = 16
    multimodal_sample_ratio: float = 0.0


def get_curriculum_phase(step: int, config: CurriculumScheduleConfig) -> CurriculumPhase:
    """Returns current active curriculum phase based on global training step."""
    if step < config.phase_1_end:
        return CurriculumPhase.PHASE_1_BASIC
    elif step < config.phase_2_end:
        return CurriculumPhase.PHASE_2_REASONING
    elif step < config.phase_3_end:
        return CurriculumPhase.PHASE_3_WORKFLOWS
    else:
        return CurriculumPhase.PHASE_4_CONSOLIDATION


def compute_curriculum_lr(step: int, config: CurriculumScheduleConfig) -> float:
    """
    Computes piecewise continuous learning rate across the 4 developmental phases:
    - Phase 1 Warmup: 0 -> max_lr
    - Phase 1 Plateau: max_lr
    - Phase 2 Cosine: max_lr -> phase_2_min_lr
    - Phase 3 Cosine: phase_2_min_lr -> phase_3_min_lr
    - Phase 4 Cosine: phase_3_min_lr -> min_lr (0.0)
    """
    if step < config.warmup_steps:
        return config.max_lr * float(step + 1) / float(max(1, config.warmup_steps))

    if step < config.phase_1_end:
        return config.max_lr

    elif step < config.phase_2_end:
        progress = float(step - config.phase_1_end) / float(config.phase_2_end - config.phase_1_end)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.phase_2_min_lr + (config.max_lr - config.phase_2_min_lr) * cosine_decay

    elif step < config.phase_3_end:
        progress = float(step - config.phase_2_end) / float(config.phase_3_end - config.phase_2_end)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.phase_3_min_lr + (config.phase_2_min_lr - config.phase_3_min_lr) * cosine_decay

    else:
        progress = float(step - config.phase_3_end) / float(max(1, config.total_steps - config.phase_3_end))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return config.min_lr + (config.phase_3_min_lr - config.min_lr) * cosine_decay


class GradedCurriculumStreamer:
    """
    Ultra-low-footprint, on-the-fly streaming dataset manager.
    Dynamically loads and yields text samples according to the active curriculum phase.
    """

    def __init__(
        self,
        tokenizer: Any,
        config: CurriculumScheduleConfig,
        search_dirs: Optional[List[str]] = None,
        seed: int = 42,
        mix: str = "auto",
        offline: bool = False,
        rank: int = 0,
        world_size: int = 1,
        online_streaming: Optional[bool] = None,
        prefetch_size: int = 32,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.mix = mix.lower()
        self.rng = random.Random(seed)
        self.rank = rank
        self.world_size = max(1, world_size)
        self.prefetch_size = prefetch_size

        if online_streaming is None:
            tok_cls = type(tokenizer).__name__
            is_mock = "Mock" in tok_cls
            env_off = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            self.online_streaming = (not offline) and (not env_off) and (not is_mock)
        else:
            self.online_streaming = online_streaming

        self.search_dirs = search_dirs or [
            "data",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        ]
        self._dataset_files = self._locate_local_datasets()
        self._local_pools: Dict[str, List[Tuple[str, str]]] = {}
        self._local_cursors: Dict[str, int] = {}

        self._stop_event = threading.Event()
        self._worker_lock = threading.Lock()
        self._online_queues: Dict[str, queue.Queue] = {}
        self._online_connected: Dict[str, bool] = {}
        self._online_failed: Dict[str, bool] = {}
        self._online_threads: Dict[str, threading.Thread] = {}

    def close(self):
        """Signal any background online streaming workers to stop."""
        self._stop_event.set()

    def shutdown(self):
        """Alias for close() matching PrefetchedBatcher API."""
        self.close()

    def _locate_local_datasets(self) -> Dict[str, List[str]]:
        """Scans search directories to catalog available parquet files."""
        catalog: Dict[str, List[str]] = {
            "basic": [],
            "reasoning": [],
            "code": [],
            "workflows": [],
            "gold": [],
        }

        found_files = []
        for d in self.search_dirs:
            if os.path.exists(d):
                found_files.extend(glob.glob(f"{d}/**/*.parquet", recursive=True))

        for f in found_files:
            fname = os.path.basename(f).lower()
            if "world_knowledge" in fname or "canonical" in fname or "cosmopedia" in fname or "fineweb" in fname:
                catalog["basic"].append(f)
            elif "openhermes" in fname or "gsm8k" in fname or "finemath" in fname:
                catalog["reasoning"].append(f)
            elif "stack" in fname or "python" in fname:
                catalog["code"].append(f)
            elif "ultrachat" in fname or "lmsys" in fname or "tool" in fname or "smoltalk" in fname:
                catalog["workflows"].append(f)
            elif "cerebro_cognitive" in fname:
                catalog["gold"].append(f)
                catalog["reasoning"].append(f)

        return catalog

    def _generate_synthetic_basic(self) -> str:
        """On-the-fly fallback generator for Phase 1 foundational syntax."""
        subjects = ["The photon", "An electron", "The neural membrane", "A harmonic oscillator", "The protein structure", "A binary tree", "The curvature tensor"]
        verbs = ["propagates through", "exhibits resonance with", "maintains equilibrium across", "governs the evolution of", "encodes information in", "modulates"]
        objects = ["the concentric radial boundary.", "the latent state manifold.", "the phase clock cycle.", "orthogonal spherical harmonics.", "the axonal shortcut pathways."]
        s = self.rng.choice(subjects)
        v = self.rng.choice(verbs)
        o = self.rng.choice(objects)
        return f"{s} {v} {o} Specifically, its spatial gradient remains bounded within contractive limits."

    def _generate_synthetic_multimodal(self, phase: CurriculumPhase) -> Tuple[str, int, torch.Tensor]:
        """Generates paired perceptual visual patches and multimodal reasoning dialogue."""
        P = getattr(self.config, "num_visual_patches", 16)
        V = getattr(self.config, "visual_dim", 512)
        spatial_freq = self.rng.uniform(0.5, 4.0)
        t = torch.linspace(0, math.pi * spatial_freq, P)
        patches = torch.zeros(P, V)
        for p in range(P):
            patches[p] = torch.sin(t[p] + torch.arange(V) * 0.05) + 0.05 * torch.randn(V)

        if phase == CurriculumPhase.PHASE_1_BASIC:
            entities = [
                "a biological cell with a prominent spherical membrane",
                "a crystalline lattice with hexagonal symmetry",
                "an electron microscope scan of neural synapses",
                "a planetary spheroid showing atmospheric circulation",
            ]
            e = self.rng.choice(entities)
            text = f"Visual Input: <perceptual_flux>\nQuery: What entity is visible in this visual field?\nResponse: The visual field displays {e}."
            return text, 0, patches

        elif phase == CurriculumPhase.PHASE_2_REASONING:
            x1, y1 = round(self.rng.uniform(0.1, 0.4), 2), round(self.rng.uniform(0.1, 0.4), 2)
            x2, y2 = round(x1 + self.rng.uniform(0.2, 0.4), 2), round(y1 + self.rng.uniform(0.2, 0.4), 2)
            text = (
                f"Visual Input: <perceptual_flux>\n"
                f"Query: Localize the primary anomaly and compute its bounding box.\n"
                f"Thought: Analyzing the spatial gradient across patches. Peak contrast occurs in the upper quadrant.\n"
                f"Answer: The target anomaly is bounded by [box: {x1}, {y1}, {x2}, {y2}]."
            )
            return text, 0, patches

        elif phase == CurriculumPhase.PHASE_3_WORKFLOWS:
            text = (
                f"Visual Input: <perceptual_flux>\n"
                f"User: Inspect this image tensor and run python code to plot its intensity profile.\n"
                f"<action:PYTHON> import numpy as np\nprofile = np.mean(image_patches, axis=-1)\nprint(f'Mean intensity: {{profile[:3]}}') </action>\n"
                f"<observation> Mean intensity: [0.42, 0.71, 0.38] </observation>\n"
                f"Assistant: The intensity profile peaks at patch 1 with a value of 0.71."
            )
            return text, 2, patches  # Action 2: PYTHON

        else:  # PHASE_4_CONSOLIDATION
            text = (
                f"Visual Input: <perceptual_flux>\n"
                f"Query: Summarize multimodal observations.\n"
                f"Thought: Grounding visual patch activations with prior cognitive state.\n"
                f"Answer: Concentric laminae exhibit stable contractive convergence across all 32 shells."
            )
            return text, 0, patches

    def _generate_synthetic_agentic(self) -> Tuple[str, int]:
        """On-the-fly generator for Phase 3 tool workflows."""
        tool_scenarios = [
            (
                "User: What is 482 multiplied by 19?\n<action:CALCULATOR> 482 * 19 </action>\n<observation> 9158 </observation>\nAssistant: The result of 482 multiplied by 19 is 9,158.",
                5,  # CALCULATOR
            ),
            (
                "User: Check how many files are in the current directory.\n<action:BASH> ls -1 | wc -l </action>\n<observation> 14 </observation>\nAssistant: There are 14 files in the current working directory.",
                1,  # BASH
            ),
            (
                "User: Write a python snippet to compute the Fibonacci sequence of length 5.\n<action:PYTHON> def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b\nprint(list(fib(5))) </action>\n<observation> [0, 1, 1, 2, 3] </observation>\nAssistant: Here is the Fibonacci sequence: [0, 1, 1, 2, 3].",
                2,  # PYTHON
            ),
            (
                "User: Read the configuration from config.json.\n<action:FILE_READ> config.json </action>\n<observation> {\"surface_dim\": 768, \"shells\": 32} </observation>\nAssistant: The config specifies 32 shells and a surface dimension of 768.",
                3,  # FILE_READ
            ),
        ]
        return self.rng.choice(tool_scenarios)

    def _stream_parquet_samples(self, file_path: str, max_samples: Optional[int] = None) -> Iterator[Tuple[str, str]]:
        """Memory-mapped streaming reader yielding extracted (prompt, response) pairs."""
        count = 0
        try:
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    q_txt, r_txt = extract_prompt_response_pair(row)
                    if q_txt and r_txt:
                        yield q_txt, r_txt
                        count += 1
                        if max_samples is not None and count >= max_samples:
                            return
        except Exception:
            return

    def _load_local_pool(self, bucket: str) -> List[Tuple[str, str]]:
        """Loads, shards across DDP ranks, and shuffles all local Parquet pairs for a bucket."""
        files = list(self._dataset_files.get(bucket, []))
        if not files:
            for fallback_key in ("gold", "reasoning", "workflows", "basic", "code"):
                files.extend(self._dataset_files.get(fallback_key, []))
        pairs: List[Tuple[str, str]] = []
        seen_files = set()
        for f in files:
            if f in seen_files:
                continue
            seen_files.add(f)
            pairs.extend(self._stream_parquet_samples(f))
        if not pairs:
            return []
        self.rng.shuffle(pairs)
        if self.world_size > 1 and len(pairs) >= self.world_size:
            pairs = pairs[self.rank :: self.world_size]
        return pairs

    def _next_local_pair(self, bucket: str) -> Optional[Tuple[str, str]]:
        """Returns the next distinct local pair for `bucket`, reshuffling at epoch boundaries."""
        pool = self._local_pools.get(bucket)
        if pool is None:
            pool = self._load_local_pool(bucket)
            self._local_pools[bucket] = pool
            self._local_cursors[bucket] = 0
        if not pool:
            return None
        idx = self._local_cursors.get(bucket, 0)
        if idx >= len(pool):
            self.rng.shuffle(pool)
            idx = 0
        pair = pool[idx]
        self._local_cursors[bucket] = idx + 1
        return pair

    def _ensure_online_worker(self, stream_key: str) -> Optional[queue.Queue]:
        if not self.online_streaming or self._online_failed.get(stream_key, False):
            return None
        with self._worker_lock:
            if self._online_failed.get(stream_key, False):
                return None
            if stream_key in self._online_queues:
                return self._online_queues[stream_key]
            q: queue.Queue = queue.Queue(maxsize=self.prefetch_size)
            self._online_queues[stream_key] = q

            def _producer():
                max_attempts = 3
                for attempt in range(max_attempts):
                    if self._stop_event.is_set():
                        return
                    try:
                        it = build_online_dataset_iterator(
                            dataset_source=stream_key,
                            rank=self.rank,
                            world_size=self.world_size,
                        )
                        for row in it:
                            if self._stop_event.is_set():
                                return
                            q_txt, r_txt = extract_prompt_response_pair(row)
                            if len(q_txt.strip()) >= 2 and len(r_txt.strip()) >= 15:
                                while not self._stop_event.is_set():
                                    try:
                                        q.put((q_txt, r_txt), timeout=0.5)
                                        break
                                    except queue.Full:
                                        continue
                        return
                    except Exception as exc:
                        err_msg = str(exc).lower()
                        is_rate_limit = "429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg
                        if is_rate_limit:
                            print(
                                f"⚠️ [online-stream:{stream_key}:r{self.rank}] "
                                f"HuggingFace 429 Rate Limit encountered. Switching immediately to resilient local/synthetic curriculum."
                            )
                            self._online_failed[stream_key] = True
                            return
                        print(
                            f"⚠️ [online-stream:{stream_key}:r{self.rank}] "
                            f"attempt {attempt + 1}/{max_attempts} failed ({type(exc).__name__}: {exc})"
                        )
                        if attempt + 1 >= max_attempts:
                            self._online_failed[stream_key] = True
                            return
                        self._stop_event.wait(timeout=min(2.0 ** attempt, 8.0))

            t = threading.Thread(
                target=_producer,
                name=f"online-stream-{stream_key}-r{self.rank}",
                daemon=True,
            )
            self._online_threads[stream_key] = t
            t.start()
            return q

    def _next_online_pair(self, stream_key: str) -> Optional[Tuple[str, str]]:
        q = self._ensure_online_worker(stream_key)
        if q is None:
            return None
        # Use low-latency timeout so queue depletion never blocks or throttles GPU throughput
        wait_s = 0.2 if not self._online_connected.get(stream_key, False) else 0.005
        try:
            pair = q.get(timeout=wait_s)
            self._online_connected[stream_key] = True
            return pair
        except queue.Empty:
            return None

    @staticmethod
    def _split_legacy_text(text: str) -> Tuple[str, str]:
        if "Response:" in text:
            p_txt, r_txt = text.split("Response:", 1)
            return p_txt.replace("Query:", "").strip(), r_txt.strip()
        if "Answer:" in text:
            p_txt, r_txt = text.split("Answer:", 1)
            return p_txt.replace("Question:", "").strip(), r_txt.strip()
        if "Assistant:" in text:
            p_txt, r_txt = text.split("Assistant:", 1)
            return p_txt.replace("User:", "").strip(), r_txt.strip()
        if "\n\n" in text:
            p_txt, r_txt = text.split("\n\n", 1)
            return p_txt.strip(), r_txt.strip()
        half = max(1, len(text) // 2)
        return text[:half].strip(), text[half:].strip()

    def _resolve_active_phase(self, phase: CurriculumPhase) -> CurriculumPhase:
        if self.mix == "foundation":
            return CurriculumPhase.PHASE_1_BASIC
        if self.mix == "reasoning":
            return CurriculumPhase.PHASE_2_REASONING
        if self.mix == "agentic":
            return CurriculumPhase.PHASE_3_WORKFLOWS
        if self.mix in ("consolidation", "balanced", "gold"):
            return CurriculumPhase.PHASE_4_CONSOLIDATION
        return phase

    def get_pair_for_phase(
        self, phase: CurriculumPhase
    ) -> Tuple[str, str, int, Optional[torch.Tensor]]:
        """Sample a `(prompt_text, response_text, action_label, perceptual_flux)` tuple."""
        active_phase = self._resolve_active_phase(phase)

        if self.mix == "code":
            pair = self._next_online_pair("code") or self._next_local_pair("code")
            if pair is not None:
                return pair[0], pair[1], 2, None
            return (
                "Write a Python function to compute quicksort cleanly.",
                "def quicksort(arr):\n"
                "    if len(arr) <= 1:\n        return arr\n"
                "    pivot = arr[len(arr) // 2]\n"
                "    left = [x for x in arr if x < pivot]\n"
                "    middle = [x for x in arr if x == pivot]\n"
                "    right = [x for x in arr if x > pivot]\n"
                "    return quicksort(left) + middle + quicksort(right)",
                2,
                None,
            )

        mm_ratio = getattr(self.config, "multimodal_sample_ratio", 0.0)
        if mm_ratio > 0.0 and self.rng.random() < mm_ratio:
            mm_text, act, patches = self._generate_synthetic_multimodal(active_phase)
            p_txt, r_txt = self._split_legacy_text(mm_text)
            return p_txt, r_txt, act, patches

        if active_phase == CurriculumPhase.PHASE_1_BASIC:
            pair = self._next_online_pair("dense_edu") or self._next_local_pair("basic")
            if pair is not None:
                return pair[0], pair[1], 0, None
            return self._split_legacy_text(self._generate_synthetic_basic()) + (0, None)

        if active_phase == CurriculumPhase.PHASE_2_REASONING:
            pair = self._next_online_pair("reasoning_anneal") or self._next_local_pair("reasoning")
            if pair is not None:
                return pair[0], pair[1], 0, None
            return self._split_legacy_text(self._generate_synthetic_basic()) + (0, None)

        if active_phase == CurriculumPhase.PHASE_3_WORKFLOWS:
            if self.rng.random() < 0.4:
                text, act = self._generate_synthetic_agentic()
                p_txt, r_txt = self._split_legacy_text(text)
                return p_txt, r_txt, act, None
            pair = self._next_online_pair("conversation_sft") or self._next_local_pair("workflows")
            if pair is not None:
                return pair[0], pair[1], 0, None
            text, act = self._generate_synthetic_agentic()
            p_txt, r_txt = self._split_legacy_text(text)
            return p_txt, r_txt, act, None

        # PHASE_4_CONSOLIDATION
        r = self.rng.random()
        if r < 0.05:
            return self.get_pair_for_phase(CurriculumPhase.PHASE_1_BASIC)
        if r < 0.20:
            return self.get_pair_for_phase(CurriculumPhase.PHASE_2_REASONING)
        if r < 0.40:
            return self.get_pair_for_phase(CurriculumPhase.PHASE_3_WORKFLOWS)
        pair = self._next_online_pair("consolidation") or self._next_local_pair("gold")
        if pair is not None:
            return pair[0], pair[1], 0, None
        return self._split_legacy_text(self._generate_synthetic_basic()) + (0, None)

    def get_sample_for_phase(self, phase: CurriculumPhase) -> Tuple[str, int, Optional[torch.Tensor]]:
        """Legacy string wrapper returning `(combined_text, action_label, perceptual_flux)`."""
        q_txt, r_txt, act, patches = self.get_pair_for_phase(phase)
        return f"{q_txt}\n\n{r_txt}", act, patches

    def get_batch(
        self,
        step: int,
        device: torch.device,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], CurriculumPhase]:
        """
        Generates a Strategy-A formatted fixed-length batch on-the-fly.
        Returns: (input_ids, targets, action_labels, perceptual_inputs, current_phase)
        """
        B = batch_size or self.config.batch_size
        T = self.config.seq_len
        phase = get_curriculum_phase(step, self.config)

        batch_input_ids = []
        batch_targets = []
        batch_actions = []
        batch_patches = []
        has_any_patches = False

        vocab_size = getattr(self.tokenizer, "vocab_size", 32000)
        eos_id = getattr(self.tokenizer, "eos_token_id", 2)
        if eos_id is None:
            eos_id = 2

        for _ in range(B):
            q_txt, r_txt, action_id, patches = self.get_pair_for_phase(phase)
            formatted = f"{format_stimulus_prompt(q_txt)}\n{format_motor_response(r_txt)}"
            raw_tokens = (
                self.tokenizer.encode(formatted, add_special_tokens=False)
                if hasattr(self.tokenizer, "encode")
                else [ord(c) % vocab_size for c in formatted]
            )
            if not raw_tokens or raw_tokens[-1] != eos_id:
                raw_tokens = raw_tokens + [eos_id]

            safe_tokens = [t % vocab_size for t in raw_tokens]

            if len(safe_tokens) < T + 1:
                pad_needed = (T + 1) - len(safe_tokens)
                valid_len = len(safe_tokens)
                padded_inp = safe_tokens + [0] * pad_needed
                padded_tgt = safe_tokens + [-100] * pad_needed
                inp = padded_inp[:T]
                tgt = padded_tgt[1 : T + 1]
                if valid_len == 0:
                    tgt = [-100] * T
            else:
                max_start = len(safe_tokens) - (T + 1)
                if max_start > 0 and self.rng.random() < 0.25:
                    start_idx = max_start
                else:
                    start_idx = self.rng.randint(0, max(0, max_start))
                window_tokens = safe_tokens[start_idx : start_idx + T + 1]
                inp = window_tokens[:T]
                tgt = window_tokens[1 : T + 1]

            batch_input_ids.append(inp)
            batch_targets.append(tgt)
            batch_actions.append(action_id)

            if patches is not None:
                has_any_patches = True
                batch_patches.append(patches)
            else:
                batch_patches.append(None)

        input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
        targets = torch.tensor(batch_targets, dtype=torch.long, device=device)
        action_labels = torch.tensor(batch_actions, dtype=torch.long, device=device)

        perceptual_inputs = None
        if has_any_patches:
            P = getattr(self.config, "num_visual_patches", 16)
            V = getattr(self.config, "visual_dim", 512)
            filled = [p if p is not None else torch.zeros(P, V) for p in batch_patches]
            perceptual_inputs = torch.stack(filled, dim=0).to(device)

        return input_ids, targets, action_labels, perceptual_inputs, phase

    def get_elastic_batch(
        self,
        step: int,
        device: torch.device,
        batch_size: Optional[int] = None,
        total_window: int = 1024,
        tail_probability: float = 0.25,
        target_pad_id: int = -100,
        buckets: Optional[List[Tuple[str, int, int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], CurriculumPhase, str]:
        """
        V4-style Strategy-A Elastic / Static Sliding Context Window batcher with teacher feedback.
        Pass explicit `buckets=[("static", prompt_len, seq_len)]` for exact TPU sequence parity.
        """
        B = batch_size or self.config.batch_size
        phase = get_curriculum_phase(step, self.config)
        active_buckets = buckets if buckets is not None else get_default_elastic_buckets(total_window)
        vocab_size = getattr(self.tokenizer, "vocab_size", 32000)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        if pad_id is None:
            pad_id = 0
        eos_id = getattr(self.tokenizer, "eos_token_id", 2)
        if eos_id is None:
            eos_id = 2

        records = []
        batch_actions = []
        batch_patches = []
        has_any_patches = False

        for _ in range(B):
            p_txt, r_txt, action_id, patches = self.get_pair_for_phase(phase)
            batch_actions.append(action_id)
            if patches is not None:
                has_any_patches = True
                batch_patches.append(patches)
            else:
                batch_patches.append(None)

            # Strategy A delimited formatting matching ElasticBucketBatcher (TPU JAX parity)
            q_fmt = format_stimulus_prompt(p_txt)
            r_fmt = format_motor_response(r_txt)

            q_ids = (
                self.tokenizer.encode(q_fmt, add_special_tokens=False)
                if hasattr(self.tokenizer, "encode")
                else [ord(c) % vocab_size for c in q_fmt]
            )
            r_ids = (
                self.tokenizer.encode(r_fmt, add_special_tokens=False)
                if hasattr(self.tokenizer, "encode")
                else [ord(c) % vocab_size for c in r_fmt]
            )
            if not r_ids or r_ids[-1] != eos_id:
                r_ids = r_ids + [eos_id]

            records.append((q_ids, r_ids))

        avg_q = max(1, sum(len(q) for q, _ in records) // len(records))
        avg_r = max(1, sum(len(r) for _, r in records) // len(records))
        b_idx = select_best_bucket(avg_q, avg_r, active_buckets)
        bucket_name, bp, bt = active_buckets[b_idx]

        batch_prompts = []
        batch_targets = []
        batch_feedback = []

        for q_ids, r_ids in records:
            if len(q_ids) > bp:
                q_padded = q_ids[:bp]
            else:
                q_padded = q_ids + [pad_id] * (bp - len(q_ids))

            if len(r_ids) > bt:
                context_token, scored = sample_window(r_ids, bt, self.rng, tail_probability)
                r_padded = scored
            else:
                context_token = None
                r_padded = r_ids + [target_pad_id] * (bt - len(r_ids))

            feed_row = [target_pad_id] * bt
            if context_token is not None:
                feed_row[0] = context_token
            for t_idx in range(1, bt):
                feed_row[t_idx] = r_padded[t_idx - 1]

            batch_prompts.append([tok % vocab_size for tok in q_padded])
            batch_targets.append([tok if tok == target_pad_id else (tok % vocab_size) for tok in r_padded])
            batch_feedback.append([tok if tok == target_pad_id else (tok % vocab_size) for tok in feed_row])

        prompts = torch.tensor(batch_prompts, dtype=torch.long, device=device)
        targets = torch.tensor(batch_targets, dtype=torch.long, device=device)
        feedback = torch.tensor(batch_feedback, dtype=torch.long, device=device)
        action_labels = torch.tensor(batch_actions, dtype=torch.long, device=device)

        perceptual_inputs = None
        if has_any_patches:
            P = getattr(self.config, "num_visual_patches", 16)
            V = getattr(self.config, "visual_dim", 512)
            filled = [p if p is not None else torch.zeros(P, V) for p in batch_patches]
            perceptual_inputs = torch.stack(filled, dim=0).to(device)

        return prompts, targets, feedback, action_labels, perceptual_inputs, phase, bucket_name
