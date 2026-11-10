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
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset


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
    seq_len: int = 256
    batch_size: int = 2
    visual_dim: int = 512
    num_visual_patches: int = 16
    multimodal_sample_ratio: float = 0.35


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
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.rng = random.Random(seed)
        self.search_dirs = search_dirs or [
            "data",
            "../reasoning_agent/data",
            "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/data",
            "/home/ajit/Documents/Startup/Active_Development/reasoning_agent/data",
        ]
        self._dataset_files = self._locate_local_datasets()

    def _locate_local_datasets(self) -> Dict[str, List[str]]:
        """Scans search directories to catalog available parquet files."""
        catalog: Dict[str, List[str]] = {
            "basic": [],
            "reasoning": [],
            "workflows": [],
            "gold": [],
        }

        found_files = []
        for d in self.search_dirs:
            if os.path.exists(d):
                found_files.extend(glob.glob(f"{d}/**/*.parquet", recursive=True))

        for f in found_files:
            fname = os.path.basename(f).lower()
            if "world_knowledge" in fname or "canonical" in fname:
                catalog["basic"].append(f)
            elif "openhermes" in fname or "gsm8k" in fname:
                catalog["reasoning"].append(f)
            elif "ultrachat" in fname or "lmsys" in fname or "tool" in fname:
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

    def _stream_parquet_samples(self, file_path: str, max_samples: int = 100) -> Iterator[str]:
        """Memory-mapped streaming reader for Parquet rows."""
        try:
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=32):
                df = batch.to_pandas()
                cols = df.columns.tolist()

                for _, row in df.iterrows():
                    # Parse based on schema
                    if "stimulus" in cols and "articulation" in cols:
                        yield f"Stimulus: {row['stimulus']}\nArticulation: {row['articulation']}"
                    elif "question" in cols and "answer" in cols:
                        yield f"Question: {row['question']}\nThought Process: Let us solve this step by step.\nAnswer: {row['answer']}"
                    elif "conversations" in cols and isinstance(row["conversations"], list):
                        dialogue = []
                        for msg in row["conversations"]:
                            role = msg.get("from", "speaker")
                            val = msg.get("value", "")
                            dialogue.append(f"{role.capitalize()}: {val}")
                        yield "\n".join(dialogue)
                    elif "messages" in cols and isinstance(row["messages"], list):
                        dialogue = []
                        for msg in row["messages"]:
                            content = msg.get("content", "")
                            dialogue.append(content)
                        yield "\n".join(dialogue)
        except Exception:
            return

    def get_sample_for_phase(self, phase: CurriculumPhase) -> Tuple[str, int, Optional[torch.Tensor]]:
        """
        Samples a training text, action label, and optional perceptual visual patch tensor.
        Returns:
            text: formatted text string
            action_label: int (0=CHAT, 1=BASH, 2=PYTHON, 3=FILE_READ, 4=FILE_WRITE, 5=CALC)
            perceptual_flux: Optional[torch.Tensor] [P, visual_dim]
        """
        # If sampling a multimodal perceptual example
        if self.rng.random() < getattr(self.config, "multimodal_sample_ratio", 0.35):
            return self._generate_synthetic_multimodal(phase)

        if phase == CurriculumPhase.PHASE_1_BASIC:
            files = self._dataset_files["basic"]
            if files:
                f = self.rng.choice(files)
                for s in self._stream_parquet_samples(f):
                    return s, 0, None  # Action 0 = CHAT
            return self._generate_synthetic_basic(), 0, None

        elif phase == CurriculumPhase.PHASE_2_REASONING:
            files = self._dataset_files["reasoning"]
            if files:
                f = self.rng.choice(files)
                for s in self._stream_parquet_samples(f):
                    return s, 0, None
            # Fallback
            return self._generate_synthetic_basic(), 0, None

        elif phase == CurriculumPhase.PHASE_3_WORKFLOWS:
            # 60% agentic tool traces, 40% multi-turn conversations
            if self.rng.random() < 0.6:
                text, act = self._generate_synthetic_agentic()
                return text, act, None
            else:
                files = self._dataset_files["workflows"]
                if files:
                    f = self.rng.choice(files)
                    for s in self._stream_parquet_samples(f):
                        return s, 0, None
                text, act = self._generate_synthetic_agentic()
                return text, act, None

        else:  # PHASE_4_CONSOLIDATION (Mixed Replay Buffer)
            r = self.rng.random()
            if r < 0.05:
                return self.get_sample_for_phase(CurriculumPhase.PHASE_1_BASIC)
            elif r < 0.20:
                return self.get_sample_for_phase(CurriculumPhase.PHASE_2_REASONING)
            elif r < 0.40:
                return self.get_sample_for_phase(CurriculumPhase.PHASE_3_WORKFLOWS)
            else:
                # 60% Gold-standard reasoning
                files = self._dataset_files["gold"]
                if files:
                    f = self.rng.choice(files)
                    for s in self._stream_parquet_samples(f):
                        return s, 0, None
                return self._generate_synthetic_basic(), 0, None

    def get_batch(
        self,
        step: int,
        device: torch.device,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], CurriculumPhase]:
        """
        Generates a tokenized batch on-the-fly without saving intermediate files.
        Returns:
            input_ids: [B, T]
            targets: [B, T]
            action_labels: [B]
            perceptual_inputs: Optional[torch.Tensor] [B, P, visual_dim] or None
            current_phase: CurriculumPhase
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

        for _ in range(B):
            text, action_id, patches = self.get_sample_for_phase(phase)
            raw_tokens = self.tokenizer.encode(text, add_special_tokens=False) if hasattr(self.tokenizer, "encode") else [ord(c) % vocab_size for c in text]

            # Enforce vocab bounds safely
            safe_tokens = [t % vocab_size for t in raw_tokens]

            if len(safe_tokens) < T + 1:
                # Pad with repetition or zeros
                pad_needed = (T + 1) - len(safe_tokens)
                safe_tokens = safe_tokens + [0] * pad_needed
            else:
                # Random window slice for diversity
                max_start = len(safe_tokens) - (T + 1)
                start_idx = self.rng.randint(0, max_start)
                safe_tokens = safe_tokens[start_idx : start_idx + T + 1]

            inp = safe_tokens[:T]
            tgt = safe_tokens[1 : T + 1]

            batch_input_ids.append(inp)
            batch_targets.append(tgt)
            batch_actions.append(action_id)

            if patches is not None:
                has_any_patches = True
                batch_patches.append(patches)
            else:
                P = getattr(self.config, "num_visual_patches", 16)
                V = getattr(self.config, "visual_dim", 512)
                batch_patches.append(torch.zeros(P, V))

        input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
        targets = torch.tensor(batch_targets, dtype=torch.long, device=device)
        action_labels = torch.tensor(batch_actions, dtype=torch.long, device=device)

        perceptual_inputs = None
        if has_any_patches:
            perceptual_inputs = torch.stack(batch_patches, dim=0).to(device)

        return input_ids, targets, action_labels, perceptual_inputs, phase
