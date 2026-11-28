"""
🧠 CerebroSpheroid V5: Exploration-Augmented Reward Engine
==========================================================
Provides an active intrinsic-motivation and exploration-driven reward model
designed specifically for CerebroSpheroid V5 reinforcement learning (PPO / GRPO).

Core Pillars:
1. Random Network Distillation (RND) Curiosity:
   - Measures representation novelty via prediction error of a fixed random target network.
   - Decays naturally as the policy masters familiar reasoning spaces, incentivizing novel paths.
2. Concentric Manifold Volumetric Exploration:
   - Evaluates active effective rank across the 16/32 concentric spherical shells.
   - Rewards non-degenerate radial wave convection and shell entropy to prevent latent mode collapse.
3. Group Trajectory Diversity (GRPO Intra-Group Dispersion):
   - Rewards pairwise semantic and n-gram divergence among parallel candidate rollouts for a prompt.
4. Process & Self-Verification Grounding:
   - Rewards active cognitive exploration (hypothesis testing, alternative paths, sanity checks)
     while verifying final correctness against ground truth.
5. Anti-Degeneration Guardrails:
   - Penalizes repetitive stuttering loops, runaway verbosity, and character gibberish.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd:
    """Tracks running mean and standard deviation for reward normalization (Welford's algorithm)."""

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon

    def update(self, x: torch.Tensor) -> None:
        batch_mean = torch.mean(x.detach().float(), dim=0)
        batch_var = torch.var(x.detach().float(), dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor, clamp_min: float = -5.0, clamp_max: float = 5.0) -> torch.Tensor:
        std = torch.sqrt(self.var + 1e-8)
        normed = (x - self.mean) / std
        return torch.clamp(normed, min=clamp_min, max=clamp_max)


class RNDTargetNetwork(nn.Module):
    """Frozen random target network mapping representations into an arbitrary embedding space."""

    def __init__(self, in_dim: int = 768, out_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        # Orthogonal initialization with fixed parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDPredictorNetwork(nn.Module):
    """Trainable predictor network learning to match the frozen target network on visited states."""

    def __init__(self, in_dim: int = 768, out_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CerebroCuriosityRND(nn.Module):
    """
    Random Network Distillation (RND) Curiosity Engine.
    Computes intrinsic exploration rewards based on state / trajectory novelty:
    R_curiosity = || f_pred(s) - f_target(s) ||^2
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 256, lr: float = 1e-4):
        super().__init__()
        self.target = RNDTargetNetwork(in_dim=in_dim, out_dim=out_dim)
        self.predictor = RNDPredictorNetwork(in_dim=in_dim, out_dim=out_dim)
        self.optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=lr, weight_decay=1e-4)
        self.running_stats = RunningMeanStd()

    def compute_novelty(self, states: torch.Tensor) -> torch.Tensor:
        """
        Computes novelty error per representation vector.
        Args:
            states: [B, in_dim]
        Returns:
            novelty: [B] normalized positive curiosity score
        """
        dev = next(self.predictor.parameters()).device
        x = states.to(dev).float()
        with torch.no_grad():
            target_feat = self.target(x)
            pred_feat = self.predictor(x)
            raw_err = (pred_feat - target_feat).pow(2).mean(dim=-1)  # [B]
            self.running_stats.update(raw_err)
            normed_err = self.running_stats.normalize(raw_err)
            # Re-scale into smooth positive reward [0, 2]
            return torch.sigmoid(normed_err) * 2.0

    def update_predictor(self, states: torch.Tensor) -> float:
        """Trains predictor network on newly visited state representations to decay familiarity."""
        dev = next(self.predictor.parameters()).device
        x = states.to(dev).float()
        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        loss = F.mse_loss(pred_feat, target_feat)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())


class ConcentricManifoldExplorer:
    """
    Evaluates volumetric geometric exploration across Cerebro's 16/32 concentric spherical shells.
    Incentivizes:
      1. Effective Subspace Rank: Multi-shell diversity preventing degenerate 1D attractor collapse.
      2. Radial Convection Flow: Wave propagation velocity (Shell 0 <-> Shell K-1).
      3. Shell Energy Entropy: Active participation of both core and mantle laminae.
    """

    @staticmethod
    def evaluate_manifold_exploration(state: torch.Tensor) -> Dict[str, float]:
        """
        Computes manifold exploration metrics on concentric state tensor: [B, K, D] or [K, D].
        """
        if state.ndim == 2:
            state = state.unsqueeze(0)  # [1, K, D]
        B, K, D = state.shape
        state_f = state.detach().float()

        # 1. Effective Rank per shell via SVD
        ranks = []
        rms_per_shell = []
        for k in range(K):
            s_k = state_f[:, k, :]
            rms = s_k.pow(2).mean().sqrt().item()
            rms_per_shell.append(rms)

            if B > 1:
                centered = s_k - s_k.mean(dim=0, keepdim=True)
                try:
                    s_vals = torch.linalg.svdvals(centered)
                    s2 = s_vals.pow(2)
                    eff_rank = (s2.sum().pow(2) / (s2.pow(2).sum() + 1e-9)).item()
                except Exception:
                    eff_rank = 1.0
            else:
                # Spatial rank across feature dimension
                try:
                    s_vals = torch.linalg.svdvals(state_f[0])
                    s2 = s_vals.pow(2)
                    eff_rank = (s2.sum().pow(2) / (s2.pow(2).sum() + 1e-9)).item()
                except Exception:
                    eff_rank = 1.0
            ranks.append(eff_rank)

        mean_rank = float(sum(ranks) / max(len(ranks), 1))

        # 2. Radial Wave Convection Velocity (Inter-shell variance & propagation)
        inter_shell_diffs = []
        for k in range(K - 1):
            diff = (state_f[:, k + 1, :] - state_f[:, k, :]).norm(dim=-1).mean().item()
            inter_shell_diffs.append(diff)
        radial_flow = float(sum(inter_shell_diffs) / max(len(inter_shell_diffs), 1))

        # 3. Shell Energy Distribution Entropy
        energies = torch.tensor(rms_per_shell, dtype=torch.float32)
        total_e = energies.sum() + 1e-8
        probs = energies / total_e
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        max_entropy = math.log(max(K, 1))
        norm_entropy = float(entropy / max(max_entropy, 1e-8))

        # Composite Manifold Exploration Bonus: [0, 1.5]
        # Rewarding dimensional rank, active radial flow, and distributed shell engagement
        rank_bonus = min(1.0, mean_rank / max(min(B, D), 8))
        flow_bonus = min(1.0, radial_flow / 5.0)
        composite_manifold = 0.40 * rank_bonus + 0.35 * flow_bonus + 0.25 * norm_entropy

        return {
            "manifold_reward": float(composite_manifold),
            "effective_rank": mean_rank,
            "radial_flow": radial_flow,
            "shell_entropy": norm_entropy,
        }


class TrajectoryDiversityEvaluator:
    """
    Computes intra-group trajectory diversity for Group Relative Policy Optimization (GRPO).
    Incentivizes sampling diverse reasoning paths, hypotheses, and proof strategies.
    """

    @staticmethod
    def _extract_ngrams(tokens: List[int], n: int = 3) -> set:
        if len(tokens) < n:
            return set()
        return set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    @classmethod
    def compute_jaccard_distance(cls, tokens_a: List[int], tokens_b: List[int], n: int = 3) -> float:
        set_a = cls._extract_ngrams(tokens_a, n)
        set_b = cls._extract_ngrams(tokens_b, n)
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return 1.0 - (intersection / max(union, 1))

    @classmethod
    def evaluate_group_diversity(
        cls,
        candidate_token_lists: List[List[int]],
        candidate_embeddings: Optional[torch.Tensor] = None,
    ) -> List[float]:
        """
        Evaluates leave-one-out trajectory distinctiveness for each candidate in a group.
        Returns:
            diversity_rewards: list of floats in [0, 1] per candidate.
        """
        G = len(candidate_token_lists)
        if G <= 1:
            return [0.0] * G

        pairwise_distances = np.zeros((G, G), dtype=np.float32)

        # 1. N-Gram Structural Diversity
        for i in range(G):
            for j in range(i + 1, G):
                dist = cls.compute_jaccard_distance(candidate_token_lists[i], candidate_token_lists[j], n=3)
                pairwise_distances[i, j] = dist
                pairwise_distances[j, i] = dist

        # 2. Semantic Embedding Cosine Distance (if provided)
        if candidate_embeddings is not None and candidate_embeddings.shape[0] == G:
            embs = F.normalize(candidate_embeddings.float(), p=2, dim=-1)
            sim_matrix = torch.matmul(embs, embs.T).cpu().numpy()
            sem_dist = 1.0 - np.clip(sim_matrix, -1.0, 1.0)
            pairwise_distances = 0.5 * pairwise_distances + 0.5 * sem_dist

        # Leave-one-out average distance
        group_diversity_rewards = []
        for i in range(G):
            other_indices = [j for j in range(G) if j != i]
            mean_dist = float(np.mean(pairwise_distances[i, other_indices]))
            group_diversity_rewards.append(mean_dist)

        return group_diversity_rewards


class ProcessVerificationEvaluator:
    """
    Rewards active cognitive exploration:
      - Alternative hypothesis generation ("Alternatively", "Suppose instead")
      - Self-correction and sanity checks ("Let's verify", "Checking if", "Wait, this implies")
      - Exact math / symbolic correctness verification against ground truth (when available).
    """

    EXPLORATION_TRIGGERS = [
        r"\balternatively\b",
        r"\banother approach\b",
        r"\bif we consider\b",
        r"\bsuppose instead\b",
        r"\blet us test\b",
        r"\bwhat if\b",
        r"\bcounterexample\b",
    ]

    VERIFICATION_TRIGGERS = [
        r"\blet me check\b",
        r"\blet's verify\b",
        r"\bchecking if\b",
        r"\bsubstituting back\b",
        r"\bthis confirms\b",
        r"\bwe can test this\b",
        r"\bto ensure correctness\b",
        r"\bwait, let's double check\b",
    ]

    CORRECTION_TRIGGERS = [
        r"\bwait, that's incorrect\b",
        r"\bcorrection:\b",
        r"\brevising\b",
        r"\bthat contradicts\b",
        r"\blet's recompute\b",
    ]

    @classmethod
    def evaluate_process_and_verification(
        cls,
        text: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        text_lower = text.lower()

        # 1. Alternative hypothesis exploration bonus
        explore_hits = sum(1 for pat in cls.EXPLORATION_TRIGGERS if re.search(pat, text_lower))
        explore_score = min(1.0, explore_hits * 0.35)

        # 2. Step-by-step verification bonus
        verify_hits = sum(1 for pat in cls.VERIFICATION_TRIGGERS if re.search(pat, text_lower))
        verify_score = min(1.0, verify_hits * 0.35)

        # 3. Self-correction recovery bonus
        correct_hits = sum(1 for pat in cls.CORRECTION_TRIGGERS if re.search(pat, text_lower))
        correction_score = min(1.0, correct_hits * 0.50)

        process_reward = 0.40 * explore_score + 0.40 * verify_score + 0.20 * correction_score

        # 4. Ground Truth Verification (if provided)
        outcome_reward = 0.0
        if ground_truth is not None and ground_truth.strip():
            gt_clean = ground_truth.strip().lower()
            # Extract boxed or final answer from candidate
            ans_match = re.findall(r"(?:the answer is|equals|####|result:?)\s*([^\n\.]+)", text_lower)
            extracted_ans = ans_match[-1].strip() if ans_match else text_lower

            if gt_clean in extracted_ans or gt_clean in text_lower:
                outcome_reward = 1.0
            else:
                # Check for numerical equivalence
                try:
                    gt_num = float(re.findall(r"-?\d+(?:\.\d+)?", gt_clean)[0])
                    cand_nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", extracted_ans)]
                    if cand_nums and abs(cand_nums[-1] - gt_num) < 1e-4:
                        outcome_reward = 1.0
                except Exception:
                    pass

        return {
            "process_reward": float(process_reward),
            "hypothesis_exploration": float(explore_score),
            "step_verification": float(verify_score),
            "self_correction": float(correction_score),
            "outcome_reward": float(outcome_reward),
        }


class ToolExecutionEvaluator:
    """
    Evaluates tool invocation and verification outcomes for candidate rollouts.
    Dispatches extracted tool calls (calc, factual, python, etc.) to UnifiedToolDispatcher.
    Awards rewards for:
      - Valid, non-crashing tool execution
      - Execution results matching expected ground truth
    Penalizes:
      - Malformed, crashing, or non-parseable tool invocations
    """

    def __init__(self, dispatcher: Optional[Any] = None):
        self._dispatcher = dispatcher

    @property
    def dispatcher(self):
        if self._dispatcher is None:
            try:
                from src.tools.unified_tool_dispatcher import UnifiedToolDispatcher
                self._dispatcher = UnifiedToolDispatcher()
            except Exception:
                self._dispatcher = None
        return self._dispatcher

    def extract_tool_calls(self, text: str) -> List[str]:
        """Extracts tool call strings from XML tags, functional calls, or JSON."""
        calls = []
        # 1. XML tags: <tool_call>...</tool_call> or <act>...</act>
        xml_matches = re.findall(r"<(?:tool_call|act)>(.*?)</(?:tool_call|act)>", text, re.DOTALL | re.IGNORECASE)
        for m in xml_matches:
            c = m.strip()
            if c:
                calls.append(c)

        # 2. Functional calls if not already in XML: calc(...), factual(...), python(...)
        if not calls:
            fn_matches = re.findall(r"\b(?:calc(?:ulator)?|factual|python|unit_convert)\s*\([^\)]*\)", text, re.IGNORECASE)
            for m in fn_matches:
                c = m.strip()
                if c:
                    calls.append(c)

        return calls

    def evaluate_tool_execution(
        self,
        text: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        calls = self.extract_tool_calls(text)
        if not calls or self.dispatcher is None:
            return {
                "has_tool_call": bool(calls),
                "tool_calls": calls,
                "tool_results": [],
                "tool_success": False,
                "tool_matches_ground_truth": False,
                "tool_reward": 0.0,
            }

        results = []
        any_success = False
        matches_gt = False

        for c_str in calls:
            try:
                res = self.dispatcher.execute_raw_call_string(c_str)
                is_err = "[Cannot parse" in res or "Traceback" in res or "Error:" in res
                results.append(res)
                if not is_err:
                    any_success = True
                    if ground_truth and ground_truth.strip():
                        gt_clean = ground_truth.strip().lower()
                        res_clean = res.strip().lower()
                        if gt_clean == res_clean or gt_clean in res_clean:
                            matches_gt = True
                        else:
                            try:
                                gt_nums = re.findall(r"-?\d+(?:\.\d+)?", gt_clean)
                                res_nums = re.findall(r"-?\d+(?:\.\d+)?", res_clean)
                                if gt_nums and res_nums:
                                    gt_val = float(gt_nums[0])
                                    res_val = float(res_nums[0])
                                    if abs(gt_val - res_val) < 1e-4:
                                        matches_gt = True
                            except Exception:
                                pass
            except Exception as e:
                results.append(f"Error: {e}")

        # Compute tool execution reward
        if matches_gt:
            tool_reward = 1.0
        elif any_success:
            tool_reward = 0.4
        else:
            # Attempted tool call but crashed / errored
            tool_reward = -0.4

        return {
            "has_tool_call": True,
            "tool_calls": calls,
            "tool_results": results,
            "tool_success": any_success,
            "tool_matches_ground_truth": matches_gt,
            "tool_reward": float(tool_reward),
        }


class ProtocolAndFormattingEvaluator:
    """
    Evaluates 3-tier cognitive boundary and agentic protocol formatting.
    Rewards:
      - Balanced cognitive tags (<think>...</think>, <tool_call>...</tool_call>)
      - Temporal preparation cues ("calling the calculator to compute...")
      - Waiting & keep-alive loop syntax ("waiting for tool response...")
      - Sensory-KV grounded response phrasing ("by calling the calculator I got...")
    Penalizes:
      - Unbalanced / dangling opened tags
      - Legacy or contaminated tag dialects (<tool_use>, <call:, etc.)
    """

    TEMPORAL_PATTERNS = [
        r"\bcalling the (?:calculator|tool|database|repl|python)\b",
        r"\bquerying the (?:knowledge|database|table|knowledgebase)\b",
        r"\bto compute this\b",
        r"\bevaluat(?:e|ing) (?:the )?request\b",
        r"\bexecuting (?:tool|query)\b",
    ]

    WAITING_PATTERNS = [
        r"\bwaiting for (?:tool|external|execution|response|database)\b",
        r"\bhas the tool responded\b",
        r"\bstill waiting for\b",
        r"\bwill return observation shortly\b",
        r"\bbackground process is running\b",
        r"\bkeep-alive\b",
    ]

    GROUNDED_ANSWER_PATTERNS = [
        r"\b(?:by|after) calling the (?:calculator|tool|database)\b",
        r"\b(?:by|after) querying the (?:knowledge database|table|database)\b",
        r"\bthe tool has responded\b",
        r"\bcalling the calculator yielded\b",
        r"\bafter waiting for the (?:database|tool)\b",
        r"\bthe answer is\b",
    ]

    LEGACY_TAG_PATTERNS = [
        r"</?tool_use>",
        r"<call:[^>]+>",
        r"</?observation>",
    ]

    @classmethod
    def evaluate_protocol_and_formatting(cls, text: str) -> Dict[str, float]:
        text_lower = text.lower()

        # 1. Tag balance checks
        unbalanced_pen = 0.0
        balanced_bonus = 0.0

        for open_tag, close_tag in [("<think>", "</think>"), ("<tool_call>", "</tool_call>"), ("<tool_response>", "</tool_response>")]:
            has_open = open_tag in text_lower
            has_close = close_tag in text_lower
            if has_open and has_close:
                balanced_bonus += 0.30
            elif has_open and not has_close:
                unbalanced_pen += 0.40

        # 2. Temporal cues
        temporal_hits = sum(1 for pat in cls.TEMPORAL_PATTERNS if re.search(pat, text_lower))
        temporal_score = min(1.0, temporal_hits * 0.35)

        # 3. Waiting / pulsating loops
        waiting_hits = sum(1 for pat in cls.WAITING_PATTERNS if re.search(pat, text_lower))
        waiting_score = min(1.0, waiting_hits * 0.35)

        # 4. Sensory-KV grounded response phrasing
        grounded_hits = sum(1 for pat in cls.GROUNDED_ANSWER_PATTERNS if re.search(pat, text_lower))
        grounded_score = min(1.0, grounded_hits * 0.35)

        # 5. Legacy / contaminated tag penalties
        legacy_hits = sum(1 for pat in cls.LEGACY_TAG_PATTERNS if re.search(pat, text_lower))
        legacy_penalty = min(1.0, legacy_hits * 0.50)

        # Composite format reward in [-1.0, 1.0]
        format_reward = (
            0.30 * min(1.0, balanced_bonus)
            + 0.25 * temporal_score
            + 0.25 * waiting_score
            + 0.20 * grounded_score
            - unbalanced_pen
            - legacy_penalty
        )

        return {
            "format_reward": float(np.clip(format_reward, -1.0, 1.0)),
            "balanced_tags": float(min(1.0, balanced_bonus)),
            "temporal_cues": float(temporal_score),
            "waiting_cues": float(waiting_score),
            "grounded_phrasing": float(grounded_score),
            "unbalanced_penalty": float(unbalanced_pen),
            "legacy_tag_penalty": float(legacy_penalty),
        }


class AntiDegenerationGuardrails:
    """Detects and penalizes repetitive loops, runaway verbosity, and character gibberish."""

    @staticmethod
    def check_repetition_penalty(tokens: List[int], n: int = 3, threshold: float = 0.25) -> float:
        if len(tokens) < 16:
            return 0.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))
        rep_ratio = 1.0 - (unique / max(total, 1))
        if rep_ratio > threshold:
            excess = rep_ratio - threshold
            return float(min(2.0, excess * 4.0))
        return 0.0

    @staticmethod
    def check_length_penalty(length: int, min_len: int = 16, target_len: int = 256, max_len: int = 1024) -> float:
        penalty = 0.0
        if length < min_len:
            # Lazy non-reasoning penalty
            penalty += (min_len - length) / float(min_len)
        elif length > max_len:
            # Runaway verbosity penalty
            penalty += min(2.0, (length - max_len) / float(max_len))
        return float(penalty)


class CerebroExplorationRewardModel(nn.Module):
    """
    Comprehensive CerebroSpheroid Exploration Reward Model.
    Combines:
      - RND Intrinsic Curiosity (Representation Novelty)
      - Concentric Manifold Volumetric Exploration (Multi-shell active rank & radial flow)
      - Group Trajectory Diversity (GRPO Leave-One-Out Dispersion)
      - Verifiable Tool Execution & Grounding (Deterministic execution of extracted tool calls)
      - Protocol & Formatting Boundary Compliance (3-tier tags, temporal, waiting, sensory-KV grounding)
      - Process & Self-Verification (Hypothesis Testing & Verification)
      - Ground Truth Outcome Verifier
      - Anti-Degeneration Guardrails
    """

    def __init__(
        self,
        surface_dim: int = 768,
        rnd_out_dim: int = 256,
        rnd_lr: float = 1e-4,
        w_outcome: float = 1.2,
        w_tool_exec: float = 1.0,
        w_format: float = 0.5,
        w_curiosity: float = 0.35,
        w_manifold: float = 0.30,
        w_diversity: float = 0.30,
        w_process: float = 0.35,
        w_penalty: float = 0.50,
        dispatcher: Optional[Any] = None,
    ):
        super().__init__()
        self.surface_dim = surface_dim
        self.curiosity_rnd = CerebroCuriosityRND(in_dim=surface_dim, out_dim=rnd_out_dim, lr=rnd_lr)
        self.manifold_evaluator = ConcentricManifoldExplorer()
        self.diversity_evaluator = TrajectoryDiversityEvaluator()
        self.process_evaluator = ProcessVerificationEvaluator()
        self.tool_evaluator = ToolExecutionEvaluator(dispatcher=dispatcher)
        self.format_evaluator = ProtocolAndFormattingEvaluator()
        self.guardrails = AntiDegenerationGuardrails()

        # Weighting factors
        self.w_outcome = w_outcome
        self.w_tool_exec = w_tool_exec
        self.w_format = w_format
        self.w_curiosity = w_curiosity
        self.w_manifold = w_manifold
        self.w_diversity = w_diversity
        self.w_process = w_process
        self.w_penalty = w_penalty

    def score_single_trajectory(
        self,
        prompt: str,
        response_text: str,
        token_ids: List[int],
        hidden_representation: Optional[torch.Tensor] = None,
        concentric_state: Optional[torch.Tensor] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Computes the exploration-augmented reward for a single rollout.
        """
        # 1. Tool execution evaluation & verification
        tool_metrics = self.tool_evaluator.evaluate_tool_execution(
            text=response_text, ground_truth=ground_truth
        )

        # 2. Protocol and formatting evaluation
        format_metrics = self.format_evaluator.evaluate_protocol_and_formatting(
            text=response_text
        )

        # 3. Process & Verification Reward
        proc_metrics = self.process_evaluator.evaluate_process_and_verification(
            text=response_text, ground_truth=ground_truth
        )

        # Outcome score: verified either via text ground truth match or tool execution match
        verified_outcome = max(
            proc_metrics["outcome_reward"],
            1.0 if tool_metrics["tool_matches_ground_truth"] else 0.0
        )

        # 4. RND Curiosity Novelty Bonus
        curiosity_score = 0.0
        if hidden_representation is not None:
            if hidden_representation.ndim == 1:
                hidden_representation = hidden_representation.unsqueeze(0)
            curiosity_score = float(self.curiosity_rnd.compute_novelty(hidden_representation)[0].item())

        # 5. Concentric Manifold Exploration Bonus
        manifold_metrics = {"manifold_reward": 0.0, "effective_rank": 1.0, "radial_flow": 0.0, "shell_entropy": 0.0}
        if concentric_state is not None:
            manifold_metrics = self.manifold_evaluator.evaluate_manifold_exploration(concentric_state)

        # 6. Anti-Degeneration Guardrail Penalties
        rep_pen = self.guardrails.check_repetition_penalty(token_ids)
        len_pen = self.guardrails.check_length_penalty(len(token_ids))
        total_penalty = rep_pen + len_pen

        # 7. Composite Reward Formulation
        total_reward = (
            self.w_outcome * verified_outcome
            + self.w_tool_exec * tool_metrics["tool_reward"]
            + self.w_format * format_metrics["format_reward"]
            + self.w_process * proc_metrics["process_reward"]
            + self.w_curiosity * curiosity_score
            + self.w_manifold * manifold_metrics["manifold_reward"]
            - self.w_penalty * total_penalty
        )

        return {
            "total_reward": float(total_reward),
            "outcome_reward": float(verified_outcome),
            "tool_reward": float(tool_metrics["tool_reward"]),
            "format_reward": float(format_metrics["format_reward"]),
            "process_reward": proc_metrics["process_reward"],
            "curiosity_novelty": curiosity_score,
            "manifold_exploration": manifold_metrics["manifold_reward"],
            "manifold_effective_rank": manifold_metrics["effective_rank"],
            "manifold_radial_flow": manifold_metrics["radial_flow"],
            "penalty": float(total_penalty),
            "repetition_penalty": float(rep_pen),
            "length_penalty": float(len_pen),
            "tool_success": tool_metrics["tool_success"],
            "tool_matches_ground_truth": tool_metrics["tool_matches_ground_truth"],
            "tool_calls": tool_metrics.get("tool_calls", []),
            "tool_results": tool_metrics.get("tool_results", []),
            "details": {
                **proc_metrics,
                **manifold_metrics,
                **tool_metrics,
                **format_metrics,
            },
        }

    def score_group_rollouts(
        self,
        prompt: str,
        candidates: List[Dict[str, Any]],
        ground_truth: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scores a group of G parallel candidates (for GRPO exploration training).
        Each candidate dict must contain:
          - 'text': str
          - 'tokens': List[int]
          - (optional) 'hidden': torch.Tensor [D]
          - (optional) 'state': torch.Tensor [K, D]
        """
        G = len(candidates)
        token_lists = [c["tokens"] for c in candidates]

        # Extract embeddings if available
        emb_list = [c.get("hidden") for c in candidates if c.get("hidden") is not None]
        emb_tensor = torch.stack(emb_list, dim=0) if len(emb_list) == G else None

        # Compute intra-group diversity bonuses
        diversity_bonuses = self.diversity_evaluator.evaluate_group_diversity(
            token_lists, candidate_embeddings=emb_tensor
        )

        scored_candidates = []
        for i, cand in enumerate(candidates):
            score_dict = self.score_single_trajectory(
                prompt=prompt,
                response_text=cand["text"],
                token_ids=cand["tokens"],
                hidden_representation=cand.get("hidden"),
                concentric_state=cand.get("state"),
                ground_truth=ground_truth,
            )
            div_reward = diversity_bonuses[i]
            score_dict["diversity_reward"] = float(div_reward)
            # Add diversity bonus to total reward
            score_dict["total_reward"] += float(self.w_diversity * div_reward)
            scored_candidates.append(score_dict)

        return scored_candidates

    def update_curiosity_with_visited_states(self, state_representations: torch.Tensor) -> float:
        """Updates the RND curiosity predictor on newly visited policy representations."""
        return self.curiosity_rnd.update_predictor(state_representations)

    def save_checkpoint(self, path: str) -> None:
        """Saves trainable exploration components (RND predictor & running stats)."""
        torch.save({
            "predictor_state": self.curiosity_rnd.predictor.state_dict(),
            "target_state": self.curiosity_rnd.target.state_dict(),
            "running_mean": self.curiosity_rnd.running_stats.mean,
            "running_var": self.curiosity_rnd.running_stats.var,
            "running_count": self.curiosity_rnd.running_stats.count,
            "config": {
                "surface_dim": self.surface_dim,
                "w_outcome": self.w_outcome,
                "w_tool_exec": self.w_tool_exec,
                "w_format": self.w_format,
                "w_curiosity": self.w_curiosity,
                "w_manifold": self.w_manifold,
                "w_diversity": self.w_diversity,
                "w_process": self.w_process,
                "w_penalty": self.w_penalty,
            },
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Restores exploration components."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.curiosity_rnd.predictor.load_state_dict(data["predictor_state"])
        if "target_state" in data:
            self.curiosity_rnd.target.load_state_dict(data["target_state"])
        self.curiosity_rnd.running_stats.mean = data["running_mean"]
        self.curiosity_rnd.running_stats.var = data["running_var"]
        self.curiosity_rnd.running_stats.count = data["running_count"]
        if "config" in data:
            cfg = data["config"]
            self.w_outcome = cfg.get("w_outcome", self.w_outcome)
            self.w_tool_exec = cfg.get("w_tool_exec", self.w_tool_exec)
            self.w_format = cfg.get("w_format", self.w_format)
            self.w_curiosity = cfg.get("w_curiosity", self.w_curiosity)
            self.w_manifold = cfg.get("w_manifold", self.w_manifold)
            self.w_diversity = cfg.get("w_diversity", self.w_diversity)
            self.w_process = cfg.get("w_process", self.w_process)
            self.w_penalty = cfg.get("w_penalty", self.w_penalty)
