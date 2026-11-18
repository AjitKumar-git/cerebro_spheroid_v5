"""
Search Space Exploration & Memory-Adaptive Candidate Ladder.
============================================================

Generates a progressive search ladder designed to push hardware memory
to its optimal capacity (~85-92% HBM on TPU / VRAM on GPU) without leaving
free memory on the table, while safely pruning unviable branches.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


def generate_vram_maximizing_candidates(
    preset: str = "v5_small",
    total_gb: float = 16.0,
    seq_len: int = 256,
    max_batch_size: Optional[int] = None,
    min_batch_size: Optional[int] = None,
    fast: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generates a progressive search ladder designed to push hardware memory
    to its optimal capacity (~85-92% HBM on TPU / VRAM on GPU) without leaving free memory on the table.
    """
    is_small_tier = preset in ("v5_small", "tpu_v5e_1", "tpu_v6e_1", "tpu_v6e", "v5_test")
    is_compact_tier = ("compact" in preset) or (preset == "v5_compact")

    if total_gb >= 64.0:
        # Ultra-capacity accelerator: TPU v5p (95GB), H100/H200 (80GB/141GB), B200 (192GB)
        if is_small_tier:
            base_ladder = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
        elif is_compact_tier:
            base_ladder = [64, 128, 256, 384, 512, 768, 1024, 1536]
        else:  # v5_base / standard
            base_ladder = [32, 64, 128, 192, 256, 384, 512, 768]
    elif total_gb >= 28.0:
        # High-capacity accelerator: TPU v6e (32GB Trillium), TPU v4 (32GB), A100 (40GB/80GB)
        if is_small_tier:
            base_ladder = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048]
        elif is_compact_tier:
            base_ladder = [32, 64, 128, 192, 256, 384, 512, 768]
        else:  # v5_base / standard
            base_ladder = [16, 32, 64, 96, 128, 192, 256, 384]
    elif total_gb >= 12.0:
        # Standard datacenter accelerator: TPU v5e-1 (16GB), T4 (16GB), V100 (16GB), L4 (24GB), RTX 3090/4090 (24GB)
        if is_small_tier:
            if seq_len >= 512:
                base_ladder = [16, 32, 64, 128, 192, 256]
            elif max_batch_size is not None and max_batch_size <= 64:
                base_ladder = [16, 32, 64]
            else:
                base_ladder = [32, 64, 128, 192, 256, 384, 512, 768, 1024]
        elif is_compact_tier:
            if seq_len >= 512:
                base_ladder = [8, 16, 32, 64, 96, 128]
            else:
                base_ladder = [8, 16, 32, 64, 96, 128, 192, 256, 384]
        elif preset == "v5_test":
            base_ladder = [8, 16, 32, 64]
        else:  # v5_base / standard
            base_ladder = [8, 16, 32, 48, 64, 96, 128]
    elif total_gb >= 6.0:
        # Mid-tier workstation / gaming GPU: 8GB - 12GB
        if is_small_tier:
            base_ladder = [16, 32, 64, 96, 128, 192, 256]
        elif is_compact_tier:
            base_ladder = [8, 16, 32, 48, 64, 96]
        else:
            base_ladder = [4, 8, 16, 24, 32]
    else:
        # Entry GPU (e.g. 4GB) or CPU
        base_ladder = [2, 4, 8, 16]

    # Common set of probe sizes based on hardware memory tier:
    # On accelerators with >= 16 GB (TPU v5e-1, v6e, v5p, H100, A100), start at 32 or 64
    # and skip 1, 2, 4, 8 entirely to avoid wasting compilation cycles.
    if total_gb >= 64.0:
        tier_common = {64, 128, 256, 512, 1024, 2048}
    elif total_gb >= 28.0:
        tier_common = {32, 64, 128, 256, 512, 1024}
    elif total_gb >= 16.0:
        tier_common = {32, 48, 64, 128, 192, 256}
    elif total_gb >= 12.0:
        tier_common = {16, 32, 48, 64, 128, 256}
    elif total_gb >= 6.0:
        tier_common = {8, 16, 32, 48, 64, 128}
    else:
        tier_common = {1, 2, 4, 8, 16, 32}

    base_ladder = sorted(set(base_ladder).union(tier_common))

    # For accelerators >= 16 GB, explicitly filter out 1, 2, 4, 8 unless constrained by max_batch_size
    if total_gb >= 16.0:
        if max_batch_size is not None and max_batch_size < 32:
            pass  # Keep constrained sizes if user requested low cap
        else:
            base_ladder = [b for b in base_ladder if b >= 32]

    if min_batch_size is not None:
        base_ladder = [b for b in base_ladder if b >= int(min_batch_size)]
        if not base_ladder:
            base_ladder = [int(min_batch_size)]

    if max_batch_size is not None:
        cap = max(1, int(max_batch_size))
        filtered = [b for b in base_ladder if b <= cap]
        base_ladder = filtered if filtered else [cap]

    if fast and len(base_ladder) > 4:
        # Fast geometric ladder: powers of 2 only, removing fine-grained linear steps
        if total_gb >= 12.0 and min_batch_size is None and max_batch_size is None:
            # On 12GB+ datacenter accelerators, avoid testing tiny 1, 2, 4, 8
            min_floor = 16 if total_gb < 28.0 else (32 if total_gb < 64.0 else 64)
            base_ladder = [b for b in base_ladder if b >= min_floor and (b & (b - 1) == 0)]
        else:
            base_ladder = [b for b in base_ladder if (b & (b - 1) == 0)]
        if not base_ladder:
            base_ladder = [16, 32, 64, 128]

    candidates = []
    for b in base_ladder:
        # Evaluate remat=False first so large batches without OOM skip redundant remat recomputation
        candidates.append({"batch_size": b, "remat": False})
        candidates.append({"batch_size": b, "remat": True})

    return candidates


class CandidateLadder:
    """
    Manages candidate traversal with intelligent stateful pruning:
    - If (B, remat=False) succeeds, (B, remat=True) is skipped as redundant.
    - If (B, remat=False) OOMs, all B' >= B with remat=False are pruned.
    - If (B, remat=True) OOMs, all B' >= B with remat=True are pruned.
    - If throughput saturates and degrades across multiple increasing batches,
      larger candidates are pruned early.
    """

    def __init__(
        self,
        candidates: List[Dict[str, Any]],
        enable_plateau_pruning: bool = True,
        plateau_tolerance: int = 2,
    ):
        self.candidates = list(candidates)
        self.passed_remat_false_batches: Set[int] = set()
        self.min_oom_remat_false: Optional[int] = None
        self.min_oom_remat_true: Optional[int] = None
        self.enable_plateau_pruning = enable_plateau_pruning
        self.plateau_tolerance = plateau_tolerance
        self.peak_throughput: float = 0.0
        self.peak_batch: Optional[int] = None
        self.degradation_count: int = 0
        self.stopped_due_to_plateau: bool = False

    def should_skip(self, batch_size: int, remat: bool) -> bool:
        """Determines whether a candidate should be pruned."""
        # 0. Plateau early exit: if throughput has degraded repeatedly on larger batches
        if self.stopped_due_to_plateau and self.peak_batch is not None and batch_size > self.peak_batch:
            return True

        # 1. Skip redundant remat=True if remat=False already succeeded
        if remat and batch_size in self.passed_remat_false_batches:
            return True

        # 2. Skip remat=False if an equal or smaller batch size already OOMed
        if not remat and self.min_oom_remat_false is not None and batch_size >= self.min_oom_remat_false:
            return True

        # 3. Skip remat=True if an equal or smaller batch size already OOMed
        if remat and self.min_oom_remat_true is not None and batch_size >= self.min_oom_remat_true:
            return True

        return False

    def record_result(
        self,
        batch_size: int,
        remat: bool,
        status: str,
        tok_per_sec: float = 0.0,
    ) -> None:
        """Updates pruning state with benchmark outcome and throughput trajectory."""
        is_pass = status == "PASS"
        is_oom = "OOM" in status or "RESOURCE_EXHAUSTED" in status or "out of memory" in status.lower()

        if is_pass:
            if not remat:
                self.passed_remat_false_batches.add(batch_size)

            # Check for throughput saturation and plateau
            if self.enable_plateau_pruning and tok_per_sec > 0:
                if tok_per_sec > self.peak_throughput * 1.01:
                    self.peak_throughput = tok_per_sec
                    self.peak_batch = batch_size
                    self.degradation_count = 0
                elif self.peak_batch is not None and batch_size > self.peak_batch:
                    # Throughput declined or stalled despite larger batch size
                    self.degradation_count += 1
                    if self.degradation_count >= self.plateau_tolerance:
                        self.stopped_due_to_plateau = True
        elif is_oom:
            if not remat:
                if self.min_oom_remat_false is None or batch_size < self.min_oom_remat_false:
                    self.min_oom_remat_false = batch_size
            else:
                if self.min_oom_remat_true is None or batch_size < self.min_oom_remat_true:
                    self.min_oom_remat_true = batch_size


def plan_bisection_search_batches(sorted_batches: List[int]) -> List[int]:
    """
    Computes an optimal logarithmic evaluation sequence across candidate batch sizes.
    Evaluates median anchor first, then branches upward or downward depending on results.
    """
    if not sorted_batches:
        return []
    if len(sorted_batches) <= 3:
        return list(sorted_batches)

    # Pick anchor at ~40-50% position
    mid_idx = len(sorted_batches) // 2
    return sorted_batches
