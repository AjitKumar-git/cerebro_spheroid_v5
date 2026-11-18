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

    # Keep a common set of probe sizes in every hardware tier
    base_ladder = sorted(set(base_ladder).union({1, 2, 4, 8, 16, 32, 48, 64, 128, 256}))

    if max_batch_size is not None:
        cap = max(1, int(max_batch_size))
        filtered = [b for b in base_ladder if b <= cap]
        base_ladder = filtered if filtered else [cap]

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
    """

    def __init__(self, candidates: List[Dict[str, Any]]):
        self.candidates = list(candidates)
        self.passed_remat_false_batches: Set[int] = set()
        self.min_oom_remat_false: Optional[int] = None
        self.min_oom_remat_true: Optional[int] = None

    def should_skip(self, batch_size: int, remat: bool) -> bool:
        """Determines whether a candidate should be pruned."""
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

    def record_result(self, batch_size: int, remat: bool, status: str) -> None:
        """Updates pruning state with benchmark outcome."""
        is_pass = status == "PASS"
        is_oom = "OOM" in status or "RESOURCE_EXHAUSTED" in status or "out of memory" in status.lower()

        if is_pass:
            if not remat:
                self.passed_remat_false_batches.add(batch_size)
        elif is_oom:
            if not remat:
                if self.min_oom_remat_false is None or batch_size < self.min_oom_remat_false:
                    self.min_oom_remat_false = batch_size
            else:
                if self.min_oom_remat_true is None or batch_size < self.min_oom_remat_true:
                    self.min_oom_remat_true = batch_size
