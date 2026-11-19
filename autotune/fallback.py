"""
OOM Detection and Dynamic Batch Size Fallback Engine for Training.
==================================================================

Provides runtime protection against TPU/GPU Out-Of-Memory (OOM) errors during
training and compilation. When an OOM occurs, it dynamically reduces the micro-batch
size (e.g. halving it), adjusts gradient accumulation to keep the effective batch
size constant, and slices existing batches so zero training tokens are lost.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


OOM_ERROR_MARKERS = (
    "resource_exhausted",
    "resourceexhausted",
    "resourceexhaustederror",
    "out of memory",
    "outofmemory",
    "hbmoom",
    "failed to allocate",
    "allocation_failed",
    "cuda out of memory",
    "std::bad_alloc",
    "outofmemoryerror",
)


def is_oom_error(exc: BaseException) -> bool:
    """
    Deterministically detects whether an exception was caused by TPU HBM,
    GPU VRAM, or host RAM exhaustion across JAX, PyTorch, CUDA, and XLA.
    """
    if exc is None:
        return False

    err_str = str(exc).lower()
    err_type = type(exc).__name__.lower()

    if any(marker in err_type for marker in ("resourceexhausted", "outofmemory", "bad_alloc")):
        return True

    return any(marker in err_str for marker in OOM_ERROR_MARKERS)


def compute_fallback_batch_size(
    current_batch_size: int,
    min_batch_size: int = 1,
    factor: float = 0.5,
) -> int:
    """
    Computes a reduced micro-batch size when an OOM occurs.
    Guarantees strictly decreasing batch size down to min_batch_size.
    """
    if current_batch_size <= min_batch_size:
        raise RuntimeError(
            f"Cannot fallback: current batch size ({current_batch_size}) is already at or below "
            f"minimum allowable batch size ({min_batch_size})."
        )

    new_bs = int(math.floor(current_batch_size * factor))

    # Keep power-of-two alignment if previous was an even multiple
    if current_batch_size >= 4 and current_batch_size % 2 == 0 and new_bs % 2 != 0:
        new_bs = max(min_batch_size, (new_bs // 2) * 2)

    new_bs = max(min_batch_size, new_bs)
    if new_bs >= current_batch_size:
        new_bs = max(min_batch_size, current_batch_size - 1)

    return new_bs


def adjust_gradient_accumulation(
    effective_batch_size: int,
    new_micro_batch_size: int,
    current_grad_accum: int = 1,
) -> int:
    """
    Calculates the adjusted gradient accumulation steps needed to preserve
    the global effective batch size when the micro-batch size is reduced.
    """
    if effective_batch_size <= 0:
        return max(1, current_grad_accum * 2)

    new_grad_accum = max(1, effective_batch_size // max(1, new_micro_batch_size))
    return new_grad_accum


class MicroBatchSlicer:
    """
    Slices batches from any data streamer into exact micro-batch chunks.
    Preserves remainder slices across iterations so zero tokens are lost
    when transitioning between batch sizes.
    """

    def __init__(self, next_batch_fn: Optional[Callable[[], Tuple[Any, ...]]] = None):
        self.next_batch_fn = next_batch_fn
        self.remainder_prompt: Optional[Any] = None
        self.remainder_target: Optional[Any] = None
        self.remainder_feedback: Optional[Any] = None
        self.remainder_weights: Optional[Any] = None

    def stash_remainder(
        self,
        prompt: Any,
        target: Any,
        feedback: Optional[Any] = None,
        weights: Optional[Any] = None,
    ) -> None:
        """Stashes an uncompleted or failed batch to be sliced in subsequent steps."""
        self.remainder_prompt = prompt
        self.remainder_target = target
        self.remainder_feedback = feedback
        self.remainder_weights = weights

    def has_remainder(self) -> bool:
        if self.remainder_prompt is None:
            return False
        shape = getattr(self.remainder_prompt, "shape", None)
        if shape is not None:
            return shape[0] > 0
        return len(self.remainder_prompt) > 0

    def get_micro_batch(
        self,
        micro_bs: int,
        next_batch_fn: Optional[Callable[[], Tuple[Any, ...]]] = None,
    ) -> Tuple[Any, ...]:
        """
        Retrieves a micro-batch of exact batch size `micro_bs`. Slices from existing
        remainder if available; otherwise fetches a new batch from `next_batch_fn`.
        Returns 3-tuple (p, t, fb) or 4-tuple (p, t, fb, w) depending on whether weights exist.
        """
        fetch_fn = next_batch_fn or self.next_batch_fn
        if fetch_fn is None and not self.has_remainder():
            raise ValueError("No next_batch_fn provided and no remainder batch stashed.")

        if self.has_remainder():
            p, t, fb, w = (
                self.remainder_prompt,
                self.remainder_target,
                self.remainder_feedback,
                self.remainder_weights,
            )
        else:
            raw = fetch_fn()
            if len(raw) == 4:
                p, t, fb, w = raw[0], raw[1], raw[2], raw[3]
            elif len(raw) == 3:
                p, t, fb, w = raw[0], raw[1], raw[2], None
            elif len(raw) == 2:
                p, t, fb, w = raw[0], raw[1], None, None
            else:
                p, t, fb, w = raw[0], raw[1], None, None

        batch_len = p.shape[0] if hasattr(p, "shape") else len(p)

        if batch_len <= micro_bs:
            # Batch fits completely in micro_bs
            self.remainder_prompt = None
            self.remainder_target = None
            self.remainder_feedback = None
            self.remainder_weights = None
            return (p, t, fb, w) if w is not None else (p, t, fb)

        # Slicing: take micro_bs items, stash remainder
        p_slice = p[:micro_bs]
        t_slice = t[:micro_bs]
        fb_slice = fb[:micro_bs] if fb is not None else None
        w_slice = w[:micro_bs] if w is not None else None

        self.remainder_prompt = p[micro_bs:]
        self.remainder_target = t[micro_bs:]
        self.remainder_feedback = fb[micro_bs:] if fb is not None else None
        self.remainder_weights = w[micro_bs:] if w is not None else None

        return (p_slice, t_slice, fb_slice, w_slice) if w is not None else (p_slice, t_slice, fb_slice)


class TrainingOOMHandler:
    """
    Stateful controller managing dynamic batch-size reduction, gradient accumulation
    adjustment, and activation rematerialization cascades during training when
    Out-Of-Memory (OOM) exceptions are triggered.
    """

    def __init__(
        self,
        initial_batch_size: int,
        effective_batch_size: Optional[int] = None,
        initial_grad_accum: int = 1,
        remat: bool = True,
        min_batch_size: int = 1,
        next_batch_fn: Optional[Callable[[], Tuple[Any, ...]]] = None,
        verbose: bool = True,
        auto_engage_remat: bool = True,
    ):
        self.batch_size = int(initial_batch_size)
        self.effective_batch_size = int(effective_batch_size) if effective_batch_size else (self.batch_size * initial_grad_accum)
        self.grad_accum = int(initial_grad_accum)
        self.remat = bool(remat)
        self.min_batch_size = int(min_batch_size)
        self.slicer = MicroBatchSlicer(next_batch_fn)
        self.verbose = verbose
        self.auto_engage_remat = auto_engage_remat
        self.fallback_count = 0
        self.history: List[Dict[str, Any]] = []

    def clean_memory(self) -> None:
        """Flushes JAX compiled dispatch caches and triggers Python garbage collection."""
        try:
            import jax
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    def handle_oom(
        self,
        exc: BaseException,
        step: Optional[int] = None,
        failed_batch: Optional[Tuple[Any, ...]] = None,
        current_remat: Optional[bool] = None,
    ) -> Tuple[int, int]:
        """
        Processes an OOM error through a multi-stage defense envelope:
        1. If remat is currently False and auto_engage_remat is True:
           Dynamically engages remat=True (reclaiming 50-70% activation memory while keeping batch size).
        2. If remat is already True:
           Halves micro-batch size and scales gradient accumulation to keep effective batch size invariant.
        3. Stashes failed batch in MicroBatchSlicer so zero tokens are lost.
        4. Cleans allocator slabs and cache memory.
        """
        if not is_oom_error(exc):
            raise exc

        old_bs = self.batch_size
        old_ga = self.grad_accum
        old_remat = current_remat if current_remat is not None else self.remat

        # Stage 1: Engage remat if it was inactive
        if (not old_remat) and self.auto_engage_remat:
            self.remat = True
            new_bs = old_bs
            new_ga = old_ga
            action = "ENGAGE_REMAT"
        else:
            # Stage 2: Halve batch size and double grad_accum
            new_bs = compute_fallback_batch_size(old_bs, min_batch_size=self.min_batch_size)
            new_ga = adjust_gradient_accumulation(self.effective_batch_size, new_bs, old_ga)
            self.batch_size = new_bs
            self.grad_accum = new_ga
            action = "HALVE_BATCH"

        self.fallback_count += 1

        if failed_batch is not None:
            p = failed_batch[0]
            t = failed_batch[1]
            fb = failed_batch[2] if len(failed_batch) > 2 else None
            w = failed_batch[3] if len(failed_batch) > 3 else None
            self.slicer.stash_remainder(p, t, fb, w)

        self.clean_memory()

        record = {
            "step": step,
            "action": action,
            "old_batch_size": old_bs,
            "new_batch_size": new_bs,
            "old_grad_accum": old_ga,
            "new_grad_accum": new_ga,
            "old_remat": old_remat,
            "new_remat": self.remat,
            "effective_batch_size": self.effective_batch_size,
        }
        self.history.append(record)

        if self.verbose:
            step_str = f" at Step {step:,}" if step is not None else ""
            print("\n" + "=" * 80)
            print(f"⚠️ [HBM/VRAM OOM Guard] Out-Of-Memory detected{step_str} (batch={old_bs}, remat={old_remat})!")
            if action == "ENGAGE_REMAT":
                print(f"   🛡️ Action: Engaged dynamic activation rematerialization (remat: False -> True)!")
                print(f"   ⚡ Batch size preserved at {new_bs} (grad_accum={new_ga}) while reclaiming ~60% activation HBM.")
            else:
                print(f"   📉 Action: Scaled micro-batch {old_bs} -> {new_bs} (remat={self.remat})")
                print(f"   ⚡ Gradient Accumulation Adjusted: {old_ga} -> {new_ga} (Preserving effective batch: {new_bs * new_ga})")
            print("=" * 80 + "\n")

        return new_bs, new_ga

    def update_streamer(self, streamer: Any) -> None:
        """Propagates the new batch size to the active data streamer or batcher."""
        if streamer is None:
            return
        if hasattr(streamer, "batch_size"):
            streamer.batch_size = self.batch_size
        if hasattr(streamer, "batcher") and getattr(streamer, "batcher", None) is not None:
            batcher = getattr(streamer, "batcher")
            if hasattr(batcher, "batch_size"):
                batcher.batch_size = self.batch_size
