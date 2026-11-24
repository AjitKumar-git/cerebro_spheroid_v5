"""
🧠 CerebroSpheroid V5: Progressive Cortical Expansion Architecture
===================================================================
Manages progressive concentric cortical expansion from pre-trained 100M foundation
weights into the 32-shell Compact model (166.3M parameters).

Directive #7 Invariant (NO LOCKED SHELLS):
  In compliance with AGENTS.md Directive #7, all concentric shells (from Shell 0 to
  Shell K-1), shortcut tracts, radial gains, and sensory/motor projections remain 100%
  unlocked, active, and receive non-zero gradients across all training iterations.
  All mask generators and anneal schedules default to 1.0 (fully active and unfrozen).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


def build_cortical_expansion_masks(
    params: Dict[str, Any],
    num_frozen_shells: int = 0,
    num_frozen_tracts: int = 0,
    framework: str = "jax",
    anneal_factor: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Constructs gradient masks for progressive cortical expansion.
    In compliance with Directive #7, defaults to 1.0 (100% unlocked shells across all layers).
    
    Args:
        params: Dictionary of model parameters (JAX arrays, PyTorch tensors, or NumPy arrays).
        num_frozen_shells: Number of innermost concentric shells to freeze (default: 0 for 100% unlocked).
        num_frozen_tracts: Number of primary shortcut tracts to freeze (default: 0 for 100% unlocked).
        framework: 'jax', 'torch', or 'numpy' for the output mask arrays.
        anneal_factor: Gradient multiplier for core [default: 1.0 for unlocked default, 0.0 if shells explicitly frozen].
    
    Returns:
        Dictionary of mask arrays matching the keys and shapes of `params`.
    """
    masks: Dict[str, Any] = {}
    if anneal_factor is None:
        anneal = 1.0 if (num_frozen_shells == 0 and num_frozen_tracts == 0) else 0.0
    else:
        anneal = float(np.clip(anneal_factor, 0.0, 1.0))

    for k, v in params.items():
        shape = tuple(v.shape)
        # Default: initialize with anneal multiplier
        mask_np = np.full(shape, anneal, dtype=np.float32)

        # Core concentric shells and shortcut tracts: outer mantle is 100% trainable (1.0)
        if k in ("core.intra_swiglu.w_gate", "core.intra_swiglu.w_val", "core.intra_swiglu.w_down"):
            # Shape: (num_shells, surface_dim, hidden_dim)
            if len(shape) >= 1 and shape[0] > num_frozen_shells:
                mask_np[num_frozen_shells:] = 1.0
        elif k == "core.intra_norms.weight":
            # Shape: (num_shells, surface_dim)
            if len(shape) >= 1 and shape[0] > num_frozen_shells:
                mask_np[num_frozen_shells:] = 1.0
        elif k == "core.shell_gains":
            # Shape: (num_shells,)
            if len(shape) >= 1 and shape[0] > num_frozen_shells:
                mask_np[num_frozen_shells:] = 1.0
        elif k == "core.shortcut_weight":
            # Shape: (num_tracts, surface_dim, surface_dim)
            if len(shape) >= 1 and shape[0] > num_frozen_tracts:
                mask_np[num_frozen_tracts:] = 1.0
        elif k == "core.shortcut_gate":
            # Shape: (num_tracts,)
            if len(shape) >= 1 and shape[0] > num_frozen_tracts:
                mask_np[num_frozen_tracts:] = 1.0
        elif k in ("core.w_radial_in", "core.w_radial_out"):
            # Shape: (num_shells - 1, radial_dim, radial_dim)
            frozen_trans = max(0, num_frozen_shells - 1)
            if len(shape) >= 1 and shape[0] > frozen_trans:
                mask_np[frozen_trans:] = 1.0
        elif k == "motor.shell_score":
            # Shape: (num_shells,) - outer shell readout scores MUST be trainable
            if len(shape) >= 1 and shape[0] > num_frozen_shells:
                mask_np[num_frozen_shells:] = 1.0
        elif k.startswith("sensory.") or k.startswith("motor.") or k.startswith("perceptual.") or k.startswith("audio_stem."):
            mask_np[...] = anneal
        elif k in ("pos_embed", "phase_embed", "affine_scale"):
            mask_np[...] = anneal
        else:
            mask_np[...] = anneal

        if framework == "jax":
            import jax.numpy as jnp
            masks[k] = jnp.asarray(mask_np, dtype=jnp.float32)
        elif framework == "torch":
            import torch
            masks[k] = torch.from_numpy(mask_np)
        else:
            masks[k] = mask_np

    return masks


def compute_cortical_anneal_factor(
    step: int,
    warmup_steps: int = 2000,
    anneal_steps: int = 1000,
    cycle_length: Optional[int] = None,
    enforce_unlocked: bool = True,
) -> float:
    """
    Computes the continuous unfreezing anneal factor [0.0, 1.0].
    Per Directive #7, enforce_unlocked=True guarantees 1.0 (fully active core).
    """
    if enforce_unlocked:
        return 1.0

    if cycle_length is not None and cycle_length > 0:
        cycle_step = step % cycle_length
        freeze_window = cycle_length - anneal_steps
        if cycle_step < freeze_window:
            return 0.0
        return float((cycle_step - freeze_window) / max(1, anneal_steps))

    if step < warmup_steps:
        return 0.0
    if step >= warmup_steps + anneal_steps:
        return 1.0
    return float((step - warmup_steps) / max(1, anneal_steps))


def count_cortical_expansion_params(
    params: Dict[str, Any],
    masks: Dict[str, Any],
) -> Dict[str, Union[int, float]]:
    """Computes exact parameter counts and percentages for frozen vs trainable subsets."""
    total_params = 0
    frozen_params = 0
    trainable_params = 0

    for k, v in params.items():
        size = int(np.prod(v.shape)) if len(v.shape) > 0 else 1
        total_params += size
        if k in masks:
            m = np.asarray(masks[k])
            trainable_count = int(np.sum(m > 0.0))
            frozen_count = size - trainable_count
            trainable_params += trainable_count
            frozen_params += frozen_count
        else:
            frozen_params += size

    frozen_pct = (frozen_params / max(1, total_params)) * 100.0
    trainable_pct = (trainable_params / max(1, total_params)) * 100.0

    return {
        "total_params": total_params,
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
        "frozen_pct": frozen_pct,
        "trainable_pct": trainable_pct,
    }


def generate_expanded_tract_wiring(
    num_shells: int = 32,
    num_tracts: int = 96,
    base_src: Optional[Any] = None,
    base_dst: Optional[Any] = None,
    num_frozen_shells: int = 16,
    framework: str = "jax",
) -> Tuple[Any, Any]:
    """
    Constructs small-world shortcut tract wiring preserving exact pre-trained base tract endpoints.
    
    Tract Partitioning:
      1. Legacy Base Tracts (0..len(base_src)-1): 100% bit-for-bit preserved from base checkpoint.
      2. Intra-Mantle Tracts: Interconnect newly expanded outer laminae (num_frozen_shells..num_shells-1).
      3. Core-to-Mantle Ascending Projections: Connect deep core laminae to outer mantle laminae.
      4. Mantle-to-Core Descending Projections: Connect outer mantle laminae back to core laminae.
    """
    srcs: list[int] = []
    dsts: list[int] = []

    # 1. Inherit pre-trained tracts
    if base_src is not None and base_dst is not None:
        b_s = [int(x) for x in np.asarray(base_src)]
        b_d = [int(x) for x in np.asarray(base_dst)]
        srcs.extend(b_s)
        dsts.extend(b_d)

    n_legacy = len(srcs)
    remaining_tracts = max(0, num_tracts - n_legacy)
    m_start = num_frozen_shells
    m_count = max(1, num_shells - m_start)

    for i in range(remaining_tracts):
        tract_id = n_legacy + i
        mode = i % 3
        if mode == 0:
            # Intra-mantle shortcuts (outer laminae interconnectivity)
            s = m_start + (i % m_count)
            d = m_start + ((i * 7 + 1) % m_count)
            if s == d:
                d = m_start + ((s - m_start + 1) % m_count)
        elif mode == 1:
            # Core-to-mantle ascending projection (thalamocortical drive)
            s = (i * 3) % num_frozen_shells
            d = m_start + ((i * 5) % m_count)
        else:
            # Mantle-to-core descending projection (corticothalamic feedback)
            s = m_start + ((i * 2) % m_count)
            d = (i * 7) % num_frozen_shells
        srcs.append(int(s))
        dsts.append(int(d))

    # Slice to exact requested num_tracts
    srcs = srcs[:num_tracts]
    dsts = dsts[:num_tracts]

    if framework == "jax":
        import jax.numpy as jnp
        return jnp.array(srcs, dtype=jnp.int32), jnp.array(dsts, dtype=jnp.int32)
    elif framework == "torch":
        import torch
        return torch.tensor(srcs, dtype=torch.int32), torch.tensor(dsts, dtype=torch.int32)
    return np.array(srcs, dtype=np.int32), np.array(dsts, dtype=np.int32)


def apply_net2net_cortical_init(
    params: Dict[str, Any],
    num_frozen_shells: int = 16,
    num_frozen_tracts: int = 40,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Initializes newly expanded cortical parameter slices for progressive cortical expansion.
    
    Guarantees active representation learning and gradient receptivity in the outer mantle:
      1. motor.shell_score[num_frozen_shells:]: Set to mean of core scores (or 0.0), ensuring
         non-zero softmax attention mass and full gradient backpropagation from Step 0.
      2. core.shortcut_gate[num_frozen_tracts:]: Set to receptive conductance (~0.0, sigma(0)=0.5,
         sigma'(0)=0.25) so newly expanded axonal shortcut tracts actively conduct flux.
      3. core.shortcut_weight[num_frozen_tracts:]: Initialized with active small Gaussian weights
         scaled to match active tracts, preventing gradient dead-zones.
      4. core.intra_swiglu.w_down[num_frozen_shells:]: Initialized with small Gaussian weights
         matching active shells so error gradients pass into SwiGLU gate/val weights.
      5. core.w_radial_in / out[num_frozen_shells - 1:]: Initialized with active radial wave
         coupling so signals traverse smoothly across the core-mantle interface.
    """
    is_jax = any("jax" in str(type(v)) for v in params.values())
    is_torch = any("torch" in str(type(v)) for v in params.values())

    rng = np.random.RandomState(seed)

    for k, v in params.items():
        arr = np.array(v)
        modified = False

        if k == "core.intra_swiglu.w_down" and arr.shape[0] > num_frozen_shells:
            rms_outer = float(np.sqrt(np.mean(arr[num_frozen_shells:] ** 2)))
            if rms_outer < 1e-5:
                scale = float(np.std(arr[:num_frozen_shells])) if num_frozen_shells > 0 else 0.02
                scale = max(scale, 0.005)
                arr[num_frozen_shells:] = rng.normal(0.0, scale * 0.5, size=arr[num_frozen_shells:].shape)
                modified = True
        elif k == "core.shortcut_weight" and arr.shape[0] > num_frozen_tracts:
            rms_outer = float(np.sqrt(np.mean(arr[num_frozen_tracts:] ** 2)))
            if rms_outer < 1e-5:
                scale = float(np.std(arr[:num_frozen_tracts])) if num_frozen_tracts > 0 else 0.01
                scale = max(scale, 0.005)
                arr[num_frozen_tracts:] = rng.normal(0.0, scale * 0.5, size=arr[num_frozen_tracts:].shape)
                modified = True
        elif k == "core.shortcut_gate" and arr.shape[0] > num_frozen_tracts:
            outer_mean = float(np.mean(arr[num_frozen_tracts:]))
            if outer_mean < -3.0 or outer_mean > 3.0:
                core_gate = float(np.mean(arr[:num_frozen_tracts])) if num_frozen_tracts > 0 else 0.0
                arr[num_frozen_tracts:] = float(np.clip(core_gate, -0.5, 0.5))
                modified = True
        elif k in ("core.w_radial_in", "core.w_radial_out") and arr.shape[0] >= num_frozen_shells:
            trans = max(0, num_frozen_shells - 1)
            rms_outer = float(np.sqrt(np.mean(arr[trans:] ** 2)))
            if rms_outer < 1e-5:
                scale = float(np.std(arr[:trans])) if trans > 0 else 0.02
                scale = max(scale, 0.005)
                arr[trans:] = rng.normal(0.0, scale * 0.5, size=arr[trans:].shape)
                modified = True
        elif k == "motor.shell_score" and arr.shape[0] > num_frozen_shells:
            outer_min = float(np.min(arr[num_frozen_shells:]))
            if outer_min < -100.0:
                core_score = float(np.mean(arr[:num_frozen_shells])) if num_frozen_shells > 0 else 0.0
                arr[num_frozen_shells:] = core_score
                modified = True

        if modified:
            if is_jax:
                import jax.numpy as jnp
                params[k] = jnp.asarray(arr, dtype=v.dtype)
            elif is_torch:
                import torch
                params[k] = torch.from_numpy(arr).to(v.device, v.dtype)
            else:
                params[k] = arr

    return params

