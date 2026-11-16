"""
🧠 CerebroSpheroid V5: Progressive Cortical Expansion Architecture
===================================================================
Manages progressive concentric cortical expansion from pre-trained 100M foundation
weights into the 32-shell Compact model (166.3M parameters).

Biological Architecture:
  Concentric cortical expansion mirrors mammalian neocortical evolution:
  The mature, stable subcortical core (Shells 0–15, thalamic projections, sensory/motor
  headers) remains structurally frozen (101.96M params | 61.3%).
  All training capacity is concentrated into the outer cortical mantle
  (Shells 16–31, small-world shortcut tracts 40–95 | 64.37M params | 38.7%) to acquire
  multi-step deliberation, deductive reasoning, and complex cognitive synthesis without
  catastrophic forgetting of core linguistic stability.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


def build_cortical_expansion_masks(
    params: Dict[str, Any],
    num_frozen_shells: int = 16,
    num_frozen_tracts: int = 40,
    framework: str = "jax",
) -> Dict[str, Any]:
    """
    Constructs binary gradient masks (0.0 = frozen, 1.0 = trainable) for progressive cortical expansion.
    
    Args:
        params: Dictionary of model parameters (JAX arrays, PyTorch tensors, or NumPy arrays).
        num_frozen_shells: Number of innermost concentric shells to freeze (default: 16).
        num_frozen_tracts: Number of primary shortcut tracts to freeze (default: 40).
        framework: 'jax', 'torch', or 'numpy' for the output mask arrays.
    
    Returns:
        Dictionary of mask arrays matching the keys and shapes of `params`.
    """
    masks: Dict[str, Any] = {}

    for k, v in params.items():
        shape = tuple(v.shape)
        # Default: initialize all zeros (frozen)
        mask_np = np.zeros(shape, dtype=np.float32)

        # Core concentric shells and shortcut tracts
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
        elif k.startswith("sensory.") or k.startswith("motor.") or k.startswith("perceptual.") or k.startswith("audio_stem."):
            # All sensory encoders, positional embeddings, motor decoders, and heads remain 100% frozen
            mask_np[...] = 0.0
        elif k in ("pos_embed", "phase_embed", "affine_scale"):
            mask_np[...] = 0.0
        else:
            # Other invariant core projections remain frozen
            mask_np[...] = 0.0

        if framework == "jax":
            import jax.numpy as jnp
            masks[k] = jnp.asarray(mask_np, dtype=jnp.float32)
        elif framework == "torch":
            import torch
            masks[k] = torch.from_numpy(mask_np)
        else:
            masks[k] = mask_np

    return masks


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
) -> Dict[str, Any]:
    """
    Applies Net2Net Zero-Regression initialization to newly expanded parameter slices.
    
    Guarantees mathematical invariance at Step 0:
      1. core.intra_swiglu.w_down[num_frozen_shells:] = 0.0 (Zero non-linear distortion in outer mantle).
      2. core.shortcut_weight[num_frozen_tracts:] = 0.0 (Zero uninitialized shortcut flux).
      3. core.shortcut_gate[num_frozen_tracts:] = -10.0 (Sigmoid conductance ~4.5e-5).
      4. core.w_radial_in[num_frozen_shells - 1:] = 0.0 (Zero inward radial noise into core).
      5. core.w_radial_out[num_frozen_shells - 1:] = 0.0 (Zero outward shock from core).
      6. motor.shell_score[num_frozen_shells:] = -1e4 (Softmax probability mass ~0 on outer shells).
    """
    is_jax = any("jax" in str(type(v)) for v in params.values())
    is_torch = any("torch" in str(type(v)) for v in params.values())

    for k, v in params.items():
        arr = np.array(v)
        modified = False

        if k == "core.intra_swiglu.w_down" and arr.shape[0] > num_frozen_shells:
            arr[num_frozen_shells:] = 0.0
            modified = True
        elif k == "core.shortcut_weight" and arr.shape[0] > num_frozen_tracts:
            arr[num_frozen_tracts:] = 0.0
            modified = True
        elif k == "core.shortcut_gate" and arr.shape[0] > num_frozen_tracts:
            arr[num_frozen_tracts:] = -10.0
            modified = True
        elif k in ("core.w_radial_in", "core.w_radial_out") and arr.shape[0] >= num_frozen_shells:
            # Transitions >= num_frozen_shells - 1 cross or lie outside the core boundary
            arr[num_frozen_shells - 1:] = 0.0
            modified = True
        elif k == "motor.shell_score" and arr.shape[0] > num_frozen_shells:
            arr[num_frozen_shells:] = -1e4
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

