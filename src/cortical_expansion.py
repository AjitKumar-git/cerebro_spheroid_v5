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
from typing import Any, Dict, Tuple, Union

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
