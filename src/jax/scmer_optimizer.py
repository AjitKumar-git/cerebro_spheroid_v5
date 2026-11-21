"""
SCMEROptimizer: Selective Cosine-Margin Embedding Regularization Optimizer for CerebroSpheroid V5.

Integrates:
1. Polar Newton-Schulz for 2D/3D manifold matrix operators (Muon-style zeroth power iteration).
2. Contractive persistent spectral clamp for recurrent core matrices (sigma_max <= 1.0).
3. SCM (Selective Cosine-Margin) regularizer on vocabulary embeddings:
   - Uniform vocabulary subsampling B = 512
   - Deterministic unordered pair bijection K = 8192 (no self-pairs, no duplicate pairs)
   - Smooth quadratic L2 normalization everywhere on R^D
   - Stop-gradient global reference centroid
   - Sparse (B, D) fused scatter-add avoiding dense (V, D) memory allocation
4. Decoupled float32 master-weight AdamW for 1D parameters.
"""

from typing import Any, Callable, Dict, Optional, Set, Tuple
import jax
import jax.numpy as jnp

from src.jax.scmer import compute_scmer_step
from src.jax.spheroid_optimizer import (
    VOCAB_KEYS,
    RECURRENT_KEYS,
    NORM_KEYS,
    is_manifold_operator,
    zeropower_via_newtonschulz5_jax,
    clamp_spectral_norm_jax,
    init_spheroid_opt_state,
    merge_spheroid_replicas,
    replicate_for_pmap,
    _apply_spheroid_grads,
    _compute_loss_and_grads,
    _is_pure_teacher_forcing,
)
from src.jax.v5_model import ModelConfigV5


def init_scmer_opt_state(
    params: Dict[str, jnp.ndarray],
    seed: int = 42,
) -> Dict[str, Any]:
    """Initializes optimizer state with PRNG seed for resume-safe SCMER pair sampling."""
    opt_state = init_spheroid_opt_state(params)
    opt_state["seed"] = seed
    return opt_state


def build_scmer_train_engine(
    config: ModelConfigV5,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    pad_id: int = 0,
    weight_decay: float = 0.01,
    repetition_penalty: float = 1.25,
    remat: bool = True,
    deliberation_mode: str = "teacher_forcing",
    param_masks: Optional[Dict[str, jnp.ndarray]] = None,
    scmer_lambda: float = 0.05,
    scmer_subsample: int = 512,
    scmer_interval: int = 8,
    scmer_margin: float = 0.12,
    scmer_pair_samples: int = 8192,
) -> Tuple[Callable, Callable]:
    """
    Builds decoupled JIT-compiled gradient computation and SCMER optimizer application functions.
    Defaults to active SCM regularization (scmer_lambda=0.001).
    """
    from src.jax.spheroid_optimizer import build_spheroid_train_engine

    return build_spheroid_train_engine(
        config=config,
        buffers=buffers,
        pad_id=pad_id,
        weight_decay=weight_decay,
        repetition_penalty=repetition_penalty,
        remat=remat,
        deliberation_mode=deliberation_mode,
        param_masks=param_masks,
        scmer_lambda=scmer_lambda,
        scmer_subsample=scmer_subsample,
        scmer_interval=scmer_interval,
        scmer_margin=scmer_margin,
        scmer_pair_samples=scmer_pair_samples,
    )


def build_train_step_scmer(
    config: ModelConfigV5,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    pad_id: int = 0,
    weight_decay: float = 0.01,
    repetition_penalty: float = 1.25,
    remat: bool = True,
    deliberation_mode: str = "teacher_forcing",
    param_masks: Optional[Dict[str, jnp.ndarray]] = None,
    scmer_lambda: float = 0.05,
    scmer_subsample: int = 512,
    scmer_interval: int = 8,
    scmer_margin: float = 0.12,
    scmer_pair_samples: int = 8192,
) -> Callable:
    """
    Builds a single fused JIT-compiled train_step combining forward, backward, and SCMEROptimizer.
    Defaults to active SCM regularization (scmer_lambda=0.001).
    """
    from src.jax.spheroid_optimizer import build_train_step_spheroid

    return build_train_step_spheroid(
        config=config,
        buffers=buffers,
        pad_id=pad_id,
        weight_decay=weight_decay,
        repetition_penalty=repetition_penalty,
        remat=remat,
        deliberation_mode=deliberation_mode,
        param_masks=param_masks,
        scmer_lambda=scmer_lambda,
        scmer_subsample=scmer_subsample,
        scmer_interval=scmer_interval,
        scmer_margin=scmer_margin,
        scmer_pair_samples=scmer_pair_samples,
    )


def build_pmap_local_sgd_step_scmer(
    config: ModelConfigV5,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    pad_id: int = 0,
    weight_decay: float = 0.01,
    repetition_penalty: float = 1.25,
    remat: bool = True,
    local_steps: int = 4,
    devices: Optional[Any] = None,
    scmer_lambda: float = 0.05,
    scmer_subsample: int = 512,
    scmer_interval: int = 8,
    scmer_margin: float = 0.12,
    scmer_pair_samples: int = 8192,
) -> Callable:
    """Multi-TPU pmap Local-SGD runner with SCMER optimizer."""
    from src.jax.spheroid_optimizer import build_pmap_local_sgd_step_spheroid

    return build_pmap_local_sgd_step_spheroid(
        config=config,
        buffers=buffers,
        pad_id=pad_id,
        weight_decay=weight_decay,
        repetition_penalty=repetition_penalty,
        remat=remat,
        local_steps=local_steps,
        devices=devices,
        scmer_lambda=scmer_lambda,
        scmer_subsample=scmer_subsample,
        scmer_interval=scmer_interval,
        scmer_margin=scmer_margin,
        scmer_pair_samples=scmer_pair_samples,
    )
