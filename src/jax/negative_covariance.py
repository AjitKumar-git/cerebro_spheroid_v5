"""
Backward-compatibility alias module.
Redirects to Selective Cosine-Margin Embedding Regularization (SCMER) in `src.jax.scmer`.
"""

from src.jax.scmer import (
    compute_vocab_centroid,
    normalize_embeddings_stabilized,
    sample_vocab_indices,
    pair_id_to_row_pair,
    sample_unordered_pairs,
    scmer_loss as negative_covariance_loss,
    scmer_gradient as negative_covariance_gradient,
    compute_scmer_step,
)
import jax.numpy as jnp
from typing import Tuple
import jax


def compute_negative_covariance_step(
    table: jnp.ndarray,
    step_key: jax.Array,
    strength: float = 0.001,
    subsample: int = 512,
    pair_samples: int = 8192,
    margin: float = 0.10,
    eps: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compatibility wrapper that scatters sparse gradient to (V, D) table."""
    token_ids, grad_rows, loss_val = compute_scmer_step(
        table=table,
        step_key=step_key,
        strength=strength,
        subsample=subsample,
        pair_samples=pair_samples,
        margin=margin,
        eps=eps,
    )
    grad_table = jnp.zeros_like(table, dtype=jnp.float32).at[token_ids].add(grad_rows)
    return grad_table, loss_val
