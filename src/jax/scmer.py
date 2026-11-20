"""
Selective Cosine-Margin Embedding Regularization (SCMER) for CerebroSpheroid V5.

Mathematical Formulation:
Operates on the token embedding table W in R^{V x D}.
Penalizes excessive pairwise positive cosine similarity between token embeddings above
a margin threshold m, using a stop-gradient global reference centroid and smooth stabilized
L2 normalization.

Loss:
    R_cos(W_S) = (1 / (2K)) * sum_{k=1}^K [max(0, cos(u_{i_k}, u_{j_k}) - m)]^2
where:
    mu_V = stop_gradient((1 / V) * sum_{i=1}^V W_i)
    x_i = W_{S_i} - mu_V
    u_i = x_i / sqrt(||x_i||_2^2 + eps^2)
    c_{i_k j_k} = u_{i_k}^T u_{j_k}

Gradient wrt x_i:
    g_{x_i} = g_{u_i} / s_i - x_i (x_i^T g_{u_i}) / s_i^3
where s_i = sqrt(||x_i||_2^2 + eps^2).

Pair Sampling:
    Deterministic bijection mapping unique pair IDs p in [0, binom(B, 2) - 1]
    to unordered index pairs (i, j) with 0 <= i < j < B.
    Guarantees no self-pairs, no duplicate pairs, and exact 1/K normalization.

Sparse Fused Allocation:
    Computes auxiliary gradient strictly as (B, D) alongside sampled token IDs (B,).
    Fused directly into task embedding gradient via indexed scatter-add, avoiding
    allocating an unnecessary second dense (V, D) buffer.
"""

from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp


def compute_vocab_centroid(W: jnp.ndarray) -> jnp.ndarray:
    """Computes the global vocabulary mean treated as a non-differentiable reference centroid."""
    return jax.lax.stop_gradient(jnp.mean(W.astype(jnp.float32), axis=0, keepdims=True))


def normalize_embeddings_stabilized(
    X: jnp.ndarray,
    eps: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Stabilized row-wise L2 normalization:
    u_i = x_i / sqrt(||x_i||^2 + eps^2)
    Provides a smooth, continuously differentiable transformation everywhere on R^D.
    """
    norms = jnp.sqrt(jnp.sum(X * X, axis=-1, keepdims=True) + eps**2)
    U = X / norms
    return U, norms


def sample_vocab_indices(
    key: jax.Array,
    vocab_size: int,
    sample_size: int,
) -> jnp.ndarray:
    """Samples B vocabulary row indices uniformly without replacement."""
    n = min(max(2, int(sample_size)), int(vocab_size))
    return jax.random.choice(key, vocab_size, shape=(n,), replace=False)


@partial(jax.jit, static_argnums=(1,))
def pair_id_to_row_pair(pair_ids: jnp.ndarray, num_rows: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Deterministic bijection mapping pair ID p in [0, binom(B, 2) - 1]
    to unique unordered row index pair (i, j) with 0 <= i < j < num_rows.

    Row-major upper-triangular ordering:
    (0, 1), (0, 2), ..., (0, B-1), (1, 2), ..., (B-2, B-1).

    Guarantees:
    - 0 <= i < j < num_rows
    - No self-pairs (i != j)
    - No duplicate pairs
    """
    B = num_rows
    b = 2 * B - 1
    b_f = jnp.asarray(b, dtype=jnp.float32)
    disc = jnp.maximum(0.0, b_f * b_f - 8.0 * pair_ids.astype(jnp.float32))
    i = jnp.floor((b_f - jnp.sqrt(disc)) / 2.0).astype(jnp.int32)

    # Numerical boundary protection for exact integer partition
    start_i = i * B - (i * (i + 1)) // 2
    i = jnp.where(start_i > pair_ids, i - 1, i)
    start_i = i * B - (i * (i + 1)) // 2
    next_start = (i + 1) * B - ((i + 1) * (i + 2)) // 2
    i = jnp.where((i < B - 2) & (next_start <= pair_ids), i + 1, i)
    start_i = i * B - (i * (i + 1)) // 2
    j = pair_ids - start_i + i + 1
    return i, j


def sample_unordered_pairs(
    key: jax.Array,
    num_rows: int,
    pair_samples: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Samples K unique pair IDs uniformly without replacement from {0, ..., binom(B, 2) - 1},
    then deterministically maps each pair ID to unique unordered row index pair (i, j).
    """
    total_pairs = (num_rows * (num_rows - 1)) // 2
    K = min(max(1, int(pair_samples)), total_pairs) if pair_samples > 0 else total_pairs
    if K == total_pairs:
        pair_ids = jnp.arange(total_pairs, dtype=jnp.int32)
    else:
        pair_ids = jax.random.choice(key, total_pairs, shape=(K,), replace=False)
    pair_i, pair_j = pair_id_to_row_pair(pair_ids, num_rows)
    return pair_i, pair_j, K


def scmer_loss(
    W: jnp.ndarray,
    sampled_indices: jnp.ndarray,
    pair_i: jnp.ndarray,
    pair_j: jnp.ndarray,
    margin: float = 0.12,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Evaluates the Selective Cosine-Margin Embedding Regularization loss:
    R_cos = (1 / 2K) * sum_{k=1}^K [max(0, cos(u_{i_k}, u_{j_k}) - m)]^2
    """
    vocab_mean = compute_vocab_centroid(W)
    X = W[sampled_indices].astype(jnp.float32) - vocab_mean
    U, _ = normalize_embeddings_stabilized(X, eps=eps)
    ui = U[pair_i]
    uj = U[pair_j]
    cosine = jnp.sum(ui * uj, axis=-1)
    hinge = jnp.maximum(cosine - margin, 0.0)
    K = float(pair_i.shape[0])
    return (0.5 / K) * jnp.sum(hinge * hinge)


def scmer_gradient(
    X: jnp.ndarray,
    U: jnp.ndarray,
    norms: jnp.ndarray,
    pair_i: jnp.ndarray,
    pair_j: jnp.ndarray,
    margin: float = 0.12,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Computes exact closed-form analytic gradient wrt centered rows X:
    g_{x_i} = g_{u_i} / s_i - x_i (x_i^T g_{u_i}) / s_i^3
    Matches jax.grad of scmer_loss bitwise.
    """
    K = float(pair_i.shape[0])
    ui = U[pair_i]
    uj = U[pair_j]
    cosine = jnp.sum(ui * uj, axis=-1)
    hinge = jnp.maximum(cosine - margin, 0.0)
    coeff = hinge / K

    # Tangent gradients wrt normalized vectors U
    gi = coeff[:, None] * (uj - cosine[:, None] * ui)
    gj = coeff[:, None] * (ui - cosine[:, None] * uj)

    grad_u = jnp.zeros_like(U)
    grad_u = grad_u.at[pair_i].add(gi)
    grad_u = grad_u.at[pair_j].add(gj)

    # Exact stabilized normalization derivative
    grad_x = (
        grad_u / norms
        - X * (jnp.sum(X * grad_u, axis=-1, keepdims=True) / (norms**3))
    )
    return grad_x


def compute_scmer_step(
    table: jnp.ndarray,
    step_key: jax.Array,
    strength: float = 0.05,
    subsample: int = 512,
    pair_samples: int = 8192,
    margin: float = 0.12,
    eps: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Executes a complete sampled SCMER regularization event:
    1. Samples B rows and K unordered pairs via deterministic pair bijection.
    2. Centers rows against global stop-gradient centroid.
    3. Computes exact analytic gradient and loss on (B, D).
    Returns (token_ids, grad_rows, loss_val).

    Does NOT allocate a second full (V, D) table. The returned (B, D) auxiliary
    gradient is fused directly into the task embedding gradient via scatter-add.
    """
    vocab_size, width = table.shape
    n = min(max(2, int(subsample)), int(vocab_size))
    key_rows, key_pairs = jax.random.split(step_key)

    # 1. Sample rows
    token_ids = sample_vocab_indices(key_rows, vocab_size, n)
    rows = table[token_ids].astype(jnp.float32)

    # 2. Stop-gradient global reference centroid
    vocab_mean = compute_vocab_centroid(table)
    X = rows - vocab_mean

    # 3. Stabilized L2 normalization
    U, norms = normalize_embeddings_stabilized(X, eps=eps)

    # 4. Sample unordered pairs via explicit mapping
    pair_i, pair_j, K = sample_unordered_pairs(key_pairs, n, pair_samples)

    # 5. Evaluate cosine and loss
    ui = U[pair_i]
    uj = U[pair_j]
    cosine = jnp.sum(ui * uj, axis=-1)
    hinge = jnp.maximum(cosine - margin, 0.0)
    loss_val = (0.5 * strength / float(K)) * jnp.sum(hinge * hinge)

    # 6. Analytic gradient on (B, D)
    grad_rows = strength * scmer_gradient(
        X=X, U=U, norms=norms, pair_i=pair_i, pair_j=pair_j, margin=margin, eps=eps
    )

    return token_ids, grad_rows, loss_val
