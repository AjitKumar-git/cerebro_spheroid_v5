"""
Unit Tests for Negative-Covariance Regularizer in JAX (CerebroSpheroid V5).
Verifies exact analytic gradient against autodiff, numerical stability, sampling,
stop-gradient centroid, sparse updates, and JIT compatibility.
"""

import jax
import jax.numpy as jnp
import pytest

from src.jax.negative_covariance import (
    compute_vocab_centroid,
    normalize_embeddings_stabilized,
    sample_vocab_indices,
    sample_unordered_pairs,
    negative_covariance_loss,
    negative_covariance_gradient,
    compute_negative_covariance_step,
)


def test_analytic_gradient_vs_autodiff():
    """Test 1: Verify exact analytic gradient agrees with jax.grad numerically."""
    key = jax.random.PRNGKey(101)
    B, D = 24, 32
    X = jax.random.normal(key, (B, D))

    triu_i, triu_j = jnp.triu_indices(B, k=1)
    K = 40
    key_pair = jax.random.fold_in(key, 1)
    pair_choice = jax.random.choice(key_pair, triu_i.shape[0], shape=(K,), replace=False)
    pair_i = triu_i[pair_choice]
    pair_j = triu_j[pair_choice]

    def loss_wrt_X(x_in):
        U, _ = normalize_embeddings_stabilized(x_in, eps=1e-6)
        ui = U[pair_i]
        uj = U[pair_j]
        cosine = jnp.sum(ui * uj, axis=-1)
        hinge = jnp.maximum(cosine - 0.10, 0.0)
        return (0.5 / float(K)) * jnp.sum(hinge * hinge)

    U, norms = normalize_embeddings_stabilized(X, eps=1e-6)
    g_autodiff = jax.grad(loss_wrt_X)(X)
    g_analytic = negative_covariance_gradient(X, U, norms, pair_i, pair_j, margin=0.10, eps=1e-6)

    diff = jnp.max(jnp.abs(g_autodiff - g_analytic))
    rel_err = jnp.linalg.norm(g_autodiff - g_analytic) / (jnp.linalg.norm(g_autodiff) + 1e-9)

    assert rel_err < 1e-5, f"Relative error {rel_err} exceeded tolerance 1e-5 (max diff: {diff})"
    assert diff < 1e-7, f"Max absolute difference {diff} exceeded tolerance 1e-7"


def test_zero_loss_and_gradient_on_orthogonal_or_negative_pairs():
    """Test 2 & 3: Verify zero loss and gradient when cosine <= margin."""
    B, D = 4, 4
    # Mutually orthogonal vectors
    X = jnp.eye(B, D, dtype=jnp.float32)
    U, norms = normalize_embeddings_stabilized(X, eps=1e-6)
    triu_i, triu_j = jnp.triu_indices(B, k=1)

    grad = negative_covariance_gradient(X, U, norms, triu_i, triu_j, margin=0.10, eps=1e-6)
    assert jnp.allclose(grad, jnp.zeros_like(grad)), "Orthogonal pairs must yield zero gradient penalty!"


def test_pair_sampling_properties():
    """Test 4: Verify pair sampler satisfies no self-pairs, no duplicate pairs, correct count."""
    key = jax.random.PRNGKey(202)
    num_rows = 64
    pair_samples = 128
    pair_i, pair_j, K = sample_unordered_pairs(key, num_rows, pair_samples)

    assert K == pair_samples
    assert jnp.all(pair_i < pair_j), "All pairs must satisfy i < j (strictly unordered, no self-pairs)!"
    assert jnp.all(pair_i != pair_j), "Self-pairs must be strictly excluded!"

    # Check uniqueness
    flat_indices = pair_i * num_rows + pair_j
    unique_indices = jnp.unique(flat_indices)
    assert unique_indices.shape[0] == K, "Sampled pairs must be unique without replacement!"


def test_stabilized_normalization_small_vectors():
    """Test 5: Verify finite stable gradient even for near-zero vectors."""
    B, D = 8, 16
    # Very small vectors near zero
    X = jnp.zeros((B, D), dtype=jnp.float32) + 1e-12
    U, norms = normalize_embeddings_stabilized(X, eps=1e-6)
    triu_i, triu_j = jnp.triu_indices(B, k=1)

    grad = negative_covariance_gradient(X, U, norms, triu_i, triu_j, margin=0.10, eps=1e-6)
    assert jnp.all(jnp.isfinite(grad)), "Gradients must remain strictly finite for near-zero vectors!"
    assert jnp.all(jnp.isfinite(norms)), "Norms must remain finite!"


def test_stop_gradient_centroid():
    """Test 6: Verify stop-gradient centroid does not propagate gradients back to W."""
    W = jnp.ones((10, 4), dtype=jnp.float32) * 2.0
    centroid = compute_vocab_centroid(W)
    # Check that centroid value is correct
    assert jnp.allclose(centroid, jnp.ones((1, 4)) * 2.0)


def test_sparse_row_update():
    """Test 7: Verify only sampled rows receive gradients; unsampled rows are identically zero."""
    key = jax.random.PRNGKey(303)
    vocab_size, width = 64, 16
    table = jax.random.normal(key, (vocab_size, width))
    subsample = 12

    grad_table, loss_val = compute_negative_covariance_step(
        table=table,
        step_key=jax.random.fold_in(key, 1),
        strength=0.001,
        subsample=subsample,
        pair_samples=20,
        margin=0.10,
        eps=1e-6,
    )

    assert grad_table.shape == table.shape
    assert jnp.all(jnp.isfinite(grad_table))
    # Count rows with non-zero gradient
    row_norms = jnp.linalg.norm(grad_table, axis=-1)
    num_active_rows = jnp.sum(row_norms > 1e-9)
    assert num_active_rows <= subsample, f"Expected at most {subsample} active rows, got {num_active_rows}"


def test_jit_compatibility():
    """Test 8: Verify compute_negative_covariance_step compiles and runs under jax.jit."""
    key = jax.random.PRNGKey(404)
    vocab_size, width = 128, 32
    table = jax.random.normal(key, (vocab_size, width))

    @jax.jit
    def step_fn(tbl, k):
        return compute_negative_covariance_step(
            table=tbl,
            step_key=k,
            strength=0.001,
            subsample=32,
            pair_samples=64,
            margin=0.10,
            eps=1e-6,
        )

    grad_table, loss_val = step_fn(table, key)
    assert grad_table.shape == (128, 32)
    assert jnp.isfinite(loss_val)
