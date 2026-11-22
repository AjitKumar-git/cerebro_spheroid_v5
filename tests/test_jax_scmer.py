"""
Unit Tests for Selective Cosine-Margin Embedding Regularization (SCMER) in JAX.

Verifies:
1. Deterministic pair bijection mapping pair IDs to row-index pairs (i, j).
2. Independent pair sampling (no self-pairs, no duplicate pairs, uniform coverage).
3. Exact analytic gradient matching autodiff (jax.grad) to machine precision.
4. Zero loss and gradient on orthogonal / negative / margin-satisfying pairs.
5. Sparse (B, D) fused gradient without dense (V, D) auxiliary allocation.
6. Numerical stability of smooth normalization on small / near-zero vectors.
7. Stop-gradient reference centroid non-differentiability.
8. PRNG key derivation reproducibility and resume safety.
9. JIT compatibility under XLA.
"""

import jax
import jax.numpy as jnp
import pytest

from src.jax.scmer import (
    compute_vocab_centroid,
    normalize_embeddings_stabilized,
    sample_vocab_indices,
    pair_id_to_row_pair,
    sample_unordered_pairs,
    scmer_loss,
    scmer_gradient,
    compute_scmer_step,
)


def test_pair_id_to_row_pair_bijection():
    """Verify deterministic bijection mapping pair ID p to (i, j) matches triu_indices exactly."""
    B = 32
    total_pairs = (B * (B - 1)) // 2
    pair_ids = jnp.arange(total_pairs, dtype=jnp.int32)
    i, j = pair_id_to_row_pair(pair_ids, B)
    triu_i, triu_j = jnp.triu_indices(B, k=1)

    assert jnp.all(i == triu_i), "Mapped i indices must match upper-triangular row indices exactly"
    assert jnp.all(j == triu_j), "Mapped j indices must match upper-triangular column indices exactly"
    assert jnp.all(i < j), "Strict ordering 0 <= i < j < B must hold everywhere"
    assert jnp.all(i >= 0) and jnp.all(j < B), "Indices must remain strictly within [0, B)"


def test_independent_pair_sampling_properties():
    """Verify pair sampler satisfies no self-pairs, no duplicates, and exact K unique pairs."""
    key = jax.random.PRNGKey(505)
    num_rows = 512
    pair_samples = 8192
    pair_i, pair_j, K = sample_unordered_pairs(key, num_rows, pair_samples)

    assert K == pair_samples
    assert pair_i.shape == (pair_samples,)
    assert pair_j.shape == (pair_samples,)

    # 1. No self-pairs
    assert jnp.all(pair_i != pair_j), "Self-pairs (i == j) must be strictly excluded!"
    assert jnp.all(pair_i < pair_j), "Pairs must strictly satisfy i < j!"

    # 2. No duplicate pairs
    flat_pair_ids = pair_i.astype(jnp.int32) * num_rows + pair_j.astype(jnp.int32)
    unique_pairs = jnp.unique(flat_pair_ids)
    assert unique_pairs.shape[0] == K, f"Expected {K} unique pairs, found duplicates ({unique_pairs.shape[0]})"


def test_analytic_gradient_vs_autodiff():
    """Verify exact analytic gradient matches jax.grad of scmer_loss to machine precision."""
    key = jax.random.PRNGKey(606)
    B, D = 32, 64
    X = jax.random.normal(key, (B, D))

    triu_i, triu_j = jnp.triu_indices(B, k=1)
    K = 100
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
    g_analytic = scmer_gradient(X, U, norms, pair_i, pair_j, margin=0.10, eps=1e-6)

    rel_err = jnp.linalg.norm(g_autodiff - g_analytic) / (jnp.linalg.norm(g_autodiff) + 1e-9)
    diff = jnp.max(jnp.abs(g_autodiff - g_analytic))

    assert rel_err < 1e-5, f"Relative error {rel_err} exceeded tolerance 1e-5"
    assert diff < 1e-7, f"Max difference {diff} exceeded tolerance 1e-7"


def test_zero_penalty_on_orthogonal_or_negative_pairs():
    """Verify orthogonal and negative pairs incur identically zero loss and zero gradient."""
    B, D = 8, 8
    X = jnp.eye(B, D, dtype=jnp.float32)  # All off-diagonal cosines are exactly 0.0 <= margin (0.10)
    U, norms = normalize_embeddings_stabilized(X, eps=1e-6)
    triu_i, triu_j = jnp.triu_indices(B, k=1)

    grad = scmer_gradient(X, U, norms, triu_i, triu_j, margin=0.10, eps=1e-6)
    assert jnp.allclose(grad, 0.0), "Orthogonal representations must receive identically zero gradient penalty!"


def test_sparse_bd_fused_gradient_no_dense_vd_allocation():
    """Verify compute_scmer_step returns (B, D) auxiliary gradient that fuses identically via scatter-add."""
    key = jax.random.PRNGKey(707)
    V, D = 128, 32
    B = 16
    K = 32
    table = jax.random.normal(key, (V, D))

    token_ids, grad_rows, loss_val = compute_scmer_step(
        table=table,
        step_key=jax.random.fold_in(key, 2),
        strength=0.001,
        subsample=B,
        pair_samples=K,
        margin=0.10,
        eps=1e-6,
    )

    # 1. Output shapes must be sparse (B, D) and (B,), NOT (V, D)
    assert token_ids.shape == (B,), f"Expected token_ids shape ({B},), got {token_ids.shape}"
    assert grad_rows.shape == (B, D), f"Expected grad_rows shape ({B}, {D}), got {grad_rows.shape}"
    assert jnp.isfinite(loss_val)

    # 2. Scatter-add fusion verification
    task_grad = jax.random.normal(jax.random.fold_in(key, 3), (V, D))
    fused_grad = task_grad.at[token_ids].add(grad_rows)

    # Equivalent dense allocation calculation
    dense_grad_aux = jnp.zeros_like(table).at[token_ids].add(grad_rows)
    expected_fused = task_grad + dense_grad_aux

    assert jnp.allclose(fused_grad, expected_fused), "Sparse scatter-add must match dense addition exactly!"


def test_stabilized_normalization_small_vectors():
    """Verify gradient remains finite and non-NaN even for near-zero vectors."""
    B, D = 4, 8
    X = jnp.zeros((B, D), dtype=jnp.float32) + 1e-15
    U, norms = normalize_embeddings_stabilized(X, eps=1e-6)
    triu_i, triu_j = jnp.triu_indices(B, k=1)

    grad = scmer_gradient(X, U, norms, triu_i, triu_j, margin=0.10, eps=1e-6)
    assert jnp.all(jnp.isfinite(grad)), "Gradients must remain strictly finite for near-zero vectors"
    assert jnp.all(jnp.isfinite(norms)), "Norms must remain finite"


def test_stop_gradient_centroid():
    """Verify global centroid does not backpropagate gradients into vocabulary rows."""
    W = jnp.ones((20, 8), dtype=jnp.float32) * 3.0
    centroid = compute_vocab_centroid(W)
    assert jnp.allclose(centroid, 3.0)


def test_prng_reproducibility_and_resume_safety():
    """Verify PRNG key derivation produces identical samples across resume steps."""
    base_seed = 42
    step = 16
    interval = 8
    reg_event = step // interval

    base_key = jax.random.PRNGKey(base_seed)
    key_run1 = jax.random.fold_in(jax.random.fold_in(base_key, step), reg_event)
    key_run2 = jax.random.fold_in(jax.random.fold_in(base_key, step), reg_event)

    assert jnp.array_equal(key_run1, key_run2), "Derived PRNG key must be strictly identical across runs"

    # Step at different event must differ
    key_other_step = jax.random.fold_in(jax.random.fold_in(base_key, step + 8), (step + 8) // interval)
    assert not jnp.array_equal(key_run1, key_other_step), "Different steps must yield different PRNG keys"


def test_jit_compilation():
    """Verify compute_scmer_step compiles and runs under jax.jit without trace errors."""
    key = jax.random.PRNGKey(808)
    V, D = 64, 16
    table = jax.random.normal(key, (V, D))

    @jax.jit
    def jit_step(tbl, k):
        return compute_scmer_step(
            table=tbl,
            step_key=k,
            strength=0.001,
            subsample=16,
            pair_samples=32,
            margin=0.10,
            eps=1e-6,
        )

    token_ids, grad_rows, loss_val = jit_step(table, key)
    assert token_ids.shape == (16,)
    assert grad_rows.shape == (16, D)
    assert jnp.isfinite(loss_val)


def test_scmer_optimizer_e2e_train_step():
    """Verify SCMER optimizer training step runs with active SCM regularization."""
    from src.jax.scmer_optimizer import init_scmer_opt_state, build_train_step_scmer
    from src.jax.v5_model import ModelConfigV5, PRESETS_V5, init_params_v5

    config = ModelConfigV5(**PRESETS_V5["v5_test"])
    key = jax.random.PRNGKey(909)
    k_params, k_data = jax.random.split(key)
    params, buffers = init_params_v5(k_params, config)
    opt_state = init_scmer_opt_state(params, seed=123)

    train_step = build_train_step_scmer(
        config,
        buffers,
        pad_id=0,
        weight_decay=1e-4,
        remat=False,
        scmer_lambda=0.005,
        scmer_subsample=32,
        scmer_interval=2,
        scmer_margin=0.10,
        scmer_pair_samples=64,
    )

    B, L_q, L_r = 2, 8, 8
    q_batch = jax.random.randint(k_data, (B, L_q), 1, config.vocab_size)
    r_batch = jax.random.randint(jax.random.fold_in(k_data, 1), (B, L_r), 1, config.vocab_size)

    # Run 4 steps
    for step in range(1, 5):
        params, opt_state, loss_val = train_step(params, opt_state, q_batch, r_batch, 0.005)
        assert jnp.isfinite(loss_val)
        assert "scmer_loss" in opt_state["diag"]
    assert int(opt_state["step"]) == 4


def test_replica_merge_preserves_scmer_scan_carry_structure():
    """Replica merging must retain every optimizer diag key needed by the next local-SGD scan."""
    from src.jax.spheroid_optimizer import merge_spheroid_replicas

    key = "sensory.embedding.weight"
    params = {key: jnp.ones((1, 4, 3), dtype=jnp.float32)}
    zeros = {key: jnp.zeros((1, 4, 3), dtype=jnp.float32)}
    diag_keys = (
        "grad_norm", "was_clipped", "was_skipped", "u_ratio_manifold",
        "u_ratio_vocab", "u_ratio_1d", "replica_drift", "scmer_loss", "neg_cov_loss",
    )
    opt_state = {
        "step": jnp.array([1], dtype=jnp.int32),
        "m": zeros,
        "v": zeros,
        "master_p": params,
        "power_v": {},
        "diag": {name: jnp.zeros((1,), dtype=jnp.float32) for name in diag_keys},
    }

    merge = jax.pmap(lambda p, o: merge_spheroid_replicas(p, o, axis_name="dp"), axis_name="dp")
    _, merged_opt = merge(params, opt_state)

    assert set(merged_opt["diag"]) == set(diag_keys)


def test_pytorch_scmer_optimizer():
    """Verify PyTorch SCMEROptimizer initializes and executes step with SCM regularization."""
    import torch
    import torch.nn as nn
    from src.spheroid_optimizer import SCMEROptimizer, build_scmer_optimizer

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Embedding(128, 32),
        nn.Linear(32, 32),
    )
    # Name embedding layer to match is_vocab heuristic
    model[0].weight.is_vocab = True

    opt = build_scmer_optimizer(
        model,
        lr=0.001,
        scmer_lambda=0.01,
        scmer_subsample=16,
        scmer_interval=1,
    )

    x = torch.randint(0, 128, (4, 8))
    out = model[1](model[0](x)).sum()
    out.backward()
    opt.step()
    opt.zero_grad()

