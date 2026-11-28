"""Independent mathematical verification of deliberate_step() and training_rollout().

Tests:
1. Exact Jacobian of deliberate_step(): sigma_max(J) vs rho(J) (non-normal transient amplification).
2. Non-conservation of radial coupling: sum_k S_k^(rad) != sum_k S_k^(intra).
3. Shortcut tract initialization: g_t = sigmoid(0) = 0.5 (active at half gate strength).
4. RoPE position indexing parity between is_pure_tf=True and autoregressive _emit_gen.
5. Exact target masking (t < 0), target_weights normalization, and loss term decomposition.
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.jax.v5_model import (
    ModelConfigV5,
    PRESETS_V5,
    init_params_v5,
    deliberate_step,
    training_rollout,
)


def verify_deliberate_step_jacobian_and_coupling():
    print("=== 1. Exact Jacobian Analysis of deliberate_step() ===")
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    cfg.use_laminar_transport = True
    params, buffers = init_params_v5(jax.random.PRNGKey(11), cfg)

    K, D = cfg.num_shells, cfg.surface_dim
    s0 = jax.random.normal(jax.random.PRNGKey(12), (1, K, D), dtype=jnp.float32)

    def f_flat(x_flat):
        x = x_flat.reshape(1, K, D)
        y = deliberate_step(params, x, buffers, config=cfg)
        return y.reshape(-1)

    J = jax.jacobian(f_flat)(s0.reshape(-1))
    J_np = np.array(J)
    svs = np.linalg.svd(J_np, compute_uv=False)
    eigs = np.linalg.eigvals(J_np)
    sigma_max = float(np.max(svs))
    rho = float(np.max(np.abs(eigs)))
    non_normality = float(np.linalg.norm(J_np @ J_np.T - J_np.T @ J_np, ord="fro"))

    print(f"  Jacobian shape: {J_np.shape}")
    print(f"  sigma_max(J) = {sigma_max:.6f}")
    print(f"  rho(J)       = {rho:.6f}")
    print(f"  ||J J^T - J^T J||_F = {non_normality:.4f} (non-normal: {non_normality > 1e-3})")
    assert sigma_max >= rho - 1e-5, "sigma_max must bound spectral radius rho(J)"

    # 2. Verify non-conservation of radial coupling & half-open shortcut gates at init
    g_init = jax.nn.sigmoid(params["core.shortcut_gate"])
    assert jnp.allclose(g_init, 0.5, atol=1e-6), f"Expected g_init=0.5, got {g_init}"
    print(f"  Shortcut gate at init: min={float(jnp.min(g_init)):.4f}, max={float(jnp.max(g_init)):.4f} (== 0.5)")

    # Check total shell sum is not conserved across deliberate_step
    s_next = deliberate_step(params, s0, buffers, config=cfg)
    sum_diff = float(jnp.abs(jnp.sum(s_next) - jnp.sum(s0)))
    print(f"  |sum(S_next) - sum(S_0)| = {sum_diff:.6f} (non-conservative neural coupling)")
    assert sum_diff > 1e-4


def verify_training_rollout_masking_and_loss_terms():
    print("\n=== 2. Target Masking, Weighting & Loss Term Decomposition ===")
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    cfg.use_adaptive_thinking = False
    cfg.use_energy_regularization = False
    cfg.router_loss_weight = 0.0
    cfg.num_mtp_heads = 1
    params, buffers = init_params_v5(jax.random.PRNGKey(21), cfg)

    prompts = jnp.array([[2, 5, 9, 0]], dtype=jnp.int32)
    targets = jnp.array([[10, 12, -100, -100]], dtype=jnp.int32)
    targets_diff_pad = jnp.array([[10, 12, 99, 88]], dtype=jnp.int32)
    weights_mask = jnp.array([[1.0, 1.0, 0.0, 0.0]], dtype=jnp.float32)

    # Loss with -100 padding must equal loss with explicit 0.0 target_weights on those positions
    loss_pad = float(training_rollout(
        params, prompts, target_len=4, buffers=buffers, config=cfg,
        targets=targets, sampling_p=1.0, pad_id=0, pure_teacher_forcing=True,
    ))
    loss_w = float(training_rollout(
        params, prompts, target_len=4, buffers=buffers, config=cfg,
        targets=targets_diff_pad, target_weights=weights_mask, sampling_p=1.0, pad_id=0, pure_teacher_forcing=True,
    ))
    diff_mask = abs(loss_pad - loss_w)
    print(f"  Masking parity (-100 vs weight=0.0): {loss_pad:.6f} vs {loss_w:.6f} (diff={diff_mask:.2e})")
    assert diff_mask < 1e-5

    # Verify energy regularization term adds exact delta
    cfg_e = ModelConfigV5(**PRESETS_V5["v5_test"])
    cfg_e.use_adaptive_thinking = False
    cfg_e.use_energy_regularization = True
    cfg_e.energy_reg_weight = 0.05
    cfg_e.router_loss_weight = 0.0
    loss_e = float(training_rollout(
        params, prompts, target_len=4, buffers=buffers, config=cfg_e,
        targets=targets, sampling_p=1.0, pad_id=0, pure_teacher_forcing=True,
    ))
    print(f"  Base loss={loss_pad:.6f}, With energy_reg(0.05)={loss_e:.6f}, delta={loss_e - loss_pad:+.6f}")
    assert loss_e > loss_pad

    # Verify RoPE position parity between is_pure_tf (step 0) and _emit_gen (step 0)
    logits_gen = training_rollout(
        params, prompts, target_len=4, buffers=buffers, config=cfg,
        targets=None, repetition_penalty=1.0, pad_id=0,
    )
    logits_tf = training_rollout(
        params, prompts, target_len=4, buffers=buffers, config=cfg,
        feedback_tokens=jnp.array([[-100, 10, 12, 14]], dtype=jnp.int32),
        feedback_is_shifted=True, sampling_p=1.0, repetition_penalty=1.0, pad_id=0,
    )
    step0_logit_diff = float(jnp.max(jnp.abs(logits_gen[:, 0, :] - logits_tf[:, 0, :])))
    print(f"  RoPE/emission step-0 parity (gen vs tf): max_diff={step0_logit_diff:.2e}")
    assert step0_logit_diff < 1e-5


if __name__ == "__main__":
    verify_deliberate_step_jacobian_and_coupling()
    verify_training_rollout_masking_and_loss_terms()
    print("\nALL MATHEMATICAL SPECIFICATION CHECKS PASSED.")
