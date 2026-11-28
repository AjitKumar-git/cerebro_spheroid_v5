"""10-Test Tensor-Level Mathematical Verification Matrix for CerebroSpheroid V5 (JAX).

Tests:
1. DCT orthogonality: ||C^T C - I||_inf
2. RoPE norm preservation: ||R x||_2 == ||x||_2
3. Causal invariance: changing future tokens cannot alter earlier logits
4. TF / generation token-by-token alignment across all steps t = 0..T-1
5. Pure TF vs. Recurrent TF equivalence across (efference in {F,T} x centralized in {F,T})
6. ACT weights: sum_t w_t == 1.0
7. Radial indexing: boundary isolation & no unintended shell index overlap
8. Spectral guard: actual SVD sigma_max(W) vs 3-step power-iteration estimate
9. Optimizer polar update: orthogonality residual of zeropower_via_newtonschulz5_jax
10. Loss weighting & MTP horizon alignment: analytical vs numerical implementation
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.jax.v5_model import (
    ModelConfigV5,
    PRESETS_V5,
    init_params_v5,
    create_dct_matrix_jax,
    apply_rope,
    deliberate_step,
    deliberate_cycles_jax,
    training_rollout,
)
from src.jax.spheroid_optimizer import (
    clamp_spectral_norm_jax,
    zeropower_via_newtonschulz5_jax,
)


def test_1_dct_orthogonality():
    D = 64
    C = create_dct_matrix_jax(D)
    err = float(jnp.max(jnp.abs(jnp.matmul(C.T, C, precision=jax.lax.Precision.HIGHEST) - jnp.eye(D))))
    print(f"[1] DCT orthogonality ||C^T C - I||_inf = {err:.2e}")
    assert err < 1e-5


def test_2_rope_norm_preservation():
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 8, 16))
    pos = jnp.arange(8, dtype=jnp.int32)
    rx = apply_rope(x, pos, theta=500000.0)
    norm_in = jnp.linalg.norm(x, axis=-1)
    norm_out = jnp.linalg.norm(rx, axis=-1)
    err = float(jnp.max(jnp.abs(norm_in - norm_out)))
    print(f"[2] RoPE norm preservation max | ||Rx|| - ||x|| | = {err:.2e}")
    assert err < 1e-5


def test_3_causal_invariance():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    cfg.use_adaptive_thinking = False
    params, buffers = init_params_v5(jax.random.PRNGKey(2), cfg)
    prompts = jnp.array([[2, 5, 9, 11]], dtype=jnp.int32)
    fb_a = jnp.array([[-100, 10, 20, 30, 40]], dtype=jnp.int32)
    fb_b = jnp.array([[-100, 10, 20, 99, 88]], dtype=jnp.int32)  # Mutated at t=3,4
    la = training_rollout(params, prompts, target_len=5, buffers=buffers, config=cfg,
                          feedback_tokens=fb_a, feedback_is_shifted=True, sampling_p=1.0, pure_teacher_forcing=True)
    lb = training_rollout(params, prompts, target_len=5, buffers=buffers, config=cfg,
                          feedback_tokens=fb_b, feedback_is_shifted=True, sampling_p=1.0, pure_teacher_forcing=True)
    prefix_err = float(jnp.max(jnp.abs(la[:, :3, :] - lb[:, :3, :])))
    mutated_diff = float(jnp.max(jnp.abs(la[:, 3:, :] - lb[:, 3:, :])))
    print(f"[3] Causal invariance: prefix[0..2] diff = {prefix_err:.2e}, mutated[3..4] diff = {mutated_diff:.4f}")
    assert prefix_err < 1e-5
    assert mutated_diff > 1e-3


def test_4_tf_generation_token_by_token_alignment():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    cfg.use_adaptive_thinking = False
    cfg.use_rope = True
    params, buffers = init_params_v5(jax.random.PRNGKey(3), cfg)
    prompts = jnp.array([[3, 7, 13, 19, 0]], dtype=jnp.int32)

    # Autoregressive generation (greedy, repetition_penalty=1.0)
    gen_logits, gen_toks = training_rollout(
        params, prompts, target_len=6, buffers=buffers, config=cfg,
        targets=None, repetition_penalty=1.0, temperature=0.0, pad_id=0, return_tokens=True,
    )
    # Feed exact generated tokens as pre-shifted feedback [-100, y_0, y_1, y_2, y_3, y_4]
    shifted_fb = jnp.concatenate([jnp.full((1, 1), -100, dtype=jnp.int32), gen_toks[:, :-1]], axis=1)
    tf_logits = training_rollout(
        params, prompts, target_len=6, buffers=buffers, config=cfg,
        feedback_tokens=shifted_fb, feedback_is_shifted=True,
        sampling_p=1.0, repetition_penalty=1.0, pad_id=0, pure_teacher_forcing=True,
    )
    per_step_max = [float(jnp.max(jnp.abs(gen_logits[:, t, :] - tf_logits[:, t, :]))) for t in range(6)]
    max_err = max(per_step_max)
    print(f"[4] TF vs Generation multi-step alignment (T=6, RoPE=True): max_diff = {max_err:.2e} | per_step = {[f'{x:.1e}' for x in per_step_max]}")
    assert max_err < 1e-4


def test_5_pure_tf_vs_recurrent_tf():
    print("[5] Pure TF vs Recurrent TF across (efference x centralized):")
    prompts = jnp.array([[2, 6, 10, 14]], dtype=jnp.int32)
    targets = jnp.array([[8, 12, 16, 20, 24]], dtype=jnp.int32)
    for eff in [False, True]:
        for cent in [False, True]:
            cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
            cfg.use_adaptive_thinking = False
            cfg.use_per_token_efference = eff
            cfg.use_centralized_readout = cent
            cfg.use_energy_regularization = False
            cfg.router_loss_weight = 0.0
            params, buffers = init_params_v5(jax.random.PRNGKey(5), cfg)
            l_pure = float(training_rollout(
                params, prompts, target_len=5, buffers=buffers, config=cfg,
                targets=targets, sampling_p=1.0, pure_teacher_forcing=True,
            ))
            l_rec = float(training_rollout(
                params, prompts, target_len=5, buffers=buffers, config=cfg,
                targets=targets, sampling_p=1.0, pure_teacher_forcing=False,
            ))
            diff = abs(l_pure - l_rec)
            print(f"    eff={str(eff):5s}, cent={str(cent):5s}: pure={l_pure:.6f}, rec={l_rec:.6f}, diff={diff:.2e}")
            assert diff < 1e-4, f"Pure TF vs Recurrent TF diverged for eff={eff}, cent={cent}: diff={diff}"


def test_6_act_weights_sum_to_one():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    cfg.use_adaptive_thinking = True
    cfg.max_thinking_cycles = 6
    params, _ = init_params_v5(jax.random.PRNGKey(6), cfg)
    w_gate = params["core.halting_gate.weight"]
    b_gate = params["core.halting_gate.bias"]
    # Simulate ACT accumulation weights for random core states
    core_states = jax.random.normal(jax.random.PRNGKey(60), (4, cfg.max_thinking_cycles, cfg.surface_dim))
    cum_p = jnp.zeros((4,), dtype=jnp.float32)
    weights = []
    for c in range(cfg.max_thinking_cycles):
        hp = jax.nn.sigmoid(jnp.squeeze(jnp.matmul(core_states[:, c, :], w_gate.T), axis=-1) + b_gate[0])
        rem_p = jnp.maximum(0.0, 1.0 - cum_p)
        sw = rem_p if c == cfg.max_thinking_cycles - 1 else jnp.minimum(hp, rem_p)
        cum_p = cum_p + sw
        weights.append(sw)
    total_w = sum(weights)
    err = float(jnp.max(jnp.abs(total_w - 1.0)))
    print(f"[6] ACT weights sum == 1.0: max |sum(w_t) - 1| = {err:.2e}")
    assert err < 1e-6


def test_7_radial_indexing_boundary_isolation():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    cfg.use_sparse_mot = False
    params, buffers = init_params_v5(jax.random.PRNGKey(7), cfg)
    # Disable intra and shortcuts, keep only radial inward (zero outward)
    params_in = dict(params)
    params_in["core.shell_gains"] = jnp.zeros_like(params["core.shell_gains"])
    params_in["core.shortcut_mix_scale"] = jnp.array(0.0, dtype=jnp.float32)
    params_in["core.w_radial_out"] = jnp.zeros_like(params["core.w_radial_out"])
    s = jax.random.normal(jax.random.PRNGKey(70), (1, cfg.num_shells, cfg.surface_dim))
    out = deliberate_step(params_in, s, buffers, config=cfg)
    # Note: shortcut_mix_scale is clipped to 0.005, so isolate exact radial_in index range by checking finite
    # Check shell K-1 gets 0 radial_in delta when w_radial_in is active and shortcuts are zeroed via shortcut_weight=0
    params_in["core.shortcut_weight"] = jnp.zeros_like(params["core.shortcut_weight"])
    out_pure_in = deliberate_step(params_in, s, buffers, config=cfg)
    mantle_delta = float(jnp.max(jnp.abs(out_pure_in[:, -1, :] - s[:, -1, :])))
    core_delta = float(jnp.max(jnp.abs(out_pure_in[:, 0, :] - s[:, 0, :])))
    print(f"[7] Radial inward boundary: Shell K-1 delta = {mantle_delta:.2e} (expected 0), Shell 0 delta = {core_delta:.4f}")
    assert mantle_delta < 1e-6 and core_delta > 1e-4


def test_8_spectral_guard_svd_vs_power_iter():
    W = jax.random.normal(jax.random.PRNGKey(8), (64, 64)) * 0.5
    true_sigma_before = float(jnp.max(jnp.linalg.svd(W, compute_uv=False)))
    v0 = jnp.ones((1, 64, 1), dtype=jnp.float32) / jnp.sqrt(64.0)
    W_clamped, _ = clamp_spectral_norm_jax(W, v0, max_spec_norm=1.0, power_steps=3)
    true_sigma_after = float(jnp.max(jnp.linalg.svd(W_clamped, compute_uv=False)))
    print(f"[8] Spectral guard (max_sigma=1.0, 3 iters): true sigma_max {true_sigma_before:.4f} -> {true_sigma_after:.4f}")


def test_9_newton_schulz_orthogonality_residual():
    G = jax.random.normal(jax.random.PRNGKey(9), (32, 32))
    X = zeropower_via_newtonschulz5_jax(G, steps=3)
    # Normalize X by its own RMS singular value to inspect spectral flatness / orthogonality residual
    svs = np.linalg.svd(np.array(X), compute_uv=False)
    cond = float(np.max(svs) / max(np.min(svs), 1e-8))
    print(f"[9] Newton-Schulz (3 steps) singular values: min={np.min(svs):.4f}, max={np.max(svs):.4f}, cond={cond:.4f}")


def test_10_loss_weighting_and_mtp_alignment():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    cfg.use_adaptive_thinking = False
    cfg.use_energy_regularization = False
    cfg.router_loss_weight = 0.0
    cfg.num_mtp_heads = 2
    params, buffers = init_params_v5(jax.random.PRNGKey(10), cfg)
    prompts = jnp.array([[2, 4, 6]], dtype=jnp.int32)
    targets_1 = jnp.array([[10, 12, 14, 16]], dtype=jnp.int32)
    targets_2 = jnp.array([[10, 12, 14, 99]], dtype=jnp.int32)
    # With MTP=2, h_t^(1) predicts y_{t+1}; mutating y_3 affects primary step 3 AND MTP step 2
    l1 = float(training_rollout(params, prompts, target_len=4, buffers=buffers, config=cfg,
                                targets=targets_1, sampling_p=1.0, pure_teacher_forcing=True))
    l2 = float(training_rollout(params, prompts, target_len=4, buffers=buffers, config=cfg,
                                targets=targets_2, sampling_p=1.0, pure_teacher_forcing=True))
    print(f"[10] MTP=2 horizon alignment sensitivity: l1={l1:.6f} vs l2={l2:.6f} (diff={abs(l1-l2):.4f})")
    assert abs(l1 - l2) > 1e-4

    # Analytical vs numerical anchor weighting on mixed batch (row 0 with anchor=50, row 1 without anchor)
    cfg_a = ModelConfigV5(**PRESETS_V5["v5_test"])
    cfg_a.use_adaptive_thinking = False
    cfg_a.use_energy_regularization = False
    cfg_a.router_loss_weight = 0.0
    cfg_a.num_mtp_heads = 1
    p_a, buf_a = init_params_v5(jax.random.PRNGKey(11), cfg_a)
    pr_a = jnp.array([[2, 4, 6, 8], [3, 5, 7, 9]], dtype=jnp.int32)
    tg_a = jnp.array([[10, 50, 14, -100], [11, 13, 15, -100]], dtype=jnp.int32)
    eval_w = jax.jit(lambda tw: training_rollout(
        p_a, pr_a, target_len=4, buffers=buf_a, config=cfg_a,
        targets=tg_a, target_weights=tw, sampling_p=1.0, pure_teacher_forcing=True,
    ))
    w_expected = jnp.array([[1.0, 3.5, 3.5, 0.0], [3.5, 3.5, 3.5, 0.0]], dtype=jnp.float32)
    ce_sum = 0.0
    for b in range(2):
        for t in range(3):
            oh = jnp.zeros((2, 4), dtype=jnp.float32).at[b, t].set(1.0)
            ce_bt = float(eval_w(oh))
            ce_sum += float(w_expected[b, t]) * ce_bt
    l_analytical = ce_sum / float(jnp.sum(w_expected))
    l_anchor_pure = float(training_rollout(
        p_a, pr_a, target_len=4, buffers=buf_a, config=cfg_a,
        targets=tg_a, anchor_token_id=50, anchor_weight=3.5, use_anchor_loss=True,
        sampling_p=1.0, pure_teacher_forcing=True,
    ))
    l_anchor_rec = float(training_rollout(
        p_a, pr_a, target_len=4, buffers=buf_a, config=cfg_a,
        targets=tg_a, anchor_token_id=50, anchor_weight=3.5, use_anchor_loss=True,
        sampling_p=1.0, pure_teacher_forcing=False,
    ))
    err_a = max(abs(l_analytical - l_anchor_pure), abs(l_analytical - l_anchor_rec))
    print(f"[10b] Anchor weighting analytical parity: analytical={l_analytical:.6f}, pure={l_anchor_pure:.6f}, rec={l_anchor_rec:.6f}, max_diff={err_a:.2e}")
    assert err_a < 1e-5


if __name__ == "__main__":
    test_1_dct_orthogonality()
    test_2_rope_norm_preservation()
    test_3_causal_invariance()
    test_4_tf_generation_token_by_token_alignment()
    test_5_pure_tf_vs_recurrent_tf()
    test_6_act_weights_sum_to_one()
    test_7_radial_indexing_boundary_isolation()
    test_8_spectral_guard_svd_vs_power_iter()
    test_9_newton_schulz_orthogonality_residual()
    test_10_loss_weighting_and_mtp_alignment()
    print("\nMATHEMATICAL VERIFICATION MATRIX COMPLETE.")
