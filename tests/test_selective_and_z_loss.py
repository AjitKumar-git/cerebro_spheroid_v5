import os
import sys
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.jax.v5_model import (
    ModelConfigV5,
    _compute_token_ce_and_z_loss,
    init_params_v5,
    training_rollout,
)
from training.train_v5_tpu_jax import parse_args


def test_selective_token_weights_downweight_easy_and_cap_noise():
    cfg_base = ModelConfigV5(
        surface_dim=64, radial_dim=32, num_shells=4, num_shortcut_tracts=8,
        vocab_size=128, num_heads=4, use_selective_loss=False, z_loss_weight=0.0,
    )
    cfg_sel = ModelConfigV5(
        surface_dim=64, radial_dim=32, num_shells=4, num_shortcut_tracts=8,
        vocab_size=128, num_heads=4, use_selective_loss=True,
        selective_focal_gamma=0.5, selective_noise_cap=8.5, z_loss_weight=1e-4,
    )

    # Token 0: mastered (ce=0.05), Token 1: learnable middle (ce=3.0), Token 2: noise spike (ce=17.0), Token 3: masked
    ce = jnp.array([0.05, 3.0, 17.0, 5.0], dtype=jnp.float32)
    lse = jnp.array([2.0, 3.0, 10.0, 1.0], dtype=jnp.float32)
    mask = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)

    loss_base, w_base = _compute_token_ce_and_z_loss(ce, lse, mask, mask, cfg_base)
    loss_sel, w_sel = _compute_token_ce_and_z_loss(ce, lse, mask, mask, cfg_sel)

    assert jnp.allclose(w_base, mask)
    # Learnable middle token (idx 1) must receive higher relative weight than mastered (idx 0) and noise-capped (idx 2)
    assert float(w_sel[1]) > float(w_sel[0])
    assert float(w_sel[1]) > float(w_sel[2])
    # Mean weight over valid tokens remains ~1.0
    assert abs(float(jnp.sum(w_sel * mask) / jnp.sum(mask)) - 1.0) < 1e-5
    # Masked token stays zero
    assert float(loss_sel[3]) == 0.0 and float(w_sel[3]) == 0.0


def test_training_rollout_selective_and_z_loss_gradients_in_both_modes():
    cfg = ModelConfigV5(
        surface_dim=64, radial_dim=32, num_shells=4, num_shortcut_tracts=8,
        vocab_size=128, num_heads=4, thinking_cycles=2,
        use_selective_loss=True, selective_focal_gamma=0.5, selective_noise_cap=8.5,
        z_loss_weight=1e-4,
    )
    params, buffers = init_params_v5(jax.random.PRNGKey(0), cfg)
    prompts = jnp.array([[1, 5, 9, 12], [2, 6, 10, 14]], dtype=jnp.int32)
    targets = jnp.array([[5, 9, 12, 3], [6, 10, 14, 4]], dtype=jnp.int32)

    for mode in ("teacher_forcing", "scheduled"):
        def loss_fn(p):
            return training_rollout(
                p, prompts, target_len=4, buffers=buffers, config=cfg,
                targets=targets, deliberation_mode=mode, sampling_p=1.0 if mode == "teacher_forcing" else 0.8,
                remat=False,
            )
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        assert jnp.isfinite(loss_val) and float(loss_val) > 0.0
        emb_grad = grads["sensory.embedding.weight"]
        assert jnp.all(jnp.isfinite(emb_grad))
        assert float(jnp.linalg.norm(emb_grad)) > 0.0


def test_cli_flags_parse_for_selective_and_z_loss():
    args = parse_args(["--selective_loss", "--selective_focal_gamma", "0.75", "--selective_noise_cap", "7.5", "--z_loss_weight", "1e-4"])
    assert args.selective_loss is True
    assert abs(args.selective_focal_gamma - 0.75) < 1e-6
    assert abs(args.selective_noise_cap - 7.5) < 1e-6
    assert abs(args.z_loss_weight - 1e-4) < 1e-9
