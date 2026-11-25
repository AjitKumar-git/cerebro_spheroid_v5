"""
SpheroidOptimizer (Pure JAX Implementation):
Manifold & Spectral-Aware Optimizer for CerebroSpheroid Architectures on Cloud TPUs and GPUs.

Architecture Parity with PyTorch / GPU SpheroidOptimizer:
1. Manifold Matrix Operators (rank >= 2):
   Updated along nearest orthogonal manifold via Quintic Newton-Schulz iterations (Muon style).
   Preserves continuous Riemannian rotational isometry without AdamW coordinate-wise distortion.
   Maintains a single momentum buffer (50% memory saving).
2. Recurrent Manifold Core (intra_shell, reentrant_field, shortcut_tracts):
   Orthogonal polar updates with persistent power-iteration contractive spectral clamp (sigma_max <= 1.0).
3. Large Vocabulary Tables & Output Heads:
   Optimized with AdamW (first and second moments) to gracefully handle sparse tokens and frequency skew.
4. 1D Parameters (Scalars, Pacemakers, Biases, RMSNorm weights):
   Optimized with AdamW (zero weight decay) to handle heterogeneous parameter scales stably.
"""

from functools import partial
from typing import Dict, Tuple, Set, Optional, Callable, Any
import math

import jax
import jax.numpy as jnp
from jax import lax

try:
    from .v5_model import ModelConfigV5, training_rollout
except ImportError:
    pass

try:
    from .scmer import compute_scmer_step
except ImportError:
    try:
        from scmer import compute_scmer_step
    except ImportError:
        from src.jax.scmer import compute_scmer_step

try:
    from .negative_covariance import compute_negative_covariance_step
except ImportError:
    from negative_covariance import compute_negative_covariance_step

RECURRENT_KEYS: Set[str] = {
    "core.intra_shell", "core.reentrant_field", "core.shortcut_tracts",
    "volumetric_core.w_radial_in", "volumetric_core.w_radial_out", "volumetric_core.shortcut_weight",
    "core.intra_swiglu.w_gate", "core.intra_swiglu.w_val", "core.intra_swiglu.w_down",
    "core.w_radial_in", "core.w_radial_out", "core.shortcut_weight",
}
VOCAB_KEYS: Set[str] = {
    "sensory.embedding.weight", "sensory_dnn.embedding.weight",
    "sensory.pos_embed.weight", "sensory_dnn.pos_embed.weight",
    "motor.head.weight", "motor_decoder.head.weight",
    "motor.phase_embed.weight", "motor_decoder.phase_embed.weight",
    "motor.persona_embeddings", "motor_decoder.persona_embeddings",
}
NORM_KEYS: Set[str] = {
    "core.intra_norms.weight",
    "volumetric_core.intra_norms.weight",
}


def is_manifold_operator(k: str, p: jnp.ndarray) -> bool:
    """Returns True for 2D, 3D, and 4D linear operator matrices updated via Polar Newton-Schulz."""
    if k in VOCAB_KEYS or k in NORM_KEYS or "norm" in k.lower():
        return False
    return p.ndim >= 2


def zeropower_via_newtonschulz5_jax(G: jnp.ndarray, steps: int = 3, eps: float = 1e-7) -> jnp.ndarray:
    """Vectorized Quintic Newton-Schulz iteration mapping update matrices to the orthogonal manifold.
    Runs strictly in float32 to prevent half-precision exponent overflow.
    For batched 3D tensors (e.g. concentric shell stacks), applies slice-relative trust scaling
    to prevent noisy, low-gradient slices from taking disproportionately massive updates."""
    orig_dtype = G.dtype
    G_f32 = G.astype(jnp.float32)
    orig_shape = G_f32.shape
    if G_f32.ndim > 2:
        G_mat = G_f32.reshape(-1, orig_shape[-2], orig_shape[-1])
    else:
        G_mat = G_f32[None, ...]

    rows, cols = G_mat.shape[-2], G_mat.shape[-1]
    transposed = rows > cols
    if transposed:
        G_mat = jnp.swapaxes(G_mat, -2, -1)

    slice_norms = jnp.sqrt(jnp.sum(jnp.square(G_mat), axis=(-2, -1), keepdims=True) + eps)
    # If 3D batched tensor with multiple slices, scale polar update by relative slice confidence
    if G_f32.ndim > 2 and orig_shape[0] > 1:
        max_slice_norm = jnp.maximum(jnp.max(slice_norms), 1e-6)
        slice_scale = jnp.clip(slice_norms / (0.05 * max_slice_norm + eps), 0.0, 1.0)
    else:
        slice_scale = 1.0

    X = G_mat / slice_norms

    # Quintic Newton-Schulz polynomial coefficients (static unroll improves XLA pipelining)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        A = jnp.matmul(X, jnp.swapaxes(X, -2, -1))
        B = b * A + c * jnp.matmul(A, A)
        X = a * X + jnp.matmul(B, X)

    X = X * slice_scale
    if transposed:
        X = jnp.swapaxes(X, -2, -1)
    return X.reshape(orig_shape).astype(orig_dtype)


def clamp_spectral_norm_jax(
    W: jnp.ndarray, v: jnp.ndarray, max_spec_norm: float = 1.0, power_steps: int = 3, eps: float = 1e-6
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Contractive spectral guard via persistent power iteration. Enforces sigma_max <= 1.0."""
    orig_dtype = W.dtype
    W_f32 = W.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)
    orig_shape = W_f32.shape
    if W_f32.ndim > 3:
        W_mat = W_f32.reshape(-1, orig_shape[-2], orig_shape[-1])
    elif W_f32.ndim == 2:
        W_mat = W_f32[None, ...]
    else:
        W_mat = W_f32

    v_final = v_f32
    for _ in range(int(power_steps)):
        u = jnp.matmul(W_mat, v_final)
        u_norm = jnp.sqrt(jnp.sum(jnp.square(u), axis=1, keepdims=True) + eps)
        u = u / u_norm
        v_next = jnp.matmul(jnp.swapaxes(W_mat, -2, -1), u)
        v_norm = jnp.sqrt(jnp.sum(jnp.square(v_next), axis=1, keepdims=True) + eps)
        v_final = v_next / v_norm

    u_final = jnp.matmul(W_mat, v_final)
    sigma = jnp.sqrt(jnp.sum(jnp.square(u_final), axis=1, keepdims=True) + eps)
    # Calibrated 1.05 safety factor accounts for power iteration underestimation of exact SVD
    scale = jnp.minimum(1.0, max_spec_norm / (sigma * 1.05 + eps))
    W_clamped = (W_mat * scale).reshape(orig_shape).astype(orig_dtype)
    return W_clamped, v_final.astype(v.dtype)


def init_spheroid_opt_state(params: Dict[str, jnp.ndarray], key: Optional[jax.Array] = None) -> Dict[str, Any]:
    """Initializes single-moment buffers for matrices, AdamW buffers for vocab/1D, and power vectors."""
    if key is None:
        key = jax.random.PRNGKey(42)

    opt_m: Dict[str, jnp.ndarray] = {}
    opt_v: Dict[str, jnp.ndarray] = {}
    master_p: Dict[str, jnp.ndarray] = {}
    power_v: Dict[str, jnp.ndarray] = {}

    for k, p in params.items():
        if not jnp.issubdtype(p.dtype, jnp.inexact):
            continue
        # Store master weights and moments in float32 to prevent BF16 ULP update freezing
        master_p[k] = p.astype(jnp.float32)
        opt_m[k] = jnp.zeros(p.shape, dtype=jnp.float32)
        if not is_manifold_operator(k, p):
            opt_v[k] = jnp.zeros(p.shape, dtype=jnp.float32)
        if k in RECURRENT_KEYS:
            key, subkey = jax.random.split(key)
            orig_shape = p.shape
            mat_b = orig_shape[0] if p.ndim == 3 else (orig_shape[0] * orig_shape[1] if p.ndim == 4 else 1)
            mat_n = orig_shape[-1]
            v_init = jax.random.normal(subkey, (mat_b, mat_n, 1), dtype=jnp.float32)
            v_init = v_init / jnp.sqrt(jnp.sum(jnp.square(v_init), axis=1, keepdims=True) + 1e-6)
            power_v[k] = v_init

    return {
        "step": jnp.array(0, dtype=jnp.int32),
        "m": opt_m,
        "v": opt_v,
        "master_p": master_p,
        "power_v": power_v,
        "diag": {
            "grad_norm": jnp.array(0.0, dtype=jnp.float32),
            "was_clipped": jnp.array(0.0, dtype=jnp.float32),
            "was_skipped": jnp.array(0.0, dtype=jnp.float32),
            "u_ratio_manifold": jnp.array(0.0, dtype=jnp.float32),
            "u_ratio_vocab": jnp.array(0.0, dtype=jnp.float32),
            "u_ratio_1d": jnp.array(0.0, dtype=jnp.float32),
            "replica_drift": jnp.array(0.0, dtype=jnp.float32),
            "scmer_loss": jnp.array(0.0, dtype=jnp.float32),
            "neg_cov_loss": jnp.array(0.0, dtype=jnp.float32),
        },
    }


def _compute_loss_and_grads(
    p: Dict[str, jnp.ndarray],
    q_batch: jnp.ndarray,
    r_batch: jnp.ndarray,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: ModelConfigV5,
    pad_id: int,
    repetition_penalty: float,
    remat: bool,
    feedback_tokens: Optional[jnp.ndarray],
    sampling_p: Any,
    pure_tf: bool,
    deliberation_mode: str = "teacher_forcing",
    encephalization_alpha: float = 1.0,
    target_weights: Optional[jnp.ndarray] = None,
):
    """Computes loss and gradients for SpheroidOptimizer with static teacher-forcing dispatch."""
    target_len = r_batch.shape[1]

    def _loss_fn(param_dict):
        return training_rollout(
            param_dict,
            q_batch,
            target_len,
            buffers,
            config=config,
            feedback_tokens=feedback_tokens,
            sampling_p=sampling_p,
            pad_id=pad_id,
            repetition_penalty=repetition_penalty,
            targets=r_batch,
            remat=remat,
            pure_teacher_forcing=pure_tf,
            deliberation_mode=deliberation_mode,
            encephalization_alpha=encephalization_alpha,
            target_weights=target_weights,
        )

    return jax.value_and_grad(_loss_fn, allow_int=True)(p)


def _scmer_grad(
    p: Dict[str, jnp.ndarray],
    opt_s: Dict[str, Any],
    strength: float,
    sample_size: int = 512,
    interval: int = 8,
    margin: float = 0.10,
    pair_samples: int = 8192,
    eps: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Selective Cosine-Margin Embedding Regularization (SCMER).

    Computes a sparse auxiliary gradient on (B, D) alongside sampled token indices (B,).
    Does NOT allocate a second full (V, D) table.

    Derives PRNG key safely from base seed, optimizer step, and regularization event:
        sampling_key = fold_in(fold_in(base_key, step), reg_event)
    """
    key = "sensory.embedding.weight"
    zero = jnp.array(0.0, dtype=jnp.float32)
    if strength < 0.0:
        raise ValueError("scmer_lambda must be non-negative")
    if sample_size < 2:
        raise ValueError("scmer_subsample must be at least 2")
    if interval < 1:
        raise ValueError("scmer_interval must be at least 1")

    if key not in p or strength == 0.0:
        width = p[key].shape[1] if key in p else 1
        dummy_ids = jnp.zeros((sample_size,), dtype=jnp.int32)
        dummy_grad = jnp.zeros((sample_size, width), dtype=jnp.float32)
        return dummy_ids, dummy_grad, zero

    table = opt_s.get("master_p", {}).get(key, p[key]).astype(jnp.float32)
    vocab_size, width = table.shape
    actual_sample_size = min(max(2, int(sample_size)), int(vocab_size))
    dummy_ids = jnp.zeros((actual_sample_size,), dtype=jnp.int32)
    dummy_grad = jnp.zeros((actual_sample_size, width), dtype=jnp.float32)

    if vocab_size < 2:
        return dummy_ids, dummy_grad, zero

    next_step = opt_s["step"] + 1
    reg_event = next_step // interval

    # Reproducible resume-safe PRNG key derived from seed, step, and regularization event
    base_seed = opt_s.get("seed", 42)
    base_key = jax.random.PRNGKey(base_seed) if (not hasattr(base_seed, "ndim") or base_seed.ndim == 0) else base_seed
    sampling_key = jax.random.fold_in(jax.random.fold_in(base_key, next_step), reg_event)

    def _apply(_):
        token_ids, grad_rows, reg_loss = compute_scmer_step(
            table=table,
            step_key=sampling_key,
            strength=strength,
            subsample=sample_size,
            pair_samples=pair_samples,
            margin=margin,
            eps=eps,
        )
        return token_ids, grad_rows, reg_loss

    def _noop(_):
        return dummy_ids, dummy_grad, zero

    return lax.cond((next_step % interval) == 0, _apply, _noop, operand=None)


def _apply_spheroid_grads(
    p: Dict[str, jnp.ndarray],
    opt_s: Dict[str, Any],
    grads: Dict[str, jnp.ndarray],
    base_lr: Any,
    weight_decay: float,
    param_masks: Optional[Dict[str, jnp.ndarray]] = None,
    scmer_lambda: float = 0.05,
    scmer_subsample: int = 512,
    scmer_interval: int = 8,
    scmer_margin: float = 0.12,
    scmer_pair_samples: int = 8192,
    neg_cov_lambda: Optional[float] = None,
    neg_cov_mode: str = "cosine_asymmetric",
    neg_cov_subsample: Optional[int] = None,
    neg_cov_interval: Optional[int] = None,
    neg_cov_margin: Optional[float] = None,
    neg_cov_pair_samples: Optional[int] = None,
    grad_clip_norm: float = 1.0,
    lr_1d_scale: float = 0.2,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """Applies Polar Newton-Schulz, recurrent spectral clamp, and float32 master-weight AdamW."""
    eff_lambda = neg_cov_lambda if (neg_cov_lambda is not None and scmer_lambda == 0.0) else scmer_lambda
    eff_subsample = neg_cov_subsample if neg_cov_subsample is not None else scmer_subsample
    eff_interval = neg_cov_interval if neg_cov_interval is not None else scmer_interval
    eff_margin = neg_cov_margin if neg_cov_margin is not None else scmer_margin
    eff_pairs = neg_cov_pair_samples if neg_cov_pair_samples is not None else scmer_pair_samples

    scmer_token_ids, scmer_grad_rows, scmer_loss = _scmer_grad(
        p,
        opt_s,
        eff_lambda,
        sample_size=eff_subsample,
        interval=eff_interval,
        margin=eff_margin,
        pair_samples=eff_pairs,
    )
    if eff_lambda > 0.0 and "sensory.embedding.weight" in grads:
        grads = dict(grads)
        k = "sensory.embedding.weight"
        # Sparse scatter-add directly onto existing task gradient buffer:
        grads[k] = grads[k].astype(jnp.float32).at[scmer_token_ids].add(scmer_grad_rows)
    if param_masks is not None:
        grads = {
            k: (g * param_masks[k] if (k in param_masks and jnp.issubdtype(g.dtype, jnp.inexact)) else g)
            for k, g in grads.items()
        }

    grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for g in grads.values() if jnp.issubdtype(g.dtype, jnp.inexact))
    )
    is_finite = jnp.isfinite(grad_norm)
    clip_scale = jnp.where(is_finite, jnp.minimum(1.0, grad_clip_norm / (grad_norm + 1e-6)), 0.0)
    clipped_grads = jax.tree_util.tree_map(lambda g: g * clip_scale if jnp.issubdtype(g.dtype, jnp.inexact) else g, grads)

    def _apply_step():
        cur_step = opt_s["step"] + 1
        has_master = "master_p" in opt_s
        new_params = {}
        new_m = {}
        new_v = {}
        new_master_p = {}
        new_power_v = dict(opt_s["power_v"])

        lr_1d = lr_1d_scale * base_lr
        lr_vocab = jnp.minimum(base_lr, 6.0e-4)

        # Track squared update and parameter norms per group for diagnostic logging
        sq_diff_m = 0.0
        sq_norm_m = 0.0
        sq_diff_v = 0.0
        sq_norm_v = 0.0
        sq_diff_1d = 0.0
        sq_norm_1d = 0.0

        for k, param in p.items():
            if not jnp.issubdtype(param.dtype, jnp.inexact):
                new_params[k] = param
                continue

            g = clipped_grads[k].astype(jnp.float32)
            m_prev = opt_s["m"][k].astype(jnp.float32)
            p_f32 = opt_s["master_p"][k] if has_master else param.astype(jnp.float32)

            if is_manifold_operator(k, param):
                m_new = 0.95 * m_prev + 0.05 * g
                new_m[k] = m_new.astype(opt_s["m"][k].dtype)

                # Recurrent manifold weights are bounded by contractive spectral clamp; exempt from weight decay
                decay_factor = 0.0 if k in RECURRENT_KEYS else weight_decay
                p_decayed = p_f32 * (1.0 - base_lr * decay_factor)
                O = zeropower_via_newtonschulz5_jax(m_new, steps=3).astype(jnp.float32)

                rows = param.shape[-2]
                cols = param.shape[-1]
                aspect_scale = math.sqrt(max(1.0, float(rows) / float(cols)))
                p_updated = p_decayed - (base_lr * aspect_scale) * O

                if k in RECURRENT_KEYS:
                    p_updated, v_next = clamp_spectral_norm_jax(p_updated, opt_s["power_v"][k], max_spec_norm=1.0)
                    new_power_v[k] = v_next

                sq_diff_m = sq_diff_m + jnp.sum(jnp.square(p_updated - p_f32))
                sq_norm_m = sq_norm_m + jnp.sum(jnp.square(p_f32))

                new_master_p[k] = p_updated.astype(jnp.float32)
                new_params[k] = p_updated.astype(param.dtype)

            elif k in VOCAB_KEYS:
                v_prev = opt_s["v"][k].astype(jnp.float32)
                m_new = 0.9 * m_prev + 0.1 * g
                v_new = 0.95 * v_prev + 0.05 * jnp.square(g)
                new_m[k] = m_new.astype(opt_s["m"][k].dtype)
                new_v[k] = v_new.astype(opt_s["v"][k].dtype)

                m_hat = m_new / (1.0 - jnp.power(0.9, cur_step))
                v_hat = v_new / (1.0 - jnp.power(0.95, cur_step))
                step_dir = m_hat / (jnp.sqrt(v_hat) + 1e-6)
                p_decayed = p_f32 * (1.0 - lr_vocab * weight_decay)
                p_updated = p_decayed - lr_vocab * step_dir

                sq_diff_v = sq_diff_v + jnp.sum(jnp.square(p_updated - p_f32))
                sq_norm_v = sq_norm_v + jnp.sum(jnp.square(p_f32))

                new_master_p[k] = p_updated.astype(jnp.float32)
                new_params[k] = p_updated.astype(param.dtype)

            else:
                v_prev = opt_s["v"][k].astype(jnp.float32)
                m_new = 0.9 * m_prev + 0.1 * g
                v_new = 0.95 * v_prev + 0.05 * jnp.square(g)
                new_m[k] = m_new.astype(opt_s["m"][k].dtype)
                new_v[k] = v_new.astype(opt_s["v"][k].dtype)

                m_hat = m_new / (1.0 - jnp.power(0.9, cur_step))
                v_hat = v_new / (1.0 - jnp.power(0.95, cur_step))
                step_dir = m_hat / (jnp.sqrt(v_hat) + 1e-6)
                p_updated = p_f32 - lr_1d * step_dir

                # Guard structural coupling scalars and shell gains from collapsing to zero
                if "radial_mix_scale" in k:
                    p_updated = jnp.clip(p_updated, 0.02, 1.0)
                elif "shortcut_mix_scale" in k:
                    p_updated = jnp.clip(p_updated, 0.01, 1.0)
                elif "shell_scale" in k:
                    p_updated = jnp.clip(p_updated, 0.05, 5.0)
                elif "shell_gains" in k:
                    p_updated = jnp.clip(p_updated, 0.02, 2.0)

                sq_diff_1d = sq_diff_1d + jnp.sum(jnp.square(p_updated - p_f32))
                sq_norm_1d = sq_norm_1d + jnp.sum(jnp.square(p_f32))

                new_master_p[k] = p_updated.astype(jnp.float32)
                new_params[k] = p_updated.astype(param.dtype)

            if param_masks is not None and k in param_masks:
                mask = param_masks[k]
                p_updated = jnp.where(mask > 0.0, new_master_p[k], p_f32)
                m_new = jnp.where(mask > 0.0, new_m[k], m_prev)
                new_m[k] = m_new.astype(opt_s["m"][k].dtype)
                if k in opt_s["v"]:
                    v_prev = opt_s["v"][k].astype(jnp.float32)
                    v_new = jnp.where(mask > 0.0, new_v[k], v_prev)
                    new_v[k] = v_new.astype(opt_s["v"][k].dtype)
                new_master_p[k] = p_updated.astype(jnp.float32)
                new_params[k] = p_updated.astype(param.dtype)

        params_finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(np_)) for np_ in new_params.values()]))
        safe_params = jax.tree_util.tree_map(lambda np_, op: jnp.where(params_finite, np_, op), new_params, p)
        safe_m = jax.tree_util.tree_map(lambda nm, om: jnp.where(params_finite, nm, om), new_m, opt_s["m"])
        safe_v = jax.tree_util.tree_map(lambda nv, ov: jnp.where(params_finite, nv, ov), new_v, opt_s["v"])
        safe_step = jnp.where(params_finite, cur_step, opt_s["step"])

        u_ratio_m = jnp.sqrt(sq_diff_m / (sq_norm_m + 1e-8))
        u_ratio_v = jnp.sqrt(sq_diff_v / (sq_norm_v + 1e-8))
        u_ratio_1d = jnp.sqrt(sq_diff_1d / (sq_norm_1d + 1e-8))
        was_clipped = jnp.where(grad_norm > 1.0, 1.0, 0.0)
        was_skipped = jnp.where(params_finite, 0.0, 1.0)

        step_diag = {
            "grad_norm": grad_norm,
            "was_clipped": was_clipped,
            "was_skipped": was_skipped,
            "u_ratio_manifold": jnp.where(params_finite, u_ratio_m, 0.0),
            "u_ratio_vocab": jnp.where(params_finite, u_ratio_v, 0.0),
            "u_ratio_1d": jnp.where(params_finite, u_ratio_1d, 0.0),
            "replica_drift": jnp.array(0.0, dtype=jnp.float32),
            "scmer_loss": scmer_loss,
            "neg_cov_loss": scmer_loss,
        }

        new_opt_state = dict(opt_s)
        new_opt_state.update({
            "step": safe_step,
            "m": safe_m,
            "v": safe_v,
            "power_v": new_power_v,
            "diag": step_diag,
        })
        if has_master:
            new_opt_state["master_p"] = jax.tree_util.tree_map(
                lambda nmp, omp: jnp.where(params_finite, nmp, omp), new_master_p, opt_s["master_p"]
            )
        return safe_params, new_opt_state

    def _skip_step():
        skip_diag = {
            "grad_norm": jnp.where(is_finite, grad_norm, jnp.array(-1.0, dtype=jnp.float32)),
            "was_clipped": jnp.array(0.0, dtype=jnp.float32),
            "was_skipped": jnp.array(1.0, dtype=jnp.float32),
            "u_ratio_manifold": jnp.array(0.0, dtype=jnp.float32),
            "u_ratio_vocab": jnp.array(0.0, dtype=jnp.float32),
            "u_ratio_1d": jnp.array(0.0, dtype=jnp.float32),
            "replica_drift": jnp.array(0.0, dtype=jnp.float32),
            "scmer_loss": scmer_loss,
            "neg_cov_loss": scmer_loss,
        }
        skip_opt_state = dict(opt_s)
        skip_opt_state["diag"] = skip_diag
        return p, skip_opt_state

    return lax.cond(is_finite, _apply_step, _skip_step)


def _is_pure_teacher_forcing(sampling_p: Any, deliberation_mode: str = "teacher_forcing") -> bool:
    if deliberation_mode not in ("teacher_forcing", "pure_tf"):
        return False
    return (not isinstance(sampling_p, jax.Array)) and (float(sampling_p) >= 1.0 - 1e-4)


def build_spheroid_train_engine(
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
    neg_cov_lambda: Optional[float] = None,
    neg_cov_mode: str = "cosine_asymmetric",
    neg_cov_subsample: Optional[int] = None,
    neg_cov_interval: Optional[int] = None,
    neg_cov_margin: Optional[float] = None,
    neg_cov_pair_samples: Optional[int] = None,
    grad_clip_norm: float = 1.0,
    lr_1d_scale: float = 0.2,
) -> Tuple[Callable, Callable]:
    """
    Builds decoupled JIT-compiled gradient computation and optimizer application functions.
    Enables gradient accumulation across batches with zero intermediate Python overhead.
    """
    eff_lambda = neg_cov_lambda if (neg_cov_lambda is not None and scmer_lambda == 0.0) else scmer_lambda
    eff_subsample = neg_cov_subsample if neg_cov_subsample is not None else scmer_subsample
    eff_interval = neg_cov_interval if neg_cov_interval is not None else scmer_interval
    eff_margin = neg_cov_margin if neg_cov_margin is not None else scmer_margin
    eff_pairs = neg_cov_pair_samples if neg_cov_pair_samples is not None else scmer_pair_samples

    @partial(jax.jit, static_argnames=("pure_tf", "deliberation_mode"))
    def _compute_grad_jit(p, q_batch, r_batch, feedback_tokens, sampling_p, pure_tf: bool, deliberation_mode: str, encephalization_alpha: float = 1.0, target_weights: Optional[jnp.ndarray] = None):
        return _compute_loss_and_grads(
            p, q_batch, r_batch, buffers, config, pad_id, repetition_penalty, remat, feedback_tokens, sampling_p, pure_tf, deliberation_mode=deliberation_mode, encephalization_alpha=encephalization_alpha, target_weights=target_weights
        )

    def compute_grad_step(p, q_batch, r_batch, feedback_tokens=None, sampling_p=1.0, deliberation_mode=deliberation_mode, encephalization_alpha=1.0, target_weights=None):
        return _compute_grad_jit(p, q_batch, r_batch, feedback_tokens, sampling_p, _is_pure_teacher_forcing(sampling_p, deliberation_mode), deliberation_mode, encephalization_alpha, target_weights)

    @jax.jit
    def apply_opt_step(p, opt_s, grads, base_lr):
        return _apply_spheroid_grads(
            p, opt_s, grads, base_lr, weight_decay, param_masks=param_masks,
            scmer_lambda=eff_lambda, scmer_subsample=eff_subsample,
            scmer_interval=eff_interval, scmer_margin=eff_margin,
            scmer_pair_samples=eff_pairs,
            grad_clip_norm=grad_clip_norm, lr_1d_scale=lr_1d_scale,
        )

    return compute_grad_step, apply_opt_step


def build_train_step_spheroid(
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
    neg_cov_lambda: Optional[float] = None,
    neg_cov_mode: str = "cosine_asymmetric",
    neg_cov_subsample: Optional[int] = None,
    neg_cov_interval: Optional[int] = None,
    neg_cov_margin: Optional[float] = None,
    neg_cov_pair_samples: Optional[int] = None,
    grad_clip_norm: float = 1.0,
    lr_1d_scale: float = 0.2,
) -> Callable:
    """Builds a single fused JIT-compiled train_step combining forward, backward, and SpheroidOptimizer."""
    eff_lambda = neg_cov_lambda if (neg_cov_lambda is not None and scmer_lambda == 0.0) else scmer_lambda
    eff_subsample = neg_cov_subsample if neg_cov_subsample is not None else scmer_subsample
    eff_interval = neg_cov_interval if neg_cov_interval is not None else scmer_interval
    eff_margin = neg_cov_margin if neg_cov_margin is not None else scmer_margin
    eff_pairs = neg_cov_pair_samples if neg_cov_pair_samples is not None else scmer_pair_samples

    @partial(jax.jit, static_argnames=("pure_tf", "deliberation_mode"))
    def _fused_train_step_jit(p, opt_s, q_batch, r_batch, base_lr, feedback_tokens, sampling_p, pure_tf: bool, deliberation_mode: str, encephalization_alpha: float = 1.0, target_weights: Optional[jnp.ndarray] = None):
        loss_val, grads = _compute_loss_and_grads(
            p, q_batch, r_batch, buffers, config, pad_id, repetition_penalty, remat, feedback_tokens, sampling_p, pure_tf, deliberation_mode=deliberation_mode, encephalization_alpha=encephalization_alpha, target_weights=target_weights
        )
        new_params, new_opt_s = _apply_spheroid_grads(
            p, opt_s, grads, base_lr, weight_decay, param_masks=param_masks,
            scmer_lambda=eff_lambda, scmer_subsample=eff_subsample,
            scmer_interval=eff_interval, scmer_margin=eff_margin,
            scmer_pair_samples=eff_pairs,
            grad_clip_norm=grad_clip_norm, lr_1d_scale=lr_1d_scale,
        )
        return new_params, new_opt_s, loss_val

    def train_step(p, opt_s, q_batch, r_batch, base_lr, feedback_tokens=None, sampling_p=1.0, deliberation_mode=deliberation_mode, encephalization_alpha=1.0, target_weights=None):
        return _fused_train_step_jit(
            p, opt_s, q_batch, r_batch, base_lr, feedback_tokens, sampling_p, _is_pure_teacher_forcing(sampling_p, deliberation_mode), deliberation_mode, encephalization_alpha, target_weights
        )

    return train_step


def merge_spheroid_replicas(
    p: Dict[str, jnp.ndarray],
    opt_s: Dict[str, Any],
    axis_name: str = "dp",
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """
    Synchronizes replicated parameters and optimizer states across TPU cores via pmean,
    preserving recurrent spectral bounds (<= 1.0) under convex averaging.
    """
    has_master = "master_p" in opt_s
    merged_m = jax.tree_util.tree_map(lambda x: lax.pmean(x, axis_name=axis_name), opt_s["m"])
    merged_v = jax.tree_util.tree_map(lambda x: lax.pmean(x, axis_name=axis_name), opt_s["v"])

    merged_master: Dict[str, jnp.ndarray] = {}
    merged_power_v: Dict[str, jnp.ndarray] = {}
    merged_params: Dict[str, jnp.ndarray] = {}

    # Track inter-TPU replica drift before averaging
    sq_drift = 0.0
    sq_norm = 0.0

    for k, param in p.items():
        if not jnp.issubdtype(param.dtype, jnp.inexact):
            merged_params[k] = param
            continue
        m_p = opt_s["master_p"][k] if has_master else param.astype(jnp.float32)
        avg_master = lax.pmean(m_p.astype(jnp.float32), axis_name=axis_name)

        sq_drift = sq_drift + jnp.sum(jnp.square(m_p.astype(jnp.float32) - avg_master))
        sq_norm = sq_norm + jnp.sum(jnp.square(avg_master))

        if k in RECURRENT_KEYS and k in opt_s.get("power_v", {}):
            v_avg = lax.pmean(opt_s["power_v"][k].astype(jnp.float32), axis_name=axis_name)
            v_norm = v_avg / jnp.sqrt(jnp.sum(jnp.square(v_avg), axis=1, keepdims=True) + 1e-6)
            avg_master, v_next = clamp_spectral_norm_jax(avg_master, v_norm, max_spec_norm=1.0)
            merged_power_v[k] = v_next.astype(opt_s["power_v"][k].dtype)
        merged_master[k] = avg_master.astype(jnp.float32)
        merged_params[k] = avg_master.astype(param.dtype)

    local_drift = jnp.sqrt(sq_drift / (sq_norm + 1e-12))
    mean_drift = lax.pmean(local_drift, axis_name=axis_name)

    for k, v_val in opt_s.get("power_v", {}).items():
        if k not in merged_power_v:
            v_avg = lax.pmean(v_val.astype(jnp.float32), axis_name=axis_name)
            v_norm = v_avg / jnp.sqrt(jnp.sum(jnp.square(v_avg), axis=1, keepdims=True) + 1e-6)
            merged_power_v[k] = v_norm.astype(v_val.dtype)

    prev_diag = opt_s.get("diag", {})
    merged_diag = {
        "grad_norm": lax.pmean(prev_diag.get("grad_norm", jnp.array(0.0, dtype=jnp.float32)), axis_name=axis_name),
        "was_clipped": lax.pmean(prev_diag.get("was_clipped", jnp.array(0.0, dtype=jnp.float32)), axis_name=axis_name),
        "was_skipped": lax.pmean(prev_diag.get("was_skipped", jnp.array(0.0, dtype=jnp.float32)), axis_name=axis_name),
        "u_ratio_manifold": lax.pmean(prev_diag.get("u_ratio_manifold", jnp.array(0.0, dtype=jnp.float32)), axis_name=axis_name),
        "u_ratio_vocab": lax.pmean(prev_diag.get("u_ratio_vocab", jnp.array(0.0, dtype=jnp.float32)), axis_name=axis_name),
        "u_ratio_1d": lax.pmean(prev_diag.get("u_ratio_1d", jnp.array(0.0, dtype=jnp.float32)), axis_name=axis_name),
        "replica_drift": mean_drift,
        "scmer_loss": lax.pmean(prev_diag.get("scmer_loss", jnp.array(0.0, dtype=jnp.float32)), axis_name=axis_name),
        "neg_cov_loss": lax.pmean(prev_diag.get("neg_cov_loss", jnp.array(0.0, dtype=jnp.float32)), axis_name=axis_name),
    }
    for k, v in prev_diag.items():
        if k not in merged_diag:
            merged_diag[k] = lax.pmean(v, axis_name=axis_name)

    merged_opt_state = {
        "step": opt_s["step"],
        "m": merged_m,
        "v": merged_v,
        "power_v": merged_power_v,
        "diag": merged_diag,
    }
    if has_master:
        merged_opt_state["master_p"] = merged_master
    return merged_params, merged_opt_state


def replicate_for_pmap(tree: Any, devices: Any) -> Any:
    """Replicate a pytree across `devices` along leading axis 0 for jax.pmap."""
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    dev_list = list(devices)
    n_dev = len(dev_list)
    mesh = Mesh(np.array(dev_list), ("dp",))
    sharding = NamedSharding(mesh, PartitionSpec("dp"))

    def _rep_leaf(x):
        arr = np.asarray(jax.device_get(x))
        return jax.device_put(np.broadcast_to(arr[None, ...], (n_dev,) + arr.shape), sharding)

    return jax.tree_util.tree_map(_rep_leaf, tree)


def build_pmap_local_sgd_step_spheroid(
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
    neg_cov_lambda: Optional[float] = None,
    neg_cov_mode: str = "cosine_asymmetric",
    neg_cov_subsample: Optional[int] = None,
    neg_cov_interval: Optional[int] = None,
    neg_cov_margin: Optional[float] = None,
    neg_cov_pair_samples: Optional[int] = None,
    grad_clip_norm: float = 1.0,
    lr_1d_scale: float = 0.2,
) -> Callable:
    """
    Multi-TPU pmap Local-SGD runner: each TPU replica executes `local_steps` sequential
    SpheroidOptimizer steps on distinct minibatches via lax.scan, then merges replicas with pmean.
    Inputs q_micro and r_micro have shape [num_devices, local_steps, B_per_tpu, L].
    """
    eff_lambda = neg_cov_lambda if (neg_cov_lambda is not None and scmer_lambda == 0.0) else scmer_lambda
    eff_subsample = neg_cov_subsample if neg_cov_subsample is not None else scmer_subsample
    eff_interval = neg_cov_interval if neg_cov_interval is not None else scmer_interval
    eff_margin = neg_cov_margin if neg_cov_margin is not None else scmer_margin
    eff_pairs = neg_cov_pair_samples if neg_cov_pair_samples is not None else scmer_pair_samples

    const_buffers = tuple(jax.device_get(b) for b in buffers)
    compiled_variants: Dict[Tuple[bool, bool], Callable] = {}

    def _get_variant(pure_tf: bool, has_fb: bool, has_weights: bool = False) -> Callable:
        key = (pure_tf, has_fb, has_weights)
        if key in compiled_variants:
            return compiled_variants[key]

        if has_weights:
            in_axes = (0, 0, 0, 0, None, 0 if has_fb else None, None, 0)

            @partial(jax.pmap, axis_name="dp", in_axes=in_axes, devices=devices)
            def _pmap_round_weighted(p, opt_s, q_micro, r_micro, base_lr, fb_micro, sampling_p, wt_micro):
                jax_buffers = tuple(jnp.asarray(b) for b in const_buffers)

                def _scan_body(carry, batch_slice):
                    cur_p, cur_opt = carry
                    if has_fb:
                        q_b, r_b, fb_b, wt_b = batch_slice
                    else:
                        q_b, r_b, wt_b = batch_slice
                        fb_b = None
                    loss_k, grads_k = _compute_loss_and_grads(
                        cur_p,
                        q_b,
                        r_b,
                        jax_buffers,
                        config,
                        pad_id,
                        repetition_penalty,
                        remat,
                        fb_b,
                        sampling_p,
                        pure_tf,
                        target_weights=wt_b,
                    )
                    nxt_p, nxt_opt = _apply_spheroid_grads(
                        cur_p, cur_opt, grads_k, base_lr, weight_decay,
                        scmer_lambda=eff_lambda, scmer_subsample=eff_subsample,
                        scmer_interval=eff_interval, scmer_margin=eff_margin,
                        scmer_pair_samples=eff_pairs,
                    )
                    return (nxt_p, nxt_opt), loss_k

                scan_xs = (q_micro, r_micro, fb_micro, wt_micro) if has_fb else (q_micro, r_micro, wt_micro)
                (end_p, end_opt), losses = lax.scan(_scan_body, (p, opt_s), scan_xs)
                merged_p, merged_opt = merge_spheroid_replicas(end_p, end_opt, axis_name="dp")
                mean_loss = lax.pmean(jnp.mean(losses), axis_name="dp")
                return merged_p, merged_opt, mean_loss

            compiled_variants[key] = _pmap_round_weighted
            return _pmap_round_weighted
        else:
            in_axes = (0, 0, 0, 0, None, 0 if has_fb else None, None)

            @partial(jax.pmap, axis_name="dp", in_axes=in_axes, devices=devices)
            def _pmap_round(p, opt_s, q_micro, r_micro, base_lr, fb_micro, sampling_p):
                jax_buffers = tuple(jnp.asarray(b) for b in const_buffers)

                def _scan_body(carry, batch_slice):
                    cur_p, cur_opt = carry
                    if has_fb:
                        q_b, r_b, fb_b = batch_slice
                    else:
                        q_b, r_b = batch_slice
                        fb_b = None
                    loss_k, grads_k = _compute_loss_and_grads(
                        cur_p,
                        q_b,
                        r_b,
                        jax_buffers,
                        config,
                        pad_id,
                        repetition_penalty,
                        remat,
                        fb_b,
                        sampling_p,
                        pure_tf,
                    )
                    nxt_p, nxt_opt = _apply_spheroid_grads(
                        cur_p, cur_opt, grads_k, base_lr, weight_decay,
                        scmer_lambda=eff_lambda, scmer_subsample=eff_subsample,
                        scmer_interval=eff_interval, scmer_margin=eff_margin,
                        scmer_pair_samples=eff_pairs,
                    )
                    return (nxt_p, nxt_opt), loss_k

                scan_xs = (q_micro, r_micro, fb_micro) if has_fb else (q_micro, r_micro)
                (end_p, end_opt), losses = lax.scan(_scan_body, (p, opt_s), scan_xs)
                merged_p, merged_opt = merge_spheroid_replicas(end_p, end_opt, axis_name="dp")
                mean_loss = lax.pmean(jnp.mean(losses), axis_name="dp")
                return merged_p, merged_opt, mean_loss

            compiled_variants[key] = _pmap_round
            return _pmap_round

    def train_round(p, opt_s, q_micro, r_micro, base_lr, feedback_micro=None, sampling_p=1.0, target_weights_micro=None):
        has_weights = target_weights_micro is not None
        fn = _get_variant(_is_pure_teacher_forcing(sampling_p), feedback_micro is not None, has_weights)
        if has_weights:
            return fn(p, opt_s, q_micro, r_micro, base_lr, feedback_micro, sampling_p, target_weights_micro)
        return fn(p, opt_s, q_micro, r_micro, base_lr, feedback_micro, sampling_p)

    return train_round
