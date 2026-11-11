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
    Runs strictly in float32 to prevent half-precision exponent overflow."""
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

    norm = jnp.sqrt(jnp.sum(jnp.square(G_mat), axis=(-2, -1), keepdims=True) + eps)
    X = G_mat / norm

    # Quintic Newton-Schulz polynomial coefficients (static unroll improves XLA pipelining)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        A = jnp.matmul(X, jnp.swapaxes(X, -2, -1))
        B = b * A + c * jnp.matmul(A, A)
        X = a * X + jnp.matmul(B, X)

    if transposed:
        X = jnp.swapaxes(X, -2, -1)
    return X.reshape(orig_shape).astype(orig_dtype)


def clamp_spectral_norm_jax(
    W: jnp.ndarray, v: jnp.ndarray, max_spec_norm: float = 1.0, power_steps: int = 2, eps: float = 1e-6
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
    scale = jnp.minimum(1.0, max_spec_norm / (sigma * 1.08 + eps))
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
        )

    return jax.value_and_grad(_loss_fn, allow_int=True)(p)


def _apply_spheroid_grads(
    p: Dict[str, jnp.ndarray],
    opt_s: Dict[str, Any],
    grads: Dict[str, jnp.ndarray],
    base_lr: Any,
    weight_decay: float,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """Applies Polar Newton-Schulz, recurrent spectral clamp, and float32 master-weight AdamW."""
    grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for g in grads.values() if jnp.issubdtype(g.dtype, jnp.inexact))
    )
    is_finite = jnp.isfinite(grad_norm)
    clip_scale = jnp.where(is_finite, jnp.minimum(1.0, 1.0 / (grad_norm + 1e-6)), 0.0)
    clipped_grads = jax.tree_util.tree_map(lambda g: g * clip_scale if jnp.issubdtype(g.dtype, jnp.inexact) else g, grads)

    def _apply_step():
        cur_step = opt_s["step"] + 1
        has_master = "master_p" in opt_s
        new_params = {}
        new_m = {}
        new_v = {}
        new_master_p = {}
        new_power_v = dict(opt_s["power_v"])

        lr_1d = 0.2 * base_lr
        lr_vocab = jnp.minimum(base_lr, 6.0e-4)

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

                p_decayed = p_f32 * (1.0 - base_lr * weight_decay)
                O = zeropower_via_newtonschulz5_jax(m_new, steps=3).astype(jnp.float32)

                rows = param.shape[-2]
                cols = param.shape[-1]
                aspect_scale = math.sqrt(max(1.0, float(rows) / float(cols)))
                p_updated = p_decayed - (base_lr * aspect_scale) * O

                if k in RECURRENT_KEYS:
                    p_updated, v_next = clamp_spectral_norm_jax(p_updated, opt_s["power_v"][k], max_spec_norm=1.0)
                    new_power_v[k] = v_next

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
                new_master_p[k] = p_updated.astype(jnp.float32)
                new_params[k] = p_updated.astype(param.dtype)

        params_finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(np_)) for np_ in new_params.values()]))
        safe_params = jax.tree_util.tree_map(lambda np_, op: jnp.where(params_finite, np_, op), new_params, p)
        safe_m = jax.tree_util.tree_map(lambda nm, om: jnp.where(params_finite, nm, om), new_m, opt_s["m"])
        safe_v = jax.tree_util.tree_map(lambda nv, ov: jnp.where(params_finite, nv, ov), new_v, opt_s["v"])
        safe_step = jnp.where(params_finite, cur_step, opt_s["step"])

        new_opt_state = {
            "step": safe_step,
            "m": safe_m,
            "v": safe_v,
            "power_v": new_power_v,
        }
        if has_master:
            new_opt_state["master_p"] = jax.tree_util.tree_map(
                lambda nmp, omp: jnp.where(params_finite, nmp, omp), new_master_p, opt_s["master_p"]
            )
        return safe_params, new_opt_state

    def _skip_step():
        return p, opt_s

    return lax.cond(is_finite, _apply_step, _skip_step)


def _is_pure_teacher_forcing(sampling_p: Any) -> bool:
    return (not isinstance(sampling_p, jax.Array)) and (float(sampling_p) >= 1.0)


def build_spheroid_train_engine(
    config: ModelConfigV5,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    pad_id: int = 0,
    weight_decay: float = 0.01,
    repetition_penalty: float = 1.25,
    remat: bool = True,
) -> Tuple[Callable, Callable]:
    """
    Builds decoupled JIT-compiled gradient computation and optimizer application functions.
    Enables gradient accumulation across batches with zero intermediate Python overhead.
    """

    @partial(jax.jit, static_argnames=("pure_tf",))
    def _compute_grad_jit(p, q_batch, r_batch, feedback_tokens, sampling_p, pure_tf: bool):
        return _compute_loss_and_grads(
            p, q_batch, r_batch, buffers, config, pad_id, repetition_penalty, remat, feedback_tokens, sampling_p, pure_tf
        )

    def compute_grad_step(p, q_batch, r_batch, feedback_tokens=None, sampling_p=1.0):
        return _compute_grad_jit(p, q_batch, r_batch, feedback_tokens, sampling_p, _is_pure_teacher_forcing(sampling_p))

    @jax.jit
    def apply_opt_step(p, opt_s, grads, base_lr):
        return _apply_spheroid_grads(p, opt_s, grads, base_lr, weight_decay)

    return compute_grad_step, apply_opt_step


def build_train_step_spheroid(
    config: ModelConfigV5,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    pad_id: int = 0,
    weight_decay: float = 0.01,
    repetition_penalty: float = 1.25,
    remat: bool = True,
) -> Callable:
    """Builds a single fused JIT-compiled train_step combining forward, backward, and SpheroidOptimizer."""

    @partial(jax.jit, static_argnames=("pure_tf",))
    def _fused_train_step_jit(p, opt_s, q_batch, r_batch, base_lr, feedback_tokens, sampling_p, pure_tf: bool):
        loss_val, grads = _compute_loss_and_grads(
            p, q_batch, r_batch, buffers, config, pad_id, repetition_penalty, remat, feedback_tokens, sampling_p, pure_tf
        )
        new_params, new_opt_s = _apply_spheroid_grads(p, opt_s, grads, base_lr, weight_decay)
        return new_params, new_opt_s, loss_val

    def train_step(p, opt_s, q_batch, r_batch, base_lr, feedback_tokens=None, sampling_p=1.0):
        return _fused_train_step_jit(
            p, opt_s, q_batch, r_batch, base_lr, feedback_tokens, sampling_p, _is_pure_teacher_forcing(sampling_p)
        )

    return train_step
