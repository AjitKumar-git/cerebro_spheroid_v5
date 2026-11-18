"""
CerebroSpheroid V5: Pure JAX High-Throughput Engine
===================================================

Native Cloud TPU & GPU implementation:
1. Affine-Tied Sensory-Motor Dictionary: 89.6% active compute parameters.
2. 32 Concentric Laminae with SwiGLU intra-shell and radial convection.
3. 8-Head Multi-Head Arcuate Fasciculus cross-resonance.
4. Scaled Phase Injection (p_scaled = 2.0).
5. Quintic Polar Newton-Schulz spectral radius guard (rho <= 1.0).
6. Pure functional scan loops for 0ms compilation overhead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax


@dataclass
class ModelConfigV5:
    """Architectural hyperparameters for CerebroSpheroid V5 JAX execution."""
    vocab_size: int = 32000
    surface_dim: int = 768
    radial_dim: int = 512
    num_shells: int = 32
    num_shortcut_tracts: int = 96
    num_heads: int = 8
    # New runs use a 4k output-position horizon.  RoPE carries relative
    # position beyond this learned phase table, so this is no longer a claimed
    # 32k capacity backed by a 1k-only clock.
    max_phase_steps: int = 4096
    sensory_position_steps: int = 1024
    thinking_cycles: int = 4
    phase_scale: float = 2.0
    use_tied_embeddings: bool = True
    use_swiglu: bool = True
    use_shell_tap: bool = True
    use_plan: bool = True
    use_sensory_self_attn: bool = True
    num_sensory_layers: int = 1
    use_rope: bool = True
    rope_theta: float = 500000.0
    max_seq_len: int = 32768
    top_k_resonance: int = 16
    use_deliberation_probe: bool = True
    use_adaptive_thinking: bool = True
    max_thinking_cycles: int = 8
    thinking_halt_threshold: float = 0.75
    num_mtp_heads: int = 1
    use_sparse_mot: bool = True
    top_k_tracts: int = 24
    use_energy_regularization: bool = True
    enable_multimodal: bool = True
    visual_dim: int = 512
    audio_dim: int = 512
    num_visual_patches: int = 16
    image_patch_size: int = 16


# Register ModelConfigV5 as static PyTree node
jax.tree_util.register_pytree_node(
    ModelConfigV5,
    lambda c: ((), (
        c.vocab_size, c.surface_dim, c.radial_dim, c.num_shells, c.num_shortcut_tracts,
        c.num_heads, c.max_phase_steps, c.sensory_position_steps, c.thinking_cycles, c.phase_scale,
        c.use_tied_embeddings, c.use_swiglu, c.use_shell_tap, c.use_plan, c.use_sensory_self_attn, c.num_sensory_layers, c.use_rope,
        c.rope_theta, c.max_seq_len, c.top_k_resonance, c.use_deliberation_probe,
        c.use_adaptive_thinking, c.max_thinking_cycles, c.thinking_halt_threshold,
        c.num_mtp_heads, c.use_sparse_mot, c.top_k_tracts, c.use_energy_regularization,
        c.enable_multimodal, c.visual_dim, c.audio_dim, c.num_visual_patches, c.image_patch_size,
    )),
    lambda aux, _: ModelConfigV5(*aux),
)


PRESETS_V5: Dict[str, Dict[str, Any]] = {
    "v5_small": {
        "surface_dim": 640,
        "radial_dim": 256,
        "num_shells": 16,
        "num_shortcut_tracts": 40,
        "num_heads": 8,
        "top_k_tracts": 16,
        "thinking_cycles": 4,
        "phase_scale": 2.0,
        "use_sensory_self_attn": True,
        "use_sparse_mot": False,
        "use_deliberation_probe": False,
        "use_adaptive_thinking": False,
        "use_energy_regularization": False,
    },
    "tpu_v5e_1": {
        "surface_dim": 640,
        "radial_dim": 256,
        "num_shells": 16,
        "num_shortcut_tracts": 40,
        "num_heads": 8,
        "top_k_tracts": 16,
        "thinking_cycles": 4,
        "phase_scale": 2.0,
        "use_sensory_self_attn": True,
        "use_sparse_mot": False,
        "use_deliberation_probe": False,
        "use_adaptive_thinking": False,
        "use_energy_regularization": False,
    },
    "tpu_v6e_1": {
        "surface_dim": 640,
        "radial_dim": 256,
        "num_shells": 16,
        "num_shortcut_tracts": 40,
        "num_heads": 8,
        "top_k_tracts": 16,
        "thinking_cycles": 4,
        "phase_scale": 2.0,
        "use_sensory_self_attn": True,
        "use_sparse_mot": False,
        "use_deliberation_probe": False,
        "use_adaptive_thinking": False,
        "use_energy_regularization": False,
    },
    "tpu_v6e": {
        "surface_dim": 640,
        "radial_dim": 256,
        "num_shells": 16,
        "num_shortcut_tracts": 40,
        "num_heads": 8,
        "top_k_tracts": 16,
        "thinking_cycles": 4,
        "phase_scale": 2.0,
        "use_sensory_self_attn": True,
        "use_sparse_mot": False,
        "use_deliberation_probe": False,
        "use_adaptive_thinking": False,
        "use_energy_regularization": False,
    },
    "v5_compact_32shells": {
        "surface_dim": 640,
        "radial_dim": 256,
        "num_shells": 32,
        "num_shortcut_tracts": 96,
        "num_heads": 8,
        "top_k_tracts": 24,
        "thinking_cycles": 4,
        "phase_scale": 2.0,
        "use_sensory_self_attn": True,
        "use_sparse_mot": False,
        "use_deliberation_probe": False,
        "use_adaptive_thinking": False,
        "use_energy_regularization": False,
        "use_tied_embeddings": True,
        "use_swiglu": True,
    },
    "v5_compact": {
        "surface_dim": 768,
        "radial_dim": 512,
        "num_shells": 32,
        "num_shortcut_tracts": 96,
        "num_heads": 8,
        "thinking_cycles": 4,
        "phase_scale": 2.0,
    },
    "v5_base": {
        "surface_dim": 1024,
        "radial_dim": 768,
        "num_shells": 48,
        "num_shortcut_tracts": 160,
        "num_heads": 16,
        "thinking_cycles": 4,
        "phase_scale": 2.0,
    },
    "v5_test": {
        "vocab_size": 2048,
        "surface_dim": 128,
        "radial_dim": 64,
        "num_shells": 8,
        "num_shortcut_tracts": 8,
        "num_heads": 4,
        "max_phase_steps": 128,
        "thinking_cycles": 2,
        "phase_scale": 2.0,
    },
}


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Root Mean Square layer normalization in JAX (accumulates variance in float32)."""
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(variance + eps)
    return (normed * weight.astype(jnp.float32)).astype(orig_dtype)


def swiglu(x: jnp.ndarray, w_gate: jnp.ndarray, w_val: jnp.ndarray, w_down: jnp.ndarray) -> jnp.ndarray:
    """Swish Gated Linear Unit in JAX."""
    g = jax.nn.silu(jnp.matmul(x, w_gate.T))
    v = jnp.matmul(x, w_val.T)
    return jnp.matmul(g * v, w_down.T)


def apply_rope(
    x: jnp.ndarray,
    positions: Union[int, jnp.ndarray],
    theta: float = 500000.0,
) -> jnp.ndarray:
    """Apply rotary position embeddings to ``[B, H, L, D]`` attention states.

    Frequencies are computed from positions rather than a finite lookup cache,
    so a new run can train past the learned absolute-position table without
    modulo collisions.
    """
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError(f"RoPE head dimension must be even, got {dim}")
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    angles = jnp.asarray(positions, dtype=jnp.float32)[..., None] * inv_freq
    cos = jnp.cos(angles).astype(x.dtype)
    sin = jnp.sin(angles).astype(x.dtype)
    # Positions are scalar for one-token decoding and [L] for block training.
    cos = cos.reshape((1,) * (x.ndim - 2) + cos.shape)
    sin = sin.reshape((1,) * (x.ndim - 2) + sin.shape)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    return jnp.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), axis=-1).reshape(x.shape)


def generate_small_world_indices(num_shells: int, num_tracts: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generates deterministic small-world axonal shortcut wiring in JAX."""
    srcs, dsts = [], []
    for i in range(num_tracts):
        if i % 3 == 0:
            s = i % max(1, num_shells // 4)
            d = num_shells - 1 - (i % max(1, num_shells // 4))
        elif i % 3 == 1:
            s = (i * 2) % num_shells
            d = (s + num_shells // 2) % num_shells
        else:
            s = (i * 7) % num_shells
            d = (i * 13) % num_shells
            if s == d:
                d = (s + 1) % num_shells
        srcs.append(s)
        dsts.append(d)
    return jnp.array(srcs, dtype=jnp.int32), jnp.array(dsts, dtype=jnp.int32)


def init_params_v5(key: jax.random.PRNGKey, config: ModelConfigV5) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Initializes CerebroSpheroid V5 parameters and small-world tract buffers."""
    keys = jax.random.split(key, 40)
    s_dim = config.surface_dim
    r_dim = config.radial_dim
    K = config.num_shells
    V = config.vocab_size
    T = config.num_shortcut_tracts

    params: Dict[str, jnp.ndarray] = {
        # 1. Sensory DNN (Shared embedding table + zero-init positional embedding)
        "sensory.embedding.weight": jax.random.normal(keys[0], (V, s_dim)) * 0.02,
        "sensory.pos_embed.weight": jnp.zeros((config.sensory_position_steps, s_dim), dtype=jnp.float32),
        "sensory.swiglu.fc_gate.weight": jax.random.normal(keys[1], (s_dim * 2, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "sensory.swiglu.fc_val.weight": jax.random.normal(keys[2], (s_dim * 2, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "sensory.swiglu.fc_down.weight": jax.random.normal(keys[3], (s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2)),
        "sensory.norm_in.weight": jnp.ones(s_dim, dtype=jnp.float32),
        "sensory.norm_out.weight": jnp.ones(s_dim, dtype=jnp.float32),
        "sensory.k_proj.weight": jax.random.normal(keys[4], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "sensory.v_proj.weight": jax.random.normal(keys[5], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        # Contextual Inter-Token Multi-Head Self-Attention (zero-initialized residual for lossless weight transfer)
        "sensory.ctx_norm.weight": jnp.ones(s_dim, dtype=jnp.float32),
        "sensory.q_self.weight": jax.random.normal(keys[34], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "sensory.k_self.weight": jax.random.normal(keys[35], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "sensory.v_self.weight": jax.random.normal(keys[36], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "sensory.out_self.weight": jnp.zeros((s_dim, s_dim), dtype=jnp.float32),

        # 2. Volumetric Core (concentric spherical shells)
        "core.shell_gains": jnp.ones(K, dtype=jnp.float32) * 0.1,
        "core.radial_mix_scale": jnp.array(0.1, dtype=jnp.float32),
        "core.shortcut_mix_scale": jnp.array(0.05, dtype=jnp.float32),
        "core.radial_in_proj.weight": jax.random.normal(keys[6], (r_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "core.radial_out_proj.weight": jax.random.normal(keys[7], (r_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "core.radial_up.weight": jax.random.normal(keys[8], (s_dim, r_dim)) * (1.0 / math.sqrt(r_dim)),
        "core.radial_up_out.weight": jax.random.normal(keys[27], (s_dim, r_dim)) * (1.0 / math.sqrt(r_dim)),
        "core.radial_norm.weight": jnp.ones(s_dim, dtype=jnp.float32),
        "core.w_radial_in": jax.random.normal(keys[9], (K - 1, r_dim, r_dim)) * (1.0 / math.sqrt(r_dim)),
        "core.w_radial_out": jax.random.normal(keys[10], (K - 1, r_dim, r_dim)) * (1.0 / math.sqrt(r_dim)),
        "core.shortcut_weight": jax.random.normal(keys[11], (T, s_dim, s_dim)) * (0.05 / math.sqrt(s_dim)),
        "core.shortcut_gate": jnp.zeros(T, dtype=jnp.float32),
        "core.efference_gate.weight": jax.random.normal(keys[12], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),

        # 3. Premotor Motor Decoder with Affine-Tied Head
        "motor.phase_embed.weight": jax.random.normal(keys[13], (config.max_phase_steps, s_dim)) * 0.02,
        "motor.phase_scale": jnp.array(config.phase_scale, dtype=jnp.float32),
        "motor.q_proj.weight": jax.random.normal(keys[14], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "motor.out_proj.weight": jax.random.normal(keys[15], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "motor.resonance_norm.weight": jnp.ones(s_dim, dtype=jnp.float32),
        "motor.shell_score": jnp.zeros(K, dtype=jnp.float32),
        "motor.shell_proj.weight": jax.random.normal(keys[16], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "motor.shell_scale": jnp.array(0.5, dtype=jnp.float32),

        # Deep motor SwiGLU fusion (5*D -> D) with per-source gating
        "motor.input_gate": jnp.ones(5, dtype=jnp.float32),
        "motor.fused_norm.weight": jnp.ones(s_dim * 5, dtype=jnp.float32),
        "motor.swiglu.fc_gate.weight": jax.random.normal(keys[17], (s_dim * 2, s_dim * 5)) * (1.0 / math.sqrt(s_dim * 5)),
        "motor.swiglu.fc_val.weight": jax.random.normal(keys[18], (s_dim * 2, s_dim * 5)) * (1.0 / math.sqrt(s_dim * 5)),
        "motor.swiglu.fc_down.weight": jax.random.normal(keys[19], (s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2)),
        "motor.norm_mid.weight": jnp.ones(s_dim, dtype=jnp.float32),

        # Recurrent Plan
        "motor.plan_init.weight": jax.random.normal(keys[20], (s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2)),
        "motor.plan_init.bias": jnp.zeros(s_dim, dtype=jnp.float32),
        "motor.plan_gate.weight": jax.random.normal(keys[21], (s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2)),
        "motor.plan_gate.bias": jnp.zeros(s_dim, dtype=jnp.float32),
        "motor.plan_update.weight": jax.random.normal(keys[22], (s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2)),
        "motor.plan_decay": jnp.array(0.85, dtype=jnp.float32),
        "motor.plan_write": jnp.array(0.15, dtype=jnp.float32),

        # Affine-Tied Emission Head
        "motor.affine_scale": jnp.ones(s_dim, dtype=jnp.float32),
        # Tri-Head Affine Scaling Parameters (Simultaneous 3-Tier Heads)
        "motor.affine_think": jnp.ones(s_dim, dtype=jnp.float32),
        "motor.affine_act": jnp.ones(s_dim, dtype=jnp.float32),
        "motor.affine_answer": jnp.ones(s_dim, dtype=jnp.float32),
        # Temporal Output Router (Decides when to activate Think, Act, Answer)
        "motor.temporal_router.weight": jax.random.normal(keys[28] if len(keys) > 28 else keys[0], (3, s_dim)) * 0.01,
        "motor.temporal_router.bias": jnp.array([0.5, -0.8, 1.0], dtype=jnp.float32),
        "motor.temperature_scale": jnp.array(1.0, dtype=jnp.float32),
        "motor.efference_proj.weight": jax.random.normal(keys[23], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "motor.refractory_decay": jnp.array(0.75, dtype=jnp.float32),
        "motor.refractory_strength": jnp.array(1.2, dtype=jnp.float32),

        # Persona Steering Basis & Scale
        "motor.persona_embeddings": jax.random.normal(keys[27], (4, s_dim)) * 0.05,
        "motor.persona_scale": jnp.array(0.5, dtype=jnp.float32),
    }

    # 3D Tensorized Batched Intra-Shell SwiGLU across all K shells in parallel
    params["core.intra_norms.weight"] = jnp.ones((K, s_dim), dtype=jnp.float32)
    params["core.intra_swiglu.w_gate"] = jax.random.normal(keys[24], (K, s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim))
    params["core.intra_swiglu.w_val"] = jax.random.normal(keys[25], (K, s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim))
    params["core.intra_swiglu.w_down"] = jax.random.normal(keys[26], (K, s_dim * 2, s_dim)) * (1.0 / math.sqrt(s_dim * 2))

    # 4. Multimodal Perceptual Projections (Images, Audio, Continuous Sensors)
    if getattr(config, "enable_multimodal", False):
        v_dim = getattr(config, "visual_dim", s_dim)
        p_size = getattr(config, "image_patch_size", 16)
        params["image_stem.conv.weight"] = jax.random.normal(keys[28], (v_dim, 3, p_size, p_size)) * (1.0 / math.sqrt(3 * p_size * p_size))
        params["image_stem.norm.weight"] = jnp.ones(v_dim, dtype=jnp.float32)

        params["audio_stem.conv.weight"] = jax.random.normal(keys[29], (v_dim, 1, 400)) * (1.0 / math.sqrt(400))
        params["audio_stem.norm.weight"] = jnp.ones(v_dim, dtype=jnp.float32)

        params["perceptual.norm_in.weight"] = jnp.ones(v_dim, dtype=jnp.float32)
        params["perceptual.linear.weight"] = jax.random.normal(keys[30], (s_dim, v_dim)) * (1.0 / math.sqrt(v_dim))
        params["perceptual.swiglu.fc_gate.weight"] = jax.random.normal(keys[31], (s_dim * 2, s_dim)) * (1.0 / math.sqrt(s_dim))
        params["perceptual.swiglu.fc_val.weight"] = jax.random.normal(keys[32], (s_dim * 2, s_dim)) * (1.0 / math.sqrt(s_dim))
        params["perceptual.swiglu.fc_down.weight"] = jax.random.normal(keys[33], (s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2))
        params["perceptual.norm_out.weight"] = jnp.ones(s_dim, dtype=jnp.float32)

    num_sens_layers = getattr(config, "num_sensory_layers", 1)
    if num_sens_layers > 1:
        extra_keys = jax.random.split(keys[34], (num_sens_layers - 1) * 6)
        for idx in range(num_sens_layers - 1):
            pfx = f"sensory_dnn.extra_layers.{idx}."
            k_base = idx * 6
            params[f"{pfx}norm1.weight"] = jnp.ones(s_dim, dtype=jnp.float32)
            params[f"{pfx}q_proj.weight"] = jax.random.normal(extra_keys[k_base], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim))
            params[f"{pfx}k_proj.weight"] = jax.random.normal(extra_keys[k_base + 1], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim))
            params[f"{pfx}v_proj.weight"] = jax.random.normal(extra_keys[k_base + 2], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim))
            params[f"{pfx}out_proj.weight"] = jnp.zeros((s_dim, s_dim), dtype=jnp.float32)
            params[f"{pfx}norm2.weight"] = jnp.ones(s_dim, dtype=jnp.float32)
            params[f"{pfx}swiglu.fc_gate.weight"] = jax.random.normal(extra_keys[k_base + 3], (s_dim * 2, s_dim)) * (1.0 / math.sqrt(s_dim))
            params[f"{pfx}swiglu.fc_val.weight"] = jax.random.normal(extra_keys[k_base + 4], (s_dim * 2, s_dim)) * (1.0 / math.sqrt(s_dim))
            params[f"{pfx}swiglu.fc_down.weight"] = jax.random.normal(extra_keys[k_base + 5], (s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2))

    # 5. Dynamic Halting Gate for Adaptive Deliberation (System-2 Reasoning)
    if getattr(config, "use_adaptive_thinking", False):
        params["core.halting_gate.weight"] = jnp.zeros((1, s_dim), dtype=jnp.float32)
        params["core.halting_gate.bias"] = jnp.array([-1.5], dtype=jnp.float32)

    src_idx, dst_idx = generate_small_world_indices(K, T)
    buffers = (src_idx, dst_idx)
    return params, buffers


def encode_image_stem(
    params: Dict[str, jnp.ndarray],
    images: jnp.ndarray,
    patch_size: int = 16,
) -> jnp.ndarray:
    """
    2D Convolutional Patchify Stem:
    images: [B, 3, H, W] -> [B, P, visual_dim]
    """
    w_conv = params["image_stem.conv.weight"]  # [v_dim, 3, p_size, p_size] (OIHW)
    out = lax.conv_general_dilated(
        lhs=images,
        rhs=w_conv,
        window_strides=(patch_size, patch_size),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    B, C, H, W = out.shape
    h = out.reshape(B, C, H * W).swapaxes(1, 2)  # [B, P, v_dim]
    return rms_norm(h, params["image_stem.norm.weight"])


def encode_audio_stem(
    params: Dict[str, jnp.ndarray],
    audio: jnp.ndarray,
    kernel_size: int = 400,
    stride: int = 160,
) -> jnp.ndarray:
    """
    1D Convolutional Acoustic Stem:
    audio: [B, T_samples] or [B, 1, T_samples] -> [B, T_frames, audio_dim]
    """
    if audio.ndim == 2:
        audio = audio[:, None, :]  # [B, 1, T]
    w_conv = params["audio_stem.conv.weight"]  # [audio_dim, 1, kernel_size] (OIW)
    pad = kernel_size // 2
    out = lax.conv_general_dilated(
        lhs=audio,
        rhs=w_conv,
        window_strides=(stride,),
        padding=[(pad, pad)],
        dimension_numbers=("NCW", "OIW", "NCW"),
    )
    h = out.swapaxes(1, 2)  # [B, T_frames, audio_dim]
    return rms_norm(h, params["audio_stem.norm.weight"])


def project_perceptual(
    params: Dict[str, jnp.ndarray],
    perceptual_flux: jnp.ndarray,
) -> jnp.ndarray:
    """
    Projects raw perceptual tokens [B, P_total, v_dim] into surface_dim space:
    RMSNorm(v_dim) -> Linear(v_dim, s_dim) -> SwiGLU(s_dim) -> RMSNorm(s_dim)
    """
    h = rms_norm(perceptual_flux, params["perceptual.norm_in.weight"])
    h_lin = jnp.matmul(h, params["perceptual.linear.weight"].T)
    h_swiglu = swiglu(
        h_lin,
        params["perceptual.swiglu.fc_gate.weight"],
        params["perceptual.swiglu.fc_val.weight"],
        params["perceptual.swiglu.fc_down.weight"],
    )
    return rms_norm(h_swiglu, params["perceptual.norm_out.weight"])


def encode_sensory(
    params: Dict[str, jnp.ndarray],
    input_tokens_or_emb: jnp.ndarray,
    config: Optional[ModelConfigV5] = None,
    pos_offset: Union[int, jnp.ndarray] = 0,
    is_causal: bool = False,
    attn_mask: Optional[jnp.ndarray] = None,
):
    """Encodes sensory input tokens into Arcuate KV pairs and position-aware representations."""
    w_emb = params["sensory.embedding.weight"]
    if input_tokens_or_emb.dtype in (jnp.int32, jnp.int64):
        x = w_emb[input_tokens_or_emb]
    else:
        x = input_tokens_or_emb

    is_2d = x.ndim == 2
    if is_2d:
        x = x[:, None, :]

    B, L, D = x.shape
    num_heads = config.num_heads if config is not None else 8
    head_dim = D // num_heads

    # Additive zero-init positional encoding (preserves legacy weights identically at step 0 of transfer)
    if "sensory.pos_embed.weight" in params:
        pos_tbl = params["sensory.pos_embed.weight"]
        max_p = pos_tbl.shape[0]
        raw_pos = jnp.arange(L) + pos_offset
        # Do not alias position 0 with 1024, 2048, ... .  RoPE supplies
        # position information outside this learned-table training range.
        pos_ids = jnp.clip(raw_pos, 0, max_p - 1)
        in_table = (raw_pos < max_p).astype(pos_tbl.dtype)[:, None]
        pos_enc = pos_tbl[pos_ids] * in_table
        x = x + pos_enc[None, :, :].astype(x.dtype)

    h = rms_norm(x, params["sensory.norm_in.weight"])
    h_swiglu = swiglu(h, params["sensory.swiglu.fc_gate.weight"], params["sensory.swiglu.fc_val.weight"], params["sensory.swiglu.fc_down.weight"])
    flux = rms_norm(x + h_swiglu, params["sensory.norm_out.weight"])

    # Contextual Inter-Token Multi-Head Self-Attention (zero-init residual)
    if L > 1 and "sensory.out_self.weight" in params:
        h_ctx = rms_norm(flux, params["sensory.ctx_norm.weight"])
        q_s = jnp.matmul(h_ctx, params["sensory.q_self.weight"].T).reshape(B, L, num_heads, head_dim)
        q_s = jnp.swapaxes(q_s, 1, 2)  # [B, H, L, head_dim]
        k_s = jnp.matmul(h_ctx, params["sensory.k_self.weight"].T).reshape(B, L, num_heads, head_dim)
        k_s = jnp.swapaxes(k_s, 1, 2)  # [B, H, L, head_dim]
        v_s = jnp.matmul(h_ctx, params["sensory.v_self.weight"].T).reshape(B, L, num_heads, head_dim)
        v_s = jnp.swapaxes(v_s, 1, 2)  # [B, H, L, head_dim]

        if config is not None and getattr(config, "use_rope", True):
            sens_pos = jnp.arange(L) + pos_offset
            q_s = apply_rope(q_s, sens_pos, theta=config.rope_theta)
            k_s = apply_rope(k_s, sens_pos, theta=config.rope_theta)

        scores = jnp.matmul(q_s, jnp.swapaxes(k_s, -1, -2)) / math.sqrt(head_dim)
        if attn_mask is not None:
            scores = jnp.where(attn_mask, scores, -1e4)
        elif is_causal:
            causal_mask = jnp.tril(jnp.ones((L, L), dtype=bool))[None, None, :, :]
            scores = jnp.where(causal_mask, scores, -1e4)
        attn_w = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(flux.dtype)
        ctx = jnp.matmul(attn_w, v_s)  # [B, H, L, head_dim]
        ctx = jnp.swapaxes(ctx, 1, 2).reshape(B, L, D)  # [B, L, D]
        flux = flux + jnp.matmul(ctx, params["sensory.out_self.weight"].T)

    # Stacked extra sensory layers (if num_sensory_layers > 1)
    num_sens_layers = getattr(config, "num_sensory_layers", 1) if config is not None else 1
    if L > 1 and num_sens_layers > 1:
        for idx in range(num_sens_layers - 1):
            pfx = f"sensory_dnn.extra_layers.{idx}."
            if f"{pfx}norm1.weight" not in params:
                pfx = f"sensory.extra_layers.{idx}."
            if f"{pfx}norm1.weight" in params:
                h_l = rms_norm(flux, params[f"{pfx}norm1.weight"])
                q_l = jnp.matmul(h_l, params[f"{pfx}q_proj.weight"].T).reshape(B, L, num_heads, head_dim)
                q_l = jnp.swapaxes(q_l, 1, 2)
                k_l = jnp.matmul(h_l, params[f"{pfx}k_proj.weight"].T).reshape(B, L, num_heads, head_dim)
                k_l = jnp.swapaxes(k_l, 1, 2)
                v_l = jnp.matmul(h_l, params[f"{pfx}v_proj.weight"].T).reshape(B, L, num_heads, head_dim)
                v_l = jnp.swapaxes(v_l, 1, 2)

                if config is not None and getattr(config, "use_rope", True):
                    sens_pos = jnp.arange(L) + pos_offset
                    q_l = apply_rope(q_l, sens_pos, theta=config.rope_theta)
                    k_l = apply_rope(k_l, sens_pos, theta=config.rope_theta)

                scores_l = jnp.matmul(q_l, jnp.swapaxes(k_l, -1, -2)) / math.sqrt(head_dim)
                if attn_mask is not None:
                    scores_l = jnp.where(attn_mask, scores_l, -1e4)
                elif is_causal:
                    causal_mask = jnp.tril(jnp.ones((L, L), dtype=bool))[None, None, :, :]
                    scores_l = jnp.where(causal_mask, scores_l, -1e4)
                attn_w_l = jax.nn.softmax(scores_l.astype(jnp.float32), axis=-1).astype(flux.dtype)
                ctx_l = jnp.matmul(attn_w_l, v_l)
                ctx_l = jnp.swapaxes(ctx_l, 1, 2).reshape(B, L, D)
                flux = flux + jnp.matmul(ctx_l, params[f"{pfx}out_proj.weight"].T)

                h2 = rms_norm(flux, params[f"{pfx}norm2.weight"])
                h2_swiglu = swiglu(
                    h2,
                    params[f"{pfx}swiglu.fc_gate.weight"],
                    params[f"{pfx}swiglu.fc_val.weight"],
                    params[f"{pfx}swiglu.fc_down.weight"],
                )
                flux = flux + h2_swiglu

    k = jnp.matmul(flux, params["sensory.k_proj.weight"].T).reshape(B, L, num_heads, head_dim)
    k = jnp.swapaxes(k, 1, 2)  # [B, H, L, head_dim]
    v = jnp.matmul(flux, params["sensory.v_proj.weight"].T).reshape(B, L, num_heads, head_dim)
    v = jnp.swapaxes(v, 1, 2)  # [B, H, L, head_dim]

    if is_2d:
        flux = flux.squeeze(1)

    return flux, (k, v)


def deliberate_step(params: Dict[str, jnp.ndarray], state: jnp.ndarray, buffers: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """Executes a single subconscious deliberation step across all concentric laminae with fused 3D tensor contractions."""
    B, K, D = state.shape
    src_idx, dst_idx = buffers
    st_dtype = state.dtype

    # 1. Batched Intra-shell Pre-Norm across all K shells simultaneously (float32 accumulation)
    state_f32 = state.astype(jnp.float32)
    var = jnp.mean(jnp.square(state_f32), axis=-1, keepdims=True)
    h = (state_f32 * jax.lax.rsqrt(var + 1e-6) * params["core.intra_norms.weight"][None, :, :].astype(jnp.float32)).astype(st_dtype)

    # 2. Batched 3D SwiGLU: Single systolic tensor contraction across all K shells in parallel
    gate = jax.nn.silu(jnp.einsum("bkd,kde->bke", h, params["core.intra_swiglu.w_gate"]))
    val = jnp.einsum("bkd,kde->bke", h, params["core.intra_swiglu.w_val"])
    down = jnp.einsum("bke,ked->bkd", gate * val, params["core.intra_swiglu.w_down"])
    state_intra = state + down * params["core.shell_gains"][None, :, None].astype(st_dtype)

    # 3. Pre-Norm before radial convection & shortcuts (eliminates global RMSNorm shell collapse)
    h_rad = rms_norm(state_intra, params["core.radial_norm.weight"])

    # Learnable radial and shortcut mixing scales (initialized to 0.1 and 0.05)
    r_mix = jnp.clip(params.get("core.radial_mix_scale", jnp.array(0.1, dtype=jnp.float32)), 0.01, 1.0).astype(st_dtype)
    s_mix = jnp.clip(params.get("core.shortcut_mix_scale", jnp.array(0.05, dtype=jnp.float32)), 0.005, 1.0).astype(st_dtype)
    w_up_out = params.get("core.radial_up_out.weight", params["core.radial_up.weight"])

    # Inter-shell bi-directional radial wave propagation with decoupled inward/outward projections
    rad_in = jnp.matmul(h_rad[:, 1:, :], params["core.radial_in_proj.weight"].T)
    in_flux = jnp.einsum("bkd,kde->bke", rad_in, params["core.w_radial_in"])
    in_up = jnp.matmul(in_flux, params["core.radial_up.weight"].T)

    rad_out = jnp.matmul(h_rad[:, :-1, :], params["core.radial_out_proj.weight"].T)
    out_flux = jnp.einsum("bkd,kde->bke", rad_out, params["core.w_radial_out"])
    out_up = jnp.matmul(out_flux, w_up_out.T)

    state_radial = state_intra.at[:, :-1, :].add(r_mix * in_up)
    state_radial = state_radial.at[:, 1:, :].add(r_mix * out_up)

    # 4. Small-world hierarchical axonal shortcut tracts in 1 vectorized step
    conductances = jax.nn.sigmoid(params["core.shortcut_gate"]).astype(st_dtype)
    src_acts = h_rad[:, src_idx, :]
    shortcut_flux = jnp.einsum("btd,tde->bte", src_acts, params["core.shortcut_weight"]) * conductances[None, :, None]

    # Return unnormalized residual state to preserve genuine radial shell amplitude hierarchy
    state_next = state_radial.at[:, dst_idx, :].add(s_mix * shortcut_flux)
    return state_next


def deliberate_cycles_jax(
    params: Dict[str, jnp.ndarray],
    state: jnp.ndarray,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: Optional[ModelConfigV5] = None,
    is_training: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Executes subconscious deliberation cycles across concentric laminae with static-iteration
    lax.scan Adaptive Computation Time (ACT) for TPU/XLA compilation efficiency.

    Returns:
        (final_state, ponder_cost)
    """
    if config is None:
        use_adapt = False
        cycles = 4
    else:
        use_adapt = bool(getattr(config, "use_adaptive_thinking", False)) and ("core.halting_gate.weight" in params)
        cycles = int(getattr(config, "max_thinking_cycles", getattr(config, "thinking_cycles", 4))) if use_adapt else int(getattr(config, "thinking_cycles", 4))

    if not use_adapt:
        def step_fn(curr_state, _):
            next_state = deliberate_step(params, curr_state, buffers)
            return next_state, None

        final_state, _ = lax.scan(step_fn, state, None, length=cycles)
        return final_state, jnp.array(0.0, dtype=jnp.float32)

    # Static ACT formulation for TPU/XLA:
    B, K, D = state.shape
    threshold = float(getattr(config, "thinking_halt_threshold", 0.75))
    w_gate = params["core.halting_gate.weight"]  # [1, D]
    b_gate = params["core.halting_gate.bias"]    # [1]

    # Carry: (curr_state, accum_state, cum_p, cum_ponder)
    init_accum = jnp.zeros_like(state)
    init_cum_p = jnp.zeros((B,), dtype=jnp.float32)
    init_ponder = jnp.zeros((B,), dtype=jnp.float32)

    def scan_body(carry, step_idx):
        curr_state, accum_state, cum_p, cum_ponder = carry
        next_state = deliberate_step(params, curr_state, buffers)

        # Deliberation state pooling across cortical/subcortical laminae
        state_mean = jnp.mean(next_state.astype(jnp.float32), axis=1)  # [B, D]
        halt_logit = jnp.squeeze(jnp.matmul(state_mean, w_gate.T), axis=-1) + b_gate[0]
        halt_prob = jax.nn.sigmoid(halt_logit)  # [B]

        # Differentiable Graves ACT ponder cost
        is_last = (step_idx == cycles - 1)
        rem_p = jnp.maximum(0.0, 1.0 - cum_p)
        step_weight = jnp.where(is_last, rem_p, jnp.minimum(halt_prob, rem_p))

        # Update cumulative probability and accumulated convex state
        new_cum_p = cum_p + step_weight
        accum_state = accum_state + step_weight[:, None, None].astype(state.dtype) * next_state
        cum_ponder = cum_ponder + rem_p

        return (next_state, accum_state, new_cum_p, cum_ponder), None

    (_, final_accum, final_cum_p, cum_ponder), _ = lax.scan(
        scan_body,
        (state, init_accum, init_cum_p, init_ponder),
        jnp.arange(cycles),
    )

    norm_factor = jnp.maximum(final_cum_p[:, None, None], 1e-6)
    final_state = (final_accum / norm_factor).astype(state.dtype)
    ponder_cost = jnp.mean(cum_ponder)

    return final_state, ponder_cost


def emit_step(
    params: Dict[str, jnp.ndarray],
    state: jnp.ndarray,
    sensory_kv: Tuple[jnp.ndarray, jnp.ndarray],
    plan: jnp.ndarray,
    refractory_trace: Optional[jnp.ndarray],
    step_idx: Union[int, jnp.ndarray],
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: Optional[ModelConfigV5] = None,
    auditory_feedback: Optional[jnp.ndarray] = None,
    repetition_penalty: float = 1.25,
    attn_mask: Optional[jnp.ndarray] = None,
    return_hidden: bool = False,
    encephalization_alpha: float = 1.0,
    query_position: Optional[Union[int, jnp.ndarray]] = None,
) -> Union[
    Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray, jnp.ndarray],
]:
    """Single motor emission step with affine-tied dictionary projection and progressive encephalization."""
    B, K, D = state.shape
    st_dtype = state.dtype
    core = state[:, 0, :]
    if K > 16:
        surface = (1.0 - encephalization_alpha) * state[:, 15, :] + encephalization_alpha * state[:, -1, :]
    else:
        surface = state[:, -1, :]
    num_heads = config.num_heads if config is not None else 8
    head_dim = D // num_heads

    # 1. Learnable Scaled Phase Injection
    max_phase = params["motor.phase_embed.weight"].shape[0]
    p_idx = jnp.clip(step_idx, 0, max_phase - 1)
    phase = params["motor.phase_embed.weight"][p_idx].astype(jnp.float32)
    p_norm = jnp.maximum(jnp.linalg.norm(phase, axis=-1, keepdims=True), 1e-6)
    p_scale_val = jnp.clip(params.get("motor.phase_scale", jnp.array(2.0, dtype=jnp.float32)), 0.05, 5.0).astype(jnp.float32)
    p_scaled = (phase * (p_scale_val / p_norm)).astype(st_dtype)

    motor_surface = surface + p_scaled
    if auditory_feedback is not None:
        motor_surface = motor_surface + auditory_feedback.astype(st_dtype)

    # 2. Multi-Head Arcuate Fasciculus Cross-Resonance with Causal Prefix
    keys, values = sensory_kv
    q = jnp.matmul(motor_surface, params["motor.q_proj.weight"].T).reshape(B, 1, num_heads, head_dim)
    q = jnp.swapaxes(q, 1, 2)  # [B, H, 1, head_dim]
    if config is not None and config.use_rope:
        q_pos = step_idx if query_position is None else query_position
        q = apply_rope(q, q_pos, theta=config.rope_theta)
        keys = apply_rope(keys, jnp.arange(keys.shape[2]), theta=config.rope_theta)
    scores = jnp.matmul(q, jnp.swapaxes(keys, -1, -2)) / math.sqrt(head_dim)
    if attn_mask is not None:
        scores = jnp.where(attn_mask, scores, -1e4)
    attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(st_dtype)
    res_heads = jnp.matmul(attn_weights, values)  # [B, H, 1, head_dim]
    res_heads = jnp.swapaxes(res_heads, 1, 2).reshape(B, D)
    resonance = rms_norm(jnp.matmul(res_heads, params["motor.out_proj.weight"].T), params["motor.resonance_norm.weight"])

    # 3. Multi-Shell Softmax Pooling
    shell_weights = jax.nn.softmax(params["motor.shell_score"].astype(jnp.float32), axis=0)[None, :, None].astype(st_dtype)
    shell_proj = jnp.matmul(state, params["motor.shell_proj.weight"].T)
    shell_pool = jnp.sum(shell_weights * shell_proj, axis=1) * params["motor.shell_scale"].astype(st_dtype)

    # 4. Gated Deep Motor Integration via SwiGLU (per-source learnable importance gating)
    ig = jnp.clip(params.get("motor.input_gate", jnp.ones(5, dtype=jnp.float32)), -5.0, 5.0).astype(st_dtype)
    fused_in = jnp.concatenate([
        ig[0] * motor_surface,
        ig[1] * core,
        ig[2] * resonance,
        ig[3] * shell_pool,
        ig[4] * plan,
    ], axis=-1)
    fused = rms_norm(fused_in, params["motor.fused_norm.weight"])
    h = rms_norm(
        swiglu(
            fused,
            params["motor.swiglu.fc_gate.weight"],
            params["motor.swiglu.fc_val.weight"],
            params["motor.swiglu.fc_down.weight"],
        ),
        params["motor.norm_mid.weight"],
    )

    # 5. Premotor Recurrent Executive Plan Update
    phase_batch = jnp.repeat(p_scaled[None, :], B, axis=0)
    plan_input = jnp.concatenate([h, phase_batch], axis=-1)
    p_gate = jax.nn.sigmoid(
        jnp.matmul(plan_input, params["motor.plan_gate.weight"].T) + params["motor.plan_gate.bias"].astype(st_dtype)
    )
    plan_decay = jnp.clip(params["motor.plan_decay"].astype(jnp.float32), 0.0, 0.92).astype(st_dtype)
    plan_write = params["motor.plan_write"].astype(st_dtype)
    plan_update = jnp.matmul(plan_input, params["motor.plan_update.weight"].T)
    next_plan = jnp.clip(
        plan_decay * plan + plan_write * p_gate * plan_update,
        -50.0,
        50.0,
    ).astype(st_dtype)

    # 6. Affine-Tied Dictionary Projection (W_motor = gamma * W_sensory)
    # Memory-optimal: Scale hidden vector [B, D] instead of vocabulary matrix [V, D]
    temp = jnp.clip(params["motor.temperature_scale"].astype(jnp.float32), 0.1, 5.0).astype(st_dtype)
    h_scaled = (h * params["motor.affine_scale"][None, :].astype(st_dtype)) / temp
    raw_logits = jnp.matmul(h_scaled, params["sensory.embedding.weight"].T)

    # 7. Two-Tier Refractory Adaptation
    if refractory_trace is not None:
        alpha = jnp.clip(params.get("motor.refractory_strength", jnp.array(1.2, dtype=jnp.float32)).astype(jnp.float32), 0.1, 5.0)
        tier1_penalty = 3.0 * alpha * refractory_trace.astype(jnp.float32)

        active_mask = (refractory_trace > 0.02).astype(jnp.float32)
        raw_f32 = raw_logits.astype(jnp.float32)
        rep_scale = jnp.where(raw_f32 > 0, 1.0 / repetition_penalty, repetition_penalty)
        scaled_logits = raw_f32 * (1.0 - active_mask) + (raw_f32 * rep_scale) * active_mask
        adapted_logits = (scaled_logits - tier1_penalty).astype(st_dtype)

        decay = jax.nn.sigmoid(params.get("motor.refractory_decay", jnp.array(0.75, dtype=jnp.float32)).astype(jnp.float32))
        current_prob = lax.stop_gradient(jax.nn.softmax(raw_f32, axis=-1))
        top1_idx = jnp.argmax(adapted_logits, axis=-1)
        top1_pulse = lax.stop_gradient(jax.nn.one_hot(top1_idx, raw_logits.shape[-1]))
        next_trace = lax.stop_gradient(decay * refractory_trace.astype(jnp.float32) + (1.0 - decay) * current_prob + 1.0 * top1_pulse).astype(st_dtype)
    else:
        adapted_logits, next_trace = raw_logits, None

    # 8. Lightweight Bounded Efference Update on Shell 0 (eliminates O(T^2 * K) full deliberate_step inside emission)
    efference = jnp.matmul(h, params["motor.efference_proj.weight"].T)
    eff_projected = jnp.matmul(jax.nn.silu(efference), params["core.efference_gate.weight"].T)
    eff_normed = rms_norm(eff_projected, params["core.radial_norm.weight"]).astype(st_dtype)
    new_core = (0.85 * core + 0.15 * eff_normed).astype(st_dtype)
    next_state = state.at[:, 0, :].set(new_core)

    if return_hidden:
        return adapted_logits, next_state, next_trace, next_plan, h
    return adapted_logits, next_state, next_trace, next_plan


def training_rollout(
    params: Dict[str, jnp.ndarray],
    prompts: jnp.ndarray,
    target_len: int,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: Optional[ModelConfigV5] = None,
    feedback_tokens: Optional[jnp.ndarray] = None,
    sampling_p: float = 1.0,
    pitfall_prob: float = 0.0,
    pad_id: int = 0,
    repetition_penalty: float = 1.25,
    targets: Optional[jnp.ndarray] = None,
    remat: bool = True,
    persona_id: Optional[int] = None,
    feedback_is_shifted: bool = True,
    pure_teacher_forcing: Optional[bool] = None,
    seed_token_id: Optional[int] = None,
    deliberation_mode: str = "scheduled",
    interleave_stride: int = 2,
    predictive_gain: float = 0.5,
    images: Optional[jnp.ndarray] = None,
    audio: Optional[jnp.ndarray] = None,
    perceptual_inputs: Optional[jnp.ndarray] = None,
    anchor_token_id: int = 3191,
    anchor_weight: float = 3.5,
    use_anchor_loss: bool = True,
    encephalization_alpha: float = 1.0,
    target_weights: Optional[jnp.ndarray] = None,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Differentiable training and autoregressive generation rollout in pure JAX."""
    # Dual-Pass Twin Resonance: runs teacher-forcing pass and deliberative rollout pass on identical tokens
    if deliberation_mode == "dual_pass" and targets is not None:
        loss_tf = training_rollout(
            params=params,
            prompts=prompts,
            target_len=target_len,
            buffers=buffers,
            config=config,
            feedback_tokens=feedback_tokens,
            sampling_p=1.0,
            pitfall_prob=0.0,
            pad_id=pad_id,
            repetition_penalty=repetition_penalty,
            targets=targets,
            remat=remat,
            persona_id=persona_id,
            feedback_is_shifted=feedback_is_shifted,
            pure_teacher_forcing=True,
            deliberation_mode="teacher_forcing",
            images=images,
            audio=audio,
            perceptual_inputs=perceptual_inputs,
            anchor_token_id=anchor_token_id,
            anchor_weight=anchor_weight,
            use_anchor_loss=use_anchor_loss,
            encephalization_alpha=encephalization_alpha,
        )
        loss_delib = training_rollout(
            params=params,
            prompts=prompts,
            target_len=target_len,
            buffers=buffers,
            config=config,
            feedback_tokens=feedback_tokens,
            sampling_p=sampling_p,
            pitfall_prob=pitfall_prob,
            pad_id=pad_id,
            repetition_penalty=repetition_penalty,
            targets=targets,
            remat=remat,
            persona_id=persona_id,
            feedback_is_shifted=feedback_is_shifted,
            pure_teacher_forcing=False,
            deliberation_mode="scheduled",
            images=images,
            audio=audio,
            perceptual_inputs=perceptual_inputs,
            anchor_token_id=anchor_token_id,
            anchor_weight=anchor_weight,
            use_anchor_loss=use_anchor_loss,
        )
        return 0.5 * loss_tf + 0.5 * loss_delib

    target_dtype = params["sensory.embedding.weight"].dtype

    # 1. Sensory Ingestion (Multimodal Images, Audio, Perceptual Flux or Text)
    sensory_tokens_list = []
    if images is not None and "image_stem.conv.weight" in params:
        p_size = getattr(config, "image_patch_size", 16) if config is not None else 16
        sensory_tokens_list.append(encode_image_stem(params, images, patch_size=p_size))
    if audio is not None and "audio_stem.conv.weight" in params:
        sensory_tokens_list.append(encode_audio_stem(params, audio))
    if perceptual_inputs is not None:
        sensory_tokens_list.append(perceptual_inputs)

    has_perceptual = len(sensory_tokens_list) > 0 and "perceptual.linear.weight" in params
    if has_perceptual:
        raw_p_flux = jnp.concatenate(sensory_tokens_list, axis=1)  # [B, P_total, v_dim]
        B_vis, P_vis, _ = raw_p_flux.shape
        vis_flux = project_perceptual(params, raw_p_flux)  # [B, P_total, surface_dim]
        w_emb = params["sensory.embedding.weight"]
        if prompts.ndim == 2 and prompts.dtype in (jnp.int32, jnp.int64):
            txt_emb = w_emb[prompts]
            combined_emb = jnp.concatenate([vis_flux, txt_emb], axis=1)
            sensory_flux, sensory_kv = encode_sensory(params, combined_emb, config=config, pos_offset=0)
            vis_mask = jnp.ones((B_vis, P_vis), dtype=jnp.float32)
            txt_mask = (prompts != pad_id).astype(jnp.float32)
            combined_mask = jnp.concatenate([vis_mask, txt_mask], axis=1)
            total_prompt_L = P_vis + prompts.shape[1]
            pos_idx = jnp.arange(total_prompt_L, dtype=jnp.float32)[None, :]
            last_valid_pos = jnp.max(jnp.where(combined_mask > 0.0, pos_idx, 0.0), axis=1, keepdims=True)
            recency_weights = jnp.exp(0.05 * (pos_idx - last_valid_pos)) * combined_mask
            weight_sum = jnp.maximum(jnp.sum(recency_weights, axis=1, keepdims=True), 1.0)
            prompt_flux = (
                jnp.sum(sensory_flux.astype(jnp.float32) * recency_weights[:, :, None], axis=1) / weight_sum
            ).astype(target_dtype)
            prompt_valid = (combined_mask > 0.0)[:, None, None, :]
        else:
            combined_emb = jnp.concatenate([vis_flux, prompts], axis=1)
            sensory_flux, sensory_kv = encode_sensory(params, combined_emb, config=config, pos_offset=0)
            total_prompt_L = combined_emb.shape[1]
            pos_idx = jnp.arange(total_prompt_L, dtype=jnp.float32)[None, :]
            recency_weights = jnp.exp(0.05 * (pos_idx - float(max(total_prompt_L - 1, 0))))
            weight_sum = jnp.maximum(jnp.sum(recency_weights, axis=1, keepdims=True), 1.0)
            prompt_flux = (
                jnp.sum(sensory_flux.astype(jnp.float32) * recency_weights[:, :, None], axis=1) / weight_sum
            ).astype(target_dtype)
            prompt_valid = jnp.ones((B_vis, 1, 1, total_prompt_L), dtype=bool)
        B = B_vis
    else:
        B, L = prompts.shape
        total_prompt_L = L
        prompt_valid = (prompts != pad_id)[:, None, None, :]
        sensory_flux, sensory_kv = encode_sensory(params, prompts, config=config, pos_offset=0, attn_mask=prompt_valid)
        mask_2d = (prompts != pad_id).astype(jnp.float32)  # [B, L]
        pos_idx = jnp.arange(L, dtype=jnp.float32)[None, :]  # [1, L]
        last_valid_pos = jnp.max(jnp.where(mask_2d > 0.0, pos_idx, 0.0), axis=1, keepdims=True)
        recency_weights = jnp.exp(0.05 * (pos_idx - last_valid_pos)) * mask_2d
        weight_sum = jnp.maximum(jnp.sum(recency_weights, axis=1, keepdims=True), 1.0)
        prompt_flux = (
            jnp.sum(sensory_flux.astype(jnp.float32) * recency_weights[:, :, None], axis=1) / weight_sum
        ).astype(target_dtype)

    num_shells = config.num_shells if config is not None else 32
    surface_dim = config.surface_dim if config is not None else 768
    vocab_size = config.vocab_size if config is not None else 32000
    thinking_cycles = config.thinking_cycles if config is not None else 4

    # Initial concentric state
    init_state = jnp.zeros((B, num_shells, surface_dim), dtype=target_dtype)
    if num_shells > 16:
        state = init_state.at[:, 15, :].set(prompt_flux).at[:, -1, :].set(prompt_flux)
    else:
        state = init_state.at[:, -1, :].set(prompt_flux)

    # Subconscious deliberation cycles with Adaptive Computation Time (ACT)
    state, ponder_cost = deliberate_cycles_jax(
        params, state, buffers, config=config, is_training=(targets is not None)
    )

    # Initial executive plan and refractory trace
    init_plan = jnp.zeros((B, surface_dim), dtype=target_dtype)
    if persona_id is not None and "motor.persona_embeddings" in params:
        p_idx = jnp.clip(persona_id, 0, params["motor.persona_embeddings"].shape[0] - 1)
        p_emb = params["motor.persona_embeddings"][p_idx].astype(target_dtype)
        p_scale = params.get("motor.persona_scale", jnp.array(0.5, dtype=jnp.float32)).astype(target_dtype)
        init_plan = init_plan + p_scale * p_emb[None, :]
    # When training (teacher-forced MLE), disable refractory penalty so loss accurately reflects true likelihood
    if targets is not None:
        init_refractory = None
        repetition_penalty = 1.0
    else:
        init_refractory = jnp.zeros((B, vocab_size), dtype=target_dtype)
    init_thought = jnp.zeros((B, surface_dim), dtype=target_dtype)

    # Pillar 2: Dynamic Answer-Anchor Loss Weighting (AALW) or Teacher Digested Weights
    if target_weights is not None:
        target_weights = target_weights[:, :target_len].astype(jnp.float32)
    elif targets is not None and use_anchor_loss:
        tgt_cur = targets[:, :target_len]
        is_anchor = (tgt_cur == anchor_token_id)
        after_anchor = (jnp.cumsum(is_anchor.astype(jnp.int32), axis=-1) > 0)
        has_anchor = jnp.any(is_anchor, axis=-1, keepdims=True)
        base_w = jnp.where(has_anchor, jnp.where(after_anchor, anchor_weight, 1.0), anchor_weight)
        valid_mask = (tgt_cur >= 0) & (tgt_cur != -100)
        target_weights = jnp.where(valid_mask, base_w, 0.0).astype(jnp.float32)
    elif targets is not None:
        target_weights = (targets[:, :target_len] >= 0).astype(jnp.float32)
    else:
        target_weights = None

    # Build strictly non-leaking prev_tokens[0..target_len-1] aligned 1:1 with emission step 0..target_len-1
    if feedback_tokens is None and targets is not None:
        bos_pad = jnp.full((B, 1), -100, dtype=targets.dtype)
        if target_len > 1:
            prev_tokens = jnp.concatenate([bos_pad, targets[:, :target_len - 1]], axis=1)
        else:
            prev_tokens = bos_pad[:, :target_len]
    elif feedback_tokens is not None:
        fb_slice = feedback_tokens[:, :target_len]
        if feedback_is_shifted:
            prev_tokens = fb_slice
        else:
            bos_pad = jnp.full((B, 1), -100, dtype=fb_slice.dtype)
            prev_tokens = jnp.concatenate([bos_pad, fb_slice[:, :target_len - 1]], axis=1) if target_len > 1 else bos_pad[:, :target_len]
    else:
        prev_tokens = None

    prompt_k, prompt_v = sensory_kv
    num_heads = prompt_k.shape[1]
    head_dim = prompt_k.shape[3]

    if prev_tokens is not None:
        safe_feed = jnp.maximum(prev_tokens, 0)
        valid_feed = (prev_tokens >= 0)[:, :, None].astype(target_dtype)
        tf_flux, (tf_k, tf_v) = encode_sensory(params, safe_feed, config=config, pos_offset=total_prompt_L, is_causal=True)
        tf_flux = (tf_flux * valid_feed).astype(target_dtype)

        # Causal Key/Value Accumulation:
        # Prompt keys [B, H, total_prompt_L, head_dim] + prev_tokens keys [B, H, target_len, head_dim]
        causal_keys = jnp.concatenate([prompt_k, tf_k], axis=2)
        causal_values = jnp.concatenate([prompt_v, tf_v], axis=2)

        target_valid = (prev_tokens >= 0)[:, None, None, :]  # [B, 1, 1, target_len]
        all_valid = jnp.concatenate([prompt_valid, target_valid], axis=-1)  # [B, 1, 1, total_prompt_L + target_len]
    else:
        causal_keys, causal_values = prompt_k, prompt_v
        tf_flux = jnp.zeros((B, target_len, surface_dim), dtype=target_dtype)
        all_valid = prompt_valid

    def _emit(carry, step_idx):
        cur_state, cur_trace, cur_plan, prev_thought = carry
        tf_aud = tf_flux[:, step_idx, :]
        internal_feed = jnp.where(step_idx == 0, 0.0, prev_thought)

        if deliberation_mode == "interleaved":
            is_delib = (step_idx % interleave_stride) == 1
            step_p = jnp.where(is_delib, sampling_p, 1.0)
            feed = step_p * tf_aud + (1.0 - step_p) * internal_feed
        elif deliberation_mode == "predictive_coding":
            feed = jnp.where(step_idx == 0, tf_aud, internal_feed + predictive_gain * (tf_aud - internal_feed))
        else:
            feed = sampling_p * tf_aud + (1.0 - sampling_p) * internal_feed

        # Causal Attention Mask: prev_tokens[0..step_idx] (up to total_prompt_L + step_idx) strictly precede targets[step_idx]
        total_len = causal_keys.shape[2]
        pos = jnp.arange(total_len)
        causal_mask = (pos <= (total_prompt_L + step_idx))[None, None, None, :]
        step_mask = all_valid & causal_mask

        logits, nxt_s, nxt_t, nxt_p = emit_step(
            params,
            cur_state,
            (causal_keys, causal_values),
            cur_plan,
            cur_trace,
            step_idx,
            buffers,
            config=config,
            auditory_feedback=feed,
            repetition_penalty=repetition_penalty,
            attn_mask=step_mask,
            encephalization_alpha=encephalization_alpha,
            query_position=total_prompt_L + step_idx,
        )

        # Scheduled sampling re-afference using full sensory encoder (position + SwiGLU aligned)
        if pitfall_prob > 0.0:
            top2 = lax.top_k(logits, 2)[1]
            hash_val = jnp.sin(step_idx * 12.9898 + 78.233) * 43758.5453
            pitfall_mask = (hash_val - jnp.floor(hash_val)) < pitfall_prob
            tok_idx = lax.stop_gradient(jnp.where(pitfall_mask, top2[:, 1], top2[:, 0]))
        else:
            tok_idx = lax.stop_gradient(jnp.argmax(logits, axis=-1))
        tok_flux, _ = encode_sensory(params, tok_idx[:, None], config=config, pos_offset=total_prompt_L + step_idx)
        nxt_thought = lax.stop_gradient(tok_flux[:, 0, :]).astype(target_dtype)

        # Enforce exact carry type alignment
        nxt_s = nxt_s.astype(target_dtype)
        nxt_p = nxt_p.astype(target_dtype)
        if nxt_t is not None:
            nxt_t = nxt_t.astype(target_dtype)

        if targets is not None:
            cur_target = targets[:, step_idx]
            log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            safe_targets = jnp.maximum(cur_target, 0)
            target_log_probs = jnp.take_along_axis(log_probs, safe_targets[:, None], axis=-1).squeeze(-1)
            mask_t = (cur_target >= 0).astype(jnp.float32)
            cur_w = target_weights[:, step_idx] if target_weights is not None else mask_t
            eff_w = mask_t * cur_w
            step_loss = -target_log_probs * eff_w
            if deliberation_mode == "predictive_coding":
                thought_err = jnp.mean((prev_thought.astype(jnp.float32) - lax.stop_gradient(tf_aud.astype(jnp.float32))) ** 2, axis=-1)
                step_loss = step_loss + (0.1 * thought_err * eff_w * (step_idx > 0).astype(jnp.float32))
            return (nxt_s, nxt_t, nxt_p, nxt_thought), (step_loss, eff_w)

        return (nxt_s, nxt_t, nxt_p, nxt_thought), logits

    # Autoregressive Generation Engine with Dynamic Causal KV Cache (when no targets or feedback provided)
    if targets is None and feedback_tokens is None:
        init_k_buf = jnp.zeros((B, num_heads, target_len, head_dim), dtype=target_dtype)
        init_v_buf = jnp.zeros((B, num_heads, target_len, head_dim), dtype=target_dtype)
        # Align with prev_tokens: slot 0 is the step-0 silence placeholder (masked out)
        target_gen_valid = jnp.concatenate(
            [jnp.zeros((B, 1, 1, 1), dtype=bool), jnp.ones((B, 1, 1, max(0, target_len - 1)), dtype=bool)], axis=-1
        ) if target_len > 1 else jnp.zeros((B, 1, 1, 1), dtype=bool)
        gen_valid = jnp.concatenate([prompt_valid, target_gen_valid], axis=-1)

        def _emit_gen(carry, step_idx):
            cur_state, cur_trace, cur_plan, prev_thought, k_buf, v_buf = carry
            internal_feed = jnp.where(step_idx == 0, 0.0, prev_thought)

            cur_all_k = jnp.concatenate([prompt_k, k_buf], axis=2)
            cur_all_v = jnp.concatenate([prompt_v, v_buf], axis=2)
            pos = jnp.arange(total_prompt_L + target_len)
            step_mask = gen_valid & (pos <= (total_prompt_L + step_idx))[None, None, None, :]

            logits, nxt_s, nxt_t, nxt_p = emit_step(
                params,
                cur_state,
                (cur_all_k, cur_all_v),
                cur_plan,
                cur_trace,
                step_idx,
                buffers,
                config=config,
                auditory_feedback=internal_feed,
                repetition_penalty=repetition_penalty,
                attn_mask=step_mask,
                encephalization_alpha=encephalization_alpha,
                query_position=total_prompt_L + step_idx,
            )

            tok_idx = lax.stop_gradient(jnp.argmax(logits, axis=-1))
            if seed_token_id is not None:
                tok_idx = jnp.where(
                    step_idx == 0,
                    jnp.asarray(seed_token_id, dtype=tok_idx.dtype),
                    tok_idx,
                )
            tok_flux, (new_k, new_v) = encode_sensory(
                params, tok_idx[:, None], config=config, pos_offset=total_prompt_L + step_idx + 1
            )
            nxt_thought = lax.stop_gradient(tok_flux[:, 0, :]).astype(target_dtype)
            new_k = lax.stop_gradient(new_k).astype(target_dtype)
            new_v = lax.stop_gradient(new_v).astype(target_dtype)

            # Insert at slot step_idx + 1 to align 1:1 with prev_tokens[1..]
            update_slot = jnp.minimum(step_idx + 1, target_len - 1)
            nxt_k_buf = lax.dynamic_update_slice(k_buf, new_k, (0, 0, update_slot, 0))
            nxt_v_buf = lax.dynamic_update_slice(v_buf, new_v, (0, 0, update_slot, 0))

            return (nxt_s.astype(target_dtype), nxt_t, nxt_p.astype(target_dtype), nxt_thought, nxt_k_buf, nxt_v_buf), logits

        _, all_logits = lax.scan(_emit_gen, (state, init_refractory, init_plan, init_thought, init_k_buf, init_v_buf), jnp.arange(target_len))
        return jnp.swapaxes(all_logits, 0, 1)

    # Use activation checkpointing (remat) for training to reduce HBM activation memory
    use_remat = remat and (targets is not None)
    is_pure_tf = (
        (targets is not None)
        and (num_shells >= 2)
        and (deliberation_mode in ("teacher_forcing", "pure_tf"))
        and (
            pure_teacher_forcing is True
            or (
                pure_teacher_forcing is None
                and (not isinstance(sampling_p, jax.Array))
                and (float(sampling_p) >= 1.0 - 1e-4)
            )
        )
    )

    if is_pure_tf:
        T = int(target_len)
        if num_shells > 16:
            surface = (1.0 - encephalization_alpha) * state[:, 15, :] + encephalization_alpha * state[:, -1, :]
        else:
            surface = state[:, -1, :]
        init_core = state[:, 0, :]

        # 1. Precompute all T normalized phase clocks and motor surface vectors outside scan
        max_phase = params["motor.phase_embed.weight"].shape[0]
        p_indices = jnp.clip(jnp.arange(T), 0, max_phase - 1)
        phases = params["motor.phase_embed.weight"][p_indices].astype(jnp.float32)
        p_norms = jnp.maximum(jnp.linalg.norm(phases, axis=-1, keepdims=True), 1e-6)
        p_scale_val = jnp.clip(
            params.get("motor.phase_scale", jnp.array(2.0, dtype=jnp.float32)), 0.05, 5.0
        ).astype(jnp.float32)
        p_scaled_all = (phases * (p_scale_val / p_norms)).astype(target_dtype)  # [T, D]
        motor_surface_all = surface[:, None, :] + p_scaled_all[None, :, :] + tf_flux.astype(target_dtype)

        # 2. Vectorized Multi-Head Arcuate Cross-Resonance across all T tokens in one systolic block
        total_len = causal_keys.shape[2]
        keys_t = jnp.swapaxes(causal_keys, -1, -2)

        def _parallel_cross_resonance(ms_all):
            q_all = jnp.matmul(ms_all, params["motor.q_proj.weight"].T).reshape(B, T, num_heads, head_dim)
            q_all = jnp.swapaxes(q_all, 1, 2)  # [B, H, T, head_dim]
            if config.use_rope:
                q_all = apply_rope(q_all, total_prompt_L + jnp.arange(T), theta=config.rope_theta)
                keys_att = apply_rope(causal_keys, jnp.arange(total_len), theta=config.rope_theta)
                keys_att_t = jnp.swapaxes(keys_att, -1, -2)
            else:
                keys_att_t = keys_t
            scores_all = jnp.matmul(q_all, keys_att_t) / math.sqrt(head_dim)
            pos = jnp.arange(total_len)[None, :]
            step_pos = (total_prompt_L + jnp.arange(T))[:, None]
            step_mask_all = all_valid & ((pos <= step_pos)[None, None, :, :])
            scores_all = jnp.where(step_mask_all, scores_all, -1e4)
            attn_w = jax.nn.softmax(scores_all.astype(jnp.float32), axis=-1).astype(target_dtype)
            res_heads = jnp.matmul(attn_w, causal_values)
            res_heads = jnp.swapaxes(res_heads, 1, 2).reshape(B, T, surface_dim)
            return rms_norm(
                jnp.matmul(res_heads, params["motor.out_proj.weight"].T),
                params["motor.resonance_norm.weight"],
            )

        res_fn = jax.checkpoint(_parallel_cross_resonance) if use_remat else _parallel_cross_resonance
        resonance_all = res_fn(motor_surface_all)  # [B, T, D]

        # 3. Precompute static shell pooling for shells 1..K-1 (only shell 0 evolves inside emission)
        shell_weights = jax.nn.softmax(params["motor.shell_score"].astype(jnp.float32), axis=0).astype(target_dtype)
        shell_scale = params["motor.shell_scale"].astype(target_dtype)
        w_shell_t = params["motor.shell_proj.weight"].T
        static_shell_proj = jnp.matmul(state[:, 1:, :], w_shell_t)
        static_shell_pool = jnp.sum(shell_weights[None, 1:, None] * static_shell_proj, axis=1) * shell_scale
        w0_shell_scale = shell_weights[0] * shell_scale

        ig = jnp.clip(params.get("motor.input_gate", jnp.ones(5, dtype=jnp.float32)), -5.0, 5.0).astype(target_dtype)
        gated_ms_seq = jnp.swapaxes(ig[0] * motor_surface_all, 0, 1)  # [T, B, D]
        gated_res_seq = jnp.swapaxes(ig[2] * resonance_all, 0, 1)     # [T, B, D]

        fused_norm_w = params["motor.fused_norm.weight"]
        fc_gate_w = params["motor.swiglu.fc_gate.weight"]
        fc_val_w = params["motor.swiglu.fc_val.weight"]
        fc_down_w = params["motor.swiglu.fc_down.weight"]
        norm_mid_w = params["motor.norm_mid.weight"]
        plan_gate_w_t = params["motor.plan_gate.weight"].T
        plan_gate_b = params["motor.plan_gate.bias"].astype(target_dtype)
        plan_decay = jnp.clip(params["motor.plan_decay"].astype(jnp.float32), 0.0, 0.92).astype(target_dtype)
        plan_write = params["motor.plan_write"].astype(target_dtype)
        plan_update_w_t = params["motor.plan_update.weight"].T
        eff_proj_t = params["motor.efference_proj.weight"].T
        eff_gate_t = params["core.efference_gate.weight"].T
        radial_norm_w = params["core.radial_norm.weight"]

        def _tf_motor_step(carry, xs):
            cur_core, cur_plan = carry
            ms_t, res_t, p_scaled_t = xs

            shell_pool = static_shell_pool + w0_shell_scale * jnp.matmul(cur_core, w_shell_t)
            fused_in = jnp.concatenate([
                ms_t,
                ig[1] * cur_core,
                res_t,
                ig[3] * shell_pool,
                ig[4] * cur_plan,
            ], axis=-1)
            fused = rms_norm(fused_in, fused_norm_w)
            h = rms_norm(swiglu(fused, fc_gate_w, fc_val_w, fc_down_w), norm_mid_w)

            phase_batch = jnp.broadcast_to(p_scaled_t[None, :], (B, surface_dim))
            plan_input = jnp.concatenate([h, phase_batch], axis=-1)
            p_gate = jax.nn.sigmoid(jnp.matmul(plan_input, plan_gate_w_t) + plan_gate_b)
            plan_update = jnp.matmul(plan_input, plan_update_w_t)
            nxt_plan = jnp.clip(plan_decay * cur_plan + plan_write * p_gate * plan_update, -50.0, 50.0).astype(target_dtype)

            efference = jnp.matmul(h, eff_proj_t)
            eff_projected = jnp.matmul(jax.nn.silu(efference), eff_gate_t)
            eff_normed = rms_norm(eff_projected, radial_norm_w).astype(target_dtype)
            nxt_core = (0.85 * cur_core + 0.15 * eff_normed).astype(target_dtype)

            return (nxt_core, nxt_plan), h

        motor_scan_fn = jax.checkpoint(_tf_motor_step) if use_remat else _tf_motor_step
        _, h_seq = lax.scan(motor_scan_fn, (init_core, init_plan), (gated_ms_seq, gated_res_seq, p_scaled_all))

        # 4. Chunked Systolic Vocabulary Projection & Cross-Entropy Outside Sequential Scan
        temp = jnp.clip(params["motor.temperature_scale"].astype(jnp.float32), 0.1, 5.0).astype(target_dtype)
        affine = params["motor.affine_scale"][None, None, :].astype(target_dtype)
        aff_th = params.get("motor.affine_think")
        aff_ac = params.get("motor.affine_act")
        aff_an = params.get("motor.affine_answer")
        if aff_th is not None and aff_ac is not None and aff_an is not None:
            combined_affine = (
                0.40 * affine
                + 0.20 * aff_th[None, None, :].astype(target_dtype)
                + 0.20 * aff_ac[None, None, :].astype(target_dtype)
                + 0.20 * aff_an[None, None, :].astype(target_dtype)
            )
        else:
            combined_affine = affine
        h_scaled = ((jnp.swapaxes(h_seq, 0, 1) * combined_affine) / temp).reshape(B * T, surface_dim)
        targets_flat = targets[:, :T].reshape(B * T)
        w_emb_t = params["sensory.embedding.weight"].T

        total_tokens = B * T
        chunk_tokens = total_tokens if total_tokens <= 4096 else next(
            (c for c in (4096, 3072, 2048, 1536, 1024, 768, 512, 384, 256, 192, 128, 64, 32, 16, 8, 4, 2, 1) if total_tokens % c == 0),
            total_tokens,
        )
        num_chunks = total_tokens // chunk_tokens
        weights_flat = target_weights[:, :T].reshape(B * T) if target_weights is not None else jnp.ones(total_tokens, dtype=jnp.float32)

        def _ce_loss_chunk(h_c, t_c, w_c):
            logits_c = jnp.matmul(h_c, w_emb_t).astype(jnp.float32)
            safe_t = jnp.maximum(t_c, 0)
            tgt_logits = jnp.take_along_axis(logits_c, safe_t[:, None], axis=-1).squeeze(-1)
            lse = jax.nn.logsumexp(logits_c, axis=-1)
            mask_c = (t_c >= 0).astype(jnp.float32)
            eff_w = mask_c * w_c
            loss_c = (lse - tgt_logits) * eff_w
            return jnp.sum(loss_c), jnp.sum(eff_w)

        if num_chunks == 1:
            loss_sum, mask_sum = _ce_loss_chunk(h_scaled, targets_flat, weights_flat)
        else:
            ce_fn = lambda _, inp: (None, _ce_loss_chunk(inp[0], inp[1], inp[2]))
            ce_scan = jax.checkpoint(ce_fn) if (use_remat or num_chunks > 2) else ce_fn
            h_chunks = h_scaled.reshape(num_chunks, chunk_tokens, surface_dim)
            t_chunks = targets_flat.reshape(num_chunks, chunk_tokens)
            w_chunks = weights_flat.reshape(num_chunks, chunk_tokens)
            _, (chunk_losses, chunk_masks) = lax.scan(ce_scan, None, (h_chunks, t_chunks, w_chunks))
            loss_sum = jnp.sum(chunk_losses)
            mask_sum = jnp.sum(chunk_masks)

        base_loss = loss_sum / jnp.maximum(mask_sum, 1.0)
        if ponder_cost is not None and getattr(config, "use_adaptive_thinking", False):
            base_loss = base_loss + float(getattr(config, "thinking_cost_weight", 0.01)) * ponder_cost

        # Supervise Temporal Router on prompt_flux
        w_r = params.get("motor.temporal_router.weight")
        b_r = params.get("motor.temporal_router.bias")
        if w_r is not None and b_r is not None:
            router_logits = jnp.matmul(prompt_flux.astype(jnp.float32), w_r.T) + b_r
            tgt_act = jnp.any(targets == 33, axis=-1).astype(jnp.float32)
            tgt_th = jnp.any(targets == 386, axis=-1).astype(jnp.float32)
            tgt_an = jnp.ones_like(tgt_act)
            tgt_gates = jnp.stack([tgt_th, tgt_act, tgt_an], axis=-1)
            max_val = jnp.maximum(router_logits, 0.0)
            router_bce = max_val - router_logits * tgt_gates + jnp.log1p(jnp.exp(-jnp.abs(router_logits)))
            gate_loss = jnp.mean(router_bce)
            return base_loss + 0.15 * gate_loss

        return base_loss

    scan_fn = jax.checkpoint(_emit) if use_remat else _emit

    if targets is not None:
        _, (all_loss, all_mask) = lax.scan(scan_fn, (state, init_refractory, init_plan, init_thought), jnp.arange(target_len))
        base_ce = jnp.sum(all_loss) / jnp.maximum(jnp.sum(all_mask), 1.0)
        if ponder_cost is not None and getattr(config, "use_adaptive_thinking", False):
            base_ce = base_ce + float(getattr(config, "thinking_cost_weight", 0.01)) * ponder_cost
        w_r = params.get("motor.temporal_router.weight")
        b_r = params.get("motor.temporal_router.bias")
        if w_r is not None and b_r is not None:
            router_logits = jnp.matmul(prompt_flux.astype(jnp.float32), w_r.T) + b_r
            tgt_act = jnp.any(targets == 33, axis=-1).astype(jnp.float32)
            tgt_th = jnp.any(targets == 386, axis=-1).astype(jnp.float32)
            tgt_an = jnp.ones_like(tgt_act)
            tgt_gates = jnp.stack([tgt_th, tgt_act, tgt_an], axis=-1)
            max_val = jnp.maximum(router_logits, 0.0)
            router_bce = max_val - router_logits * tgt_gates + jnp.log1p(jnp.exp(-jnp.abs(router_logits)))
            gate_loss = jnp.mean(router_bce)
            return base_ce + 0.15 * gate_loss
        return base_ce

    _, all_logits = lax.scan(scan_fn, (state, init_refractory, init_plan, init_thought), jnp.arange(target_len))
    return jnp.swapaxes(all_logits, 0, 1)


def compute_manifold_diagnostics_jax(state: jnp.ndarray) -> Dict[str, Any]:
    """
    Computes geometric diagnostics across all concentric spheroid shells in JAX.
    Tracks:
    - Shell RMS energy propagation (thalamic core -> cortical surface)
    - Inter-shell cosine similarity (radial wave convection smoothness)
    """
    state_f = state.astype(jnp.float32)
    num_shells = state.shape[1]
    rms_per_shell = [float(jnp.sqrt(jnp.mean(jnp.square(state_f[:, k, :])))) for k in range(num_shells)]
    adjacent_cosines = []
    for k in range(num_shells - 1):
        s_curr = state_f[:, k, :]
        s_next = state_f[:, k + 1, :]
        cos = jnp.mean(
            jnp.sum(s_curr * s_next, axis=-1)
            / (jnp.linalg.norm(s_curr, axis=-1) * jnp.linalg.norm(s_next, axis=-1) + 1e-6)
        )
        adjacent_cosines.append(float(cos))

    return {
        "mean_shell_rms": float(sum(rms_per_shell) / len(rms_per_shell)),
        "shell_rms": rms_per_shell,
        "mean_adjacent_cosine": float(sum(adjacent_cosines) / len(adjacent_cosines)) if adjacent_cosines else 1.0,
        "adjacent_cosines": adjacent_cosines,
    }


def generate_text_jax(
    params: Dict[str, jnp.ndarray],
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: ModelConfigV5,
    prompt: str,
    tokenizer: Any,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    repetition_penalty: float = 1.2,
    seed_token_id: Optional[int] = None,
    strict_act: bool = False,
    return_parsed: bool = False,
    images: Optional[jnp.ndarray] = None,
    audio: Optional[jnp.ndarray] = None,
    perceptual_inputs: Optional[jnp.ndarray] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Generates text using the 3-tier cognitive protocol in JAX:
      1. Thinking   : <think> ... </think> -> Private scratchpad
      2. Talking    : Unbounded plain text to EOS -> Fluid zero-overhead answering
      3. Tool Action: <act> ... </act> -> Structured external tool execution boundary
    """
    from src.corpus import (
        format_stimulus_prompt,
        clean_motor_response,
        encode_no_special,
        get_bio_token_ids,
        extract_strict_act_payload,
        parse_model_output,
    )

    bio_ids = get_bio_token_ids(tokenizer)
    seed = seed_token_id  # Unbiased generation: do not force <act> by default

    formatted_prompt = format_stimulus_prompt(prompt)
    prompt_ids = encode_no_special(tokenizer, formatted_prompt)
    prompts_arr = jnp.array([prompt_ids], dtype=jnp.int32)

    # Slot query single-token optimization
    slot_keywords = (
        "analogy",
        "::",
        "is to",
        "antonym",
        "opposite of",
        "synonym",
        "what is the capital",
        "what is the past tense",
        "past tense of",
        "plural of",
        "definition of",
        "define ",
    )
    is_slot = any(kw in prompt.lower() for kw in slot_keywords)
    effective_max_tokens = min(max_new_tokens, 48) if is_slot else max_new_tokens

    logits = training_rollout(
        params=params,
        prompts=prompts_arr,
        target_len=effective_max_tokens,
        buffers=buffers,
        config=config,
        repetition_penalty=repetition_penalty,
        seed_token_id=seed,
        images=images,
        audio=audio,
        perceptual_inputs=perceptual_inputs,
    )
    token_ids = [int(x) for x in jnp.argmax(logits[0], axis=-1)]

    if strict_act:
        return extract_strict_act_payload(token_ids, tokenizer)

    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    # Check text-level stops: ChatML <|im_end|>, </stim>, </s>
    for stop_str in ("<|im_end|>", "</stim>", "</s>"):
        if stop_str in decoded:
            decoded = decoded.split(stop_str)[0]

    parsed = parse_model_output(decoded)
    if return_parsed:
        return parsed

    return parsed["answer"] if parsed["answer"] else (parsed["thinking"] if parsed["thinking"] else clean_motor_response(decoded))


def emit_tri_head_logits_jax(
    params: Dict[str, jnp.ndarray],
    h: jnp.ndarray,
    temp: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simultaneously computes vocabulary logits across all three cognitive tiers in JAX:
      1. logits_think : Deliberation / Reasoning scratchpad (<think>)
      2. logits_act   : Non-conversational actions (Coding, Calculator, Tools) (<act>)
      3. logits_answer: Conversational dialogue / Response (Plain Text)
    """
    w_emb = params["sensory.embedding.weight"]
    aff_think = params.get("motor.affine_think", params["motor.affine_scale"])
    aff_act = params.get("motor.affine_act", params["motor.affine_scale"])
    aff_ans = params.get("motor.affine_answer", params["motor.affine_scale"])

    l_think = jnp.matmul((h * aff_think) / temp, w_emb.T)
    l_act = jnp.matmul((h * aff_act) / temp, w_emb.T)
    l_ans = jnp.matmul((h * aff_ans) / temp, w_emb.T)
    return l_think, l_act, l_ans


def compute_temporal_gates_jax(
    params: Dict[str, jnp.ndarray],
    h: jnp.ndarray,
    action_trigger: Optional[bool] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes dynamic temporal emission probabilities for Think, Act, and Answer in JAX.
    Returns:
      (gate_think, gate_act, gate_answer) in [0.0, 1.0]
    """
    w_r = params.get("motor.temporal_router.weight")
    b_r = params.get("motor.temporal_router.bias")
    if w_r is not None and b_r is not None:
        h_feat = h if h.ndim == 2 else jnp.reshape(h, (-1, w_r.shape[1]))
        logits = jnp.matmul(h_feat, w_r.T) + b_r
        if action_trigger is True:
            logits = logits + jnp.array([0.0, 3.0, 0.0])
        elif action_trigger is False:
            logits = logits - jnp.array([0.0, 4.0, 0.0])
        gates = jax.nn.sigmoid(logits)
        return gates[:, 0], gates[:, 1], gates[:, 2]
    # Default fallback: Think active (0.8), Act selective (0.2), Answer active (0.95)
    return jnp.array([0.8]), jnp.array([0.2]), jnp.array([0.95])


def generate_tri_stream_jax(
    params: Dict[str, jnp.ndarray],
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: ModelConfigV5,
    prompt: str,
    tokenizer: Any,
    max_new_tokens: int = 128,
    temperature: float = 0.6,
    repetition_penalty: float = 1.15,
) -> Dict[str, Any]:
    """
    Simultaneous Tri-Head Cognitive Emission in JAX:
    Emits all 3 streams simultaneously from the shared manifold hidden state:
      - Head 1 (Think)  : Emits cognitive reasoning scratchpad (<think>)
      - Head 2 (Act)    : Emits non-conversational code / calculator / tool actions (<act>)
      - Head 3 (Answer) : Emits natural conversational answer (Plain Text)
    """
    import numpy as np
    from src.corpus import format_stimulus_prompt, encode_no_special

    formatted_prompt = format_stimulus_prompt(prompt)
    prompt_ids = encode_no_special(tokenizer, formatted_prompt)
    prompts_arr = jnp.array([prompt_ids], dtype=jnp.int32)
    B, L = prompts_arr.shape
    total_prompt_L = L
    target_dtype = params["sensory.embedding.weight"].dtype

    # Check prompt semantics for action triggers
    p_lower = prompt.lower()
    has_action_trigger = any(kw in p_lower for kw in (
        "python", "code", "script", "function", "def ", "calculate", "solve", "+", "-", "*", "/",
        "sqrt", "sql", "select ", "tool", "bash", "grep", "curl", "regex", "test", "assert", "datetime", "mermaid"
    ))
    is_pure_conversation = any(p_lower.strip().startswith(kw) for kw in (
        "hi", "hello", "hey", "good morning", "good evening", "how are you", "who are you",
        "what is earth", "why is the sky", "what is a pixel", "what is a computer port"
    )) and not has_action_trigger
    action_intent = True if has_action_trigger else (False if is_pure_conversation else None)

    # 1. Sensory encoding
    sensory_flux, (sensory_keys, sensory_values) = encode_sensory(params, prompts_arr, config=config, pos_offset=0)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    mask_2d = (prompts_arr != pad_id).astype(jnp.float32)
    pos_idx = jnp.arange(L, dtype=jnp.float32)[None, :]
    last_valid_pos = jnp.max(jnp.where(mask_2d > 0.0, pos_idx, 0.0), axis=1, keepdims=True)
    recency_weights = jnp.exp(0.05 * (pos_idx - last_valid_pos)) * mask_2d
    weight_sum = jnp.maximum(jnp.sum(recency_weights, axis=1, keepdims=True), 1.0)
    prompt_centroid = (
        jnp.sum(sensory_flux.astype(jnp.float32) * recency_weights[:, :, None], axis=1) / weight_sum
    ).astype(target_dtype)

    # 2. Autonomous Temporal Decision: the model dynamically decides which heads emit
    g_th, g_ac, g_an = compute_temporal_gates_jax(params, prompt_centroid, action_trigger=action_intent)
    gate_think_val = float(g_th[0])
    gate_act_val = float(g_ac[0])
    gate_ans_val = float(g_an[0])

    emit_think = gate_think_val >= 0.35
    emit_act = gate_act_val >= 0.45
    emit_ans = gate_ans_val >= 0.35

    # 3. Concentric state & subconscious deliberation cycles
    num_shells = config.num_shells
    surface_dim = config.surface_dim
    vocab_size = config.vocab_size
    thinking_cycles = config.thinking_cycles

    init_state = jnp.zeros((B, num_shells, surface_dim), dtype=target_dtype)
    state = init_state.at[:, -1, :].set(prompt_centroid)
    state, _ = deliberate_cycles_jax(params, state, buffers, config=config, is_training=False)

    plan = jnp.zeros((B, surface_dim), dtype=target_dtype)
    refractory = jnp.zeros((B, vocab_size), dtype=target_dtype)
    eos_id = getattr(tokenizer, "eos_token_id", 2) or 2

    tokens_think, tokens_act, tokens_ans = [], [], []
    fin_think = not emit_think
    fin_act = not emit_act
    fin_ans = not emit_ans

    # Keep an autoregressive KV cache with silence/BOS slot 0 prepended
    # to match teacher-forcing layout (masked out from attention).
    dummy_pad = jnp.zeros((B, 1), dtype=jnp.int32)
    _, (sil_k, sil_v) = encode_sensory(params, dummy_pad, config=config, pos_offset=total_prompt_L, is_causal=True)
    causal_keys = jnp.concatenate([sensory_keys, sil_k], axis=2)
    causal_values = jnp.concatenate([sensory_values, sil_v], axis=2)
    all_valid = jnp.concatenate([
        (prompts_arr != pad_id)[:, None, None, :],
        jnp.zeros((B, 1, 1, 1), dtype=bool)
    ], axis=-1)

    def _sample_from_logits(lg: jnp.ndarray, prev_toks: List[int]) -> int:
        lg_np = np.array(lg[0], dtype=np.float32)
        if prev_toks:
            for pt in set(prev_toks):
                if lg_np[pt] > 0:
                    lg_np[pt] /= repetition_penalty
                else:
                    lg_np[pt] *= repetition_penalty
        lg_np = lg_np / max(temperature, 1e-4)
        exp_lg = np.exp(lg_np - np.max(lg_np))
        probs = exp_lg / np.sum(exp_lg)
        return int(np.random.choice(len(probs), p=probs))

    for t in range(max_new_tokens):
        # The cache contains the prompt plus *previous* generated tokens.
        # Every valid cached key is causal at this point; the newly sampled
        # token is appended only after this emission step.
        step_mask = all_valid

        last_tok = None
        if not fin_ans and tokens_ans:
            last_tok = tokens_ans[-1]
        elif not fin_act and tokens_act:
            last_tok = tokens_act[-1]
        elif not fin_think and tokens_think:
            last_tok = tokens_think[-1]
        feed = params["sensory.embedding.weight"][jnp.array([last_tok])] if last_tok is not None else None

        _, next_state, refractory, plan, h = emit_step(
            params=params,
            state=state,
            sensory_kv=(causal_keys, causal_values),
            plan=plan,
            refractory_trace=refractory,
            step_idx=t,
            buffers=buffers,
            config=config,
            auditory_feedback=feed,
            repetition_penalty=repetition_penalty,
            attn_mask=step_mask,
            return_hidden=True,
            query_position=total_prompt_L + t,
        )

        l_th, l_ac, l_an = emit_tri_head_logits_jax(params, h, temp=temperature)

        if not fin_think:
            tok_th = _sample_from_logits(l_th, tokens_think)
            if tok_th == eos_id:
                fin_think = True
            else:
                tokens_think.append(tok_th)

        if not fin_act:
            tok_ac = _sample_from_logits(l_ac, tokens_act)
            if tok_ac == eos_id:
                fin_act = True
            else:
                tokens_act.append(tok_ac)

        if not fin_ans:
            tok_an = _sample_from_logits(l_an, tokens_ans)
            if tok_an == eos_id:
                fin_ans = True
            else:
                tokens_ans.append(tok_an)

        if fin_think and fin_act and fin_ans:
            break

        # The recurrent manifold is shared by the three display streams.  Its
        # next-step feedback prioritizes answer, then action, then thought;
        # cache exactly that same token so attention and feedback describe one
        # coherent autoregressive trajectory.
        emitted_tok = tok_an if not fin_ans else (tok_ac if not fin_act else tok_th)
        _, (new_k, new_v) = encode_sensory(
            params,
            jnp.asarray([[emitted_tok]], dtype=jnp.int32),
            config=config,
            pos_offset=total_prompt_L + t + 1,
            is_causal=True,
        )
        causal_keys = jnp.concatenate([causal_keys, new_k], axis=2)
        causal_values = jnp.concatenate([causal_values, new_v], axis=2)
        all_valid = jnp.concatenate(
            [all_valid, jnp.ones((B, 1, 1, 1), dtype=bool)], axis=-1
        )

        state = next_state

    text_think = tokenizer.decode(tokens_think, skip_special_tokens=True).strip() if emit_think else ""
    text_act = tokenizer.decode(tokens_act, skip_special_tokens=True).strip() if emit_act else ""
    text_ans = tokenizer.decode(tokens_ans, skip_special_tokens=True).strip() if emit_ans else ""

    return {
        "thinking": text_think,
        "action": text_act,
        "answer": text_ans,
        "temporal_decisions": {
            "think": emit_think,
            "act": emit_act,
            "answer": emit_ans,
            "gates": {
                "p_think": gate_think_val,
                "p_act": gate_act_val,
                "p_ans": gate_ans_val,
            }
        }
    }
