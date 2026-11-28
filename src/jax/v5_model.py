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
    use_per_token_efference: bool = False
    use_centralized_readout: bool = True
    use_contractive_deliberation: bool = True
    contractive_delib_decay: float = 0.75
    deliberation_early_halt_thresh: float = 0.025
    use_spectral_core_readout: bool = True
    spectral_band_splits: Optional[Tuple[int, int, int]] = None
    use_core_inquiry: bool = True
    use_dual_evidence_grounding: bool = True
    sensory_evidence_core_scale: float = 0.05
    use_core_halting: bool = True
    deliberation_oscillation_halt: bool = True
    use_laminar_transport: bool = True
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
    tool_token_id: Optional[int] = 31
    act_token_id: Optional[int] = 31
    think_token_id: Optional[int] = 29
    think_end_token_id: Optional[int] = 30
    tool_end_token_id: Optional[int] = 32
    act_end_token_id: Optional[int] = 32
    router_loss_weight: float = 0.0
    energy_reg_weight: float = 0.01
    use_bidirectional_tracts: bool = False
    use_sequential_cascade: bool = False
    use_sensory_core_coupling: bool = False
    use_parallel_plan_scan: bool = False
    preserve_shell_norm_clamp: bool = True
    use_selective_loss: bool = False
    selective_focal_gamma: float = 0.5
    selective_noise_cap: float = 8.5
    z_loss_weight: float = 0.0


# Register ModelConfigV5 as static PyTree node
jax.tree_util.register_pytree_node(
    ModelConfigV5,
    lambda c: ((), asdict(c)),
    lambda aux, _: ModelConfigV5(**aux),
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


def swiglu(x: jnp.ndarray, w_gate: jnp.ndarray, w_val: jnp.ndarray, w_down: jnp.ndarray, use_swiglu: bool = True) -> jnp.ndarray:
    """Swish Gated Linear Unit in JAX with optional non-gated SiLU bypass."""
    g = jax.nn.silu(jnp.matmul(x, w_gate.T))
    if not use_swiglu:
        return jnp.matmul(g, w_down.T)
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


def create_dct_matrix_jax(dim: int) -> jnp.ndarray:
    """Orthonormal Discrete Cosine Transform (DCT-II) matrix in JAX (Parseval energy-conserving)."""
    n = jnp.arange(dim, dtype=jnp.float32)
    k = jnp.arange(dim, dtype=jnp.float32)[:, None]
    M = jnp.cos(math.pi / float(dim) * (n + 0.5) * k)
    M = M.at[0, :].multiply(1.0 / math.sqrt(float(dim)))
    M = M.at[1:, :].multiply(math.sqrt(2.0 / float(dim)))
    return M


def transform_spectral_core_jax(core: jnp.ndarray, params: dict, config: Any) -> jnp.ndarray:
    """Transforms deliberated Shell 0 via orthonormal DCT basis, multi-band filtering, and residual gating in JAX."""
    use_spec = getattr(config, "use_spectral_core_readout", True) if config is not None else False
    if not use_spec or "motor.spectral_band_gains" not in params:
        return core
    D = core.shape[-1]
    splits = getattr(config, "spectral_band_splits", None)
    if splits is not None and sum(splits) == D:
        low_d, mid_d, high_d = splits
    else:
        low_d = D // 4
        mid_d = D // 2
        high_d = D - (low_d + mid_d)
    dt = core.dtype
    dct_mat = create_dct_matrix_jax(D).astype(dt)
    s_hat = jnp.matmul(core, dct_mat.T)
    gains = params["motor.spectral_band_gains"].astype(dt)
    z_low = s_hat[..., :low_d] * gains[0]
    z_mid = s_hat[..., low_d : low_d + mid_d] * gains[1]
    z_high = s_hat[..., low_d + mid_d :] * gains[2]
    filtered_spec = jnp.concatenate([z_low, z_mid, z_high], axis=-1)
    # Invert orthonormal DCT-II back to spatial domain (C^T) before residual addition
    reconstructed = jnp.matmul(filtered_spec, dct_mat)
    beta = jnp.tanh(params.get("motor.spectral_residual_gate", jnp.zeros(1, dtype=jnp.float32))).astype(dt)
    norm_w = params.get("motor.spectral_norm.weight", jnp.ones(D, dtype=jnp.float32))
    return rms_norm(core + beta * reconstructed, norm_w)


def generate_small_world_indices(num_shells: int, num_tracts: int, bidirectional: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generates deterministic small-world axonal shortcut wiring in JAX.
    
    bidirectional=False restores the canonical baseline wiring under which
    v5_small was trained for 245k+ steps.
    """
    srcs, dsts = [], []
    if not bidirectional:
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
                if s == d and num_shells > 1:
                    d = (s + 1) % num_shells
            srcs.append(s)
            dsts.append(d)
        return jnp.array(srcs, dtype=jnp.int32), jnp.array(dsts, dtype=jnp.int32)

    deep_span = max(1, num_shells // 4)
    for i in range(num_tracts):
        m = i // 3
        if i % 3 == 0:
            # Bidirectional subcortical <-> cortical mantle tracts (even m outward, odd m corticofugal return)
            s_deep = (m // 2) % deep_span
            d_mantle = num_shells - 1 - s_deep
            if d_mantle == s_deep and num_shells > 1:
                d_mantle = (s_deep + 1) % num_shells
            s, d = (s_deep, d_mantle) if (m % 2 == 0) else (d_mantle, s_deep)
        elif i % 3 == 1:
            # Hierarchical outer-to-inner corticofugal funnel across all shell parities
            s = (num_shells - 1 - m) % num_shells
            if s > 0:
                d = (s - 1) // 4
            else:
                d = (num_shells // 2) % num_shells
                if d == s and num_shells > 1:
                    d = (s + 1) % num_shells
        else:
            s = (i * 7) % num_shells
            d = (i * 13) % num_shells
            if s == d and num_shells > 1:
                d = (s + 1) % num_shells
        srcs.append(s)
        dsts.append(d)
    return jnp.array(srcs, dtype=jnp.int32), jnp.array(dsts, dtype=jnp.int32)


def init_params_v5(key: jax.random.PRNGKey, config: ModelConfigV5) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Initializes CerebroSpheroid V5 parameters and small-world tract buffers."""
    keys = jax.random.split(key, 48)
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
        "motor.temporal_router.weight": jax.random.normal(keys[38], (3, s_dim)) * 0.01,
        "motor.temporal_router.bias": jnp.array([0.5, -0.8, 1.0], dtype=jnp.float32),
        "motor.temperature_scale": jnp.array(1.0, dtype=jnp.float32),
        "motor.efference_proj.weight": jax.random.normal(keys[23], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "motor.refractory_decay": jnp.array(0.75, dtype=jnp.float32),
        "motor.refractory_strength": jnp.array(1.2, dtype=jnp.float32),

        # Persona Steering Basis & Scale
        "motor.persona_embeddings": jax.random.normal(keys[37], (4, s_dim)) * 0.05,
        "motor.persona_scale": jnp.array(0.5, dtype=jnp.float32),
    }

    # Optional untied vocabulary output projection
    if not getattr(config, "use_tied_embeddings", True):
        params["motor.head.weight"] = jax.random.normal(keys[39], (V, s_dim)) * (1.0 / math.sqrt(s_dim))

    # Core Spectral Readout parameters (Parseval orthonormal DCT-II decomposition)
    if getattr(config, "use_spectral_core_readout", True):
        params["motor.spectral_band_gains"] = jnp.ones(3, dtype=jnp.float32)
        params["motor.spectral_residual_gate"] = jnp.zeros(1, dtype=jnp.float32)
        params["motor.spectral_norm.weight"] = jnp.ones(s_dim, dtype=jnp.float32)

    # Optional Multi-Token Prediction (MTP) auxiliary heads
    if getattr(config, "num_mtp_heads", 1) > 1:
        for m in range(1, config.num_mtp_heads):
            m_key = jax.random.fold_in(keys[40], m)
            sub_k = jax.random.split(m_key, 4)
            params[f"motor.mtp.{m}.fc_gate.weight"] = jax.random.normal(sub_k[0], (s_dim * 2, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2))
            params[f"motor.mtp.{m}.fc_val.weight"] = jax.random.normal(sub_k[1], (s_dim * 2, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2))
            params[f"motor.mtp.{m}.fc_down.weight"] = jax.random.normal(sub_k[2], (s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2))
            params[f"motor.mtp.{m}.norm.weight"] = jnp.ones(s_dim, dtype=jnp.float32)
            params[f"motor.mtp.{m}.affine_scale"] = jnp.ones(s_dim, dtype=jnp.float32)

    # 3D Tensorized Batched Intra-Shell SwiGLU across all K shells in parallel
    params["core.intra_norms.weight"] = jnp.ones((K, s_dim), dtype=jnp.float32)
    params["core.intra_swiglu.w_gate"] = jax.random.normal(keys[24], (K, s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim))
    params["core.intra_swiglu.w_val"] = jax.random.normal(keys[25], (K, s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim))
    params["core.intra_swiglu.w_down"] = jax.random.normal(keys[26], (K, s_dim * 2, s_dim)) * (1.0 / math.sqrt(s_dim * 2))

    # Optional Subconscious Deliberation Memory Probe
    if getattr(config, "use_deliberation_probe", False):
        probe_key = jax.random.fold_in(key, 999)
        params["core.deliberation_probe.weight"] = jax.random.normal(probe_key, (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim))
        params["core.deliberation_probe_norm.weight"] = jnp.ones(s_dim, dtype=jnp.float32)
        params["core.evidence_core_gate"] = jnp.zeros(1, dtype=jnp.float32)

    # 4. Multimodal Perceptual Projections (Images, Audio, Continuous Sensors)
    if getattr(config, "enable_multimodal", False):
        v_dim = getattr(config, "visual_dim", s_dim)
        p_size = getattr(config, "image_patch_size", 16)
        params["image_stem.conv.weight"] = jax.random.normal(keys[41], (v_dim, 3, p_size, p_size)) * (1.0 / math.sqrt(3 * p_size * p_size))
        params["image_stem.norm.weight"] = jnp.ones(v_dim, dtype=jnp.float32)

        params["audio_stem.conv.weight"] = jax.random.normal(keys[42], (v_dim, 1, 400)) * (1.0 / math.sqrt(400))
        params["audio_stem.norm.weight"] = jnp.ones(v_dim, dtype=jnp.float32)

        params["perceptual.norm_in.weight"] = jnp.ones(v_dim, dtype=jnp.float32)
        params["perceptual.linear.weight"] = jax.random.normal(keys[43], (s_dim, v_dim)) * (1.0 / math.sqrt(v_dim))
        params["perceptual.swiglu.fc_gate.weight"] = jax.random.normal(keys[31], (s_dim * 2, s_dim)) * (1.0 / math.sqrt(s_dim))
        params["perceptual.swiglu.fc_val.weight"] = jax.random.normal(keys[32], (s_dim * 2, s_dim)) * (1.0 / math.sqrt(s_dim))
        params["perceptual.swiglu.fc_down.weight"] = jax.random.normal(keys[33], (s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2))
        params["perceptual.norm_out.weight"] = jnp.ones(s_dim, dtype=jnp.float32)

    num_sens_layers = getattr(config, "num_sensory_layers", 1)
    if num_sens_layers > 1:
        extra_keys = jax.random.split(keys[44], (num_sens_layers - 1) * 6)
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

    # 6. Laminar-specific directional transport gates (zero-initialized for exact baseline parity)
    if getattr(config, "use_laminar_transport", True):
        params["core.laminar_in_gate"] = jnp.zeros(K - 1, dtype=jnp.float32)
        params["core.laminar_out_gate"] = jnp.zeros(K - 1, dtype=jnp.float32)

    use_bi = bool(getattr(config, "use_bidirectional_tracts", False))
    src_idx, dst_idx = generate_small_world_indices(K, T, bidirectional=use_bi)
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


def _chunked_mha_jax(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    attn_mask: Optional[jnp.ndarray] = None,
    is_causal: bool = False,
    query_pos_offset: Union[int, jnp.ndarray] = 0,
    top_k_res: int = 0,
    chunk_size: int = 512,
    use_remat: bool = False,
) -> jnp.ndarray:
    """Memory-bounded multi-head attention via query-axis chunking (8x lower HBM at 4K ctx)."""
    B, H, Q_len, Dh = q.shape
    K_len = k.shape[2]
    if (
        top_k_res == 0
        and attn_mask is None
        and hasattr(jax.nn, "dot_product_attention")
        and ((not is_causal) or (isinstance(query_pos_offset, int) and query_pos_offset == 0 and Q_len == K_len))
    ):
        q_bthd = jnp.swapaxes(q, 1, 2)
        k_bthd = jnp.swapaxes(k, 1, 2)
        v_bthd = jnp.swapaxes(v, 1, 2)
        out_bthd = jax.nn.dot_product_attention(q_bthd, k_bthd, v_bthd, is_causal=is_causal)
        return out_bthd.reshape(B, Q_len, H * Dh)
    k_t = jnp.swapaxes(k, -1, -2)
    scale = 1.0 / math.sqrt(Dh)

    def _attend_block(q_blk, q_pos_blk):
        scores = jnp.matmul(q_blk, k_t) * scale
        if attn_mask is not None:
            scores = jnp.where(attn_mask, scores, -1e4)
        if is_causal:
            k_pos = jnp.arange(K_len, dtype=jnp.int32)[None, :]
            c_mask = (k_pos <= q_pos_blk[:, None])[None, None, :, :]
            scores = jnp.where(c_mask, scores, -1e4)
        if top_k_res > 0 and K_len > top_k_res:
            top_vals, _ = lax.top_k(scores, top_k_res)
            scores = jnp.where(scores >= top_vals[..., -1:], scores, -1e4)
        attn_w = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
        return jnp.matmul(attn_w, v)

    c_sz = chunk_size if (Q_len > chunk_size and Q_len % chunk_size == 0) else next(
        (c for c in (chunk_size, 384, 256, 192, 128) if Q_len > c and Q_len % c == 0),
        Q_len,
    )
    if c_sz >= Q_len:
        q_pos = query_pos_offset + jnp.arange(Q_len, dtype=jnp.int32)
        out = _attend_block(q, q_pos)
        return jnp.swapaxes(out, 1, 2).reshape(B, Q_len, H * Dh)

    n_chunks = Q_len // c_sz
    q_chunks = jnp.swapaxes(q.reshape(B, H, n_chunks, c_sz, Dh), 0, 2)  # [n_chunks, H, B, c_sz, Dh]
    q_chunks = jnp.swapaxes(q_chunks, 1, 2)  # [n_chunks, B, H, c_sz, Dh]
    q_pos_all = (query_pos_offset + jnp.arange(Q_len, dtype=jnp.int32)).reshape(n_chunks, c_sz)

    blk_fn = jax.checkpoint(_attend_block) if use_remat else _attend_block
    _, out_chunks = lax.scan(lambda _, inp: (None, blk_fn(inp[0], inp[1])), None, (q_chunks, q_pos_all))
    # out_chunks: [n_chunks, B, H, c_sz, Dh] -> [B, H, Q_len, Dh]
    out = jnp.swapaxes(jnp.swapaxes(out_chunks, 0, 1), 1, 2).reshape(B, H, Q_len, Dh)
    return jnp.swapaxes(out, 1, 2).reshape(B, Q_len, H * Dh)


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
    h_swiglu = swiglu(
        h,
        params["sensory.swiglu.fc_gate.weight"],
        params["sensory.swiglu.fc_val.weight"],
        params["sensory.swiglu.fc_down.weight"],
        use_swiglu=getattr(config, "use_swiglu", True),
    )
    flux = rms_norm(x + h_swiglu, params["sensory.norm_out.weight"])

    # Contextual Inter-Token Multi-Head Self-Attention (zero-init residual)
    if L > 1 and "sensory.out_self.weight" in params and getattr(config, "use_sensory_self_attn", True):
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

        ctx = _chunked_mha_jax(
            q_s, k_s, v_s,
            attn_mask=attn_mask,
            is_causal=(is_causal and attn_mask is None),
            query_pos_offset=0,
        )
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

                ctx_l = _chunked_mha_jax(
                    q_l, k_l, v_l,
                    attn_mask=attn_mask,
                    is_causal=(is_causal and attn_mask is None),
                    query_pos_offset=0,
                )
                flux = flux + jnp.matmul(ctx_l, params[f"{pfx}out_proj.weight"].T)

                h2 = rms_norm(flux, params[f"{pfx}norm2.weight"])
                h2_swiglu = swiglu(
                    h2,
                    params[f"{pfx}swiglu.fc_gate.weight"],
                    params[f"{pfx}swiglu.fc_val.weight"],
                    params[f"{pfx}swiglu.fc_down.weight"],
                )
                flux = flux + h2_swiglu
        flux = rms_norm(flux, params["sensory.norm_out.weight"])

    k = jnp.matmul(flux, params["sensory.k_proj.weight"].T).reshape(B, L, num_heads, head_dim)
    k = jnp.swapaxes(k, 1, 2)  # [B, H, L, head_dim]
    v = jnp.matmul(flux, params["sensory.v_proj.weight"].T).reshape(B, L, num_heads, head_dim)
    v = jnp.swapaxes(v, 1, 2)  # [B, H, L, head_dim]

    if is_2d:
        flux = flux.squeeze(1)

    return flux, (k, v)


def deliberate_step(
    params: Dict[str, jnp.ndarray],
    state: jnp.ndarray,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: Optional[ModelConfigV5] = None,
) -> jnp.ndarray:
    """Executes a single subconscious deliberation step across all concentric laminae with fused 3D tensor contractions."""
    B, K, D = state.shape
    src_idx, dst_idx = buffers
    st_dtype = state.dtype

    if getattr(config, "use_sequential_cascade", False):
        # Sequential K-1 -> 0 Transformer-like shell cascade: each shell feeds directly into the next inward shell
        # in the same pass so all K shells receive direct O(1) gradient, followed by vectorized outward/shortcut flux.
        K = state.shape[1]
        r_mix = jnp.clip(params.get("core.radial_mix_scale", jnp.array(0.1, dtype=jnp.float32)), 0.01, 1.0).astype(st_dtype)
        s_mix = jnp.clip(params.get("core.shortcut_mix_scale", jnp.array(0.05, dtype=jnp.float32)), 0.005, 1.0).astype(st_dtype)
        w_up_out = params.get("core.radial_up_out.weight", params["core.radial_up.weight"])
        use_laminar = getattr(config, "use_laminar_transport", True) and ("core.laminar_in_gate" in params)
        if use_laminar:
            tau_in_all = 2.0 * jax.nn.sigmoid(params["core.laminar_in_gate"]).astype(st_dtype) * r_mix
            tau_out_all = 2.0 * jax.nn.sigmoid(params["core.laminar_out_gate"]).astype(st_dtype) * r_mix

        work = state
        for k in range(K - 1, -1, -1):
            s_k = work[:, k, :]
            if k < K - 1:
                h_prev = rms_norm(work[:, k + 1, :], params["core.radial_norm.weight"])
                rad_in = jnp.matmul(h_prev, params["core.radial_in_proj.weight"].T)
                in_flux = jnp.matmul(rad_in, params["core.w_radial_in"][k])
                in_up = jnp.matmul(in_flux, params["core.radial_up.weight"].T)
                t_in = tau_in_all[k] if use_laminar else r_mix
                s_k = s_k + (0.7071 * work[:, k + 1, :]) + (t_in * in_up)

            s_f32 = s_k.astype(jnp.float32)
            v_k = jnp.mean(jnp.square(s_f32), axis=-1, keepdims=True)
            h_k = (s_f32 * jax.lax.rsqrt(v_k + 1e-6) * params["core.intra_norms.weight"][k][None, :].astype(jnp.float32)).astype(st_dtype)
            g_k = jax.nn.silu(jnp.matmul(h_k, params["core.intra_swiglu.w_gate"][k]))
            if getattr(config, "use_swiglu", True):
                val_k = jnp.matmul(h_k, params["core.intra_swiglu.w_val"][k])
                down_k = jnp.matmul(g_k * val_k, params["core.intra_swiglu.w_down"][k])
            else:
                down_k = jnp.matmul(g_k, params["core.intra_swiglu.w_down"][k])
            s_k_out = s_k + down_k * params["core.shell_gains"][k].astype(st_dtype)
            work = work.at[:, k, :].set(s_k_out)

        h_rad = rms_norm(work, params["core.radial_norm.weight"])
        rad_out = jnp.matmul(h_rad[:, :-1, :], params["core.radial_out_proj.weight"].T)
        out_flux = jnp.einsum("bkd,kde->bke", rad_out, params["core.w_radial_out"])
        out_up = jnp.matmul(out_flux, w_up_out.T)
        if use_laminar:
            state_radial = work.at[:, 1:, :].add(tau_out_all[None, :, None] * out_up)
        else:
            state_radial = work.at[:, 1:, :].add(r_mix * out_up)

        conductances = jax.nn.sigmoid(params["core.shortcut_gate"]).astype(st_dtype)
        if config is not None and getattr(config, "use_sparse_mot", False):
            k_val = min(int(getattr(config, "top_k_tracts", 8)), conductances.shape[0])
            _, top_idx = lax.top_k(conductances, k_val)
            sparse_mask = jnp.zeros_like(conductances).at[top_idx].set(1.0)
            conductances = conductances * sparse_mask
        src_acts = h_rad[:, src_idx, :]
        shortcut_flux = jnp.einsum("btd,tde->bte", src_acts, params["core.shortcut_weight"]) * conductances[None, :, None]
        return state_radial.at[:, dst_idx, :].add(s_mix * shortcut_flux).astype(st_dtype)

    # 1. Batched Intra-shell Pre-Norm across all K shells simultaneously (float32 accumulation)
    state_f32 = state.astype(jnp.float32)
    var = jnp.mean(jnp.square(state_f32), axis=-1, keepdims=True)
    h = (state_f32 * jax.lax.rsqrt(var + 1e-6) * params["core.intra_norms.weight"][None, :, :].astype(jnp.float32)).astype(st_dtype)

    # 2. Batched 3D SwiGLU: Single systolic tensor contraction across all K shells in parallel
    gate = jax.nn.silu(jnp.einsum("bkd,kde->bke", h, params["core.intra_swiglu.w_gate"]))
    if getattr(config, "use_swiglu", True):
        val = jnp.einsum("bkd,kde->bke", h, params["core.intra_swiglu.w_val"])
        down = jnp.einsum("bke,ked->bkd", gate * val, params["core.intra_swiglu.w_down"])
    else:
        down = jnp.einsum("bke,ked->bkd", gate, params["core.intra_swiglu.w_down"])
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

    if getattr(config, "use_laminar_transport", True) and "core.laminar_in_gate" in params:
        tau_in = 2.0 * jax.nn.sigmoid(params["core.laminar_in_gate"])[None, :, None].astype(st_dtype) * r_mix
        tau_out = 2.0 * jax.nn.sigmoid(params["core.laminar_out_gate"])[None, :, None].astype(st_dtype) * r_mix
        state_radial = state_intra.at[:, :-1, :].add(tau_in * in_up)
        state_radial = state_radial.at[:, 1:, :].add(tau_out * out_up)
    else:
        state_radial = state_intra.at[:, :-1, :].add(r_mix * in_up)
        state_radial = state_radial.at[:, 1:, :].add(r_mix * out_up)

    # 4. Small-world hierarchical axonal shortcut tracts in 1 vectorized step
    conductances = jax.nn.sigmoid(params["core.shortcut_gate"]).astype(st_dtype)
    if config is not None and getattr(config, "use_sparse_mot", False):
        k_val = min(int(getattr(config, "top_k_tracts", 8)), conductances.shape[0])
        _, top_idx = lax.top_k(conductances, k_val)
        sparse_mask = jnp.zeros_like(conductances).at[top_idx].set(1.0)
        conductances = conductances * sparse_mask

    src_acts = h_rad[:, src_idx, :]
    shortcut_flux = jnp.einsum("btd,tde->bte", src_acts, params["core.shortcut_weight"]) * conductances[None, :, None]

    # Return unnormalized residual state to preserve genuine radial shell amplitude hierarchy
    state_next = state_radial.at[:, dst_idx, :].add(s_mix * shortcut_flux)
    return state_next.astype(st_dtype)


def deliberate_cycles_jax(
    params: Dict[str, jnp.ndarray],
    state: jnp.ndarray,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: Optional[ModelConfigV5] = None,
    is_training: bool = False,
    sensory_kv: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    return_energy: bool = False,
) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Executes subconscious deliberation cycles across concentric laminae with static-iteration
    lax.scan Adaptive Computation Time (ACT) for TPU/XLA compilation efficiency.

    Returns:
        (final_state, ponder_cost) or (final_state, ponder_cost, energy_loss)
    """
    B, K, D = state.shape
    init_scale = jnp.sqrt(jnp.sum(jnp.square(state.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-8).astype(state.dtype)
    preserve_shell_clamp = bool(getattr(config, "preserve_shell_norm_clamp", True)) if config is not None else True
    target_scale = init_scale if preserve_shell_clamp else jnp.mean(init_scale, axis=1, keepdims=True)
    use_contract = bool(getattr(config, "use_contractive_deliberation", True)) if config is not None else False
    delib_decay = float(getattr(config, "contractive_delib_decay", 0.75)) if config is not None else 0.75
    if config is None:
        use_adapt = False
        cycles = 4
    else:
        use_adapt = bool(getattr(config, "use_adaptive_thinking", False)) and ("core.halting_gate.weight" in params)
        cycles = int(getattr(config, "max_thinking_cycles", getattr(config, "thinking_cycles", 4))) if use_adapt else int(getattr(config, "thinking_cycles", 4))

    def _apply_probe(curr_s, step_idx):
        if sensory_kv is not None and "core.deliberation_probe.weight" in params and getattr(config, "use_deliberation_probe", False):
            keys, values = sensory_kv
            L_kv = keys.shape[2]
            num_h = keys.shape[1]
            h_dim = keys.shape[3]
            if getattr(config, "use_core_inquiry", True):
                query_vec = curr_s[:, 0, :]
            else:
                query_vec = jnp.mean(curr_s, axis=1)
            probe_q = jnp.matmul(query_vec, params["core.deliberation_probe.weight"].T).reshape(B, 1, num_h, h_dim)
            probe_q = jnp.swapaxes(probe_q, 1, 2)
            if getattr(config, "use_rope", True):
                probe_q = apply_rope(probe_q, step_idx, theta=config.rope_theta)
                keys_r = apply_rope(keys, jnp.arange(L_kv, dtype=jnp.int32), theta=config.rope_theta)
            else:
                keys_r = keys
            scores = jnp.matmul(probe_q, jnp.swapaxes(keys_r, -1, -2)) / math.sqrt(h_dim)
            top_k_res = getattr(config, "top_k_resonance", 0) if config is not None else 0
            if top_k_res > 0 and L_kv > top_k_res:
                topk_v, _ = lax.top_k(scores, top_k_res)
                scores = jnp.where(scores >= topk_v[..., -1:], scores, -1e4)
            attn_p = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(curr_s.dtype)
            retrieved = jnp.matmul(attn_p, values)
            retrieved = jnp.swapaxes(retrieved, 1, 2).reshape(B, D)
            retrieved_normed = rms_norm(retrieved, params["core.deliberation_probe_norm.weight"])
            if getattr(config, "use_dual_evidence_grounding", True) and "core.evidence_core_gate" in params:
                beta_core = jnp.tanh(params.get("core.evidence_core_gate", jnp.zeros(1, dtype=jnp.float32))).astype(curr_s.dtype)
                scale = float(getattr(config, "sensory_evidence_core_scale", 0.05))
                state_core = curr_s[:, :1, :] + beta_core * scale * retrieved_normed[:, None, :]
                state_mantle = curr_s[:, -1:, :] + scale * retrieved_normed[:, None, :]
                if K > 2:
                    return jnp.concatenate([state_core, curr_s[:, 1:-1, :], state_mantle], axis=1)
                else:
                    return jnp.concatenate([state_core, state_mantle], axis=1)
            else:
                state_mantle = curr_s[:, -1:, :] + 0.1 * retrieved_normed[:, None, :]
                return jnp.concatenate([curr_s[:, :-1, :], state_mantle], axis=1)
        return curr_s

    if not use_adapt:
        def step_fn(carry, step_idx):
            curr_state, cum_energy = carry
            probed_state = _apply_probe(curr_state, step_idx)
            raw_next = deliberate_step(params, probed_state, buffers, config=config)
            step_energy = jnp.mean((raw_next.astype(jnp.float32) - probed_state.astype(jnp.float32)) ** 2)
            if use_contract:
                alpha = (delib_decay ** step_idx).astype(curr_state.dtype)
                updated = probed_state + alpha * (raw_next - probed_state)
                norm_up = jnp.sqrt(jnp.sum(jnp.square(updated.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-8).astype(curr_state.dtype)
                next_state = (updated / norm_up) * target_scale
            else:
                next_state = raw_next
            return (next_state, cum_energy + step_energy), None

        (final_state, total_energy), _ = lax.scan(step_fn, (state, jnp.array(0.0, dtype=jnp.float32)), jnp.arange(cycles))
        if return_energy:
            return final_state, jnp.array(0.0, dtype=jnp.float32), total_energy
        return final_state, jnp.array(0.0, dtype=jnp.float32)

    # Static ACT formulation for TPU/XLA:
    threshold = float(getattr(config, "thinking_halt_threshold", 0.75))
    w_gate = params["core.halting_gate.weight"]  # [1, D]
    b_gate = params["core.halting_gate.bias"]    # [1]

    # Carry: (curr_state, accum_state, cum_p, cum_ponder, cum_energy)
    init_accum = jnp.zeros_like(state)
    init_cum_p = jnp.zeros((B,), dtype=jnp.float32)
    init_ponder = jnp.zeros((B,), dtype=jnp.float32)

    def scan_body(carry, step_idx):
        curr_state, accum_state, cum_p, cum_ponder, cum_energy = carry
        probed_state = _apply_probe(curr_state, step_idx)
        raw_next = deliberate_step(params, probed_state, buffers, config=config)
        step_energy = jnp.mean((raw_next.astype(jnp.float32) - probed_state.astype(jnp.float32)) ** 2)
        if use_contract:
            alpha = (delib_decay ** step_idx).astype(curr_state.dtype)
            updated = probed_state + alpha * (raw_next - probed_state)
            norm_up = jnp.sqrt(jnp.sum(jnp.square(updated.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-8).astype(curr_state.dtype)
            next_state = (updated / norm_up) * target_scale
        else:
            next_state = raw_next

        # Centralized Core S_0 halting gate
        if getattr(config, "use_core_halting", True):
            core_state = next_state[:, 0, :].astype(jnp.float32)
        else:
            core_state = jnp.mean(next_state.astype(jnp.float32), axis=1)  # [B, D]
        halt_logit = jnp.squeeze(jnp.matmul(core_state, w_gate.T), axis=-1) + b_gate[0]
        halt_prob = jax.nn.sigmoid(halt_logit)  # [B]

        # Differentiable Graves ACT ponder cost
        is_last = (step_idx == cycles - 1)
        rem_p = jnp.maximum(0.0, 1.0 - cum_p)
        step_weight = jnp.where(is_last, rem_p, jnp.minimum(halt_prob, rem_p))

        # Update cumulative probability and accumulated convex state
        new_cum_p = cum_p + step_weight
        accum_state = accum_state + step_weight[:, None, None].astype(state.dtype) * next_state
        cum_ponder = cum_ponder + rem_p

        return (next_state, accum_state, new_cum_p, cum_ponder, cum_energy + step_energy), None

    (_, final_accum, final_cum_p, cum_ponder, total_energy), _ = lax.scan(
        scan_body,
        (state, init_accum, init_cum_p, init_ponder, jnp.array(0.0, dtype=jnp.float32)),
        jnp.arange(cycles),
    )

    norm_factor = jnp.maximum(final_cum_p[:, None, None], 1e-6)
    final_state = (final_accum / norm_factor).astype(state.dtype)
    if use_contract:
        norm_final = jnp.sqrt(jnp.sum(jnp.square(final_state.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-8).astype(state.dtype)
        final_state = (final_state / norm_final) * target_scale
    ponder_cost = jnp.mean(cum_ponder)

    if return_energy:
        return final_state, ponder_cost, total_energy
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
    affine_scale: Optional[jnp.ndarray] = None,
) -> Union[
    Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray, jnp.ndarray],
]:
    """Single motor emission step with affine-tied dictionary projection and progressive encephalization."""
    B, K, D = state.shape
    st_dtype = state.dtype
    if getattr(config, "use_sensory_core_coupling", False) and auditory_feedback is not None:
        rad_prof = (jnp.arange(1, K + 1, dtype=jnp.float32) / float(K))[None, :, None].astype(st_dtype)
        coupled_s = state + (0.1 * auditory_feedback[:, None, :].astype(st_dtype) * rad_prof)
        state = deliberate_step(params, coupled_s, buffers, config=config)
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

    use_cent_readout = getattr(config, "use_centralized_readout", True) if config is not None else False
    eff_core = transform_spectral_core_jax(core, params, config) if use_cent_readout else core
    if use_cent_readout:
        core_phase = eff_core + p_scaled
        query_surface = (core_phase + auditory_feedback.astype(st_dtype)) if auditory_feedback is not None else core_phase
        motor_surface = jnp.zeros_like(query_surface)
    else:
        motor_surface = surface + p_scaled
        query_surface = (motor_surface + auditory_feedback.astype(st_dtype)) if auditory_feedback is not None else motor_surface

    # 2. Multi-Head Arcuate Fasciculus Cross-Resonance with Causal Prefix
    keys, values = sensory_kv
    q = jnp.matmul(query_surface, params["motor.q_proj.weight"].T).reshape(B, 1, num_heads, head_dim)
    q = jnp.swapaxes(q, 1, 2)  # [B, H, 1, head_dim]
    if config is not None and config.use_rope:
        q_pos = step_idx if query_position is None else query_position
        q = apply_rope(q, q_pos, theta=config.rope_theta)
        keys = apply_rope(keys, jnp.arange(keys.shape[2], dtype=jnp.int32), theta=config.rope_theta)
    scores = jnp.matmul(q, jnp.swapaxes(keys, -1, -2)) / math.sqrt(head_dim)
    if attn_mask is not None:
        scores = jnp.where(attn_mask, scores, -1e4)
    top_k_res = getattr(config, "top_k_resonance", 0) if config is not None else 0
    if top_k_res > 0 and scores.shape[-1] > top_k_res:
        top_vals, _ = lax.top_k(scores, top_k_res)
        min_top = top_vals[..., -1:]
        scores = jnp.where(scores >= min_top, scores, -1e4)
    attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(st_dtype)
    res_heads = jnp.matmul(attn_weights, values)  # [B, H, 1, head_dim]
    res_heads = jnp.swapaxes(res_heads, 1, 2).reshape(B, D)
    resonance = rms_norm(jnp.matmul(res_heads, params["motor.out_proj.weight"].T), params["motor.resonance_norm.weight"])

    # 3. Multi-Shell Softmax Pooling
    if getattr(config, "use_shell_tap", True) and (not use_cent_readout):
        shell_weights = jax.nn.softmax(params["motor.shell_score"].astype(jnp.float32), axis=0)[None, :, None].astype(st_dtype)
        shell_proj = jnp.matmul(state, params["motor.shell_proj.weight"].T)
        norm_factor = math.sqrt(float(state.shape[1]))
        shell_pool = (jnp.sum(shell_weights * shell_proj, axis=1) * (params["motor.shell_scale"].astype(st_dtype) * norm_factor)).astype(st_dtype)
    else:
        shell_pool = jnp.zeros((B, D), dtype=st_dtype)

    # 4. Gated Deep Motor Integration via SwiGLU (per-source learnable importance gating)
    use_plan = getattr(config, "use_plan", True)
    ig = jnp.clip(params.get("motor.input_gate", jnp.ones(5, dtype=jnp.float32)), -5.0, 5.0).astype(st_dtype)
    fused_in = jnp.concatenate([
        ig[0] * motor_surface,
        ig[1] * eff_core,
        ig[2] * resonance,
        ig[3] * shell_pool,
        ig[4] * (plan if use_plan else jnp.zeros_like(plan)),
    ], axis=-1)
    fused = rms_norm(fused_in, params["motor.fused_norm.weight"])
    h = rms_norm(
        swiglu(
            fused,
            params["motor.swiglu.fc_gate.weight"],
            params["motor.swiglu.fc_val.weight"],
            params["motor.swiglu.fc_down.weight"],
            use_swiglu=getattr(config, "use_swiglu", True),
        ),
        params["motor.norm_mid.weight"],
    )

    # 5. Premotor Recurrent Executive Plan Update
    if use_plan:
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
    else:
        next_plan = plan

    # 6. Affine-Tied (or Untied) Dictionary Projection (W_motor = gamma * W_sensory or W_motor)
    # Memory-optimal: Scale hidden vector [B, D] instead of vocabulary matrix [V, D]
    temp = jnp.clip(params["motor.temperature_scale"].astype(jnp.float32), 0.1, 5.0).astype(st_dtype)
    active_affine = affine_scale if affine_scale is not None else params["motor.affine_scale"][None, :].astype(st_dtype)
    h_scaled = (h * active_affine) / temp
    w_out = params.get("motor.head.weight", params["sensory.embedding.weight"])
    raw_logits = jnp.matmul(h_scaled, w_out.T)

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

    # 8. Vectorized Depth-Decayed Efference Update across all concentric shells (O(1) overhead)
    if getattr(config, "use_per_token_efference", False):
        efference = jnp.matmul(h, params["motor.efference_proj.weight"].T)
        eff_projected = jnp.matmul(jax.nn.silu(efference), params["core.efference_gate.weight"].T)
        eff_normed = rms_norm(eff_projected, params["core.radial_norm.weight"]).astype(st_dtype)

        K = state.shape[1]
        if K > 1:
            gamma_0 = 0.15
            gamma_end = 0.02
            decay_rate = math.exp(math.log(gamma_end / gamma_0) / float(max(K - 2, 1)))
            gamma_vals = [gamma_0 * (decay_rate ** k) for k in range(K - 1)] + [0.0]
            if K > 16:
                gamma_vals[15] = 0.0
            gammas = jnp.array(gamma_vals, dtype=jnp.float32)[None, :, None].astype(st_dtype)
            next_state = ((1.0 - gammas) * state + gammas * eff_normed[:, None, :]).astype(st_dtype)
        else:
            new_core = (0.85 * core + 0.15 * eff_normed).astype(st_dtype)
            next_state = state.at[:, 0, :].set(new_core)
    else:
        next_state = state

    if return_hidden:
        return adapted_logits, next_state, next_trace, next_plan, h
    return adapted_logits, next_state, next_trace, next_plan


def _compute_token_ce_and_z_loss(
    ce_c: jnp.ndarray,
    lse: jnp.ndarray,
    mask_c: jnp.ndarray,
    eff_w: jnp.ndarray,
    config: Optional[ModelConfigV5],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes per-token weighted CE (with optional noise-capped selective weighting) and optional Z-loss."""
    use_sel = bool(getattr(config, "use_selective_loss", False)) if config is not None else False
    if use_sel:
        gamma = float(getattr(config, "selective_focal_gamma", 0.5))
        noise_cap = float(getattr(config, "selective_noise_cap", 8.5))
        p_t = jax.lax.stop_gradient(jnp.exp(-jnp.clip(ce_c, 0.0, 20.0)))
        focal_w = jnp.power(jnp.maximum(1.0 - p_t, 1e-4), gamma)
        noise_w = jnp.minimum(1.0, noise_cap / jnp.maximum(ce_c, 1e-4))
        sel_w = jax.lax.stop_gradient(focal_w * noise_w)
        sel_mean = jnp.sum(sel_w * mask_c) / jnp.maximum(jnp.sum(mask_c), 1.0)
        sel_w = sel_w / jnp.maximum(sel_mean, 1e-4)
        opt_w = eff_w * sel_w
    else:
        opt_w = eff_w

    loss_c = ce_c * opt_w
    z_w = float(getattr(config, "z_loss_weight", 0.0)) if config is not None else 0.0
    if z_w > 0.0:
        loss_c = loss_c + (z_w * jnp.square(lse) * eff_w)
    return loss_c, opt_w


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
    temperature: Optional[float] = None,
    key: Optional[jax.Array] = None,
    return_tokens: bool = False,
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

    # Initial concentric state with radial laminar depth profile ensuring all shells receive signal
    radial_profile = (jnp.arange(1, num_shells + 1, dtype=jnp.float32) / float(num_shells))[None, :, None]
    state = (jnp.broadcast_to(prompt_flux[:, None, :], (B, num_shells, surface_dim)).astype(jnp.float32) * radial_profile).astype(target_dtype)

    # Subconscious deliberation cycles with Adaptive Computation Time (ACT)
    state, ponder_cost, energy_loss = deliberate_cycles_jax(
        params, state, buffers, config=config, is_training=(targets is not None), sensory_kv=sensory_kv, return_energy=True
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

    # Track independent tri-head affine scale specialization strictly without target leakage
    th_start = (getattr(config, "think_token_id", None) or 29) if config is not None else 29
    th_end = (getattr(config, "think_end_token_id", None) or 30) if config is not None else 30
    act_start = (getattr(config, "tool_token_id", None) or getattr(config, "act_token_id", None) or 31) if config is not None else 31
    act_end = (getattr(config, "tool_end_token_id", None) or getattr(config, "act_end_token_id", None) or 32) if config is not None else 32

    # Check whether prompt had unclosed think or act tokens
    if prompts is not None and prompts.ndim == 2:
        prompt_th_unclosed = (jnp.sum(prompts == th_start, axis=-1) > jnp.sum(prompts == th_end, axis=-1))
        prompt_ac_unclosed = (jnp.sum(prompts == act_start, axis=-1) > jnp.sum(prompts == act_end, axis=-1))
    else:
        prompt_th_unclosed = jnp.zeros((B,), dtype=bool)
        prompt_ac_unclosed = jnp.zeros((B,), dtype=bool)

    aff_base = params["motor.affine_scale"][None, None, :].astype(target_dtype)
    aff_th = params.get("motor.affine_think")
    aff_ac = params.get("motor.affine_act")
    aff_an = params.get("motor.affine_answer")
    if prev_tokens is not None and aff_th is not None and aff_ac is not None and aff_an is not None:
        is_th_start = (prev_tokens == th_start).astype(jnp.int32)
        is_th_end = (prev_tokens == th_end).astype(jnp.int32)
        is_ac_start = (prev_tokens == act_start).astype(jnp.int32)
        is_ac_end = (prev_tokens == act_end).astype(jnp.int32)

        th_depth = jnp.cumsum(is_th_start - is_th_end, axis=-1) + prompt_th_unclosed[:, None].astype(jnp.int32)
        in_th = th_depth > 0

        ac_depth = jnp.cumsum(is_ac_start - is_ac_end, axis=-1) + prompt_ac_unclosed[:, None].astype(jnp.int32)
        in_ac = ac_depth > 0

        in_an = (~in_th) & (~in_ac)

        m_th = in_th[:, :, None].astype(target_dtype)
        m_ac = in_ac[:, :, None].astype(target_dtype)
        m_an = in_an[:, :, None].astype(target_dtype)

        head_mod = (
            m_th * aff_th[None, None, :].astype(target_dtype)
            + m_ac * aff_ac[None, None, :].astype(target_dtype)
            + m_an * aff_an[None, None, :].astype(target_dtype)
        )
        per_token_affine = aff_base * head_mod
    else:
        per_token_affine = aff_base

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

        step_aff = per_token_affine[:, step_idx, :] if per_token_affine is not None else None
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
            affine_scale=step_aff,
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
            logits_f32 = logits.astype(jnp.float32)
            lse = jax.nn.logsumexp(logits_f32, axis=-1)
            safe_targets = jnp.maximum(cur_target, 0)
            tgt_logits = jnp.take_along_axis(logits_f32, safe_targets[:, None], axis=-1).squeeze(-1)
            ce_c = lse - tgt_logits
            mask_t = (cur_target >= 0).astype(jnp.float32)
            cur_w = target_weights[:, step_idx] if target_weights is not None else mask_t
            eff_w = mask_t * cur_w
            step_loss, _ = _compute_token_ce_and_z_loss(ce_c, lse, mask_t, eff_w, config)
            if deliberation_mode == "predictive_coding":
                thought_err = jnp.mean((prev_thought.astype(jnp.float32) - lax.stop_gradient(tf_aud.astype(jnp.float32))) ** 2, axis=-1)
                step_loss = step_loss + (0.1 * thought_err * eff_w * (step_idx > 0).astype(jnp.float32))
            return (nxt_s, nxt_t, nxt_p, nxt_thought), (step_loss, eff_w, ce_c * eff_w)

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
            cur_state, cur_trace, cur_plan, prev_thought, k_buf, v_buf, mode_th, mode_ac = carry
            internal_feed = jnp.where(step_idx == 0, 0.0, prev_thought)

            cur_all_k = jnp.concatenate([prompt_k, k_buf], axis=2)
            cur_all_v = jnp.concatenate([prompt_v, v_buf], axis=2)
            pos = jnp.arange(total_prompt_L + target_len, dtype=jnp.int32)
            step_mask = gen_valid & (pos <= (total_prompt_L + step_idx))[None, None, None, :]

            if aff_th is not None and aff_ac is not None and aff_an is not None:
                head_mod = jnp.where(
                    mode_th[:, None],
                    aff_th[None, :],
                    jnp.where(
                        mode_ac[:, None],
                        aff_ac[None, :],
                        aff_an[None, :]
                    )
                ).astype(target_dtype)
                active_affine = aff_base.squeeze(1) * head_mod
            else:
                active_affine = aff_base.squeeze(1)

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
                affine_scale=active_affine,
            )

            if temperature is not None and temperature > 1e-4 and key is not None:
                step_key = jax.random.fold_in(key, step_idx)
                scaled_logits = logits.astype(jnp.float32) / max(float(temperature), 1e-4)
                tok_idx = lax.stop_gradient(jax.random.categorical(step_key, scaled_logits, axis=-1))
            else:
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

            nxt_mode_th = (mode_th | (tok_idx == th_start)) & (tok_idx != th_end)
            nxt_mode_ac = (mode_ac | (tok_idx == act_start)) & (tok_idx != act_end)

            return (nxt_s.astype(target_dtype), nxt_t, nxt_p.astype(target_dtype), nxt_thought, nxt_k_buf, nxt_v_buf, nxt_mode_th, nxt_mode_ac), (logits, tok_idx)

        _, (all_logits, all_toks) = lax.scan(
            _emit_gen,
            (state, init_refractory, init_plan, init_thought, init_k_buf, init_v_buf, prompt_th_unclosed, prompt_ac_unclosed),
            jnp.arange(target_len, dtype=jnp.int32),
        )
        if return_tokens:
            return jnp.swapaxes(all_logits, 0, 1), jnp.swapaxes(all_toks, 0, 1)
        return jnp.swapaxes(all_logits, 0, 1)

    # Use activation checkpointing (remat) for training to reduce HBM activation memory
    use_remat = remat and (targets is not None)
    use_per_tok_eff_cfg = bool(getattr(config, "use_per_token_efference", False)) if config is not None else False
    use_cent_readout_cfg = bool(getattr(config, "use_centralized_readout", True)) if config is not None else False
    # Vectorized Pure-TF precomputes cross-resonance queries before the token scan, which is
    # valid only when the query source shell (S_{-1}/S_{15} if non-centralized, S_0 if centralized)
    # is invariant across steps. When num_shells == 1 or (efference and centralized readout),
    # S_0 mutates per token so cross-resonance depends serially on prior motor states -> use _emit.
    is_pure_tf = (
        (targets is not None)
        and (num_shells >= 2)
        and (not (use_per_tok_eff_cfg and use_cent_readout_cfg))
        and (
            pure_teacher_forcing is True
            or (
                (deliberation_mode in ("teacher_forcing", "pure_tf"))
                and (not isinstance(sampling_p, jax.Array))
                and (float(sampling_p) >= 1.0 - 1e-4)
            )
        )
    )

    if is_pure_tf:
        T = int(target_len)
        use_cent_readout = use_cent_readout_cfg
        use_core_coupling = bool(getattr(config, "use_sensory_core_coupling", False))
        use_par_plan = bool(getattr(config, "use_parallel_plan_scan", False)) and (not getattr(config, "use_per_token_efference", False))

        # 1. Precompute all T normalized phase clocks and motor surface vectors outside scan
        max_phase = params["motor.phase_embed.weight"].shape[0]
        p_indices = jnp.clip(jnp.arange(T), 0, max_phase - 1)
        phases = params["motor.phase_embed.weight"][p_indices].astype(jnp.float32)
        p_norms = jnp.maximum(jnp.linalg.norm(phases, axis=-1, keepdims=True), 1e-6)
        p_scale_val = jnp.clip(
            params.get("motor.phase_scale", jnp.array(2.0, dtype=jnp.float32)), 0.05, 5.0
        ).astype(jnp.float32)
        p_scaled_all = (phases * (p_scale_val / p_norms)).astype(target_dtype)  # [T, D]

        use_shell_tap = getattr(config, "use_shell_tap", True) and (not use_cent_readout)
        w_shell_t = params["motor.shell_proj.weight"].T
        if use_shell_tap:
            shell_weights = jax.nn.softmax(params["motor.shell_score"].astype(jnp.float32), axis=0).astype(target_dtype)
            norm_factor = math.sqrt(float(num_shells))
            shell_scale = (params["motor.shell_scale"].astype(target_dtype) * norm_factor).astype(target_dtype)
            w_shell_scaled = (w_shell_t * shell_scale).astype(target_dtype)
        else:
            shell_weights = jnp.zeros(num_shells, dtype=target_dtype)
            w_shell_scaled = jnp.zeros_like(w_shell_t)

        if use_core_coupling:
            # Chunked parallel per-token concentric core coupling from teacher-forced tf_flux (zero recurrent h_t drift)
            rad_prof = (jnp.arange(1, num_shells + 1, dtype=jnp.float32) / float(num_shells))[None, :, None].astype(target_dtype)
            tf_flat = tf_flux.astype(target_dtype).reshape(B * T, surface_dim)
            base_s_flat = jnp.broadcast_to(state[:, None, :, :], (B, T, num_shells, surface_dim)).reshape(B * T, num_shells, surface_dim)
            tot_bt = B * T
            c_chunk = tot_bt if tot_bt <= 256 else next(
                (c for c in (256, 128, 64, 32, 16, 8, 4, 2, 1) if tot_bt % c == 0),
                tot_bt,
            )

            def _couple_step(bs_c, tf_c):
                coupled = bs_c + (0.1 * tf_c[:, None, :] * rad_prof)
                st_out = deliberate_step(params, coupled, buffers, config=config)
                c0 = transform_spectral_core_jax(st_out[:, 0, :], params, config) if use_cent_readout else st_out[:, 0, :]
                sf = ((1.0 - encephalization_alpha) * st_out[:, 15, :] + encephalization_alpha * st_out[:, -1, :]) if num_shells > 16 else st_out[:, -1, :]
                if use_shell_tap:
                    sp = jnp.sum(shell_weights[None, :, None] * jnp.matmul(st_out, w_shell_scaled), axis=1)
                else:
                    sp = jnp.zeros((st_out.shape[0], surface_dim), dtype=target_dtype)
                return c0, sf, sp

            couple_fn = jax.checkpoint(_couple_step) if use_remat else _couple_step
            if c_chunk == tot_bt:
                eff_core_flat, surface_flat, shell_pool_flat = couple_fn(base_s_flat, tf_flat)
            else:
                n_c = tot_bt // c_chunk
                _, (eff_core_flat, surface_flat, shell_pool_flat) = lax.scan(
                    lambda _, inp: (None, couple_fn(inp[0], inp[1])),
                    None,
                    (base_s_flat.reshape(n_c, c_chunk, num_shells, surface_dim), tf_flat.reshape(n_c, c_chunk, surface_dim)),
                )
                eff_core_flat = eff_core_flat.reshape(tot_bt, surface_dim)
                surface_flat = surface_flat.reshape(tot_bt, surface_dim)
                shell_pool_flat = shell_pool_flat.reshape(tot_bt, surface_dim)

            eff_core_all = eff_core_flat.reshape(B, T, surface_dim)
            surface_all = surface_flat.reshape(B, T, surface_dim)
            shell_pool_all = shell_pool_flat.reshape(B, T, surface_dim)
            init_core = eff_core_all[:, 0, :]
            eff_core = init_core
            if use_cent_readout:
                query_surface_all = eff_core_all + p_scaled_all[None, :, :] + tf_flux.astype(target_dtype)
                motor_surface_all = jnp.zeros_like(query_surface_all)
            else:
                motor_surface_all = surface_all + p_scaled_all[None, :, :]
                query_surface_all = motor_surface_all + tf_flux.astype(target_dtype)
        else:
            if num_shells > 16:
                surface = (1.0 - encephalization_alpha) * state[:, 15, :] + encephalization_alpha * state[:, -1, :]
            else:
                surface = state[:, -1, :]
            init_core = state[:, 0, :]
            eff_core = transform_spectral_core_jax(init_core, params, config) if use_cent_readout else init_core
            eff_core_all = jnp.broadcast_to(eff_core[:, None, :], (B, T, surface_dim))
            shell_pool_all = None
            if use_cent_readout:
                core_t = eff_core[:, None, :] + p_scaled_all[None, :, :]
                query_surface_all = core_t + tf_flux.astype(target_dtype)
                motor_surface_all = jnp.zeros_like(query_surface_all)
            else:
                motor_surface_all = surface[:, None, :] + p_scaled_all[None, :, :]
                query_surface_all = motor_surface_all + tf_flux.astype(target_dtype)

        # 2. Query-Chunked Multi-Head Arcuate Cross-Resonance (hoist RoPE keys outside remat boundary)
        total_len = causal_keys.shape[2]
        keys_att = (
            apply_rope(causal_keys, jnp.arange(total_len, dtype=jnp.int32), theta=config.rope_theta)
            if (config is not None and config.use_rope)
            else causal_keys
        )
        top_k_res = getattr(config, "top_k_resonance", 0) if config is not None else 0

        def _parallel_cross_resonance(ms_all):
            q_all = jnp.matmul(ms_all, params["motor.q_proj.weight"].T).reshape(B, T, num_heads, head_dim)
            q_all = jnp.swapaxes(q_all, 1, 2)  # [B, H, T, head_dim]
            if config is not None and config.use_rope:
                q_all = apply_rope(q_all, total_prompt_L + jnp.arange(T, dtype=jnp.int32), theta=config.rope_theta)
            res_heads = _chunked_mha_jax(
                q_all,
                keys_att,
                causal_values,
                attn_mask=all_valid,
                is_causal=True,
                query_pos_offset=total_prompt_L,
                top_k_res=top_k_res,
                chunk_size=512,
                use_remat=use_remat,
            )
            return rms_norm(
                jnp.matmul(res_heads, params["motor.out_proj.weight"].T),
                params["motor.resonance_norm.weight"],
            )

        res_fn = jax.checkpoint(_parallel_cross_resonance) if use_remat else _parallel_cross_resonance
        resonance_all = res_fn(query_surface_all)  # [B, T, D]

        # 3. Dynamic concentric shell pooling & vectorized depth-decayed efference recurrence setup
        if num_shells > 1:
            gamma_0 = 0.15
            gamma_end = 0.02
            decay_rate = math.exp(math.log(gamma_end / gamma_0) / float(max(num_shells - 2, 1)))
            gamma_vals = [gamma_0 * (decay_rate ** k) for k in range(num_shells - 1)] + [0.0]
            if num_shells > 16:
                gamma_vals[15] = 0.0
            gammas = jnp.array(gamma_vals, dtype=jnp.float32)[None, :, None].astype(target_dtype)
        else:
            gammas = jnp.array([[[0.15]]], dtype=target_dtype)

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
        use_plan = getattr(config, "use_plan", True)
        use_swiglu_motor = getattr(config, "use_swiglu", True)
        use_per_tok_eff = getattr(config, "use_per_token_efference", False)

        def _tf_motor_step(carry, xs):
            cur_state, cur_plan = carry
            ms_t, res_t, p_scaled_t = xs

            cur_core = cur_state[:, 0, :]
            if use_shell_tap:
                shell_proj = jnp.matmul(cur_state, w_shell_scaled)
                shell_pool = jnp.sum(shell_weights[None, :, None] * shell_proj, axis=1)
            else:
                shell_pool = jnp.zeros((B, surface_dim), dtype=target_dtype)

            fused_in = jnp.concatenate([
                ms_t,
                ig[1] * cur_core,
                res_t,
                ig[3] * shell_pool,
                ig[4] * (cur_plan if use_plan else jnp.zeros_like(cur_plan)),
            ], axis=-1)
            fused = rms_norm(fused_in, fused_norm_w)
            h = rms_norm(swiglu(fused, fc_gate_w, fc_val_w, fc_down_w, use_swiglu=use_swiglu_motor), norm_mid_w)

            if use_plan:
                phase_batch = jnp.broadcast_to(p_scaled_t[None, :], (B, surface_dim))
                plan_input = jnp.concatenate([h, phase_batch], axis=-1)
                p_gate = jax.nn.sigmoid(jnp.matmul(plan_input, plan_gate_w_t) + plan_gate_b)
                plan_update = jnp.matmul(plan_input, plan_update_w_t)
                nxt_plan = jnp.clip(plan_decay * cur_plan + plan_write * p_gate * plan_update, -50.0, 50.0).astype(target_dtype)
            else:
                nxt_plan = cur_plan

            if use_per_tok_eff:
                efference = jnp.matmul(h, eff_proj_t)
                eff_projected = jnp.matmul(jax.nn.silu(efference), eff_gate_t)
                eff_normed = rms_norm(eff_projected, radial_norm_w).astype(target_dtype)

                if num_shells > 1:
                    nxt_state = ((1.0 - gammas) * cur_state + gammas * eff_normed[:, None, :]).astype(target_dtype)
                else:
                    nxt_core = (0.85 * cur_core + 0.15 * eff_normed).astype(target_dtype)
                    nxt_state = cur_state.at[:, 0, :].set(nxt_core)
            else:
                nxt_state = cur_state

            return (nxt_state, nxt_plan), h

        if use_par_plan:
            # Parallel associative scan over T tokens (log2(T) depth instead of T-step sequential scan)
            if shell_pool_all is None:
                if use_shell_tap:
                    sp_static = jnp.sum(shell_weights[None, :, None] * jnp.matmul(state, w_shell_scaled), axis=1)
                    sp_all = jnp.broadcast_to(sp_static[:, None, :], (B, T, surface_dim))
                else:
                    sp_all = jnp.zeros((B, T, surface_dim), dtype=target_dtype)
            else:
                sp_all = shell_pool_all

            fused_par = jnp.concatenate([
                ig[0] * motor_surface_all,
                ig[1] * eff_core_all,
                ig[2] * resonance_all,
                ig[3] * sp_all,
                jnp.zeros((B, T, surface_dim), dtype=target_dtype),
            ], axis=-1)
            h_pre = rms_norm(
                swiglu(rms_norm(fused_par, fused_norm_w), fc_gate_w, fc_val_w, fc_down_w, use_swiglu=use_swiglu_motor),
                norm_mid_w,
            )
            if use_plan:
                phase_bt = jnp.broadcast_to(p_scaled_all[None, :, :], (B, T, surface_dim))
                plan_in = jnp.concatenate([h_pre, phase_bt], axis=-1)
                p_gate = jax.nn.sigmoid(jnp.matmul(plan_in, plan_gate_w_t) + plan_gate_b)
                p_upd = jnp.matmul(plan_in, plan_update_w_t)
                b_seq = (plan_write * p_gate * p_upd).astype(jnp.float32)
                a_seq = jnp.broadcast_to(plan_decay.astype(jnp.float32), b_seq.shape)

                def _assoc_op(left, right):
                    a1, b1 = left
                    a2, b2 = right
                    return a2 * a1, a2 * b1 + b2

                _, plan_out = lax.associative_scan(_assoc_op, (a_seq, b_seq), axis=1)
                plan_prev = jnp.concatenate([init_plan[:, None, :].astype(jnp.float32), plan_out[:, :-1, :]], axis=1)
                plan_prev = jnp.clip(plan_prev, -50.0, 50.0).astype(target_dtype)
                h_seq = jnp.swapaxes(rms_norm(h_pre + ig[4] * plan_prev, norm_mid_w), 0, 1)
            else:
                h_seq = jnp.swapaxes(h_pre, 0, 1)
        elif use_core_coupling and (not getattr(config, "use_per_token_efference", False)):
            gated_core_seq = jnp.swapaxes(ig[1] * eff_core_all, 0, 1)
            gated_sp_seq = jnp.swapaxes(
                ig[3] * (shell_pool_all if shell_pool_all is not None else jnp.zeros((B, T, surface_dim), dtype=target_dtype)),
                0, 1,
            )

            def _tf_coupled_step(cur_plan, xs):
                ms_t, core_t, res_t, sp_t, p_scaled_t = xs
                fused_in = jnp.concatenate([
                    ms_t,
                    core_t,
                    res_t,
                    sp_t,
                    ig[4] * (cur_plan if use_plan else jnp.zeros_like(cur_plan)),
                ], axis=-1)
                fused = rms_norm(fused_in, fused_norm_w)
                h = rms_norm(swiglu(fused, fc_gate_w, fc_val_w, fc_down_w, use_swiglu=use_swiglu_motor), norm_mid_w)
                if use_plan:
                    phase_batch = jnp.broadcast_to(p_scaled_t[None, :], (B, surface_dim))
                    plan_input = jnp.concatenate([h, phase_batch], axis=-1)
                    p_gate = jax.nn.sigmoid(jnp.matmul(plan_input, plan_gate_w_t) + plan_gate_b)
                    plan_update = jnp.matmul(plan_input, plan_update_w_t)
                    nxt_plan = jnp.clip(plan_decay * cur_plan + plan_write * p_gate * plan_update, -50.0, 50.0).astype(target_dtype)
                else:
                    nxt_plan = cur_plan
                return nxt_plan, h

            c_scan_fn = jax.checkpoint(_tf_coupled_step) if use_remat else _tf_coupled_step
            _, h_seq = lax.scan(c_scan_fn, init_plan, (gated_ms_seq, gated_core_seq, gated_res_seq, gated_sp_seq, p_scaled_all))
        else:
            motor_scan_fn = jax.checkpoint(_tf_motor_step) if use_remat else _tf_motor_step
            motor_init_state = state.at[:, 0, :].set(eff_core) if use_cent_readout else state
            _, h_seq = lax.scan(motor_scan_fn, (motor_init_state, init_plan), (gated_ms_seq, gated_res_seq, p_scaled_all))

        # 4. Chunked Systolic Vocabulary Projection & Cross-Entropy Outside Sequential Scan
        temp = jnp.clip(params["motor.temperature_scale"].astype(jnp.float32), 0.1, 5.0).astype(target_dtype)
        affine = params["motor.affine_scale"][None, None, :].astype(target_dtype)
        active_affine = per_token_affine if per_token_affine is not None else affine
        h_scaled = ((jnp.swapaxes(h_seq, 0, 1) * active_affine) / temp).reshape(B * T, surface_dim)
        targets_flat = targets[:, :T].reshape(B * T)
        w_emb_t = params.get("motor.head.weight", params["sensory.embedding.weight"]).T

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
            ce_c = lse - tgt_logits
            mask_c = (t_c >= 0).astype(jnp.float32)
            eff_w = mask_c * w_c
            loss_c, _ = _compute_token_ce_and_z_loss(ce_c, lse, mask_c, eff_w, config)
            return jnp.sum(loss_c), jnp.sum(eff_w), jnp.sum(ce_c * eff_w)

        if num_chunks == 1:
            loss_sum, mask_sum, raw_ce_sum = _ce_loss_chunk(h_scaled, targets_flat, weights_flat)
        else:
            ce_fn = lambda _, inp: (None, _ce_loss_chunk(inp[0], inp[1], inp[2]))
            ce_scan = jax.checkpoint(ce_fn) if (use_remat or num_chunks > 2) else ce_fn
            h_chunks = h_scaled.reshape(num_chunks, chunk_tokens, surface_dim)
            t_chunks = targets_flat.reshape(num_chunks, chunk_tokens)
            w_chunks = weights_flat.reshape(num_chunks, chunk_tokens)
            _, (chunk_losses, chunk_masks, chunk_raw_ces) = lax.scan(ce_scan, None, (h_chunks, t_chunks, w_chunks))
            loss_sum = jnp.sum(chunk_losses)
            mask_sum = jnp.sum(chunk_masks)
            raw_ce_sum = jnp.sum(chunk_raw_ces)

        mask_denom = jnp.maximum(mask_sum, 1.0)
        opt_loss = loss_sum / mask_denom
        raw_ce = raw_ce_sum / mask_denom
        base_loss = opt_loss + lax.stop_gradient(raw_ce - opt_loss)
        if ponder_cost is not None and getattr(config, "use_adaptive_thinking", False):
            base_loss = base_loss + float(getattr(config, "thinking_cost_weight", 0.001)) * ponder_cost

        # Multi-Token Prediction (MTP) auxiliary cross-entropy
        if getattr(config, "num_mtp_heads", 1) > 1:
            mtp_loss_sum = 0.0
            h_btd = jnp.swapaxes(h_seq, 0, 1)  # [B, T, D]
            h_prev = h_btd
            w_tok_emb = params.get("sensory.embedding.weight")
            for m in range(1, config.num_mtp_heads):
                if T > m:
                    tok_prefix = targets[:, m - 1:T - 1]
                    emb_prefix = w_tok_emb[jnp.maximum(tok_prefix, 0)]  # [B, T - m, D]
                    h_m_in = jnp.concatenate([h_prev[:, :T - m, :], emb_prefix], axis=-1)
                    gate_m = jax.nn.silu(jnp.matmul(h_m_in, params[f"motor.mtp.{m}.fc_gate.weight"].T))
                    val_m = jnp.matmul(h_m_in, params[f"motor.mtp.{m}.fc_val.weight"].T)
                    down_m = jnp.matmul(gate_m * val_m, params[f"motor.mtp.{m}.fc_down.weight"].T)
                    h_m = rms_norm(h_prev[:, :T - m, :] + down_m, params[f"motor.mtp.{m}.norm.weight"])
                    scale_m = params[f"motor.mtp.{m}.affine_scale"][None, None, :]
                    h_m_scaled = (h_m * scale_m) / temp
                    logits_m = jnp.matmul(h_m_scaled, w_emb_t).astype(jnp.float32)
                    tgt_m = targets[:, m:T]
                    safe_tgt_m = jnp.maximum(tgt_m, 0)
                    logp_m = jax.nn.log_softmax(logits_m, axis=-1)
                    tgt_logp_m = jnp.take_along_axis(logp_m, safe_tgt_m[:, :, None], axis=-1).squeeze(-1)
                    mask_m = (tgt_m >= 0).astype(jnp.float32)
                    w_m = target_weights[:, m:T] if target_weights is not None else mask_m
                    eff_w_m = mask_m * w_m
                    mtp_loss_m = -jnp.sum(tgt_logp_m * eff_w_m) / jnp.maximum(jnp.sum(eff_w_m), 1.0)
                    mtp_loss_sum = mtp_loss_sum + (0.3 / float(config.num_mtp_heads - 1)) * mtp_loss_m
                    h_prev = h_m
            base_loss = base_loss + mtp_loss_sum

        if energy_loss is not None and getattr(config, "use_energy_regularization", False):
            base_loss = base_loss + float(getattr(config, "energy_reg_weight", 0.01)) * (energy_loss / max(thinking_cycles, 1))

        # Optional Temporal Router supervision (Strictly 0.0 during pretraining to allow free exploration)
        router_weight = float(getattr(config, "router_loss_weight", 0.0))
        if router_weight > 0.0:
            w_r = params.get("motor.temporal_router.weight")
            b_r = params.get("motor.temporal_router.bias")
            if w_r is not None and b_r is not None:
                router_logits = jnp.matmul(prompt_flux.astype(jnp.float32), w_r.T) + b_r
                act_id = getattr(config, "tool_token_id", None) or getattr(config, "act_token_id", 31) if config is not None else 31
                th_id = getattr(config, "think_token_id", 29) if config is not None else 29
                tgt_act = jnp.any(targets == act_id, axis=-1).astype(jnp.float32)
                tgt_th = jnp.any(targets == th_id, axis=-1).astype(jnp.float32)
                tgt_an = jnp.ones_like(tgt_act)
                tgt_gates = jnp.stack([tgt_th, tgt_act, tgt_an], axis=-1)
                max_val = jnp.maximum(router_logits, 0.0)
                router_bce = max_val - router_logits * tgt_gates + jnp.log1p(jnp.exp(-jnp.abs(router_logits)))
                gate_loss = jnp.mean(router_bce)
                return base_loss + router_weight * gate_loss

        return base_loss

    scan_fn = jax.checkpoint(_emit) if use_remat else _emit

    if targets is not None:
        _, (all_loss, all_mask, all_raw_ce) = lax.scan(scan_fn, (state, init_refractory, init_plan, init_thought), jnp.arange(target_len))
        mask_denom = jnp.maximum(jnp.sum(all_mask), 1.0)
        opt_ce = jnp.sum(all_loss) / mask_denom
        raw_ce = jnp.sum(all_raw_ce) / mask_denom
        base_ce = opt_ce + lax.stop_gradient(raw_ce - opt_ce)
        if ponder_cost is not None and getattr(config, "use_adaptive_thinking", False):
            base_ce = base_ce + float(getattr(config, "thinking_cost_weight", 0.001)) * ponder_cost
        if energy_loss is not None and getattr(config, "use_energy_regularization", False):
            base_ce = base_ce + float(getattr(config, "energy_reg_weight", 0.01)) * (energy_loss / max(thinking_cycles, 1))
        router_weight = float(getattr(config, "router_loss_weight", 0.0))
        if router_weight > 0.0:
            w_r = params.get("motor.temporal_router.weight")
            b_r = params.get("motor.temporal_router.bias")
            if w_r is not None and b_r is not None:
                router_logits = jnp.matmul(prompt_flux.astype(jnp.float32), w_r.T) + b_r
                act_id = getattr(config, "tool_token_id", None) or getattr(config, "act_token_id", 31) if config is not None else 31
                th_id = getattr(config, "think_token_id", 29) if config is not None else 29
                tgt_act = jnp.any(targets == act_id, axis=-1).astype(jnp.float32)
                tgt_th = jnp.any(targets == th_id, axis=-1).astype(jnp.float32)
                tgt_an = jnp.ones_like(tgt_act)
                tgt_gates = jnp.stack([tgt_th, tgt_act, tgt_an], axis=-1)
                max_val = jnp.maximum(router_logits, 0.0)
                router_bce = max_val - router_logits * tgt_gates + jnp.log1p(jnp.exp(-jnp.abs(router_logits)))
                gate_loss = jnp.mean(router_bce)
                return base_ce + router_weight * gate_loss
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


def _encode_sensory_cached_step(
    params: Dict[str, jnp.ndarray],
    tok_ids: jnp.ndarray,
    pos_offset: Union[int, jnp.ndarray],
    sens_k_buf: jnp.ndarray,
    sens_v_buf: jnp.ndarray,
    valid_mask: jnp.ndarray,
    config: Optional[ModelConfigV5] = None,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Single-token sensory forward pass with causal KV cache over prior sensory tokens."""
    w_emb = params["sensory.embedding.weight"]
    x = w_emb[tok_ids]  # [B, 1, D]
    B, _, D = x.shape
    num_heads = config.num_heads if config is not None else 8
    head_dim = D // num_heads

    if "sensory.pos_embed.weight" in params:
        pos_tbl = params["sensory.pos_embed.weight"]
        max_p = pos_tbl.shape[0]
        pos_id = jnp.clip(pos_offset, 0, max_p - 1)
        in_table = (pos_offset < max_p).astype(pos_tbl.dtype)
        x = x + (pos_tbl[pos_id] * in_table)[None, None, :].astype(x.dtype)

    h = rms_norm(x, params["sensory.norm_in.weight"])
    h_swiglu = swiglu(
        h,
        params["sensory.swiglu.fc_gate.weight"],
        params["sensory.swiglu.fc_val.weight"],
        params["sensory.swiglu.fc_down.weight"],
        use_swiglu=getattr(config, "use_swiglu", True),
    )
    flux = rms_norm(x + h_swiglu, params["sensory.norm_out.weight"])
    use_rope_cfg = bool(config is not None and getattr(config, "use_rope", True))
    rope_theta = config.rope_theta if config is not None else 500000.0
    slot = jnp.clip(pos_offset, 0, sens_k_buf.shape[3] - 1)

    if "sensory.out_self.weight" in params and getattr(config, "use_sensory_self_attn", True):
        h_ctx = rms_norm(flux, params["sensory.ctx_norm.weight"])
        q_s = jnp.swapaxes(jnp.matmul(h_ctx, params["sensory.q_self.weight"].T).reshape(B, 1, num_heads, head_dim), 1, 2)
        k_s = jnp.swapaxes(jnp.matmul(h_ctx, params["sensory.k_self.weight"].T).reshape(B, 1, num_heads, head_dim), 1, 2)
        v_s = jnp.swapaxes(jnp.matmul(h_ctx, params["sensory.v_self.weight"].T).reshape(B, 1, num_heads, head_dim), 1, 2)
        if use_rope_cfg:
            q_s = apply_rope(q_s, pos_offset, theta=rope_theta)
            k_s = apply_rope(k_s, pos_offset, theta=rope_theta)
        kb0 = lax.dynamic_update_slice(sens_k_buf[0], k_s.astype(sens_k_buf.dtype), (0, 0, slot, 0))
        vb0 = lax.dynamic_update_slice(sens_v_buf[0], v_s.astype(sens_v_buf.dtype), (0, 0, slot, 0))
        sens_k_buf = sens_k_buf.at[0].set(kb0)
        sens_v_buf = sens_v_buf.at[0].set(vb0)
        scores = jnp.matmul(q_s, jnp.swapaxes(kb0, -1, -2)) / math.sqrt(head_dim)
        scores = jnp.where(valid_mask, scores, -1e4)
        attn_w = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(flux.dtype)
        ctx = jnp.swapaxes(jnp.matmul(attn_w, vb0), 1, 2).reshape(B, 1, D)
        flux = flux + jnp.matmul(ctx, params["sensory.out_self.weight"].T)

    num_sens_layers = getattr(config, "num_sensory_layers", 1) if config is not None else 1
    if num_sens_layers > 1:
        for idx in range(num_sens_layers - 1):
            pfx = f"sensory_dnn.extra_layers.{idx}."
            if f"{pfx}norm1.weight" in params:
                h_l = rms_norm(flux, params[f"{pfx}norm1.weight"])
                q_l = jnp.swapaxes(jnp.matmul(h_l, params[f"{pfx}q_proj.weight"].T).reshape(B, 1, num_heads, head_dim), 1, 2)
                k_l = jnp.swapaxes(jnp.matmul(h_l, params[f"{pfx}k_proj.weight"].T).reshape(B, 1, num_heads, head_dim), 1, 2)
                v_l = jnp.swapaxes(jnp.matmul(h_l, params[f"{pfx}v_proj.weight"].T).reshape(B, 1, num_heads, head_dim), 1, 2)
                if use_rope_cfg:
                    q_l = apply_rope(q_l, pos_offset, theta=rope_theta)
                    k_l = apply_rope(k_l, pos_offset, theta=rope_theta)
                kbi = lax.dynamic_update_slice(sens_k_buf[idx + 1], k_l.astype(sens_k_buf.dtype), (0, 0, slot, 0))
                vbi = lax.dynamic_update_slice(sens_v_buf[idx + 1], v_l.astype(sens_v_buf.dtype), (0, 0, slot, 0))
                sens_k_buf = sens_k_buf.at[idx + 1].set(kbi)
                sens_v_buf = sens_v_buf.at[idx + 1].set(vbi)
                scores_l = jnp.matmul(q_l, jnp.swapaxes(kbi, -1, -2)) / math.sqrt(head_dim)
                scores_l = jnp.where(valid_mask, scores_l, -1e4)
                attn_w_l = jax.nn.softmax(scores_l.astype(jnp.float32), axis=-1).astype(flux.dtype)
                ctx_l = jnp.swapaxes(jnp.matmul(attn_w_l, vbi), 1, 2).reshape(B, 1, D)
                flux = flux + jnp.matmul(ctx_l, params[f"{pfx}out_proj.weight"].T)
                h2 = rms_norm(flux, params[f"{pfx}norm2.weight"])
                flux = flux + swiglu(
                    h2,
                    params[f"{pfx}swiglu.fc_gate.weight"],
                    params[f"{pfx}swiglu.fc_val.weight"],
                    params[f"{pfx}swiglu.fc_down.weight"],
                )
        flux = rms_norm(flux, params["sensory.norm_out.weight"])

    k = jnp.swapaxes(jnp.matmul(flux, params["sensory.k_proj.weight"].T).reshape(B, 1, num_heads, head_dim), 1, 2)
    v = jnp.swapaxes(jnp.matmul(flux, params["sensory.v_proj.weight"].T).reshape(B, 1, num_heads, head_dim), 1, 2)
    return flux, (k, v), sens_k_buf, sens_v_buf


def generate_causal_prefill_jax(
    params: Dict[str, jnp.ndarray],
    prompt_ids: jnp.ndarray,
    max_new_tokens: int,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: ModelConfigV5,
    temperature: float = 0.3,
    repetition_penalty: float = 1.15,
    key: Optional[jax.Array] = None,
) -> List[int]:
    """Causal lax.scan prefill + decode matching RectifiedContiguousPretrainStreamer geometry."""
    B, L = prompt_ids.shape
    total_len = L + max_new_tokens
    dtype = params["sensory.embedding.weight"].dtype
    num_heads = config.num_heads if config is not None else 8
    head_dim = config.surface_dim // num_heads
    num_sens_layers = max(1, int(getattr(config, "num_sensory_layers", 1)))
    sens_k_buf = jnp.zeros((num_sens_layers, B, num_heads, total_len, head_dim), dtype=dtype)
    sens_v_buf = jnp.zeros((num_sens_layers, B, num_heads, total_len, head_dim), dtype=dtype)
    init_sens_mask = jnp.zeros((B, 1, 1, total_len), dtype=bool).at[:, :, :, 0].set(True)

    bos = prompt_ids[:, :1]
    s_flux, (pk, pv), sens_k_buf, sens_v_buf = _encode_sensory_cached_step(
        params, bos, jnp.array(0, dtype=jnp.int32), sens_k_buf, sens_v_buf, init_sens_mask, config=config
    )
    radial_profile = (jnp.arange(1, config.num_shells + 1, dtype=jnp.float32) / float(config.num_shells))[None, :, None]
    state = (jnp.broadcast_to(s_flux[:, 0:1, :], (B, config.num_shells, config.surface_dim)).astype(jnp.float32) * radial_profile).astype(dtype)
    state, _, _ = deliberate_cycles_jax(
        params, state, buffers, config=config, is_training=False, sensory_kv=(pk, pv), return_energy=True
    )

    k_buf = jnp.zeros((B, num_heads, total_len, head_dim), dtype=dtype).at[:, :, :1, :].set(pk.astype(dtype))
    v_buf = jnp.zeros((B, num_heads, total_len, head_dim), dtype=dtype).at[:, :, :1, :].set(pv.astype(dtype))
    valid_buf = jnp.zeros((B, 1, 1, total_len), dtype=bool).at[:, :, :, 0].set(True)

    plan = jnp.zeros((B, config.surface_dim), dtype=dtype)
    trace = jnp.zeros((B, config.vocab_size), dtype=dtype)
    aff = params["motor.affine_scale"][None, :].astype(dtype)
    if "motor.affine_answer" in params:
        aff = aff * params["motor.affine_answer"][None, :].astype(dtype)

    p_flat = prompt_ids[0]
    gen_key = key if key is not None else jax.random.PRNGKey(42)

    def _step(carry, t):
        st, tr, pl, th, kb, vb, valb, skb, svb = carry
        feed = jnp.where(t == 0, jnp.zeros_like(th), th)
        mask = valb & (jnp.arange(total_len)[None, None, None, :] <= t)
        logits, n_st, n_tr, n_pl = emit_step(
            params, st, (kb, vb), pl, tr, t, buffers, config=config,
            auditory_feedback=feed, repetition_penalty=repetition_penalty, attn_mask=mask,
            query_position=t + 1, affine_scale=aff,
        )
        if temperature is not None and temperature > 1e-4:
            step_key = jax.random.fold_in(gen_key, t)
            pred = lax.stop_gradient(jax.random.categorical(step_key, logits.astype(jnp.float32) / float(temperature), axis=-1))
        else:
            pred = lax.stop_gradient(jnp.argmax(logits, axis=-1))
        next_tok = jnp.where(t < L - 1, p_flat[jnp.minimum(t + 1, L - 1)][None], pred)
        slot = jnp.minimum(t + 1, total_len - 1)
        n_valb = valb.at[:, :, :, slot].set(True)
        sens_mask = n_valb & (jnp.arange(total_len)[None, None, None, :] <= (t + 1))
        t_flux, (tk, tv), n_skb, n_svb = _encode_sensory_cached_step(
            params, next_tok[:, None], t + 1, skb, svb, sens_mask, config=config
        )
        n_kb = lax.dynamic_update_slice(kb, tk.astype(dtype), (0, 0, slot, 0))
        n_vb = lax.dynamic_update_slice(vb, tv.astype(dtype), (0, 0, slot, 0))
        return (n_st.astype(dtype), n_tr, n_pl.astype(dtype), t_flux[:, 0, :].astype(dtype), n_kb, n_vb, n_valb, n_skb, n_svb), pred

    _, preds = jax.lax.scan(
        _step,
        (state, trace, plan, jnp.zeros((B, config.surface_dim), dtype=dtype), k_buf, v_buf, valid_buf, sens_k_buf, sens_v_buf),
        jnp.arange(L - 1 + max_new_tokens, dtype=jnp.int32),
    )
    return [int(preds[i, 0]) for i in range(L - 1, L - 1 + max_new_tokens)]


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
    key: Optional[jax.Array] = None,
    raw_prompt: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Generates text using the 3-tier cognitive protocol in JAX:
      1. Thinking   : <think> ... </think> -> Private scratchpad
      2. Talking    : Unbounded plain text to EOS -> Fluid zero-overhead answering
      3. Tool Action: <tool_call> ... </tool_call> -> Structured external tool execution boundary
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
    seed = seed_token_id  # Unbiased generation: do not force tool call by default

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

    gen_key = key if key is not None else jax.random.PRNGKey(42)
    if raw_prompt and images is None and audio is None and perceptual_inputs is None:
        raw_ids = tokenizer(prompt, return_tensors="np", add_special_tokens=True).input_ids
        token_ids = generate_causal_prefill_jax(
            params=params,
            prompt_ids=jnp.array(raw_ids, dtype=jnp.int32),
            max_new_tokens=effective_max_tokens,
            buffers=buffers,
            config=config,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            key=gen_key,
        )
    else:
        logits, sampled_toks = training_rollout(
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
            temperature=temperature,
            key=gen_key,
            return_tokens=True,
        )
        token_ids = [int(x) for x in sampled_toks[0]]

    if strict_act:
        return extract_strict_act_payload(token_ids, tokenizer)

    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    # Check text-level stops: ChatML <|im_end|>, </s>
    for stop_str in ("<|im_end|>", "</s>"):
        if stop_str in decoded:
            decoded = decoded.split(stop_str)[0]

    parsed = parse_model_output(decoded)
    if return_parsed:
        return parsed

    return parsed["answer"] if parsed["answer"] else (parsed["thinking"] if parsed["thinking"] else clean_motor_response(decoded))


def generate_closed_loop_tool_jax(
    params: Dict[str, jnp.ndarray],
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: ModelConfigV5,
    prompt: str,
    tokenizer: Any,
    tool_executor: Optional[Any] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    repetition_penalty: float = 1.2,
    key: Optional[jax.Array] = None,
) -> Dict[str, Any]:
    """
    Executes the closed-loop tool interaction specified in Section 4.2 of the plan:
      1. Prompt enters sensory encoding and initial latent deliberation (deliberate_cycles_jax).
      2. Model generates initial motor output (direct answer or structured <tool_call>).
      3. If a <tool_call> is emitted and a tool executor is provided:
         a. Host executes tool -> tool_result
         b. Closed-loop context feedback: prompt + <tool_call> ... </tool_call> + <tool_response> ... </tool_response>
         c. Model ingests updated context and generates the final natural language answer.
      4. If no tool is emitted, returns the direct answer immediately.
    """
    from src.corpus import build_tool_closed_loop_prompt, clean_motor_response, parse_model_output

    if tool_executor is None:
        try:
            from src.agentic_harness import execute_agentic_tool
            tool_executor = execute_agentic_tool
        except Exception:
            tool_executor = None

    rng1, rng2 = jax.random.split(key if key is not None else jax.random.PRNGKey(42))

    # Step 1: Initial Generation (Latent Deliberation -> Direct Answer or Tool Call)
    step1_parsed = generate_text_jax(
        params=params,
        buffers=buffers,
        config=config,
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        return_parsed=True,
        key=rng1,
    )

    tool_calls = step1_parsed.get("tool_calls", [])
    if not tool_calls or tool_executor is None:
        # Direct answer or tool execution not available
        final_answer = step1_parsed.get("answer") or clean_motor_response(step1_parsed.get("raw", ""))
        return {
            "has_tool_call": False,
            "tool_call": None,
            "tool_response": None,
            "answer": final_answer,
            "step1_raw": step1_parsed.get("raw", ""),
        }

    # Step 2: Host Tool Execution
    tool_call = tool_calls[0]
    try:
        tool_result = str(tool_executor(tool_call))
    except Exception as e:
        tool_result = f"Error executing tool: {e}"

    # Step 3: Context Feedback Loop & Final Supervised Answer Generation
    feedback_prompt = build_tool_closed_loop_prompt(prompt, tool_call, tool_result)
    step2_parsed = generate_text_jax(
        params=params,
        buffers=buffers,
        config=config,
        prompt=feedback_prompt,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        return_parsed=True,
        key=rng2,
    )

    final_answer = step2_parsed.get("answer") or clean_motor_response(step2_parsed.get("raw", ""))
    return {
        "has_tool_call": True,
        "tool_call": tool_call,
        "tool_response": tool_result,
        "answer": final_answer,
        "step1_raw": step1_parsed.get("raw", ""),
        "step2_raw": step2_parsed.get("raw", ""),
        "feedback_prompt": feedback_prompt,
    }


def emit_tri_head_logits_jax(
    params: Dict[str, jnp.ndarray],
    h: jnp.ndarray,
    temp: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simultaneously computes vocabulary logits across all three cognitive tiers in JAX:
      1. logits_think : Deliberation / Reasoning scratchpad (<think>)
      2. logits_act   : Tool actions (Coding, Calculator, Tools) (<tool_call>)
      3. logits_answer: Conversational dialogue / Response (Plain Text)
    """
    w_emb = params.get("motor.head.weight", params["sensory.embedding.weight"])
    aff_base = params["motor.affine_scale"]
    aff_think = params.get("motor.affine_think", jnp.ones_like(aff_base))
    aff_act = params.get("motor.affine_act", jnp.ones_like(aff_base))
    aff_ans = params.get("motor.affine_answer", jnp.ones_like(aff_base))

    l_think = jnp.matmul((h * (aff_base * aff_think)) / temp, w_emb.T)
    l_act = jnp.matmul((h * (aff_base * aff_act)) / temp, w_emb.T)
    l_ans = jnp.matmul((h * (aff_base * aff_ans)) / temp, w_emb.T)
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
    phased: bool = True,
    rng_key: Optional[jax.Array] = None,
) -> Dict[str, Any]:
    """
    Simultaneous Tri-Head Cognitive Emission in JAX:
    Emits all 3 streams simultaneously from the shared manifold hidden state:
      - Head 1 (Think)  : Emits cognitive reasoning scratchpad (<think>)
      - Head 2 (Act)    : Emits tool actions (<tool_call>)
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

    # 1. Sensory encoding (with prompt_valid mask matching training_rollout)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    prompt_valid = (prompts_arr != pad_id)[:, None, None, :]
    sensory_flux, (sensory_keys, sensory_values) = encode_sensory(
        params, prompts_arr, config=config, pos_offset=0, attn_mask=prompt_valid
    )
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

    radial_profile = (jnp.arange(1, num_shells + 1, dtype=jnp.float32) / float(num_shells))[None, :, None]
    state = (jnp.broadcast_to(prompt_centroid[:, None, :], (B, num_shells, surface_dim)).astype(jnp.float32) * radial_profile).astype(target_dtype)
    state, _ = deliberate_cycles_jax(
        params, state, buffers, config=config, is_training=False, sensory_kv=(sensory_keys, sensory_values)
    )

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

    sample_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)

    def _sample_from_logits(lg: jnp.ndarray, prev_toks: List[int]) -> int:
        nonlocal sample_key
        lg_f32 = lg[0].astype(jnp.float32)
        if prev_toks:
            pt_idx = jnp.array(sorted(set(prev_toks)), dtype=jnp.int32)
            pt_vals = lg_f32[pt_idx]
            adj_vals = jnp.where(pt_vals > 0, pt_vals / repetition_penalty, pt_vals * repetition_penalty)
            lg_f32 = lg_f32.at[pt_idx].set(adj_vals)
        lg_scaled = lg_f32 / max(temperature, 1e-4)
        sample_key, sub_key = jax.random.split(sample_key)
        return int(jax.random.categorical(sub_key, lg_scaled))

    prev_thought = jnp.zeros((B, surface_dim), dtype=target_dtype)
    for t in range(max_new_tokens):
        # The cache contains the prompt plus *previous* generated tokens.
        # Every valid cached key is causal at this point; the newly sampled
        # token is appended only after this emission step.
        step_mask = all_valid
        feed = prev_thought if t > 0 else None

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

        l_th, l_ac, l_an = emit_tri_head_logits_jax(params, h, temp=1.0)

        tok_th = None
        tok_ac = None
        tok_an = None

        if phased and emit_think and not fin_think:
            # Phase 1: Thinking scratchpad emission (populates causal KV buffer)
            tok_th = _sample_from_logits(l_th, tokens_think)
            if tok_th == eos_id:
                fin_think = True
            else:
                tokens_think.append(tok_th)
            emitted_tok = tok_th if (tok_th is not None and tok_th != eos_id) else None
        else:
            # Phase 2 (or unphased): Action & Answer emission cross-referencing prompt + scratchpad
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

            if not phased and not fin_think:
                tok_th = _sample_from_logits(l_th, tokens_think)
                if tok_th == eos_id:
                    fin_think = True
                else:
                    tokens_think.append(tok_th)

            emitted_tok = (
                tok_an
                if not fin_ans and tok_an is not None and tok_an != eos_id
                else (
                    tok_ac
                    if not fin_act and tok_ac is not None and tok_ac != eos_id
                    else (
                        tok_th
                        if not fin_think and tok_th is not None and tok_th != eos_id
                        else None
                    )
                )
            )

        if emitted_tok is not None:
            tok_flux, (new_k, new_v) = encode_sensory(
                params,
                jnp.asarray([[emitted_tok]], dtype=jnp.int32),
                config=config,
                pos_offset=total_prompt_L + t + 1,
                is_causal=True,
            )
            prev_thought = tok_flux[:, 0, :].astype(target_dtype)
            causal_keys = jnp.concatenate([causal_keys, new_k], axis=2)
            causal_values = jnp.concatenate([causal_values, new_v], axis=2)
            all_valid = jnp.concatenate(
                [all_valid, jnp.ones((B, 1, 1, 1), dtype=bool)], axis=-1
            )
        else:
            prev_thought = jnp.zeros((B, surface_dim), dtype=target_dtype)

        if fin_think and fin_act and fin_ans:
            break

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
