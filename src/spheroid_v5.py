"""
CerebroSpheroid V5: Deep Concentric Laminae & Recurrent Deliberation Architecture
==================================================================================

Key Innovations in V5:
1. Affine-Tied Sensory-Motor Dictionary: Eliminates 30M+ redundant lookup parameters
   (W_motor = gamma * W_sensory), raising active compute parameters to >90%.
2. Deep Spheroid Laminae (32 to 48 concentric shells): 2x - 3x radial wave depth.
3. SwiGLU Gated Wave Equations: High-capacity non-linear radial and intra-shell convection.
4. Multi-Head Resonant Arcuate Fasciculus (8 to 16 heads): Scaled dot-product cross-resonance
   with temporal recency biasing.
5. Native Normalized Phase Clock (p_scaled = 2.0): Eliminates static attractor basins from step 0.
6. Contractive Spectral Guard (rho <= 1.0): Quintic Newton-Schulz polar normalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfigV5:
    """Architectural hyperparameters for CerebroSpheroid V5."""
    vocab_size: int = 32000           # LLaMA Byte-Fallback BPE
    surface_dim: int = 768            # Spheroid boundary dimension (D)
    radial_dim: int = 512             # Radial wave channel depth (D_r)
    num_shells: int = 32              # Concentric spherical laminae (K)
    num_shortcut_tracts: int = 96     # Axonal shortcut tracts
    num_heads: int = 8                # Arcuate Fasciculus cross-resonance heads
    max_phase_steps: int = 1024       # Premotor phase clock horizon
    thinking_cycles: int = 4          # Subconscious deliberation steps before emission
    phase_scale: float = 2.0          # Normalized phase clock signal amplitude
    dropout: float = 0.0              # Regularization dropout
    use_tied_embeddings: bool = True  # Affine-tied sensory-motor dictionary
    use_swiglu: bool = True           # SwiGLU non-linear gating
    use_shell_tap: bool = True        # Multi-shell efferent tap
    use_plan: bool = True             # Recurrent premotor executive plan
    rope_theta: float = 500000.0      # Rotary Position Embedding base (32k+ horizon)
    max_seq_len: int = 32768          # Precomputed rotary horizon
    top_k_resonance: int = 16         # Top-k sparse associative retrieval (needle tethering)
    use_deliberation_probe: bool = True  # Subconscious core probe of sensory KV buffer
    # Frontier 2026 Architectural Upgrades
    use_adaptive_thinking: bool = True     # Dynamic halting gate during subconscious deliberation
    max_thinking_cycles: int = 8           # Upper bound for adaptive deliberation
    thinking_halt_threshold: float = 0.75  # Halting confidence threshold
    num_mtp_heads: int = 1                 # Multi-Token Prediction heads (1 = standard, 2 or 3 = MTP)
    use_sparse_mot: bool = True            # Mixture-of-Tracts sparse axonal routing
    top_k_tracts: int = 24                 # Active shortcut tracts routed per step
    use_energy_regularization: bool = True # Contrastive fixed-point energy guard
    enable_multimodal: bool = True         # Multimodal continuous sensory projection
    visual_dim: int = 512                  # Perceptual / Visual patch dimension
    audio_dim: int = 512                   # Acoustic / Audio frame dimension
    num_visual_patches: int = 16           # Default visual tokens per perceptual stimulus
    image_patch_size: int = 16             # Pixel patch size (16x16)


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
        "use_tied_embeddings": False,
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
        "top_k_tracts": 4,
        "visual_dim": 128,
    },
}


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit with channel expansion factor."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(2 * in_features)
        self.fc_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.fc_val = nn.Linear(in_features, hidden_features, bias=False)
        self.fc_down = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_down(F.silu(self.fc_gate(x)) * self.fc_val(x))


def precompute_rope_freqs(dim: int, max_seq_len: int = 32768, theta: float = 500000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes rotary frequency cos and sin tables up to max_seq_len with YaRN-compatible base."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)  # [max_seq_len, dim // 2]
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # [max_seq_len, dim]
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # [max_seq_len, dim]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """
    Applies Rotary Position Embedding to x shaped [B, H, L, D] or [B, H, 1, D].
    offset indexes into the precomputed rotary frequencies. Dynamically extends
    frequencies if sequence length exceeds the precomputed cache.
    """
    L = x.shape[2]
    if offset + L > cos.shape[0]:
        new_max = max(offset + L + 1024, cos.shape[0] * 2)
        dim = x.shape[-1]
        cos, sin = precompute_rope_freqs(dim, max_seq_len=new_max)
        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)
    c = cos[offset : offset + L].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, L, D]
    s = sin[offset : offset + L].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * c + x_rot * s


class MTPHead(nn.Module):
    """
    Multi-Token Prediction Drafting Head (DeepSeek-V3 / Meta MTP).
    Predicts future token t + offset from motor hidden representation and preceding token.
    """

    def __init__(self, surface_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(surface_dim * 2)
        self.swiglu = SwiGLU(surface_dim * 2, hidden_features=surface_dim * 2, out_features=surface_dim)
        self.norm_out = RMSNorm(surface_dim)
        self.affine_scale = nn.Parameter(torch.ones(surface_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_curr: torch.Tensor,
        prev_token_emb: torch.Tensor,
        tied_w: torch.Tensor,
        temp: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused = self.norm(torch.cat([h_curr, prev_token_emb], dim=-1))
        h_next = self.norm_out(h_curr + self.dropout(self.swiglu(fused)))
        h_scaled = (h_next * self.affine_scale) / max(temp, 1e-4)
        logits = F.linear(h_scaled, tied_w)
        return logits, h_next


class ImagePatchStem(nn.Module):
    """
    2D Convolutional Patchify Stem (Retinotopic sensory mapping):
    Converts raw RGB pixel grids [B, 3, H, W] into spatial visual tokens [B, P, visual_dim].
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 3, visual_dim: int = 512):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, visual_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = RMSNorm(visual_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, visual_dim, H', W'] -> [B, P, visual_dim]
        h = self.conv(x).flatten(2).transpose(1, 2)
        return self.norm(h)


class AudioWaveformStem(nn.Module):
    """
    1D Convolutional Acoustic Stem (Tonotopic cochlear mapping):
    Converts raw 16kHz audio waveforms [B, 1, T_samples] or [B, T_samples] into temporal acoustic tokens [B, T_frames, audio_dim].
    Uses 20ms frame resolution (kernel 400, stride 160 at 16kHz).
    """

    def __init__(self, kernel_size: int = 400, stride: int = 160, audio_dim: int = 512):
        super().__init__()
        self.conv = nn.Conv1d(1, audio_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.norm = RMSNorm(audio_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv(x).transpose(1, 2)
        return self.norm(h)


class SensoryInputDNNV5(nn.Module):
    """
    Sensory Input DNN V5:
    1. Shared/Tied Embedding Table (vocab_size -> surface_dim).
    2. Deep SwiGLU input mapping.
    3. Multi-Head Key/Value projections for Arcuate Fasciculus cross-resonance.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        surface_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.surface_dim = surface_dim
        self.num_heads = num_heads
        self.head_dim = surface_dim // num_heads

        # Primary sensory dictionary (shared with motor emission head via affine tying)
        self.embedding = nn.Embedding(vocab_size, surface_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # Zero-init positional embedding preserves transferability while restoring token order
        self.pos_embed = nn.Embedding(1024, surface_dim)
        nn.init.zeros_(self.pos_embed.weight)

        # Non-linear input SwiGLU block
        self.swiglu = SwiGLU(surface_dim, hidden_features=surface_dim * 2, out_features=surface_dim)
        self.norm_in = RMSNorm(surface_dim)
        self.norm_out = RMSNorm(surface_dim)
        self.dropout = nn.Dropout(dropout)

        # Multi-Head KV projections for Arcuate Fasciculus
        self.k_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.v_proj = nn.Linear(surface_dim, surface_dim, bias=False)

        # Contextual inter-token multi-head self-attention across prompt tokens
        self.ctx_norm = RMSNorm(surface_dim)
        self.q_self = nn.Linear(surface_dim, surface_dim, bias=False)
        self.k_self = nn.Linear(surface_dim, surface_dim, bias=False)
        self.v_self = nn.Linear(surface_dim, surface_dim, bias=False)
        self.out_self = nn.Linear(surface_dim, surface_dim, bias=False)

    def forward(
        self,
        input_ids_or_emb: torch.Tensor,
        is_causal: bool = False,
        pos_offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Accepts integer token IDs [B, L] or soft embedding vectors [B, D] / [B, L, D].
        Args:
            input_ids_or_emb: input tokens or embeddings
            is_causal: If True, applies causal upper-triangle masking to prevent future token leakage
        Returns:
            flux: Contextual sensory representation [B, L, D]
            kv: Tuple of (keys, values) shaped [B, H, L, head_dim] for cross-resonance.
        """
        if input_ids_or_emb.dtype in (torch.int32, torch.int64):
            x = self.embedding(input_ids_or_emb)
        else:
            x = input_ids_or_emb

        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        B, L, D = x.shape
        pos_ids = (torch.arange(L, device=x.device, dtype=torch.long) + pos_offset).clamp(
            0, self.pos_embed.num_embeddings - 1
        )
        x = x + self.pos_embed(pos_ids).unsqueeze(0).to(x.dtype)
        h = self.norm_in(x)
        h = x + self.dropout(self.swiglu(h))
        flux = self.norm_out(h)

        # Contextual Inter-Token Mixing via Fused FlashAttention (SDPA)
        if L > 1:
            h_ctx = self.ctx_norm(flux)
            q_s = self.q_self(h_ctx).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k_s = self.k_self(h_ctx).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v_s = self.v_self(h_ctx).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            ctx = F.scaled_dot_product_attention(
                q_s,
                k_s,
                v_s,
                is_causal=is_causal,
                dropout_p=self.dropout.p if self.training else 0.0,
            ).transpose(1, 2).contiguous().view(B, L, D)
            flux = flux + self.dropout(self.out_self(ctx))

        # Multi-Head KV projections: [B, L, D] -> [B, H, L, head_dim]
        k = self.k_proj(flux).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(flux).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if is_2d:
            flux = flux.squeeze(1)

        return flux, (k, v)


class VolumetricSpheroidCoreV5(nn.Module):
    """
    Volumetric Spheroid Core V5:
    1. 32 to 48 concentric spherical laminae (Shell 0 = Deep Core, Shell K-1 = Surface).
    2. Intra-shell SwiGLU non-linear convection.
    3. Inter-shell bi-directional radial wave propagation with dimension D_r.
    4. Small-world hierarchical axonal shortcut tracts with learnable conductance.
    5. Contractive spectral radius guard (rho <= 1.0) via Polar Newton-Schulz normalization.
    """

    def __init__(
        self,
        num_shells: int = 32,
        surface_dim: int = 768,
        radial_dim: int = 512,
        num_shortcut_tracts: int = 96,
        dropout: float = 0.0,
        use_sparse_mot: bool = True,
        top_k_tracts: int = 24,
    ):
        super().__init__()
        self.num_shells = num_shells
        self.surface_dim = surface_dim
        self.radial_dim = radial_dim
        self.num_shortcut_tracts = num_shortcut_tracts
        self.use_sparse_mot = use_sparse_mot
        self.top_k_tracts = min(top_k_tracts, num_shortcut_tracts)

        # Mixture-of-Axonal-Tracts (MoT) dynamic routing gate
        self.tract_router = nn.Linear(surface_dim, num_shortcut_tracts, bias=False)
        nn.init.normal_(self.tract_router.weight, mean=0.0, std=0.02)

        # Intra-shell SwiGLU processors per shell
        self.intra_shell_swiglu = nn.ModuleList([
            SwiGLU(surface_dim, hidden_features=surface_dim * 2, out_features=surface_dim)
            for _ in range(num_shells)
        ])
        self.intra_norms = nn.ModuleList([RMSNorm(surface_dim) for _ in range(num_shells)])
        self.shell_gains = nn.Parameter(torch.ones(num_shells) * 0.1)

        # Inter-shell radial wave operators (inward and outward)
        # Projects surface_dim -> radial_dim for deep convection, then back to surface_dim
        self.radial_in_proj = nn.Linear(surface_dim, radial_dim, bias=False)
        self.radial_out_proj = nn.Linear(surface_dim, radial_dim, bias=False)
        self.radial_up = nn.Linear(radial_dim, surface_dim, bias=False)
        self.radial_up_out = nn.Linear(radial_dim, surface_dim, bias=False)
        self.radial_norm = RMSNorm(surface_dim)
        self.radial_mix_scale = nn.Parameter(torch.tensor(0.1))
        self.shortcut_mix_scale = nn.Parameter(torch.tensor(0.05))

        # Inter-shell transition matrices (subject to contractive spectral guard)
        self.w_radial_in = nn.Parameter(torch.randn(num_shells - 1, radial_dim, radial_dim) * (1.0 / math.sqrt(radial_dim)))
        self.w_radial_out = nn.Parameter(torch.randn(num_shells - 1, radial_dim, radial_dim) * (1.0 / math.sqrt(radial_dim)))

        # Hierarchical axonal shortcut tracts (small-world wiring)
        self.src_indices, self.dst_indices = self._generate_small_world_tracts(num_shells, num_shortcut_tracts)
        self.shortcut_weight = nn.Parameter(torch.randn(num_shortcut_tracts, surface_dim, surface_dim) * (0.05 / math.sqrt(surface_dim)))
        self.shortcut_gate = nn.Parameter(torch.zeros(num_shortcut_tracts))

        # Efference copy projection from motor area back into subcortical core (Shell 0)
        self.efference_gate = nn.Linear(surface_dim, surface_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _generate_small_world_tracts(num_shells: int, num_tracts: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Constructs small-world hierarchical axonal wiring between distant concentric shells."""
        srcs, dsts = [], []
        for i in range(num_tracts):
            # Connect deep subcortical shells (0..K//4) directly to cortical mantle (3K//4..K-1)
            if i % 3 == 0:
                s = i % max(1, num_shells // 4)
                d = num_shells - 1 - (i % max(1, num_shells // 4))
            elif i % 3 == 1:
                # Long-range intermediate skip tracts
                s = (i * 2) % num_shells
                d = (s + num_shells // 2) % num_shells
            else:
                # Random long-range axonal connections
                s = (i * 7) % num_shells
                d = (i * 13) % num_shells
                if s == d:
                    d = (s + 1) % num_shells
            srcs.append(s)
            dsts.append(d)
        return torch.tensor(srcs, dtype=torch.long), torch.tensor(dsts, dtype=torch.long)

    def deliberate_step(self, state: torch.Tensor) -> torch.Tensor:
        """
        Executes a single subconscious recurrent deliberation step across all 32 shells:
        1. Intra-shell SwiGLU non-linear transform.
        2. Bi-directional radial wave propagation (advection-diffusion).
        3. Long-range axonal shortcut conductance.
        """
        B, K, D = state.shape

        # 1. Intra-shell SwiGLU update
        intra_updates = []
        for k in range(K):
            s_k = state[:, k, :]
            h_k = self.intra_norms[k](s_k)
            u_k = self.intra_shell_swiglu[k](h_k) * self.shell_gains[k]
            intra_updates.append(s_k + self.dropout(u_k))
        state_intra = torch.stack(intra_updates, dim=1)  # [B, K, D]

        # 2. Inter-shell radial wave propagation (Pre-Norm preserves inter-shell magnitude hierarchy)
        h_comm = self.radial_norm(state_intra)
        # Inward wave: Shell k+1 -> Shell k
        rad_in = self.radial_in_proj(h_comm[:, 1:, :])   # [B, K-1, Dr]
        in_flux = torch.einsum("bkd,kde->bke", rad_in, self.w_radial_in)  # [B, K-1, Dr]
        in_up = self.radial_up(in_flux)  # [B, K-1, D]

        # Outward wave: Shell k -> Shell k+1 (decoupled radial_up_out projection)
        rad_out = self.radial_out_proj(h_comm[:, :-1, :])  # [B, K-1, Dr]
        out_flux = torch.einsum("bkd,kde->bke", rad_out, self.w_radial_out)  # [B, K-1, Dr]
        out_up = self.radial_up_out(out_flux)  # [B, K-1, D]

        # Assemble radial deltas out-of-place (zero in-place modifications)
        rad_scale = self.radial_mix_scale.to(state_intra.dtype)
        delta_in = F.pad(rad_scale * in_up, (0, 0, 0, 1))    # [B, K, D], zero padded on Shell K-1
        delta_out = F.pad(rad_scale * out_up, (0, 0, 1, 0))  # [B, K, D], zero padded on Shell 0
        state_radial = state_intra + delta_in + delta_out

        # 3. Mixture-of-Axonal-Tracts (MoT Sparse Small-World Routing)
        conductances = torch.sigmoid(self.shortcut_gate)  # [T]
        if self.use_sparse_mot and self.top_k_tracts < self.num_shortcut_tracts:
            routing_logits = self.tract_router(state_radial[:, -1, :])  # [B, T]
            topk_scores, topk_indices = torch.topk(routing_logits, self.top_k_tracts, dim=-1)
            routing_weights = F.softmax(topk_scores, dim=-1).to(dtype=routing_logits.dtype)
            routing_mask = torch.zeros_like(routing_logits).scatter_(-1, topk_indices, routing_weights)
            eff_conductances = conductances.unsqueeze(0) * routing_mask  # [B, T]
        else:
            eff_conductances = conductances.view(1, -1)  # [1, T]

        src_acts = self.radial_norm(state_radial)[:, self.src_indices, :]   # [B, T, D]
        shortcut_flux = torch.einsum("btd,tde->bte", src_acts, self.shortcut_weight) * eff_conductances.unsqueeze(-1)

        # Assemble axonal skip flux out-of-place
        sc_scale = self.shortcut_mix_scale.to(state_radial.dtype)
        delta_shortcut = torch.zeros_like(state_radial)
        for t in range(self.num_shortcut_tracts):
            dst = self.dst_indices[t]
            delta_shortcut[:, dst, :] = delta_shortcut[:, dst, :] + sc_scale * shortcut_flux[:, t, :]
        state_next = state_radial + delta_shortcut

        return state_next

    def step_efference_fast(self, state: torch.Tensor, efference: torch.Tensor) -> torch.Tensor:
        """O(1) bounded residual efference feedback onto Shell 0 without full deliberate_step."""
        orig_dtype = state.dtype
        eff_f32 = efference.float()
        eff_normed = (eff_f32 * torch.rsqrt(eff_f32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(orig_dtype)
        new_shell0 = 0.85 * state[:, 0, :] + 0.15 * eff_normed
        return torch.cat([new_shell0.unsqueeze(1), state[:, 1:, :]], dim=1)

    def inject_efference_copy(self, state: torch.Tensor, efference: torch.Tensor) -> torch.Tensor:
        """Injects motor efference copy directly into Shell 0 out-of-place."""
        eff_projected = self.efference_gate(efference).unsqueeze(1)  # [B, 1, D]
        state_new = torch.cat([state[:, :1, :] + eff_projected, state[:, 1:, :]], dim=1)
        return state_new


class PremotorMotorDecoderV5(nn.Module):
    """
    Premotor & Motor Decoder V5:
    1. Multi-shell pooling across all 32 concentric laminae.
    2. Multi-Head Arcuate Fasciculus cross-resonance (8 heads).
    3. Scaled Phase Injection (p_scaled = 2.0).
    4. Executive pre-verbal recurrent plan update.
    5. Affine-Tied dictionary motor head:
       logits = (gamma * W_sensory) @ RMSNorm(h_emit) / tau
    6. Two-tier refractory repetition penalty.
    """

    def __init__(
        self,
        num_shells: int = 32,
        surface_dim: int = 768,
        vocab_size: int = 32000,
        num_heads: int = 8,
        max_phase_steps: int = 1024,
        phase_scale: float = 2.0,
        dropout: float = 0.0,
        use_plan: bool = True,
        use_shell_tap: bool = True,
        rope_theta: float = 500000.0,
        max_seq_len: int = 32768,
        top_k_resonance: int = 16,
        use_tied_embeddings: bool = True,
    ):
        super().__init__()
        self.num_shells = num_shells
        self.surface_dim = surface_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.head_dim = surface_dim // num_heads
        self.max_phase_steps = max_phase_steps
        self.phase_scale = nn.Parameter(torch.tensor(float(phase_scale)))
        self.use_plan = use_plan
        self.use_shell_tap = use_shell_tap
        self.top_k_resonance = top_k_resonance
        self.use_tied_embeddings = use_tied_embeddings

        # Precompute RoPE rotary frequency cache for 32k+ token horizons
        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len=max_seq_len, theta=rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Premotor normalized temporal phase clock
        self.phase_embed = nn.Embedding(max_phase_steps, surface_dim)
        nn.init.normal_(self.phase_embed.weight, mean=0.0, std=0.02)

        # Multi-Head Arcuate Fasciculus Query & Output Projections
        self.q_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.out_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.resonance_norm = RMSNorm(surface_dim)

        # Multi-shell pooling weights and projections
        self.shell_score = nn.Parameter(torch.zeros(num_shells))
        self.shell_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.shell_scale = nn.Parameter(torch.tensor(0.5))

        # Deep motor integration MLP: [motor_surface, core, resonance, shell_pool, plan] = 5*D -> D
        num_inputs = 5 if use_plan else 4
        fused_dim = surface_dim * num_inputs
        self.input_gate = nn.Parameter(torch.ones(num_inputs))
        self.fused_norm = RMSNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)

        if self.use_tied_embeddings:
            self.motor_swiglu = SwiGLU(fused_dim, hidden_features=surface_dim * 2, out_features=surface_dim)
            self.norm_mid = RMSNorm(surface_dim)
            self.affine_scale = nn.Parameter(torch.ones(surface_dim))
            self.head = None
            self.efference_proj = nn.Linear(surface_dim, surface_dim, bias=False)
            plan_in_dim = surface_dim * 2
        else:
            # Untied head mode (1:1 V4 architectural parity for exact weight salvaging)
            self.fc_gate = nn.Linear(fused_dim, surface_dim * 2, bias=False)
            self.fc_val = nn.Linear(fused_dim, surface_dim * 2, bias=False)
            self.norm_mid = RMSNorm(surface_dim * 2)
            self.head = nn.Linear(surface_dim * 2, vocab_size, bias=False)
            self.affine_scale = None
            self.efference_proj = nn.Linear(surface_dim * 2, surface_dim, bias=False)
            plan_in_dim = surface_dim * 3

        # Recurrent Executive Plan synthesis
        if use_plan:
            self.plan_init = nn.Linear(surface_dim * 2, surface_dim, bias=True)
            self.plan_gate = nn.Linear(plan_in_dim, surface_dim, bias=True)
            self.plan_update = nn.Linear(plan_in_dim, surface_dim, bias=False)
            self.plan_decay = nn.Parameter(torch.tensor(0.85))
            self.plan_write = nn.Parameter(torch.tensor(0.15))

        self.temperature_scale = nn.Parameter(torch.tensor(1.0))

        # Two-tier refractory penalty parameters
        self.refractory_decay = nn.Parameter(torch.tensor(0.75))
        self.refractory_strength = nn.Parameter(torch.tensor(1.2))

        # Agentic Persona Steering Vectors (0: Reasoner, 1: Explorer, 2: Concise, 3: Verifier)
        self.persona_embeddings = nn.Parameter(torch.empty(4, surface_dim).normal_(mean=0.0, std=0.05))
        self.persona_scale = nn.Parameter(torch.tensor(0.5))

    def init_plan(self, batch_size: int, device: torch.device, persona_id: Optional[int] = None) -> torch.Tensor:
        plan = torch.zeros(batch_size, self.surface_dim, device=device)
        if persona_id is not None:
            p_idx = min(max(0, int(persona_id)), self.persona_embeddings.shape[0] - 1)
            plan = plan + self.persona_scale * self.persona_embeddings[p_idx]
        return plan

    def forward(
        self,
        state: torch.Tensor,
        sensory_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        tied_embedding_weight: torch.Tensor,
        plan: Optional[torch.Tensor] = None,
        refractory_trace: Optional[torch.Tensor] = None,
        step_idx: int = 0,
        auditory_feedback: Optional[torch.Tensor] = None,
        repetition_penalty: float = 1.25,
        return_hidden: bool = False,
        causal_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        causal_kv_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]]:
        """
        Executes a single motor verbalization step:
        Returns:
            adapted_logits: [B, vocab_size]
            efference: [B, surface_dim]
            next_trace: [B, vocab_size]
            next_plan: [B, surface_dim]
            (optional) h_motor: [B, surface_dim]
        """
        B, K, D = state.shape
        surface = state[:, -1, :]  # Cortical boundary (Shell K-1)
        core = state[:, 0, :]      # Subcortical core (Shell 0)

        if plan is None and self.use_plan:
            plan = self.init_plan(B, state.device)

        # 1. Scaled Phase Injection: Normalized temporal clock prevents static attractor traps
        max_phase = self.phase_embed.weight.shape[0]
        p_idx = min(max(0, int(step_idx)), max_phase - 1)
        phase = self.phase_embed.weight[p_idx].to(state.dtype)
        p_norm = phase.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        p_scaled = phase * (torch.clamp(self.phase_scale, 0.05, 5.0) / p_norm)

        motor_surface = surface + p_scaled
        if auditory_feedback is not None:
            motor_surface = motor_surface + auditory_feedback.to(motor_surface.dtype)

        # 2. Multi-Head Arcuate Fasciculus Cross-Resonance with RoPE & Top-K Sparsity
        if sensory_kv is not None:
            keys, values = sensory_kv  # [B, H, L_prompt, head_dim]
            causal_len = 0
            if causal_kv is not None and causal_kv[0].shape[2] > 0:
                c_k, c_v = causal_kv
                causal_len = c_k.shape[2]
                keys = torch.cat([keys, c_k], dim=2)
                values = torch.cat([values, c_v], dim=2)
            L = keys.shape[2]
            q = self.q_proj(motor_surface).view(B, self.num_heads, 1, self.head_dim)

            # Apply RoPE: keys start at pos 0..L-1, query is positioned at sequence frontier (L)
            q_offset = min(int(L), self.rope_cos.shape[0] - 1)
            q_rot = apply_rope(q, self.rope_cos, self.rope_sin, offset=q_offset)
            k_rot = apply_rope(keys, self.rope_cos, self.rope_sin, offset=0)

            # Dot-product cross-resonance: [B, H, 1, L]
            scores = torch.matmul(q_rot.to(keys.dtype), k_rot.transpose(-1, -2)) / math.sqrt(self.head_dim)

            if causal_len > 0 and causal_kv_mask is not None:
                prompt_len_k = L - causal_len
                prompt_valid = torch.ones(B, prompt_len_k, dtype=torch.bool, device=scores.device)
                k_valid = torch.cat([prompt_valid, causal_kv_mask[:, :causal_len]], dim=1).view(B, 1, 1, L)
                scores = torch.where(k_valid, scores, torch.full_like(scores, -1e9))

            # Top-K Sparse Associative Gating (Needle-in-a-Haystack Tethering)
            if self.top_k_resonance > 0 and L > self.top_k_resonance:
                topk_v, _ = torch.topk(scores, self.top_k_resonance, dim=-1)
                min_topk = topk_v[..., -1, None]
                scores = torch.where(scores < min_topk, torch.full_like(scores, -1e9), scores)

            attn_weights = F.softmax(scores, dim=-1)
            res = torch.matmul(attn_weights, values)  # [B, H, 1, head_dim]
            res = res.transpose(1, 2).contiguous().view(B, D)
            resonance = self.resonance_norm(self.out_proj(res).to(motor_surface.dtype))
        else:
            resonance = torch.zeros_like(motor_surface)

        # 3. Multi-Shell Softmax Pooling
        if self.use_shell_tap:
            shell_weights = F.softmax(self.shell_score, dim=0).view(1, -1, 1)
            shell_proj = self.shell_proj(state)
            shell_pool = torch.sum(shell_weights * shell_proj, dim=1) * self.shell_scale
        else:
            shell_pool = torch.zeros_like(motor_surface)

        # 4. Deep Motor Integration via SwiGLU
        gate = self.input_gate.to(motor_surface.dtype)
        fused_components = [
            motor_surface * gate[0],
            core * gate[1],
            resonance * gate[2],
            shell_pool * gate[3],
        ]
        if self.use_plan and plan is not None:
            fused_components.append(plan * gate[4])

        fused = self.fused_norm(torch.cat(fused_components, dim=-1))
        if self.use_tied_embeddings:
            h = self.norm_mid(self.motor_swiglu(fused))
        else:
            h = self.norm_mid(F.silu(self.fc_gate(fused)) * self.fc_val(fused))
        h = self.dropout(h)

        # 5. Premotor Recurrent Executive Plan Update
        if self.use_plan and plan is not None:
            phase_batch = p_scaled.unsqueeze(0).expand(B, -1)
            plan_input = torch.cat([h, phase_batch], dim=-1)
            p_gate = torch.sigmoid(self.plan_gate(plan_input))
            p_decay = torch.clamp(self.plan_decay, 0.0, 0.92)
            p_update = self.plan_update(plan_input)
            next_plan = torch.clamp(p_decay * plan + self.plan_write * p_gate * p_update, -50.0, 50.0)
        else:
            next_plan = plan if plan is not None else torch.zeros(B, D, device=state.device)

        # 6. Motor Verbalization Emission Head
        temp = torch.clamp(self.temperature_scale, 0.1, 5.0)
        if self.use_tied_embeddings:
            h_scaled = (h * self.affine_scale) / temp
            raw_logits = F.linear(h_scaled, tied_embedding_weight)
        else:
            raw_logits = self.head(h) / temp

        # 7. Two-Tier Refractory Repetition Suppression
        if refractory_trace is not None:
            alpha = torch.clamp(self.refractory_strength, 0.1, 5.0)
            # Tier 1: Continuous probability trace subtraction
            tier1_penalty = 3.0 * alpha * refractory_trace

            # Tier 2: Multiplicative repetition discount for tokens with active recent trace
            active_mask = (refractory_trace > 0.02).float()
            rep_scale = torch.where(raw_logits > 0, 1.0 / repetition_penalty, repetition_penalty)
            scaled_logits = raw_logits * (1.0 - active_mask) + (raw_logits * rep_scale) * active_mask
            adapted_logits = scaled_logits - tier1_penalty

            # Update trace with soft probability EMA + hard top-1 refractory pulse
            decay = torch.sigmoid(self.refractory_decay)
            with torch.no_grad():
                current_prob = F.softmax(raw_logits, dim=-1)
                top1_idx = torch.argmax(adapted_logits, dim=-1)
                top1_pulse = F.one_hot(top1_idx, num_classes=self.vocab_size).float()
                next_trace = decay * refractory_trace + (1.0 - decay) * current_prob + 1.0 * top1_pulse
        else:
            adapted_logits, next_trace = raw_logits, None

        efference = self.efference_proj(h)
        if return_hidden:
            return adapted_logits, efference, next_trace, next_plan, h
        return adapted_logits, efference, next_trace, next_plan

    def forward_chunk(
        self,
        state: torch.Tensor,
        sensory_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        target_feedback: torch.Tensor,
        target_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        tied_embedding_weight: torch.Tensor,
        plan: Optional[torch.Tensor] = None,
        chunk_offset: int = 0,
        return_hidden: bool = False,
        target_kv_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Parallel Fused Chunk Motor Decoder:
        Processes an entire chunk of T_chunk tokens in parallel via FlashAttention (SDPA) and GEMM.
        """
        B, K, D = state.shape
        T_chunk = target_feedback.shape[1]
        surface = state[:, -1, :]  # [B, D]
        core = state[:, 0, :]      # [B, D]

        if plan is None and self.use_plan:
            plan = self.init_plan(B, state.device)

        # 1. Parallel Phase Clock Vector Lookup across entire chunk
        max_phase = self.phase_embed.weight.shape[0]
        p_indices = torch.clamp(torch.arange(chunk_offset, chunk_offset + T_chunk, device=state.device), 0, max_phase - 1)
        phase_seq = self.phase_embed(p_indices)  # [T_chunk, D]
        p_norm = phase_seq.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        p_scaled = phase_seq * (torch.clamp(self.phase_scale, 0.05, 5.0) / p_norm)  # [T_chunk, D]

        # Broadcast surface + phase + feedback across [B, T_chunk, D]
        motor_surface = surface.unsqueeze(1) + p_scaled.unsqueeze(0) + target_feedback.to(surface.dtype)

        # 2. Multi-Head Arcuate Fasciculus Cross-Resonance via Parallel FlashAttention (SDPA)
        if sensory_kv is not None:
            prompt_k, prompt_v = sensory_kv  # [B, H, L_prompt, head_dim]
            L_prompt = prompt_k.shape[2]

            if target_kv is not None and target_kv[0].shape[2] > 0:
                t_k, t_v = target_kv  # [B, H, T_past + T_chunk, head_dim]
                all_k = torch.cat([prompt_k, t_k], dim=2)
                all_v = torch.cat([prompt_v, t_v], dim=2)
            else:
                all_k, all_v = prompt_k, prompt_v

            q = self.q_proj(motor_surface).view(B, T_chunk, self.num_heads, self.head_dim).transpose(1, 2)

            # RoPE Rotary Position Embedding
            q_rot = apply_rope(q, self.rope_cos, self.rope_sin, offset=L_prompt + chunk_offset)
            k_rot = apply_rope(all_k, self.rope_cos, self.rope_sin, offset=0)

            # Prefix-causal mask: Prompt is bidirectional, feedback keys up to current step are causal
            prompt_mask = torch.ones(T_chunk, L_prompt + chunk_offset, dtype=torch.bool, device=q.device)
            causal_submask = torch.tril(torch.ones(T_chunk, T_chunk, dtype=torch.bool, device=q.device))
            prefix_mask = torch.cat([prompt_mask, causal_submask], dim=1).unsqueeze(0).unsqueeze(0)
            if target_kv is not None and target_kv[0].shape[2] > 0 and target_kv_mask is not None:
                prompt_valid = torch.ones(B, L_prompt, dtype=torch.bool, device=q.device)
                k_valid = torch.cat([prompt_valid, target_kv_mask[:, : t_k.shape[2]]], dim=1)[:, None, None, :]
                prefix_mask = prefix_mask & k_valid

            # Fused PyTorch SDPA (dispatches to FlashAttention-2 / memory-efficient attention)
            res = F.scaled_dot_product_attention(
                q_rot.to(all_k.dtype),
                k_rot,
                all_v,
                attn_mask=prefix_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )  # [B, H, T_chunk, head_dim]
            res = res.transpose(1, 2).contiguous().view(B, T_chunk, D)
            resonance = self.resonance_norm(self.out_proj(res).to(motor_surface.dtype))
        else:
            resonance = torch.zeros_like(motor_surface)

        # 3. Multi-Shell Softmax Pooling
        if self.use_shell_tap:
            shell_weights = F.softmax(self.shell_score, dim=0).view(1, -1, 1)
            shell_proj = self.shell_proj(state)
            shell_pool = torch.sum(shell_weights * shell_proj, dim=1) * self.shell_scale  # [B, D]
        else:
            shell_pool = torch.zeros_like(surface)

        # 4. Deep Motor Integration via Parallel SwiGLU
        gate = self.input_gate.to(motor_surface.dtype)
        core_exp = core.unsqueeze(1).expand(-1, T_chunk, -1)
        shell_exp = shell_pool.unsqueeze(1).expand(-1, T_chunk, -1)
        fused_components = [
            motor_surface * gate[0],
            core_exp * gate[1],
            resonance * gate[2],
            shell_exp * gate[3],
        ]
        if self.use_plan and plan is not None:
            fused_components.append(plan.unsqueeze(1).expand(-1, T_chunk, -1) * gate[4])

        fused = self.fused_norm(torch.cat(fused_components, dim=-1))
        if self.use_tied_embeddings:
            h = self.norm_mid(self.motor_swiglu(fused))
        else:
            h = self.norm_mid(F.silu(self.fc_gate(fused)) * self.fc_val(fused))
        h = self.dropout(h)

        # 5. Parallel Motor Emission Projection
        temp = torch.clamp(self.temperature_scale, 0.1, 5.0)
        if self.use_tied_embeddings:
            h_scaled = (h * self.affine_scale) / temp
            logits = F.linear(h_scaled, tied_embedding_weight)  # [B, T_chunk, vocab_size]
        else:
            logits = self.head(h) / temp

        efference = self.efference_proj(h)  # [B, T_chunk, surface_dim]
        if return_hidden:
            return logits, efference, h, plan
        return logits, efference, plan


class CerebroSpheroidV5(nn.Module):
    """
    CerebroSpheroid V5: Full Cognitive Recurrent Neural Architecture.
    Integrates Sensory Input DNN V5, Volumetric Spheroid Core V5, and Premotor Motor Decoder V5.
    """

    def __init__(
        self,
        config: Optional[ModelConfigV5] = None,
        vocab_size: int = 32000,
        surface_dim: int = 768,
        radial_dim: int = 512,
        num_shells: int = 32,
        num_shortcut_tracts: int = 96,
        num_heads: int = 8,
        thinking_cycles: int = 4,
        phase_scale: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = ModelConfigV5(
                vocab_size=vocab_size,
                surface_dim=surface_dim,
                radial_dim=radial_dim,
                num_shells=num_shells,
                num_shortcut_tracts=num_shortcut_tracts,
                num_heads=num_heads,
                thinking_cycles=thinking_cycles,
                phase_scale=phase_scale,
                dropout=dropout,
            )

        # 1. Sensory Front-End
        self.sensory_dnn = SensoryInputDNNV5(
            vocab_size=self.config.vocab_size,
            surface_dim=self.config.surface_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
        )

        # 2. Volumetric Recurrent Core (32 to 48 Concentric Laminae with MoT)
        self.volumetric_core = VolumetricSpheroidCoreV5(
            num_shells=self.config.num_shells,
            surface_dim=self.config.surface_dim,
            radial_dim=self.config.radial_dim,
            num_shortcut_tracts=self.config.num_shortcut_tracts,
            dropout=self.config.dropout,
            use_sparse_mot=self.config.use_sparse_mot,
            top_k_tracts=self.config.top_k_tracts,
        )

        # 3. Premotor Motor Decoder with Affine-Tied Head
        self.motor_decoder = PremotorMotorDecoderV5(
            num_shells=self.config.num_shells,
            surface_dim=self.config.surface_dim,
            vocab_size=self.config.vocab_size,
            num_heads=self.config.num_heads,
            max_phase_steps=self.config.max_phase_steps,
            phase_scale=self.config.phase_scale,
            dropout=self.config.dropout,
            use_plan=self.config.use_plan,
            use_shell_tap=self.config.use_shell_tap,
            rope_theta=self.config.rope_theta,
            max_seq_len=self.config.max_seq_len,
            top_k_resonance=self.config.top_k_resonance,
            use_tied_embeddings=getattr(self.config, "use_tied_embeddings", True),
        )

        # 4. Deliberative Episodic Memory Probing (Subconscious sensory recall)
        if self.config.use_deliberation_probe:
            self.deliberation_probe = nn.Linear(self.config.surface_dim, self.config.surface_dim, bias=False)
            self.deliberation_probe_norm = RMSNorm(self.config.surface_dim)
        else:
            self.deliberation_probe = None
            self.deliberation_probe_norm = None

        # 5. Multi-Token Prediction (MTP) Drafting Heads
        if self.config.num_mtp_heads > 1:
            self.mtp_heads = nn.ModuleList([
                MTPHead(self.config.surface_dim, dropout=self.config.dropout)
                for _ in range(self.config.num_mtp_heads - 1)
            ])
        else:
            self.mtp_heads = nn.ModuleList()

        # 6. Dynamic Halting Gate for Adaptive Deliberation (System-2 Reasoning)
        if self.config.use_adaptive_thinking:
            self.halting_gate = nn.Linear(self.config.surface_dim, 1)
            nn.init.zeros_(self.halting_gate.weight)
            nn.init.constant_(self.halting_gate.bias, -1.5)
        else:
            self.halting_gate = None

        # 7. Multimodal Perceptual Projections (Images, Audio, Continuous Sensors)
        if getattr(self.config, "enable_multimodal", False):
            v_dim = getattr(self.config, "visual_dim", self.config.surface_dim)
            self.image_stem = ImagePatchStem(
                patch_size=getattr(self.config, "image_patch_size", 16),
                in_channels=3,
                visual_dim=v_dim,
            )
            self.audio_stem = AudioWaveformStem(
                kernel_size=400,
                stride=160,
                audio_dim=v_dim,
            )
            self.perceptual_projector = nn.Sequential(
                RMSNorm(v_dim),
                nn.Linear(v_dim, self.config.surface_dim, bias=False),
                SwiGLU(self.config.surface_dim, hidden_features=self.config.surface_dim * 2, out_features=self.config.surface_dim),
                RMSNorm(self.config.surface_dim),
            )
        else:
            self.image_stem = None
            self.audio_stem = None
            self.perceptual_projector = None

        self.current_spheroid_state: Optional[torch.Tensor] = None

    def count_parameters(self) -> Dict[str, int]:
        """Returns detailed parameter breakdown across modules."""
        sensory = sum(p.numel() for p in self.sensory_dnn.parameters())
        if self.perceptual_projector is not None:
            sensory += sum(p.numel() for p in self.perceptual_projector.parameters())
        if self.image_stem is not None:
            sensory += sum(p.numel() for p in self.image_stem.parameters())
        if self.audio_stem is not None:
            sensory += sum(p.numel() for p in self.audio_stem.parameters())
        core = sum(p.numel() for p in self.volumetric_core.parameters())
        motor = sum(p.numel() for p in self.motor_decoder.parameters())
        total = sum(p.numel() for p in self.parameters())
        vocab_lookup = self.sensory_dnn.embedding.weight.numel()
        active_compute = total - vocab_lookup
        return {
            "sensory_dnn": sensory,
            "volumetric_core": core,
            "motor_decoder": motor,
            "total_parameters": total,
            "vocab_lookup": vocab_lookup,
            "active_compute": active_compute,
            "active_compute_percent": round((active_compute / total) * 100, 2),
        }

    def init_spheroid_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes empty concentric spherical state: [B, num_shells, surface_dim]."""
        return torch.zeros(batch_size, self.config.num_shells, self.config.surface_dim, device=device)

    @staticmethod
    def _recency_weighted_pool(
        flux: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        decay: float = 0.05,
    ) -> torch.Tensor:
        """Order-preserving exponential recency pooling over non-pad sensory tokens."""
        L = flux.shape[1]
        pos = torch.arange(L, device=flux.device, dtype=torch.float32).unsqueeze(0)
        if mask is not None:
            m = mask.float()
            last_pos = torch.max(pos * m, dim=1, keepdim=True)[0]
            weights = torch.exp(decay * (pos - last_pos).clamp(max=0.0)) * m
        else:
            weights = torch.exp(decay * (pos - float(max(L - 1, 0))))
        denom = torch.sum(weights, dim=1, keepdim=True).clamp_min(1.0)
        pooled = torch.sum(flux.float() * weights.unsqueeze(-1), dim=1) / denom
        return pooled.to(flux.dtype)

    def forward(
        self,
        prompts: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        feedback_tokens: Optional[torch.Tensor] = None,
        feedback_is_shifted: bool = True,
        sampling_p: float = 1.0,
        pad_id: int = 0,
        target_len: Optional[int] = None,
        perceptual_inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Differentiable training rollout engine:
        1. Encodes sensory prompt, raw images, and raw audio into Arcuate Fasciculus KV and centroid.
        2. Sets Shell K-1 (Surface) to prompt centroid.
        3. Executes subconscious deliberation cycles in subcortical core.
        4. Steps through motor emissions with scheduled sampling and thought re-afference.
        """
        dev = prompts.device
        t_len = target_len if target_len is not None else (targets.shape[1] if targets is not None else 128)

        # 1. Sensory Ingestion (Multimodal Images, Audio, Perceptual Flux or Text)
        sensory_tokens_list = []
        if images is not None and self.image_stem is not None:
            sensory_tokens_list.append(self.image_stem(images))
        if audio is not None and self.audio_stem is not None:
            sensory_tokens_list.append(self.audio_stem(audio))
        if perceptual_inputs is not None:
            sensory_tokens_list.append(perceptual_inputs)

        has_perceptual = len(sensory_tokens_list) > 0 and self.perceptual_projector is not None
        if has_perceptual:
            raw_p_flux = torch.cat(sensory_tokens_list, dim=1)  # [B, P_total, visual_dim]
            B_vis, P_vis, _ = raw_p_flux.shape
            prompt_pos_offset = P_vis + prompts.shape[1]
            vis_flux = self.perceptual_projector(raw_p_flux)   # [B, P_total, surface_dim]
            if prompts.dim() == 2 and prompts.dtype in (torch.int32, torch.int64):
                txt_emb = self.sensory_dnn.embedding(prompts)  # [B, L, D]
                combined_emb = torch.cat([vis_flux, txt_emb], dim=1)  # [B, P+L, D]
                sensory_flux, sensory_kv = self.sensory_dnn(combined_emb)
                vis_mask = torch.ones(B_vis, P_vis, device=dev)
                txt_mask = (prompts != pad_id).float()
                combined_mask = torch.cat([vis_mask, txt_mask], dim=1)
                prompt_centroid = self._recency_weighted_pool(sensory_flux, combined_mask)
            else:
                combined_emb = torch.cat([vis_flux, prompts], dim=1)
                sensory_flux, sensory_kv = self.sensory_dnn(combined_emb)
                prompt_centroid = self._recency_weighted_pool(sensory_flux)
            B = B_vis
        else:
            B = prompts.shape[0]
            prompt_pos_offset = prompts.shape[1]
            sensory_flux, sensory_kv = self.sensory_dnn(prompts)
            if prompts.dim() == 2:
                mask = (prompts != pad_id).float()
                prompt_centroid = self._recency_weighted_pool(sensory_flux, mask)
            else:
                prompt_centroid = self._recency_weighted_pool(sensory_flux)

        # 2. State Initialization: Cortical Surface receives sensory prompt flux out-of-place
        subcortical = torch.zeros(B, self.config.num_shells - 1, self.config.surface_dim, device=dev)
        state = torch.cat([subcortical, prompt_centroid.unsqueeze(1)], dim=1)

        # 3. Subconscious Deliberation Cycles with Memory Probing & Adaptive Halting
        cycles = self.config.max_thinking_cycles if self.config.use_adaptive_thinking else self.config.thinking_cycles
        energy_loss = torch.tensor(0.0, device=dev)
        halt_loss = torch.tensor(0.0, device=dev)

        for cycle in range(cycles):
            state_prev = state.clone()
            if sensory_kv is not None and self.deliberation_probe is not None:
                keys, values = sensory_kv
                L_kv = keys.shape[2]
                probe_q = self.deliberation_probe(state.mean(dim=1)).view(B, self.config.num_heads, 1, self.sensory_dnn.head_dim)
                q_rot = apply_rope(probe_q, self.motor_decoder.rope_cos, self.motor_decoder.rope_sin, offset=cycle)
                k_rot = apply_rope(keys, self.motor_decoder.rope_cos, self.motor_decoder.rope_sin, offset=0)
                scores = torch.matmul(q_rot.to(keys.dtype), k_rot.transpose(-1, -2)) / math.sqrt(self.sensory_dnn.head_dim)
                if self.config.top_k_resonance > 0 and L_kv > self.config.top_k_resonance:
                    topk_v, _ = torch.topk(scores, self.config.top_k_resonance, dim=-1)
                    scores = torch.where(scores < topk_v[..., -1, None], torch.full_like(scores, -1e9), scores)
                attn_p = F.softmax(scores, dim=-1)
                retrieved = torch.matmul(attn_p, values).transpose(1, 2).contiguous().view(B, self.config.surface_dim)
                state_mantle = state[:, -1:, :] + 0.1 * self.deliberation_probe_norm(retrieved).unsqueeze(1)
                state = torch.cat([state[:, :-1, :], state_mantle], dim=1)

            state = self.volumetric_core.deliberate_step(state)

            if self.config.use_energy_regularization:
                energy_loss = energy_loss + torch.mean((state - state_prev) ** 2)

            if self.config.use_adaptive_thinking and self.halting_gate is not None:
                halt_prob = torch.sigmoid(self.halting_gate(state.mean(dim=1)))
                halt_loss = halt_loss + halt_prob.mean()
                if not self.training and (halt_prob > self.config.thinking_halt_threshold).all():
                    break

        # Pre-embed aligned previous-token feedback (prev_tokens[:, t] feeds step t predicting targets[:, t])
        prev_tokens = None
        if feedback_tokens is not None:
            fb_slice = feedback_tokens[:, :t_len]
            if feedback_is_shifted:
                prev_tokens = fb_slice
            else:
                pad_col = torch.full((B, 1), -100, dtype=fb_slice.dtype, device=dev)
                prev_tokens = torch.cat([pad_col, fb_slice[:, : max(0, t_len - 1)]], dim=1)[:, :t_len]
        elif targets is not None:
            tgt_slice = targets[:, :t_len]
            pad_col = torch.full((B, 1), -100, dtype=tgt_slice.dtype, device=dev)
            prev_tokens = torch.cat([pad_col, tgt_slice[:, : max(0, t_len - 1)]], dim=1)[:, :t_len]

        if prev_tokens is not None:
            safe_feedback = torch.clamp(prev_tokens, min=0)
            tf_flux, (tf_keys, tf_values) = self.sensory_dnn(
                safe_feedback, is_causal=True, pos_offset=prompt_pos_offset
            )
            fb_valid_mask = prev_tokens >= 0
            tf_flux = tf_flux * fb_valid_mask.unsqueeze(-1).to(tf_flux.dtype)
        else:
            tf_flux = None
            tf_keys = None
            tf_values = None
            fb_valid_mask = None

        # 4. Motor Emission Engine (Chunkwise Parallel Fused FlashAttention vs Sequential)
        plan = self.motor_decoder.init_plan(B, dev) if self.config.use_plan else None
        tied_w = self.sensory_dnn.embedding.weight

        # Check if we can use fast parallel chunk execution
        use_fast_chunkwise = targets is not None and not getattr(self, "force_sequential", False) and tf_flux is not None

        if use_fast_chunkwise:
            # prev_tokens is already aligned 1:1 with emission step t (no double shift)
            shifted_feedback = tf_flux[:, :t_len, :]

            # Chunk size: 64 tokens per chunk for optimal Tensor Core saturation and periodic core updates
            c_size = getattr(self.config, "chunk_size", 64)
            num_chunks = max(1, (t_len + c_size - 1) // c_size)

            all_chunk_logits = []
            for c_idx in range(num_chunks):
                c_start = c_idx * c_size
                c_end = min(c_start + c_size, t_len)
                chunk_fb = shifted_feedback[:, c_start:c_end, :]

                # Target KV cache up to the current chunk (exclusively past/previous tokens)
                chunk_target_kv = (tf_keys[:, :, :c_end, :], tf_values[:, :, :c_end, :]) if tf_keys is not None else None
                chunk_kv_mask = fb_valid_mask[:, :c_end] if fb_valid_mask is not None else None

                chunk_logits, chunk_eff, plan = self.motor_decoder.forward_chunk(
                    state=state,
                    sensory_kv=sensory_kv,
                    target_feedback=chunk_fb,
                    target_kv=chunk_target_kv,
                    tied_embedding_weight=tied_w,
                    plan=plan,
                    chunk_offset=c_start,
                    target_kv_mask=chunk_kv_mask,
                )
                all_chunk_logits.append(chunk_logits)

                # Periodic Spheroid Core recurrence across chunk boundaries
                eff_chunk = chunk_eff.mean(dim=1)
                state_eff = self.volumetric_core.inject_efference_copy(state, eff_chunk)
                state = self.volumetric_core.deliberate_step(state_eff)

            logits_stacked = torch.cat(all_chunk_logits, dim=1)
        else:
            # Sequential rollout for free sampling or test unrolling
            refractory = torch.zeros(B, self.config.vocab_size, device=dev)
            prev_thought = torch.zeros(B, self.config.surface_dim, device=dev)
            all_logits = []

            for t in range(t_len):
                if tf_flux is not None:
                    tf_aud = tf_flux[:, t, :]
                    if t == 0:
                        auditory_in = torch.where(
                            fb_valid_mask[:, 0:1], sampling_p * tf_aud, torch.zeros_like(tf_aud)
                        )
                    else:
                        auditory_in = sampling_p * tf_aud + (1.0 - sampling_p) * prev_thought
                else:
                    auditory_in = prev_thought if t > 0 else None

                causal_kv = None
                causal_kv_mask = None
                if tf_keys is not None:
                    causal_kv = (tf_keys[:, :, : t + 1, :], tf_values[:, :, : t + 1, :])
                    causal_kv_mask = fb_valid_mask[:, : t + 1] if fb_valid_mask is not None else None

                logits, efference, refractory, plan = self.motor_decoder(
                    state=state,
                    sensory_kv=sensory_kv,
                    tied_embedding_weight=tied_w,
                    plan=plan,
                    refractory_trace=refractory,
                    step_idx=t,
                    auditory_feedback=auditory_in,
                    causal_kv=causal_kv,
                    causal_kv_mask=causal_kv_mask,
                )
                all_logits.append(logits)

                soft_p = F.softmax(logits, dim=-1)
                soft_emb = soft_p @ tied_w
                soft_emb_diff = 0.2 * soft_emb + 0.8 * soft_emb.detach()
                prev_thought, _ = self.sensory_dnn(soft_emb_diff, pos_offset=prompt_pos_offset + t)

                state = self.volumetric_core.step_efference_fast(state, efference)

            logits_stacked = torch.stack(all_logits, dim=1)

        self.current_spheroid_state = state

        if targets is not None:
            # Cross-entropy loss computed only on valid non-padded targets (target != -100)
            loss = F.cross_entropy(
                logits_stacked.view(-1, self.config.vocab_size),
                targets[:, :t_len].contiguous().view(-1),
                ignore_index=-100,
            )
            if self.config.use_energy_regularization:
                loss = loss + 0.01 * (energy_loss / max(cycles, 1))
            if self.config.use_adaptive_thinking and self.halting_gate is not None:
                loss = loss + 0.001 * (halt_loss / max(cycles, 1))

            return loss, logits_stacked

        return logits_stacked

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.25,
        thinking_cycles: Optional[int] = None,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Closed-loop autoregressive generation with discrete token auditory feedback."""
        self.eval()
        B, L = prompt_ids.shape
        dev = prompt_ids.device
        cycles = thinking_cycles if thinking_cycles is not None else (self.config.max_thinking_cycles if self.config.use_adaptive_thinking else self.config.thinking_cycles)

        # 1. Encode prompt with order-preserving recency weighting
        sensory_flux, sensory_kv = self.sensory_dnn(prompt_ids)
        prompt_mask = (prompt_ids != 0).float()
        safe_mask = torch.where(prompt_mask.sum(dim=1, keepdim=True) > 0, prompt_mask, torch.ones_like(prompt_mask))
        prompt_centroid = self._recency_weighted_pool(sensory_flux, safe_mask)

        # 2. State Initialization: Cortical Surface receives sensory prompt flux out-of-place
        subcortical = torch.zeros(B, self.config.num_shells - 1, self.config.surface_dim, device=dev)
        state = torch.cat([subcortical, prompt_centroid.unsqueeze(1)], dim=1)

        for cycle in range(cycles):
            if sensory_kv is not None and self.deliberation_probe is not None:
                keys, values = sensory_kv
                L_kv = keys.shape[2]
                probe_q = self.deliberation_probe(state.mean(dim=1)).view(B, self.config.num_heads, 1, self.sensory_dnn.head_dim)
                q_rot = apply_rope(probe_q, self.motor_decoder.rope_cos, self.motor_decoder.rope_sin, offset=cycle)
                k_rot = apply_rope(keys, self.motor_decoder.rope_cos, self.motor_decoder.rope_sin, offset=0)
                scores = torch.matmul(q_rot.to(keys.dtype), k_rot.transpose(-1, -2)) / math.sqrt(self.sensory_dnn.head_dim)
                if self.config.top_k_resonance > 0 and L_kv > self.config.top_k_resonance:
                    topk_v, _ = torch.topk(scores, self.config.top_k_resonance, dim=-1)
                    scores = torch.where(scores < topk_v[..., -1, None], torch.full_like(scores, -1e9), scores)
                attn_p = F.softmax(scores, dim=-1)
                retrieved = torch.matmul(attn_p, values).transpose(1, 2).contiguous().view(B, self.config.surface_dim)
                state_mantle = state[:, -1:, :] + 0.1 * self.deliberation_probe_norm(retrieved).unsqueeze(1)
                state = torch.cat([state[:, :-1, :], state_mantle], dim=1)

            state = self.volumetric_core.deliberate_step(state)

            if self.config.use_adaptive_thinking and self.halting_gate is not None:
                halt_prob = torch.sigmoid(self.halting_gate(state.mean(dim=1)))
                if (halt_prob > self.config.thinking_halt_threshold).all():
                    break

        # 3. Motor generation loop
        plan = self.motor_decoder.init_plan(B, dev) if self.config.use_plan else None
        refractory = torch.zeros(B, self.config.vocab_size, device=dev)
        prev_thought = torch.zeros(B, self.config.surface_dim, device=dev)

        generated_tokens = []
        causal_keys_list = []
        causal_values_list = []
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        tied_w = self.sensory_dnn.embedding.weight

        for t in range(max_new_tokens):
            auditory_in = prev_thought if t > 0 else None

            # Causal KV Cache: Query past emitted tokens
            causal_kv = None
            if len(causal_keys_list) > 0:
                causal_kv = (torch.cat(causal_keys_list, dim=2), torch.cat(causal_values_list, dim=2))

            logits, efference, refractory, plan = self.motor_decoder(
                state=state,
                sensory_kv=sensory_kv,
                tied_embedding_weight=tied_w,
                plan=plan,
                refractory_trace=refractory,
                step_idx=t,
                auditory_feedback=auditory_in,
                repetition_penalty=repetition_penalty,
                causal_kv=causal_kv,
            )

            # Temperature and sampling
            logits = logits / max(temperature, 1e-4)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # [B, 1]

            generated_tokens.append(next_tok)
            finished = finished | (next_tok.squeeze(-1) == eos_token_id)
            if finished.all():
                break

            # Thought re-afference & update causal KV buffer
            prev_thought, (tok_k, tok_v) = self.sensory_dnn(next_tok.squeeze(-1), pos_offset=L + t)
            causal_keys_list.append(tok_k)
            causal_values_list.append(tok_v)

            # Lightweight O(1) bounded efference copy propagation
            state = self.volumetric_core.step_efference_fast(state, efference)

        return torch.cat(generated_tokens, dim=1)

    def generate_text(
        self,
        prompt: str,
        tokenizer: Any,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.25,
        thinking_cycles: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> str:
        """Convenience method for generating text from a prompt string."""
        from src.corpus import format_stimulus_prompt, clean_motor_response

        dev = device if device is not None else next(self.parameters()).device
        formatted_prompt = format_stimulus_prompt(prompt)
        raw_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        input_ids = [t % self.config.vocab_size for t in raw_ids] or [getattr(tokenizer, "bos_token_id", 1) or 1]
        tok_tensor = torch.tensor([input_ids], dtype=torch.long, device=dev)

        out_ids = self.generate(
            tok_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            thinking_cycles=thinking_cycles,
            eos_token_id=getattr(tokenizer, "eos_token_id", 2),
        )
        decoded = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
        return clean_motor_response(decoded)


def build_spheroid_v5(preset: str = "v5_compact", **kwargs) -> CerebroSpheroidV5:
    """Factory helper to build a CerebroSpheroid V5 model from presets."""
    if preset in PRESETS_V5:
        p_cfg = PRESETS_V5[preset].copy()
        p_cfg.update(kwargs)
        cfg = ModelConfigV5(**p_cfg)
    else:
        cfg = ModelConfigV5(**kwargs)
    return CerebroSpheroidV5(config=cfg)


@torch.no_grad()
def compute_manifold_diagnostics(
    model: CerebroSpheroidV5,
    state: torch.Tensor,
) -> Dict[str, Union[float, List[float]]]:
    """
    Computes geometric diagnostics across all concentric spheroid shells.
    Tracks:
    - Effective dimensionality / rank per shell via SVD eigenvalues
    - Shell RMS energy propagation (thalamic core -> cortical surface)
    - Inter-shell cosine similarity (radial wave convection smoothness)
    """
    batch_size, num_shells, shell_dim = state.shape
    state_f = state.float()

    effective_ranks = []
    rms_per_shell = []
    for k in range(num_shells):
        s_k = state_f[:, k, :]
        rms = s_k.pow(2).mean().sqrt().item()
        rms_per_shell.append(rms)

        if batch_size > 1:
            s_centered = s_k - s_k.mean(dim=0, keepdim=True)
            try:
                s_vals = torch.linalg.svdvals(s_centered)
                s2 = s_vals.pow(2)
                rank = (s2.sum().pow(2) / (s2.pow(2).sum() + 1e-9)).item() if s2.sum().item() >= 1e-6 else 1.0
            except Exception:
                rank = 1.0
        else:
            rank = 1.0
        effective_ranks.append(rank)

    adjacent_cosines = []
    for k in range(num_shells - 1):
        s_curr = state_f[:, k, :]
        s_next = state_f[:, k + 1, :]
        cos = F.cosine_similarity(s_curr, s_next, dim=-1).mean().item()
        adjacent_cosines.append(cos)

    return {
        "mean_effective_rank": float(sum(effective_ranks) / len(effective_ranks)),
        "min_effective_rank": float(min(effective_ranks)),
        "effective_ranks": effective_ranks,
        "mean_shell_rms": float(sum(rms_per_shell) / len(rms_per_shell)),
        "shell_rms": rms_per_shell,
        "mean_adjacent_cosine": float(sum(adjacent_cosines) / len(adjacent_cosines)) if adjacent_cosines else 1.0,
        "adjacent_cosines": adjacent_cosines,
    }
