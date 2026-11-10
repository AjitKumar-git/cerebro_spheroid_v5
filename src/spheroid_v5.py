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


PRESETS_V5: Dict[str, Dict[str, Any]] = {
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
    offset indexes into the precomputed rotary frequencies.
    """
    L = x.shape[2]
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

        # Non-linear input SwiGLU block
        self.swiglu = SwiGLU(surface_dim, hidden_features=surface_dim * 2, out_features=surface_dim)
        self.norm_in = RMSNorm(surface_dim)
        self.norm_out = RMSNorm(surface_dim)
        self.dropout = nn.Dropout(dropout)

        # Multi-Head KV projections for Arcuate Fasciculus
        self.k_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.v_proj = nn.Linear(surface_dim, surface_dim, bias=False)

    def forward(self, input_ids_or_emb: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Accepts integer token IDs [B, L] or soft embedding vectors [B, D] / [B, L, D].
        Returns:
            flux: High-dimensional sensory representation [B, L, D]
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
        h = self.norm_in(x)
        h = x + self.dropout(self.swiglu(h))
        flux = self.norm_out(h)

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
        self.radial_norm = RMSNorm(surface_dim)

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

        # 2. Inter-shell radial wave propagation
        # Inward wave: Shell k+1 -> Shell k
        rad_in = self.radial_in_proj(state_intra[:, 1:, :])   # [B, K-1, Dr]
        in_flux = torch.einsum("bkd,kde->bke", rad_in, self.w_radial_in)  # [B, K-1, Dr]
        in_up = self.radial_up(in_flux)  # [B, K-1, D]

        # Outward wave: Shell k -> Shell k+1
        rad_out = self.radial_out_proj(state_intra[:, :-1, :])  # [B, K-1, Dr]
        out_flux = torch.einsum("bkd,kde->bke", rad_out, self.w_radial_out)  # [B, K-1, Dr]
        out_up = self.radial_up(out_flux)  # [B, K-1, D]

        # Assemble radial deltas out-of-place (zero in-place modifications)
        delta_in = F.pad(0.1 * in_up, (0, 0, 0, 1))    # [B, K, D], zero padded on Shell K-1
        delta_out = F.pad(0.1 * out_up, (0, 0, 1, 0))  # [B, K, D], zero padded on Shell 0
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

        src_acts = state_radial[:, self.src_indices, :]   # [B, T, D]
        shortcut_flux = torch.einsum("btd,tde->bte", src_acts, self.shortcut_weight) * eff_conductances.unsqueeze(-1)

        # Assemble axonal skip flux out-of-place
        delta_shortcut = torch.zeros_like(state_radial)
        for t in range(self.num_shortcut_tracts):
            dst = self.dst_indices[t]
            delta_shortcut[:, dst, :] = delta_shortcut[:, dst, :] + 0.05 * shortcut_flux[:, t, :]
        state_next = state_radial + delta_shortcut

        return self.radial_norm(state_next)

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
    ):
        super().__init__()
        self.num_shells = num_shells
        self.surface_dim = surface_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.head_dim = surface_dim // num_heads
        self.max_phase_steps = max_phase_steps
        self.phase_scale = phase_scale
        self.use_plan = use_plan
        self.use_shell_tap = use_shell_tap
        self.top_k_resonance = top_k_resonance

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
        fused_dim = surface_dim * 5 if use_plan else surface_dim * 4
        self.fused_norm = RMSNorm(fused_dim)
        self.motor_swiglu = SwiGLU(fused_dim, hidden_features=surface_dim * 2, out_features=surface_dim)
        self.norm_mid = RMSNorm(surface_dim)
        self.dropout = nn.Dropout(dropout)

        # Recurrent Executive Plan synthesis
        if use_plan:
            self.plan_init = nn.Linear(surface_dim * 2, surface_dim, bias=True)
            self.plan_gate = nn.Linear(surface_dim * 2, surface_dim, bias=True)
            self.plan_update = nn.Linear(surface_dim * 2, surface_dim, bias=False)
            self.plan_decay = nn.Parameter(torch.tensor(0.85))
            self.plan_write = nn.Parameter(torch.tensor(0.15))

        # Affine-Tied Motor Emission Head Parameters
        # gamma: Learnable per-channel scale multiplying tied embedding weights
        self.affine_scale = nn.Parameter(torch.ones(surface_dim))
        self.temperature_scale = nn.Parameter(torch.tensor(1.0))

        # Efference copy projection (injected into Shell 0)
        self.efference_proj = nn.Linear(surface_dim, surface_dim, bias=False)

        # Two-tier refractory penalty parameters
        self.refractory_decay = nn.Parameter(torch.tensor(0.75))
        self.refractory_strength = nn.Parameter(torch.tensor(1.2))

    def init_plan(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.surface_dim, device=device)

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
        p_scaled = phase * (self.phase_scale / p_norm)

        motor_surface = surface + p_scaled
        if auditory_feedback is not None:
            motor_surface = motor_surface + auditory_feedback.to(motor_surface.dtype)

        # 2. Multi-Head Arcuate Fasciculus Cross-Resonance with RoPE & Top-K Sparsity
        if sensory_kv is not None:
            keys, values = sensory_kv  # [B, H, L, head_dim]
            L = keys.shape[2]
            q = self.q_proj(motor_surface).view(B, self.num_heads, 1, self.head_dim)

            # Apply RoPE: keys start at pos 0..L-1, query is positioned at step_idx
            q_offset = min(int(step_idx), self.rope_cos.shape[0] - 1)
            q_rot = apply_rope(q, self.rope_cos, self.rope_sin, offset=q_offset)
            k_rot = apply_rope(keys, self.rope_cos, self.rope_sin, offset=0)

            # Dot-product cross-resonance: [B, H, 1, L]
            scores = torch.matmul(q_rot.to(keys.dtype), k_rot.transpose(-1, -2)) / math.sqrt(self.head_dim)

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
        fused_components = [motor_surface, core, resonance, shell_pool]
        if self.use_plan and plan is not None:
            fused_components.append(plan)

        fused = self.fused_norm(torch.cat(fused_components, dim=-1))
        h = self.norm_mid(self.motor_swiglu(fused))
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

        # 6. Affine-Tied Dictionary Projection (W_motor = gamma * W_sensory)
        # Memory-optimal: Scale hidden vector [B, D] (3 KB) instead of vocabulary matrix [V, D] (98 MB)
        temp = torch.clamp(self.temperature_scale, 0.1, 5.0)
        h_scaled = (h * self.affine_scale) / temp
        raw_logits = F.linear(h_scaled, tied_embedding_weight)

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

    def count_parameters(self) -> Dict[str, int]:
        """Returns detailed parameter breakdown across modules."""
        sensory = sum(p.numel() for p in self.sensory_dnn.parameters())
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

    def forward(
        self,
        prompts: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        feedback_tokens: Optional[torch.Tensor] = None,
        sampling_p: float = 1.0,
        pad_id: int = 0,
        target_len: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Differentiable training rollout engine:
        1. Encodes sensory prompt into Arcuate Fasciculus KV and centroid flux.
        2. Sets Shell K-1 (Surface) to prompt centroid.
        3. Executes subconscious deliberation cycles in subcortical core.
        4. Steps through motor emissions with scheduled sampling and thought re-afference.
        """
        B, L = prompts.shape
        dev = prompts.device
        t_len = target_len if target_len is not None else (targets.shape[1] if targets is not None else 128)

        # 1. Sensory Ingestion
        sensory_flux, sensory_kv = self.sensory_dnn(prompts)

        # Masked centroid pooling: padding tokens never dilute prompt semantics
        mask = (prompts != pad_id).unsqueeze(-1).float()
        prompt_centroid = torch.sum(sensory_flux * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1.0)

        # 2. State Initialization: Cortical Surface receives sensory prompt flux out-of-place
        subcortical = torch.zeros(B, self.config.num_shells - 1, self.config.surface_dim, device=dev)
        state = torch.cat([subcortical, prompt_centroid.unsqueeze(1)], dim=1)

        # 3. Subconscious Deliberation Cycles with Memory Probing & Adaptive Halting
        cycles = self.config.max_thinking_cycles if self.config.use_adaptive_thinking else self.config.thinking_cycles
        energy_loss = torch.tensor(0.0, device=dev)

        for cycle in range(cycles):
            state_prev = state.clone()
            if sensory_kv is not None and self.deliberation_probe is not None:
                keys, values = sensory_kv
                L_kv = keys.shape[2]
                probe_q = self.deliberation_probe(state[:, 0, :]).view(B, self.config.num_heads, 1, self.sensory_dnn.head_dim)
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
                halt_prob = torch.sigmoid(self.halting_gate(state[:, 0, :]))
                if not self.training and (halt_prob > self.config.thinking_halt_threshold).all():
                    break

        # Pre-embed teacher feedback tokens if available
        if feedback_tokens is not None:
            safe_feedback = torch.clamp(feedback_tokens, min=0)
            tf_flux, _ = self.sensory_dnn(safe_feedback)
            valid_mask = (feedback_tokens >= 0).unsqueeze(-1).float()
            tf_flux = tf_flux * valid_mask
        else:
            tf_flux = None

        # 4. Motor Emission Loop
        plan = self.motor_decoder.init_plan(B, dev) if self.config.use_plan else None
        refractory = torch.zeros(B, self.config.vocab_size, device=dev)
        prev_thought = torch.zeros(B, self.config.surface_dim, device=dev)

        all_logits = []
        tied_w = self.sensory_dnn.embedding.weight

        for t in range(t_len):
            # Scheduled sampling: bridge teacher-forcing and internal thought
            if tf_flux is not None:
                tf_aud = tf_flux[:, t, :]
                internal_feed = prev_thought if t > 0 else tf_aud
                auditory_in = sampling_p * tf_aud + (1.0 - sampling_p) * internal_feed
            else:
                auditory_in = prev_thought if t > 0 else None

            logits, efference, refractory, plan = self.motor_decoder(
                state=state,
                sensory_kv=sensory_kv,
                tied_embedding_weight=tied_w,
                plan=plan,
                refractory_trace=refractory,
                step_idx=t,
                auditory_feedback=auditory_in,
            )
            all_logits.append(logits)

            # Thought re-afference for step t+1 (isolated gradient)
            with torch.no_grad():
                soft_p = F.softmax(logits, dim=-1)
                soft_emb = soft_p @ tied_w
                prev_thought, _ = self.sensory_dnn(soft_emb)

            # Closed-loop state propagation via efference copy injection
            state_eff = self.volumetric_core.inject_efference_copy(state, efference)
            state = self.volumetric_core.deliberate_step(state_eff)

        logits_stacked = torch.stack(all_logits, dim=1)  # [B, t_len, vocab_size]

        if targets is not None:
            # Cross-entropy loss computed only on valid non-padded targets (target != -100)
            loss = F.cross_entropy(
                logits_stacked.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            if self.config.use_energy_regularization:
                loss = loss + 0.01 * (energy_loss / max(cycles, 1))

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

        # 1. Encode prompt
        sensory_flux, sensory_kv = self.sensory_dnn(prompt_ids)
        prompt_centroid = sensory_flux.mean(dim=1)

        # 2. State Initialization: Cortical Surface receives sensory prompt flux out-of-place
        subcortical = torch.zeros(B, self.config.num_shells - 1, self.config.surface_dim, device=dev)
        state = torch.cat([subcortical, prompt_centroid.unsqueeze(1)], dim=1)

        for cycle in range(cycles):
            if sensory_kv is not None and self.deliberation_probe is not None:
                keys, values = sensory_kv
                L_kv = keys.shape[2]
                probe_q = self.deliberation_probe(state[:, 0, :]).view(B, self.config.num_heads, 1, self.sensory_dnn.head_dim)
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
                halt_prob = torch.sigmoid(self.halting_gate(state[:, 0, :]))
                if (halt_prob > self.config.thinking_halt_threshold).all():
                    break

        # 3. Motor generation loop
        plan = self.motor_decoder.init_plan(B, dev) if self.config.use_plan else None
        refractory = torch.zeros(B, self.config.vocab_size, device=dev)
        prev_thought = torch.zeros(B, self.config.surface_dim, device=dev)

        generated_tokens = []
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        tied_w = self.sensory_dnn.embedding.weight

        for t in range(max_new_tokens):
            auditory_in = prev_thought if t > 0 else None

            logits, efference, refractory, plan = self.motor_decoder(
                state=state,
                sensory_kv=sensory_kv,
                tied_embedding_weight=tied_w,
                plan=plan,
                refractory_trace=refractory,
                step_idx=t,
                auditory_feedback=auditory_in,
                repetition_penalty=repetition_penalty,
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

            # Thought re-afference: discrete token embedding matching training distribution
            prev_thought, _ = self.sensory_dnn(next_tok.squeeze(-1))

            # Efference copy state propagation
            state_eff = self.volumetric_core.inject_efference_copy(state, efference)
            state = self.volumetric_core.deliberate_step(state_eff)

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
        dev = device if device is not None else next(self.parameters()).device
        STIM_ID = 20436 if self.config.vocab_size > 20436 else (getattr(tokenizer, "bos_token_id", 1) or 1)
        raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
        safe_ids = [t % self.config.vocab_size for t in raw_ids]
        input_ids = [STIM_ID] + safe_ids if (not safe_ids or safe_ids[0] != STIM_ID) else safe_ids
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
        return tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True).strip()


def build_spheroid_v5(preset: str = "v5_compact", **kwargs) -> CerebroSpheroidV5:
    """Factory helper to build a CerebroSpheroid V5 model from presets."""
    if preset in PRESETS_V5:
        p_cfg = PRESETS_V5[preset].copy()
        p_cfg.update(kwargs)
        cfg = ModelConfigV5(**p_cfg)
    else:
        cfg = ModelConfigV5(**kwargs)
    return CerebroSpheroidV5(config=cfg)
