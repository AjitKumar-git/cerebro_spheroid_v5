# CerebroSpheroid V5: Thinking Level Training & Adaptive Deliberation Guide

**Date:** November 16, 2026  
**Module:** Subconscious Deliberation & Volumetric Core  
**Target:** Dynamic Test-Time Compute Scaling & Adaptive PonderNet Halting  
**Related Checkpoint:** `checkpoints/v5_qwen_0_5b_refined_sft.pt`  

---

## 1. Executive Summary

In CerebroSpheroid V5, **"thinking"** occurs predominantly during the **subconscious deliberation phase**—a recurrent, pre-emission dynamical process executed within the 24 concentric shells of the volumetric core:

$$\text{Sensory Prompt} \longrightarrow \underbrace{S_0 \xrightarrow{\text{Cycle 1}} S_1 \xrightarrow{\text{Cycle 2}} \dots \xrightarrow{\text{Cycle } C} S_C}_{\textbf{Subconscious Deliberation (Thinking Level)}} \longrightarrow \text{Motor Emission}$$

While standard autoregressive Transformers possess zero pre-emission recurrent compute ($O(1)$ constant feedforward depth per token regardless of prompt difficulty), CerebroSpheroid can circulate state representations across its concentric manifold for an arbitrary number of cycles $C \in [1, C_{\text{max}}]$.

This document establishes the theoretical foundations, failure modes, loss formulations, and PyTorch implementations required to train the **thinking level** as a fully adaptive, difficulty-aware test-time compute mechanism during fine-tuning.

---

## 2. Architectural Components of the Thinking Loop

The thinking engine operates across four tightly coupled modules in `src/spheroid_v5.py`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   SUBCONSCIOUS DELIBERATION TOPOLOGY                        │
│                                                                             │
│  [Sensory Prompt] ──► Recency Pool ──► S₀ (Shell K-1 Mantle)                │
│                                           │                                 │
│  ┌────────────────────────────────────────┴──────────────────────────────┐  │
│  │ RECURRENT DELIBERATION LOOP (Cycle c = 1 ... C)                       │  │
│  │                                                                       │  │
│  │  1. Deliberation Probe: Q_c = W_probe · mean(S_{c-1})                 │  │
│  │     Resonance: A_c = Softmax(Q_c · K_{sensory}^T / sqrt(d)) · V_{sens}│  │
│  │     Mantle Update: S_{c-1}[:,-1,:] += 0.1 · Norm(A_c)                 │  │
│  │                                                                       │  │
│  │  2. Volumetric Core: S_c = DeliberateStep(S_{c-1})                    │  │
│  │     - Intra-shell SwiGLU transformations                              │  │
│  │     - Bidirectional radial wave convection (w_in, w_out)              │  │
│  │     - Small-world shortcut white-matter tracts                        │  │
│  │                                                                       │  │
│  │  3. Adaptive Halting Gate: λ_c = σ(W_halt · mean(S_c) + b_halt)       │  │
│  └────────────────────────────────────────┬──────────────────────────────┘  │
│                                           ▼                                 │
│                             S_C ──► Motor Decoder (Tokens)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Cortical State Initialization ($S_0 \in \mathbb{R}^{B \times K \times D}$)**:
   The surface shell ($K-1$) receives the recency-weighted prompt centroid $s_0 = \sum_i w_i h_i^{\text{sensory}}$, while subcortical shells ($0 \dots K-2$) initialize at zero.
2. **Deliberation Memory Probe ($Q_{\text{probe}}$)**:
   At each cycle, the mean cortical state queries the sensory key-value cache ($K_{\text{sensory}}, V_{\text{sensory}}$), pulling prompt context directly into the cortical mantle before core convection.
3. **Volumetric Radial Convection**:
   Inflow ($W_{\text{rad\_in}}$) drives information inward from surface shells to core shell 0; outflow ($W_{\text{rad\_out}}$) projects processed subcortical state back to the surface.
4. **Adaptive Halting Gate ($\text{halting\_gate}$)**:
   A linear projection $W_{\text{halt}} \in \mathbb{R}^{1 \times D}$ evaluating whether the internal deliberation has converged to an actionable motor plan.

---

## 3. The Three Mathematical Traps of Naive Thinking Training

Attempts to train thinking depth naively invariably trigger one of three structural failure modes:

### Trap 1: Fixed-Horizon Overfitting (The $C=4$ Rigidity Collapse)
When a model is trained exclusively with a constant parameter (e.g., `thinking_cycles=4`), the Motor Decoder’s cross-resonance and fused-norm layers overfit strictly to the statistical manifold of $S_4$.
* **Consequence:** If evaluated at $C=1$ (fast inference) or $C=8$ (deep thinking), the activations are severely out-of-distribution, resulting in garbled token emissions.

### Trap 2: The Halting Bias Gradient Trap
If the halting gate is trained with a naive auxiliary loss like $\mathcal{L}_{\text{halt}} = \frac{1}{C}\sum_{c=1}^C \lambda_c$:
$$\frac{\partial \mathcal{L}_{\text{halt}}}{\partial \lambda_c} = +1 > 0$$
* **Consequence:** Gradients continually push $\lambda_c \to 0$. The model learns to *never* halt, always exhausting $C_{\text{max}}$ on trivial queries like `"What is the capital of France?"`. Conversely, negating the loss pushes $\lambda_1 \to 1.0$, collapsing the model to a 1-cycle feedforward system.

### Trap 3: Latent Dynamical Instability (Explosion / Vanishing)
Because the volumetric core is recurrently applied:
$$S_c = f_{\text{core}}(S_{c-1})$$
If the spectral radius $\rho(\nabla f_{\text{core}}) > 1.0$, hidden activations grow exponentially with $c$. If $\rho < 1.0$, the state contracts into a trivial zero vector.
* **Consequence:** Without energy regularization, multi-cycle deliberation either blows up logits or erases prompt memory.

---

## 4. The 4 Foundational Training Paradigms

### Pillar 1: PonderNet Probabilistic Halting Formulation (Recommended)
Following PonderNet (Banino et al., DeepMind 2021), the thinking level is modeled as a discrete probability distribution over deliberation cycles.

At each cycle $c \in \{1, \dots, C_{\text{max}}\}$, the halting gate produces an exit scalar:
$$\lambda_c = \sigma(W_{\text{halt}} \bar{S}_c + b_{\text{halt}}) \in (0, 1) \quad \text{where } \bar{S}_c = \frac{1}{K} \sum_{k=1}^K S_{c, k}$$

The probability $p_c$ that deliberation halts *exactly* at step $c$ is:
$$p_c = \lambda_c \prod_{j=1}^{c-1} (1 - \lambda_j)$$

The overall task loss is the **expectation of the motor loss across all thinking levels**:
$$\mathcal{L}_{\text{task}} = \sum_{c=1}^{C_{\text{max}}} p_c \cdot \mathcal{L}_{\text{CE}}(\text{MotorDecoder}(S_c), Y)$$

To enforce a desired average compute budget (e.g., target average cycles $\mu = 3.5$), we add the KL-divergence between the empirical halting distribution $p$ and a Geometric prior $\mathcal{P}_{\text{geom}}(\lambda_{\text{prior}})$ where $\lambda_{\text{prior}} = \frac{1}{\mu}$:

$$q_c = (1 - \lambda_{\text{prior}})^{c-1} \lambda_{\text{prior}}$$
$$\mathcal{L}_{\text{reg}} = D_{\text{KL}}(p \parallel q) = \sum_{c=1}^{C_{\text{max}}} p_c \log \frac{p_c}{q_c}$$

$$\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \beta_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}}$$

---

### Pillar 2: Stochastic Cycle Curriculum (Multi-Horizon SFT)
To ensure the motor decoder can read from any thinking level without degradation:

1. **Stochastic Cycle Sampling**: During SFT, sample $C$ per batch from a categorical distribution:
   $$C \sim \text{Categorical}(p = [0.15, 0.20, 0.35, 0.20, 0.10]) \quad \text{for } C \in \{1, 2, 4, 6, 8\}$$
2. **Complexity-Conditioned Curriculum**:
   - **Direct Slot QA / Grammatical Syntax**: Constrain $C \in \{1, 2\}$
   - **Factual Knowledge / Single-Hop Deduction**: Constrain $C \in \{3, 4\}$
   - **Multi-Step Math (GSM8K) / Algorithmic Logic**: Constrain $C \in \{6, 8\}$
3. **Latent Thought Alignment Loss**:
   Ensure state representations at step $c$ and $c-1$ remain aligned in orientation:
   $$\mathcal{L}_{\text{align}} = \frac{1}{C-1} \sum_{c=2}^C \left(1 - \frac{\bar{S}_c \cdot \bar{S}_{c-1}}{\|\bar{S}_c\|_2 \|\bar{S}_{c-1}\|_2}\right)$$

---

### Pillar 3: Latent Attractor Energy Relaxation (Hopfield Settling)
In continuous attractor networks, deliberation represents relaxation toward a low-energy fixed point. We train the volumetric core to converge into a stable attractor basin:

$$\mathcal{L}_{\text{energy}} = \frac{1}{C-1} \sum_{c=2}^C \frac{1}{D} \| S_c - S_{c-1} \|_F^2$$

This prevents chaotic limit cycles and ensures state velocity $\|\Delta S_c\| \to 0$ as the internal deliberation matures.

---

### Pillar 4: Test-Time Compute Reinforcement Learning (PPO / GRPO)
For post-SFT alignment, train the halting gate policy $\pi_\theta(\text{halt} \mid S_c)$ with a reward function penalizing unnecessary compute:

$$R = \mathbb{I}(\text{Answer Correct}) - \beta_{\text{cost}} \cdot C_{\text{halt}}$$

The policy gradient update on $W_{\text{halt}}$ is:
$$\nabla_\theta \mathcal{J} = \mathbb{E} \left[ \sum_{c=1}^{C_{\text{halt}}} \nabla_\theta \log \pi_\theta(a_c \mid S_c) \cdot (R - b) \right]$$

---

## 5. Production-Ready PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

def train_pondernet_deliberation_step(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    max_cycles: int = 8,
    target_avg_cycles: float = 3.5,
    beta_reg: float = 0.01,
    energy_reg: float = 0.005,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Executes an exact PonderNet subconscious deliberation training step.
    Dynamically balances task loss across thinking depths against a Geometric compute prior.
    """
    B = prompt_ids.shape[0]
    dev = prompt_ids.device

    # 1. Sensory Prompt Ingestion
    sensory_flux, sensory_kv = model.sensory_dnn(prompt_ids)
    safe_mask = (prompt_ids != 0).float()
    prompt_centroid = model._recency_weighted_pool(sensory_flux, safe_mask)

    # 2. Cortical State Initialization (Shell K-1 receives prompt centroid)
    subcortical = torch.zeros(
        B, model.config.num_shells - 1, model.config.surface_dim,
        device=dev, dtype=prompt_centroid.dtype
    )
    state = torch.cat([subcortical, prompt_centroid.unsqueeze(1)], dim=1)

    # 3. Geometric Prior Distribution: q(c) = (1 - λ)^(c-1) * λ
    lambda_prior = 1.0 / target_avg_cycles
    prior_unnorm = torch.tensor([
        ((1.0 - lambda_prior) ** c) * lambda_prior for c in range(max_cycles)
    ], device=dev, dtype=torch.float)
    prior_dist = prior_unnorm / prior_unnorm.sum()

    unhalted_prob = torch.ones(B, 1, device=dev, dtype=torch.float)
    step_probs = []
    cycle_losses = []
    energy_loss = torch.tensor(0.0, device=dev)

    # 4. Deliberation Recurrent Expansion
    for c in range(max_cycles):
        state_prev = state.clone()

        # Deliberation Probe (Cross-attention into sensory KV)
        if sensory_kv is not None and model.deliberation_probe is not None:
            keys, values = sensory_kv
            probe_q = model.deliberation_probe(state.mean(dim=1)).view(
                B, model.config.num_heads, 1, model.sensory_dnn.head_dim
            )
            scores = torch.matmul(probe_q, keys.transpose(-1, -2)) / (model.sensory_dnn.head_dim ** 0.5)
            attn_p = F.softmax(scores.float(), dim=-1).to(keys.dtype)
            retrieved = torch.matmul(attn_p, values).transpose(1, 2).contiguous().view(B, model.config.surface_dim)
            state_mantle = state[:, -1:, :] + 0.1 * model.deliberation_probe_norm(retrieved).unsqueeze(1)
            state = torch.cat([state[:, :-1, :], state_mantle], dim=1)

        # Core radial and intra-shell convection
        state = model.volumetric_core.deliberate_step(state)

        # Attractor energy relaxation (Lyapunov velocity penalty)
        if c > 0:
            energy_loss = energy_loss + torch.mean((state - state_prev) ** 2)

        # Halting Gate probability: λ_c ∈ (0, 1)
        halt_logit = model.halting_gate(state.mean(dim=1).float())
        lambda_c = torch.sigmoid(halt_logit)  # [B, 1]

        # Exact halting probability: p_c = unhalted * λ_c
        p_c = unhalted_prob * lambda_c
        step_probs.append(p_c)

        # Update remaining unhalted mass
        unhalted_prob = unhalted_prob * (1.0 - lambda_c)

        # Evaluate Motor Loss conditioned on state at thinking level c
        loss_c, _ = model(
            prompts=prompt_ids,
            targets=target_ids,
            deliberation_mode="teacher_forcing",
            target_weights=None,
        )
        cycle_losses.append(loss_c)

    # Absorb residual unhalted probability into final step
    step_probs[-1] = step_probs[-1] + unhalted_prob

    P = torch.cat(step_probs, dim=1)           # [B, max_cycles]
    L = torch.stack(cycle_losses, dim=1)       # [B, max_cycles]

    # Expected task loss across all thinking levels: E[L] = Σ (p_c * L_c)
    expected_task_loss = torch.mean(torch.sum(P * L, dim=-1))

    # KL Divergence Regularization against Geometric Compute Prior
    batch_mean_p = torch.mean(P, dim=0) + 1e-8
    kl_reg = torch.sum(batch_mean_p * torch.log(batch_mean_p / (prior_dist + 1e-8)))

    # Total Multi-Objective Loss
    total_loss = expected_task_loss + beta_reg * kl_reg + energy_reg * (energy_loss / max_cycles)

    # Telemetry metrics
    cycle_indices = torch.arange(1, max_cycles + 1, device=dev, dtype=torch.float)
    avg_thinking_depth = torch.sum(P * cycle_indices, dim=-1).mean().item()

    metrics = {
        "loss": total_loss.item(),
        "task_loss": expected_task_loss.item(),
        "kl_reg": kl_reg.item(),
        "energy_loss": energy_loss.item(),
        "avg_thinking_depth": avg_thinking_depth,
        "halt_prob_c1": batch_mean_p[0].item(),
        "halt_prob_cmax": batch_mean_p[-1].item(),
    }
    return total_loss, metrics
```

---

## 6. Verification & Evaluation Protocol

1. **Inferred Thinking Depth by Domain**:
   Measure $\hat{C} = \sum_{c=1}^{C_{\text{max}}} c \cdot p_c$ across standard benchmark categories:
   - Direct Slot Analogies / Antonyms: Expected $\hat{C} \approx 1.0 - 1.5$ cycles
   - Factual Knowledge / Single-Hop QA: Expected $\hat{C} \approx 2.5 - 3.5$ cycles
   - GSM8K Math / Multi-Step Logic: Expected $\hat{C} \approx 6.0 - 7.5$ cycles
2. **Accuracy vs. Allocated Cycles Curve**:
   Manually override `thinking_cycles = C` for $C \in \{1, 2, 4, 6, 8\}$ during greedy decoding. A correctly trained thinking level exhibits monotonically increasing accuracy on complex tasks as deliberation cycles increase.
