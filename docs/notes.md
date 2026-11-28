# Notes

## Types of Reasoning in Artificial Intelligence
*Source: GeeksforGeeks (Last Updated: 27 May, 2026) | Added: 2026-11-25*

In AI systems, **reasoning** is the process of using available information and logical methods to make decisions, solve problems, and generate conclusions. Different types of reasoning enable AI systems to learn from data, handle uncertainty, and respond intelligently in dynamic real-world situations.

### 1. Deductive Reasoning
- **Definition**: Starts with general principles and applies them to specific cases to arrive at certain conclusions. If the premises are true, the conclusion must also be true. ([Reference](https://www.geeksforgeeks.org/artificial-intelligence/deductive-reasoning-in-ai/))
- **Example**: If all humans are mortal and Socrates is a human, then Socrates is mortal.
- **Application**: Used in expert systems where predefined rules help derive accurate decisions, such as identifying diseases from known symptom patterns.

### 2. Inductive Reasoning
- **Definition**: Works by drawing general conclusions from specific observations. The conclusions reached are not certain but are based on probability and patterns observed in the data. ([Reference](https://www.geeksforgeeks.org/artificial-intelligence/inductive-reasoning-in-ai/))
- **Example**: If we observe that the sun rises in the east every day, we might infer that it will rise in the east tomorrow.
- **Application**: Widely used in machine learning systems to identify patterns and make predictions, such as movie recommendation systems.

### 3. Abductive Reasoning
- **Definition**: Starts with incomplete observations and seeks the most likely explanation. It is about making educated guesses based on available data, even if not all facts are known. ([Reference](https://www.geeksforgeeks.org/artificial-intelligence/abductive-reasoning-in-ai/))
- **Example**: If a patient has a fever and cough, a doctor might hypothesize that they have the flu even though other illnesses could cause similar symptoms.
- **Application**: Used in diagnostic AI systems to predict probable causes of problems or diseases from available symptoms or data.

### 4. Analogical Reasoning
- **Definition**: Involves comparing two situations that are similar and using knowledge from one to solve problems in another. It helps AI systems solve problems in new domains by applying solutions from related areas.
- **Example**: If flying a helicopter is similar to flying a drone, knowledge from piloting a helicopter can be transferred to flying a drone.
- **Application**: Used in robotics and cognitive systems where AI transfers solutions from previously solved tasks to similar new tasks.

### 5. Common Sense Reasoning
- **Definition**: Allows AI systems to make decisions using everyday knowledge and practical understanding of the world. It involves making judgments about the world that are obvious to humans but difficult for machines to understand.
- **Example**: If it rains, we can expect the ground to get wet even without explicitly stating it.
- **Application**: Used in conversational AI and autonomous systems to interpret everyday situations and respond naturally.

### 6. Monotonic Reasoning
- **Definition**: A form of reasoning where conclusions, once drawn, cannot be reversed even if new information becomes available. This ensures that conclusions remain consistent regardless of updates to the knowledge base.
- **Example**: The statement *"The Sahara is a desert"* remains true even if more information about the world's deserts is introduced.
- **Application**: Used in formal verification and rule-based systems where consistent conclusions are required.

### 7. Nonmonotonic Reasoning
- **Definition**: In contrast to monotonic reasoning, allows AI systems to revise their conclusions based on new information. Important for decision-making in dynamic and unpredictable environments. ([Reference](https://www.geeksforgeeks.org/artificial-intelligence/monotonic-reasoning-vs-non-monotonic-reasoning/))
- **Example**: Initially, we might think that all birds can fly, but we revise this conclusion after learning about penguins, which cannot fly.
- **Application**: Used in adaptive AI systems like autonomous vehicles and traffic management systems that continuously update decisions using new data.

### 8. Fuzzy Reasoning
- **Definition**: Deals with uncertainty by allowing for degrees of truth rather than binary true/false values. Useful when data is vague or incomplete. ([Reference](https://www.geeksforgeeks.org/artificial-intelligence/fuzzy-logic-introduction/))
- **Example**: The statement *"It's warm outside"* is vague. Fuzzy reasoning might assign a value like `0.7` to represent how warm it is.
- **Application**: Used in smart appliances and autonomous systems where decisions must be made under uncertain or imprecise conditions.

---

## Information Extraction from a Single Forward Pass (and Classical TPU vs. Quantum)
*Added: 2026-11-27*

### 1. Hard Information-Theoretic Ceiling (Target Bits)
- From ground-truth labels alone, 1 forward pass over $B \cdot T$ tokens supplies at most the conditional entropy of the targets:
  $$I_{\text{targets}} \approx B \cdot T \cdot \frac{\mathcal{L}_{\text{CE}}}{\ln 2} \text{ bits}$$
- At $\mathcal{L}_{\text{CE}} \approx 3.0\text{ nats}$ ($\approx 4.33\text{ bits/token}$), a batch of $16,384$ tokens contains only $\sim 71\text{ kbits}$ ($\sim 8.9\text{ KB}$) of irreducible prediction error.
- Standard Cross-Entropy compresses the entire forward pass down to **1 scalar per token**, discarding rich structural and geometric information already computed in memory.

### 2. Extra Optimization Signal Extractable from 1 Forward Pass (TPU / JAX Native)
Without adding a second forward pass, the activations ($T \times K \times D$) and logits ($T \times V$) already in memory provide:
- **Logit Scale & Selective Weighting** *(active in V5)*: `z_loss_weight * lse^2`, `selective_focal_gamma`, and `selective_noise_cap` in `src/jax/v5_model.py` (`_compute_token_ce_and_z_loss`).
- **Matrix Spectral Preconditioning & Covariance** *(active in V5)*: `power_v` and `scmer_loss` (`neg_cov_loss`) in `src/jax/spheroid_optimizer.py`.
- **Token Entropy $H(p_t)$ vs. $\text{CE}_t$**: Distinguishes *confidently wrong* tokens (high $\text{CE}_t$, low $H(p_t)$) from *genuinely ambiguous* tokens (high $\text{CE}_t$, high $H(p_t)$) with 0 new parameters (~1–2% FLOPs).
- **Latent Predictive Coding ($h_t \to h_{t+1}$)**: Supervising $h_t$ against `lax.stop_gradient(h_{t+1})` along the sequence already in memory provides dense $D$-dimensional continuous targets (~1–2% FLOPs).
- **Per-Shell Deep Supervision**: Distilling inner/motor logits (with `stop_gradient`) into outer shell states `state[:, k, :]` gives outer shells direct gradient signal without altering `use_shell_tap` on the live motor readout (~2–5% FLOPs).
- **Gauss-Newton / K-FAC Curvature**: Combining forward activation covariance $a_\ell a_\ell^\top$ with backward gradient covariance $g_\ell g_\ell^\top$ yields second-order preconditioning with zero extra forward passes (~3–5% FLOPs).

### 3. Limits of 1 Forward Pass
- **Multi-step exposure bias / rollout error**: Requires autoregressive rollout or `dual_pass` ($>1$ forward pass).
- **Exact non-diagonal Hessian directions**: Requires JVP / second-backward passes.

### 4. Quantum Computing vs. Current TPU Architecture
- **No quantum computer is needed**, and quantum hardware would not help here: all of the signals above are dense classical linear-algebra operations (`bfloat16`/`float32` matmuls and reductions) that TPUs are built for.
- Loading $16\text{k}$ tokens + $85.4\text{M}$ parameters into quantum states and reading out $85.4\text{M}$ classical gradient updates every step (QRAM/readout bottleneck) would negate any theoretical quantum speedup.
- Everything above can be emulated directly inside Cerebro Spheroid V5's existing 1 forward + 1 backward pass in JAX on TPU.

---

## Refined GRPO & Multi-Tier Exploration Reward Engine (V5 Small)
*Added: 2026-11-28*

### 1. Architectural Core
- **Policy Checkpoint**: Anchored to aligned base checkpoint `checkpoints/tool_aligned/v5_small_tool_aligned_step_500.pt`.
- **Dataset Integration**: Ingests 8,743 verified agentic records from `data/consolidated/02_agentic_tools/agentic_tools_sensory_grounded.jsonl` covering tool calling, sensory KV needle grounding, grounded answers, and pulsating waiting loops.
- **Rollout Sampling**: Batched group generation ($G=4$) with temperature sampling ($T=0.85$, $p=0.95$).

### 2. Multi-Tier Reward Model (`CerebroExplorationRewardModel`)
- **Verifiable Tool Dispatching** (`ToolExecutionEvaluator`): Deterministic execution of parsed tool calls (`calc`, `factual`, `python`) via `UnifiedToolDispatcher`. Awards $+1.0$ for exact ground-truth matches, $+0.4$ for clean execution, and $-0.4$ for unparseable syntax crashes.
- **Cognitive Boundary Compliance** (`ProtocolAndFormattingEvaluator`): Evaluates balanced 3-tier tags (`<think>`, `<tool_call>`, `<tool_response>`), temporal prefixes ("calling the calculator..."), waiting/pulsating loops ("waiting for tool response..."), and sensory-KV grounded phrasing ("by calling the calculator I got..."). Penalizes legacy/hallucinated tag dialects (`<tool_use>`, `<call:`, `<observation>`).
- **Process Verification** (`ProcessVerificationEvaluator`): Evaluates hypothesis generation, self-verification, and self-correction.
- **Curiosity & Manifold Exploration**: Random Network Distillation (RND) novelty with predictor familiarity decay, plus active concentric SVD rank, radial convection velocity, and shell energy entropy.
- **Trajectory Diversity**: Leave-one-out 3-gram Jaccard distance and representation cosine dispersion.

### 3. Clipped Surrogate & Unbiased KL
- Unbiased token-level KL divergence against frozen reference model $\pi_{\text{ref}}$:
  $$\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})_{i, t} = \exp(-\Delta_{i, t}) + \Delta_{i, t} - 1 \ge 0$$
- Clipped surrogate policy loss with group advantage standardization: $A_i = (R_i - \bar{R}) / (\sigma_R + \epsilon)$.
- Memory-efficient AMP BF16 training on local RTX 3050 (< 1.2 GB VRAM, ~0.5s/step).
- Bare BF16 serialization purging all optimizer momentum fluff from checkpoints.

