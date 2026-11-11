# CerebroSpheroid V5: Architectural & Biological Sensorimotor Specification

## 1. Core Principle: Strategy A (Step 0 Sensorimotor Invariants)

CerebroSpheroid V5 models diverge fundamentally from monolithic decoder-only transformers. Rather than treating prompts and responses as arbitrary slices of a flat autoregressive sequence, V5 is built on an explicit **sensorimotor cognitive architecture**:

```
                         AFFERENT SENSORY FLUX
                     ┌────────────────────────────┐
                     │  <stim> {prompt} </stim>   │
                     └─────────────┬──────────────┘
                                   │
                                   ▼
                    [ Sensory Encoder (32k BPE) ]
                                   │
                                   ▼ Masked Centroid Pooling
                     ┌────────────────────────────┐
                     │ Shell 15: Cerebral Cortex  │
                     └─────────────┬──────────────┘
                                   │
                          Concentric Recurrence
                        (Subconscious Deliberation)
                                   │
                                   ▼
                     ┌────────────────────────────┐
                     │ Shell 0: Subcortical Core  │
                     └─────────────┬──────────────┘
                                   │
                             Efference Copy
                                   │
                                   ▼
                     ┌────────────────────────────┐
                     │   <act> {response} </act>  │
                     └────────────────────────────┘
                         EFFERENT MOTOR ACTIONS
```

### Invariants Enforced Under Strategy A:
1. **Sensory Afference (`<stim> ... </stim>`)**:
   - Every input prompt, instruction, educational query, or contextual situation is strictly enclosed within `<stim>` and `</stim>`.
   - Seeded exclusively into the outer concentric cortical shell (Shell 15) via masked centroid pooling.
2. **Motor Efference (`<act> ... </act>`)**:
   - Every response, derivation, chain-of-thought, or agentic tool invocation is strictly enclosed within `<act>` and `</act>`.
   - Motor emission terminates when `</act>` (or EOS) is discharged.
   - Efference copies of active motor emissions are projected back into Shell 0 (Subcortical Core) to maintain causal feedback.
3. **Step 0 Grounding**:
   - Both online streaming (`dense_edu`, `ElasticBucketBatcher`) and offline prefetching (`LinguisticDataStreamer`) apply biological delimiters from the very first gradient step (Step 0).
   - This eliminates format shift shock when transitioning from pretraining to instruction/tool tuning.

---

## 2. Vocabulary & Tokenizer Specification

| Property | Value | Architectural Constraint |
| :--- | :--- | :--- |
| **Tokenizer Architecture** | Byte-Fallback BPE | Strictly 32,000 Vocabulary |
| **Tokenizer Source** | `data/tokenizer/` | Fully bundled, 100% offline |
| **Special Tokens** | PAD=0, BOS=1, EOS=2 | Zero `<unk>` due to byte-level fallback |
| **`<stim>` Encoding** | Subword sequence | `[529, 303, 326, 29958]` (`<`, `st`, `im`, `>`) |
| **`</stim>` Encoding** | Subword sequence | `[1533, 303, 326, 29958]` (`</`, `st`, `im`, `>`) |
| **`<act>` Encoding** | Subword sequence | `[529, 627, 29958]` (`<`, `act`, `>`) |
| **`</act>` Encoding** | Subword sequence | `[1533, 627, 29958]` (`</`, `act`, `>`) |

### Strict Prohibition of Legacy Artifacts:
* **Zero GPT-2 Artifacts**: The 50,257-vocabulary GPT-2 tokenizer, legacy token IDs (`20436`, `1044`), and GPT-2 configurations are completely deprecated and barred from V5. Calling `build_tokenizer("gpt2")` raises a `ValueError`.
* **Zero Hardcoded Paths**: No `/home/user/` or absolute machine-specific cache paths are permitted in active source or scripts. All tokenizer and data paths resolve relative to the repository root (`data/tokenizer/`).

---

## 3. High-Throughput Training Invariants

1. **Native TPU BF16 Execution**:
   - Pure JAX systolic MXU acceleration on Cloud TPU v5e-1 (`tpu_v5e_1` / `v5_small`, 101.6M parameters).
   - Delivers 60,000 – 85,000 tokens/sec sustained throughput at batch size 128 and sequence length 256.
2. **SpheroidOptimizer**:
   - Riemannian quintic Newton-Schulz NS-5 polar orthogonalization for all 2D and 3D synaptic weight tensors.
   - Recurrent contractive spectral clamp ($\rho(W) \le 1.0$) prevents gradient explosion.
   - Decoupled AdamW for sensory embeddings and motor projection heads.
3. **Causal Teacher-Forced Feedback**:
   - Internal auditory feedback is causally shifted (`jnp.roll(targets, shift=1)`) so step $t$ conditions on token $t-1$.
4. **Resumption & Scratch Invariants**:
   - `from_scratch=True` strictly bypasses all cloud sync downloads and staging directories, initializing clean Step 0 weights.
   - `cloud_sync=True` mirrors `{model}_latest.pt` and `{model}_best.pt` to `bioflo.in` asynchronously with zero TPU stalls.
