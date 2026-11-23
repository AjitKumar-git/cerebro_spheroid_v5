# CerebroSpheroid V5: Architectural & Biological Sensorimotor Specification

## 1. Core Principle: Sensorimotor Cognitive Architecture

CerebroSpheroid V5 models diverge fundamentally from monolithic decoder-only transformers. Rather than treating prompts and responses as arbitrary slices of a flat autoregressive sequence, V5 is built on an explicit **sensorimotor cognitive architecture**:

```
                         AFFERENT SENSORY FLUX
                     ┌────────────────────────────┐
                     │     Prompt / Input Text    │
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
                     │ <think> ... </think>       │
                     │ <tool_call> ...            │
                     │ Grounded Motor Answer      │
                     └────────────────────────────┘
                         EFFERENT MOTOR ACTIONS
```

### Sensorimotor Principles:
1. **Sensory Afference**:
   - Input prompts, instructions, educational queries, or contextual problems are ingested directly as clean natural text.
   - Seeded into the outer concentric cortical shell (Shell 15) via masked centroid pooling and sensory projection.
2. **Motor Efference & Cognitive Boundaries**:
   - Motor emissions proceed through explicit cognitive boundaries: `<think> ... </think>` for internal latent deliberation, `<tool_call> ... </tool_call>` for tool dispatch, and grounded natural language answers.
   - Motor emission terminates when `</s>` (EOS) is discharged.
   - Efference copies of active motor emissions are projected back into Shell 0 (Subcortical Core) to maintain causal feedback.
3. **Standard Cognitive Delimiters**:
   - Training operates cleanly on native standard delimiters (`<s>`, `</s>`, `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`).
   - All input queries, educational prompts, and instructions use natural text without wrapper tags.

---

## 2. Vocabulary & Tokenizer Specification

| Property | Value | Architectural Constraint |
| :--- | :--- | :--- |
| **Tokenizer Architecture** | Byte-Fallback BPE | Strictly 32,000 Vocabulary |
| **Tokenizer Source** | `data/tokenizer/` | Fully bundled, 100% offline |
| **Special Tokens** | PAD=0, BOS=1 (`<s>`), EOS=2 (`</s>`) | Zero `<unk>` due to byte-level fallback |
| **`<|im_start|>` / `<|im_end|>`** | ChatML Delimiter Tokens | ID 27 / ID 28 |
| **`<think>` / `</think>`** | Dedicated Cognitive Tokens | ID 29 / ID 30 |
| **`<tool_call>` / `</tool_call>`** | Dedicated Motor Action Tokens | ID 31 / ID 32 |
| **`<tool_response>` / `</tool_response>`** | Dedicated Sensory Feedback Tokens | ID 33 / ID 34 |

### Strict Prohibition of Legacy Artifacts:
* **Standard Input Representation**: All prompts and inputs use clean natural text without artificial wrapper envelopes.
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
