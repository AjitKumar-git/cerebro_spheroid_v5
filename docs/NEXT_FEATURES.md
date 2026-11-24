# CerebroSpheroid V5: Next Features & Engineering Backlog

Planned improvements tracked for future development cycles and scaling phases.

---

## 1. Sequence Packing & Multi-Window Extraction

- **Current State**:
  - Training supports fixed sequence lengths (e.g. 512, 1024, 2048, 4096) with homogeneous static shapes for JAX `pmap` / TPU XLA compilation.
  - Documents shorter than the sequence length are padded, while long articles are sampled in slices.
- **Planned Upgrades**:
  1. **Multi-Document Sequence Packing**: Concatenate multiple short documents with attention masking boundaries into a single uniform static shape, driving token loss utilization to ~99.8%.
  2. **Sequential Multi-Window Extraction**: For long documents exceeding 4,096 tokens, stream sequential overlapping windows with recurrent state handoff rather than discarding downstream tokens.

---

## 2. Multi-TPU Distributed Scaling & Pod Coordination

- **Current State**:
  - Single-slice TPU execution (TPU v5e-8, TPU v4-8, TPU v3-8) fully operational via JAX `pmap` Local-SGD.
  - Zero Google Drive dependency; checkpoints saved locally to `/content/checkpoints/` and synced asynchronously to `bioflo.in`.
- **Planned Upgrades**:
  1. **Multi-Node TPU Pod Coordination (`shard_map` / Mesh)**: Transition from Local-SGD `pmap` to JAX `shard_map` across multi-host TPU v4-16 and v5e-16/32 pods.
  2. **Automated Ring All-Gather**: Integrate NCCL/ICI collective primitives for multi-node gradient reduction without host CPU bottlenecks.

---

## 3. Post-Pretraining Cognitive & Tool Dispatch Evaluation

- **Current State**:
  - Strict 3-tier boundary (`<think>`, `<tool_call>`, plain text response) enforced with zero tool loss penalty during pretraining.
  - Test suites verify tool isolation and syntactic boundary emission.
- **Planned Upgrades**:
  1. **Standardized Tool Benchmark Suite**: Implement an automated evaluation benchmark scoring discrete action accuracy across `CALCULATOR`, `FACTUAL_DB`, and `PYTHON` execution on holdout task splits.
  2. **Latent Pondering Energy Metric**: Track the relationship between recurrent thinking cycles ($K=4\dots 8$) and problem complexity across GSM8K and ARC-Challenge tasks.

---

## 4. Token-Free Dynamic Byte Chunking (BLT & H-Net Paradigms for Cerebro V6)

- **Current State (V5)**:
  - 32,000-vocabulary LLaMA BPE tokenizer with tied embedding matrix ($32k \times 768 = 24.6\text{M}$ parameters), consuming $23.2\%$ of `v5_small`'s 106M parameter budget.
  - Active training run streaming past 11.73B tokens on TPU v5e with static BPE sequence shapes.
- **Planned Upgrades (V6)**:
  1. **H-Net Dynamic Chunking (DC) Layer**: Implement an end-to-end learned segmentation boundary gate (Hwang, Wang & Gu, 2025) directly within the outer sensory cortical mantle (Shell 31), replacing static BPE tokenization with $3\times\text{--}6\times$ dynamic byte compression without needing external entropy models.
  2. **Parameter Reallocation to Cortical Shells**: Recover $\sim 24.4\text{M}$ parameters from the 32k vocabulary embedding matrix into deep concentric laminae (expanding shell count or widening radial dimensions).
  3. **TPU Static Patch Masking**: Implement fixed-capacity patch padding ($P_{\text{max}} = 16$) via `jax.lax.associative_scan` and cumulative sum masks to retain native TPU v5e XLA graph caching and avoid compilation jitter.
  4. **Latent Refractory Tracking**: Migrate the inhibition-of-return mechanism from discrete token ID penalties to latent patch representation distances.

