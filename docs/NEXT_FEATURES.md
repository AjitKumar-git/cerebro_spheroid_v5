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
  - Strict 3-tier boundary (`<think>`, `<act>`, plain text response) enforced with zero tool loss penalty during pretraining.
  - Test suites verify tool isolation and syntactic boundary emission.
- **Planned Upgrades**:
  1. **Standardized Tool Benchmark Suite**: Implement an automated evaluation benchmark scoring discrete action accuracy across `CALCULATOR`, `FACTUAL_DB`, and `PYTHON` execution on holdout task splits.
  2. **Latent Pondering Energy Metric**: Track the relationship between recurrent thinking cycles ($K=4\dots 8$) and problem complexity across GSM8K and ARC-Challenge tasks.
