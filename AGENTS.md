# Repository Operational Directives & Agent Standards

This document establishes the mandatory operational rules, architectural invariants, and workflow constraints for any AI agent or automated developer working within the `cerebro_spheroid_v5` codebase.

---

## 1. Absolute Directives & Strict Prohibitions

1. **NEVER CREATE ANY `README.md` FILE**:
   - **STRICT PROHIBITION**: Do NOT create `README.md` anywhere in `cerebro_spheroid_v5`, `cerebro_spheroid_v6`, or subdirectories.
   - Architectural and project documentation belongs in `docs/` (e.g., `docs/RESEARCH_NOTES.md`, `docs/V5_SYSTEM_SPECIFICATION.md`) or this `AGENTS.md`.

2. **ZERO GOOGLE DRIVE DEPENDENCY**:
   - Do NOT import `google.colab.drive` or invoke `drive.mount()`.
   - Do NOT hardcode `/content/drive/MyDrive/...` paths in scripts, notebooks, or configs.
   - All checkpoints default to local high-speed scratch storage (`/content/checkpoints/` or `./checkpoints/`).
   - Remote checkpoint backup and telemetry synchronization are handled asynchronously via `deploy/bioflo/sync.php`.

3. **NO V4 CODE OR LEGACY JUNK**:
   - CerebroSpheroid V5 is a native architecture trained from scratch.
   - Never import, adapt, or reintroduce obsolete V4 weight-transfer scripts, archived notebooks, or V4 migration helpers.

4. **NEVER RUN `git push` WITHOUT EXPLICIT PERMISSION**:
   - Never push changes to remote git branches unless the user explicitly requests a push.

---

## 2. Pretraining Curriculum Invariants (SmolLM2 Parity)

Autonomous pretraining (`src/elastic_window.py`, `scripts/run_autonomous_training.py`, `training/train_v5_tpu_jax.py`) strictly implements the **SmolLM2 4-stage developmental mixture** (arXiv:2502.02737):

| Stage | Progress | English Web (`FineWeb`) | Code (`The Stack-Edu`) | Math (`FineMath` / `OWM`) | Synthetic (`Cosmopedia v2`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | $0\% - 55\%$ | **90%** | **10%** | 0% | 0% |
| **Stage 2** | $55\% - 73\%$ | **75%** | **20%** | **5%** | 0% |
| **Stage 3** | $73\% - 91\%$ | **70%** | **20%** | **10%** | 0% |
| **Stage 4** | $91\% - 100\%$ | **58%** | **24%** | **14%** | **4%** |

### Critical Pretraining Rules:
- **100% Loss-Bearing Tokens**: Every token in pretraining windows must contribute to cross-entropy loss. Never mask tokens with `-100` label masking during pretraining.
- **Zero SFT Contamination**: Pretraining streams pure contiguous documents. Do NOT inject conversational SFT Q&A templates or prompt-response boundaries into pretraining.
- **Unrestricted Tool Exploration**: Pretraining tool loss is strictly **`0.0`**. Allow the model to freely explore tool tokens (`<act>`) without loss penalties.

---

## 3. Cognitive Architecture & Output Protocol

CerebroSpheroid V5 enforces an industry-standard **3-tier cognitive boundary**:
1. `<think> ... </think>`: Latent deliberation and verbal derivation across concentric shells (DeepSeek-R1 / Qwen2.5 standard).
2. `<tool_call> ... </tool_call>`: Deterministic external tool invocations (`CALCULATOR`, `FACTUAL_DB`, `BASH`, `PYTHON`). Tool observations feed back as `<tool_response> ... </tool_response>`.
3. Plain text: Direct, objective response after deliberation completes.

### Token Standardization Invariants:
- **PURGE NON-STANDARD TOKENS**: Custom non-standard pseudo-biological tokens (`<act>`, `</act>`, `<stim>`, `</stim>`) are strictly purged from all new outputs, data formatting, and generation pipelines.
- Prompts are ingested as clean natural language or standard ChatML (`<|im_start|>user\n...<|im_end|>`), NEVER wrapped in `<stim> ... </stim>`.
- Python code generation is emitted in standard Markdown code fences (````python\n...````), NOT wrapped in tool or action tags.

---

## 4. Optimization & Embedding Regularization

- **SCMER (Selective Cosine-Margin Embedding Regularization)**:
  - Must remain enabled in `SpheroidOptimizer` and `src/jax/scmer_optimizer.py`.
  - Default parameters: $\lambda = 0.05$, cosine margin $m = 0.12$.
  - Actively penalizes positive off-diagonal embedding correlations to prevent representational collapse and embedding anisotropy.

---

## 5. Verification Protocol

After modifying any training logic, data loader, or architectural module, **always** run the standard test suite:

```bash
pytest tests/test_autonomous_smollm2_curriculum.py \
       tests/test_act_tool_calling_boundary.py \
       tests/test_stage_transitions_and_chunking.py \
       tests/test_graded_curriculum.py \
       tests/test_jax_scmer.py
```

All unit tests must pass before declaring any task complete.
