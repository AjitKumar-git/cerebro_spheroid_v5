"""
CerebroSpheroid V5: Script Catalog & Functional Registry
========================================================

Scripts in this directory are cataloged and tracked by their functional domain:

1. OPERATIONAL & LAUNCHERS:
   - `train_v5_tpu_jax.py`: Unified CLI forwarding launcher to `training.train_v5_tpu_jax`.
   - `auto_tune_tpu.py`: Standalone CLI for accelerator probing, ladder benchmarking, and caching.
   - `patch_colab_notebooks.py`: Synchronizes Colab notebooks with canonical 3-step execution flow.
   - `launch_kaggle_tpu.py`: TPU v5 lite / v3-8 launcher for Kaggle environments.
   - `track_loss_regression.py`: Telemetry tracker synchronizing loss curves with bioflo.in.

2. DATASET BUILDERS & DISTILLATION:
   - `build_chat_thinking_dataset.py`: Multi-turn deliberation and reasoning dataset builder.
   - `build_layman_sft_dataset.py`: Conversational alignment and grounded instruction builder.
   - `build_frontier_100k_dataset.py`: 100k high-complexity instruction synthesis.
   - `build_anti_sycophancy_dataset.py`: Truthfulness and anti-sycophancy evaluation corpus.
   - `build_dictionary_dataset.py`: Rare token preservation and lexical corpus builder.
   - `digest_cosmopedia_v2.py`: Sharded pre-tokenized Cosmopedia v2 digestion.
   - `distill_gemma_to_stream_dataset.py`: Gemma 2 distillation dataset extractor.
   - `upload_*_to_hf.py`: Push generated datasets to Hugging Face Hub.

3. BENCHMARKS & EVALUATION:
   - `benchmark_industry_standard.py`: MMLU, GSM8k, ARC, HellaSwag evaluation suite.
   - `benchmark_frontier_suite.py`: High-difficulty reasoning and cognitive benchmark.
   - `eval_passkey_retrieval.py`: Long-context passkey needle-in-a-haystack retrieval evaluation.
   - `compare_cerebro_vs_smollm.py`: Parity evaluation against SmolLM2 baselines.

4. AUDITS & MANIFOLD DIAGNOSTICS:
   - `audit_agi_emergence.py`: Emergence metric analysis across deliberate thinking cycles.
   - `audit_anti_sycophancy.py`: Verification of model truthfulness against user bias.
   - `audit_dataset_stream.py`: Multi-corpus mixture validation and domain health.
   - `audit_weight_saturation.py`: Singular value decomposition and weight spectrum tracking.
   - `analyze_weight_distribution.py`: Shell parameter histograms and manifold drift.
"""
