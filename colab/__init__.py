"""
CerebroSpheroid V5: Colab & Remote Launchers
============================================

Interactive notebooks and headless execution scripts for Google Colab environments:

1. `CerebroSpheroid_V5_Colab_TPU_JAX.ipynb`:
   - Canonical 3-step interactive training notebook on Google Cloud TPU v5e/v5-lite.
   - Automatically clones repo, installs JAX TPU wheels, pulls hardware profile, and starts pretraining.

2. `CerebroSpheroid_V5_Colab_TPU_JAX_small.ipynb`:
   - Dedicated preset configuration targeting `v5_small` (105.9M params / 85.4M active compute).

3. `run_one_click_tpu.py`:
   - Headless one-click execution script that configures runtime environment and triggers `training.train_v5_tpu_jax`.

4. `distill_gemma_to_v5_small.py`:
   - End-to-end distillation pipeline transferring representations from Gemma 2 into CerebroSpheroid V5.
"""
