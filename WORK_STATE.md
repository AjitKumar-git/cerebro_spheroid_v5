# Agent Work State & Coordination

## Status: IN PROGRESS
- **Active Task**: Stage 1 Context Curriculum (1024 Tokens) & Pre-Seeded HBM Safety Guard
- **Engine**: CerebroSpheroid V5 pure JAX high-throughput engine on Cloud TPU v5e-1 (16GB)
- **Current Checkpoint**: Step 122,844 (Loss: 3.5508 eval, 3.03 EMA, ~249k tok/s)
- **Deliberation Cycles**: 8 Subconscious Deliberation Cycles
- **Context Horizon**: Stage 1 (1024 total tokens = 256 prompt + 768 target)
- **Batch Resolution**: Micro-batch 64 with `remat=True` (grad accum = 4, effective batch = 256, 4.5 GB / 28.9% HBM)

## Recent Completed Changes
1. **Resolved Pre-Seeded Winner Lookup with Cap Suffix (`autotune/engine.py`)**:
   - `AutoTuner` now strips `_cap<BS>` when querying verified local and remote configurations, loading `TPU_v5_lite_16GB_seq1024` instantly even when `max_batch_size=128` is passed in Colab.
2. **Pre-Seeded Winner Calibrated to Batch 64 Remat (`configs/hardware_configs.json`)**:
   - Updated `TPU_v5_lite_16GB_seq1024` to `batch_size: 64, remat: True` (249,305 tok/s, 4.5 GB / 28.9% HBM).
3. **Hard HBM Micro-Token Guard Capping (`training/train_v5_tpu_jax.py`)**:
   - Bounded micro-batch tokens to 65,536 (`max_micro_tokens = 65536`), automatically scaling micro-batch size from 128 -> 64 at 1024 context, and scaling gradient accumulation (4 steps) to preserve the exact effective batch size of 256.
4. **Candidate Ladder Search Pruning (`autotune/search.py`)**:
   - Restricted ladder at seq >= 1024 on 16GB accelerators to `[16, 32, 64]` and bypassed non-remat probes for batch >= 64, preventing XLA compiler aborts during fallback searches.
