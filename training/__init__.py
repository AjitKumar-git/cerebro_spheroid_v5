"""
CerebroSpheroid V5: Training Engines
====================================

Authoritative training pipelines for CerebroSpheroid V5:

1. `train_v5_tpu_jax.py`:
   - Pure JAX high-throughput pretraining engine optimized for Google Cloud TPU v5e/v5-lite.
   - 100% loss-bearing contiguous pretrain streamer with 4-stage SmolLM2 developmental curriculum.
   - SCMER (Selective Cosine-Margin Embedding Regularization) with Polar Order-5 Newton-Schulz iterations.
   - Zero-pause async checkpoint dispatch (bare BF16 cloud sync + full local optimizer state).
   - Dynamic website token vault resolution without manual environment secrets.

2. `train_v5_sft.py`:
   - Cognitive reasoning and specialized tri-head supervised fine-tuning.
   - Aligns thinking deliberation (<think>...</think>), deterministic tool calling (<tool_call>...),
     and conversational responses.
   - Integrates grounded database tools and dynamic length scaling.

3. `auto_tune_tpu.py`:
   - Automated hardware detection, VRAM maximization, and remat ladder tuning.
"""

from .train_v5_tpu_jax import (
    main as train_tpu_main,
    run_training as run_tpu_training,
    parse_args as parse_tpu_args,
    build_data_stream as build_tpu_data_stream,
)
from .train_v5_sft import (
    main as train_sft_main,
    parse_sft_args,
    run_sft_training,
)

__all__ = [
    "train_tpu_main",
    "run_tpu_training",
    "parse_tpu_args",
    "build_tpu_data_stream",
    "train_sft_main",
    "parse_sft_args",
    "run_sft_training",
]
