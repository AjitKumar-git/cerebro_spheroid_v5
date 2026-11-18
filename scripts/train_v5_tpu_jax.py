#!/usr/bin/env python3
"""
CerebroSpheroid V5: Pure JAX High-Throughput Engine (CLI Forwarding Entrypoint)
==============================================================================

Delegates directly to `training.train_v5_tpu_jax` to maintain a single source
of truth across CLI training, scripts, and Jupyter / Colab notebooks.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.train_v5_tpu_jax import (
    main,
    parse_args,
    run_training,
    build_data_stream,
)

if __name__ == "__main__":
    main()
