#!/usr/bin/env python3
"""
CerebroSpheroid V5: Colab TPU Runner (Forwarding Entrypoint)
============================================================

Delegates directly to `scripts.train_v5_small_colab_tpu` to maintain
a single source of truth across repository scripts and notebooks.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.train_v5_small_colab_tpu import (
    main,
    parse_args,
    run_training,
    run_generalization_probe,
    compute_cosine_lr,
)

if __name__ == "__main__":
    main()
