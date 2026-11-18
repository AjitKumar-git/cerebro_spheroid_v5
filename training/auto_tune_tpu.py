#!/usr/bin/env python3
"""
CerebroSpheroid V5: Standalone TPU Auto-Tuner & Hardware Benchmark (Forwarding Entrypoint)
========================================================================================

Delegates directly to `scripts.auto_tune_tpu` to maintain a single source
of truth across training modules and scripts.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.auto_tune_tpu import main, parse_args

if __name__ == "__main__":
    main()
