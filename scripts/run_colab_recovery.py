#!/usr/bin/env python3
"""
scripts/run_colab_recovery.py
=============================

Forwarding entrypoint to `colab.run_colab_recovery` to maintain a single source
of truth across Colab notebooks and terminal scripts.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from colab.run_colab_recovery import main

if __name__ == "__main__":
    main()
