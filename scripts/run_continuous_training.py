#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience launcher for CerebroSpheroid V5 Continuous Self-Healing Trainer.
Directly forwards execution to training/continuous_trainer.py.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.continuous_trainer import parse_args, run_continuous_training_supervisor

if __name__ == "__main__":
    args, unknown = parse_args()
    sys.exit(run_continuous_training_supervisor(args, unknown))
