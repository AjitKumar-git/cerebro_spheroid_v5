import os
import sys

# Prevent JAX/XLA from greedily pre-allocating 75% of GPU VRAM in mixed PyTorch+JAX pytest runs.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
