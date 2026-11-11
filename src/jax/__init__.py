"""
CerebroSpheroid V5: Pure JAX Spheroid Engine Package.
"""

from .v5_model import (
    ModelConfigV5,
    PRESETS_V5,
    init_params_v5,
    generate_small_world_indices,
    encode_sensory,
    deliberate_step,
    emit_step,
    training_rollout,
    compute_manifold_diagnostics_jax,
)

from .spheroid_optimizer import (
    zeropower_via_newtonschulz5_jax,
    clamp_spectral_norm_jax,
    init_spheroid_opt_state,
    build_spheroid_train_engine,
    build_train_step_spheroid,
)

from .checkpoint import (
    save_jax_to_torch_pt,
    load_torch_pt_to_jax,
    adapt_and_load_checkpoint_weights,
    discover_latest_checkpoint,
    AsyncCheckpointManager,
)

from .data_streamer import LinguisticDataStreamer
from .auto_tuner import run_auto_tuner

__all__ = [
    "ModelConfigV5",
    "PRESETS_V5",
    "init_params_v5",
    "generate_small_world_indices",
    "encode_sensory",
    "deliberate_step",
    "emit_step",
    "training_rollout",
    "compute_manifold_diagnostics_jax",
    "zeropower_via_newtonschulz5_jax",
    "clamp_spectral_norm_jax",
    "init_spheroid_opt_state",
    "build_spheroid_train_engine",
    "build_train_step_spheroid",
    "save_jax_to_torch_pt",
    "load_torch_pt_to_jax",
    "adapt_and_load_checkpoint_weights",
    "discover_latest_checkpoint",
    "AsyncCheckpointManager",
    "LinguisticDataStreamer",
    "run_auto_tuner",
]
