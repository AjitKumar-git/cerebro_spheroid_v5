"""
Architecture Detection, Profiling & Fingerprinting Engine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class ArchitectureInfo:
    architecture_name: str
    param_count: Optional[int] = None
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    seq_len: int = 256
    prompt_len: int = 256
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    architecture_signature: str = ""

    def __post_init__(self):
        if not self.architecture_signature:
            self.architecture_signature = self.compute_signature()

    def compute_signature(self) -> str:
        """Computes a deterministic architectural fingerprint."""
        norm_name = str(self.architecture_name).strip().lower().replace(" ", "_").replace("/", "_")
        components = [norm_name]

        if self.param_count:
            if self.param_count >= 1_000_000_000:
                components.append(f"{self.param_count / 1e9:.1f}B")
            elif self.param_count >= 1_000_000:
                components.append(f"{self.param_count / 1e6:.1f}M")
            else:
                components.append(f"{self.param_count}p")

        if self.hidden_dim:
            components.append(f"d{self.hidden_dim}")
        if self.num_layers:
            components.append(f"l{self.num_layers}")

        sig = "_".join(components)
        return sig


def extract_architecture_signature(
    model_or_config: Any,
    seq_len: int = 256,
    prompt_len: int = 256,
    preset: Optional[str] = None,
) -> ArchitectureInfo:
    """
    Extracts structural hyperparameters from any model, configuration object,
    dataclass, dictionary, or preset string.
    """
    if isinstance(model_or_config, ArchitectureInfo):
        return model_or_config

    arch_name = preset or "custom_model"
    param_count = None
    hidden_dim = None
    num_layers = None
    num_heads = None
    vocab_size = None
    metadata: Dict[str, Any] = {}

    # Case 1: Simple string preset
    if isinstance(model_or_config, str):
        arch_name = model_or_config
    elif isinstance(model_or_config, dict):
        arch_name = model_or_config.get("name", model_or_config.get("preset", preset or "dict_model"))
        param_count = model_or_config.get("param_count") or model_or_config.get("n_params")
        hidden_dim = (
            model_or_config.get("hidden_dim")
            or model_or_config.get("surface_dim")
            or model_or_config.get("d_model")
            or model_or_config.get("dim")
        )
        num_layers = (
            model_or_config.get("num_layers")
            or model_or_config.get("num_shells")
            or model_or_config.get("n_layers")
            or model_or_config.get("depth")
        )
        num_heads = (
            model_or_config.get("num_heads")
            or model_or_config.get("surface_heads")
            or model_or_config.get("n_heads")
        )
        vocab_size = model_or_config.get("vocab_size")
        metadata = {k: v for k, v in model_or_config.items() if isinstance(v, (int, float, str, bool))}
    elif hasattr(model_or_config, "__dict__"):
        d = getattr(model_or_config, "__dict__", {})
        arch_name = getattr(model_or_config, "name", preset or type(model_or_config).__name__)
        hidden_dim = (
            getattr(model_or_config, "hidden_dim", None)
            or getattr(model_or_config, "surface_dim", None)
            or getattr(model_or_config, "d_model", None)
            or getattr(model_or_config, "dim", None)
        )
        num_layers = (
            getattr(model_or_config, "num_layers", None)
            or getattr(model_or_config, "num_shells", None)
            or getattr(model_or_config, "n_layers", None)
            or getattr(model_or_config, "depth", None)
        )
        num_heads = (
            getattr(model_or_config, "num_heads", None)
            or getattr(model_or_config, "surface_heads", None)
            or getattr(model_or_config, "n_heads", None)
        )
        vocab_size = getattr(model_or_config, "vocab_size", None)
        metadata = {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}

    return ArchitectureInfo(
        architecture_name=arch_name,
        param_count=param_count,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        vocab_size=vocab_size,
        seq_len=seq_len,
        prompt_len=prompt_len,
        custom_metadata=metadata,
    )


class ArchitectureDetector:
    """
    Tracks seen architectures and detects when an unoptimized or newly altered
    model architecture is introduced to the training pipeline.
    """

    def __init__(self, cache_manager: Any = None):
        self.cache_manager = cache_manager

    def is_new_architecture(
        self,
        arch_info: ArchitectureInfo,
        hardware_sig: str,
        seq_len: int = 256,
        max_batch_size: Optional[int] = None,
        tune_objective: str = "balanced",
    ) -> bool:
        """
        Determines whether the given architecture needs auto-tuning on this hardware.
        Returns False if a verified config already exists in cache, True if new.
        """
        if self.cache_manager is None:
            from .cache import load_local_hardware_config
            load_fn = load_local_hardware_config
        else:
            load_fn = self.cache_manager.load

        seq_suffix = f"_seq{seq_len}" if seq_len != 256 else ""
        cap_suffix = f"_cap{int(max_batch_size)}" if max_batch_size is not None else ""
        cache_sig = f"{hardware_sig}{seq_suffix}{cap_suffix}"

        cached = load_fn(arch_info.architecture_name, cache_sig)
        if cached is None and cache_sig != hardware_sig and max_batch_size is None:
            fallback = load_fn(arch_info.architecture_name, hardware_sig)
            if fallback and fallback.get("seq_len") == seq_len:
                cached = fallback

        if cached is None:
            return True

        if cached.get("seq_len", 256) != seq_len:
            return True

        if max_batch_size is not None and int(cached.get("batch_size", 0)) > int(max_batch_size):
            return True

        cached_obj = cached.get("tune_objective")
        if cached_obj is not None and cached_obj != tune_objective:
            return True

        if cached.get("tok_per_sec", 0) <= 0 and cached.get("steps_per_sec", 0) <= 0:
            return True

        return False
