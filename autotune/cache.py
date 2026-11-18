"""
Configuration Caching & Accelerator/Architecture Persistence.
============================================================

Handles persistent local filesystem caching and optional remote cloud sync,
ensuring verified optimal configurations can be loaded in zero seconds.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

LOCAL_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "hardware_configs.json")
)

_FALLBACK_PATHS = [
    LOCAL_CONFIG_PATH,
    os.path.join("configs", "hardware_configs.json"),
    os.path.join("checkpoints", "hardware_configs.json"),
]


def load_local_hardware_config(
    preset_or_arch: str,
    device_signature: str,
    custom_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Loads cached configuration from local filesystem if present."""
    norm_preset = (
        "v5_small"
        if preset_or_arch in ("tpu_v5e_1", "tpu_v6e_1", "tpu_v6e", "tpu", "v5_tpu")
        else ("v5_compact" if "compact" in preset_or_arch else preset_or_arch)
    )

    search_paths = [custom_path] if custom_path else _FALLBACK_PATHS

    for p_key in [preset_or_arch, norm_preset]:
        for path in search_paths:
            if not path:
                continue
            try:
                if os.path.exists(path) and os.path.getsize(path) > 2:
                    with open(path, "r") as f:
                        data = json.load(f)
                        if p_key in data and device_signature in data[p_key]:
                            return data[p_key][device_signature]
            except Exception:
                pass
    return None


def save_local_hardware_config(
    preset_or_arch: str,
    device_signature: str,
    config: Dict[str, Any],
    custom_path: Optional[str] = None,
) -> None:
    """Saves verified configuration to local filesystem."""
    norm_preset = (
        "v5_small"
        if preset_or_arch in ("tpu_v5e_1", "tpu_v6e_1", "tpu_v6e", "tpu", "v5_tpu")
        else ("v5_compact" if "compact" in preset_or_arch else preset_or_arch)
    )

    target_path = custom_path or LOCAL_CONFIG_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    data = {}
    if os.path.exists(target_path) and os.path.getsize(target_path) > 2:
        try:
            with open(target_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

    for p_key in set([preset_or_arch, norm_preset]):
        if p_key not in data:
            data[p_key] = {}
        data[p_key][device_signature] = config

    try:
        with open(target_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠️ [AutoTuner] Could not write local config cache: {e}")


class HardwareConfigCache:
    """Object-oriented wrapper around local and remote configuration storage."""

    def __init__(self, cache_path: Optional[str] = None):
        self.cache_path = cache_path or LOCAL_CONFIG_PATH

    def load(self, preset_or_arch: str, device_signature: str) -> Optional[Dict[str, Any]]:
        return load_local_hardware_config(preset_or_arch, device_signature, custom_path=self.cache_path)

    def save(self, preset_or_arch: str, device_signature: str, config: Dict[str, Any]) -> None:
        save_local_hardware_config(preset_or_arch, device_signature, config, custom_path=self.cache_path)
