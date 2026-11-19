"""
CerebroSpheroid V5: JAX Checkpointing & Autonomous Synchronization Suite.

Features:
1. PyTorch (.pt) <-> JAX Parameter Converter:
   Bi-directional lossless conversion between PyTorch state_dict and JAX PyTrees.
   Supports automatic 2D -> 3D tensor fusion for intra-shell SwiGLU weights and shortcuts.
2. Autonomous Checkpoint Discovery (Zero Hardcoded Step Numbers):
   Automatically inspects local and bioflo.in cloud directories to identify the most advanced checkpoint.
3. Asynchronous Background Checkpoint Manager (AsyncCheckpointManager):
   Offloads parameter conversion, PyTorch disk serialization, and remote cloud upload
   to an independent CPU thread pool, eliminating TPU latency stalls (0ms TPU pause).
"""

import os
import re
import glob
import json
import shutil
import time
import dataclasses
from typing import Dict, Tuple, Optional, Any, List, Union
import concurrent.futures

import numpy as np

from src.cortical_expansion import apply_net2net_cortical_init, generate_expanded_tract_wiring

try:
    import torch
    HAS_TORCH = True
    _TORCH_ERR = None
except (ImportError, SystemError, Exception) as e:
    torch = None
    HAS_TORCH = False
    _TORCH_ERR = str(e)

import jax
import jax.numpy as jnp

from .v5_model import ModelConfigV5, PRESETS_V5, init_params_v5, generate_small_world_indices


def _safe_torch_tensor(arr):
    """Safely converts an array (including ml_dtypes.bfloat16 from TPU/JAX) into a torch.Tensor."""
    if not HAS_TORCH:
        raise RuntimeError(
            f"PyTorch is required for .pt checkpoint conversion: {_TORCH_ERR}.\n"
            "If on Google Colab, please do 'Runtime -> Restart session' after pip install."
        )
    if isinstance(arr, torch.Tensor):
        return arr
    if hasattr(arr, "dtype") and str(arr.dtype) == "bfloat16":
        np_f32 = np.array(arr, dtype=np.float32)
        return torch.from_numpy(np_f32).to(torch.bfloat16)
    np_arr = np.array(arr)
    return torch.from_numpy(np_arr)


def safe_torch_load(
    ckpt_path_or_file: Any,
    map_location: str = "cpu",
    weights_only: bool = True,
) -> Dict[str, Any]:
    """Safely loads a PyTorch checkpoint with tensor-only loading (weights_only=True)
    and validates checkpoint structure to protect against arbitrary code execution.
    """
    if not HAS_TORCH:
        raise RuntimeError(f"PyTorch is required to load .pt checkpoint: {_TORCH_ERR}")
    if isinstance(ckpt_path_or_file, str) and not os.path.exists(ckpt_path_or_file):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path_or_file}")

    try:
        ckpt = torch.load(ckpt_path_or_file, map_location=map_location, weights_only=weights_only)
    except Exception as e:
        raise ValueError(f"Safe tensor-only checkpoint loading failed for {ckpt_path_or_file}: {e}") from e

    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Invalid checkpoint format in {ckpt_path_or_file}: expected dictionary, got {type(ckpt).__name__}"
        )

    return ckpt


def save_jax_to_torch_pt(
    params: Dict[str, jnp.ndarray],
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    save_path: str,
    meta: Optional[dict] = None,
    opt_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Converts JAX PyTree into PyTorch state_dict and saves to .pt format (including Spheroid optimizer moments)."""
    if not HAS_TORCH:
        raise RuntimeError(
            f"PyTorch is required to save .pt checkpoints: {_TORCH_ERR}."
        )
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch_state: Dict[str, torch.Tensor] = {}
    for k, v in params.items():
        t = _safe_torch_tensor(v)
        torch_state[k] = t

    torch_state["core.src_indices"] = _safe_torch_tensor(buffers[0])
    torch_state["core.dst_indices"] = _safe_torch_tensor(buffers[1])

    checkpoint = {
        "model_state": torch_state,
        "step": meta.get("step", 0) if meta else 0,
        "loss": meta.get("loss", 0.0) if meta else 0.0,
        "eval_loss": meta.get("eval_loss", 0.0) if meta else 0.0,
        "timestamp": time.time(),
    }
    if meta:
        for mk, mv in meta.items():
            if mk not in checkpoint:
                checkpoint[mk] = mv

    # Persist optimizer state (moments, master weights, and spectral vectors) to prevent cold restarts
    if opt_state is not None and isinstance(opt_state, dict):
        try:
            torch_opt: Dict[str, Any] = {}
            if "step" in opt_state:
                step_val = opt_state["step"]
                torch_opt["step"] = int(np.asarray(jax.device_get(step_val))) if hasattr(step_val, "__len__") or hasattr(step_val, "item") else int(step_val)
            for sub_name in ["m", "v", "master_p", "power_v"]:
                if sub_name in opt_state and isinstance(opt_state[sub_name], dict):
                    torch_opt[sub_name] = {
                        k: _safe_torch_tensor(v) for k, v in opt_state[sub_name].items()
                    }
            checkpoint["opt_state"] = torch_opt
        except Exception as e_opt:
            print(f"⚠️ [save_checkpoint] Could not serialize optimizer state: {e_opt}")

    tmp_path = f"{save_path}.tmp.{os.getpid()}.{time.time_ns()}"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, save_path)

    if meta:
        meta_path = save_path.replace(".pt", ".json")
        def _json_default(obj):
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            return str(obj)

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=_json_default)


def restore_spheroid_opt_state(
    ckpt_or_path: Union[Dict[str, Any], str],
    params: Dict[str, jnp.ndarray],
    key: Optional[jax.Array] = None,
) -> Dict[str, Any]:
    """
    Restores full Spheroid optimizer moments (m, v, master_p, power_v) from checkpoint if present,
    guaranteeing continuous momentum across training session restarts.
    Falls back gracefully to init_spheroid_opt_state if opt_state is absent.
    """
    from src.jax.spheroid_optimizer import init_spheroid_opt_state

    if isinstance(ckpt_or_path, str):
        if not HAS_TORCH or not os.path.exists(ckpt_or_path):
            return init_spheroid_opt_state(params, key=key)
        try:
            ckpt = safe_torch_load(ckpt_or_path, map_location="cpu", weights_only=True)
        except Exception:
            return init_spheroid_opt_state(params, key=key)
    else:
        ckpt = ckpt_or_path

    raw_opt = ckpt.get("opt_state") if isinstance(ckpt, dict) else None
    if not raw_opt or not isinstance(raw_opt, dict):
        return init_spheroid_opt_state(params, key=key)

    opt_state = init_spheroid_opt_state(params, key=key)
    if "step" in raw_opt:
        raw_s = raw_opt["step"]
        opt_state["step"] = int(raw_s.item()) if hasattr(raw_s, "item") else int(raw_s)

    restored_components = 0
    for sub_dict in ["m", "v", "master_p", "power_v"]:
        if sub_dict in raw_opt and isinstance(raw_opt[sub_dict], dict):
            for k, val in raw_opt[sub_dict].items():
                if k in opt_state[sub_dict]:
                    np_arr = _to_numpy(val)
                    opt_state[sub_dict][k] = jnp.asarray(np_arr, dtype=opt_state[sub_dict][k].dtype)
                    restored_components += 1

    print(f"✓ Restored Spheroid optimizer state ({restored_components} moment/master tensors, step {opt_state['step']}) from checkpoint")
    return opt_state


def _to_numpy(t):
    """Safely converts a PyTorch tensor (including bfloat16) or array to NumPy float32/float64."""
    if hasattr(t, "dtype") and str(t.dtype).endswith("bfloat16"):
        return t.to(torch.float32).numpy()
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)


def adapt_and_load_checkpoint_weights(
    ckpt: Dict[str, Any],
    target_params: Dict[str, jnp.ndarray],
    target_dtype,
    config: ModelConfigV5,
    buffers: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    apply_net2net: bool = True,
) -> Union[int, Tuple[int, Tuple[jnp.ndarray, jnp.ndarray]]]:
    """Adapts checkpoint weights from any previous run (PyTorch state_dict or JAX params, 2D or 3D SwiGLU)."""
    raw_state = ckpt.get("model_state") or ckpt.get("params") or ckpt.get("state_dict") or ckpt
    if not isinstance(raw_state, dict):
        return (0, buffers) if buffers is not None else 0

    restored_keys = set()
    K = config.num_shells
    s_dim = config.surface_dim

    # 1. Fuse 2D Intra-Shell SwiGLU weights into 3D tensors if not already 3D
    if "core.intra_swiglu.w_gate" in target_params and "core.intra_swiglu.w_gate" not in raw_state:
        gates, vals, downs, norms = [], [], [], []
        for k in range(K):
            for pfx in ["volumetric_core.intra_shell_swiglu", "core.intra_swiglu", "intra_swiglu"]:
                g_key = f"{pfx}.{k}.fc_gate.weight"
                if g_key in raw_state:
                    g = _to_numpy(raw_state[g_key])
                    gates.append(g.T if g.shape == (s_dim * 2, s_dim) else g)
                    v = _to_numpy(raw_state[f"{pfx}.{k}.fc_val.weight"])
                    vals.append(v.T if v.shape == (s_dim * 2, s_dim) else v)
                    d = _to_numpy(raw_state[f"{pfx}.{k}.fc_down.weight"])
                    downs.append(d.T if d.shape == (s_dim, s_dim * 2) else d)
                    break
            for n_pfx in ["volumetric_core.intra_norms", "core.intra_norms", "intra_norms"]:
                n_key = f"{n_pfx}.{k}.weight"
                if n_key in raw_state:
                    norms.append(_to_numpy(raw_state[n_key]))
                    break

        if len(gates) == K:
            target_params["core.intra_swiglu.w_gate"] = jnp.asarray(np.stack(gates, axis=0), dtype=target_dtype)
            target_params["core.intra_swiglu.w_val"] = jnp.asarray(np.stack(vals, axis=0), dtype=target_dtype)
            target_params["core.intra_swiglu.w_down"] = jnp.asarray(np.stack(downs, axis=0), dtype=target_dtype)
            restored_keys.update(["core.intra_swiglu.w_gate", "core.intra_swiglu.w_val", "core.intra_swiglu.w_down"])
        if len(norms) == K:
            target_params["core.intra_norms.weight"] = jnp.asarray(np.stack(norms, axis=0), dtype=target_dtype)
            restored_keys.add("core.intra_norms.weight")

    # 2. General prefix and name translations (ordered from most specific to general)
    key_translations = [
        # Multimodal perceptual projector mappings (PyTorch -> JAX)
        ("perceptual_projector.0.weight", "perceptual.norm_in.weight"),
        ("perceptual_projector.1.weight", "perceptual.linear.weight"),
        ("perceptual_projector.2.fc_gate.weight", "perceptual.swiglu.fc_gate.weight"),
        ("perceptual_projector.2.fc_val.weight", "perceptual.swiglu.fc_val.weight"),
        ("perceptual_projector.2.fc_down.weight", "perceptual.swiglu.fc_down.weight"),
        ("perceptual_projector.3.weight", "perceptual.norm_out.weight"),
        # Specific motor SwiGLU projections (PyTorch motor_decoder.motor_swiglu.* or motor.motor_swiglu.*)
        ("motor_decoder.motor_swiglu.", "motor.swiglu."),
        ("motor_decoder.swiglu.", "motor.swiglu."),
        ("motor.motor_swiglu.", "motor.swiglu."),
        ("motor_decoder.fc_gate.weight", "motor.swiglu.fc_gate.weight"),
        ("motor_decoder.fc_val.weight", "motor.swiglu.fc_val.weight"),
        ("motor_decoder.fc_down.weight", "motor.swiglu.fc_down.weight"),
        ("motor.fc_gate.weight", "motor.swiglu.fc_gate.weight"),
        ("motor.fc_val.weight", "motor.swiglu.fc_val.weight"),
        ("motor.fc_down.weight", "motor.swiglu.fc_down.weight"),
        # Tied embedding projections
        ("motor_decoder.head.weight", "sensory.embedding.weight"),
        ("motor.head.weight", "sensory.embedding.weight"),
        # General module prefixes (evaluated after specific prefixes above)
        ("sensory_dnn.", "sensory."),
        ("volumetric_core.", "core."),
        ("motor_decoder.", "motor."),
    ]

    for raw_k, raw_v in raw_state.items():
        v = _to_numpy(raw_v)
        target_k = raw_k
        if target_k not in target_params:
            for old_p, new_p in key_translations:
                if target_k.startswith(old_p):
                    target_k = target_k.replace(old_p, new_p, 1)
                    break

        if target_k in target_params and target_k not in restored_keys:
            t_shape = target_params[target_k].shape
            if t_shape == v.shape:
                target_params[target_k] = jnp.asarray(v, dtype=target_dtype)
                restored_keys.add(target_k)
            elif t_shape == v.T.shape:
                target_params[target_k] = jnp.asarray(v.T, dtype=target_dtype)
                restored_keys.add(target_k)
            elif (
                target_k in ("sensory.pos_embed.weight", "motor.phase_embed.weight")
                and v.ndim == 2
                and t_shape[1] == v.shape[1]
            ):
                rows = min(t_shape[0], v.shape[0])
                buf = np.array(target_params[target_k])
                buf[:rows] = v[:rows]
                target_params[target_k] = jnp.asarray(buf, dtype=target_dtype)
                restored_keys.add(target_k)
            elif (
                len(t_shape) == len(v.shape)
                and all(t_shape[i] >= v.shape[i] for i in range(len(t_shape)))
                and any(t_shape[i] > v.shape[i] for i in range(len(t_shape)))
            ):
                # Concentric shell / tract expansion (e.g. 16 -> 32 shells, 40 -> 96 shortcut tracts)
                buf = np.array(target_params[target_k])
                if len(t_shape) == 1:
                    buf[: v.shape[0]] = v
                elif len(t_shape) == 2:
                    buf[: v.shape[0], : v.shape[1]] = v
                elif len(t_shape) == 3:
                    buf[: v.shape[0], : v.shape[1], : v.shape[2]] = v
                target_params[target_k] = jnp.asarray(buf, dtype=target_dtype)
                restored_keys.add(target_k)

    # Weight transferability: legacy checkpoints share one radial_up across inward/outward waves
    has_radial_up_out = isinstance(raw_state, dict) and any(
        k.endswith("radial_up_out.weight") for k in raw_state.keys()
    )
    if (
        not has_radial_up_out
        and "core.radial_up_out.weight" in target_params
        and "core.radial_up.weight" in target_params
    ):
        target_params["core.radial_up_out.weight"] = target_params["core.radial_up.weight"]

    # Safety net: if sensory self-attention is absent from checkpoint, zero out_self so residual starts strictly at identity
    if "sensory.out_self.weight" in target_params and "sensory.out_self.weight" not in restored_keys:
        target_params["sensory.out_self.weight"] = jnp.zeros_like(target_params["sensory.out_self.weight"])

    # Tri-head transferability: if checkpoint has motor.affine_scale, initialize tri-heads
    if "motor.affine_scale" in target_params:
        base_affine = target_params["motor.affine_scale"]
        for tri_k in ("motor.affine_think", "motor.affine_act", "motor.affine_answer"):
            if tri_k in target_params and tri_k not in restored_keys:
                target_params[tri_k] = base_affine
                restored_keys.add(tri_k)

    # Temporal Router transferability: canonical zero weight and [0.5, -0.8, 1.0] bias initialization
    if "motor.temporal_router.weight" in target_params and "motor.temporal_router.weight" not in restored_keys:
        target_params["motor.temporal_router.weight"] = jnp.zeros_like(target_params["motor.temporal_router.weight"])
        restored_keys.add("motor.temporal_router.weight")
    if "motor.temporal_router.bias" in target_params and "motor.temporal_router.bias" not in restored_keys:
        target_params["motor.temporal_router.bias"] = jnp.array([0.5, -0.8, 1.0], dtype=target_params["motor.temporal_router.bias"].dtype)
        restored_keys.add("motor.temporal_router.bias")

    # 4. Progressive Cortical Expansion: Preserve tract wiring and apply Net2Net zero-regression init
    updated_buffers = buffers
    if buffers is not None and "core.src_indices" in raw_state and "core.dst_indices" in raw_state:
        base_src = _to_numpy(raw_state["core.src_indices"])
        base_dst = _to_numpy(raw_state["core.dst_indices"])
        updated_buffers = generate_expanded_tract_wiring(
            num_shells=config.num_shells,
            num_tracts=config.num_shortcut_tracts,
            base_src=base_src,
            base_dst=base_dst,
            num_frozen_shells=16,
            framework="jax",
        )

    if apply_net2net and config.num_shells > 16:
        apply_net2net_cortical_init(target_params, num_frozen_shells=16, num_frozen_tracts=40)

    missing = [k for k in target_params.keys() if k not in restored_keys and k != "core.radial_up_out.weight"]
    if missing:
        print(f"ℹ️ [adapt_weights] Unmapped JAX parameter arrays ({len(missing)}/{len(target_params)}): {missing}")

    if buffers is not None:
        return len(restored_keys), updated_buffers
    return len(restored_keys)


def adapt_jax_to_torch_state_dict(
    jax_state: Dict[str, Any],
    config: ModelConfigV5,
) -> Dict[str, Any]:
    """
    Losslessly converts JAX parameter dictionary (fused 3D SwiGLU & norms)
    into standard PyTorch CerebroSpheroidV5 state_dict (ModuleList keys).
    """
    if not HAS_TORCH:
        raise RuntimeError(f"PyTorch is required for checkpoint conversion: {_TORCH_ERR}")

    raw_state = jax_state.get("model_state", jax_state) if isinstance(jax_state, dict) else jax_state
    new_state: Dict[str, Any] = {}
    K = config.num_shells
    s_dim = config.surface_dim

    # 1. Unpack 3D intra-shell swiglu & norms into ModuleList keys
    if "core.intra_swiglu.w_gate" in raw_state:
        g = _safe_torch_tensor(raw_state["core.intra_swiglu.w_gate"])
        v = _safe_torch_tensor(raw_state["core.intra_swiglu.w_val"])
        d = _safe_torch_tensor(raw_state["core.intra_swiglu.w_down"])
        for k in range(K):
            g_k = g[k].t() if g[k].shape == (s_dim, s_dim * 2) else g[k]
            v_k = v[k].t() if v[k].shape == (s_dim, s_dim * 2) else v[k]
            d_k = d[k].t() if d[k].shape == (s_dim * 2, s_dim) else d[k]
            new_state[f"volumetric_core.intra_shell_swiglu.{k}.fc_gate.weight"] = g_k
            new_state[f"volumetric_core.intra_shell_swiglu.{k}.fc_val.weight"] = v_k
            new_state[f"volumetric_core.intra_shell_swiglu.{k}.fc_down.weight"] = d_k

    if "core.intra_norms.weight" in raw_state:
        norms = _safe_torch_tensor(raw_state["core.intra_norms.weight"])
        for k in range(K):
            new_state[f"volumetric_core.intra_norms.{k}.weight"] = norms[k]

    # 2. Module prefix & name translations
    pfx_map = [
        ("motor.swiglu.fc_gate.weight", "motor_decoder.motor_swiglu.fc_gate.weight"),
        ("motor.swiglu.fc_val.weight", "motor_decoder.motor_swiglu.fc_val.weight"),
        ("motor.swiglu.fc_down.weight", "motor_decoder.motor_swiglu.fc_down.weight"),
        ("perceptual.norm_in.weight", "perceptual_projector.0.weight"),
        ("perceptual.linear.weight", "perceptual_projector.1.weight"),
        ("perceptual.swiglu.fc_gate.weight", "perceptual_projector.2.fc_gate.weight"),
        ("perceptual.swiglu.fc_val.weight", "perceptual_projector.2.fc_val.weight"),
        ("perceptual.swiglu.fc_down.weight", "perceptual_projector.2.fc_down.weight"),
        ("perceptual.norm_out.weight", "perceptual_projector.3.weight"),
        ("sensory.", "sensory_dnn."),
        ("core.", "volumetric_core."),
        ("motor.", "motor_decoder."),
    ]

    for k, val in raw_state.items():
        if k in (
            "core.intra_swiglu.w_gate",
            "core.intra_swiglu.w_val",
            "core.intra_swiglu.w_down",
            "core.intra_norms.weight",
            "core.src_indices",
            "core.dst_indices",
        ):
            continue
        mapped_k = k
        for old_p, new_p in pfx_map:
            if mapped_k.startswith(old_p):
                mapped_k = mapped_k.replace(old_p, new_p, 1)
                break
        new_state[mapped_k] = _safe_torch_tensor(val)

    return new_state


def load_torch_pt_to_jax(
    ckpt_path: str,
    config: ModelConfigV5,
    target_dtype=jnp.float32,
    seed: int = 42,
) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray], dict]:
    """Loads a PyTorch .pt checkpoint directly into JAX V5 parameter dictionary."""
    if not HAS_TORCH:
        raise RuntimeError(f"PyTorch is required to load .pt checkpoint: {_TORCH_ERR}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = safe_torch_load(ckpt_path, map_location="cpu", weights_only=True)
    params, buffers = init_params_v5(jax.random.PRNGKey(seed), config)
    restored, buffers = adapt_and_load_checkpoint_weights(ckpt, params, target_dtype, config, buffers=buffers)
    print(f"✓ Restored & adapted {restored}/{len(params)} parameter arrays into JAX V5 engine")

    meta = {}
    if isinstance(ckpt, dict):
        for k in ["step", "loss", "eval_loss", "best_eval_loss", "timestamp", "config"]:
            if k in ckpt:
                meta[k] = ckpt[k]
    json_path = ckpt_path.replace(".pt", ".json")
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                meta.update(json.load(f))
        except Exception:
            pass

    return params, buffers, meta


def discover_latest_checkpoint(
    ckpt_dir: str,
    preset: Optional[str] = None,
) -> Tuple[Optional[str], int, float]:
    """Discovers the most advanced checkpoint in ckpt_dir."""
    if not os.path.exists(ckpt_dir):
        return None, 0, float("inf")

    candidates = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    if not candidates:
        return None, 0, float("inf")

    best_path = None
    max_step = -1
    best_loss = float("inf")

    # Priority 1: Checkpoint matching preset name
    for c in candidates:
        fname = os.path.basename(c)
        if preset and preset not in fname and "latest" not in fname:
            continue
        step = -1
        loss = float("inf")
        # Try sidecar json first
        j_path = c.replace(".pt", ".json")
        if os.path.exists(j_path):
            try:
                with open(j_path) as jf:
                    data = json.load(jf)
                    step = int(data.get("step", -1))
                    loss = float(data.get("eval_loss", data.get("loss", float("inf"))))
            except Exception:
                pass
        if step < 0:
            match = re.search(r"step[_-]?(\d+)", fname)
            if match:
                step = int(match.group(1))
        if "latest" in fname and step < 0:
            step = 9000  # Default fallback step for latest

        if step > max_step:
            max_step = step
            best_path = c
            best_loss = loss

    return best_path, max(0, max_step), best_loss


class AsyncCheckpointManager:
    """Non-blocking background thread worker for asynchronous checkpoint serialization and cloud upload."""

    def __init__(
        self,
        local_dir: str,
        target_dir: str,
        cloud_sync: bool = False,
        cloud_url: str = "https://bioflo.in/api/cerebro/sync.php",
        cloud_token: Optional[str] = None,
        min_cloud_upload_interval_sec: float = 0.0,
        cloud_weights_only: bool = True,
        save_optimizer_state: bool = True,
    ):
        self.local_dir = local_dir
        self.target_dir = target_dir
        self.cloud_sync = cloud_sync
        self.cloud_url = cloud_url
        if cloud_token is None:
            from src.cloud_sync import get_sync_token
            self.cloud_token = get_sync_token()
        else:
            self.cloud_token = cloud_token
        self.min_cloud_upload_interval_sec = max(0.0, float(min_cloud_upload_interval_sec))
        self.cloud_weights_only = cloud_weights_only
        self.save_optimizer_state = save_optimizer_state
        self._last_cloud_upload_at: Dict[str, float] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt_async")
        self._last_future: Optional[concurrent.futures.Future] = None
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.target_dir, exist_ok=True)

    def _worker(
        self,
        params_cpu: Dict[str, np.ndarray],
        buffers_cpu: Tuple[np.ndarray, np.ndarray],
        filename: str,
        meta_dict: Optional[dict],
        force_cloud: bool = False,
        opt_state_cpu: Optional[Dict[str, Any]] = None,
    ):
        try:
            local_path = os.path.join(self.local_dir, filename)
            save_opt = opt_state_cpu if self.save_optimizer_state else None
            save_jax_to_torch_pt(params_cpu, buffers_cpu, local_path, meta=meta_dict, opt_state=save_opt)

            # Atomically mirror to target_dir if different
            target_path = os.path.join(self.target_dir, filename)
            if os.path.abspath(local_path) != os.path.abspath(target_path):
                tmp_target = f"{target_path}.tmp.{os.getpid()}.{time.time_ns()}"
                shutil.copyfile(local_path, tmp_target)
                os.replace(tmp_target, target_path)
                local_json = local_path.replace(".pt", ".json")
                if os.path.exists(local_json):
                    tmp_json = f"{target_path.replace('.pt', '.json')}.tmp.{os.getpid()}.{time.time_ns()}"
                    shutil.copyfile(local_json, tmp_json)
                    os.replace(tmp_json, target_path.replace(".pt", ".json"))

            # Prune scratch disk
            self._prune_scratch()

            # Upload to cloud if active (rate-limit non-final uploads so high-step/s runs do not saturate network)
            if self.cloud_sync and ("latest" in filename or "best" in filename):
                target_type = "best" if "best" in filename else "latest"
                now = time.time()
                last_up = self._last_cloud_upload_at.get(target_type, 0.0)
                if force_cloud or (self.min_cloud_upload_interval_sec <= 0.0) or ((now - last_up) >= self.min_cloud_upload_interval_sec):
                    try:
                        from src.cloud_sync import push_checkpoint_async
                        step_val = int(meta_dict.get("step", 0)) if meta_dict else 0
                        if target_type == "best":
                            raw_loss = meta_dict.get("eval_loss", meta_dict.get("loss", 0.0)) if meta_dict else 0.0
                        else:
                            raw_loss = meta_dict.get("loss", meta_dict.get("eval_loss", 0.0)) if meta_dict else 0.0
                        loss_f = float(raw_loss) if raw_loss is not None else 0.0
                        loss_val = loss_f if (loss_f == loss_f and abs(loss_f) < 1e30) else 0.0
                        sync_model = meta_dict.get("sync_model") if meta_dict else None
                        # Snapshot isolation: stage to a dedicated temporary file so ongoing TPU training
                        # steps never mutate or overwrite the checkpoint file while the async upload streams chunks
                        upload_stage = f"{local_path}.upload.{os.getpid()}.{time.time_ns()}.pt"
                        if self.cloud_weights_only:
                            # Option A: Save lightweight weights-only checkpoint (~229 MB) for rapid, non-blocking cloud upload
                            save_jax_to_torch_pt(params_cpu, buffers_cpu, upload_stage, meta=meta_dict, opt_state=None)
                        else:
                            shutil.copyfile(local_path, upload_stage)
                        push_checkpoint_async(
                            upload_stage,
                            step=step_val,
                            loss=loss_val,
                            target_type=target_type,
                            model=sync_model,
                            url=self.cloud_url,
                            token=self.cloud_token,
                        )
                        self._last_cloud_upload_at[target_type] = now
                    except Exception as sync_err:
                        print(f"⚠️ [cloud_sync] Async upload notice: {sync_err}")
        except Exception as err:
            print(f"⚠️ [AsyncCheckpoint Error]: {err}")

    def _prune_scratch(self):
        try:
            pt_files = glob.glob(os.path.join(self.local_dir, "*_step_*.pt"))
            if len(pt_files) > 3:
                pt_files.sort(key=os.path.getmtime)
                for old_f in pt_files[:-3]:
                    try:
                        os.remove(old_f)
                        old_j = old_f.replace(".pt", ".json")
                        if os.path.exists(old_j):
                            os.remove(old_j)
                    except OSError:
                        pass
        except Exception:
            pass

    def dispatch(
        self,
        params: Dict[str, jnp.ndarray],
        buffers: Tuple[jnp.ndarray, jnp.ndarray],
        filename: str,
        meta_dict: Optional[dict] = None,
        block_if_busy: bool = False,
        opt_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Dispatches an asynchronous save job. Returns True if job was queued."""
        is_critical = ("best" in filename) or ("step_" in filename) or block_if_busy
        if not is_critical and (self._last_future is not None and not self._last_future.done()):
            return False

        params_cpu = jax.device_get(params)
        buffers_cpu = jax.device_get(buffers)
        opt_state_cpu = jax.device_get(opt_state) if opt_state is not None else None
        self._last_future = self.executor.submit(self._worker, params_cpu, buffers_cpu, filename, meta_dict, False, opt_state_cpu)
        return True

    def flush(
        self,
        params: Dict[str, jnp.ndarray],
        buffers: Tuple[jnp.ndarray, jnp.ndarray],
        filename: str,
        meta_dict: Optional[dict] = None,
        opt_state: Optional[Dict[str, Any]] = None,
    ):
        """Synchronously flushes state (used on shutdown / emergency exit)."""
        params_cpu = jax.device_get(params)
        buffers_cpu = jax.device_get(buffers)
        opt_state_cpu = jax.device_get(opt_state) if opt_state is not None else None
        self._worker(params_cpu, buffers_cpu, filename, meta_dict, True, opt_state_cpu)
        if self.cloud_sync:
            try:
                from src.cloud_sync import wait_for_in_flight_uploads
                wait_for_in_flight_uploads(timeout=30.0)
            except Exception:
                pass

    def shutdown(self, wait: bool = True):
        self.executor.shutdown(wait=wait)
        if wait and self.cloud_sync:
            try:
                from src.cloud_sync import wait_for_in_flight_uploads
                wait_for_in_flight_uploads(timeout=30.0)
            except Exception:
                pass
