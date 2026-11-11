"""
CerebroSpheroid V5: Pure JAX Auto-Tuning & Environment Benchmark Engine
========================================================================

Automatically tests the execution environment (Cloud TPU v5e-1, GPU, or CPU)
across candidate hyperparameters (batch sizes, remat/activation checkpointing)
to discover and configure the MAXIMUM tokens/second (tok/s) throughput while
pushing VRAM / HBM utilization to its optimal capacity (~85-92%) without leaving
memory on the table.

Key Features:
1. Deep Hardware Memory Profiling: Accurately identifies TPU, CUDA/ROCm GPU,
   and memory tiers (e.g. 16GB, 24GB, 40GB, 80GB).
2. Dynamic VRAM-Maximizing Search Ladder: Explores ascending batch sizes up to
   hardware saturation and compares remat=False vs remat=True.
3. Safe OOM Mitigation: Gracefully catches HbmOom / RESOURCE_EXHAUSTED / CUDA OOM
   and prunes invalid search branches without terminating execution.
4. Two-Way Remote Website Sync: Queries bioflo.in (sync.php?action=get_config)
   to load verified optimal configs in zero seconds. Once benchmarked, pushes the
   winning config back to bioflo.in (action=save_config) so you never waste time
   benchmarking again!
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from .v5_model import ModelConfigV5, PRESETS_V5, init_params_v5
from .spheroid_optimizer import init_spheroid_opt_state, build_train_step_spheroid

LOCAL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "hardware_configs.json")


def get_hardware_info() -> Dict[str, Any]:
    """
    Profiles active accelerator devices and determines memory capacity.
    """
    devices = jax.devices()
    dev = devices[0] if devices else None
    dev_kind = dev.device_kind if dev else "cpu"
    platform = dev.platform.lower() if dev else "cpu"

    is_tpu = "tpu" in platform or "tpu" in dev_kind.lower()
    is_gpu = "gpu" in platform or "cuda" in platform or "gpu" in dev_kind.lower() or "nvidia" in dev_kind.lower()

    total_memory_bytes: Optional[int] = None

    # Method 1: JAX device memory_stats API
    if dev and hasattr(dev, "memory_stats"):
        try:
            stats = dev.memory_stats()
            if stats and "bytes_limit" in stats and stats["bytes_limit"] > 0:
                total_memory_bytes = stats["bytes_limit"]
        except Exception:
            pass

    # Method 2: TPU known architectural specifications
    if is_tpu and not total_memory_bytes:
        k = dev_kind.lower()
        if "v4" in k:
            total_memory_bytes = 32 * (1024**3)
        elif "v5e" in k or "v5p" in k or "v5" in k or "v3" in k or "v2" in k:
            total_memory_bytes = 16 * (1024**3)
        else:
            total_memory_bytes = 16 * (1024**3)

    # Method 3: PyTorch CUDA probe for GPUs
    if is_gpu and not total_memory_bytes:
        try:
            import torch
            if torch.cuda.is_available():
                total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        except Exception:
            pass

    # Fallback conservative defaults
    if not total_memory_bytes:
        if is_tpu:
            total_memory_bytes = 16 * (1024**3)
        elif is_gpu:
            total_memory_bytes = 16 * (1024**3)
        else:
            total_memory_bytes = 8 * (1024**3)

    total_gb = total_memory_bytes / (1024**3)
    clean_kind = dev_kind.replace(" ", "_").replace("/", "_").replace("-", "_")
    sig = f"{clean_kind}_{int(round(total_gb))}GB"
    mem_type = "HBM" if is_tpu else ("VRAM" if is_gpu else "RAM")

    return {
        "device_kind": dev_kind,
        "platform": platform,
        "is_tpu": is_tpu,
        "is_gpu": is_gpu,
        "memory_type": mem_type,
        "total_memory_bytes": total_memory_bytes,
        "total_gb": total_gb,
        "hardware_signature": sig,
        "device_count": len(devices),
    }


def generate_vram_maximizing_candidates(
    preset: str,
    total_gb: float = 16.0,
    seq_len: int = 256,
) -> List[Dict[str, Any]]:
    """
    Generates a progressive search ladder designed to push hardware memory
    to its optimal capacity (~85-92% HBM on TPU / VRAM on GPU) without leaving free memory on the table.
    """
    if total_gb >= 32.0:
        # High-capacity accelerator: TPU v4 (32GB), A100/H100 (40GB/80GB)
        if preset in ("v5_small", "tpu_v5e_1", "v5_test"):
            base_ladder = [64, 128, 256, 384, 512]
        elif preset == "v5_compact":
            base_ladder = [32, 64, 128, 192, 256]
        else:  # v5_base
            base_ladder = [16, 32, 64, 96, 128]
    elif total_gb >= 12.0:
        # Standard datacenter accelerator: TPU v5e-1 (16GB), T4 (16GB), V100 (16GB), L4 (24GB), RTX 3090/4090 (24GB)
        if preset in ("v5_small", "tpu_v5e_1"):
            base_ladder = [32, 64, 128, 192, 256, 384]
        elif preset == "v5_compact":
            base_ladder = [16, 32, 64, 96, 128]
        elif preset == "v5_test":
            base_ladder = [8, 16, 32, 64]
        else:  # v5_base
            base_ladder = [8, 16, 32, 48, 64]
    elif total_gb >= 6.0:
        # Mid-tier workstation / gaming GPU: 8GB - 12GB
        if preset in ("v5_small", "tpu_v5e_1"):
            base_ladder = [16, 32, 64, 96]
        elif preset == "v5_compact":
            base_ladder = [8, 16, 32, 48]
        else:
            base_ladder = [4, 8, 16, 24]
    else:
        # Entry GPU (e.g. 4GB) or CPU
        base_ladder = [2, 4, 8]

    candidates = []
    for idx, b in enumerate(base_ladder):
        # Always test remat=True (memory-efficient baseline)
        candidates.append({"batch_size": b, "remat": True})
        # Test remat=False on lower batch sizes where memory allows
        # (remat=False avoids recomputation and often achieves 20-30% faster tok/s)
        if idx <= min(1, len(base_ladder) - 1):
            candidates.append({"batch_size": b, "remat": False})

    return candidates


def load_local_hardware_config(preset: str, device_signature: str) -> Optional[Dict[str, Any]]:
    """Loads cached configuration from local filesystem if present."""
    norm_preset = "v5_small" if preset in ("tpu_v5e_1", "tpu", "v5_tpu") else preset
    for path in [LOCAL_CONFIG_PATH, os.path.join("configs", "hardware_configs.json"), os.path.join("checkpoints", "hardware_configs.json")]:
        try:
            if os.path.exists(path) and os.path.getsize(path) > 2:
                with open(path, "r") as f:
                    data = json.load(f)
                    if norm_preset in data and device_signature in data[norm_preset]:
                        return data[norm_preset][device_signature]
        except Exception:
            pass
    return None


def save_local_hardware_config(preset: str, device_signature: str, config: Dict[str, Any]) -> None:
    """Saves verified configuration to local filesystem."""
    norm_preset = "v5_small" if preset in ("tpu_v5e_1", "tpu", "v5_tpu") else preset
    os.makedirs(os.path.dirname(os.path.abspath(LOCAL_CONFIG_PATH)), exist_ok=True)
    data = {}
    if os.path.exists(LOCAL_CONFIG_PATH) and os.path.getsize(LOCAL_CONFIG_PATH) > 2:
        try:
            with open(LOCAL_CONFIG_PATH, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

    if norm_preset not in data:
        data[norm_preset] = {}
    data[norm_preset][device_signature] = config

    try:
        with open(LOCAL_CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠️ [AutoTuner] Could not write local config cache: {e}")


def benchmark_config(
    preset: str = "v5_small",
    batch_size: int = 32,
    remat: bool = True,
    seq_len: int = 256,
    num_steps: int = 3,
    dtype: str = "auto",
) -> Dict[str, Any]:
    """
    Benchmarks a single training configuration and returns throughput & memory metrics.
    Gracefully catches OOM and compilation errors.
    """
    hw = get_hardware_info()
    is_tpu = hw["is_tpu"]
    is_gpu = hw["is_gpu"]
    total_mem_bytes = hw["total_memory_bytes"]

    # Select precision: BF16 on TPU and GPU with tensor cores
    if dtype == "auto":
        target_dtype = jnp.bfloat16 if (is_tpu or is_gpu) else jnp.float32
    elif dtype in ("bf16", "bfloat16"):
        target_dtype = jnp.bfloat16
    elif dtype in ("fp16", "float16"):
        target_dtype = jnp.float16
    else:
        target_dtype = jnp.float32

    cfg_dict = PRESETS_V5.get(preset, PRESETS_V5["v5_small"])
    config = ModelConfigV5(**cfg_dict)

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)

    try:
        # 1. Initialize parameters
        params, buffers = init_params_v5(init_key, config)
        if target_dtype != jnp.float32:
            params = {k: v.astype(target_dtype) if jnp.issubdtype(v.dtype, jnp.floating) else v for k, v in params.items()}

        opt_state = init_spheroid_opt_state(params)
        train_step_fn = build_train_step_spheroid(
            config=config,
            buffers=buffers,
            pad_id=0,
            weight_decay=1e-4,
            remat=remat,
        )

        # 2. Ahead-Of-Time XLA Compilation
        dummy_prompts = jnp.zeros((batch_size, 32), dtype=jnp.int32)
        dummy_targets = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

        t0_compile = time.time()
        params, opt_state, init_loss = train_step_fn(params, opt_state, dummy_prompts, dummy_targets, 5e-4)
        jax.block_until_ready(init_loss)
        compile_time = time.time() - t0_compile

        # 3. Micro-benchmark execution loop
        tokens_per_step = batch_size * seq_len
        step_times = []

        for step in range(num_steps):
            key, step_key = jax.random.split(key)
            prompt_b = jax.random.randint(step_key, (batch_size, 32), 0, config.vocab_size)
            target_b = jax.random.randint(step_key, (batch_size, seq_len), 0, config.vocab_size)

            t0_step = time.time()
            params, opt_state, loss = train_step_fn(params, opt_state, prompt_b, target_b, 5e-4)
            jax.block_until_ready(loss)
            step_times.append(time.time() - t0_step)

        avg_step_sec = sum(step_times) / max(len(step_times), 1)
        tok_per_sec = tokens_per_step / max(avg_step_sec, 1e-6)

        # 4. Memory Profiling
        dev = jax.devices()[0] if jax.devices() else None
        vram_gb = 0.0
        vram_utilization_pct = 0.0

        if dev and hasattr(dev, "memory_stats"):
            try:
                stats = dev.memory_stats()
                if stats and "peak_bytes_in_use" in stats:
                    peak_bytes = stats["peak_bytes_in_use"]
                    vram_gb = peak_bytes / (1024**3)
                    limit = stats.get("bytes_limit", total_mem_bytes)
                    if limit and limit > 0:
                        vram_utilization_pct = (peak_bytes / limit) * 100.0
            except Exception:
                pass

        if vram_gb == 0.0:
            # Analytical memory estimation
            param_bytes = sum(p.size * p.itemsize for p in params.values())
            # State + momentum
            opt_bytes = param_bytes * 2
            # Activation buffer
            act_mult = 4 if remat else (config.num_shells * 3)
            act_bytes = batch_size * seq_len * config.surface_dim * (2 if target_dtype == jnp.bfloat16 else 4) * act_mult
            est_total = param_bytes + opt_bytes + act_bytes
            vram_gb = est_total / (1024**3)
            if total_mem_bytes and total_mem_bytes > 0:
                vram_utilization_pct = min(99.0, (est_total / total_mem_bytes) * 100.0)

        mem_label = "HBM" if is_tpu else ("VRAM" if is_gpu else "RAM")
        return {
            "preset": preset,
            "batch_size": batch_size,
            "remat": remat,
            "seq_len": seq_len,
            "dtype": str(target_dtype).replace("jax.numpy.", ""),
            "compile_time": compile_time,
            "step_ms": avg_step_sec * 1000.0,
            "tok_per_sec": tok_per_sec,
            "memory_gb": vram_gb,
            "memory_utilization_pct": vram_utilization_pct,
            "memory_type": mem_label,
            "vram_gb": vram_gb,
            "vram_utilization_pct": vram_utilization_pct,
            "initial_loss": float(init_loss),
            "status": "PASS",
        }

    except Exception as e:
        err_msg = str(e)
        if "RESOURCE_EXHAUSTED" in err_msg or "HbmOom" in err_msg or "out of memory" in err_msg.lower():
            status_desc = "OOM (Memory Exceeded)"
        else:
            status_desc = f"FAIL ({type(e).__name__}: {err_msg[:40]})"

        mem_label = "HBM" if is_tpu else ("VRAM" if is_gpu else "RAM")
        return {
            "preset": preset,
            "batch_size": batch_size,
            "remat": remat,
            "seq_len": seq_len,
            "dtype": str(dtype),
            "compile_time": None,
            "step_ms": None,
            "tok_per_sec": 0.0,
            "memory_gb": 0.0,
            "memory_utilization_pct": 0.0,
            "memory_type": mem_label,
            "vram_gb": 0.0,
            "vram_utilization_pct": 0.0,
            "initial_loss": None,
            "status": status_desc,
        }


def auto_tune_environment(
    preset: str = "v5_small",
    candidates: Optional[List[Dict[str, Any]]] = None,
    seq_len: int = 256,
    dtype: str = "auto",
    use_cached_config: bool = True,
    force_tune: bool = False,
    upload_winner: bool = True,
    cloud_url: Optional[str] = None,
    cloud_token: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Explores candidate execution configurations and identifies the setting that
    maximizes token throughput on the active accelerator while utilizing free VRAM.

    Synchronizes with bioflo.in so optimal settings are permanently remembered.
    """
    hw = get_hardware_info()
    devices = jax.devices()
    dev_kind = hw["device_kind"]
    is_tpu = hw["is_tpu"]
    is_gpu = hw["is_gpu"]
    total_gb = hw["total_gb"]
    sig = hw["hardware_signature"]

    mem_type = hw.get("memory_type", "HBM" if is_tpu else ("VRAM" if is_gpu else "RAM"))

    # 1. First Priority: Check remote website bioflo.in or local cache to avoid wasting time
    if use_cached_config and not force_tune and candidates is None:
        # Check local cache
        cached_local = load_local_hardware_config(preset, sig)
        if cached_local and cached_local.get("tok_per_sec", 0) > 0:
            if verbose:
                print("=" * 80)
                print(f"⚡ [AutoTuner] Loaded verified local config for [{preset}] on [{sig}]:")
                print(f"   • Batch Size  : {cached_local['batch_size']}")
                print(f"   • Remat Scan  : {cached_local['remat']}")
                print(f"   • Peak Speed  : {cached_local.get('tok_per_sec', 0):,.0f} tok/s")
                m_gb = cached_local.get("memory_gb", cached_local.get("vram_gb", 0))
                m_pct = cached_local.get("memory_utilization_pct", cached_local.get("vram_utilization_pct", 0))
                print(f"   • {mem_type} Util   : {m_gb:.1f} GB ({m_pct:.1f}%)")
                print(f"⏩ Skipping micro-benchmarks and adopting verified parameters!\n" + "=" * 80)
            return cached_local

        # Check bioflo.in website
        try:
            from ..cloud_sync import fetch_remote_hardware_config
            cached_remote = fetch_remote_hardware_config(model=preset, device_signature=sig, url=cloud_url, token=cloud_token)
            if cached_remote and cached_remote.get("tok_per_sec", 0) > 0:
                if verbose:
                    print("=" * 80)
                    print(f"🌐 [AutoTuner] Retrieved verified optimal config from bioflo.in for [{preset}] on [{sig}]:")
                    print(f"   • Batch Size  : {cached_remote['batch_size']}")
                    print(f"   • Remat Scan  : {cached_remote['remat']}")
                    print(f"   • Peak Speed  : {cached_remote.get('tok_per_sec', 0):,.0f} tok/s")
                    m_gb = cached_remote.get("memory_gb", cached_remote.get("vram_gb", 0))
                    m_pct = cached_remote.get("memory_utilization_pct", cached_remote.get("vram_utilization_pct", 0))
                    print(f"   • {mem_type} Util   : {m_gb:.1f} GB ({m_pct:.1f}%)")
                    print(f"   • Verified At : {cached_remote.get('updated_at', 'recently')}")
                    print(f"⏩ Skipping micro-benchmarks and adopting verified parameters from website!\n" + "=" * 80)
                save_local_hardware_config(preset, sig, cached_remote)
                return cached_remote
        except Exception as e:
            if verbose:
                print(f"ℹ️ [AutoTuner] Remote config check notice: {e}")

    # 2. Benchmarking execution
    if verbose:
        print("=" * 80)
        print(f"🚀 CEREBROSPHEROID V5: AUTOMATIC THROUGHPUT BENCHMARK & {mem_type} AUTO-TUNER")
        print("=" * 80)
        print(f"Hardware Platform  : {len(devices)} x {dev_kind} ({sig})")
        print(f"Detected Memory    : {total_gb:.1f} GB Available {mem_type}")
        print(f"Target Preset      : {preset} (SeqLen: {seq_len})")
        print(f"Search Strategy    : Maximize {mem_type} utilization without OOM; test remat True vs False")
        print("-" * 80)

    if candidates is None:
        candidates = generate_vram_maximizing_candidates(preset=preset, total_gb=total_gb, seq_len=seq_len)

    results = []
    best_config = None
    max_tok_s = -1.0
    oom_remat_true_seen = False

    for idx, c in enumerate(candidates, 1):
        b = c["batch_size"]
        r = c.get("remat", True)

        # Prune larger batch sizes if smaller remat=True already OOMed
        if r and oom_remat_true_seen:
            continue

        if verbose:
            print(f"[{idx}/{len(candidates)}] Testing: Batch {b:3d} | Remat: {str(r):5s} ... ", end="", flush=True)

        res = benchmark_config(
            preset=preset,
            batch_size=b,
            remat=r,
            seq_len=seq_len,
            num_steps=3 if (is_tpu or is_gpu) else 2,
            dtype=dtype,
        )
        results.append(res)

        if res["status"] == "PASS":
            tok_s = res["tok_per_sec"]
            m_gb = res.get("memory_gb", res.get("vram_gb", 0))
            m_pct = res.get("memory_utilization_pct", res.get("vram_utilization_pct", 0))
            if verbose:
                print(f"✓ {tok_s:8,.0f} tok/s | {mem_type}: {m_gb:4.1f} GB ({m_pct:4.1f}%) | Latency: {res['step_ms']:6.1f}ms | Compile: {res['compile_time']:4.1f}s")
            if tok_s > max_tok_s:
                max_tok_s = tok_s
                best_config = res
        else:
            if "OOM" in res["status"] and r:
                oom_remat_true_seen = True
            if verbose:
                print(f"❌ {res['status']}")

    if verbose:
        print("\n" + "=" * 80)
        print(f"📊 AUTO-TUNING {mem_type} & THROUGHPUT BENCHMARK RESULTS")
        print("=" * 80)
        mem_hdr = f"{mem_type} Util"
        print(f"{'Preset':<10} | {'Batch':<6} | {'Remat':<7} | {'Compile (s)':<12} | {'Step (ms)':<10} | {mem_hdr:<15} | {'Throughput':<15} | {'Status':<10}")
        print("-" * 80)
        for r in results:
            c_str = f"{r['compile_time']:.1f}s" if r['compile_time'] is not None else "-"
            s_str = f"{r['step_ms']:.1f}ms" if r['step_ms'] is not None else "-"
            m_g = r.get('memory_gb', r.get('vram_gb', 0))
            m_p = r.get('memory_utilization_pct', r.get('vram_utilization_pct', 0))
            v_str = f"{m_g:.1f}GB ({m_p:.0f}%)" if m_g > 0 else "-"
            t_str = f"{r['tok_per_sec']:,.0f} tok/s" if r['tok_per_sec'] > 0 else "-"
            winner_mark = " 🏆 (WINNER)" if (best_config and r == best_config) else ""
            print(f"{r['preset']:<10} | {r['batch_size']:<6} | {str(r['remat']):<7} | {c_str:<12} | {s_str:<10} | {v_str:<15} | {t_str:<15} | {r['status'] + winner_mark:<10}")
        print("=" * 80)

    if best_config is None:
        best_config = {
            "preset": preset,
            "batch_size": 16 if (is_tpu or is_gpu) else 2,
            "remat": True,
            "seq_len": seq_len,
            "dtype": "bf16" if (is_tpu or is_gpu) else "fp32",
            "tok_per_sec": 0.0,
            "vram_gb": 0.0,
            "vram_utilization_pct": 0.0,
            "status": "DEFAULT_FALLBACK",
        }
        if verbose:
            print("⚠️ All benchmark candidates failed. Falling back to conservative defaults.")
    else:
        if verbose:
            print(f"🏆 OPTIMAL CONFIGURATION SELECTED FOR MAXIMUM THROUGHPUT & VRAM UTILIZATION:")
            print(f"   • Batch Size  : {best_config['batch_size']}")
            print(f"   • Remat Scan  : {best_config['remat']}")
            print(f"   • VRAM Footprint: {best_config['vram_gb']:.1f} GB ({best_config['vram_utilization_pct']:.1f}% of total {total_gb:.1f} GB)")
            print(f"   • Peak Speed  : {best_config['tok_per_sec']:,.0f} tokens/second")
            print("=" * 80)

        # 3. Cache locally
        save_local_hardware_config(preset, sig, best_config)

        # 4. Send config back to bioflo.in website so we don't waste time testing again
        if upload_winner:
            try:
                from ..cloud_sync import push_remote_hardware_config
                ok = push_remote_hardware_config(
                    model=preset,
                    device_signature=sig,
                    config=best_config,
                    url=cloud_url,
                    token=cloud_token,
                )
                if ok and verbose:
                    print(f"🌐 [AutoTuner] Winning configuration saved to website bioflo.in!")
                    print(f"   Subsequent runs on {sig} will immediately reuse this config.\n")
            except Exception as e:
                if verbose:
                    print(f"ℹ️ [AutoTuner] Website sync notice: {e}\n")

    return best_config


# Backward-compatible alias
run_auto_tuner = auto_tune_environment
