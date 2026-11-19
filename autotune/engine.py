"""
Core AutoTuner Engine & Execution Coordinator.
==============================================

Orchestrates accelerator discovery, architecture fingerprinting,
memory-ladder candidate exploration, safe OOM branch pruning, and
Pareto-optimal parameter selection with fast bisection & theoretical HBM bounds.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .objectives import TuneObjective
from .hardware import get_hardware_info, resolve_jax_compute_dtype
from .architecture import (
    ArchitectureDetector,
    ArchitectureInfo,
    extract_architecture_signature,
)
from .scoring import score_benchmark_result
from .search import generate_vram_maximizing_candidates, CandidateLadder
from .cache import (
    HardwareConfigCache,
    load_local_hardware_config,
    save_local_hardware_config,
)
from .benchmark import BenchmarkRunner
from .memory import estimate_peak_hbm


class AutoTuner:
    """
    High-Speed Modular Accelerator and Architecture Auto-Tuner.

    Features:
    - Zero-overhead cached retrieval (0ms) for previously analyzed architectures.
    - Automatic detection when a new model architecture or hardware topology is introduced.
    - Fast geometric bisection exploration with theoretical HBM OOM pre-check.
    - Memory-maximizing exploration across TPU and GPU tiers.
    - Multi-objective optimization: tokens/sec, steps/s, loss reduction/sec, balanced.
    """

    def __init__(
        self,
        objective: Union[str, TuneObjective] = TuneObjective.BALANCED,
        cache_manager: Optional[HardwareConfigCache] = None,
        benchmark_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        use_cached_config: bool = True,
        upload_winner: bool = True,
        cloud_url: Optional[str] = None,
        cloud_token: Optional[str] = None,
        verbose: bool = True,
        fast: bool = True,
        hardware_info: Optional[Union[HardwareInfo, Dict[str, Any]]] = None,
        get_hardware_info_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        self.objective = TuneObjective.from_str(objective)
        self.cache_manager = cache_manager or HardwareConfigCache()
        self.benchmark_runner = BenchmarkRunner(benchmark_fn)
        self.architecture_detector = ArchitectureDetector(self.cache_manager)
        self.use_cached_config = use_cached_config
        self.upload_winner = upload_winner
        self.cloud_url = cloud_url
        self.cloud_token = cloud_token
        self.verbose = verbose
        self.fast = fast
        self.hardware_info = hardware_info
        self.get_hardware_info_fn = get_hardware_info_fn or get_hardware_info

    def is_new_architecture(
        self,
        architecture: Any,
        hardware_sig: Optional[str] = None,
        seq_len: int = 256,
        max_batch_size: Optional[int] = None,
    ) -> bool:
        """Determines if the given architecture needs auto-tuning on this hardware."""
        arch_info = extract_architecture_signature(architecture, seq_len=seq_len)
        if hardware_sig is None:
            hw = self.hardware_info if self.hardware_info is not None else self.get_hardware_info_fn()
            hardware_sig = hw["hardware_signature"]
        return self.architecture_detector.is_new_architecture(
            arch_info=arch_info,
            hardware_sig=hardware_sig,
            seq_len=seq_len,
            max_batch_size=max_batch_size,
            tune_objective=self.objective.value,
        )

    def tune_if_new(
        self,
        architecture: Any,
        benchmark_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        seq_len: int = 256,
        prompt_len: int = 256,
        dtype: str = "auto",
        max_batch_size: Optional[int] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        force_tune: bool = False,
        fast: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Executes auto-tuning ONLY when a new architecture or unknown accelerator
        is detected. If already benchmarked and cached, returns optimal config in 0ms.
        """
        arch_info = extract_architecture_signature(
            architecture, seq_len=seq_len, prompt_len=prompt_len
        )
        hw = self.hardware_info if self.hardware_info is not None else self.get_hardware_info_fn()
        sig = hw["hardware_signature"]

        is_new = force_tune or self.architecture_detector.is_new_architecture(
            arch_info=arch_info,
            hardware_sig=sig,
            seq_len=seq_len,
            max_batch_size=max_batch_size,
            tune_objective=self.objective.value,
        )

        if not is_new and self.use_cached_config:
            seq_suffix = f"_seq{seq_len}" if seq_len != 256 else ""
            cap_suffix = f"_cap{int(max_batch_size)}" if max_batch_size is not None else ""
            cache_sig = f"{sig}{seq_suffix}{cap_suffix}"

            cached = self.cache_manager.load(arch_info.architecture_name, cache_sig)
            if cached is None and cache_sig != sig and max_batch_size is None:
                cached = self.cache_manager.load(arch_info.architecture_name, sig)

            if cached and cached.get("tok_per_sec", 0) > 0:
                if self.verbose:
                    print("=" * 80)
                    print(
                        f"⚡ [AutoTuner] Architecture [{arch_info.architecture_name}] already verified on [{cache_sig}]."
                    )
                    print(f"   • Batch Size   : {cached['batch_size']}")
                    print(f"   • Remat Scan   : {cached['remat']}")
                    print(f"   • Peak Speed   : {cached.get('tok_per_sec', 0):,.0f} tok/s")
                    print(
                        f"⏩ Skipping micro-benchmarks and adopting verified parameters!\n"
                        + "=" * 80
                    )
                return cached

        if self.verbose:
            print(f"🔍 [AutoTuner] New architecture or altered configuration detected: [{arch_info.architecture_name}]")
            print(f"   Initiating smart memory and throughput auto-tuning...")

        return self.tune(
            preset=arch_info.architecture_name,
            candidates=candidates,
            seq_len=seq_len,
            dtype=dtype,
            use_cached_config=self.use_cached_config,
            force_tune=force_tune,
            tune_objective=self.objective.value,
            prompt_len=prompt_len,
            max_batch_size=max_batch_size,
            benchmark_fn=benchmark_fn,
            fast=self.fast if fast is None else fast,
        )

    def _tune_bisection(
        self,
        preset: str,
        candidates: List[Dict[str, Any]],
        runner: BenchmarkRunner,
        ladder: CandidateLadder,
        dtype: str,
        seq_len: int,
        prompt_len: int,
        benchmark_steps: int,
        obj_name: str,
        verbose: bool,
        mem_type: str,
        **runner_kwargs,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Executes fast logarithmic geometric bisection search across candidate batch sizes.
        Reduces exploration complexity from O(N) to O(log N), finishing in 3-5 runs.
        """
        unique_batches = sorted(list(set(c["batch_size"] for c in candidates)))
        results: List[Dict[str, Any]] = []
        best_config: Optional[Dict[str, Any]] = None
        best_score = -1.0
        eval_cache: Dict[Tuple[int, bool], Dict[str, Any]] = {}

        def eval_batch(b: int, remat: bool = False) -> Dict[str, Any]:
            nonlocal best_config, best_score
            r = remat
            key = (b, r)
            if key in eval_cache:
                return eval_cache[key]

            if ladder.should_skip(b, r):
                skip_res = {
                    "preset": preset,
                    "batch_size": b,
                    "remat": r,
                    "tok_per_sec": 0.0,
                    "steps_per_sec": 0.0,
                    "step_ms": None,
                    "compile_time": None,
                    "memory_gb": 0.0,
                    "memory_utilization_pct": 0.0,
                    "status": "SKIPPED (Pruned by Ladder)",
                }
                eval_cache[key] = skip_res
                return skip_res

            if verbose:
                print(f"[Bisection Probe] Testing: Batch {b:3d} | Remat: {str(r):5s} ... ", end="", flush=True)

            res = runner.run(
                preset=preset,
                batch_size=b,
                remat=r,
                seq_len=seq_len,
                num_steps=benchmark_steps,
                dtype=dtype,
                prompt_len=prompt_len,
                **runner_kwargs,
            )
            results.append(res)
            eval_cache[key] = res
            ladder.record_result(b, r, res.get("status", "FAIL"), tok_per_sec=res.get("tok_per_sec", 0.0))

            if res.get("status") == "PASS":
                tok_s = res.get("tok_per_sec", 0.0)
                steps_s = res.get("steps_per_sec", 0.0)
                m_gb = res.get("memory_gb", res.get("vram_gb", 0))
                m_pct = res.get("memory_utilization_pct", res.get("vram_utilization_pct", 0))
                loss_v = res.get("loss_reduction_per_sec", 0.0)
                loss_str = f" | dLoss/s: {loss_v:6.4f}" if loss_v > 0 else ""

                if verbose:
                    print(
                        f"✓ {steps_s:5.2f} steps/s | {tok_s:8,.0f} tok/s | "
                        f"{mem_type}: {m_gb:4.1f} GB ({m_pct:4.1f}%){loss_str} | Latency: {res.get('step_ms', 0):6.1f}ms"
                    )
                score = score_benchmark_result(res, obj_name)
                if score > best_score:
                    best_score = score
                    best_config = res
            else:
                if verbose:
                    print(f"❌ {res.get('status')}")

            return res

        if not unique_batches:
            return results, best_config

        # 1. Anchor probe: test the median batch size with remat=False
        low = 0
        high = len(unique_batches) - 1
        mid = (low + high) // 2
        anchor_batch = unique_batches[mid]

        res_anchor = eval_batch(anchor_batch, remat=False)

        if res_anchor.get("status") == "PASS":
            best_pass_idx = mid
            # Probe highest candidate to find upper bound
            res_high = eval_batch(unique_batches[high], remat=False)
            if res_high.get("status") == "PASS":
                best_pass_idx = high
            else:
                # Bracket is [mid, high]: bisect between them
                l, r = mid, high
                while r - l > 1:
                    m = (l + r) // 2
                    res_m = eval_batch(unique_batches[m], remat=False)
                    if res_m.get("status") == "PASS":
                        l = m
                        best_pass_idx = m
                    else:
                        r = m
        else:
            # Anchor OOMed! Bisect downward between low and mid
            best_pass_idx = None
            l, r = low, mid
            while r - l > 1:
                m = (l + r) // 2
                res_m = eval_batch(unique_batches[m], remat=False)
                if res_m.get("status") == "PASS":
                    l = m
                    best_pass_idx = m
                else:
                    r = m
            if best_pass_idx is None:
                res_low = eval_batch(unique_batches[low], remat=False)
                if res_low.get("status") == "PASS":
                    best_pass_idx = low

        # 2. Check activation rematerialization (remat=True)
        # Test if next larger batch fits with activation checkpointing and beats best score
        if best_pass_idx is not None and best_pass_idx + 1 < len(unique_batches):
            next_b = unique_batches[best_pass_idx + 1]
            res_remat = eval_batch(next_b, remat=True)
            if res_remat.get("status") == "PASS":
                curr_idx = best_pass_idx + 1
                while curr_idx + 1 < len(unique_batches):
                    higher_b = unique_batches[curr_idx + 1]
                    res_h = eval_batch(higher_b, remat=True)
                    if res_h.get("status") == "PASS":
                        curr_idx += 1
                    else:
                        break
        elif best_pass_idx is None:
            # If all remat=False OOMed, bisect with remat=True
            l, r = low, high
            while l <= r:
                m = (l + r) // 2
                res_m = eval_batch(unique_batches[m], remat=True)
                if res_m.get("status") == "PASS":
                    l = m + 1
                else:
                    r = m - 1

        return results, best_config

    def tune(
        self,
        preset: str = "v5_small",
        candidates: Optional[List[Dict[str, Any]]] = None,
        seq_len: int = 256,
        dtype: str = "auto",
        use_cached_config: bool = True,
        force_tune: bool = False,
        upload_winner: Optional[bool] = None,
        cloud_url: Optional[str] = None,
        cloud_token: Optional[str] = None,
        verbose: Optional[bool] = None,
        tune_objective: Optional[str] = None,
        prompt_len: int = 256,
        max_batch_size: Optional[int] = None,
        benchmark_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        fast: Optional[bool] = None,
        num_steps: Optional[int] = None,
        strategy: Optional[str] = None,
        hardware_info: Optional[Union[HardwareInfo, Dict[str, Any]]] = None,
        worst_case_seq_len: Optional[int] = None,
        model_config: Optional[Any] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        num_devices: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Explores candidate configurations and selects the optimal hyperparameter settings."""
        if verbose is None:
            verbose = self.verbose
        if upload_winner is None:
            upload_winner = self.upload_winner
        if cloud_url is None:
            cloud_url = self.cloud_url
        if cloud_token is None:
            cloud_token = self.cloud_token

        # Default strategy: bisection when fast=True and candidates is None
        is_custom_candidates = candidates is not None
        if fast is None:
            fast = self.fast if not is_custom_candidates else False
        if strategy is None:
            strategy = "bisection" if (fast and not is_custom_candidates) else "linear"

        runner_kwargs = {
            "worst_case_seq_len": worst_case_seq_len,
            "model_config": model_config,
            "config_overrides": config_overrides,
            "num_devices": num_devices or 1,
            **kwargs,
        }

        obj_enum = TuneObjective.from_str(tune_objective or self.objective)
        obj_name = obj_enum.value

        hw = hardware_info or (self.hardware_info if self.hardware_info is not None else self.get_hardware_info_fn())
        dev_kind = hw["device_kind"]
        is_tpu = hw.get("is_tpu", False)
        is_gpu = hw.get("is_gpu", False)
        total_gb = hw["total_gb"]
        sig = hw["hardware_signature"]
        eff_devs = int(num_devices) if num_devices and num_devices > 0 else int(hw.get("device_count", 1))
        dev_suffix = f"_dev{eff_devs}" if eff_devs > 1 else ""
        seq_suffix = f"_seq{seq_len}" if seq_len != 256 else ""
        cap_suffix = f"_cap{int(max_batch_size)}" if max_batch_size is not None else ""
        cache_sig = f"{sig}{seq_suffix}{cap_suffix}{dev_suffix}"
        mem_type = hw.get("memory_type", "HBM" if is_tpu else ("VRAM" if is_gpu else "RAM"))

        runner = BenchmarkRunner(benchmark_fn) if benchmark_fn else self.benchmark_runner

        def _is_cache_safe(cfg: Optional[Dict[str, Any]]) -> bool:
            if not cfg or cfg.get("tok_per_sec", 0) <= 0:
                return False
            if cfg.get("seq_len", 256) != seq_len:
                return False
            if max_batch_size is not None and int(cfg.get("batch_size", 0)) > int(max_batch_size):
                return False
            if cfg.get("num_devices", 1) != eff_devs:
                return False
            cfg_obj = cfg.get("tune_objective")
            if cfg_obj is not None and cfg_obj != obj_name:
                return False
            if worst_case_seq_len and worst_case_seq_len > seq_len:
                est = estimate_peak_hbm(
                    param_bytes=212_000_000,
                    batch_size=cfg["batch_size"],
                    seq_len=worst_case_seq_len,
                    prompt_len=prompt_len,
                    remat=cfg.get("remat", True),
                    total_memory_bytes=hw.get("total_memory_bytes"),
                    num_devices=eff_devs,
                    vocab_size=getattr(model_config, "vocab_size", 32000),
                )
                if est.predicted_oom:
                    if verbose:
                        print(f"⚠️ [AutoTuner Cache Guard] Cached candidate (B={cfg['batch_size']}, remat={cfg.get('remat')}) predicts OOM at worst_case_seq_len={worst_case_seq_len}. Re-benchmarking safe ladder.")
                    return False
            return True

        # 1. First Priority: Check cached configs if caching is enabled
        if use_cached_config and not force_tune and candidates is None:
            cached_local = self.cache_manager.load(preset, cache_sig)
            if cached_local is None and cap_suffix:
                fallback_local = self.cache_manager.load(preset, f"{sig}{seq_suffix}{dev_suffix}")
                if fallback_local and _is_cache_safe(fallback_local):
                    cached_local = fallback_local
            if cached_local is None and cache_sig != sig and max_batch_size is None and eff_devs == 1:
                fallback_local = self.cache_manager.load(preset, sig)
                if fallback_local and _is_cache_safe(fallback_local):
                    cached_local = fallback_local

            if _is_cache_safe(cached_local):
                if verbose:
                    steps_s = cached_local.get("steps_per_sec") or (
                        1000.0 / cached_local["step_ms"] if cached_local.get("step_ms") else 0.0
                    )
                    print("=" * 80)
                    print(f"⚡ [AutoTuner] Loaded verified local config for [{preset}] on [{cache_sig}]:")
                    print(f"   • Batch Size  : {cached_local['batch_size']}")
                    print(f"   • Remat Scan  : {cached_local['remat']}")
                    print(f"   • Step Rate   : {steps_s:.2f} steps/s ({cached_local.get('step_ms', 0):.1f} ms)")
                    print(f"   • Peak Speed  : {cached_local.get('tok_per_sec', 0):,.0f} tok/s")
                    m_gb = cached_local.get("memory_gb", cached_local.get("vram_gb", 0))
                    m_pct = cached_local.get("memory_utilization_pct", cached_local.get("vram_utilization_pct", 0))
                    print(f"   • {mem_type} Util   : {m_gb:.1f} GB ({m_pct:.1f}%)")
                    print(f"⏩ Skipping micro-benchmarks and adopting verified parameters!\n" + "=" * 80)
                return cached_local

            # Optional remote sync (bioflo.in)
            try:
                from src.cloud_sync import fetch_remote_hardware_config
                cached_remote = fetch_remote_hardware_config(
                    model=preset, device_signature=cache_sig, url=cloud_url, token=cloud_token
                )
                if cached_remote is None and cap_suffix:
                    cached_remote = fetch_remote_hardware_config(
                        model=preset, device_signature=f"{sig}{seq_suffix}{dev_suffix}", url=cloud_url, token=cloud_token
                    )
                if cached_remote is None and cache_sig != sig and eff_devs == 1:
                    fallback_remote = fetch_remote_hardware_config(
                        model=preset, device_signature=sig, url=cloud_url, token=cloud_token
                    )
                    if fallback_remote and _is_cache_safe(fallback_remote):
                        cached_remote = fallback_remote

                if _is_cache_safe(cached_remote):
                    if verbose:
                        print("=" * 80)
                        print(f"🌐 [AutoTuner] Retrieved verified optimal config from bioflo.in for [{preset}] on [{cache_sig}]:")
                        print(f"   • Batch Size  : {cached_remote['batch_size']}")
                        print(f"   • Remat Scan  : {cached_remote['remat']}")
                        print(f"   • Peak Speed  : {cached_remote.get('tok_per_sec', 0):,.0f} tok/s")
                        m_gb = cached_remote.get("memory_gb", cached_remote.get("vram_gb", 0))
                        m_pct = cached_remote.get("memory_utilization_pct", cached_remote.get("vram_utilization_pct", 0))
                        print(f"   • {mem_type} Util   : {m_gb:.1f} GB ({m_pct:.1f}%)")
                        print(f"⏩ Skipping micro-benchmarks and adopting verified parameters from website!\n" + "=" * 80)
                    self.cache_manager.save(preset, cache_sig, cached_remote)
                    return cached_remote
            except Exception as e:
                if verbose:
                    print(f"ℹ️ [AutoTuner] Remote config check notice: {e}")

        # 2. Benchmarking execution
        speed_tag = " [GEOMETRIC BISECTION SEARCH]" if strategy == "bisection" else ""
        if verbose:
            print("=" * 80)
            print(f"🚀 AUTOTUNER: AUTOMATIC {obj_enum.label.upper()} & {mem_type} OPTIMIZATION{speed_tag}")
            print("=" * 80)
            print(f"Hardware Platform  : {hw.get('device_count', 1)} x {dev_kind} ({sig})")
            print(f"Detected Memory    : {total_gb:.1f} GB Available {mem_type}")
            cap_str = f", MaxBatch: {max_batch_size}" if max_batch_size is not None else ""
            print(f"Target Preset/Arch : {preset} (PromptLen: {prompt_len}, SeqLen: {seq_len}{cap_str})")
            print(f"Tune Objective     : {obj_enum.label}")
            print("-" * 80)

        # Smart candidate ladder generation: on >=16GB accelerators, start at 32 or 64
        min_batch = None
        if total_gb >= 16.0 and max_batch_size is None:
            min_batch = 32 if total_gb < 64.0 else 64
        elif total_gb >= 12.0 and max_batch_size is None:
            min_batch = 16

        if candidates is None:
            target_ladder_seq = worst_case_seq_len if (worst_case_seq_len and worst_case_seq_len > seq_len) else seq_len
            candidates = generate_vram_maximizing_candidates(
                preset=preset,
                total_gb=total_gb,
                seq_len=target_ladder_seq,
                max_batch_size=max_batch_size,
                min_batch_size=min_batch,
                fast=fast,
            )

        ladder = CandidateLadder(
            candidates,
            enable_plateau_pruning=fast,
            plateau_tolerance=2,
        )

        benchmark_steps = (
            num_steps if num_steps is not None else (2 if (is_tpu or is_gpu) else 1)
        )

        # Execution mode: Bisection vs Linear
        if strategy == "bisection" and not is_custom_candidates:
            results, best_config = self._tune_bisection(
                preset=preset,
                candidates=candidates,
                runner=runner,
                ladder=ladder,
                dtype=dtype,
                seq_len=seq_len,
                prompt_len=prompt_len,
                benchmark_steps=benchmark_steps,
                obj_name=obj_name,
                verbose=verbose,
                mem_type=mem_type,
                **runner_kwargs,
            )
        else:
            results = []
            best_config = None
            best_score = -1.0

            for idx, c in enumerate(candidates, 1):
                b = c["batch_size"]
                r = c.get("remat", True)

                # Smart branch pruning & plateau early-stopping
                if ladder.should_skip(b, r):
                    continue

                if verbose:
                    print(f"[{idx}/{len(candidates)}] Testing: Batch {b:3d} | Remat: {str(r):5s} ... ", end="", flush=True)

                res = runner.run(
                    preset=preset,
                    batch_size=b,
                    remat=r,
                    seq_len=seq_len,
                    num_steps=benchmark_steps,
                    dtype=dtype,
                    prompt_len=prompt_len,
                    **runner_kwargs,
                )
                results.append(res)
                ladder.record_result(
                    b, r, res.get("status", "FAIL"), tok_per_sec=res.get("tok_per_sec", 0.0)
                )

                if res.get("status") == "PASS":
                    tok_s = res.get("tok_per_sec", 0.0)
                    steps_s = res.get("steps_per_sec", 0.0)
                    m_gb = res.get("memory_gb", res.get("vram_gb", 0))
                    m_pct = res.get("memory_utilization_pct", res.get("vram_utilization_pct", 0))
                    loss_v = res.get("loss_reduction_per_sec", 0.0)
                    loss_str = f" | dLoss/s: {loss_v:6.4f}" if loss_v > 0 else ""

                    if verbose:
                        print(
                            f"✓ {steps_s:5.2f} steps/s | {tok_s:8,.0f} tok/s | "
                            f"{mem_type}: {m_gb:4.1f} GB ({m_pct:4.1f}%){loss_str} | Latency: {res.get('step_ms', 0):6.1f}ms"
                        )
                    score = score_benchmark_result(res, obj_name)
                    if score > best_score:
                        best_score = score
                        best_config = res
                else:
                    if verbose:
                        print(f"❌ {res.get('status')}")

                if ladder.stopped_due_to_plateau and verbose:
                    print(
                        f"🛑 [AutoTuner] Peak throughput reached ({ladder.peak_throughput:,.0f} tok/s at Batch {ladder.peak_batch}). "
                        f"Early-stopping search ladder to save time."
                    )
                    break

        if verbose:
            print("\n" + "=" * 92)
            print(f"📊 AUTO-TUNING {mem_type}, STEPS/S & THROUGHPUT BENCHMARK RESULTS")
            print("=" * 92)
            mem_hdr = f"{mem_type} Util"
            print(f"{'Preset':<10} | {'Batch':<5} | {'Remat':<5} | {'Compile':<8} | {'Step':<8} | {'Steps/s':<7} | {mem_hdr:<13} | {'Throughput':<14} | {'Status'}")
            print("-" * 92)
            for r in results:
                c_str = f"{r['compile_time']:.1f}s" if r.get('compile_time') is not None else "-"
                s_str = f"{r['step_ms']:.1f}ms" if r.get('step_ms') is not None else "-"
                st_s = f"{r.get('steps_per_sec', 0.0):.2f}/s" if r.get('steps_per_sec', 0.0) > 0 else "-"
                m_g = r.get('memory_gb', r.get('vram_gb', 0))
                m_p = r.get('memory_utilization_pct', r.get('vram_utilization_pct', 0))
                v_str = f"{m_g:.1f}GB ({m_p:.0f}%)" if m_g > 0 else "-"
                t_str = f"{r.get('tok_per_sec', 0):,.0f} tok/s" if r.get('tok_per_sec', 0) > 0 else "-"
                winner_mark = " 🏆 (WINNER)" if (best_config and r == best_config) else ""
                print(f"{r['preset']:<10} | {r['batch_size']:<5} | {str(r['remat']):<5} | {c_str:<8} | {s_str:<8} | {st_s:<7} | {v_str:<13} | {t_str:<14} | {r['status'] + winner_mark}")
            print("=" * 92)

        if best_config is None:
            fb_dtype, _ = resolve_jax_compute_dtype(dtype)
            best_config = {
                "preset": preset,
                "batch_size": 16 if (is_tpu or is_gpu) else 2,
                "remat": True,
                "seq_len": seq_len,
                "prompt_len": prompt_len,
                "dtype": "bf16" if str(fb_dtype).endswith("bfloat16") else "fp32",
                "steps_per_sec": 0.0,
                "tok_per_sec": 0.0,
                "vram_gb": 0.0,
                "vram_utilization_pct": 0.0,
                "tune_objective": obj_name,
                "tuner_version": 2,
                "status": "DEFAULT_FALLBACK",
            }
            if verbose:
                print("⚠️ All benchmark candidates failed. Falling back to conservative defaults.")
        else:
            best_config["tune_objective"] = obj_name
            best_config["max_batch_size"] = max_batch_size
            if worst_case_seq_len is not None:
                best_config["worst_case_seq_len"] = worst_case_seq_len
            best_config["tuner_version"] = 2
            if verbose:
                print(f"🏆 OPTIMAL CONFIGURATION SELECTED ({obj_name.upper()}):")
                print(f"   • Batch Size  : {best_config['batch_size']}")
                print(f"   • Remat Scan  : {best_config['remat']}")
                print(f"   • Step Rate   : {best_config.get('steps_per_sec', 0.0):.2f} steps/second ({best_config.get('step_ms', 0):.1f} ms)")
                print(f"   • Peak Speed  : {best_config.get('tok_per_sec', 0):,.0f} tokens/second")
                m_gb = best_config.get("memory_gb", best_config.get("vram_gb", 0))
                m_pct = best_config.get("memory_utilization_pct", best_config.get("vram_utilization_pct", 0))
                print(f"   • {mem_type} Util   : {m_gb:.1f} GB ({m_pct:.1f}% of {total_gb:.1f} GB)")
                if best_config.get("loss_reduction_per_sec", 0) > 0:
                    print(f"   • dLoss/sec   : {best_config['loss_reduction_per_sec']:.6f}")
                print("=" * 92)

            # Save locally
            best_config["num_devices"] = eff_devs
            if worst_case_seq_len:
                best_config["worst_case_seq_len"] = worst_case_seq_len
            self.cache_manager.save(preset, cache_sig, best_config)

            # Push remote if configured
            if upload_winner:
                try:
                    from src.cloud_sync import push_remote_hardware_config
                    ok = push_remote_hardware_config(
                        model=preset,
                        device_signature=cache_sig,
                        config=best_config,
                        url=cloud_url,
                        token=cloud_token,
                    )
                    if ok and verbose:
                        print(f"🌐 [AutoTuner] Winning configuration saved to website bioflo.in!\n")
                except Exception as e:
                    if verbose:
                        print(f"ℹ️ [AutoTuner] Website sync notice: {e}\n")

        return best_config


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
    tune_objective: str = "balanced",
    prompt_len: int = 256,
    max_batch_size: Optional[int] = None,
    fast: bool = True,
    num_steps: Optional[int] = None,
    strategy: Optional[str] = None,
    hardware_info: Optional[Union[HardwareInfo, Dict[str, Any]]] = None,
    worst_case_seq_len: Optional[int] = None,
    model_config: Optional[Any] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    num_devices: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Functional convenience entrypoint matching legacy CerebroSpheroid auto_tune_environment."""
    tuner = AutoTuner(
        objective=tune_objective,
        use_cached_config=use_cached_config,
        upload_winner=upload_winner,
        cloud_url=cloud_url,
        cloud_token=cloud_token,
        verbose=verbose,
        fast=fast,
    )
    return tuner.tune(
        preset=preset,
        candidates=candidates,
        seq_len=seq_len,
        dtype=dtype,
        use_cached_config=use_cached_config,
        force_tune=force_tune,
        tune_objective=tune_objective,
        prompt_len=prompt_len,
        max_batch_size=max_batch_size,
        fast=fast,
        num_steps=num_steps,
        strategy=strategy,
        hardware_info=hardware_info,
        worst_case_seq_len=worst_case_seq_len,
        model_config=model_config,
        config_overrides=config_overrides,
        num_devices=num_devices,
        **kwargs,
    )


# Backward-compatible alias
run_auto_tuner = auto_tune_environment
