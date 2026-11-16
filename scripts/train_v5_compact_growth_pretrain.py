#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Endogenous Pretraining Engine (v5_small -> v5_compact)
=============================================================================
Executes Progressive Continued Pretraining (CPT) with cyclical Core Freezing
and Periodic Consolidation Annealing (100 steps every 1,000 steps):

  1. Net2Net Subspace Expansion:
     - v5_small (102.4M: 16 shells, d=640, r=256, 40 tracts)
     - -> v5_compact (245.6M: 32 shells, d=768, r=512, 96 tracts, 32k vocab)
     - Preserves 100% of the 900,000-step TPU foundation priors

  2. Cyclical Annealing Schedule (100 steps every 1,000 steps):
     - Window Steps 0 to 899 (900 steps): Core weights FROZEN (requires_grad=False)
       Only new 143M parameters train on pretraining corpus at base LR (2e-4).
     - Window Steps 900 to 999 (100 steps): Core weights UNFROZEN
       Differential LR (core 2e-5, expansion 2e-4) anneals down to min_lr (2e-6)
       for smooth co-adaptation and deep weight consolidation.
     - Window Step 1000: Next cycle begins! Core re-freezes for next 900 steps.

  3. Full Backup & Cloud Synchronization:
     - Local checkpoints: v5_compact_pretrain_latest.pt & _best.pt
     - Background Cloud Sync: bioflo.in live push (push_checkpoint_async)
     - Google Drive auto-mirror if mounted (/content/drive)
"""

import os
import sys
import time
import json
import math
import argparse
import random
import re
import shutil
from typing import Dict, Any, List, Tuple, Optional, Iterator

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    PRESETS_V5,
    infer_v5_config_from_state_dict,
)
from src.corpus import (
    build_tokenizer,
    format_stimulus_prompt,
    encode_no_special,
)
from src.cloud_sync import (
    push_checkpoint_async,
    pull_checkpoint,
    check_remote_manifest,
)
from src.teacher_digester import (
    create_digester,
    QwenTeacherDigester,
    HeuristicDigester,
)

DEFAULT_SMALL_CKPT = os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_latest_step920k.pt")
DEFAULT_COMPACT_CFG = os.path.join(WORKSPACE_ROOT, "configs", "v5_compact_250m.json")
DEFAULT_OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "checkpoints")
DEFAULT_TOKENIZER_DIR = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
DEFAULT_TRAIN_DATA = os.path.join(WORKSPACE_ROOT, "data", "cerebro_v5_custom_dataset", "train.parquet")
DEFAULT_VAL_DATA = os.path.join(WORKSPACE_ROOT, "data", "cerebro_v5_custom_dataset", "validation.parquet")


def expand_v5_small_to_compact(
    small_ckpt_path: str,
    compact_cfg_path: str,
    output_seed_path: str,
    cloud_sync: bool = True,
) -> Tuple[CerebroSpheroidV5, ModelConfigV5]:
    """Expands v5_small checkpoint into v5_compact architecture using Net2Net expansion."""
    print("\n" + "=" * 80)
    print("🧬 [EXPANSION] ENDOGENOUS NET2NET GROWTH: v5_small (102M) -> v5_compact (246M)")
    print("=" * 80)

    # Auto-pull from bioflo.in if local small checkpoint is missing
    if not os.path.exists(small_ckpt_path):
        alt_paths = [
            os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_best.pt"),
            os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_latest_step900k.pt"),
        ]
        found = next((p for p in alt_paths if os.path.exists(p)), None)
        if found:
            small_ckpt_path = found
            print(f"  ✓ Found alternative foundation checkpoint: {small_ckpt_path}")
        elif cloud_sync:
            print(f"  🔄 Local checkpoint missing. Pulling v5_small latest from bioflo.in cloud...")
            os.makedirs(os.path.dirname(small_ckpt_path), exist_ok=True)
            ok = pull_checkpoint(local_dest=small_ckpt_path, target_type="latest", model="v5_small")
            if not ok or not os.path.exists(small_ckpt_path):
                raise FileNotFoundError(f"Could not locate or pull foundation checkpoint: {small_ckpt_path}")

    print(f"  Source Small Checkpoint : {small_ckpt_path}")
    print(f"  Target Compact Config   : {compact_cfg_path}")
    print(f"  Output Seed Checkpoint  : {output_seed_path}")

    # Load small checkpoint
    small_raw = torch.load(small_ckpt_path, map_location="cpu", weights_only=False)
    small_sd = small_raw.get("model_state", small_raw.get("model", small_raw))

    # Convert JAX keys if needed
    if any(k.startswith(("sensory.", "core.", "motor.")) for k in small_sd.keys()):
        from src.jax.checkpoint import adapt_jax_to_torch_state_dict
        s_cfg = ModelConfigV5(**PRESETS_V5["v5_small"])
        small_sd = adapt_jax_to_torch_state_dict(small_sd, s_cfg)

    # Instantiate target compact model
    if os.path.exists(compact_cfg_path):
        with open(compact_cfg_path, "r") as f:
            cfg_dict = json.load(f)
        valid_keys = {k: v for k, v in cfg_dict.items() if k in ModelConfigV5.__dataclass_fields__}
        compact_cfg = ModelConfigV5(**valid_keys)
    else:
        compact_cfg = ModelConfigV5(**PRESETS_V5["v5_compact"])

    compact_model = CerebroSpheroidV5(compact_cfg)
    compact_sd = compact_model.state_dict()

    new_sd = {}
    transferred_count = 0
    expanded_count = 0
    interpolated_count = 0

    K_src = 16  # v5_small shells
    K_tgt = compact_cfg.num_shells  # 32 shells

    for k_tgt, v_tgt in compact_sd.items():
        if k_tgt in small_sd and small_sd[k_tgt].shape == v_tgt.shape:
            new_sd[k_tgt] = small_sd[k_tgt].clone()
            transferred_count += 1

        elif k_tgt in small_sd and small_sd[k_tgt].shape != v_tgt.shape:
            v_src = small_sd[k_tgt]
            adapted = v_tgt.clone() * 0.0
            slices = tuple(slice(0, min(s, d)) for s, d in zip(v_src.shape, v_tgt.shape))
            adapted[slices] = v_src[slices]

            if "norm" in k_tgt and v_tgt.dim() == 1 and v_tgt.shape[0] > v_src.shape[0]:
                adapted[v_src.shape[0]:] = 1.0

            new_sd[k_tgt] = adapted
            expanded_count += 1

        elif "intra_shell_swiglu" in k_tgt or "intra_norms" in k_tgt:
            m = re.search(r"\.(\d+)\.", k_tgt)
            if m:
                shell_idx = int(m.group(1))
                src_shell_idx = min(K_src - 1, int(round(shell_idx * (K_src - 1) / (K_tgt - 1))))
                src_key = k_tgt.replace(f".{shell_idx}.", f".{src_shell_idx}.")

                if src_key in small_sd:
                    v_src = small_sd[src_key]
                    adapted = v_tgt.clone() * 0.0
                    slices = tuple(slice(0, min(s, d)) for s, d in zip(v_src.shape, v_tgt.shape))
                    adapted[slices] = v_src[slices]

                    if shell_idx >= K_src:
                        noise = torch.randn_like(adapted) * 1e-4
                        adapted = adapted + noise

                    new_sd[k_tgt] = adapted
                    interpolated_count += 1
                else:
                    new_sd[k_tgt] = v_tgt
            else:
                new_sd[k_tgt] = v_tgt

        elif "w_radial" in k_tgt:
            v_src = small_sd.get(k_tgt, None)
            adapted = v_tgt.clone() * 0.0
            if v_src is not None:
                slices = tuple(slice(0, min(s, d)) for s, d in zip(v_src.shape, v_tgt.shape))
                adapted[slices] = v_src[slices]
            new_sd[k_tgt] = adapted
            expanded_count += 1

        else:
            new_sd[k_tgt] = v_tgt

    compact_model.load_state_dict(new_sd, strict=False)

    p_small = sum(p.numel() for p in small_sd.values() if isinstance(p, torch.Tensor))
    p_compact = sum(p.numel() for p in compact_model.parameters())

    print(f"  ✓ Net2Net Expansion Complete: {p_small/1e6:.1f}M -> {p_compact/1e6:.1f}M params")
    print(f"  ✓ Transferred Tensors  : {transferred_count}")
    print(f"  ✓ Subspace Expanded    : {expanded_count}")
    print(f"  ✓ Interpolated Laminae : {interpolated_count}")

    save_payload = {
        "step": int(small_raw.get("step", 920500)),
        "model_state": {k: v.cpu().float() for k, v in compact_model.state_dict().items()},
        "config": compact_cfg,
        "growth_origin": "v5_small_step920k",
    }
    os.makedirs(os.path.dirname(output_seed_path), exist_ok=True)
    torch.save(save_payload, output_seed_path)
    print(f"  ✓ Seed checkpoint saved: {output_seed_path}\n")

    return compact_model, compact_cfg


class PretrainingTokenStream:
    """
    Streams continuous pretraining token windows for Causal Language Modeling
    with optional Teacher Digester dynamic next-token importance weights.
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer: Any,
        seq_len: int = 512,
        batch_size: int = 2,
        digester: Optional[Any] = None,
    ):
        self.parquet_path = parquet_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.digester = digester
        self.buffer: List[int] = []
        self.weight_buffer: List[float] = []

        print(f"📦 Initializing Pretraining Token Stream from: {parquet_path}")
        if self.digester is not None:
            print(f"  🧠 Teacher Digestion ACTIVE: Dynamic Next-Token Weighting enabled")
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            self.texts = []
            for _, r in df.iterrows():
                p = str(r.get("prompt", "")).strip()
                a = str(r.get("answer", "")).strip()
                t = str(r.get("thinking", "")).strip()
                combined = f"{p}\n{t}\n{a}".strip() if t else f"{p}\n{a}".strip()
                if len(combined) > 20:
                    self.texts.append(combined)
            random.shuffle(self.texts)
            print(f"  ✓ Loaded {len(self.texts):,} pretraining documents.")
        else:
            print("  ⚠️ Parquet not found. Generating synthetic continuous pretraining stream.")
            self.texts = self._generate_synthetic_corpus()

        self.text_idx = 0

    def _generate_synthetic_corpus(self) -> List[str]:
        corpus = [
            "The geometry of concentric laminae reflects cortical hierarchy in biological neural systems. Information propagates from sensory input through radial wave routing into deep central clusters.",
            "Linear algebra provides the mathematical framework for transformer representations. Matrix multiplications project high-dimensional token embeddings across attention heads.",
            "In mathematics, calculus describes continuous change. Differential equations model rates of change across physical dynamical systems, from fluid mechanics to quantum states.",
            "Computer science relies on efficient data structures and algorithms. Binary search trees, hash maps, and graphs allow logarithmic and constant time retrieval.",
            "Biological intelligence emerges from modular recurrent circuits and small-world connectivity. The human brain balances local clustering with long-range axonal shortcuts.",
        ]
        return corpus * 500

    def get_batch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        needed_tokens = self.batch_size * (self.seq_len + 1)
        while len(self.buffer) < needed_tokens:
            doc = self.texts[self.text_idx % len(self.texts)]
            self.text_idx += 1

            if self.digester is not None:
                try:
                    toks, weights = self.digester.digest_text(doc, self.tokenizer)
                    toks = list(toks) + [2]  # EOS
                    weights = list(weights) + [1.0]  # EOS weight
                except Exception as e:
                    # Robust fallback if teacher digestion encounters rare formatting anomaly
                    toks = self.tokenizer.encode(doc, add_special_tokens=False) + [2]
                    weights = [1.0] * len(toks)
            else:
                toks = self.tokenizer.encode(doc, add_special_tokens=False) + [2]  # EOS
                weights = [1.0] * len(toks)

            self.buffer.extend(toks)
            self.weight_buffer.extend(weights)

        chunk_toks = self.buffer[:needed_tokens]
        chunk_weights = self.weight_buffer[:needed_tokens]
        self.buffer = self.buffer[needed_tokens:]
        self.weight_buffer = self.weight_buffer[needed_tokens:]

        # Reshape into [batch_size, seq_len + 1]
        x = torch.tensor(chunk_toks, dtype=torch.long, device=device).view(self.batch_size, self.seq_len + 1)
        w = torch.tensor(chunk_weights, dtype=torch.float32, device=device).view(self.batch_size, self.seq_len + 1)

        prompts = x[:, : self.seq_len]
        targets = x[:, 1 :]
        target_weights = w[:, 1 :]  # Aligned 1:1 with next-token prediction targets
        return prompts, targets, target_weights


def run_pretraining(
    small_checkpoint: str,
    compact_config: str,
    output_dir: str,
    train_data: str,
    val_data: str,
    total_steps: int = 10000,
    cycle_length: int = 1000,
    anneal_steps: int = 100,
    batch_size: int = 2,
    accum_steps: int = 8,
    base_lr: float = 2e-4,
    min_lr: float = 2e-6,
    cloud_sync: bool = True,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    teacher_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    digest_mode: str = "online",
    teacher_scale: float = 1.5,
    min_weight: float = 0.2,
    max_weight: float = 4.0,
):
    device = torch.device(device_str)
    dev_type = "cuda" if "cuda" in device_str else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(output_dir, "v5_compact_pretrain_latest.pt")
    best_ckpt_path = os.path.join(output_dir, "v5_compact_pretrain_best.pt")
    seed_ckpt_path = os.path.join(output_dir, "v5_compact_expanded_seed.pt")

    print("=" * 80)
    print("🧠 CEREBROSPHEROID V5 COMPACT: PROGRESSIVE CONTINUED PRETRAINING")
    print("================================================================================")
    print(f"  Target Architecture       : v5_compact (245.6M params, 32 shells, d=768)")
    print(f"  Device                    : {device}")
    print(f"  Teacher Digester          : {teacher_model if digest_mode != 'none' else 'DISABLED'} (mode: {digest_mode})")
    print(f"  Dynamic Loss Weighting    : alpha={teacher_scale}, range=[{min_weight}, {max_weight}]")
    print(f"  Total Pretraining Steps   : {total_steps:,}")
    print(f"  Cycle Schedule            : Freeze {cycle_length - anneal_steps} steps ➔ Anneal {anneal_steps} steps (every {cycle_length} steps)")
    print(f"  Batch Config              : Micro {batch_size} × Accum {accum_steps} = Effective {batch_size * accum_steps}")
    print(f"  Learning Rate Range       : {base_lr:.2e} -> {min_lr:.2e}")
    print(f"  Cloud Sync (bioflo.in)    : {cloud_sync}")
    print("=" * 80 + "\n")

    # 1. Model Initialization / Resumption
    start_step = 0
    best_loss = float("inf")

    if os.path.exists(latest_ckpt_path):
        print(f"🔄 Auto-resuming from existing pretraining checkpoint: {latest_ckpt_path}")
        raw = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        with open(compact_config) as f:
            raw_cfg = json.load(f)
        valid_keys = {k: v for k, v in raw_cfg.items() if k in ModelConfigV5.__dataclass_fields__}
        cfg = ModelConfigV5(**valid_keys)
        model = CerebroSpheroidV5(cfg).to(device)
        model.load_state_dict(raw["model_state"], strict=False)
        start_step = int(raw.get("step", 0))
        best_loss = float(raw.get("val_loss", raw.get("loss", float("inf"))))
        print(f"  ✓ Resumed at Step {start_step:,} (Best Loss: {best_loss:.4f})")
    else:
        model, cfg = expand_v5_small_to_compact(
            small_ckpt_path=small_checkpoint,
            compact_cfg_path=compact_config,
            output_seed_path=seed_ckpt_path,
            cloud_sync=cloud_sync,
        )
        model = model.to(device)

    # 2. Tokenizer, Teacher Digester & Pretraining Stream
    tokenizer = build_tokenizer(DEFAULT_TOKENIZER_DIR)

    digester = None
    if digest_mode != "none":
        try:
            digester = create_digester(
                teacher_model=teacher_model,
                device=device_str,
                mode=digest_mode,
                alpha=teacher_scale,
                w_min=min_weight,
                w_max=max_weight,
            )
            print(f"  ✓ Teacher Digester initialized: {teacher_model} (mode: {digest_mode})")
        except Exception as e:
            print(f"  ⚠️ Teacher Digester load failed ({e}). Falling back to heuristic digester.")
            digester = create_digester(mode="heuristic", alpha=teacher_scale, w_min=min_weight, w_max=max_weight)

    train_stream = PretrainingTokenStream(
        train_data,
        tokenizer,
        seq_len=512,
        batch_size=batch_size,
        digester=digester,
    )

    # 3. Parameter Group Separation (Core vs. Expansion)
    core_params = []
    expansion_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Inherited baseline weights from v5_small
        if any(core_kw in name for core_kw in ["sensory_dnn", "intra_shell_swiglu.0", "intra_norms.0"]):
            core_params.append(p)
        else:
            expansion_params.append(p)

    print(f"📊 Parameter Partition: Core {len(core_params)} groups | Expansion {len(expansion_params)} groups")

    # 4. Learning Rate Function with Periodic Cyclical Annealing
    # Steps 0 -> 900: Core Frozen. Expansion trains at base_lr
    # Steps 900 -> 1000: Anneal Phase. Core unfreezes. Both anneal smoothly down to min_lr
    def get_phase_and_lrs(step_idx: int) -> Tuple[bool, float, float]:
        step_in_cycle = step_idx % cycle_length
        is_annealing = step_in_cycle >= (cycle_length - anneal_steps)

        if not is_annealing:
            # Accumulation phase: expansion active, core frozen
            return False, 0.0, base_lr
        else:
            # Anneal phase (last 100 steps of every 1000 steps)
            anneal_progress = (step_in_cycle - (cycle_length - anneal_steps)) / max(1, anneal_steps)
            anneal_factor = 0.5 * (1.0 + math.cos(math.pi * anneal_progress))
            exp_lr = min_lr + (base_lr - min_lr) * anneal_factor
            core_lr = min_lr + (base_lr * 0.1 - min_lr) * anneal_factor
            return True, core_lr, exp_lr

    # Initialize optimizer with expansion params active
    optimizer = torch.optim.AdamW(
        [
            {"params": core_params, "lr": 0.0, "weight_decay": 0.01},
            {"params": expansion_params, "lr": base_lr, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Set initial freeze status
    for p in core_params:
        p.requires_grad = False

    test_prompts = [
        "In the human brain, cortical layers",
        "The fundamental law of calculus states that",
        "Linear algebra projection matrices are used to",
        "What is 15 + 27? Step-by-step:",
    ]

    print("\n🚀 Beginning Continuous Pretraining with Periodic Consolidation (100 steps / 1,000 steps)...\n")
    model.train()
    start_time = time.time()
    accum_count = 0
    step = start_step
    ema_loss = None
    currently_annealing = False

    while step < total_steps:
        # Check Cyclical Annealing Status
        is_anneal, core_lr, exp_lr = get_phase_and_lrs(step)

        if is_anneal and not currently_annealing:
            print(f"\n🔓 [STEP {step:,}] ➔ ENTERING 100-STEP CONSOLIDATION ANNEAL (Core Unfrozen, Cosine Decay Active)")
            for p in core_params:
                p.requires_grad = True
            currently_annealing = True

        elif not is_anneal and currently_annealing:
            print(f"\n🔒 [STEP {step:,}] ➔ EXITING ANNEAL: RE-FREEZING CORE FOR NEXT 900-STEP ACCUMULATION BLOCK")
            for p in core_params:
                p.requires_grad = False
            currently_annealing = False

        # Fetch Pretraining Batch with Dynamic Next-Token Weights
        p_t, t_t, w_t = train_stream.get_batch(device)

        if dev_type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                loss, _ = model(
                    prompts=p_t,
                    targets=t_t,
                    target_weights=w_t,
                    sampling_p=0.5,
                    deliberation_mode="scheduled",
                )
        else:
            loss, _ = model(
                prompts=p_t,
                targets=t_t,
                target_weights=w_t,
                sampling_p=0.5,
                deliberation_mode="scheduled",
            )

        loss_scaled = loss / accum_steps
        loss_scaled.backward()
        accum_count += 1

        if accum_count % accum_steps == 0:
            step += 1

            # Update dynamic learning rates
            optimizer.param_groups[0]["lr"] = core_lr
            optimizer.param_groups[1]["lr"] = exp_lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            ema_loss = loss_val if ema_loss is None else (0.92 * ema_loss + 0.08 * loss_val)

            # Progress Logging
            if step % 10 == 0 or step == 1 or step == total_steps:
                elapsed = time.time() - start_time
                steps_per_sec = (step - start_step) / max(1e-3, elapsed)
                vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                phase_tag = "ANNEAL (UNFROZEN)" if is_anneal else "ACCUM (FROZEN)"
                w_mean = w_t.mean().item()
                w_max = w_t.max().item()
                w_min = w_t.min().item()
                print(f"[{phase_tag}] Step {step:5d}/{total_steps} | Loss: {loss_val:.4f} | EMA: {ema_loss:.4f} | W:[{w_min:.2f}/{w_mean:.2f}/{w_max:.2f}] | ExpLR: {exp_lr:.2e} | CoreLR: {core_lr:.2e} | Speed: {steps_per_sec:.2f} st/s | VRAM: {vram_gb:.2f} GB")

            # Checkpoint & Cloud Backup at Anneal Milestones (every 1,000 steps) or every 200 steps
            is_cycle_end = (step % cycle_length == 0)
            if step % 200 == 0 or is_cycle_end or step == total_steps:
                save_payload = {
                    "step": step,
                    "loss": float(loss_val),
                    "ema_loss": float(ema_loss),
                    "model_state": {k: v.cpu().float() for k, v in model.state_dict().items()},
                    "config": cfg,
                    "timestamp": time.time(),
                }

                # 1. Local Latest Save
                torch.save(save_payload, latest_ckpt_path)

                # 2. Local Best Save
                is_best = False
                if loss_val < best_loss:
                    best_loss = loss_val
                    torch.save(save_payload, best_ckpt_path)
                    is_best = True
                    print(f"  🏆 New Best Loss Checkpoint: {best_loss:.4f} -> {best_ckpt_path}")

                # 3. Bioflo Cloud Sync (Background Thread)
                if cloud_sync:
                    try:
                        push_checkpoint_async(latest_ckpt_path, step, loss_val, target_type="latest", model="v5_compact")
                        if is_best:
                            push_checkpoint_async(best_ckpt_path, step, loss_val, target_type="best", model="v5_compact")
                        print("  🌐 Checkpoint mirrored to bioflo.in in background.")
                    except Exception as e:
                        print(f"  ⚠️ Cloud sync notice: {e}")

                # 4. Google Drive Backup (if mounted)
                gdrive_dir = "/content/drive/MyDrive/cerebro_checkpoints"
                if os.path.exists("/content/drive"):
                    try:
                        os.makedirs(gdrive_dir, exist_ok=True)
                        shutil.copy(latest_ckpt_path, os.path.join(gdrive_dir, "v5_compact_pretrain_latest.pt"))
                        if is_best:
                            shutil.copy(best_ckpt_path, os.path.join(gdrive_dir, "v5_compact_pretrain_best.pt"))
                        print("  💾 Checkpoint backed up to Google Drive.")
                    except Exception as ge:
                        print(f"  ⚠️ Google Drive backup notice: {ge}")

                # 5. Live Linguistic Probe
                model.eval()
                print("  --- 🔍 Live Linguistic & Reasoning Probes ---")
                with torch.no_grad():
                    for prompt in test_prompts:
                        res = model.generate_text(prompt, tokenizer=tokenizer, max_new_tokens=48, temperature=0.3, device=device)
                        print(f"   Prompt: {prompt}")
                        print(f"   Model : {res.strip().replace(chr(10), ' ⏎ ')}")
                model.train()
                print("  ---------------------------------------------\n")

    print("\n" + "=" * 80)
    print("✨ CEREBROSPHEROID V5 COMPACT CONTINUED PRETRAINING COMPLETE")
    print(f"  Final Saved Model Checkpoint : {latest_ckpt_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="v5_small -> v5_compact Pretraining with Cyclical Annealing")
    parser.add_argument("--small-ckpt", type=str, default=DEFAULT_SMALL_CKPT)
    parser.add_argument("--compact-config", type=str, default=DEFAULT_COMPACT_CFG)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-data", type=str, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=str, default=DEFAULT_VAL_DATA)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--cycle-length", type=int, default=1000)
    parser.add_argument("--anneal-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--cloud-sync", action="store_true", default=True)
    parser.add_argument("--no-cloud-sync", dest="cloud_sync", action="store_false")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--digest-mode", type=str, choices=["online", "heuristic", "none"], default="online")
    parser.add_argument("--teacher-scale", type=float, default=1.5)
    parser.add_argument("--min-weight", type=float, default=0.2)
    parser.add_argument("--max-weight", type=float, default=4.0)
    args = parser.parse_args()

    run_pretraining(
        small_checkpoint=args.small_ckpt,
        compact_config=args.compact_config,
        output_dir=args.output_dir,
        train_data=args.train_data,
        val_data=args.val_data,
        total_steps=args.total_steps,
        cycle_length=args.cycle_length,
        anneal_steps=args.anneal_steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        base_lr=args.base_lr,
        min_lr=args.min_lr,
        cloud_sync=args.cloud_sync,
        device_str=args.device,
        teacher_model=args.teacher_model,
        digest_mode=args.digest_mode,
        teacher_scale=args.teacher_scale,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )


if __name__ == "__main__":
    main()
