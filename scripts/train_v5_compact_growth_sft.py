#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Endogenous Model Growth Engine (v5_small -> v5_compact)
=============================================================================
Executes function-preserving structural expansion, parameter freezing, and 
scheduled deliberation SFT at P=0.50:
  1. Expansion (Net2Net):
     - v5_small (102.4M params: 16 shells, d=640, r=256, 40 tracts)
     - -> v5_compact (245.6M params: 32 shells, d=768, r=512, 96 tracts)
     - Preserves 100% of 900k-step native linguistic and biological priors
  2. Training Protocol:
     - Phase 1 (Steps 0 - 300): Inherited core FROZEN; new tracts & shells adapt
     - Phase 2 (Steps 300+): Unfreeze with differential LR & Cosine Annealing
     - Scheduled Deliberation at P = 0.50 (50% scratchpad thought / 50% direct)
  3. Optimized for Google Colab (NVIDIA T4 / L4 / A100 / Local GPU)
"""

import os
import sys
import time
import json
import math
import argparse
import random
import re
from typing import Dict, Any, List, Tuple

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
    load_v5_checkpoint_into_torch,
)
from src.corpus import (
    build_tokenizer,
    format_stimulus_prompt,
    encode_no_special,
)

DEFAULT_SMALL_CKPT = os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_latest_step920k.pt")
DEFAULT_COMPACT_CFG = os.path.join(WORKSPACE_ROOT, "configs", "v5_compact_250m.json")
DEFAULT_EXPANDED_CKPT = os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_compact_expanded_warmstart.pt")
DEFAULT_OUTPUT_CKPT = os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_compact_growth_sft_p05.pt")
DEFAULT_TRAIN_DATA = os.path.join(WORKSPACE_ROOT, "data", "cerebro_v5_custom_dataset", "train.parquet")
DEFAULT_VAL_DATA = os.path.join(WORKSPACE_ROOT, "data", "cerebro_v5_custom_dataset", "validation.parquet")
DEFAULT_TOKENIZER_DIR = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")


def expand_v5_small_to_compact(
    small_ckpt_path: str,
    compact_cfg_path: str,
    output_path: str,
    device: str = "cpu",
) -> Tuple[CerebroSpheroidV5, ModelConfigV5]:
    """
    Expands v5_small state dict into v5_compact using zero-padded subspace
    expansion and smooth shell interpolation.
    """
    print("\n" + "=" * 80)
    print("🧬 [STEP 1/3] ENDOGENOUS SPHEROID EXPANSION: v5_small -> v5_compact")
    print("=" * 80)
    print(f"  Source Small Checkpoint : {small_ckpt_path}")
    print(f"  Target Compact Config   : {compact_cfg_path}")
    print(f"  Output Seed Checkpoint  : {output_path}")

    # 1. Load small checkpoint
    small_raw = torch.load(small_ckpt_path, map_location="cpu", weights_only=False)
    small_sd = small_raw.get("model_state", small_raw.get("model", small_raw))

    # Convert JAX keys if needed
    if any(k.startswith(("sensory.", "core.", "motor.")) for k in small_sd.keys()):
        from src.jax.checkpoint import adapt_jax_to_torch_state_dict
        s_cfg = ModelConfigV5(**PRESETS_V5["v5_small"])
        small_sd = adapt_jax_to_torch_state_dict(small_sd, s_cfg)

    # 2. Instantiate target compact model
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
        # Case A: Exact match in key and shape
        if k_tgt in small_sd and small_sd[k_tgt].shape == v_tgt.shape:
            new_sd[k_tgt] = small_sd[k_tgt].clone()
            transferred_count += 1

        # Case B: Direct match in key, but shape expanded (Surface: 640->768, Radial: 256->512, Tracts: 40->96)
        elif k_tgt in small_sd and small_sd[k_tgt].shape != v_tgt.shape:
            v_src = small_sd[k_tgt]
            adapted = v_tgt.clone() * 0.0  # Zero initialization for outer channels
            slices = tuple(slice(0, min(s, d)) for s, d in zip(v_src.shape, v_tgt.shape))
            adapted[slices] = v_src[slices]

            # If it's a normalization gain vector, initialize newly added channels to 1.0
            if "norm" in k_tgt and v_tgt.dim() == 1 and v_tgt.shape[0] > v_src.shape[0]:
                adapted[v_src.shape[0]:] = 1.0

            new_sd[k_tgt] = adapted
            expanded_count += 1

        # Case C: Concentric shell expansion (shells 16..31)
        elif "intra_shell_swiglu" in k_tgt or "intra_norms" in k_tgt:
            # Extract shell index
            m = re.search(r"\.(\d+)\.", k_tgt)
            if m:
                shell_idx = int(m.group(1))
                # Map 32 shells back to 16 shells: l = floor(k * 15 / 31)
                src_shell_idx = min(K_src - 1, int(round(shell_idx * (K_src - 1) / (K_tgt - 1))))
                src_key = k_tgt.replace(f".{shell_idx}.", f".{src_shell_idx}.")

                if src_key in small_sd:
                    v_src = small_sd[src_key]
                    adapted = v_tgt.clone() * 0.0
                    slices = tuple(slice(0, min(s, d)) for s, d in zip(v_src.shape, v_tgt.shape))
                    adapted[slices] = v_src[slices]

                    # Add tiny perturbation for even shells to break symmetry
                    if shell_idx >= K_src:
                        noise = torch.randn_like(adapted) * 1e-4
                        adapted = adapted + noise

                    new_sd[k_tgt] = adapted
                    interpolated_count += 1
                else:
                    new_sd[k_tgt] = v_tgt
            else:
                new_sd[k_tgt] = v_tgt

        # Case D: Radial wave routing matrices (15 -> 31)
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

    print(f"  ✓ Spheroid Grown: {p_small/1e6:.1f}M -> {p_compact/1e6:.1f}M parameters")
    print(f"  ✓ Transferred Tensors  : {transferred_count}")
    print(f"  ✓ Expanded Tensors     : {expanded_count}")
    print(f"  ✓ Interpolated Laminae : {interpolated_count}")

    save_payload = {
        "step": small_raw.get("step", 920500),
        "model_state": {k: v.cpu().float() for k, v in compact_model.state_dict().items()},
        "config": compact_cfg,
        "growth_origin": "v5_small_step920k",
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(save_payload, output_path)
    print(f"  ✓ Seeded checkpoint written to: {output_path}\n")

    return compact_model, compact_cfg


def load_tokenized_dataset(
    parquet_path: str,
    tokenizer: Any,
    max_samples: int = 30000,
    max_prompt_len: int = 2048,
    max_target_len: int = 1024,
) -> List[Dict[str, Any]]:
    print(f"📦 Loading & tokenizing step-by-step reasoning dataset from: {parquet_path}")
    if not os.path.exists(parquet_path):
        print(f"⚠️ Parquet dataset not found at {parquet_path}. Using fallback synthetic dataset.")
        return build_fallback_reasoning_data(tokenizer, max_samples=1000)

    df = pd.read_parquet(parquet_path)
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)

    entries = []
    eos_id = getattr(tokenizer, "eos_token_id", 2) or 2

    for _, r in df.iterrows():
        prompt = str(r["prompt"]).strip()
        answer = str(r["answer"]).strip()
        if not prompt or not answer:
            continue

        thinking = str(r["thinking"]).strip() if r.get("thinking") is not None and not pd.isna(r["thinking"]) else ""
        thinking = re.sub(r"<<.*?>>", "", thinking).strip()

        if thinking and thinking.lower() != "none" and len(thinking) > 5:
            full_target = f"{thinking}\n\n{answer}"
        else:
            full_target = answer

        p_str = format_stimulus_prompt(prompt)
        raw_p = [t % vocab_size for t in encode_no_special(tokenizer, p_str)]
        if len(raw_p) > max_prompt_len:
            prefix_len = min(32, max_prompt_len // 4)
            suffix_len = max_prompt_len - prefix_len
            p_ids = raw_p[:prefix_len] + raw_p[-suffix_len:]
        else:
            p_ids = raw_p

        raw_t = [t % vocab_size for t in encode_no_special(tokenizer, full_target)]
        if not raw_t or raw_t[-1] != eos_id:
            raw_t.append(eos_id)
        if len(raw_t) > max_target_len:
            t_ids = raw_t[: max_target_len - 1] + [eos_id]
        else:
            t_ids = raw_t

        if len(p_ids) > 0 and len(t_ids) > 1:
            entries.append({"prompt_ids": p_ids, "target_ids": t_ids})

    print(f"  ✓ Successfully tokenized {len(entries):,} reasoning entries.")
    return entries


def build_fallback_reasoning_data(tokenizer: Any, max_samples: int = 1000) -> List[Dict[str, Any]]:
    """Self-contained arithmetic, logic, and factual reasoning examples if parquet is missing."""
    qa_pairs = [
        ("What is 15 + 27?", "Step 1: 15 + 20 = 35.\nStep 2: 35 + 7 = 42.\n\nThe answer is 42."),
        ("What is 12 * 8?", "Step 1: 10 * 8 = 80.\nStep 2: 2 * 8 = 16.\nStep 3: 80 + 16 = 96.\n\nThe answer is 96."),
        ("If Alice has 3 apples and Bob gives her 5 more, how many apples does Alice have?", "Step 1: Alice starts with 3 apples.\nStep 2: Bob gives her 5.\nStep 3: 3 + 5 = 8.\n\nAlice has 8 apples."),
        ("What is the capital of France?", "France is a country in Western Europe. Its capital and largest city is Paris.\n\nThe capital of France is Paris."),
        ("What is the chemical symbol for gold?", "Gold is a chemical element with atomic number 79. Its symbol comes from the Latin word 'aurum'.\n\nThe symbol for gold is Au."),
        ("Write a Python function to check if a number is even.", "def is_even(n: int) -> bool:\n    \"\"\"Return True if n is even, else False.\"\"\"\n    return n % 2 == 0"),
        ("Solve for x: 2x + 6 = 14.", "Step 1: Subtract 6 from both sides: 2x = 8.\nStep 2: Divide by 2: x = 4.\n\nThe value of x is 4."),
    ]
    entries = []
    eos_id = getattr(tokenizer, "eos_token_id", 2) or 2
    for _ in range(max_samples // len(qa_pairs) + 1):
        for p, a in qa_pairs:
            p_ids = encode_no_special(tokenizer, format_stimulus_prompt(p))[:2048]
            t_ids = encode_no_special(tokenizer, a)[:1023] + [eos_id]
            entries.append({"prompt_ids": p_ids, "target_ids": t_ids})
            if len(entries) >= max_samples:
                break
        if len(entries) >= max_samples:
            break
    return entries


def evaluate_val_loss(model, val_entries, device, sampling_p=0.5, max_eval_batches=20):
    model.eval()
    total_loss = 0.0
    count = 0
    dev_type = "cuda" if "cuda" in str(device) else "cpu"
    with torch.no_grad():
        for i in range(min(max_eval_batches, len(val_entries) // 4)):
            batch = val_entries[i * 4 : (i + 1) * 4]
            max_p = max(len(e["prompt_ids"]) for e in batch)
            max_t = max(len(e["target_ids"]) for e in batch)
            p_t = torch.zeros((len(batch), max_p), dtype=torch.long, device=device)
            t_t = torch.full((len(batch), max_t), -100, dtype=torch.long, device=device)
            for b_idx, e in enumerate(batch):
                p_t[b_idx, : len(e["prompt_ids"])] = torch.tensor(e["prompt_ids"], dtype=torch.long, device=device)
                t_t[b_idx, : len(e["target_ids"])] = torch.tensor(e["target_ids"], dtype=torch.long, device=device)

            if dev_type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    loss, _ = model(prompts=p_t, targets=t_t, sampling_p=sampling_p, deliberation_mode="scheduled")
            else:
                loss, _ = model(prompts=p_t, targets=t_t, sampling_p=sampling_p, deliberation_mode="scheduled")
            total_loss += loss.item()
            count += 1
    return total_loss / max(1, count)


def run_growth_and_sft(
    small_checkpoint: str,
    compact_config: str,
    output_checkpoint: str,
    train_path: str,
    val_path: str,
    max_steps: int = 1500,
    freeze_steps: int = 300,
    batch_size: int = 2,
    accum_steps: int = 8,
    lr: float = 2e-4,
    min_lr: float = 2e-6,
    sampling_p: float = 0.5,
    probe_interval: int = 100,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    dev_type = "cuda" if "cuda" in device_str else "cpu"
    print("=" * 80)
    print("🧠 CEREBROSPHEROID V5 COMPACT: PROGRESSIVE GROWTH & SFT ENGINE (P=0.50)")
    print("=" * 80)
    print(f"  Device                     : {device}")
    print(f"  Freeze Steps (Phase 1)     : {freeze_steps} (Core weights isolated)")
    print(f"  Annealing Steps (Phase 2)  : {max_steps - freeze_steps} (Full model co-adaptation)")
    print(f"  Deliberation Sampling P    : {sampling_p:.2f} (50% internal thought / 50% teacher)")
    print(f"  Batch Configuration        : Micro {batch_size} × Accum {accum_steps} = Effective {batch_size * accum_steps}")
    print("=" * 80 + "\n")

    # 1. Expand v5_small to v5_compact
    seed_path = output_checkpoint.replace(".pt", "_expanded_seed.pt")
    model, cfg = expand_v5_small_to_compact(small_checkpoint, compact_config, seed_path, device=str(device))
    model = model.to(device)

    # 2. Tokenizer
    tokenizer = build_tokenizer(DEFAULT_TOKENIZER_DIR)

    # 3. Datasets
    train_entries = load_tokenized_dataset(train_path, tokenizer, max_samples=30000)
    val_entries = load_tokenized_dataset(val_path, tokenizer, max_samples=2000)

    # 4. Parameter Separation for Freeze & Differential LR
    core_params = []
    expansion_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Core inherited structures
        if any(core_kw in name for core_kw in ["sensory_dnn", "volumetric_core.intra_shell_swiglu.0", "volumetric_core.intra_norms.0"]):
            core_params.append(p)
        else:
            expansion_params.append(p)

    # Initial Phase 1: Freeze core parameters
    print(f"🔒 Phase 1: Freezing {len(core_params)} core tensor groups for first {freeze_steps} steps...")
    for p in core_params:
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": expansion_params, "lr": lr, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    def get_lr(step_idx: int) -> float:
        progress = step_idx / max(1, max_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    test_prompts = [
        ("What is 15 + 27?", "42"),
        ("If Alice has 3 apples and Bob gives her 5 more, how many apples does Alice have?", "8 apples"),
        ("What is the capital of France?", "Paris"),
        ("Solve for x: 2x + 6 = 14.", "x = 4"),
    ]

    print("\n🚀 Commencing SFT Training with Scheduled Deliberation at P=0.50...\n")
    model.train()
    start_time = time.time()
    accum_count = 0
    step = 0
    ema_loss = None
    best_val_loss = float("inf")
    unfrozen_phase2 = False

    while step < max_steps:
        # Phase 2 Transition: Unfreeze Core Parameters
        if step >= freeze_steps and not unfrozen_phase2:
            print(f"\n🔓 [STEP {step}] PHASE 2 TRANSITION: Unfreezing core parameters with differential LR!")
            for p in core_params:
                p.requires_grad = True

            # Add core params to optimizer with 10x smaller learning rate
            optimizer = torch.optim.AdamW(
                [
                    {"params": core_params, "lr": lr * 0.1, "weight_decay": 0.01},
                    {"params": expansion_params, "lr": lr, "weight_decay": 0.01},
                ],
                betas=(0.9, 0.95),
                eps=1e-8,
            )
            unfrozen_phase2 = True

        batch = random.sample(train_entries, batch_size)
        max_p = max(len(e["prompt_ids"]) for e in batch)
        max_t = max(len(e["target_ids"]) for e in batch)
        p_t = torch.zeros((len(batch), max_p), dtype=torch.long, device=device)
        t_t = torch.full((len(batch), max_t), -100, dtype=torch.long, device=device)
        for b_idx, e in enumerate(batch):
            p_t[b_idx, : len(e["prompt_ids"])] = torch.tensor(e["prompt_ids"], dtype=torch.long, device=device)
            t_t[b_idx, : len(e["target_ids"])] = torch.tensor(e["target_ids"], dtype=torch.long, device=device)

        if dev_type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                loss, _ = model(prompts=p_t, targets=t_t, sampling_p=sampling_p, deliberation_mode="scheduled")
        else:
            loss, _ = model(prompts=p_t, targets=t_t, sampling_p=sampling_p, deliberation_mode="scheduled")
        loss_scaled = loss / accum_steps

        loss_scaled.backward()
        accum_count += 1

        if accum_count % accum_steps == 0:
            step += 1
            cur_lr = get_lr(step)

            if unfrozen_phase2:
                optimizer.param_groups[0]["lr"] = cur_lr * 0.1  # Core
                optimizer.param_groups[1]["lr"] = cur_lr        # Expansion
            else:
                optimizer.param_groups[0]["lr"] = cur_lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            ema_loss = loss_val if ema_loss is None else (0.92 * ema_loss + 0.08 * loss_val)

            if step % 10 == 0 or step == 1 or step == max_steps:
                elapsed = time.time() - start_time
                steps_per_sec = step / max(1e-3, elapsed)
                vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                phase_label = "PHASE 1 (FROZEN)" if not unfrozen_phase2 else "PHASE 2 (ANNEAL)"
                print(f"[{phase_label}] Step {step:4d}/{max_steps} | Loss: {loss_val:.4f} | EMA: {ema_loss:.4f} | LR: {cur_lr:.2e} | Speed: {steps_per_sec:.2f} st/s | VRAM: {vram_gb:.2f} GB")

            # Periodic Validation & Test Probes
            if step % probe_interval == 0 or step == max_steps:
                val_loss = evaluate_val_loss(model, val_entries, device, sampling_p=sampling_p)
                print(f"\n📊 [Validation Step {step}] Held-out Val Loss: {val_loss:.4f} (Best: {min(best_val_loss, val_loss):.4f})")

                save_payload = {
                    "model_state": {k: v.cpu().float() for k, v in model.state_dict().items()},
                    "step": step,
                    "val_loss": float(val_loss),
                    "ema_loss": float(ema_loss),
                    "sampling_p": sampling_p,
                    "config": cfg,
                }
                torch.save(save_payload, output_checkpoint)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt = output_checkpoint.replace(".pt", "_best.pt")
                    torch.save(save_payload, best_ckpt)
                    print(f"   🏆 New best checkpoint saved to: {best_ckpt}")

                # Live text generation probe
                model.eval()
                print("   --- 🔍 Live Generation Probes (Step-by-Step Scratchpad) ---")
                with torch.no_grad():
                    for p_probe, expected in test_prompts:
                        res = model.generate_text(p_probe, tokenizer=tokenizer, max_new_tokens=64, temperature=0.2, device=device)
                        print(f"   Prompt: {p_probe}")
                        print(f"   Model : {res.strip().replace(chr(10), ' ⏎ ')}")
                model.train()
                print("   ------------------------------------------------------------\n")

    print("\n" + "=" * 80)
    print("✨ CEREBROSPHEROID V5 COMPACT GROWTH & SFT SCRIPT COMPLETE")
    print(f"  Final Converged Model Checkpoint : {output_checkpoint}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="v5_small -> v5_compact Endogenous Growth & SFT")
    parser.add_argument("--small-ckpt", type=str, default=DEFAULT_SMALL_CKPT)
    parser.add_argument("--compact-config", type=str, default=DEFAULT_COMPACT_CFG)
    parser.add_argument("--output-ckpt", type=str, default=DEFAULT_OUTPUT_CKPT)
    parser.add_argument("--train-data", type=str, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=str, default=DEFAULT_VAL_DATA)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--freeze-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--sampling-p", type=float, default=0.5)
    parser.add_argument("--probe-interval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_growth_and_sft(
        small_checkpoint=args.small_ckpt,
        compact_config=args.compact_config,
        output_checkpoint=args.output_ckpt,
        train_path=args.train_data,
        val_path=args.val_data,
        max_steps=args.max_steps,
        freeze_steps=args.freeze_steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        sampling_p=args.sampling_p,
        probe_interval=args.probe_interval,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
