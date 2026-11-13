#!/usr/bin/env python3
"""
Quick inference and sample generation runner for CerebroSpheroid V5.
Loads a checkpoint (PyTorch or JAX-adapted) and tests text generation across
pedagogical, logical, and conversational stimuli.
"""

import os
import sys
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from src.jax.checkpoint import adapt_jax_to_torch_state_dict
from transformers import AutoTokenizer

def run_sample_generation(checkpoint_path: str, preset: str = "tpu_v5e_1"):
    print(f"📦 Loading checkpoint from {checkpoint_path}...")
    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    raw_cfg = PRESETS_V5.get(preset, PRESETS_V5.get("v5_small"))
    cfg = ModelConfigV5(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    model = CerebroSpheroidV5(cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    step = ckpt.get("step", "unknown")
    loss = ckpt.get("loss", "unknown")
    print(f"✓ Checkpoint loaded: Step {step:,} (Loss: {loss})")

    # Adapt JAX keys if needed
    if any(k.startswith("core.intra_swiglu") for k in state.keys()):
        state = adapt_jax_to_torch_state_dict(state, cfg)

    model.load_state_dict(state, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompts = [
        "What is an atom?",
        "What is gravity and how does it work?",
        "Question: If 2 + 2 = 4, then 4 + 4 equals what? Answer:",
        "Explain the importance of water for living organisms:",
    ]

    print("\n" + "="*80)
    print(f"🧠 CEREBRO-SPHEROID V5 SAMPLE OUTPUTS (Step {step:,} | Loss {loss})")
    print("   [Mode: Strict output inside <act> ... </act>]")
    print("="*80)

    for p in prompts:
        print(f"\n[Stimulus Prompt]: {p}")
        with torch.no_grad():
            out = model.generate_text(
                prompt=p,
                tokenizer=tokenizer,
                max_new_tokens=64,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                device=device,
                strict_act=True,
            )
        print(f"[Strict Motor Action Output]:\n{out.strip()}\n" + "-"*60)

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_cloud_best_live.pt"
    run_sample_generation(ckpt)
