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
from transformers import AutoTokenizer

def run_sample_generation(checkpoint_path: str, preset: str = "tpu_v5e_1"):
    print(f"📦 Loading checkpoint from {checkpoint_path}...")
    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    raw_cfg = PRESETS_V5[preset]
    cfg = ModelConfigV5(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    model = CerebroSpheroidV5(cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    step = ckpt.get("step", "unknown")
    loss = ckpt.get("loss", "unknown")
    print(f"✓ Checkpoint loaded: Step {step:,} (Loss: {loss})")

    # Adapt keys if necessary
    model.load_state_dict(state, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompts = [
        "What is gravity and how does it work?",
        "Explain photosynthesis in plants:",
        "The fundamental law of energy states that",
        "def fibonacci(n):",
    ]

    print("\n" + "="*80)
    print(f"🧠 CEREBRO-SPHEROID V5 SAMPLE OUTPUTS (Step {step:,} | Loss {loss})")
    print("="*80)

    for p in prompts:
        print(f"\n[Prompt]: {p}")
        with torch.no_grad():
            out = model.generate_text(
                prompt=p,
                tokenizer=tokenizer,
                max_new_tokens=64,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                device=device,
            )
        print(f"[Model Output]:\n{out.strip()}\n" + "-"*60)

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_cloud_best_live.pt"
    run_sample_generation(ckpt)
