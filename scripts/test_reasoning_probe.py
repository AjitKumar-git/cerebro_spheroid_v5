#!/usr/bin/env python3
"""
Test reasoning and generation capabilities of CerebroSpheroid V5 Step 593,000 checkpoint.
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

def test_checkpoint(checkpoint_path: str, preset: str = "tpu_v5e_1"):
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    raw_cfg = PRESETS_V5.get(preset, PRESETS_V5.get("v5_small"))
    cfg = ModelConfigV5(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    model = CerebroSpheroidV5(cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    step = ckpt.get("step", "unknown")
    loss = ckpt.get("loss", "unknown")
    print(f"✓ Checkpoint loaded: Step {step:,} (Loss: {loss:.4f} if isinstance(loss, (int, float)) else loss)")

    # Adapt JAX keys if needed
    if any(k.startswith("core.intra_swiglu") for k in state.keys()) or any("proj_in" in k for k in state.keys()):
        state = adapt_jax_to_torch_state_dict(state, cfg)

    model.load_state_dict(state, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompts = [
        # Arithmetic & Word Problems (GSM8K style)
        "Question: What is 2 + 2?\nAnswer:",
        "Question: If a box has 5 red balls and 3 blue balls, how many balls are in the box in total?\nAnswer:",
        "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
        # Deductive Logic
        "Question: All humans are mortal. Socrates is a human. What can we conclude about Socrates?\nAnswer:",
        "Question: Which number is bigger: 9.11 or 9.9?\nAnswer:",
        # Coding & Functional
        "Write a Python function to add two numbers a and b:\n",
        # World Knowledge
        "What is photosynthesis?",
    ]

    print("\n" + "="*80)
    print(f"🧠 CEREBRO-SPHEROID V5 REASONING PROBE (Step {step:,} | Loss {loss})")
    print("="*80)

    for p in prompts:
        print(f"\n👉 [PROMPT]:\n{p}")
        
        # Test 1: Greedy (temperature=0.1, deterministic best-path)
        with torch.no_grad():
            out_greedy = model.generate_text(
                prompt=p,
                tokenizer=tokenizer,
                max_new_tokens=96,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.2,
                device=device,
                strict_act=False,
            )
        print(f"\n[Generated Output (Greedy / Deterministic)]:\n{out_greedy.strip()}")

        # Test 2: Standard sampling (temperature=0.6)
        with torch.no_grad():
            out_sample = model.generate_text(
                prompt=p,
                tokenizer=tokenizer,
                max_new_tokens=96,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                device=device,
                strict_act=False,
            )
        print(f"\n[Generated Output (Sampling T=0.6)]:\n{out_sample.strip()}")
        print("-" * 70)

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_latest_step593k.pt"
    test_checkpoint(ckpt)
