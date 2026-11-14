#!/usr/bin/env python3
"""
Comprehensive Generation and Reasoning Probe for CerebroSpheroid V5 Checkpoints.
Tests both PyTorch and JAX engines under greedy (T=0.1) and sampling (T=0.7) regimes.
"""

import os
import sys
import time
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from src.jax.checkpoint import adapt_jax_to_torch_state_dict, load_torch_pt_to_jax
from src.jax.v5_model import generate_text_jax

PROMPTS = [
    ("1. Arithmetic (Simple)", "Question: What is 2 + 2?\nAnswer:"),
    ("2. Arithmetic (GSM8K Word Problem)", "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:"),
    ("3. Deductive Logic (Syllogism)", "Question: All humans are mortal. Socrates is a human. What can we conclude about Socrates?\nAnswer:"),
    ("4. Number Comparison (Decimals)", "Question: Which number is bigger: 9.11 or 9.9?\nAnswer:"),
    ("5. Science / Chemistry", "Question: What is the chemical formula for water?\nAnswer:"),
    ("6. Multiple Choice (Biology)", "Question: What is the primary organ responsible for pumping blood through the human body?\nA) Brain\nB) Heart\nC) Liver\nD) Kidney\nAnswer:"),
    ("7. Python Coding", "def is_prime(n):\n    \"\"\"Return True if n is prime, else False.\"\"\"\n"),
]

def run_probe(checkpoint_path: str, preset: str = "v5_small"):
    print(f"\n{'='*80}")
    print(f"🔬 CEREBRO-SPHEROID V5 UPGRADED REASONING PROBE")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"{'='*80}\n")

    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    raw_cfg = PRESETS_V5.get(preset, PRESETS_V5["v5_small"])
    cfg = ModelConfigV5(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    cfg.use_sensory_self_attn = True

    # 1. Load into JAX
    print("⏳ Loading checkpoint into JAX engine with sensory self-attention...")
    jax_params, jax_buffers, meta = load_torch_pt_to_jax(checkpoint_path, cfg)
    step = meta.get("step", "unknown")
    loss = meta.get("loss", "unknown")
    eval_loss = meta.get("eval_loss", "unknown")
    print(f"✓ JAX Engine Ready | Step: {step} | Train Loss: {loss:.4f} | Eval Loss: {eval_loss:.4f}")
    
    out_self_norm = float(jnp.linalg.norm(jax_params["sensory.out_self.weight"]))
    print(f"✓ Sensory Self-Attention Identity Parity: ||out_self|| = {out_self_norm:.6f} (0.0 = exact identity)\n")

    # 2. Load into PyTorch
    print("⏳ Loading checkpoint into PyTorch engine with sensory self-attention...")
    pt_model = CerebroSpheroidV5(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    if any(k.startswith(("sensory.", "core.", "motor.")) for k in state.keys()):
        state = adapt_jax_to_torch_state_dict(state, cfg)
    pt_model.load_state_dict(state, strict=False)
    pt_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_model.to(device)
    print(f"✓ PyTorch Engine Ready on {device}\n")

    print(f"{'='*80}")
    print(f"PROMPT EVALUATION BATTERY (Step {step})")
    print(f"{'='*80}\n")

    for cat, prompt in PROMPTS:
        print(f"--- [{cat}] ---")
        print(f"Prompt: {prompt.strip()}")

        # PyTorch Greedy (T=0.1)
        with torch.no_grad():
            pt_greedy = pt_model.generate_text(
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=64,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.2,
                device=device,
                strict_act=False,
            )
        print(f"\n[PyTorch Greedy (T=0.1)]:\n{pt_greedy.strip()}")

        # PyTorch Sampling (T=0.7)
        with torch.no_grad():
            pt_sample = pt_model.generate_text(
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=64,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                device=device,
                strict_act=False,
            )
        print(f"\n[PyTorch Sampling (T=0.7)]:\n{pt_sample.strip()}")

        # JAX Autoregressive Rollout (Greedy Argmax)
        try:
            jax_out = generate_text_jax(
                params=jax_params,
                buffers=jax_buffers,
                config=cfg,
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=64,
                repetition_penalty=1.2,
                strict_act=False,
            )
            print(f"\n[JAX Engine (Deterministic Rollout)]:\n{jax_out.strip()}")
        except Exception as e:
            print(f"\n[JAX Engine]: Rollout note: {e}")

        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_latest_step612k.pt")
    run_probe(ckpt)
