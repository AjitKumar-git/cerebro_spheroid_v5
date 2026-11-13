#!/usr/bin/env python3
"""
Head-to-Head Comparative Benchmark:
CerebroSpheroid V5 (97M params) vs. Hugging Face SmolLM2-135M-Instruct (135M params).
Tests both models side-by-side with identical prompts.
Cerebro uses strict <act> ... </act> output extraction.
"""

import os
import sys
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/hf_home")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/hf_home")

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from src.jax.checkpoint import adapt_jax_to_torch_state_dict


def run_comparison(
    cerebro_ckpt: str = "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_cloud_best_live.pt",
    preset: str = "tpu_v5e_1",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("🥊 HEAD-TO-HEAD COMPARISON: SmolLM2-135M-Instruct vs. CerebroSpheroid V5 (97M)")
    print("=" * 80)

    # 1. Load SmolLM2-135M-Instruct
    print("\n📦 Loading SmolLM2-135M-Instruct from local cache...")
    smol_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    smol_tok = AutoTokenizer.from_pretrained(smol_id, local_files_only=True)
    smol_model = AutoModelForCausalLM.from_pretrained(smol_id, local_files_only=True).to(device)
    smol_model.eval()
    smol_params = sum(p.numel() for p in smol_model.parameters())
    print(f"✓ SmolLM2-135M Loaded: {smol_params:,} parameters ({smol_params/1e6:.1f}M).")

    # 2. Load CerebroSpheroid V5
    print(f"\n🧠 Loading CerebroSpheroid V5 ({cerebro_ckpt})...")
    raw_cfg = PRESETS_V5.get(preset, PRESETS_V5.get("v5_small"))
    cfg = ModelConfigV5(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    cerebro_tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    cerebro_tok = AutoTokenizer.from_pretrained(cerebro_tok_dir)

    cerebro_model = CerebroSpheroidV5(cfg)
    ckpt = torch.load(cerebro_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    step = ckpt.get("step", "unknown")
    loss = ckpt.get("loss", "unknown")

    if any(k.startswith("core.intra_swiglu") for k in state.keys()):
        state = adapt_jax_to_torch_state_dict(state, cfg)

    cerebro_model.load_state_dict(state, strict=False)
    cerebro_model.to(device)
    cerebro_model.eval()
    cerebro_params = sum(p.numel() for p in cerebro_model.parameters())
    print(f"✓ CerebroSpheroid V5 Loaded: Step {step:,} | Loss {loss:.4f} | {cerebro_params:,} parameters ({cerebro_params/1e6:.1f}M).")

    test_queries = [
        "What is an atom?",
        "What is gravity and how does it work?",
        "Question: If 2 + 2 = 4, then 4 + 4 equals what? Answer:",
        "Explain the importance of water for living organisms:",
        "Write a python function to compute factorial(n):",
    ]

    print("\n" + "=" * 80)
    print("🔬 COMPARATIVE GENERATION RESULTS")
    print("=" * 80)

    for idx, q in enumerate(test_queries, 1):
        print(f"\n[Test {idx}] Prompt: \"{q}\"")
        print("-" * 80)

        # SmolLM2 Generation
        smol_chat = [{"role": "user", "content": q}]
        smol_input_text = smol_tok.apply_chat_template(smol_chat, tokenize=False, add_generation_prompt=True)
        smol_inputs = smol_tok(smol_input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            smol_out = smol_model.generate(
                **smol_inputs,
                max_new_tokens=64,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=smol_tok.eos_token_id,
            )
        smol_gen_ids = smol_out[0][smol_inputs["input_ids"].shape[1]:]
        smol_ans = smol_tok.decode(smol_gen_ids, skip_special_tokens=True).strip()

        # CerebroSpheroid V5 Generation (Strict inside <act> ... </act>)
        with torch.no_grad():
            cerebro_ans = cerebro_model.generate_text(
                prompt=q,
                tokenizer=cerebro_tok,
                max_new_tokens=64,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                device=device,
                strict_act=True,
            ).strip()

        print(f"🤖 SmolLM2-135M-Instruct:\n{smol_ans}\n")
        print(f"🧠 CerebroSpheroid V5 (Strict <act>):\n{cerebro_ans}")
        print("-" * 80)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_cloud_best_live.pt"
    run_comparison(ckpt)
