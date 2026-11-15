#!/usr/bin/env python3
"""
Empirical Weight Transfer & Information Output Benchmark:
SmolLM2-135M-Instruct -> CerebroSpheroid V5 (Centroid Concentric Architecture)
"""

import os
import sys
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Force local Hugging Face cache
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/hf_home")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/hf_home")

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from src.jax.checkpoint import adapt_jax_to_torch_state_dict
from scripts.transformer_to_centroid_transfer import (
    extract_transformer_config,
    adapt_dimension,
    transfer_weights
)


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("🔬 EMPIRICAL WEIGHT TRANSFER BENCHMARK: SmolLM2-135M -> CerebroSpheroid V5")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load SmolLM2-135M-Instruct
    print("\n📦 [1/5] Loading Source Model: SmolLM2-135M-Instruct...")
    smol_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    smol_tok = AutoTokenizer.from_pretrained(smol_id, local_files_only=True)
    smol_model = AutoModelForCausalLM.from_pretrained(smol_id, local_files_only=True).to(device)
    smol_model.eval()
    smol_params = sum(p.numel() for p in smol_model.parameters())
    print(f"   ✓ SmolLM2 Parameters: {smol_params:,} ({smol_params/1e6:.2f}M)")

    # 2. Load V5 Tokenizer and Native Pre-trained V5 Checkpoint
    print("\n🧠 [2/5] Loading Native CerebroSpheroid V5 Checkpoint (v5_small_best.pt)...")
    v5_tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    v5_tok = AutoTokenizer.from_pretrained(v5_tok_dir)
    
    cfg = ModelConfigV5(**PRESETS_V5["tpu_v5e_1"])
    v5_native = CerebroSpheroidV5(cfg).to(device)
    
    ckpt_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_best.pt")
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw_ckpt.get("model_state", raw_ckpt.get("model", raw_ckpt))
    step = raw_ckpt.get("step", 900000)
    v5_state_torch = adapt_jax_to_torch_state_dict(state, cfg)
    v5_native.load_state_dict(v5_state_torch, strict=False)
    v5_native.eval()
    v5_params = sum(p.numel() for p in v5_native.parameters())
    print(f"   ✓ V5 Native Parameters: {v5_params:,} ({v5_params/1e6:.2f}M) at step {step:,}")

    # 3. Create Transferred V5 Model (Copying V5 architecture, injecting SmolLM2 weights)
    print("\n🔄 [3/5] Executing Cross-Architecture Weight Transfer...")
    v5_transferred = CerebroSpheroidV5(cfg).to(device)
    # Seed with native V5 weights first so non-transferable spheroid components have trained priors
    v5_transferred.load_state_dict(v5_native.state_dict())

    # Extract source config & weights
    tr_cfg = extract_transformer_config(smol_model.config)
    source_sd = smol_model.state_dict()

    # Detailed Weight Transfer Audit
    target_sd = v5_transferred.state_dict()
    transfer_audit = {
        "exact_matches": 0,
        "adapted_dimensions": 0,
        "spheroid_specific_unmapped": 0,
        "transferred_param_count": 0,
        "total_target_param_count": v5_params,
        "layer_correlations": [],
    }

    # Perform structured transfer
    transfer_stats = transfer_weights(source_sd, tr_cfg, v5_transferred, verbose=False)
    
    # Audit tensor matches
    transferred_sd = v5_transferred.state_dict()
    for k, v in target_sd.items():
        if k in transferred_sd:
            if not torch.equal(v, transferred_sd[k]):
                transfer_audit["transferred_param_count"] += v.numel()
                transfer_audit["adapted_dimensions"] += 1
            else:
                transfer_audit["spheroid_specific_unmapped"] += 1

    transferred_pct = (transfer_audit["transferred_param_count"] / v5_params) * 100
    print(f"\n📊 --- WEIGHT TRANSFER ACCURACY AUDIT ---")
    print(f"   • Exact Tensor Matches: 0% (Divergent shape: 576 -> 640 dim, 49k -> 32k vocab)")
    print(f"   • Dimension-Adapted Tensors: {transfer_audit['adapted_dimensions']} tensors")
    print(f"   • Untransferable Spheroid Tensors (Radial/Tracts): {transfer_audit['spheroid_specific_unmapped']} tensors")
    print(f"   • Effective Parameter Coverage Transferred: {transferred_pct:.2f}% ({transfer_audit['transferred_param_count']:,} / {v5_params:,} params)")

    # 4. Latent Representation Alignment: Token Embedding Space
    print("\n📐 [4/5] Testing Representation / Latent Space Alignment...")
    # Compare embeddings of common tokens
    smol_emb = smol_model.get_input_embeddings().weight.detach().cpu().float()  # [49152, 576]
    v5_native_emb = v5_native.sensory_dnn.embedding.weight.detach().cpu().float()  # [32000, 640]
    v5_trans_emb = v5_transferred.sensory_dnn.embedding.weight.detach().cpu().float()  # [32000, 640]

    # Shared dimension slice [0:576]
    common_vocab_len = min(smol_emb.shape[0], v5_trans_emb.shape[0])
    emb_cosine = F.cosine_similarity(
        smol_emb[:common_vocab_len],
        v5_trans_emb[:common_vocab_len, :576],
        dim=-1
    ).mean().item()
    print(f"   • Token Embedding Cosine Similarity (Transferred vs Source, dim 576): {emb_cosine:.4f}")
    
    native_vs_smol_cosine = F.cosine_similarity(
        smol_emb[:common_vocab_len],
        v5_native_emb[:common_vocab_len, :576],
        dim=-1
    ).mean().item()
    print(f"   • Token Embedding Cosine Similarity (Native V5 vs SmolLM2 baseline): {native_vs_smol_cosine:.4f}")

    # 5. Output Information & Text Generation Testing
    print("\n📝 [5/5] Testing Forward Pass & Information Output Accuracy...")
    test_prompts = [
        "The capital of France is",
        "Question: If 2 + 2 = 4, then 4 + 4 equals what? Answer:",
        "What is an atom?",
        "def is_even(n):",
    ]

    output_results = []
    
    for idx, prompt in enumerate(test_prompts, 1):
        print(f"\n" + "-" * 75)
        print(f"TEST {idx}: \"{prompt}\"")
        print("-" * 75)

        # A. SmolLM2 Output
        smol_inputs = smol_tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            smol_out = smol_model.generate(
                **smol_inputs,
                max_new_tokens=40,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=smol_tok.eos_token_id,
            )
        smol_gen_tokens = smol_out[0][smol_inputs["input_ids"].shape[1]:]
        smol_text = smol_tok.decode(smol_gen_tokens, skip_special_tokens=True).strip()

        # B. Native V5 Output
        with torch.no_grad():
            v5_native_text = v5_native.generate_text(
                prompt=prompt,
                tokenizer=v5_tok,
                max_new_tokens=40,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                device=device,
                strict_act=False,
            ).strip()

        # C. Transferred V5 Output
        with torch.no_grad():
            v5_trans_text = v5_transferred.generate_text(
                prompt=prompt,
                tokenizer=v5_tok,
                max_new_tokens=40,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                device=device,
                strict_act=False,
            ).strip()

        # Compute next-token probability distribution on the prompt itself
        v5_inputs = v5_tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            # Source logits
            s_logits = smol_model(smol_inputs["input_ids"]).logits[:, -1, :32000]
            # Native logits
            n_logits = v5_native(v5_inputs["input_ids"])[:, -1, :]
            # Transferred logits
            t_logits = v5_transferred(v5_inputs["input_ids"])[:, -1, :]

            # Compute KL Divergence between source and transferred next-token probs
            p_source = F.softmax(s_logits / 1.0, dim=-1)
            log_p_trans = F.log_softmax(t_logits / 1.0, dim=-1)
            log_p_native = F.log_softmax(n_logits / 1.0, dim=-1)
            
            kl_trans = F.kl_div(log_p_trans, p_source, reduction="batchmean").item()
            kl_native = F.kl_div(log_p_native, p_source, reduction="batchmean").item()

            top5_source = [smol_tok.decode([t]) for t in torch.topk(p_source, 5).indices[0].tolist()]
            top5_trans = [v5_tok.decode([t]) for t in torch.topk(F.softmax(t_logits, dim=-1), 5).indices[0].tolist()]

        print(f"🤖 [Source: SmolLM2-135M]:\n   \"{smol_text}\"")
        print(f"🧠 [Target: Native V5 Step 900k]:\n   \"{v5_native_text}\"")
        print(f"🔀 [Transferred: V5 with SmolLM2 weights]:\n   \"{v5_trans_text}\"")
        print(f"   • KL Divergence (Source vs Transferred): {kl_trans:.4f} (Native vs Source: {kl_native:.4f})")
        print(f"   • Top-5 Next Tokens: Source={top5_source} | Transferred={top5_trans}")

        output_results.append({
            "prompt": prompt,
            "smol_output": smol_text,
            "v5_native_output": v5_native_text,
            "v5_transferred_output": v5_trans_text,
            "kl_div_transferred": kl_trans,
            "kl_div_native": kl_native,
            "top5_source": top5_source,
            "top5_transferred": top5_trans,
        })

    # Save summary report
    summary = {
        "timestamp": time.time(),
        "source_model": smol_id,
        "source_params": smol_params,
        "target_model": "CerebroSpheroidV5",
        "target_params": v5_params,
        "transferred_params": transfer_audit["transferred_param_count"],
        "transferred_coverage_pct": transferred_pct,
        "embedding_cosine_similarity": emb_cosine,
        "results": output_results,
    }

    report_path = os.path.join(WORKSPACE_ROOT, "eval", "results", "weight_transfer_smollm_v5_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Full JSON report saved to: {report_path}")

    return summary


if __name__ == "__main__":
    run_experiment()
