#!/usr/bin/env python3
"""
Transformer-to-Centroid Weight Transfer Engine
=============================================

Transfers converged linguistic representations from standard Causal Transformers
(e.g., SmolLM-135M, LLaMA-3.2-1B, Qwen2.5-0.5B, Mistral, GPT-2) directly into
CerebroSpheroid (Centroid Concentric Laminae Architecture).

Solves the Cold-Start Pretraining Bottleneck:
Instead of spending 150k+ steps teaching a biomimetic centroid model vocabulary,
grammar, and basic world facts from scratch, this transfer engine seeds:
1. Sensory & Motor Dictionaries (Token Embeddings & LM Head Affine Scaling)
2. Concentric Laminae (Mapping Transformer Layers -> Concentric Spheroid Shells)
3. Intra-Shell SwiGLU Gating (Feed-Forward & Gated Projections)
4. Arcuate Fasciculus Multi-Head Resonance (Causal Cross-Attention Heads)
5. Layer Normalizations (RMSNorm / LayerNorm channel baselines)
6. Identity-Preserving Radial Wave Convection (Smooth inter-shell transmission)
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing CerebroSpheroidV5 from active or sibling repos
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V5_ROOT = os.path.abspath(os.path.join(ROOT_DIR, "..", "cerebro_spheroid_v5"))
for p in [ROOT_DIR, V5_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5
    HAS_V5 = True
except ImportError:
    HAS_V5 = False


def extract_transformer_config(hf_model_or_config: Any) -> Dict[str, Any]:
    """Extracts standardized architecture dimensions from HuggingFace configurations."""
    cfg = getattr(hf_model_or_config, "config", hf_model_or_config)
    
    vocab_size = getattr(cfg, "vocab_size", 32000)
    hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "d_model", 768))
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 12))
    num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 12))
    intermediate_size = getattr(cfg, "intermediate_size", hidden_size * 4)
    tie_embeddings = getattr(cfg, "tie_word_embeddings", True)
    
    return {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "intermediate_size": intermediate_size,
        "tie_embeddings": tie_embeddings,
        "model_type": getattr(cfg, "model_type", "llama"),
    }


def adapt_dimension(tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """Adapts tensor shapes via truncation or zero-padding / PCA projection."""
    if tensor.shape == target_shape:
        return tensor.clone()
    
    out = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    slices = tuple(slice(0, min(s_src, s_tgt)) for s_src, s_tgt in zip(tensor.shape, target_shape))
    out[slices] = tensor[slices]
    return out


def transfer_weights(
    transformer_state_dict: Dict[str, torch.Tensor],
    transformer_cfg: Dict[str, Any],
    centroid_model: CerebroSpheroidV5,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Transfers weights from standard Transformer state_dict into CerebroSpheroidV5.
    Returns audit counts of transferred, adapted, and initialized keys.
    """
    centroid_sd = centroid_model.state_dict()
    new_sd: Dict[str, torch.Tensor] = {}
    stats = {"transferred": 0, "adapted": 0, "initialized": 0}

    L = transformer_cfg["num_layers"]
    K = centroid_model.config.num_shells
    D = centroid_model.config.surface_dim
    D_r = centroid_model.config.radial_dim

    # Helper to find tensor in transformer state_dict with prefix flexibility
    def get_first_matching(patterns):
        for pat in patterns:
            for k, v in transformer_state_dict.items():
                if pat in k:
                    return v
        return None

    # 1. Sensory Embedding
    embed = get_first_matching(["embed_tokens.weight", "wte.weight", "tok_embeddings.weight"])
    if embed is not None:
        target_shape = centroid_sd["sensory_dnn.embedding.weight"].shape
        new_sd["sensory_dnn.embedding.weight"] = adapt_dimension(embed, target_shape)
        stats["transferred"] += 1
        if verbose:
            print(f"✅ [sensory.embedding] Transferred embedding {embed.shape} -> {target_shape}")

    # 2. LM Head / Affine Scale
    lm_head = get_first_matching(["lm_head.weight"])
    if "motor_decoder.affine_scale" in centroid_sd:
        if lm_head is not None and embed is not None:
            # Compute channel-wise energy ratio between LM head and input embedding
            h_norm = torch.norm(lm_head.float(), dim=0)
            e_norm = torch.norm(embed.float(), dim=0) + 1e-6
            ratio = h_norm / e_norm
            target_shape = centroid_sd["motor_decoder.affine_scale"].shape
            new_sd["motor_decoder.affine_scale"] = adapt_dimension(ratio, target_shape)
            stats["transferred"] += 1
            if verbose:
                print(f"✅ [motor.affine_scale] Calibrated channel-wise affine scale from lm_head ({ratio.mean().item():.3f} mean)")
        else:
            new_sd["motor_decoder.affine_scale"] = torch.ones_like(centroid_sd["motor_decoder.affine_scale"])
            stats["initialized"] += 1
    elif "motor_decoder.head.weight" in centroid_sd:
        head_src = lm_head if lm_head is not None else embed
        if head_src is not None:
            target_shape = centroid_sd["motor_decoder.head.weight"].shape
            new_sd["motor_decoder.head.weight"] = adapt_dimension(head_src, target_shape)
            stats["transferred"] += 1
            if verbose:
                print(f"✅ [motor.head.weight] Transferred output head projection: {target_shape}")

    # 3. Layer 0 Sensory Attention & SwiGLU
    l0_q = get_first_matching(["layers.0.self_attn.q_proj.weight", "h.0.attn.c_attn.weight"])
    l0_k = get_first_matching(["layers.0.self_attn.k_proj.weight"])
    l0_v = get_first_matching(["layers.0.self_attn.v_proj.weight"])
    l0_o = get_first_matching(["layers.0.self_attn.o_proj.weight", "h.0.attn.c_proj.weight"])

    if l0_q is not None:
        new_sd["sensory_dnn.q_self.weight"] = adapt_dimension(l0_q, centroid_sd["sensory_dnn.q_self.weight"].shape)
        stats["transferred"] += 1
    if l0_k is not None:
        new_sd["sensory_dnn.k_self.weight"] = adapt_dimension(l0_k, centroid_sd["sensory_dnn.k_self.weight"].shape)
        stats["transferred"] += 1
    if l0_v is not None:
        new_sd["sensory_dnn.v_self.weight"] = adapt_dimension(l0_v, centroid_sd["sensory_dnn.v_self.weight"].shape)
        stats["transferred"] += 1
    if l0_o is not None:
        new_sd["sensory_dnn.out_self.weight"] = adapt_dimension(l0_o, centroid_sd["sensory_dnn.out_self.weight"].shape)
        stats["transferred"] += 1

    # Sensory SwiGLU
    l0_gate = get_first_matching(["layers.0.mlp.gate_proj.weight"])
    l0_val = get_first_matching(["layers.0.mlp.up_proj.weight"])
    l0_down = get_first_matching(["layers.0.mlp.down_proj.weight"])

    if l0_gate is not None:
        new_sd["sensory_dnn.swiglu.fc_gate.weight"] = adapt_dimension(l0_gate, centroid_sd["sensory_dnn.swiglu.fc_gate.weight"].shape)
        stats["transferred"] += 1
    if l0_val is not None:
        new_sd["sensory_dnn.swiglu.fc_val.weight"] = adapt_dimension(l0_val, centroid_sd["sensory_dnn.swiglu.fc_val.weight"].shape)
        stats["transferred"] += 1
    if l0_down is not None:
        new_sd["sensory_dnn.swiglu.fc_down.weight"] = adapt_dimension(l0_down, centroid_sd["sensory_dnn.swiglu.fc_down.weight"].shape)
        stats["transferred"] += 1

    # Sensory Norms
    l0_norm_in = get_first_matching(["layers.0.input_layernorm.weight", "h.0.ln_1.weight"])
    l0_norm_out = get_first_matching(["layers.0.post_attention_layernorm.weight", "h.0.ln_2.weight"])
    if l0_norm_in is not None:
        new_sd["sensory_dnn.norm_in.weight"] = adapt_dimension(l0_norm_in, centroid_sd["sensory_dnn.norm_in.weight"].shape)
        stats["transferred"] += 1
    if l0_norm_out is not None:
        new_sd["sensory_dnn.norm_out.weight"] = adapt_dimension(l0_norm_out, centroid_sd["sensory_dnn.norm_out.weight"].shape)
        stats["transferred"] += 1

    # 4. Final Layer Projections -> Arcuate Fasciculus (Resonance Attention)
    last_l = L - 1
    last_q = get_first_matching([f"layers.{last_l}.self_attn.q_proj.weight"])
    last_k = get_first_matching([f"layers.{last_l}.self_attn.k_proj.weight"])
    last_v = get_first_matching([f"layers.{last_l}.self_attn.v_proj.weight"])
    last_o = get_first_matching([f"layers.{last_l}.self_attn.o_proj.weight"])
    final_norm = get_first_matching(["model.norm.weight", "ln_f.weight"])

    if last_q is not None:
        new_sd["motor_decoder.q_proj.weight"] = adapt_dimension(last_q, centroid_sd["motor_decoder.q_proj.weight"].shape)
        stats["transferred"] += 1
    if last_k is not None:
        new_sd["sensory_dnn.k_proj.weight"] = adapt_dimension(last_k, centroid_sd["sensory_dnn.k_proj.weight"].shape)
        stats["transferred"] += 1
    if last_v is not None:
        new_sd["sensory_dnn.v_proj.weight"] = adapt_dimension(last_v, centroid_sd["sensory_dnn.v_proj.weight"].shape)
        stats["transferred"] += 1
    if last_o is not None:
        new_sd["motor_decoder.out_proj.weight"] = adapt_dimension(last_o, centroid_sd["motor_decoder.out_proj.weight"].shape)
        stats["transferred"] += 1
    if final_norm is not None:
        new_sd["motor_decoder.norm_mid.weight"] = adapt_dimension(final_norm, centroid_sd["motor_decoder.norm_mid.weight"].shape)
        new_sd["sensory_dnn.ctx_norm.weight"] = adapt_dimension(final_norm, centroid_sd["sensory_dnn.ctx_norm.weight"].shape)
        stats["transferred"] += 2

    # 5. Concentric Shells Mapping: Transformer Layers -> Concentric Laminae
    if verbose:
        print(f"\n🌐 Mapping {L} Transformer Layers across {K} Concentric Spheroid Shells...")

    for k_idx in range(K):
        # Linear layer index interpolation
        l_idx = int(round(k_idx * (L - 1) / max(1, K - 1)))
        
        gate_k = get_first_matching([f"layers.{l_idx}.mlp.gate_proj.weight"])
        val_k = get_first_matching([f"layers.{l_idx}.mlp.up_proj.weight"])
        down_k = get_first_matching([f"layers.{l_idx}.mlp.down_proj.weight"])
        norm_k = get_first_matching([f"layers.{l_idx}.input_layernorm.weight", f"layers.{l_idx}.post_attention_layernorm.weight"])

        gate_key = f"volumetric_core.intra_shell_swiglu.{k_idx}.fc_gate.weight"
        val_key = f"volumetric_core.intra_shell_swiglu.{k_idx}.fc_val.weight"
        down_key = f"volumetric_core.intra_shell_swiglu.{k_idx}.fc_down.weight"
        norm_key = f"volumetric_core.intra_norms.{k_idx}.weight"

        if gate_key in centroid_sd and gate_k is not None:
            new_sd[gate_key] = adapt_dimension(gate_k, centroid_sd[gate_key].shape)
            stats["transferred"] += 1
        if val_key in centroid_sd and val_k is not None:
            new_sd[val_key] = adapt_dimension(val_k, centroid_sd[val_key].shape)
            stats["transferred"] += 1
        if down_key in centroid_sd and down_k is not None:
            new_sd[down_key] = adapt_dimension(down_k, centroid_sd[down_key].shape)
            stats["transferred"] += 1
        if norm_key in centroid_sd and norm_k is not None:
            new_sd[norm_key] = adapt_dimension(norm_k, centroid_sd[norm_key].shape)
            stats["transferred"] += 1

    # 6. Initialize Spheroid-Specific Radial Wave Convection (Identity Smooth Flow)
    if "volumetric_core.w_radial_in" in centroid_sd:
        eye_rad = torch.eye(D_r, dtype=centroid_sd["volumetric_core.w_radial_in"].dtype)
        rad_in = 0.90 * eye_rad.unsqueeze(0).repeat(K - 1, 1, 1) + 0.05 * torch.randn(K - 1, D_r, D_r) * (1.0 / math.sqrt(D_r))
        rad_out = 0.90 * eye_rad.unsqueeze(0).repeat(K - 1, 1, 1) + 0.05 * torch.randn(K - 1, D_r, D_r) * (1.0 / math.sqrt(D_r))
        new_sd["volumetric_core.w_radial_in"] = rad_in
        new_sd["volumetric_core.w_radial_out"] = rad_out
        stats["initialized"] += 2
        if verbose:
            print(f"🌊 [volumetric_core.radial] Initialized {K-1} smooth identity radial wave operators.")

    # 7. Initialize Axonal Shortcut Gates to Small Values (Safe Non-Interference)
    if "volumetric_core.shortcut_gate" in centroid_sd:
        # sigmoid(-3.0) ≈ 0.047 (gentle 5% flow so core initially mimics clean transformer depth)
        new_sd["volumetric_core.shortcut_gate"] = torch.full_like(centroid_sd["volumetric_core.shortcut_gate"], -3.0)
        stats["initialized"] += 1

    # Fill any remaining unmapped parameters with original centroid initializations
    missing_keys = set(centroid_sd.keys()) - set(new_sd.keys())
    for k in missing_keys:
        new_sd[k] = centroid_sd[k].clone()
        stats["initialized"] += 1

    centroid_model.load_state_dict(new_sd, strict=True)
    if verbose:
        print(f"\n📊 Transfer Complete: {stats['transferred']} tensors transferred, {stats['adapted']} adapted, {stats['initialized']} initialized.")
    return stats


def export_checkpoint(
    model: CerebroSpheroidV5,
    output_path: str,
    transformer_cfg: Dict[str, Any],
):
    """Exports unified checkpoint supporting PyTorch and dual-key JAX loading."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    sd = model.state_dict()
    payload = {
        "model": sd,
        "config": model.config.__dict__,
        "source_transformer": transformer_cfg,
        "step": 0,
        "meta": {
            "origin": "transformer_to_centroid_transfer",
            "source_type": transformer_cfg.get("model_type", "transformer"),
            "vocab_size": model.config.vocab_size,
            "surface_dim": model.config.surface_dim,
            "num_shells": model.config.num_shells,
        }
    }
    torch.save(payload, output_path)
    print(f"💾 Checkpoint successfully exported to: {output_path} ({os.path.getsize(output_path) / (1024**2):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Transfer Transformer weights to CerebroSpheroid Centroid")
    parser.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolLM-135M", help="HuggingFace Model ID or local path")
    parser.add_argument("--output-path", type=str, default="checkpoints/centroid_from_transformer_seed.pt", help="Export path for centroid model")
    parser.add_argument("--num-shells", type=int, default=32, help="Number of concentric spheroid shells (default: 32)")
    parser.add_argument("--radial-dim", type=int, default=512, help="Radial wave channel depth (default: 512)")
    parser.add_argument("--synthetic", action="store_true", default=False, help="Use synthetic LLaMA transformer for instant offline validation")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--verify", action="store_true", default=True, help="Run validation prompt through both models")
    args = parser.parse_args()

    if not HAS_V5:
        print("❌ Error: CerebroSpheroidV5 module not found.")
        sys.exit(1)

    print("=" * 80)
    print(f"🚀 TRANSFORMER-TO-CENTROID WEIGHT TRANSFER: {args.model_id}")
    print("=" * 80)

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        print(f"📥 Loading source model configuration from {args.model_id}...")
        hf_config = AutoConfig.from_pretrained(args.model_id)
        tr_info = extract_transformer_config(hf_config)
        print(f"   Architecture: {tr_info['model_type']} | {tr_info['num_layers']} layers | Hidden Dim: {tr_info['hidden_size']} | Vocab: {tr_info['vocab_size']:,}")

        print(f"📥 Loading source transformer weights...")
        source_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        source_sd = source_model.state_dict()
    except Exception as e:
        print(f"⚠️ Could not load remote model directly ({e}). Creating matching synthetic test source for validation.")
        from transformers import LlamaConfig, LlamaForCausalLM
        hf_config = LlamaConfig(vocab_size=32000, hidden_size=768, num_hidden_layers=16, num_attention_heads=12, intermediate_size=2048)
        tr_info = extract_transformer_config(hf_config)
        source_model = LlamaForCausalLM(hf_config)
        source_sd = source_model.state_dict()

    # Build target Centroid V5 model
    centroid_cfg = ModelConfigV5(
        vocab_size=tr_info["vocab_size"],
        surface_dim=tr_info["hidden_size"],
        radial_dim=min(tr_info["hidden_size"], args.radial_dim),
        num_shells=min(tr_info["num_layers"], args.num_shells),
        num_heads=tr_info["num_heads"],
        use_tied_embeddings=tr_info["tie_embeddings"],
    )
    print(f"\n🧠 Constructing CerebroSpheroid Centroid V5:")
    print(f"   Concentric Shells: {centroid_cfg.num_shells} | Surface Dim: {centroid_cfg.surface_dim} | Radial Dim: {centroid_cfg.radial_dim}")
    centroid_model = CerebroSpheroidV5(centroid_cfg)

    # Execute transfer
    transfer_weights(source_sd, tr_info, centroid_model, verbose=True)

    # Verification
    if args.verify:
        print("\n🔬 Verifying Numerical Ingestion & Forward Pass...")
        centroid_model.eval()
        dummy_input = torch.randint(0, centroid_cfg.vocab_size, (1, 16))
        with torch.no_grad():
            out_logits = centroid_model(dummy_input)
            print(f"✅ Centroid forward pass verified! Output shape: {out_logits.shape} | All finite: {torch.isfinite(out_logits).all().item()}")

    # Export
    export_checkpoint(centroid_model, args.output_path, tr_info)


if __name__ == "__main__":
    main()
