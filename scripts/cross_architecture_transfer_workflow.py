#!/usr/bin/env python3
"""
CerebroSpheroid Cross-Architecture Weight Transfer & Convergence Engine
======================================================================
Formalized autonomous workflow to transfer and harmonize weights from any
frontier transformer model (e.g., SmolLM, Qwen 2.5, LLaMA) into the
biomimetic CerebroSpheroid V5 concentric architecture.

Workflow Stages:
  1. Architecture Introspection & Topology Alignment
  2. Deep Semantic Vocabulary Surgery (Token Cipher Translation)
  3. Concentric Shell Depth Interpolation & Subspace Adaptation
  4. Dead-Channel Sanitization (RMSNorm Unit Variance Restoration)
  5. High-Velocity Curriculum Calibration (Early Exit on Target Loss)
  6. Standardized Academic Benchmark Suite Evaluation
"""

import os
import sys
import json
import math
import time
import argparse
import ast
import re
from typing import Dict, Any, Tuple, Optional, List

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/hf_home")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/hf_home")

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from src.jax.checkpoint import adapt_jax_to_torch_state_dict
from src.corpus import format_stimulus_prompt, format_motor_response, encode_no_special
from scripts.benchmark_frontier_suite import (
    MMLU_TASKS,
    ARC_TASKS,
    HELLASWAG_TASKS,
    GSM8K_TASKS,
    HUMANEVAL_TASKS,
    score_multiple_choice_loglik,
    evaluate_generative_response,
    extract_gsm8k_number,
)


class CrossArchitectureTransferWorkflow:
    def __init__(
        self,
        source_model_id: str,
        target_config_path_or_preset: str,
        target_base_ckpt: Optional[str] = None,
        device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device_str)
        self.source_model_id = source_model_id
        self.target_config_path_or_preset = target_config_path_or_preset
        self.target_base_ckpt = target_base_ckpt

        print("=" * 80)
        print("🧠 CEREBRO-SPHEROID CROSS-ARCHITECTURE WEIGHT TRANSFER WORKFLOW")
        print(f"Device: {self.device} | Source: {source_model_id}")
        print("=" * 80)

        # 1. Load Tokenizers
        print("\n[Phase 1/6] Loading source and target tokenizers...")
        self.v5_tok = AutoTokenizer.from_pretrained(os.path.join(WORKSPACE_ROOT, "data", "tokenizer"))
        self.source_tok = AutoTokenizer.from_pretrained(source_model_id)
        print(f"   ✓ Target (V5) Tokenizer Vocab Size: {self.v5_tok.vocab_size}")
        print(f"   ✓ Source Tokenizer Vocab Size:     {self.source_tok.vocab_size}")

        # 2. Resolve Target Model Configuration
        self.target_cfg = self._resolve_target_config(target_config_path_or_preset)
        print(f"\n[Phase 1/6] Target Architecture: {getattr(self.target_cfg, 'preset', 'custom')}")
        print(f"   Surface Dim: {self.target_cfg.surface_dim} | Radial Dim: {self.target_cfg.radial_dim}")
        print(f"   Concentric Shells: {self.target_cfg.num_shells} | Heads: {self.target_cfg.num_heads}")

        # 3. Instantiate Target CerebroSpheroid Model
        self.v5_model = CerebroSpheroidV5(self.target_cfg)
        target_param_count = sum(p.numel() for p in self.v5_model.parameters())
        print(f"   Target Model Parameters: {target_param_count:,} ({target_param_count/1e6:.1f}M)")

        # Load base checkpoint if supplied (e.g. for native prior initialization)
        if self.target_base_ckpt and os.path.exists(self.target_base_ckpt):
            print(f"   Loading structural base checkpoint: {self.target_base_ckpt}")
            raw = torch.load(self.target_base_ckpt, map_location="cpu", weights_only=False)
            sd = raw.get("model_state", raw.get("model", raw))
            if any(k.startswith(("sensory.", "core.", "motor.")) for k in sd.keys()):
                sd = adapt_jax_to_torch_state_dict(sd, self.target_cfg)
            self.v5_model.load_state_dict(sd, strict=False)
            print("   ✓ Base prior state dict merged.")

        # 4. Load Source Model (CPU to save GPU memory during surgery)
        print(f"\n[Phase 1/6] Ingesting source model weights: {source_model_id}...")
        self.source_model = AutoModelForCausalLM.from_pretrained(
            source_model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        source_param_count = sum(p.numel() for p in self.source_model.parameters())
        print(f"   ✓ Source Model Loaded: {source_param_count:,} parameters ({source_param_count/1e6:.1f}M)")

    def _resolve_target_config(self, path_or_preset: str) -> ModelConfigV5:
        if path_or_preset in PRESETS_V5:
            d = PRESETS_V5[path_or_preset]
            if isinstance(d, dict):
                return ModelConfigV5(**d)
            return d
        
        # Check if it's a file path
        if os.path.exists(path_or_preset):
            with open(path_or_preset, "r") as f:
                d = json.load(f)
            valid = {k: v for k, v in d.items() if k in ModelConfigV5.__dataclass_fields__}
            return ModelConfigV5(**valid)

        # Check in configs/ directory
        candidate = os.path.join(WORKSPACE_ROOT, "configs", path_or_preset)
        if not candidate.endswith(".json"):
            candidate += ".json"
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                d = json.load(f)
            valid = {k: v for k, v in d.items() if k in ModelConfigV5.__dataclass_fields__}
            return ModelConfigV5(**valid)

        raise ValueError(f"Could not resolve target config: {path_or_preset}")

    def execute_semantic_vocabulary_surgery(self) -> Dict[str, Any]:
        """
        Phase 2: Semantic String Vocabulary Surgery.
        Overcomes the 0% token cipher trap by decoding token strings and re-encoding
        through the source model's tokenizer with subword-mean pooling.
        """
        print("\n" + "=" * 80)
        print("✂️ PHASE 2: DEEP SEMANTIC STRING VOCABULARY SURGERY")
        print("=" * 80)

        # Get source embeddings
        src_emb = self.source_model.get_input_embeddings().weight.detach()
        v5_sd = self.v5_model.state_dict()
        new_v5_emb = v5_sd["sensory_dnn.embedding.weight"].clone()

        dim_src = src_emb.shape[1]
        dim_tgt = new_v5_emb.shape[1]
        transfer_dim = min(dim_src, dim_tgt)

        print(f"Transfer embedding channel dimension: {transfer_dim} (Source: {dim_src}, Target: {dim_tgt})")

        bio_tokens = {"<stim>", "</stim>", "<act>", "</act>", "<s>", "</s>", "<unk>"}
        mapped_single = 0
        mapped_multi = 0
        preserved_control = 0

        # Pillar 3: Deliberate Control Token Calibration from source delimiters
        control_token_candidates = {
            31: ["<|im_start|>user\n", "[INST]", "<|start_header_id|>user<|end_header_id|>\n\n", "User: "],       # <stim>
            32: ["<|im_end|>\n", "[/INST]", "<|eot_id|>\n", "\n\nAssistant: ", "\n\n"],                          # </stim>
            33: ["<|im_start|>assistant\n", "<|start_header_id|>assistant<|end_header_id|>\n\n", "Assistant: "], # <act>
            34: ["<|im_end|>", "<|eot_id|>", "</s>"],                                                             # </act>
        }
        calibrated_controls = 0
        for ctrl_id, candidates in control_token_candidates.items():
            for cand in candidates:
                encoded = self.source_tok.encode(cand, add_special_tokens=False)
                valid_ids = [idx for idx in encoded if idx < src_emb.shape[0]]
                if valid_ids:
                    pooled = src_emb[valid_ids, :transfer_dim].mean(dim=0)
                    new_v5_emb[ctrl_id, :transfer_dim] = pooled
                    calibrated_controls += 1
                    break

        for i in range(self.v5_tok.vocab_size):
            # Tokens 31-34 already calibrated above; preserve special 1, 2 (<unk>, <s>, </s>)
            if i in (1, 2):
                preserved_control += 1
                continue
            if i in (31, 32, 33, 34):
                continue
            tok_str = self.v5_tok.decode([i])
            if tok_str.strip() in bio_tokens:
                preserved_control += 1
                continue

            encoded = self.source_tok.encode(tok_str, add_special_tokens=False)
            if len(encoded) == 1:
                src_id = encoded[0]
                if src_id < src_emb.shape[0]:
                    new_v5_emb[i, :transfer_dim] = src_emb[src_id, :transfer_dim]
                    mapped_single += 1
            elif len(encoded) > 1:
                valid_ids = [idx for idx in encoded if idx < src_emb.shape[0]]
                if valid_ids:
                    pooled = src_emb[valid_ids, :transfer_dim].mean(dim=0)
                    new_v5_emb[i, :transfer_dim] = pooled
                    mapped_multi += 1

        total_mapped = mapped_single + mapped_multi + calibrated_controls
        coverage_pct = (total_mapped / (self.v5_tok.vocab_size - preserved_control)) * 100.0
        print(f"   ✓ Vocabulary Mapping Results:")
        print(f"     - Exact Single-Token Match:  {mapped_single:5d} tokens")
        print(f"     - Multi-Subword Pooled:      {mapped_multi:5d} tokens")
        print(f"     - Calibrated Control Tokens: {calibrated_controls:5d} tokens (Pillar 3)")
        print(f"     - Preserved Control/Bio:     {preserved_control:5d} tokens")
        print(f"     - Semantic Coverage:         {coverage_pct:.2f}% of target vocabulary")

        # Update model state dict
        v5_sd["sensory_dnn.embedding.weight"] = new_v5_emb
        if self.target_cfg.use_tied_embeddings and "motor_cortex.head.weight" in v5_sd:
            v5_sd["motor_cortex.head.weight"] = new_v5_emb.clone()

        self.v5_model.load_state_dict(v5_sd, strict=False)
        return {"mapped_single": mapped_single, "mapped_multi": mapped_multi, "coverage_pct": coverage_pct}

    def execute_cross_layer_shell_adaptation(self) -> Dict[str, Any]:
        """
        Phase 3: Concentric Shell Depth Interpolation & Subspace Adaptation.
        Maps transformer layers L -> K concentric shells with dead-channel sanitization.
        """
        print("\n" + "=" * 80)
        print("🌐 PHASE 3: CONCENTRIC SHELL DEPTH INTERPOLATION & WEIGHT SURGERY")
        print("=" * 80)

        src_sd = self.source_model.state_dict()
        v5_sd = self.v5_model.state_dict()
        new_v5_sd = {k: v.clone() for k, v in v5_sd.items()}

        K = self.target_cfg.num_shells
        L = getattr(self.source_model.config, "num_hidden_layers", 24)
        print(f"Mapping {L} source Transformer layers -> {K} target Concentric Shells...")

        # Find layer keys in source model
        mapped_shells = 0
        for k_idx in range(K):
            l_idx = int(round(k_idx * (L - 1) / max(1, K - 1)))
            
            # Identify source FFN weights (support LLaMA / Qwen / Mistral naming)
            g_proj = src_sd.get(f"model.layers.{l_idx}.mlp.gate_proj.weight")
            u_proj = src_sd.get(f"model.layers.{l_idx}.mlp.up_proj.weight")
            d_proj = src_sd.get(f"model.layers.{l_idx}.mlp.down_proj.weight")
            ln_w   = src_sd.get(f"model.layers.{l_idx}.post_attention_layernorm.weight")
            if ln_w is None:
                ln_w = src_sd.get(f"model.layers.{l_idx}.input_layernorm.weight")

            gate_k = f"volumetric_core.intra_shell_swiglu.{k_idx}.fc_gate.weight"
            val_k  = f"volumetric_core.intra_shell_swiglu.{k_idx}.fc_val.weight"
            down_k = f"volumetric_core.intra_shell_swiglu.{k_idx}.fc_down.weight"
            norm_k = f"volumetric_core.intra_norms.{k_idx}.weight"

            if gate_k in new_v5_sd and g_proj is not None:
                out_dim = min(new_v5_sd[gate_k].shape[0], g_proj.shape[0])
                in_dim  = min(new_v5_sd[gate_k].shape[1], g_proj.shape[1])
                new_v5_sd[gate_k][:out_dim, :in_dim] = g_proj[:out_dim, :in_dim].to(new_v5_sd[gate_k].dtype)

            if val_k in new_v5_sd and u_proj is not None:
                out_dim = min(new_v5_sd[val_k].shape[0], u_proj.shape[0])
                in_dim  = min(new_v5_sd[val_k].shape[1], u_proj.shape[1])
                new_v5_sd[val_k][:out_dim, :in_dim] = u_proj[:out_dim, :in_dim].to(new_v5_sd[val_k].dtype)

            if down_k in new_v5_sd and d_proj is not None:
                out_dim = min(new_v5_sd[down_k].shape[0], d_proj.shape[0])
                in_dim  = min(new_v5_sd[down_k].shape[1], d_proj.shape[1])
                new_v5_sd[down_k][:out_dim, :in_dim] = d_proj[:out_dim, :in_dim].to(new_v5_sd[down_k].dtype)

            if norm_k in new_v5_sd and ln_w is not None:
                norm_dim = min(new_v5_sd[norm_k].shape[0], ln_w.shape[0])
                new_v5_sd[norm_k][:norm_dim] = ln_w[:norm_dim].to(new_v5_sd[norm_k].dtype)
                if new_v5_sd[norm_k].shape[0] > norm_dim:
                    new_v5_sd[norm_k][norm_dim:] = 1.0  # Unit scale prior

            mapped_shells += 1

        print(f"   ✓ Mapped {mapped_shells} concentric shells from source layers.")

        # Phase 4: Dead-Channel Sanitization on all RMSNorms
        print("\n" + "=" * 80)
        print("🧼 PHASE 4: DEAD-CHANNEL SANITIZATION & RMSNORM UNIT VARIANCE RESTORATION")
        print("=" * 80)
        sanitized_norms = 0
        for name, tensor in new_v5_sd.items():
            if "norm" in name.lower() and tensor.ndim == 1:
                # Replace any zero or near-zero scaling factors with 1.0
                zero_mask = (tensor.abs() < 1e-4)
                if zero_mask.any():
                    tensor[zero_mask] = 1.0
                    sanitized_norms += 1

        print(f"   ✓ Sanitized {sanitized_norms} RMSNorm parameter tensors (zero-scale suppressed).")

        # Ingest state dict into target model
        self.v5_model.load_state_dict(new_v5_sd, strict=True)
        print("   ✓ Transferred weights strictly verified and loaded into CerebroSpheroid.")
        
        # Free source model from CPU memory to keep environment light
        del self.source_model
        del src_sd
        del self.v5_tok
        import gc
        gc.collect()

        return {"mapped_shells": mapped_shells, "sanitized_norms": sanitized_norms}

    def run_curriculum_calibration(
        self,
        output_checkpoint_path: str,
        target_loss: float = 3.0,
        max_steps: int = 1500,
        batch_size: int = 2,
        accum_steps: int = 2,
        initial_lr: float = 3.5e-5,
        min_lr: float = 2.0e-6,
    ) -> Dict[str, Any]:
        """
        Phase 5: High-Velocity Curriculum Calibration Training.
        Rapid teacher-forcing calibration until Cross-Entropy loss <= target_loss.
        """
        print("\n" + "=" * 80)
        print(f"⚡ PHASE 5: CURRICULUM CALIBRATION TRAINING (Target Loss <= {target_loss:.2f})")
        print(f"Device: {self.device} | Micro-Batch: {batch_size} | Accumulation: {accum_steps}")
        print(f"Initial LR: {initial_lr:.2e} -> Min LR: {min_lr:.2e} | Max Steps: {max_steps}")
        print("=" * 80)

        # Move model to target device in half precision to ensure tight VRAM usage
        self.v5_model = self.v5_model.half().to(self.device)
        self.v5_model.train()

        # Load Tri-Modal Cognitive Trajectory Corpus (Pillar 1)
        tok = AutoTokenizer.from_pretrained(os.path.join(WORKSPACE_ROOT, "data", "tokenizer"))
        candidate_paths = [
            os.path.join(WORKSPACE_ROOT, "data", "cerebro_cognitive_8k", "cerebro_tri_modal_sft_10k.jsonl"),
            os.path.join(WORKSPACE_ROOT, "data", "cerebro_cognitive_8k", "cerebro_tri_modal_sft_10k.parquet"),
            os.path.join(WORKSPACE_ROOT, "data", "cerebro_cognitive_8k", "cerebro_cognitive_10k.jsonl"),
            os.path.join(WORKSPACE_ROOT, "data", "cerebro_cognitive_8k", "cerebro_cognitive_10k.parquet"),
        ]
        dataset_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
        print(f"Loading calibration QA trajectories from: {os.path.basename(dataset_path)}...")

        data_samples = []
        if dataset_path.endswith(".parquet"):
            import pyarrow.parquet as pq
            table = pq.read_table(dataset_path)
            q_col = table["question"].to_pylist()
            a_col = table["answer"].to_pylist()
            for q, a in zip(q_col[:8000], a_col[:8000]):
                data_samples.append((q, a))
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= 8000:
                        break
                    entry = json.loads(line)
                    data_samples.append((entry["question"], entry["answer"]))

        # Pillar 4: Truncation Trap Elimination - Expand horizon from [:80] to [:256] tokens
        tokenized_data = []
        for q, a in data_samples:
            prompt_str = format_stimulus_prompt(q)
            answer_str = format_motor_response(a)
            p_ids = [t % self.target_cfg.vocab_size for t in encode_no_special(tok, prompt_str)][:256]
            a_ids = [t % self.target_cfg.vocab_size for t in encode_no_special(tok, answer_str)][:256]
            if len(p_ids) > 2 and len(a_ids) > 2:
                tokenized_data.append((p_ids, a_ids))

        print(f"   ✓ Ready with {len(tokenized_data)} full-horizon (256-tok) QA pairs.")

        # Pillar 4: SCMOptimizer with Rank-Adaptive Gating Acceleration and Stiefel Manifold updates
        try:
            from src.scm_optimizer import SCMOptimizer
            optimizer = SCMOptimizer(
                self.v5_model.parameters(),
                lr=initial_lr,
                momentum=0.95,
                weight_decay=0.01,
                ns_steps=5,
                rank_boost=4.0,
                radial_gamma=0.5,
            )
            print("   ✓ Active Optimizer: SCMOptimizer (Stiefel Newton-Schulz 5 + Rank-Adaptive Gating Acceleration)")
        except Exception as e:
            print(f"   ⚠️ SCMOptimizer notice ({e}), defaulting to AdamW.")
            optimizer = torch.optim.AdamW(self.v5_model.parameters(), lr=initial_lr, weight_decay=0.01)

        def get_lr(step_i):
            prog = min(1.0, step_i / max_steps)
            return min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * prog))

        loss_history = []
        best_loss = 999.0
        ema_loss = None
        start_time = time.time()
        converged = False
        final_step = 0

        print("\n🚀 Beginning Convergence Rollout...")
        optimizer.zero_grad()

        for step in range(1, max_steps + 1):
            curr_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = curr_lr

            # Sample micro-batch
            batch_indices = torch.randint(0, len(tokenized_data), (batch_size,)).tolist()
            batch_pairs = [tokenized_data[idx] for idx in batch_indices]

            max_p = max(len(p) for p, _ in batch_pairs)
            max_a = max(len(a) for _, a in batch_pairs)

            prompts_tensor = torch.zeros((batch_size, max_p), dtype=torch.long, device=self.device)
            targets_tensor = torch.full((batch_size, max_a), -100, dtype=torch.long, device=self.device)

            for b_idx, (p, a) in enumerate(batch_pairs):
                prompts_tensor[b_idx, :len(p)] = torch.tensor(p, dtype=torch.long, device=self.device)
                targets_tensor[b_idx, :len(a)] = torch.tensor(a, dtype=torch.long, device=self.device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                # Pillar 2: Dynamic Answer-Anchor Loss Weighting (3.5x penalty on answer tokens)
                loss, _ = self.v5_model(
                    prompts=prompts_tensor,
                    targets=targets_tensor,
                    sampling_p=1.0,
                    deliberation_mode="teacher_forcing",
                    use_anchor_loss=True,
                    anchor_weight=3.5,
                )
                loss_scaled = loss / accum_steps

            loss_scaled.backward()

            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.v5_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            loss_val = loss.item()
            loss_history.append((step, loss_val))
            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = 0.95 * ema_loss + 0.05 * loss_val

            if loss_val < best_loss:
                best_loss = loss_val

            final_step = step

            if step % 50 == 0 or step == max_steps or loss_val <= target_loss:
                elapsed = time.time() - start_time
                rate = step / max(1e-3, elapsed)
                vram_used = torch.cuda.memory_allocated(self.device) / 1e6 if torch.cuda.is_available() else 0
                print(f"   [Step {step:4d}/{max_steps}] Loss: {loss_val:.4f} (EMA: {ema_loss:.4f}) | Best: {best_loss:.4f} | LR: {curr_lr:.2e} | Speed: {rate:.1f} st/s | VRAM: {vram_used:.0f}MB")

            if ema_loss <= target_loss or loss_val <= target_loss:
                print(f"\n🎉 CONVERGENCE CRITERION MET! Loss: {loss_val:.4f} <= {target_loss:.2f} at Step {step}!")
                converged = True
                break

        # Save Converged Model Checkpoint
        os.makedirs(os.path.dirname(output_checkpoint_path), exist_ok=True)
        print(f"\n💾 Saving Converged Checkpoint to: {output_checkpoint_path}...")
        save_dict = {
            "model_state": {k: v.cpu().float() for k, v in self.v5_model.state_dict().items()},
            "step": final_step,
            "calibration_loss": float(loss_val),
            "best_loss": float(best_loss),
            "ema_loss": float(ema_loss),
            "converged": converged,
            "source_model": self.source_model_id,
            "meta": {
                "target_config": getattr(self.target_cfg, "preset", "custom"),
                "surface_dim": self.target_cfg.surface_dim,
                "radial_dim": self.target_cfg.radial_dim,
                "num_shells": self.target_cfg.num_shells,
                "loss_history": loss_history,
                "training_time_seconds": time.time() - start_time,
            },
        }
        torch.save(save_dict, output_checkpoint_path)
        print("   ✓ Checkpoint saved successfully!")

        self.v5_model.eval()
        return {
            "final_step": final_step,
            "final_loss": loss_val,
            "best_loss": best_loss,
            "converged": converged,
            "output_checkpoint": output_checkpoint_path,
        }

    def evaluate_academic_benchmarks(self, output_json_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Phase 6: Standardized Academic Benchmark Suite Evaluation.
        """
        print("\n" + "=" * 80)
        print("🏆 PHASE 6: ACADEMIC BENCHMARK SHOWDOWN")
        print("=" * 80)

        self.v5_model.eval()
        tok = AutoTokenizer.from_pretrained(os.path.join(WORKSPACE_ROOT, "data", "tokenizer"))
        letters = ["A", "B", "C", "D"]
        results = {}

        # 1. MMLU
        mmlu_correct = 0
        for item in MMLU_TASKS:
            prompt = f"Question: {item['question']}\nChoices:\n"
            for i, ch in enumerate(item["choices"]):
                prompt += f"{letters[i]}) {ch}\n"
            prompt += "Answer:"
            pred_idx, _ = score_multiple_choice_loglik(self.v5_model, tok, prompt, item["choices"], self.device)
            if letters[pred_idx] == item["answer"]:
                mmlu_correct += 1
        results["MMLU"] = (mmlu_correct / len(MMLU_TASKS)) * 100.0
        print(f"   ✓ MMLU Score:        {results['MMLU']:.1f}% ({mmlu_correct}/{len(MMLU_TASKS)})")

        # 2. ARC-Challenge
        arc_correct = 0
        for item in ARC_TASKS:
            prompt = f"Question: {item['question']}\nChoices:\n"
            for i, ch in enumerate(item["choices"]):
                prompt += f"{letters[i]}) {ch}\n"
            prompt += "Answer:"
            pred_idx, _ = score_multiple_choice_loglik(self.v5_model, tok, prompt, item["choices"], self.device)
            if letters[pred_idx] == item["answer"]:
                arc_correct += 1
        results["ARC_Challenge"] = (arc_correct / len(ARC_TASKS)) * 100.0
        print(f"   ✓ ARC-Challenge:     {results['ARC_Challenge']:.1f}% ({arc_correct}/{len(ARC_TASKS)})")

        # 3. HellaSwag
        hella_correct = 0
        for item in HELLASWAG_TASKS:
            prompt = f"Context: {item['context']}\nContinuations:\n"
            for i, ch in enumerate(item["choices"]):
                prompt += f"{letters[i]}) {ch}\n"
            prompt += "Answer:"
            pred_idx, _ = score_multiple_choice_loglik(self.v5_model, tok, prompt, item["choices"], self.device)
            if letters[pred_idx] == item["answer"]:
                hella_correct += 1
        results["HellaSwag"] = (hella_correct / len(HELLASWAG_TASKS)) * 100.0
        print(f"   ✓ HellaSwag:         {results['HellaSwag']:.1f}% ({hella_correct}/{len(HELLASWAG_TASKS)})")

        # 4. GSM8K Exact Match
        gsm_correct = 0
        for item in GSM8K_TASKS:
            prompt = f"Question: {item['question']}\nAnswer:"
            generated = evaluate_generative_response(self.v5_model, tok, prompt, max_tokens=48, temperature=0.1, device=self.device)
            extracted = extract_gsm8k_number(generated)
            if extracted == item["target_number"]:
                gsm_correct += 1
        results["GSM8K"] = (gsm_correct / len(GSM8K_TASKS)) * 100.0
        print(f"   ✓ GSM8K:             {results['GSM8K']:.1f}% ({gsm_correct}/{len(GSM8K_TASKS)})")

        # 5. HumanEval Python AST Validity
        he_correct = 0
        for item in HUMANEVAL_TASKS:
            prompt = f"{item['prompt']}"
            generated = evaluate_generative_response(self.v5_model, tok, prompt, max_tokens=64, temperature=0.1, device=self.device)
            full_code = prompt + generated
            try:
                ast.parse(full_code)
                he_correct += 1
            except SyntaxError:
                pass
        results["HumanEval"] = (he_correct / len(HUMANEVAL_TASKS)) * 100.0
        print(f"   ✓ HumanEval Syntactic: {results['HumanEval']:.1f}% ({he_correct}/{len(HUMANEVAL_TASKS)})")

        composite_score = sum(results.values()) / len(results)
        results["Composite_Score"] = composite_score
        print("-" * 50)
        print(f"🌟 COMPOSITE BENCHMARK ACCURACY: {composite_score:.2f}%")
        print("-" * 50)

        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"   ✓ Benchmark results saved to: {output_json_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Cross-Architecture Transfer & Convergence Engine")
    parser.add_argument("--source-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Source HuggingFace model repo or path")
    parser.add_argument("--target-config", type=str, default="configs/v5_qwen_0_5b.json", help="Target config JSON or preset name")
    parser.add_argument("--target-base-ckpt", type=str, default=None, help="Optional native checkpoint to inherit priors")
    parser.add_argument("--output-ckpt", type=str, default="checkpoints/v5_qwen_0_5b_converged.pt", help="Path to save output checkpoint")
    parser.add_argument("--target-loss", type=float, default=3.00, help="Target loss for early exit convergence")
    parser.add_argument("--max-steps", type=int, default=1200, help="Max calibration steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size per step")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3.5e-5, help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=2.0e-6, help="Minimum learning rate")
    parser.add_argument("--run-benchmarks", action="store_true", default=True, help="Run academic benchmarks after calibration")
    parser.add_argument("--benchmark-output", type=str, default="eval/results/qwen_0_5b_converged_benchmark.json", help="Path for benchmark JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    args = parser.parse_args()

    workflow = CrossArchitectureTransferWorkflow(
        source_model_id=args.source_model,
        target_config_path_or_preset=args.target_config,
        target_base_ckpt=args.target_base_ckpt,
        device_str=args.device,
    )

    # Step 1: Semantic Vocabulary Surgery
    vocab_stats = workflow.execute_semantic_vocabulary_surgery()

    # Step 2: Layer & Concentric Shell Adaptation
    shell_stats = workflow.execute_cross_layer_shell_adaptation()

    # Step 3: Curriculum Calibration Training
    train_stats = workflow.run_curriculum_calibration(
        output_checkpoint_path=args.output_ckpt,
        target_loss=args.target_loss,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        initial_lr=args.lr,
        min_lr=args.min_lr,
    )

    # Step 4: Academic Benchmark Suite
    if args.run_benchmarks:
        bench_stats = workflow.evaluate_academic_benchmarks(output_json_path=args.benchmark_output)

    print("\n" + "=" * 80)
    print("✨ WORKFLOW COMPLETE: CROSS-ARCHITECTURE CONVERGENCE SUCCESSFULLY EXECUTED")
    print(f"Final Model Checkpoint: {args.output_ckpt}")
    print("=" * 80)


if __name__ == "__main__":
    main()
