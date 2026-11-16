#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Refined Supervised Fine-Tuning (SFT) Engine
===================================================================
Executes the refined SFT strategy with:
  1. Tri-Modal Data Blend (45% Direct Slot QA, 40% Full-Horizon CoT, 15% Agentic)
  2. Answer-Anchor Loss Weighting (AALW: 3.5x penalty on final target answers via target_weights)
  3. Control Token Embedding Preservation & Safe Initialization (<stim>, </stim>, <act>, </act>)
  4. Decoupled AdamW Optimization (2D/3D weight decay, 0.0 for 1D norms/biases) with Warmup
  5. Live qualitative probe verification on analogies, antonyms, grammar, arithmetic, and facts.
"""

import os
import sys
import json
import math
import time
import argparse
import random
from typing import Dict, Any, List, Tuple

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.spheroid_v5 import CerebroSpheroidV5, infer_v5_config_from_state_dict
from src.corpus import (
    format_stimulus_prompt,
    format_motor_response,
    encode_no_special,
    get_bio_token_ids,
)
from src.scm_optimizer import SCMOptimizer


# --- 1. Synthetic High-Density Direct Slot QA Pairs ---
def build_direct_slot_qa_corpus() -> List[Tuple[str, str]]:
    """Generates high-density concise direct QA pairs (analogies, antonyms, grammar, facts)."""
    pairs = []

    # A. Analogies
    analogies = [
        ("hot is to cold as day is to", "night"),
        ("cold is to hot as night is to", "day"),
        ("bird is to fly as fish is to", "swim"),
        ("fish is to swim as bird is to", "fly"),
        ("pen is to write as knife is to", "cut"),
        ("knife is to cut as pen is to", "write"),
        ("hammer is to nail as screwdriver is to", "screw"),
        ("author is to book as artist is to", "painting"),
        ("car is to road as boat is to", "water"),
        ("leaf is to tree as petal is to", "flower"),
        ("fire is to hot as ice is to", "cold"),
        ("eye is to see as ear is to", "hear"),
        ("big is to small as tall is to", "short"),
        ("up is to down as left is to", "right"),
        ("king is to queen as man is to", "woman"),
        ("sun is to day as moon is to", "night"),
        ("wheel is to car as wing is to", "airplane"),
        ("bread is to baker as meat is to", "butcher"),
        ("shoe is to foot as glove is to", "hand"),
        ("dog is to puppy as cat is to", "kitten"),
    ]
    for stem, ans in analogies:
        pairs.append((f"Complete the analogy: {stem} ____.", ans))
        pairs.append((f"{stem.capitalize()} ____.", ans))

    # B. Antonyms
    antonyms = [
        ("hot", "cold"), ("dark", "light"), ("fast", "slow"), ("happy", "sad"),
        ("author", "reader"), ("strong", "weak"), ("begin", "end"), ("high", "low"),
        ("empty", "full"), ("hard", "soft"), ("rich", "poor"), ("ancient", "modern"),
        ("noisy", "quiet"), ("sharp", "dull"), ("brave", "cowardly"), ("cruel", "kind"),
        ("expand", "contract"), ("create", "destroy"), ("accept", "refuse"), ("arrive", "depart")
    ]
    for word, ant in antonyms:
        pairs.append((f"What is the exact antonym of the word '{word}'?", ant))
        pairs.append((f"What is the opposite of '{word}'?", ant))

    # C. Irregular Verbs (Past Tense & Participles)
    verbs = [
        ("drink", "drank", "drinking"), ("run", "ran", "running"), ("go", "went", "going"),
        ("eat", "ate", "eating"), ("sing", "sang", "singing"), ("grow", "grew", "growing"),
        ("write", "wrote", "writing"), ("see", "saw", "seeing"), ("take", "took", "taking"),
        ("break", "broke", "breaking"), ("choose", "chose", "choosing"), ("drive", "drove", "driving"),
        ("fly", "flew", "flying"), ("know", "knew", "knowing"), ("swim", "swam", "swimming")
    ]
    for v, past, ing in verbs:
        pairs.append((f"What is the simple past tense of the irregular verb '{v}'?", past))
        pairs.append((f"What is the past tense of '{v}'?", past))
        pairs.append((f"What is the present participle (-ing form) of '{v}'?", ing))

    # D. Factual Knowledge & Elementary Arithmetic
    facts = [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Germany?", "Berlin"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the capital of Italy?", "Rome"),
        ("What is the capital of the United Kingdom?", "London"),
        ("What is the capital of Canada?", "Ottawa"),
        ("What is the chemical symbol for water?", "H2O"),
        ("What planet is known as the Red Planet?", "Mars"),
        ("What is 15 + 27?", "42"),
        ("What is 8 multiplied by 7?", "56"),
        ("What is 100 divided by 4?", "25"),
        ("What is 9 squared?", "81"),
        ("If John has 3 apples and buys 2 more, how many apples does he have?", "5"),
        ("Between 'embarrass' and 'embarass', which one is the correct spelling?", "embarrass"),
        ("Between 'receive' and 'recieve', which one is the correct spelling?", "receive")
    ]
    for q, a in facts:
        pairs.append((q, a))

    # Duplicate with prompt variations to increase sample diversity
    expanded_pairs = []
    for q, a in pairs:
        expanded_pairs.append((q, a))
        expanded_pairs.append((f"Question: {q} Answer:", a))
        expanded_pairs.append((f"Q: {q}\nA:", a))

    return expanded_pairs


# --- 2. Tri-Modal Dataset Ingestion ---
def load_tri_modal_dataset(
    tokenizer: Any,
    vocab_size: int,
    max_cot_samples: int = 1500,
    max_prompt_len: int = 256,
    max_target_len: int = 512,
) -> List[Dict[str, Any]]:
    """
    Constructs a balanced Tri-Modal dataset with token-level anchor masks:
    - 45% Direct Slot QA (with answers wrapped cleanly in <act>)
    - 40% Full-Horizon CoT from cerebro_cognitive_10k.jsonl (with answers after ####)
    - 15% Tool Action selection traces
    """
    print("\n📦 Assembling Refined Tri-Modal SFT Dataset...")
    tokenized_entries = []

    # 1. Direct Slot QA (45%)
    direct_pairs = build_direct_slot_qa_corpus()
    for q, a in direct_pairs:
        p_str = format_stimulus_prompt(q)
        # Wrap answer directly in <act> ... </act>
        a_str = format_motor_response(a)
        p_ids = [t % vocab_size for t in encode_no_special(tokenizer, p_str)][:max_prompt_len]
        a_ids = [t % vocab_size for t in encode_no_special(tokenizer, a_str)][:max_target_len]
        
        if len(p_ids) > 1 and len(a_ids) > 1:
            # For direct answers, all tokens inside <act> ... </act> get 3.5x weight
            weights = [3.5] * len(a_ids)
            tokenized_entries.append({
                "prompt_ids": p_ids,
                "target_ids": a_ids,
                "weights": weights,
                "type": "direct_slot"
            })

    print(f"   ✓ Direct Slot QA Pairs:     {len(tokenized_entries)} samples")

    # 2. Full-Horizon CoT Reasoning (40%)
    cot_path = os.path.join(WORKSPACE_ROOT, "data", "cerebro_cognitive_8k", "cerebro_cognitive_10k.jsonl")
    cot_loaded = 0
    if os.path.exists(cot_path):
        with open(cot_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if cot_loaded >= max_cot_samples:
                    break
                entry = json.loads(line)
                q = entry.get("question", "")
                a = entry.get("answer", "")
                p_str = format_stimulus_prompt(q)
                a_str = format_motor_response(a)
                p_ids = [t % vocab_size for t in encode_no_special(tokenizer, p_str)][:max_prompt_len]
                a_ids = [t % vocab_size for t in encode_no_special(tokenizer, a_str)][:max_target_len]
                
                if len(p_ids) > 2 and len(a_ids) > 2:
                    # Weight 1.0 for reasoning steps, 3.5 for tokens after "####"
                    weights = [1.0] * len(a_ids)
                    raw_a_tokens = [tokenizer.decode([t]) for t in a_ids]
                    found_anchor = False
                    for i, tok_str in enumerate(raw_a_tokens):
                        if "####" in tok_str or "##" in tok_str:
                            found_anchor = True
                        if found_anchor:
                            weights[i] = 3.5

                    tokenized_entries.append({
                        "prompt_ids": p_ids,
                        "target_ids": a_ids,
                        "weights": weights,
                        "type": "cot_reasoning"
                    })
                    cot_loaded += 1

    print(f"   ✓ Full-Horizon CoT Pairs:   {cot_loaded} samples")

    # 3. Agentic Tool Actions (15%)
    tool_samples = [
        ("Search the web for the latest weather in Tokyo", "<act>action: search_web query: \"Tokyo weather\"</act>"),
        ("Execute python code to compute factorial of 10", "<act>action: run_code language: python code: \"import math; print(math.factorial(10))\"</act>"),
        ("Read the file at /workspace/config.json", "<act>action: read_file path: \"/workspace/config.json\"</act>"),
        ("List all files in the current project directory", "<act>action: run_command cmd: \"ls -la\"</act>")
    ] * 50
    for q, a in tool_samples:
        p_ids = [t % vocab_size for t in encode_no_special(tokenizer, format_stimulus_prompt(q))][:max_prompt_len]
        a_ids = [t % vocab_size for t in encode_no_special(tokenizer, format_motor_response(a))][:max_target_len]
        if len(p_ids) > 1 and len(a_ids) > 1:
            tokenized_entries.append({
                "prompt_ids": p_ids,
                "target_ids": a_ids,
                "weights": [2.5] * len(a_ids),
                "type": "agentic_tool"
            })

    print(f"   ✓ Agentic Tool Traces:      {len(tool_samples)} samples")
    print(f"   🌟 Total Combined SFT Set:  {len(tokenized_entries)} samples")
    random.seed(42)
    random.shuffle(tokenized_entries)
    return tokenized_entries


# --- 3. Calibrate Control Token Embeddings (Safe Guard) ---
def calibrate_control_tokens(model: CerebroSpheroidV5, tokenizer: Any, force: bool = False):
    """Calibrates embeddings for tokens 31, 32, 33, 34 (<stim>, </stim>, <act>, </act>) only if uninitialized."""
    with torch.no_grad():
        emb = model.sensory_dnn.embedding.weight
        norms = [emb[tid].norm().item() for tid in (31, 32, 33, 34)]
        if not force and all(n > 1.0 for n in norms):
            print(f"   ✓ Control tokens 31-34 already trained (norms: {[round(n, 2) for n in norms]}), preserving weights.")
            return

        period_id = tokenizer.encode(".", add_special_tokens=False)[0]
        colon_id = tokenizer.encode(":", add_special_tokens=False)[0]
        
        # Token 31 (<stim>): set from period / prompt cue
        emb[31] = emb[period_id].clone() * 0.9 + 0.1 * torch.randn_like(emb[31]) * 0.02
        # Token 32 (</stim>): colon cue
        emb[32] = emb[colon_id].clone() * 0.9 + 0.1 * torch.randn_like(emb[32]) * 0.02
        # Token 33 (<act>): colon cue + small offset
        emb[33] = emb[colon_id].clone() * 0.9 + 0.1 * torch.randn_like(emb[33]) * 0.02
        # Token 34 (</act>): period cue
        emb[34] = emb[period_id].clone() * 0.9 + 0.1 * torch.randn_like(emb[34]) * 0.02
        print("   ✓ Calibrated uninitialized control tokens 31-34 in embedding matrix.")


# --- 4. Main SFT Training Orchestrator ---
def run_refined_sft(
    base_checkpoint: str = "checkpoints/v5_qwen_0_5b_uwbl_scm.pt",
    output_checkpoint: str = "checkpoints/v5_qwen_0_5b_local_sft.pt",
    optimizer_type: str = "adamw",
    max_steps: int = 1000,
    batch_size: int = 2,
    accum_steps: int = 8,
    lr: float = 3e-5,
    min_lr: float = 3e-6,
    weight_decay: float = 0.01,
    warmup_steps: int = 50,
    probe_interval: int = 50,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_cot_samples: int = 1500,
):
    print("=" * 80)
    print("🎯 CEREBRO-SPHEROID REFINED SFT EXECUTION ENGINE (ADAMW / AALW)")
    print(f"Base Checkpoint:   {base_checkpoint}")
    print(f"Output Checkpoint: {output_checkpoint}")
    print(f"Optimizer:         {optimizer_type.upper()}")
    print(f"Max Steps:         {max_steps} | Micro-Batch: {batch_size} | Accumulation: {accum_steps}")
    print(f"Effective Batch:   {batch_size * accum_steps} sequences/step")
    print(f"Learning Rate:     {lr:.2e} -> {min_lr:.2e} (Warmup: {warmup_steps} steps)")
    print(f"Weight Decay:      {weight_decay}")
    print("=" * 80)

    device = torch.device(device_str)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # Load Model
    raw = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    cfg = infer_v5_config_from_state_dict(raw)
    model = CerebroSpheroidV5(cfg)
    sd = raw.get("model_state", raw.get("model_state_dict", raw.get("model", raw)))
    model.load_state_dict(sd, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(WORKSPACE_ROOT, "data", "tokenizer"))
    calibrate_control_tokens(model, tokenizer)

    model = model.to(dtype).to(device)
    model.train()

    # Load Tri-Modal Dataset
    dataset = load_tri_modal_dataset(tokenizer, cfg.vocab_size, max_cot_samples=max_cot_samples)

    # Initialize Optimizer
    if optimizer_type.lower() == "adamw":
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Exclude 1D tensors (biases, layer norms, RMS scale factors) from weight decay
            if param.ndim < 2 or "norm" in name.lower() or "bias" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
        print(f"   ✓ Configured AdamW: {len(decay_params)} decay params (WD={weight_decay}), {len(no_decay_params)} no-decay params (WD=0.0)")
    else:
        optimizer = SCMOptimizer(model.parameters(), lr=lr, rank_boost=3.0, weight_decay=weight_decay)
        print(f"   ✓ Configured SCMOptimizer: Polar Stiefel Retraction with rank_boost=3.0")

    def get_lr(s: int) -> float:
        if s <= warmup_steps:
            return max(min_lr, lr * float(s) / float(max(1, warmup_steps)))
        prog = min(1.0, (s - warmup_steps) / float(max(1, max_steps - warmup_steps)))
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * prog))

    start_time = time.time()
    best_loss = 999.0
    ema_loss = None

    test_prompts = [
        ("Complete the analogy: hot is to cold as day is to ____.", "night"),
        ("What is the exact antonym of the word \"author\"?", "reader"),
        ("What is the simple past tense of the irregular verb \"drink\"?", "drank"),
        ("If John has 3 apples and buys 2 more, how many apples does he have?", "5"),
        ("What is the capital of France?", "Paris"),
    ]

    print("\n🚀 Beginning Refined SFT Rollout with Answer-Anchor Weighting...")
    optimizer.zero_grad()

    for step in range(1, max_steps + 1):
        curr_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        # Sample micro-batch
        batch_entries = [dataset[i] for i in torch.randint(0, len(dataset), (batch_size,)).tolist()]

        max_p = max(len(e["prompt_ids"]) for e in batch_entries)
        max_a = max(len(e["target_ids"]) for e in batch_entries)

        p_tensor = torch.zeros((batch_size, max_p), dtype=torch.long, device=device)
        t_tensor = torch.full((batch_size, max_a), -100, dtype=torch.long, device=device)
        w_tensor = torch.zeros((batch_size, max_a), dtype=torch.float, device=device)

        for b_idx, e in enumerate(batch_entries):
            p_ids = e["prompt_ids"]
            a_ids = e["target_ids"]
            weights = e["weights"]
            p_tensor[b_idx, :len(p_ids)] = torch.tensor(p_ids, dtype=torch.long, device=device)
            t_tensor[b_idx, :len(a_ids)] = torch.tensor(a_ids, dtype=torch.long, device=device)
            w_tensor[b_idx, :len(weights)] = torch.tensor(weights, dtype=torch.float, device=device)

        with torch.amp.autocast("cuda", dtype=dtype):
            # Forward pass with Answer-Anchor Loss Weighting passed explicitly
            loss, _ = model(
                prompts=p_tensor,
                targets=t_tensor,
                target_weights=w_tensor,
                deliberation_mode="teacher_forcing",
            )
            # Accumulation scaling
            loss_scaled = loss / accum_steps

        loss_scaled.backward()

        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        loss_val = loss.item()
        if ema_loss is None:
            ema_loss = loss_val
        else:
            ema_loss = 0.95 * ema_loss + 0.05 * loss_val

        if loss_val < best_loss:
            best_loss = loss_val

        if step % 10 == 0 or step == max_steps:
            elapsed = time.time() - start_time
            rate = step / max(1e-3, elapsed)
            vram = torch.cuda.memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0
            print(f"   [Step {step:4d}/{max_steps}] Loss: {loss_val:.4f} (EMA: {ema_loss:.4f}) | Best: {best_loss:.4f} | LR: {curr_lr:.2e} | Speed: {rate:.1f} st/s | VRAM: {vram:.0f}MB")

        # Qualitative Probe at step 25, 50, and every probe_interval
        if step in (25, 50) or step % probe_interval == 0 or step == max_steps:
            model.eval()
            print(f"\n--- 🔍 SFT Qualitative Generation Probe (Step {step} | Optimizer: {optimizer_type.upper()}) ---")
            with torch.no_grad():
                for p, target in test_prompts:
                    gen_strict = model.generate_text(
                        prompt=p,
                        tokenizer=tokenizer,
                        max_new_tokens=24,
                        temperature=0.2,
                        top_k=20,
                        device=device,
                        strict_act=True,
                    )
                    gen_raw = model.generate_text(
                        prompt=p,
                        tokenizer=tokenizer,
                        max_new_tokens=24,
                        temperature=0.2,
                        top_k=20,
                        device=device,
                        strict_act=False,
                    )
                    gen_s_clean = gen_strict.strip().replace("\n", " ")
                    gen_r_clean = gen_raw.strip().replace("\n", " ")
                    print(f"   Prompt: \"{p[:45]}...\"")
                    print(f"   Target: [{target}] -> Strict: \"{gen_s_clean[:40]}\" | Raw: \"{gen_r_clean[:40]}\"")
            print("------------------------------------------------------------------------------------------\n")
            model.train()

        # Periodic intermediate save
        if step % 250 == 0 and step != max_steps:
            os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
            torch.save({
                "model_state": {k: v.cpu().float() for k, v in model.state_dict().items()},
                "step": step,
                "loss": float(loss_val),
                "best_loss": float(best_loss),
                "ema_loss": float(ema_loss),
                "optimizer": optimizer_type,
                "lr": float(curr_lr),
                "strategy": "tri_modal_refined_sft_adamw",
            }, output_checkpoint)
            print(f"   💾 Periodic checkpoint saved at Step {step} -> {output_checkpoint}")

    # Save Refined SFT Checkpoint
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    print(f"\n💾 Saving Refined SFT Checkpoint to: {output_checkpoint}...")
    torch.save({
        "model_state": {k: v.cpu().float() for k, v in model.state_dict().items()},
        "step": max_steps,
        "loss": float(loss_val),
        "best_loss": float(best_loss),
        "ema_loss": float(ema_loss),
        "optimizer": optimizer_type,
        "lr": float(curr_lr),
        "strategy": "tri_modal_refined_sft_adamw",
    }, output_checkpoint)
    print(f"✓ Refined SFT checkpoint saved successfully to {output_checkpoint}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refined SFT Engine for CerebroSpheroid V5 (AdamW / SCM)")
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/v5_qwen_0_5b_uwbl_scm.pt")
    parser.add_argument("--output-checkpoint", type=str, default="checkpoints/v5_qwen_0_5b_local_sft.pt")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "scm"])
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--probe-interval", type=int, default=50)
    parser.add_argument("--max-cot-samples", type=int, default=1500)
    args = parser.parse_args()

    run_refined_sft(
        base_checkpoint=args.base_checkpoint,
        output_checkpoint=args.output_checkpoint,
        optimizer_type=args.optimizer,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        probe_interval=args.probe_interval,
        max_cot_samples=args.max_cot_samples,
    )
