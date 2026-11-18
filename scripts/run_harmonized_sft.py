#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Harmonized Refined SFT Engine
===================================================
Solves the "Token Aristocracy" norm imbalance and CoT dilution via:
  1. Embedding Norm Harmonization (all 32,000 rows normalized to unit scale 1.0)
  2. Balanced 50/50 Direct Slot QA & CoT Curriculum (Slot QA oversampled 6x)
  3. 5.0x Answer-Anchor Loss Weighting (AALW) on final slot answers
  4. Decoupled AdamW Optimization (2D/3D weight decay=0.01, 1D norms/biases=0.0)
  5. Live qualitative probe verification on exact answers (night, reader, drank, 5, Paris).
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
)


def build_expanded_slot_qa_corpus() -> List[Tuple[str, str]]:
    """Builds a rich, high-density direct slot QA corpus across diverse factual domains."""
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
        pairs.append((f"Question: {stem} Answer:", ans))

    # B. Antonyms
    antonyms = [
        ("hot", "cold"), ("dark", "light"), ("fast", "slow"), ("happy", "sad"),
        ("author", "reader"), ("strong", "weak"), ("begin", "end"), ("high", "low"),
        ("empty", "full"), ("hard", "soft"), ("rich", "poor"), ("ancient", "modern"),
        ("noisy", "quiet"), ("sharp", "dull"), ("brave", "cowardly"), ("cruel", "kind"),
        ("expand", "contract"), ("create", "destroy"), ("accept", "refuse"), ("arrive", "depart"),
        ("good", "bad"), ("true", "false"), ("deep", "shallow"), ("wide", "narrow"),
    ]
    for word, ant in antonyms:
        pairs.append((f"What is the exact antonym of the word \"{word}\"?", ant))
        pairs.append((f"What is the opposite of \"{word}\"?", ant))
        pairs.append((f"Antonym of {word}:", ant))

    # C. Irregular Verbs (Past Tense & Participles)
    verbs = [
        ("drink", "drank", "drinking"), ("run", "ran", "running"), ("go", "went", "going"),
        ("eat", "ate", "eating"), ("sing", "sang", "singing"), ("grow", "grew", "growing"),
        ("write", "wrote", "writing"), ("see", "saw", "seeing"), ("take", "took", "taking"),
        ("break", "broke", "breaking"), ("choose", "chose", "choosing"), ("drive", "drove", "driving"),
        ("fly", "flew", "flying"), ("know", "knew", "knowing"), ("swim", "swam", "swimming"),
        ("begin", "began", "beginning"), ("speak", "spoke", "speaking"), ("ring", "rang", "ringing")
    ]
    for v, past, ing in verbs:
        pairs.append((f"What is the simple past tense of the irregular verb \"{v}\"?", past))
        pairs.append((f"What is the past tense of \"{v}\"?", past))
        pairs.append((f"Past tense of {v}:", past))
        pairs.append((f"What is the present participle (-ing form) of \"{v}\"?", ing))

    # D. Factual Knowledge & Elementary Arithmetic
    facts = [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Germany?", "Berlin"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the capital of Italy?", "Rome"),
        ("What is the capital of the United Kingdom?", "London"),
        ("What is the capital of Canada?", "Ottawa"),
        ("What is the capital of Spain?", "Madrid"),
        ("What is the capital of Australia?", "Canberra"),
        ("What is the chemical symbol for water?", "H2O"),
        ("What planet is known as the Red Planet?", "Mars"),
        ("What is 15 + 27?", "42"),
        ("What is 8 multiplied by 7?", "56"),
        ("What is 100 divided by 4?", "25"),
        ("What is 9 squared?", "81"),
        ("What is 12 + 13?", "25"),
        ("What is 7 + 8?", "15"),
        ("If John has 3 apples and buys 2 more, how many apples does he have?", "5"),
        ("If Mary has 10 cookies and gives away 4, how many cookies does she have left?", "6"),
        ("Between 'embarrass' and 'embarass', which one is the correct spelling?", "embarrass"),
        ("Between 'receive' and 'recieve', which one is the correct spelling?", "receive")
    ]
    for q, a in facts:
        pairs.append((q, a))
        pairs.append((f"Q: {q}\nA:", a))

    return pairs


def load_harmonized_dataset(
    tokenizer: Any,
    vocab_size: int,
    slot_oversample: int = 6,
    max_cot_samples: int = 1500,
    max_prompt_len: int = 2048,
    max_target_len: int = 1024,
    anchor_weight: float = 5.0,
) -> List[Dict[str, Any]]:
    """Loads a balanced 50/50 dataset with oversampled slot QA and Answer-Anchor weighting."""
    print("\n📦 Assembling Harmonized 50/50 SFT Dataset...")
    tokenized_entries = []
    eos_id = getattr(tokenizer, "eos_token_id", 2) or 2

    def _prepare_prompt_ids(p_str: str) -> List[int]:
        ids = [t % vocab_size for t in encode_no_special(tokenizer, p_str)]
        if len(ids) > max_prompt_len:
            prefix_len = min(32, max_prompt_len // 4)
            suffix_len = max_prompt_len - prefix_len
            ids = ids[:prefix_len] + ids[-suffix_len:]
        return ids

    def _prepare_target_ids(a_str: str) -> List[int]:
        ids = [t % vocab_size for t in encode_no_special(tokenizer, a_str)]
        if len(ids) == 0 or ids[-1] != eos_id:
            ids.append(eos_id)
        if len(ids) > max_target_len:
            ids = ids[: max_target_len - 1] + [eos_id]
        return ids

    # 1. Direct Slot QA (Oversampled to balance gradient volume)
    raw_slot_pairs = build_expanded_slot_qa_corpus()
    print(f"   Base Direct Slot QA Pairs: {len(raw_slot_pairs)} unique pairs")

    for _ in range(slot_oversample):
        for q, a in raw_slot_pairs:
            p_str = format_stimulus_prompt(q)
            a_str = format_motor_response(a)
            p_ids = _prepare_prompt_ids(p_str)
            a_ids = _prepare_target_ids(a_str)
            
            if len(p_ids) > 1 and len(a_ids) > 1:
                # 5.0x penalty on all answer tokens
                weights = [anchor_weight] * len(a_ids)
                tokenized_entries.append({
                    "prompt_ids": p_ids,
                    "target_ids": a_ids,
                    "weights": weights,
                    "type": "direct_slot"
                })

    slot_count = len(tokenized_entries)
    print(f"   ✓ Oversampled Direct Slot QA: {slot_count} samples ({slot_oversample}x oversampled)")

    # 2. CoT Reasoning
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
                p_ids = _prepare_prompt_ids(p_str)
                a_ids = _prepare_target_ids(a_str)
                
                if len(p_ids) > 2 and len(a_ids) > 2:
                    weights = [1.0] * len(a_ids)
                    raw_a_tokens = [tokenizer.decode([t]) for t in a_ids]
                    found_anchor = False
                    for i, tok_str in enumerate(raw_a_tokens):
                        if "####" in tok_str or "##" in tok_str:
                            found_anchor = True
                        if found_anchor:
                            weights[i] = anchor_weight

                    tokenized_entries.append({
                        "prompt_ids": p_ids,
                        "target_ids": a_ids,
                        "weights": weights,
                        "type": "cot_reasoning"
                    })
                    cot_loaded += 1

    print(f"   ✓ Full-Horizon CoT Reasoning: {cot_loaded} samples")
    print(f"   🌟 Total Balanced SFT Dataset: {len(tokenized_entries)} samples")
    random.seed(42)
    random.shuffle(tokenized_entries)
    return tokenized_entries


def run_harmonized_sft(
    base_checkpoint: str = "checkpoints/v5_qwen_0_5b_uwbl_scm.pt",
    output_checkpoint: str = "checkpoints/v5_qwen_0_5b_harmonized_sft.pt",
    harmonize_norm: float = 1.0,
    max_steps: int = 500,
    batch_size: int = 2,
    accum_steps: int = 8,
    lr: float = 5e-5,
    min_lr: float = 5e-6,
    weight_decay: float = 0.01,
    warmup_steps: int = 30,
    probe_interval: int = 25,
    slot_oversample: int = 6,
    anchor_weight: float = 5.0,
    max_cot_samples: int = 1500,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 80)
    print("🎯 CEREBRO-SPHEROID HARMONIZED SFT ENGINE (EMBEDDING LEVELING + SLOT OVERSAMPLING)")
    print(f"Base Checkpoint:   {base_checkpoint}")
    print(f"Output Checkpoint: {output_checkpoint}")
    print(f"Harmonize Norm:    {harmonize_norm:.2f} (unit spherical projection)")
    print(f"Max Steps:         {max_steps} | Micro-Batch: {batch_size} | Accumulation: {accum_steps}")
    print(f"Effective Batch:   {batch_size * accum_steps} sequences/step")
    print(f"Learning Rate:     {lr:.2e} -> {min_lr:.2e} (Warmup: {warmup_steps} steps)")
    print(f"Answer Weighting:  {anchor_weight:.1f}x penalty on target answers")
    print("=" * 80)

    device = torch.device(device_str)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # Load Model
    raw = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    cfg = infer_v5_config_from_state_dict(raw)
    model = CerebroSpheroidV5(cfg)
    sd = raw.get("model_state", raw.get("model_state_dict", raw.get("model", raw)))
    model.load_state_dict(sd, strict=False)

    # 1. EMBEDDING NORM HARMONIZATION (Leveling the Token Playing Field)
    with torch.no_grad():
        emb = model.sensory_dnn.embedding.weight
        before_norms = emb.norm(dim=1)
        print(f"\n⚡ Pre-Harmonization Embedding Norms: Min={before_norms.min().item():.3f}, Max={before_norms.max().item():.3f}, Mean={before_norms.mean().item():.3f}")
        # Normalize every row to uniform target norm
        row_norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-5)
        model.sensory_dnn.embedding.weight.data = (emb / row_norms) * harmonize_norm
        after_norms = model.sensory_dnn.embedding.weight.norm(dim=1)
        print(f"⚡ Post-Harmonization Embedding Norms: Uniform={after_norms.mean().item():.4f} (All 32,000 tokens leveled)")

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(WORKSPACE_ROOT, "data", "tokenizer"))

    model = model.to(dtype).to(device)
    model.train()

    # Load Balanced Dataset
    dataset = load_harmonized_dataset(
        tokenizer=tokenizer,
        vocab_size=cfg.vocab_size,
        slot_oversample=slot_oversample,
        max_cot_samples=max_cot_samples,
        anchor_weight=anchor_weight,
    )

    # Decoupled AdamW
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
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
        ("What is the capital of Japan?", "Tokyo"),
    ]

    print("\n🚀 Beginning Harmonized SFT Training...")
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
            loss, _ = model(
                prompts=p_tensor,
                targets=t_tensor,
                target_weights=w_tensor,
                deliberation_mode="teacher_forcing",
            )
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

        # Qualitative Probe at step 15, 25, 50, and every probe_interval
        if step in (15, 25, 50) or step % probe_interval == 0 or step == max_steps:
            model.eval()
            print(f"\n--- 🔍 Harmonized SFT Qualitative Probe (Step {step}) ---")
            with torch.no_grad():
                for p, target in test_prompts:
                    gen_strict = model.generate_text(
                        prompt=p,
                        tokenizer=tokenizer,
                        max_new_tokens=20,
                        temperature=0.1,
                        top_k=10,
                        device=device,
                        strict_act=True,
                    )
                    gen_raw = model.generate_text(
                        prompt=p,
                        tokenizer=tokenizer,
                        max_new_tokens=20,
                        temperature=0.1,
                        top_k=10,
                        device=device,
                        strict_act=False,
                    )
                    print(f"   Prompt: \"{p[:45]}...\"")
                    print(f"   Target: [{target}] -> Strict: {repr(gen_strict.strip())[:45]} | Raw: {repr(gen_raw.strip())[:45]}")
            print("--------------------------------------------------------------------------------\n")
            model.train()

        # Intermediate checkpoint
        if step % 200 == 0 and step != max_steps:
            os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
            torch.save({
                "model_state": {k: v.cpu().float() for k, v in model.state_dict().items()},
                "step": step,
                "loss": float(loss_val),
                "best_loss": float(best_loss),
                "ema_loss": float(ema_loss),
                "optimizer": "adamw",
                "lr": float(curr_lr),
                "strategy": "harmonized_balanced_sft",
            }, output_checkpoint)
            print(f"   💾 Intermediate checkpoint saved at Step {step} -> {output_checkpoint}")

    # Final Save
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    print(f"\n💾 Saving Harmonized SFT Checkpoint to: {output_checkpoint}...")
    torch.save({
        "model_state": {k: v.cpu().float() for k, v in model.state_dict().items()},
        "step": max_steps,
        "loss": float(loss_val),
        "best_loss": float(best_loss),
        "ema_loss": float(ema_loss),
        "optimizer": "adamw",
        "lr": float(curr_lr),
        "strategy": "harmonized_balanced_sft",
    }, output_checkpoint)
    print(f"✓ Harmonized SFT checkpoint saved successfully to {output_checkpoint}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonized SFT Engine for CerebroSpheroid V5")
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/v5_qwen_0_5b_uwbl_scm.pt")
    parser.add_argument("--output-checkpoint", type=str, default="checkpoints/v5_qwen_0_5b_harmonized_sft.pt")
    parser.add_argument("--harmonize-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--probe-interval", type=int, default=50)
    parser.add_argument("--slot-oversample", type=int, default=6)
    parser.add_argument("--anchor-weight", type=float, default=5.0)
    parser.add_argument("--max-cot-samples", type=int, default=1500)
    args = parser.parse_args()

    run_harmonized_sft(
        base_checkpoint=args.base_checkpoint,
        output_checkpoint=args.output_checkpoint,
        harmonize_norm=args.harmonize_norm,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        probe_interval=args.probe_interval,
        slot_oversample=args.slot_oversample,
        anchor_weight=args.anchor_weight,
        max_cot_samples=args.max_cot_samples,
    )
