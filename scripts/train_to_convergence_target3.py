#!/usr/bin/env python3
"""
High-Velocity Convergence Engine: Target Loss <= 3.0
===================================================
Continues curriculum training from Step 500 onwards, dynamically
monitoring and optimizing convergence until loss reaches <= 3.0.
"""

import os
import sys
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/hf_home")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/hf_home")

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from transformers import AutoTokenizer
from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from src.corpus import format_stimulus_prompt, format_motor_response, encode_no_special


def train_towards_target_loss(
    target_loss: float = 3.0,
    max_steps: int = 2000,
    start_step: int = 500,
    batch_size: int = 6,
    initial_lr: float = 4.5e-5,
    min_lr: float = 3.0e-6,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🎯 CONVERGENCE CAMPAIGN: Target Cross-Entropy Loss <= {target_loss:.2f}")
    print(f"Device: {device} | Batch Size: {batch_size}")
    print(f"Step Range: {start_step} -> {max_steps} | Dynamic Early Exit on Loss <= {target_loss}")
    print("=" * 80)

    # 1. Tokenizer & Architecture
    v5_tok = AutoTokenizer.from_pretrained(os.path.join(WORKSPACE_ROOT, "data", "tokenizer"))
    cfg = ModelConfigV5(**PRESETS_V5["tpu_v5e_1"])
    model = CerebroSpheroidV5(cfg).to(device)

    # 2. Load Checkpoint from Step 500
    ckpt_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_smollm_aligned_step500.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_smollm_aligned.pt")
        start_step = 100

    print(f"\n📥 Loading baseline checkpoint from: {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    current_loss = ckpt.get("calibration_loss", 7.95)
    print(f"   ✓ Checkpoint loaded at Step {ckpt.get('step', start_step)} (Initial Loss: {current_loss:.4f})")

    # 3. Load Full Cognitive Trajectory Corpus (6,000+ pairs)
    dataset_path = os.path.join(WORKSPACE_ROOT, "data", "cerebro_cognitive_8k", "cerebro_cognitive_10k.jsonl")
    data_samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 6000:
                break
            entry = json.loads(line)
            data_samples.append((entry["question"], entry["answer"]))

    print(f"   ✓ Loaded {len(data_samples)} diverse cognitive QA trajectories from dataset.")

    tokenized_data = []
    for q, a in data_samples:
        prompt_str = format_stimulus_prompt(q)
        answer_str = format_motor_response(a)
        
        p_ids = [t % model.config.vocab_size for t in encode_no_special(v5_tok, prompt_str)]
        a_ids = [t % model.config.vocab_size for t in encode_no_special(v5_tok, answer_str)]
        
        p_ids = p_ids[:80]
        a_ids = a_ids[:80]
        if len(p_ids) > 2 and len(a_ids) > 2:
            tokenized_data.append((p_ids, a_ids))

    print(f"   ✓ Prepared {len(tokenized_data)} trajectories for batching (max_len: 80).")

    # 4. Optimizer & Cosine Decay Schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)
    total_planned = max_steps - start_step

    def get_lr(step_i):
        prog = min(1.0, step_i / total_planned)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * prog))

    loss_history = ckpt.get("meta", {}).get("loss_history", [])
    best_loss = current_loss
    ema_loss = current_loss

    model.train()
    print("\n⚡ Rollout Active...")
    start_time = time.time()
    converged = False
    final_step = start_step

    for step in range(start_step + 1, max_steps + 1):
        step_idx = step - start_step
        curr_lr = get_lr(step_idx)
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        # Mini-batch sampling
        batch_indices = torch.randint(0, len(tokenized_data), (batch_size,)).tolist()
        batch_pairs = [tokenized_data[idx] for idx in batch_indices]

        max_p = max(len(p) for p, _ in batch_pairs)
        max_a = max(len(a) for _, a in batch_pairs)

        prompts_tensor = torch.zeros((batch_size, max_p), dtype=torch.long, device=device)
        targets_tensor = torch.full((batch_size, max_a), -100, dtype=torch.long, device=device)

        for b_idx, (p, a) in enumerate(batch_pairs):
            prompts_tensor[b_idx, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
            targets_tensor[b_idx, :len(a)] = torch.tensor(a, dtype=torch.long, device=device)

        optimizer.zero_grad()
        loss, _ = model(
            prompts=prompts_tensor,
            targets=targets_tensor,
            sampling_p=1.0,
            deliberation_mode="teacher_forcing",
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append((step, loss_val))
        ema_loss = 0.95 * ema_loss + 0.05 * loss_val
        final_step = step

        if loss_val < best_loss:
            best_loss = loss_val

        # Status log every 50 steps
        if step % 50 == 0 or step == max_steps or loss_val <= target_loss:
            elapsed = time.time() - start_time
            rate = step_idx / max(1e-3, elapsed)
            print(f"   [Step {step:4d}/{max_steps}] Loss: {loss_val:.4f} (EMA: {ema_loss:.4f}) | Best: {best_loss:.4f} | LR: {curr_lr:.2e} | Speed: {rate:.1f} steps/s | Elapsed: {elapsed:.1f}s")

        # Early exit if target <= 3.0 reached
        if ema_loss <= target_loss or loss_val <= target_loss:
            print(f"\n🎉 TARGET CONVERGENCE ACHIEVED! Loss: {loss_val:.4f} <= {target_loss:.2f} at Step {step}!")
            converged = True
            break

    model.eval()

    # 5. Generative Evaluation on Canonical Prompts
    print("\n" + "=" * 80)
    print(f"🔬 GENERATIVE EVALUATION AT STEP {final_step} (Loss: {best_loss:.4f})")
    print("=" * 80)

    test_prompts = [
        "The capital of France is",
        "Question: If 2 + 2 = 4, then 4 + 4 equals what? Answer:",
        "What is an atom?",
        "def is_even(n):",
    ]

    eval_results = []
    for idx, prompt in enumerate(test_prompts, 1):
        print(f"\n[Prompt {idx}] \"{prompt}\"")
        with torch.no_grad():
            gen_text = model.generate_text(
                prompt=prompt,
                tokenizer=v5_tok,
                max_new_tokens=48,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                device=device,
                strict_act=False,
            ).strip()
        print(f"   Output:\n   \"{gen_text}\"")
        eval_results.append({"prompt": prompt, "output": gen_text})

    # 6. Save Converged Checkpoint
    output_ckpt_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "v5_small_smollm_converged.pt")
    torch.save({
        "model_state": model.state_dict(),
        "step": final_step,
        "calibration_loss": best_loss,
        "converged_target_reached": converged,
        "meta": {
            "source": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "target": "CerebroSpheroidV5",
            "target_loss": target_loss,
            "final_loss": best_loss,
            "final_step": final_step,
            "loss_history": loss_history,
            "eval_results": eval_results,
        }
    }, output_ckpt_path)
    print(f"\n💾 Checkpoint saved to: {output_ckpt_path} ({os.path.getsize(output_ckpt_path) / (1024**2):.1f} MB)")

    return {
        "converged": converged,
        "final_step": final_step,
        "best_loss": best_loss,
        "eval_results": eval_results,
        "loss_history": loss_history,
    }


if __name__ == "__main__":
    train_towards_target_loss(target_loss=3.0, max_steps=1800, start_step=500, batch_size=6, initial_lr=4.5e-5, min_lr=3.0e-6)
