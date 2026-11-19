#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Exploration-Augmented GRPO Reinforcement Learning Engine
================================================================================
Implements Group Relative Policy Optimization (GRPO) for CerebroSpheroid V5,
specifically powered by the CerebroExplorationRewardModel.

Core Mechanisms:
1. Batched Group Rollouts:
   - For each prompt q, samples G parallel candidate trajectories {o_1, ..., o_G}
     using temperature sampling (T=0.7 - 0.9).
2. Multi-Tier Exploration Reward Scoring:
   - Outcome Grounding: Verifies final symbolic/numerical correctness.
   - Process Reasoning: Rewards hypothesis exploration, step verification, and self-correction.
   - RND Curiosity: Rewards representation novelty using frozen orthogonal target network.
   - Concentric Manifold Exploration: Rewards active SVD effective rank, radial flow velocity,
     and shell energy entropy across Cerebro's 16 concentric shells.
   - Intra-Group Diversity: Rewards leave-one-out n-gram and semantic trajectory dispersion.
   - Anti-Degeneration Guardrails: Penalizes repetition loops and runaway verbosity.
3. Group Advantage Standardization:
   - A_i = (R_i - mean(R)) / (std(R) + eps)
4. Clipped Surrogate Policy Gradient (GRPO):
   - Optimizes clipped token ratio r_i,t * A_i with KL divergence penalty against ref policy.
5. Dynamic Curiosity Familiarity Decay:
   - Trains RND predictor network on visited states so explored paths become familiar.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# Memory allocation optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    PRESETS_V5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.cerebro_reward_model import (
    CerebroExplorationRewardModel,
)
from src.corpus import format_stimulus_prompt, encode_no_special


FALLBACK_PROMPTS = [
    {
        "prompt": "What is 15 + 27?",
        "ground_truth": "42",
        "category": "arithmetic",
    },
    {
        "prompt": "If Alice has 3 apples and Bob gives her 5 more, how many apples does Alice have?",
        "ground_truth": "8",
        "category": "word_problem",
    },
    {
        "prompt": "What is 12 multiplied by 8?",
        "ground_truth": "96",
        "category": "arithmetic",
    },
    {
        "prompt": "A train travels at 60 mph for 3 hours. What is the total distance traveled in miles?",
        "ground_truth": "180",
        "category": "physics_math",
    },
    {
        "prompt": "What is the square root of 81?",
        "ground_truth": "9",
        "category": "arithmetic",
    },
    {
        "prompt": "Solve for x: 2*x + 6 = 20.",
        "ground_truth": "7",
        "category": "algebra",
    },
    {
        "prompt": "What is the capital of France?",
        "ground_truth": "Paris",
        "category": "factual",
    },
    {
        "prompt": "Which planet is known as the Red Planet?",
        "ground_truth": "Mars",
        "category": "science",
    },
]


def load_exploration_dataset(
    dataset_path: Optional[str] = None,
    max_samples: int = 5000,
) -> List[Dict[str, Any]]:
    """Loads prompt-ground_truth pairs from jsonl or falls back to built-in curriculum."""
    records = []
    if dataset_path and os.path.exists(dataset_path):
        print(f"📖 Loading exploration prompts from {dataset_path}...")
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        prompt = data.get("prompt", "").strip()
                        raw_resp = data.get("response", data.get("answer", "")).strip()
                        if not prompt:
                            continue

                        # Extract ground truth target
                        gt = ""
                        # Check for numerical / final answer
                        m = re.findall(r"(?:answer is|equals|####|result:?)\s*([^\n\.]+)", raw_resp.lower())
                        if m:
                            gt = m[-1].strip()
                        elif raw_resp:
                            gt = raw_resp.split("\n")[-1].strip()

                        records.append({
                            "prompt": prompt,
                            "ground_truth": gt,
                            "category": data.get("category", "general"),
                        })
                        if len(records) >= max_samples:
                            break
                    except Exception:
                        continue
        except Exception as e:
            print(f"⚠️ Error reading dataset ({e}), using built-in curriculum.")

    if not records:
        print("ℹ️ Using built-in reasoning & math curriculum prompts.")
        records = FALLBACK_PROMPTS.copy()

    print(f"✓ Total exploration prompts loaded: {len(records):,}")
    return records


def setup_model_and_tokenizer(
    model_path: Optional[str],
    preset: str = "v5_small",
    device: str = "cuda",
) -> Tuple[CerebroSpheroidV5, CerebroSpheroidV5, Any, ModelConfigV5]:
    """Loads policy model, reference policy model, tokenizer, and config."""
    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    print(f"🖥️ Execution target device: {dev}")

    cfg = None
    state = None
    if model_path and os.path.exists(model_path):
        print(f"⏳ Loading policy checkpoint from: {model_path}")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config")
        state = ckpt.get("model_state", ckpt.get("model", ckpt))
        if cfg is None or not hasattr(cfg, "surface_dim"):
            cfg = infer_v5_config_from_state_dict(state, preset_name=preset)
    else:
        print(f"⚡ Initializing {preset} from scratch (no base checkpoint specified or found).")
        cfg = ModelConfigV5(**PRESETS_V5[preset])

    # Alignment settings
    cfg.use_sensory_self_attn = True
    if preset == "v5_small":
        cfg.use_rope = False
        cfg.top_k_resonance = 0

    policy_model = CerebroSpheroidV5(cfg)
    if state is not None:
        matched, total = load_v5_checkpoint_into_torch(policy_model, state, verbose=False)
        print(f"✓ Loaded {matched}/{total} parameters into policy model.")

    policy_model.to(dev)

    # Reference policy model (frozen)
    print("❄️ Creating frozen reference policy for KL stabilization...")
    ref_model = CerebroSpheroidV5(cfg)
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.to(dev)

    return policy_model, ref_model, tokenizer, cfg


def compute_grpo_loss(
    policy_model: CerebroSpheroidV5,
    ref_model: CerebroSpheroidV5,
    prompt_ids: torch.Tensor,       # [G, L_p]
    completion_ids: torch.Tensor,   # [G, L_c]
    advantages: torch.Tensor,       # [G]
    clip_ratio: float = 0.20,
    kl_coef: float = 0.04,
    pad_id: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes Group Relative Policy Optimization (GRPO) clipped surrogate loss
    with token-level KL divergence penalty against reference policy.
    """
    G, L_c = completion_ids.shape
    dev = prompt_ids.device
    mask = (completion_ids != pad_id).float()  # [G, L_c]

    # 1. Forward through Reference Model (no grad)
    with torch.no_grad():
        _, ref_logits = ref_model(prompts=prompt_ids, targets=completion_ids, use_anchor_loss=False)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_tok_lp = torch.gather(ref_log_probs, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)  # [G, L_c]

    # 2. Forward through Policy Model (with grad)
    _, policy_logits = policy_model(prompts=prompt_ids, targets=completion_ids, use_anchor_loss=False)
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_tok_lp = torch.gather(policy_log_probs, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)  # [G, L_c]

    # 3. Probability Ratio
    # r_t = exp(log pi_theta - log pi_ref)
    log_ratio = policy_tok_lp - ref_tok_lp
    ratio = torch.exp(torch.clamp(log_ratio, min=-10.0, max=10.0))

    # 4. GRPO Clipped Surrogate Objective
    adv = advantages.unsqueeze(1).to(dev)  # [G, 1]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss_token = -torch.min(surr1, surr2)  # [G, L_c]

    # 5. Token-level KL Divergence Penalty
    # D_KL = exp(log pi_ref - log pi_theta) - 1 - (log pi_ref - log pi_theta)
    kl_div = torch.exp(-log_ratio) - 1.0 + log_ratio  # Non-negative approx

    # Weighted sum over completion tokens
    token_denom = mask.sum(dim=-1).clamp(min=1.0)  # [G]
    seq_policy_loss = (policy_loss_token * mask).sum(dim=-1) / token_denom  # [G]
    seq_kl = (kl_div * mask).sum(dim=-1) / token_denom  # [G]

    total_loss = (seq_policy_loss + kl_coef * seq_kl).mean()

    return total_loss, {
        "policy_loss": float(seq_policy_loss.mean().item()),
        "kl_div": float(seq_kl.mean().item()),
        "mean_ratio": float(ratio.mean().item()),
        "clip_fraction": float((torch.abs(ratio - 1.0) > clip_ratio).float().mean().item()),
    }


def train_grpo_exploration(
    model_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    output_dir: str = "checkpoints/grpo_exploration",
    preset: str = "v5_small",
    device: str = "cuda",
    num_steps: int = 50,
    group_size: int = 4,
    learning_rate: float = 1e-5,
    max_prompt_len: int = 128,
    max_new_tokens: int = 48,
    temperature: float = 0.85,
    top_p: float = 0.95,
    clip_ratio: float = 0.20,
    kl_coef: float = 0.04,
    rnd_lr: float = 1e-4,
    save_interval: int = 25,
) -> None:
    """Executes the complete exploration-driven GRPO training loop."""
    os.makedirs(output_dir, exist_ok=True)
    dev_str = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    dev = torch.device(dev_str)

    print(f"\n{'='*80}")
    print(f"🚀 CEREBRO-SPHEROID V5 EXPLORATION-AUGMENTED GRPO TRAINER")
    print(f"Device: {dev} | Steps: {num_steps} | Group Size: {group_size} | LR: {learning_rate}")
    print(f"Temperature: {temperature} | Clip Ratio: {clip_ratio} | KL Coef: {kl_coef}")
    print(f"{'='*80}\n")

    policy_model, ref_model, tokenizer, cfg = setup_model_and_tokenizer(
        model_path=model_path, preset=preset, device=dev_str
    )

    reward_model = CerebroExplorationRewardModel(
        surface_dim=cfg.surface_dim,
        rnd_out_dim=256,
        rnd_lr=rnd_lr,
        w_outcome=1.2,
        w_curiosity=0.40,
        w_manifold=0.35,
        w_diversity=0.30,
        w_process=0.40,
        w_penalty=0.50,
    ).to(dev)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    prompts_dataset = load_exploration_dataset(dataset_path)

    eos_id = getattr(tokenizer, "eos_token_id", 2) or 2
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    running_rewards: List[float] = []
    running_outcomes: List[float] = []
    running_curiosity: List[float] = []
    running_diversity: List[float] = []
    running_manifold: List[float] = []

    print("\n🏁 Starting GRPO Exploration Reinforcement Loop...\n")

    for step in range(1, num_steps + 1):
        t0 = time.time()

        # 1. Sample Prompt
        sample = random.choice(prompts_dataset)
        raw_prompt = sample["prompt"]
        gt_answer = sample.get("ground_truth", "")

        formatted_p = format_stimulus_prompt(raw_prompt)
        p_ids = encode_no_special(tokenizer, formatted_p)
        if len(p_ids) > max_prompt_len:
            p_ids = p_ids[:max_prompt_len]

        # 2. Batched Parallel Rollout Generation
        # Prompt tensor: [G, L_p]
        p_tensor = torch.tensor([p_ids] * group_size, dtype=torch.long, device=dev)

        policy_model.eval()
        with torch.no_grad():
            rollout_tokens = policy_model.generate(
                prompt_ids=p_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.15,
                eos_token_id=eos_id,
            )

            # Extract Spheroid concentric state & hidden representation
            # current_spheroid_state: [G, K, D]
            spheroid_state = policy_model.current_spheroid_state
            if spheroid_state is None:
                spheroid_state = torch.zeros(group_size, cfg.num_shells, cfg.surface_dim, device=dev)
            
            # Representation for curiosity RND: mean across shells or cortical surface [G, D]
            state_representations = spheroid_state.mean(dim=1).detach()

        # 3. Decode Candidates and Score via Exploration Reward Model
        candidates = []
        for g_idx in range(group_size):
            cand_tokens = rollout_tokens[g_idx].tolist()
            # Truncate at EOS if present
            if eos_id in cand_tokens:
                eos_idx = cand_tokens.index(eos_id)
                cand_tokens = cand_tokens[: eos_idx + 1]

            cand_text = tokenizer.decode(cand_tokens, skip_special_tokens=True).strip()
            candidates.append({
                "text": cand_text,
                "tokens": cand_tokens,
                "hidden": state_representations[g_idx],
                "state": spheroid_state[g_idx],
            })

        # Score with multi-tier exploration reward model
        scored_candidates = reward_model.score_group_rollouts(
            prompt=raw_prompt,
            candidates=candidates,
            ground_truth=gt_answer,
        )

        # 4. Group Advantage Standardization
        rewards = [c["total_reward"] for c in scored_candidates]
        r_mean = float(np.mean(rewards))
        r_std = float(np.std(rewards))

        if r_std > 1e-6:
            advantages = torch.tensor([(r - r_mean) / r_std for r in rewards], dtype=torch.float32, device=dev)
        else:
            advantages = torch.zeros(group_size, dtype=torch.float32, device=dev)

        # 5. Prepare padded completion tensor for policy forward pass
        max_c_len = max(len(c["tokens"]) for c in candidates)
        if max_c_len < 2:
            # Fallback pad
            max_c_len = 2

        padded_completions = torch.full((group_size, max_c_len), pad_id, dtype=torch.long, device=dev)
        for g_idx, c in enumerate(candidates):
            c_len = len(c["tokens"])
            if c_len > 0:
                padded_completions[g_idx, :c_len] = torch.tensor(c["tokens"][:max_c_len], dtype=torch.long, device=dev)

        # 6. GRPO Policy Gradient Update
        policy_model.train()
        optimizer.zero_grad()

        loss, metrics = compute_grpo_loss(
            policy_model=policy_model,
            ref_model=ref_model,
            prompt_ids=p_tensor,
            completion_ids=padded_completions,
            advantages=advantages,
            clip_ratio=clip_ratio,
            kl_coef=kl_coef,
            pad_id=pad_id,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()

        # 7. Dynamic Curiosity Familiarity Decay
        # Update predictor network on the visited representations
        curiosity_loss = reward_model.update_curiosity_with_visited_states(state_representations)

        # 8. Tracking & Reporting
        dt = time.time() - t0
        mean_curiosity = float(np.mean([c["curiosity_novelty"] for c in scored_candidates]))
        mean_diversity = float(np.mean([c["diversity_reward"] for c in scored_candidates]))
        mean_manifold = float(np.mean([c["manifold_exploration"] for c in scored_candidates]))
        mean_outcome = float(np.mean([c["outcome_reward"] for c in scored_candidates]))

        running_rewards.append(r_mean)
        running_outcomes.append(mean_outcome)
        running_curiosity.append(mean_curiosity)
        running_diversity.append(mean_diversity)
        running_manifold.append(mean_manifold)

        # Logging output
        print(
            f"Step {step:03d}/{num_steps:03d} [{dt:.2f}s] | "
            f"R_tot: {r_mean:+.3f} | "
            f"Outcome: {mean_outcome:.2f} | "
            f"Curiosity: {mean_curiosity:.3f} | "
            f"Diversity: {mean_diversity:.3f} | "
            f"Manifold: {mean_manifold:.3f} | "
            f"Loss: {loss.item():.4f} | "
            f"KL: {metrics['kl_div']:.4f}"
        )

        # Detailed rollout sample display periodically
        if step == 1 or step % 10 == 0:
            print(f"\n--- [Step {step} Prompt: \"{raw_prompt}\" (GT: \"{gt_answer}\")] ---")
            for g_idx, sc in enumerate(scored_candidates):
                txt_preview = sc['details'].get('response_text', candidates[g_idx]['text'])
                txt_snippet = txt_preview.replace('\n', ' ')[:90]
                print(
                    f"  Cand {g_idx}: R={sc['total_reward']:+.3f} "
                    f"[Out:{sc['outcome_reward']:.1f}, Proc:{sc['process_reward']:.2f}, Cur:{sc['curiosity_novelty']:.2f}, Div:{sc['diversity_reward']:.2f}] "
                    f"-> \"{txt_snippet}...\""
                )
            print()

        # Checkpointing
        if step % save_interval == 0 or step == num_steps:
            save_path = os.path.join(output_dir, f"cerebro_grpo_step_{step}.pt")
            reward_save_path = os.path.join(output_dir, f"reward_model_step_{step}.pt")

            torch.save({
                "step": step,
                "model_state": policy_model.state_dict(),
                "config": cfg,
                "metrics": {
                    "running_rewards": running_rewards,
                    "running_curiosity": running_curiosity,
                    "running_diversity": running_diversity,
                    "running_manifold": running_manifold,
                },
            }, save_path)

            reward_model.save_checkpoint(reward_save_path)
            print(f"💾 Checkpoints saved to {save_path} & {reward_save_path}\n")

        # GPU cache cleanup
        if dev_str.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n✅ GRPO Exploration Reinforcement Training Complete!")
    print(f"Final Mean Reward (last 10 steps): {np.mean(running_rewards[-10:]):+.4f}")
    print(f"Final Mean Curiosity Novelty: {np.mean(running_curiosity[-10:]):.4f}")
    print(f"Final Mean Intra-Group Diversity: {np.mean(running_diversity[-10:]):.4f}")
    print(f"Final Mean Manifold Rank/Flow Bonus: {np.mean(running_manifold[-10:]):.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CerebroSpheroid V5 with Exploration-Augmented GRPO")
    parser.add_argument("--model_path", type=str, default="checkpoints/v5_small_reasoning_sft_p05_best.pt", help="Base model checkpoint")
    parser.add_argument("--dataset_path", type=str, default="data/gemini_golden_reasoning_sft.jsonl", help="Exploration prompt dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints/grpo_exploration", help="Checkpoint output directory")
    parser.add_argument("--preset", type=str, default="v5_small", help="Architecture preset")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of GRPO exploration steps")
    parser.add_argument("--group_size", type=int, default=4, help="Candidate rollouts per prompt (G)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Policy learning rate")
    parser.add_argument("--max_prompt_len", type=int, default=128, help="Max prompt length")
    parser.add_argument("--max_new_tokens", type=int, default=48, help="Max rollout tokens per candidate")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus top-p")
    parser.add_argument("--clip_ratio", type=float, default=0.20, help="GRPO / PPO clip ratio")
    parser.add_argument("--kl_coef", type=float, default=0.04, help="KL divergence penalty coefficient")
    parser.add_argument("--rnd_lr", type=float, default=1e-4, help="RND curiosity predictor learning rate")
    parser.add_argument("--save_interval", type=int, default=25, help="Checkpoint save interval")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_grpo_exploration(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        preset=args.preset,
        device=args.device,
        num_steps=args.num_steps,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        clip_ratio=args.clip_ratio,
        kl_coef=args.kl_coef,
        rnd_lr=args.rnd_lr,
        save_interval=args.save_interval,
    )
