#!/usr/bin/env python3
"""
💬 CerebroSpheroid V5: Interactive Local Conversational Terminal
================================================================
Interactive CLI chat interface for CerebroSpheroid V5 Small.
Features:
- Pure native PyTorch inference with concentric spheroidal manifold deliberation
- Full support for biological stimulus-action delimiters: <stim> ... </stim> <act> ... </act>
- Configurable subconscious deliberation cycles (1 -> 8 thinking passes per token)
- Generation speed (tokens/sec) and latency diagnostics
- Multi-turn conversational memory tracking

Usage:
  python3 scripts/chat_cerebro_v5.py
  python3 scripts/chat_cerebro_v5.py --checkpoint checkpoints/v5_small_sft_best.pt
  python3 scripts/chat_cerebro_v5.py --cycles 4 --temp 0.6
"""

import os
import sys
import time
import argparse
import readline  # For smooth arrow-key navigation in terminal
from typing import List, Dict, Optional

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.corpus import (
    build_tokenizer,
    format_stimulus_prompt,
    format_motor_response,
    clean_motor_response,
    encode_no_special,
)


def load_chat_engine(checkpoint_path: str, device: torch.device):
    """Loads CerebroSpheroid checkpoint into evaluation mode."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"\n🧠 Loading CerebroSpheroid V5 from: {os.path.basename(checkpoint_path)}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = infer_v5_config_from_state_dict(ckpt, default_preset="v5_small")
    model = CerebroSpheroidV5(config=cfg)
    loaded, total = load_v5_checkpoint_into_torch(model, ckpt, verbose=False)
    model.to(device).eval()

    step = ckpt.get("step", ckpt.get("global_step", "unknown"))
    loss = float(ckpt.get("loss", ckpt.get("eval_loss", 0.0)))
    eval_loss = float(ckpt.get("eval_loss", 0.0))

    return model, cfg, step, loss, eval_loss


class CerebroChatSession:
    """Manages multi-turn conversation and token generation."""

    def __init__(
        self,
        model: CerebroSpheroidV5,
        tokenizer: any,
        device: torch.device,
        thinking_cycles: int = 4,
        temperature: float = 0.6,
        top_p: float = 0.9,
        repetition_penalty: float = 1.25,
        max_new_tokens: int = 64,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.thinking_cycles = thinking_cycles
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.history: List[Dict[str, str]] = []

    def reset(self):
        self.history.clear()

    def generate_reply(self, user_text: str) -> str:
        """Processes user input, runs subconscious deliberation, and emits response."""
        # Build prompt context with conversational framing
        conversation_context = ""
        for turn in self.history[-3:]:  # Keep up to 3 previous turns in memory
            conversation_context += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"
        conversation_context += f"User: {user_text}\nAssistant:"

        formatted_stim = format_stimulus_prompt(conversation_context) + " <act> "
        tokens = encode_no_special(self.tokenizer, formatted_stim)
        p_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

        orig_cycles = self.model.config.thinking_cycles
        self.model.config.thinking_cycles = self.thinking_cycles

        t0 = time.time()
        stop_ids = [self.tokenizer.eos_token_id] if getattr(self.tokenizer, "eos_token_id", None) else []
        with torch.no_grad():
            out_ids = self.model.generate(
                p_tensor,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                stop_token_ids=stop_ids,
            )
        elapsed = time.time() - t0
        self.model.config.thinking_cycles = orig_cycles

        gen_tokens = out_ids[0].tolist()
        num_generated = len(gen_tokens)
        speed = num_generated / max(elapsed, 1e-4)

        raw_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)
        clean_text = clean_motor_response(raw_text)

        # Fallback if clean response was emptied by delimiter strip
        if not clean_text:
            clean_text = raw_text.replace("<act>", "").replace("</act>", "").replace("</s>", "").strip()

        # Update conversational memory
        self.history.append({"user": user_text, "bot": clean_text})

        return clean_text, elapsed, speed, num_generated


def interactive_chat_loop(checkpoint_path: str, thinking_cycles: int = 4, temperature: float = 0.6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer("data/tokenizer")

    model, cfg, step, loss, eval_loss = load_chat_engine(checkpoint_path, device)
    session = CerebroChatSession(
        model=model,
        tokenizer=tokenizer,
        device=device,
        thinking_cycles=thinking_cycles,
        temperature=temperature,
    )

    print("\n" + "=" * 80)
    print("🧠 CEREBROSPHEROID V5: INTERACTIVE LOCAL CHAT TERMINAL")
    print("=" * 80)
    print(f"  • Model Checkpoint   : {os.path.basename(checkpoint_path)}")
    print(f"  • Training Milestone : Step {step:,} | Loss: {loss:.4f} | Eval Loss: {eval_loss:.4f}")
    print(f"  • Spheroid Manifold  : {cfg.num_shells} concentric shells | Surface: {cfg.surface_dim}D | Radial: {cfg.radial_dim}D")
    print(f"  • Subconscious Passes: {session.thinking_cycles} cycles/token (Deliberation Engine)")
    print(f"  • Hardware Device    : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("-" * 80)
    print("Commands:")
    print("  /cycles <1-8>  : Adjust subconscious thinking passes")
    print("  /temp <0.1-1.0>: Adjust sampling temperature")
    print("  /tokens <len>  : Adjust maximum output tokens")
    print("  /reset         : Clear conversational memory")
    print("  /exit or exit  : Quit chat session")
    print("=" * 80 + "\n")

    while True:
        try:
            user_input = input("\n👤 You > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                print("👋 Ending session. Exiting CerebroSpheroid V5...")
                break

            if user_input.startswith("/cycles"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    session.thinking_cycles = max(1, min(8, int(parts[1])))
                    print(f"⚙️ Subconscious thinking cycles set to: {session.thinking_cycles}")
                continue

            if user_input.startswith("/temp"):
                parts = user_input.split()
                if len(parts) > 1:
                    try:
                        session.temperature = float(parts[1])
                        print(f"⚙️ Temperature set to: {session.temperature:.2f}")
                    except ValueError:
                        pass
                continue

            if user_input.startswith("/tokens"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    session.max_new_tokens = int(parts[1])
                    print(f"⚙️ Max new tokens set to: {session.max_new_tokens}")
                continue

            if user_input == "/reset":
                session.reset()
                print("🧹 Conversational memory reset.")
                continue

            reply, elapsed_sec, tok_speed, num_toks = session.generate_reply(user_input)

            print(f"\n🧠 CerebroSpheroid V5 > {reply}")
            print(f"   ⏱️  {elapsed_sec*1000:.1f}ms | {tok_speed:.1f} tok/s | {num_toks} tokens | {session.thinking_cycles} thinking cycles")

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session interrupted. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Chat with CerebroSpheroid V5 Small.")
    default_ckpt = os.path.join(REPO_ROOT, "checkpoints", "v5_small_sft_best.pt")
    if not os.path.exists(default_ckpt):
        default_ckpt = os.path.join(REPO_ROOT, "checkpoints", "v5_small_sft_step50k.pt")
    if not os.path.exists(default_ckpt):
        default_ckpt = os.path.join(REPO_ROOT, "checkpoints", "v5_small_latest_step900k.pt")

    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Path to checkpoint")
    parser.add_argument("--cycles", type=int, default=4, help="Subconscious deliberation cycles (1-8)")
    parser.add_argument("--temp", type=float, default=0.6, help="Sampling temperature")
    args = parser.parse_args()

    interactive_chat_loop(args.checkpoint, thinking_cycles=args.cycles, temperature=args.temp)


if __name__ == "__main__":
    main()
