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
        max_new_tokens: int = 256,
        show_thinking: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.thinking_cycles = thinking_cycles
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.show_thinking = show_thinking
        self.tri_head_mode = True
        self.history: List[Dict[str, str]] = []

    def reset(self):
        self.history.clear()

    def generate_tri_reply(self, user_text: str):
        """Generates all three cognitive tiers simultaneously in parallel using the Three Heads."""
        t0 = time.time()
        with torch.no_grad():
            res = self.model.generate_tri_stream(
                prompt=user_text,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=40,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                device=self.device,
            )
        elapsed = time.time() - t0
        all_text = res.get("thinking", "") + res.get("action", "") + res.get("answer", "")
        tok_count = max(1, len(self.tokenizer.encode(all_text, add_special_tokens=False)))
        speed = tok_count / max(elapsed, 1e-4)

        self.history.append({"user": user_text, "bot": res["answer"]})
        parsed = {
            "thinking": res["thinking"],
            "tool_calls": [res["action"]] if res["action"] else [],
            "answer": res["answer"],
            "temporal_decisions": res.get("temporal_decisions", {}),
            "raw": f"<think>\n{res['thinking']}\n</think>\n<act>\n{res['action']}\n</act>\n{res['answer']}",
        }

        # Sensory Afference ReAct Loop: Ground Answer in Tool Observation
        if res.get("action"):
            from src.tools.tool_dispatcher import execute_action
            act = res["action"]
            obs = execute_action(act)
            if obs and not obs.startswith("[Tool Error"):
                parsed["tool_results"] = [obs]
                # If answer is missing or truncated, ground it in observation
                if not res.get("answer") or len(res.get("answer", "").strip()) < 3:
                    afferent_prompt = (
                        f"{user_text}\n"
                        f"<think>\n{res.get('thinking', '')}\n</think>\n"
                        f"<act>\n{act}\n</act>\n"
                        f"Observation: {obs}\n"
                    )
                    with torch.no_grad():
                        grounded = self.model.generate_text(
                            prompt=afferent_prompt,
                            tokenizer=self.tokenizer,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            device=self.device,
                            return_parsed=True,
                        )
                        grounded_ans = grounded.get("answer") or grounded.get("raw") or ""
                        if grounded_ans:
                            parsed["answer"] = grounded_ans
        return parsed, elapsed, speed, tok_count

    def generate_reply(self, user_text: str):
        """Processes user input, runs subconscious deliberation, and emits parsed response."""
        t0 = time.time()
        with torch.no_grad():
            parsed = self.model.generate_text(
                prompt=user_text,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=5,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                thinking_cycles=self.thinking_cycles,
                device=self.device,
                strict_act=False,
                return_parsed=True,
            )

        # Sensory Afference ReAct Loop: Execute Action and Ground Answer
        if parsed.get("tool_calls"):
            from src.tools.tool_dispatcher import execute_action
            tool_results = []
            for act in parsed["tool_calls"]:
                tool_res = execute_action(act)
                if tool_res and not tool_res.startswith("[Tool Error"):
                    tool_results.append(tool_res)

            if tool_results:
                parsed["tool_results"] = tool_results
                # If answer is truncated or missing after </act>, articulate grounded answer
                if not parsed.get("answer") or len(parsed.get("answer", "").strip()) < 3:
                    afferent_prompt = (
                        f"{user_text}\n"
                        f"<think>\n{parsed.get('thinking', '')}\n</think>\n"
                        f"<act>\n{parsed['tool_calls'][0]}\n</act>\n"
                        f"Observation: {tool_results[0]}\n"
                    )
                    with torch.no_grad():
                        grounded = self.model.generate_text(
                            prompt=afferent_prompt,
                            tokenizer=self.tokenizer,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            top_k=5,
                            top_p=self.top_p,
                            repetition_penalty=self.repetition_penalty,
                            thinking_cycles=1,
                            device=self.device,
                            strict_act=False,
                            return_parsed=True,
                        )
                        grounded_ans = grounded.get("answer") or grounded.get("raw") or ""
                        if grounded_ans:
                            parsed["answer"] = grounded_ans

        elapsed = time.time() - t0
        raw_text = parsed.get("raw", "")
        tok_count = max(1, len(self.tokenizer.encode(raw_text, add_special_tokens=False)))
        speed = tok_count / max(elapsed, 1e-4)

        # Update conversational memory with clean answer
        answer = parsed.get("answer") or raw_text
        self.history.append({"user": user_text, "bot": answer})

        return parsed, elapsed, speed, tok_count


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
    print("  /think [on|off]: Toggle deliberation scratchpad (<think>) display")
    print("  /tri [on|off]  : Toggle simultaneous 3-Head parallel generation")
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

            if user_input.startswith("/think"):
                parts = user_input.split()
                if len(parts) > 1:
                    arg = parts[1].lower()
                    if arg.isdigit():
                        session.thinking_cycles = max(1, min(16, int(arg)))
                        session.show_thinking = True
                        print(f"⚙️ Deliberation scratchpad visibility: ENABLED | Thinking passes set to: {session.thinking_cycles} cycles/token")
                    elif arg in ("on", "true", "1", "show"):
                        session.show_thinking = True
                        print(f"⚙️ Deliberation scratchpad visibility: ENABLED (Visible)")
                    elif arg in ("off", "false", "0", "hide"):
                        session.show_thinking = False
                        print(f"⚙️ Deliberation scratchpad visibility: HIDDEN (Private scratchpad)")
                else:
                    session.show_thinking = not session.show_thinking
                    status_str = "ENABLED (Visible)" if session.show_thinking else "HIDDEN (Private scratchpad)"
                    print(f"⚙️ Deliberation scratchpad visibility: {status_str}")
                continue

            if user_input.startswith("/tri"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1].lower() in ("off", "false", "0"):
                    session.tri_head_mode = False
                    print("⚙️ Simultaneous 3-Head Parallel Generation: DISABLED (Sequential Single-Stream)")
                else:
                    session.tri_head_mode = True
                    print("⚙️ Simultaneous 3-Head Parallel Generation: ENABLED (Simultaneous Thinking, Action & Talking)")
                continue

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

            if getattr(session, "tri_head_mode", False):
                parsed, elapsed_sec, tok_speed, num_toks = session.generate_tri_reply(user_input)
            else:
                parsed, elapsed_sec, tok_speed, num_toks = session.generate_reply(user_input)

            # 1. Deliberation Scratchpad (<think>)
            if session.show_thinking and parsed.get("thinking"):
                print(f"\n┌─ 💭 Deliberation Scratchpad (<think>) {'─' * 39}")
                for line in parsed["thinking"].splitlines():
                    print(f"│  {line}")
                print(f"└{'─' * 79}")

            # 2. Non-Conversational Action Execution (<act>: Coding, Calculator, SQL, Tools, Diagrams)
            act_category = "Action"
            if parsed.get("tool_calls"):
                for act in parsed["tool_calls"]:
                    act_lower = act.lower().strip()
                    if any(act_lower.startswith(p) for p in ("calculator:", "calc:", "math:")):
                        title = "Calculator Action (<act>)"
                        act_category = "Calculator"
                    elif act_lower.startswith("sql:") or any(act_lower.startswith(p) for p in ("select ", "insert ", "update ", "delete ")):
                        title = "SQL Query (<act>)"
                        act_category = "SQL"
                    elif any(act_lower.startswith(p) for p in ("tool:", "call:", "function:")) or (act.strip().startswith("{") and "tool" in act.lower()):
                        title = "Tool Invocation (<act>)"
                        act_category = "Tool Call"
                    elif any(act_lower.startswith(p) for p in ("mermaid:", "graph ", "flowchart ", "sequencediagram", "plot:")):
                        title = "Diagram & Visual Plot (<act>)"
                        act_category = "Diagram/Plot"
                    elif act_lower.startswith("regex:") or "re.search" in act_lower or "re.findall" in act_lower:
                        title = "Regular Expression (<act>)"
                        act_category = "Regex"
                    elif any(act_lower.startswith(p) for p in ("test:", "assert ")):
                        title = "Unit Test Assertion (<act>)"
                        act_category = "Unit Test"
                    elif any(act_lower.startswith(p) for p in ("datetime:", "time:", "clock:")):
                        title = "Datetime Service (<act>)"
                        act_category = "Datetime"
                    elif any(act_lower.startswith(p) for p in ("bash:", "sh:", "cmd:")):
                        title = "System Shell (<act>)"
                        act_category = "Shell"
                    elif any(act.strip().startswith(p) for p in ("def ", "class ", "import ", "from ", "```", "return ", "print(", "for ", "while ", "# ")) or ("\ndef " in act or "\nclass " in act or "=" in act):
                        title = "Code Synthesis (<act>)"
                        act_category = "Code"
                    else:
                        title = "Action Execution (<act>)"
                        act_category = "Action"
                    dash_count = max(5, 74 - len(title))
                    print(f"\n┌─ ⚡ {title} {'─' * dash_count}")
                    for line in act.splitlines():
                        print(f"│  {line}")
                    from src.tools.tool_dispatcher import execute_action
                    cached_results = parsed.get("tool_results") or []
                    tool_res = cached_results[0] if cached_results else execute_action(act)
                    if tool_res and tool_res != act and not tool_res.startswith("[Tool Error"):
                        print(f"│  ↳ 🔬 Result: {tool_res}")
                    print(f"└{'─' * 79}")

            # 3. Main plain-text response (unbounded to EOS)
            answer = parsed.get("answer") or parsed.get("raw") or "..."
            print(f"\n🧠 CerebroSpheroid V5 > {answer}")
            print(f"   ⏱️  {elapsed_sec*1000:.1f}ms | {tok_speed:.1f} tok/s | {num_toks} tokens | {session.thinking_cycles} thinking cycles")

            # 4. Temporal Gating Route Display
            temporal_info = parsed.get("temporal_decisions", {})
            if temporal_info:
                th_ok = temporal_info.get("think", False)
                ac_ok = temporal_info.get("act", False)
                an_ok = temporal_info.get("answer", True)

                th_str = "✓" if th_ok else "✗ (Suppressed)"
                ac_str = f"✓ ({act_category})" if (ac_ok and parsed.get("tool_calls")) else ("✓" if ac_ok else "✗ (Suppressed)")
                an_str = "✓" if an_ok else "✗ (Suppressed)"

                print(f"   🛣️  Temporal Route   : [Think: {th_str} | Act: {ac_str} | Answer: {an_str}]")

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session interrupted. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Chat with CerebroSpheroid V5 Small.")
    default_ckpt = os.path.join(REPO_ROOT, "checkpoints", "v5_small_sft_master_diverse.pt")
    if not os.path.exists(default_ckpt):
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
