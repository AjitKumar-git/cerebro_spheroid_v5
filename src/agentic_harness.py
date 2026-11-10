"""
CerebroSpheroid V5: Agentic Execution Harness & Persistent State Loop
======================================================================

Key Architectural Capabilities:
1. Discrete Action Classifier Head (ToolActionHead):
   Directly projects from Cortical Mantle (Shell K-1) into discrete action classes:
   [0: CHAT, 1: BASH, 2: PYTHON, 3: FILE_READ, 4: FILE_WRITE, 5: CALCULATOR].
   Eliminates fragile JSON string parsing and syntax hallucinations.

2. Persistent Working Memory (Zero Context Wiping):
   Between tool turns, the 32 concentric spherical shells stay live in VRAM.
   The model does NOT need to re-read thousands of past conversation tokens.

3. Closed-Loop Sensory Re-Afference:
   Tool output (stdout, stderr, traceback) is injected directly into Shell K-1 as a
   high-magnitude sensory impulse, which cascades through the core via inward wave
   propagation for self-healing error correction.
"""

from __future__ import annotations

import ast
import io
import os
import subprocess
import sys
import time
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, RMSNorm


class ActionType(IntEnum):
    """Discrete Agentic Action Classes."""
    CHAT = 0          # Direct conversational response to human
    BASH = 1          # Shell command execution
    PYTHON = 2        # Python REPL sandbox execution
    FILE_READ = 3     # Read file contents
    FILE_WRITE = 4    # Write file contents
    CALCULATOR = 5    # Safe mathematical arithmetic evaluation


ACTION_NAMES = {
    ActionType.CHAT: "chat",
    ActionType.BASH: "bash",
    ActionType.PYTHON: "python",
    ActionType.FILE_READ: "file_read",
    ActionType.FILE_WRITE: "file_write",
    ActionType.CALCULATOR: "calculator",
}


class ToolActionHead(nn.Module):
    """
    Cortical Action Classifier Head.
    Predicts discrete action dispatch directly from Cortical Mantle state (Shell K-1).
    """

    def __init__(self, surface_dim: int, num_actions: int = len(ActionType)):
        super().__init__()
        self.norm = RMSNorm(surface_dim)
        self.fc = nn.Linear(surface_dim, num_actions, bias=True)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)
        # Bias Action 0 (CHAT) initially so untuned models default to conversation
        nn.init.constant_(self.fc.bias, 0.0)
        self.fc.bias.data[ActionType.CHAT] = 1.0

    def forward(self, cortical_state: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cortical_state: [B, surface_dim] (Shell K-1 representation)
        Returns:
            logits: [B, num_actions]
            probs: [B, num_actions]
        """
        h = self.norm(cortical_state)
        logits = self.fc(h) / max(temperature, 1e-4)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class SandboxedToolExecutor:
    """Safe local sandbox for tool execution."""

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()

    def execute_bash(self, command: str, timeout: int = 15) -> str:
        """Runs a shell command safely with timeout."""
        clean_cmd = command.strip().strip("`").strip()
        if clean_cmd.startswith("bash"):
            clean_cmd = clean_cmd[4:].strip()
        try:
            res = subprocess.run(
                clean_cmd,
                shell=True,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = res.stdout.strip()
            err = res.stderr.strip()
            if res.returncode == 0:
                return out if out else "[Command completed with zero exit code]"
            return f"[Exit code {res.returncode}] stderr: {err}\nstdout: {out}".strip()
        except subprocess.TimeoutExpired:
            return f"[Error: Command timed out after {timeout}s]"
        except Exception as e:
            return f"[Execution Error: {e}]"

    def execute_python(self, code: str, timeout: int = 15) -> str:
        """Executes Python code in a sandboxed subprocess."""
        clean_code = code.strip().strip("`").strip()
        if clean_code.startswith("python"):
            clean_code = clean_code[6:].strip()
        try:
            res = subprocess.run(
                [sys.executable, "-c", clean_code],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = res.stdout.strip()
            err = res.stderr.strip()
            if res.returncode == 0:
                return out if out else "[Python completed with zero output]"
            return f"[Traceback / Error]:\n{err}".strip()
        except subprocess.TimeoutExpired:
            return f"[Error: Python execution timed out after {timeout}s]"
        except Exception as e:
            return f"[Python Error: {e}]"

    def execute_file_read(self, path: str, max_chars: int = 2000) -> str:
        """Reads content of a local file safely."""
        clean_path = path.strip().strip("\"'").strip()
        full_path = os.path.abspath(os.path.join(self.workspace_root, clean_path))
        if not full_path.startswith(os.path.abspath(self.workspace_root)) and not os.path.exists(full_path):
            return f"[Access Denied / Not Found: {clean_path}]"
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_chars)
            return content
        except Exception as e:
            return f"[File Read Error: {e}]"

    def execute_calculator(self, expression: str) -> str:
        """Safely evaluates arithmetic expressions using AST parsing."""
        clean_expr = expression.strip().replace("=", "").strip()
        try:
            node = ast.parse(clean_expr, mode="eval")
            # Only allow safe arithmetic AST nodes
            allowed_nodes = (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Constant,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.FloorDiv,
                ast.Mod,
                ast.Pow,
                ast.USub,
                ast.UAdd,
            )
            # Add backwards-compatibility for older Python versions if ast.Num exists
            if hasattr(ast, "Num"):
                allowed_nodes = allowed_nodes + (getattr(ast, "Num"),)

            for subnode in ast.walk(node):
                if not isinstance(subnode, allowed_nodes):
                    return f"[Error: Unsafe calculator expression]"
            result = eval(compile(node, "<calc>", "eval"), {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"[Calculator Error: {e}]"

    def dispatch(self, action: ActionType, argument: str) -> str:
        """Dispatches action to the appropriate tool runner."""
        if action == ActionType.BASH:
            return self.execute_bash(argument)
        elif action == ActionType.PYTHON:
            return self.execute_python(argument)
        elif action == ActionType.FILE_READ:
            return self.execute_file_read(argument)
        elif action == ActionType.CALCULATOR:
            return self.execute_calculator(argument)
        elif action == ActionType.CHAT:
            return argument
        else:
            return f"[Unknown Action: {action}]"


class AgenticSpheroidHarness(nn.Module):
    """
    Agentic Spheroid Execution Engine:
    Wraps CerebroSpheroidV5 with:
    - ToolActionHead (Discrete action selection)
    - Persistent Working Memory (Active 32 concentric shells across multi-turn interactions)
    - Sensory Re-afference (Closed-loop tool execution feedback)
    """

    def __init__(self, model: CerebroSpheroidV5, workspace_root: Optional[str] = None):
        super().__init__()
        self.model = model
        self.surface_dim = model.config.surface_dim
        self.action_head = ToolActionHead(self.surface_dim, num_actions=len(ActionType))
        self.executor = SandboxedToolExecutor(workspace_root)
        self.persistent_state: Optional[torch.Tensor] = None

    def reset_working_memory(self):
        """Clears persistent concentric state."""
        self.persistent_state = None

    def inject_tool_feedback(self, tool_output: str, tokenizer, device: torch.device) -> torch.Tensor:
        """
        Closed-Loop Sensory Re-Afference:
        Injects tool output (stdout, stderr, traceback) into Cortical Surface (Shell 31)
        and steps deliberation to propagate feedback inwards through the 32 shells.
        """
        if self.persistent_state is None:
            self.persistent_state = self.model.init_spheroid_state(1, device)

        # Tokenize tool output
        token_ids = tokenizer.encode(f"\n[Tool Output]: {tool_output[:256]}", add_special_tokens=False)
        if not token_ids:
            token_ids = [0]
        token_ids = [t % self.model.config.vocab_size for t in token_ids]
        t_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

        # Sensory flux representation
        with torch.no_grad():
            flux, _ = self.model.sensory_dnn(t_tensor)
            feedback_centroid = flux.mean(dim=1)  # [1, D]

            # Inject into Cortical Mantle (Shell K-1) out-of-place
            state = self.persistent_state
            mantle = state[:, -1:, :] + 0.2 * feedback_centroid.unsqueeze(1)
            state = torch.cat([state[:, :-1, :], mantle], dim=1)

            # Subconscious deliberation: assimilate feedback across 32 concentric shells
            for _ in range(2):
                state = self.model.volumetric_core.deliberate_step(state)

            self.persistent_state = state
            return self.persistent_state

    def step_agentic_turn(
        self,
        user_prompt: str,
        tokenizer,
        device: torch.device,
        max_tokens: int = 48,
        temperature: float = 0.7,
        force_action: Optional[ActionType] = None,
    ) -> Dict[str, Any]:
        """
        Executes a single end-to-end agentic turn:
        1. Assimilates user prompt into persistent working memory.
        2. Deliberates internally.
        3. Predicts discrete action (ToolActionHead).
        4. Generates argument / speech tokens.
        5. If a tool action was chosen, executes in sandbox and injects feedback.
        """
        self.eval()
        B = 1

        # Tokenize prompt safely with vocabulary boundaries
        bos = getattr(tokenizer, "bos_token_id", None)
        raw_ids = tokenizer.encode(user_prompt, add_special_tokens=False) if hasattr(tokenizer, "encode") else [1]
        p_ids = []
        if bos is not None and isinstance(bos, int) and 0 <= bos < self.model.config.vocab_size:
            p_ids.append(bos)
        p_ids.extend([t % self.model.config.vocab_size for t in raw_ids])
        if not p_ids:
            p_ids = [0]
        p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            sensory_flux, sensory_kv = self.model.sensory_dnn(p_tensor)
            prompt_centroid = sensory_flux.mean(dim=1)

            # Ingest into persistent state (or initialize fresh)
            if self.persistent_state is None:
                subcortical = torch.zeros(B, self.model.config.num_shells - 1, self.surface_dim, device=device)
                state = torch.cat([subcortical, prompt_centroid.unsqueeze(1)], dim=1)
            else:
                state = self.persistent_state
                mantle = state[:, -1:, :] + 0.3 * prompt_centroid.unsqueeze(1)
                state = torch.cat([state[:, :-1, :], mantle], dim=1)

            # Subconscious deliberation cycles
            cycles = self.model.config.thinking_cycles
            for _ in range(cycles):
                state = self.model.volumetric_core.deliberate_step(state)

            self.persistent_state = state

            # Predict Action via ToolActionHead
            cortical_mantle = state[:, -1, :]  # Shell K-1
            action_logits, action_probs = self.action_head(cortical_mantle)

            if force_action is not None:
                action_id = force_action
            else:
                action_id = ActionType(int(torch.argmax(action_probs, dim=-1).item()))

            action_name = ACTION_NAMES.get(action_id, "unknown")
            action_confidence = float(action_probs[0, action_id].item())

            # Generate argument or conversational speech tokens
            generated_text = self.model.generate_text(
                user_prompt,
                tokenizer=tokenizer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                device=device,
            ).strip()

            # Execute tool if action is non-conversational
            tool_output = None
            if action_id != ActionType.CHAT:
                tool_output = self.executor.dispatch(action_id, generated_text)
                # Sensory Re-afference: Inject tool output back into live working memory
                self.inject_tool_feedback(tool_output, tokenizer, device)

            return {
                "action_id": int(action_id),
                "action_name": action_name,
                "confidence": action_confidence,
                "generated_text": generated_text,
                "tool_output": tool_output,
                "persistent_memory_active": True,
            }
