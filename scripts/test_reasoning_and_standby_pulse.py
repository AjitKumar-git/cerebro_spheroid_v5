#!/usr/bin/env python3
"""
Diagnostic Script: Reasoning Verification & Standby 4 bps Resting Pulse Dynamics.
1. Tests model reasoning capabilities (arithmetic, deductive logic, analogies, slot QA).
2. Simulates standby resting state (zero sensory input with a 4 bps oscillatory pulse).
"""

import os
import sys
import time
import math
import torch
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from transformers import AutoTokenizer
from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    infer_v5_config_from_state_dict,
)
from src.corpus import (
    format_stimulus_prompt,
    extract_strict_act_payload,
    get_bio_token_ids,
)


REASONING_PROMPTS = [
    ("Analogy (Direct Slot)", "hot : cold :: day : ?"),
    ("Analogy (Natural)", "Complete the analogy: doctor is to hospital as chef is to ____."),
    ("Antonym", "What is the opposite of huge?"),
    ("Factual Slot", "What is the capital of France?"),
    ("Elementary Math", "What is 7 * 8?"),
    ("Word Problem (GSM8K)", "If John has 15 apples and gives 4 to Mary and 3 to Bob, how many apples does John have left?"),
    ("Counterfactual Logic", "What is the capital city of Antarctica?"),
    ("Python Code", "def is_even(n):\n    \"\"\"Return True if n is even, else False.\"\"\"\n"),
]


def find_best_checkpoint():
    candidate_dirs = [
        os.path.join(WORKSPACE_ROOT, "checkpoints"),
        "/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints",
    ]
    priorities = [
        "v5_qwen_0_5b_uwbl_scm.pt",
        "v5_qwen_0_5b_refined_sft.pt",
        "v5_qwen_0_5b_converged.pt",
        "v5_small_sft_best.pt",
        "v5_small_smollm_converged.pt",
        "v5_small_latest_step900k.pt",
    ]
    for fname in priorities:
        for d in candidate_dirs:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                return path
    return None


def run_reasoning_evaluation(model, tokenizer, device):
    print("\n" + "=" * 80)
    print("🧠 PART 1: REASONING CAPABILITY EVALUATION BATTERY")
    print("=" * 80)

    results = []
    for category, prompt in REASONING_PROMPTS:
        print(f"\n--- [{category}] ---")
        print(f"Prompt: {prompt.strip()}")

        try:
            # Greedy generation (T=0.1) for deterministic reasoning
            out_greedy = model.generate_text(
                prompt,
                tokenizer,
                max_new_tokens=48,
                temperature=0.1,
                top_k=20,
                device=device,
                strict_act=True,
            )
            print(f"  👉 Response (Greedy T=0.1): {repr(out_greedy)}")
            results.append((category, prompt, out_greedy))
        except Exception as e:
            print(f"  ❌ Generation error: {e}")
            results.append((category, prompt, f"ERROR: {e}"))

    return results


def run_standby_4bps_pulse_simulation(model, tokenizer, device, num_pulses=40, pulse_freq_hz=4.0):
    """
    Simulates standby resting state (zero external sensory prompt input) with an endogenous 4 bps clock pulse.
    Tracks:
    - Shell RMS energy profile across concentric layers (subcortical core -> cortical surface)
    - Inter-shell cosine coherence
    - Attractor stability vs dissipative collapse or explosive divergence
    - Spontaneous motor output discharge under resting potential
    """
    print("\n" + "=" * 80)
    print(f"💓 PART 2: STANDBY RESTING STATE SIMULATION ({pulse_freq_hz} BPS / {pulse_freq_hz} Hz PULSE)")
    print("=" * 80)
    print(f"Input: Pure Standby (Zero Sensory Prompt)")
    print(f"Duration: {num_pulses} pulses ({num_pulses / pulse_freq_hz:.1f} seconds of quiet rest)")
    print("-" * 80)

    model.eval()
    B = 1
    K = model.config.num_shells
    D = model.config.surface_dim
    R = model.config.radial_dim

    # Zero sensory input: model sits in pure resting state
    # Shell state: [B, K, D] initialized to zero
    state = torch.zeros(B, K, D, device=device, dtype=torch.float32)

    # Motor plan initialized to baseline resting potential
    plan = torch.zeros(B, D, device=device, dtype=torch.float32)

    shell_rms_history = []
    coherence_history = []
    spontaneous_tokens = []
    bio_ids = get_bio_token_ids(tokenizer)

    for pulse in range(1, num_pulses + 1):
        with torch.no_grad():
            # 1. Subconscious cortical deliberation step (radial flow + shortcut tracts)
            state = model.volumetric_core.deliberate_step(state)

            # 2. Motor readout under resting state (no external sensory prompt)
            tied_w = model.sensory_dnn.embedding.weight
            logits, efference, _, plan = model.motor_decoder(
                state=state,
                sensory_kv=None,
                tied_embedding_weight=tied_w,
                plan=plan,
                step_idx=pulse,
            )

            # 3. Lightweight bounded efference copy propagation back into subcortical core
            state = model.volumetric_core.step_efference_fast(state, efference)

            # 4. Measure concentric shell energy (RMS)
            rms_per_shell = [float(torch.sqrt(torch.mean(state[0, k, :].pow(2))).item()) for k in range(K)]
            mean_rms = sum(rms_per_shell) / K

            # 5. Measure inter-shell adjacent cosine coherence
            cos_list = []
            for k in range(K - 1):
                s1 = state[0, k, :]
                s2 = state[0, k + 1, :]
                c = torch.dot(s1, s2) / (torch.norm(s1) * torch.norm(s2) + 1e-6)
                cos_list.append(float(c.item()))
            mean_cos = sum(cos_list) / len(cos_list) if cos_list else 1.0

            shell_rms_history.append((pulse, mean_rms, rms_per_shell))
            coherence_history.append((pulse, mean_cos))

            # 6. Check spontaneous motor cortex discharge
            top_tok = int(torch.argmax(logits[0], dim=-1).item())
            tok_text = tokenizer.decode([top_tok])
            spontaneous_tokens.append((pulse, top_tok, tok_text))

        if pulse % 8 == 0 or pulse == 1 or pulse == num_pulses:
            core_rms = rms_per_shell[0]
            surface_rms = rms_per_shell[-1]
            print(f"   [Pulse {pulse:2d}/{num_pulses}] ({pulse / pulse_freq_hz:4.1f}s) | "
                  f"Mean RMS: {mean_rms:.4f} | Core RMS: {core_rms:.4f} | Surface RMS: {surface_rms:.4f} | "
                  f"Coherence: {mean_cos:.3f} | Spontaneous: {repr(tok_text.strip())}")

    # Summary Analysis
    initial_rms = shell_rms_history[0][1]
    final_rms = shell_rms_history[-1][1]
    rms_ratio = final_rms / max(1e-6, initial_rms)
    final_coherence = coherence_history[-1][1]

    print("\n" + "=" * 80)
    print("📊 STANDBY RESTING DYNAMICS DIAGNOSTIC REPORT")
    print("=" * 80)
    print(f"• Resting State Stability : {'STABLE ATTRACTOR (Non-Explosive)' if final_rms < 10.0 else 'UNSTABLE DIVERGENCE'}")
    print(f"• Energy Retention Ratio  : {rms_ratio:.2f}x (Final RMS: {final_rms:.4f} vs Initial: {initial_rms:.4f})")
    print(f"• Inter-Shell Coherence   : {final_coherence:.3f} (Cos similarity across adjacent concentric shells)")
    print(f"• Core / Surface Energy   : {shell_rms_history[-1][2][0]:.4f} (Core) / {shell_rms_history[-1][2][-1]:.4f} (Surface)")
    
    unique_spontaneous = len(set(t[1] for t in spontaneous_tokens))
    print(f"• Spontaneous Token Modes : {unique_spontaneous} unique resting tokens across {num_pulses} pulses")
    sample_stream = "".join(t[2] for t in spontaneous_tokens[:16]).replace("\n", "\\n")
    print(f"• Sample Resting Stream   : \"{sample_stream}\"")
    print("=" * 80)

    return {
        "final_rms": final_rms,
        "final_coherence": final_coherence,
        "is_stable": final_rms < 10.0,
        "rms_history": shell_rms_history,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else find_best_checkpoint()
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading Checkpoint: {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Infer config
    cfg = infer_v5_config_from_state_dict(ckpt)
    print(f"Config: Shells={cfg.num_shells}, SurfaceDim={cfg.surface_dim}, RadialDim={cfg.radial_dim}")

    model = CerebroSpheroidV5(cfg)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    # Part 1: Reasoning Evaluation
    reasoning_results = run_reasoning_evaluation(model, tokenizer, device)

    # Part 2: Standby 4 bps Resting Pulse Simulation
    standby_results = run_standby_4bps_pulse_simulation(model, tokenizer, device, num_pulses=40, pulse_freq_hz=4.0)


if __name__ == "__main__":
    main()
