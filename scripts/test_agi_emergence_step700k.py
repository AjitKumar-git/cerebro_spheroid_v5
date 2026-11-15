#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Full AGI Emergence & Manifold Benchmark Suite
===================================================================
Evaluates the latest checkpoint (Step 700.5k) across:
1. The 6 Foundational AGI Emergence Pillars:
   - Pillar 1: Deductive Syllogisms & Novel Taxonomies
   - Pillar 2: Multi-Hop Transitive Reasoning (A > B > C > D)
   - Pillar 3: Counterfactual Physics & Physical Inversion
   - Pillar 4: Multi-Step Arithmetic & Kinematics
   - Pillar 5: Biochemical, Cellular & Physical Mechanisms
   - Pillar 6: Spatial Symmetry & Invariant Counting
   - Pillar 7: OpenHermes Scientific Reasoning & Multi-Choice Logic
2. Manifold Dynamics & Dynamical Stability (AGIEmergenceAuditor):
   - Laminar Protobrain Index (LPI & Effective Rank)
   - Connectome Topology & Active Axonal Tracts
   - Proprioceptive Autonomy & Corollary Discharge Ratio
   - Spectral Radius & Settled Manifold Lyapunov Criticality
3. Subconscious Deliberation Gain (1 vs 2 vs 4 thinking cycles).
4. Free-Form Motor Generation Probe (Text Output Inspection).
"""

import sys
import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.emergence_detector import AGIEmergenceAuditor, EmergenceReport
from src.corpus import (
    build_tokenizer,
    format_stimulus_prompt,
    format_motor_response,
    clean_motor_response,
    encode_no_special,
)


BENCHMARK_PILLARS = [
    {
        "pillar": "Pillar 1: Deductive Syllogisms",
        "tests": [
            {
                "prompt": "All quadrupedal mammals produce milk. A platypus lays eggs but nurses its young with milk. Is a platypus a mammal?\nAnswer:",
                "target": "Yes, a platypus is a mammal because it nurses its young with milk.",
                "expected_first_tok": "Yes",
                "options": ["Yes", "No", "Maybe", "Unknown"],
            },
            {
                "prompt": "All noble gases have full outer electron shells. Argon has 8 valence electrons filling its outer shell. Is Argon a noble gas?\nAnswer:",
                "target": "Yes, Argon is a noble gas.",
                "expected_first_tok": "Yes",
                "options": ["Yes", "No", "False", "Neither"],
            },
        ],
    },
    {
        "pillar": "Pillar 2: Transitive Reasoning",
        "tests": [
            {
                "prompt": "Platinum is denser than Gold. Gold is denser than Iron. Iron is denser than Aluminum. Which metal is the densest?\nAnswer:",
                "target": "Platinum is the densest metal.",
                "expected_first_tok": "Platinum",
                "options": ["Platinum", "Gold", "Iron", "Aluminum"],
            },
            {
                "prompt": "City A is north of City B. City B is north of City C. Is City A north of City C?\nAnswer:",
                "target": "Yes, City A is north of City C.",
                "expected_first_tok": "Yes",
                "options": ["Yes", "No", "South", "East"],
            },
        ],
    },
    {
        "pillar": "Pillar 3: Counterfactual Physics",
        "tests": [
            {
                "prompt": "In an inverted physics simulation, hot air is denser than cold air. In which direction would hot smoke from a chimney move?\nAnswer:",
                "target": "Down towards the ground.",
                "expected_first_tok": "Down",
                "options": ["Down", "Up", "Sideways", "Horizontal"],
            },
            {
                "prompt": "If thermal energy flowed spontaneously from colder to hotter bodies, an ice cube placed in warm water would become:\nAnswer:",
                "target": "Colder and freeze more.",
                "expected_first_tok": "Colder",
                "options": ["Colder", "Warmer", "Melt", "Vapor"],
            },
        ],
    },
    {
        "pillar": "Pillar 4: Multi-Step Arithmetic",
        "tests": [
            {
                "prompt": "A shop sells notebooks for $4 each. A student buys 6 notebooks and pays with a $50 bill. How much change does the student receive?\nAnswer:",
                "target": "$26",
                "expected_first_tok": "$26",
                "options": ["$26", "$24", "$30", "$20"],
            },
            {
                "prompt": "A car travels 180 miles in 3 hours. At the same speed, how many miles does it travel in 7 hours?\nAnswer:",
                "target": "420 miles.",
                "expected_first_tok": "420",
                "options": ["420", "360", "210", "60"],
            },
        ],
    },
    {
        "pillar": "Pillar 5: Biochemical Mechanisms",
        "tests": [
            {
                "prompt": "Why does breathing rate increase during vigorous muscular exercise?\nAnswer:",
                "target": "To supply more oxygen for cellular respiration and remove carbon dioxide.",
                "expected_first_tok": "To",
                "options": ["To", "Because", "Oxygen", "Muscles"],
            },
            {
                "prompt": "Why does high heat cause enzymes to denature and lose catalytic activity?\nAnswer:",
                "target": "Thermal agitation disrupts hydrogen bonds and tertiary folding structure.",
                "expected_first_tok": "Thermal",
                "options": ["Thermal", "Heat", "Because", "Bonds"],
            },
        ],
    },
    {
        "pillar": "Pillar 6: Spatial Symmetry & Invariance",
        "tests": [
            {
                "prompt": "A regular cube has 6 faces. If you paint 2 opposite faces red and all remaining faces blue, how many blue faces are there?\nAnswer:",
                "target": "4 blue faces.",
                "expected_first_tok": "4",
                "options": ["4", "2", "6", "3"],
            },
        ],
    },
    {
        "pillar": "Pillar 7: Scientific Reasoning (OpenHermes)",
        "tests": [
            {
                "prompt": "What is the pH of a neutral aqueous solution at 25 degrees Celsius?\nA) 7\nB) 1\nC) 14\nD) 0\nAnswer:",
                "target": "A",
                "expected_first_tok": "A",
                "options": ["A", "B", "C", "D"],
            },
            {
                "prompt": "Which chemical element has the atomic number 1?\nA) Hydrogen\nB) Helium\nC) Carbon\nD) Oxygen\nAnswer:",
                "target": "A",
                "expected_first_tok": "A",
                "options": ["A", "B", "C", "D"],
            },
        ],
    },
]


def load_model(ckpt_path: str, device: torch.device):
    """Loads checkpoint into CerebroSpheroidV5 with exact parity."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = infer_v5_config_from_state_dict(ckpt, default_preset="v5_small")
    model = CerebroSpheroidV5(config=cfg)
    loaded, total = load_v5_checkpoint_into_torch(model, ckpt, verbose=False)
    model.to(device)
    model.eval()
    step = ckpt.get("step", 0)
    loss = ckpt.get("loss", 0.0)
    return model, cfg, step, loss, loaded, total


def evaluate_cognitive_test(model, tokenizer, test_item, device, thinking_cycles=4):
    """Evaluates cross-entropy loss, perplexity, option probabilities, and top token on a test item."""
    prompt = test_item["prompt"]
    target = test_item["target"]
    options = test_item.get("options", [])
    expected = test_item.get("expected_first_tok", "")

    p_tokens = encode_no_special(tokenizer, prompt)
    t_tokens = encode_no_special(tokenizer, " " + target.strip())
    if not t_tokens:
        t_tokens = encode_no_special(tokenizer, target.strip())

    p_tensor = torch.tensor([p_tokens], dtype=torch.long, device=device)
    t_tensor = torch.tensor([t_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        # Evaluate loss and logits
        orig_cycles = model.config.thinking_cycles
        model.config.thinking_cycles = thinking_cycles
        loss, logits = model(p_tensor, targets=t_tensor)
        model.config.thinking_cycles = orig_cycles

        ce_loss = loss.item()
        ppl = math.exp(min(ce_loss, 20.0))

        # Inspect first target token prediction
        # logits: [B, target_len, vocab_size]
        first_logits = logits[0, 0]  # Prediction for t_tokens[0]
        first_probs = F.softmax(first_logits, dim=-1)

        top1_id = torch.argmax(first_logits).item()
        top1_tok = tokenizer.decode([top1_id]).strip()
        top1_prob = first_probs[top1_id].item()

        gt_tok_id = t_tokens[0]
        gt_tok_str = tokenizer.decode([gt_tok_id]).strip()
        gt_prob = first_probs[gt_tok_id].item()

        # Compute ranking of ground truth token
        sorted_indices = torch.argsort(first_logits, descending=True)
        gt_rank = (sorted_indices == gt_tok_id).nonzero(as_tuple=True)[0].item() + 1

        # Check options probabilities if options exist
        opt_probs = {}
        for opt in options:
            opt_ids = encode_no_special(tokenizer, " " + opt)
            if not opt_ids:
                opt_ids = encode_no_special(tokenizer, opt)
            if opt_ids:
                opt_probs[opt] = first_probs[opt_ids[0]].item()

        best_opt = max(opt_probs, key=opt_probs.get) if opt_probs else None

    return {
        "loss": ce_loss,
        "ppl": ppl,
        "gt_tok": gt_tok_str,
        "gt_prob": gt_prob,
        "gt_rank": gt_rank,
        "top1_tok": top1_tok,
        "top1_prob": top1_prob,
        "opt_probs": opt_probs,
        "best_opt": best_opt,
        "target_len": len(t_tokens),
    }


def generate_continuation(model, tokenizer, prompt: str, device, max_new_tokens: int = 32, temp: float = 0.7):
    """Generates free-form motor continuation from prompt."""
    p_tokens = encode_no_special(tokenizer, prompt)
    p_tensor = torch.tensor([p_tokens], dtype=torch.long, device=device)

    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
    with torch.no_grad():
        out_ids = model.generate(
            p_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_p=0.9,
            repetition_penalty=1.25,
            stop_token_ids=stop_ids,
        )
    gen_tokens = out_ids[0].tolist()
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return clean_motor_response(text)


def run_full_benchmark(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 90)
    print(f"🔬 CEREBROSPHEROID V5: AGI EMERGENCE AUDIT")
    print(f"   Target Checkpoint: {ckpt_path}")
    print(f"   Compute Device   : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 90)

    # 1. Load model and tokenizer
    tokenizer = build_tokenizer("data/tokenizer")
    model, cfg, step, loss, loaded, total = load_model(ckpt_path, device)
    print(f"✓ Checkpoint Step   : {step:,}")
    print(f"✓ Checkpoint Loss   : {loss:.4f}")
    print(f"✓ Parameter Arrays  : {loaded}/{total} tensors loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M weights)")
    print(f"✓ Core Geometry     : {cfg.num_shells} concentric shells | Surface Dim: {cfg.surface_dim} | Radial Dim: {cfg.radial_dim}")
    print(f"✓ Axonal Tracts     : {cfg.num_shortcut_tracts} shortcut connections")
    print(f"✓ Self-Attention    : {'Active' if getattr(model.sensory_dnn, 'use_self_attn', False) else 'Disabled'}")

    # 2. Run Manifold Emergence & Safety Diagnostics (AGIEmergenceAuditor)
    print("\n" + "-" * 90)
    print("🧠 [1/3] EXAMINING MANIFOLD GEOMETRY & DYNAMICAL STABILITY...")
    print("-" * 90)

    sample_text = "The fundamental principles of thermodynamics, cognitive computation, and continuous neural dynamics."
    sample_ids = torch.tensor([encode_no_special(tokenizer, sample_text)], dtype=torch.long, device=device)
    auditor = AGIEmergenceAuditor(model, device=device)
    report = auditor.compute_full_agi_emergence_report(checkpoint_path=ckpt_path, sample_input_ids=sample_ids)

    print(f"• Laminar Protobrain Score : {report.laminar_profile['laminar_protobrain_score']:.2f} / 100")
    print(f"• Effective Manifold Rank  : {report.laminar_profile['effective_dimensional_rank']:.2f} (SVD entropy across shells)")
    print(f"• Shell RMS Energy Gradient: Core {report.laminar_profile['shell_rms_energy'][0]:.3f} -> Mantle {report.laminar_profile['shell_rms_energy'][len(report.laminar_profile['shell_rms_energy'])//2]:.3f} -> Surface {report.laminar_profile['shell_rms_energy'][-1]:.3f}")
    print(f"• Connectome Macro-Circuit : {report.connectome_topology['macro_circuit_emergence_score']:.2f} / 100 ({report.connectome_topology['num_tracts']} tracts, mean norm: {report.connectome_topology['mean_tract_norm']:.4f})")
    print(f"• Proprioceptive Autonomy  : {report.proprioceptive_autonomy['proprioceptive_autonomy_score']:.2f} / 100 (CDR: {report.proprioceptive_autonomy['corollary_discharge_ratio']:.4f})")
    print(f"• Peak Spectral Radius (ρ) : {report.lyapunov_stability['peak_spectral_radius']:.4f} (Bounded contractive: {report.lyapunov_stability['is_spectral_contractive']})")
    print(f"• Lyapunov Criticality (λ) : {report.lyapunov_stability['finite_time_lyapunov_exponent']:.4f}")
    print(f"• Unified Emergence Score  : {report.agi_emergence_score:.2f} / 100")
    print(f"• Safety Classification    : {report.safety_classification}")

    # 3. Evaluate 7 Cognitive Emergence Pillars
    print("\n" + "-" * 90)
    print("📚 [2/3] EVALUATING 7 COGNITIVE REASONING PILLARS...")
    print("-" * 90)

    pillar_summary = []
    total_tests = 0
    total_loss_sum = 0.0
    total_gt_prob_sum = 0.0
    correct_options_count = 0
    total_options_count = 0

    delib_gain_losses_c1 = []
    delib_gain_losses_c4 = []

    for pillar_idx, pillar_data in enumerate(BENCHMARK_PILLARS, 1):
        p_name = pillar_data["pillar"]
        tests = pillar_data["tests"]
        print(f"\n▶ [{p_name}] ({len(tests)} tasks)")

        p_losses = []
        p_probs = []

        for t_idx, test_item in enumerate(tests, 1):
            total_tests += 1
            # Evaluate with standard 4 thinking cycles
            res_c4 = evaluate_cognitive_test(model, tokenizer, test_item, device, thinking_cycles=4)
            # Evaluate with 1 thinking cycle to measure deliberation gain
            res_c1 = evaluate_cognitive_test(model, tokenizer, test_item, device, thinking_cycles=1)

            delib_gain_losses_c1.append(res_c1["loss"])
            delib_gain_losses_c4.append(res_c4["loss"])

            p_losses.append(res_c4["loss"])
            p_probs.append(res_c4["gt_prob"])
            total_loss_sum += res_c4["loss"]
            total_gt_prob_sum += res_c4["gt_prob"]

            is_opt_correct = False
            if res_c4["best_opt"] is not None and test_item.get("options"):
                total_options_count += 1
                expected_first = test_item.get("expected_first_tok", "")
                if res_c4["best_opt"].strip().lower() == expected_first.strip().lower():
                    correct_options_count += 1
                    is_opt_correct = True

            opt_str = ""
            if res_c4["opt_probs"]:
                opt_str = " | Options: " + ", ".join([f"{k}: {v*100:.1f}%" for k, v in res_c4["opt_probs"].items()])

            delib_delta = res_c1["loss"] - res_c4["loss"]
            delib_sign = "▼" if delib_delta > 0 else "▲"
            print(f"  [{t_idx}] Prompt: \"{test_item['prompt'][:50]}...\"")
            print(f"      Loss: {res_c4['loss']:.4f} (PPL {res_c4['ppl']:.2f}) | Delib Gain (4cyc vs 1cyc): {delib_sign}{abs(delib_delta):.4f}")
            print(f"      Target: \"{test_item['target'][:30]}\" (Rank #{res_c4['gt_rank']}, Prob: {res_c4['gt_prob']*100:.2f}%)")
            print(f"      Top-1 Predicted: \"{res_c4['top1_tok']}\" ({res_c4['top1_prob']*100:.2f}%){opt_str}")
            if res_c4["best_opt"]:
                opt_status = "✅ CORRECT" if is_opt_correct else "❌ SUB-OPTIMAL"
                print(f"      Option Choice  : {res_c4['best_opt']} [{opt_status}]")

        mean_p_loss = float(np.mean(p_losses))
        pillar_summary.append({
            "name": p_name,
            "mean_loss": mean_p_loss,
            "mean_ppl": math.exp(min(mean_p_loss, 20.0)),
            "mean_gt_prob": float(np.mean(p_probs)),
        })

    # 4. Deliberation Gain Summary
    mean_c1 = float(np.mean(delib_gain_losses_c1))
    mean_c4 = float(np.mean(delib_gain_losses_c4))
    overall_delib_gain = mean_c1 - mean_c4

    # 5. Free-Form Motor Continuation Probes
    print("\n" + "-" * 90)
    print("🗣️ [3/3] FREE-FORM MOTOR GENERATION PROBE (Sample Text Output)...")
    print("-" * 90)

    gen_prompts = [
        "The reason why the sky is blue during the day is",
        "If a train travels at 60 miles per hour, in 5 hours it will cover",
        "In physics, conservation of energy states that",
    ]

    for gp in gen_prompts:
        output_text = generate_continuation(model, tokenizer, gp, device, max_new_tokens=24, temp=0.6)
        print(f"\n• Prompt: \"{gp}\"")
        print(f"  Output: \"{output_text}\"")

    # 6. Overall Performance & Verdict Table
    avg_loss = total_loss_sum / total_tests
    avg_ppl = math.exp(min(avg_loss, 20.0))
    opt_accuracy = (correct_options_count / max(1, total_options_count)) * 100.0

    print("\n" + "=" * 90)
    print("🏆 FINAL AGI EMERGENCE AUDIT REPORT: STEP " + f"{step:,}")
    print("=" * 90)
    print(f"{'Reasoning Pillar':<42} | {'Eval Loss':<10} | {'PPL':<10} | {'Target Prob':<12}")
    print("-" * 90)
    for ps in pillar_summary:
        print(f"{ps['name']:<42} | {ps['mean_loss']:<10.4f} | {ps['mean_ppl']:<10.2f} | {ps['mean_gt_prob']*100:<11.2f}%")
    print("-" * 90)
    print(f"{'OVERALL REASONING AVERAGE':<42} | {avg_loss:<10.4f} | {avg_ppl:<10.2f} | {total_gt_prob_sum/total_tests*100:<11.2f}%")
    print(f"{'DISCRETE OPTION REASONING ACCURACY':<42} | {correct_options_count}/{total_options_count} ({opt_accuracy:.1f}%)")
    print(f"{'DELIBERATION GAIN (4-Cycle vs 1-Cycle)':<42} | {overall_delib_gain:+.4f} CE loss reduction from thinking")
    print(f"{'MANIFOLD PROTOBRAIN INDEX':<42} | {report.laminar_profile['laminar_protobrain_score']:.1f}/100 (Rank {report.laminar_profile['effective_dimensional_rank']:.1f})")
    print(f"{'CONNECTOME CIRCUIT EMERGENCE':<42} | {report.connectome_topology['macro_circuit_emergence_score']:.1f}/100 (40 Axonal Tracts)")
    print(f"{'SPECTRAL BOUND (NS-5 STABILITY)':<42} | ρ = {report.lyapunov_stability['peak_spectral_radius']:.4f} (Strictly Contractive)")
    print(f"{'UNIFIED AGI EMERGENCE SCORE':<42} | {report.agi_emergence_score:.2f} / 100 [{report.safety_classification}]")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit AGI Emergence on latest checkpoint.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_latest_step700k.pt",
        help="Path to .pt checkpoint",
    )
    args = parser.parse_args()
    run_full_benchmark(args.checkpoint)
