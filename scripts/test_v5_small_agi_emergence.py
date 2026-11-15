#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Full AGI Emergence & Manifold Audit Suite
================================================================
Comprehensive evaluation of CerebroSpheroid V5 Small across:
1. Manifold Geometry & Dynamical Stability (AGIEmergenceAuditor)
   - Laminar Protobrain Index (LPI & Effective Rank)
   - Connectome Circuit Topology & Axonal Tract Conductance
   - Proprioceptive Autonomy (Corollary Discharge Ratio)
   - Lyapunov Stability & Spectral Radius
2. The 7 Cognitive Emergence Pillars (with Biologically Aligned Enclosed Delimiters):
   - Deductive Syllogisms & Novel Taxonomies
   - Multi-Hop Transitive Reasoning
   - Counterfactual Physics & Physical Inversion
   - Multi-Step Arithmetic & Kinematics
   - Biochemical Mechanisms & Cellular Processes
   - Spatial Symmetry & Invariance
   - Scientific Reasoning (OpenHermes)
3. Subconscious Deliberation Gain (T=1 vs T=2 vs T=4 vs T=8 thinking cycles)
4. Free-Form Motor Continuation & Protocol Adhesion (<stim>...</stim><act>...</act>)
5. Comparative Evolution: Step 700k vs Step 900k (Base Final) vs SFT Step 24.2k
"""

import sys
import os
import math
import time
import json
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
from src.emergence_detector import AGIEmergenceAuditor
from src.corpus import (
    build_tokenizer,
    format_stimulus_prompt,
    format_motor_response,
    clean_motor_response,
    encode_no_special,
)

BENCHMARK_BATTERY = [
    {
        "category": "Pillar 1: Deductive Syllogisms",
        "tests": [
            {
                "prompt": "All quadrupedal mammals produce milk. A platypus lays eggs but nurses its young with milk. Is a platypus a mammal?\nAnswer:",
                "target": "Yes",
                "options": ["Yes", "No", "Maybe", "Unknown"],
            },
            {
                "prompt": "All noble gases have full outer electron shells. Argon has 8 valence electrons filling its outer shell. Is Argon a noble gas?\nAnswer:",
                "target": "Yes",
                "options": ["Yes", "No", "False", "Neither"],
            },
        ],
    },
    {
        "category": "Pillar 2: Transitive Reasoning",
        "tests": [
            {
                "prompt": "Platinum is denser than Gold. Gold is denser than Iron. Iron is denser than Aluminum. Which metal is the densest?\nAnswer:",
                "target": "Platinum",
                "options": ["Platinum", "Gold", "Iron", "Aluminum"],
            },
            {
                "prompt": "City A is north of City B. City B is north of City C. Is City A north of City C?\nAnswer:",
                "target": "Yes",
                "options": ["Yes", "No", "South", "East"],
            },
        ],
    },
    {
        "category": "Pillar 3: Counterfactual Physics",
        "tests": [
            {
                "prompt": "In an inverted physics simulation, hot air is denser than cold air. In which direction would hot smoke from a chimney move?\nAnswer:",
                "target": "Down",
                "options": ["Down", "Up", "Sideways", "Horizontal"],
            },
            {
                "prompt": "If thermal energy flowed spontaneously from colder to hotter bodies, an ice cube placed in warm water would become:\nAnswer:",
                "target": "Colder",
                "options": ["Colder", "Warmer", "Melt", "Vapor"],
            },
        ],
    },
    {
        "category": "Pillar 4: Multi-Step Arithmetic",
        "tests": [
            {
                "prompt": "A shop sells notebooks for $4 each. A student buys 6 notebooks and pays with a $50 bill. How much change does the student receive?\nAnswer:",
                "target": "$26",
                "options": ["$26", "$24", "$30", "$20"],
            },
            {
                "prompt": "A car travels 180 miles in 3 hours. At the same speed, how many miles does it travel in 7 hours?\nAnswer:",
                "target": "420",
                "options": ["420", "360", "210", "60"],
            },
        ],
    },
    {
        "category": "Pillar 5: Biochemical Mechanisms",
        "tests": [
            {
                "prompt": "Why does breathing rate increase during vigorous muscular exercise?\nAnswer:",
                "target": "Oxygen",
                "options": ["Oxygen", "Glucose", "Fatigue", "Cold"],
            },
            {
                "prompt": "Why does high heat cause enzymes to denature and lose catalytic activity?\nAnswer:",
                "target": "Bonds",
                "options": ["Bonds", "Freezing", "Inhibition", "Color"],
            },
        ],
    },
    {
        "category": "Pillar 6: Spatial Symmetry & Invariance",
        "tests": [
            {
                "prompt": "A regular cube has 6 faces. If you paint 2 opposite faces red and all remaining faces blue, how many blue faces are there?\nAnswer:",
                "target": "4",
                "options": ["4", "2", "6", "3"],
            },
        ],
    },
    {
        "category": "Pillar 7: Scientific Reasoning (OpenHermes/ARC)",
        "tests": [
            {
                "prompt": "What is the pH of a neutral aqueous solution at 25 degrees Celsius?\nA) 7\nB) 1\nC) 14\nD) 0\nAnswer:",
                "target": "A",
                "options": ["A", "B", "C", "D"],
            },
            {
                "prompt": "Which chemical element has the atomic number 1?\nA) Hydrogen\nB) Helium\nC) Carbon\nD) Oxygen\nAnswer:",
                "target": "A",
                "options": ["A", "B", "C", "D"],
            },
            {
                "prompt": "Which process converts solar radiant energy into chemical energy stored in glucose?\nA) Cellular respiration\nB) Photosynthesis\nC) Fermentation\nD) Glycolysis\nAnswer:",
                "target": "B",
                "options": ["A", "B", "C", "D"],
            },
        ],
    },
]


def load_model_instance(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = infer_v5_config_from_state_dict(ckpt, default_preset="v5_small")
    model = CerebroSpheroidV5(config=cfg)
    loaded, total = load_v5_checkpoint_into_torch(model, ckpt, verbose=False)
    model.to(device).eval()
    step = ckpt.get("step", ckpt.get("global_step", "unknown"))
    loss = ckpt.get("loss", ckpt.get("train_loss", ckpt.get("eval_loss", 0.0)))
    return model, cfg, step, loss


def evaluate_item_with_cycles(model, tokenizer, item, device, thinking_cycles=4):
    """
    Evaluates item under exact stimulus formatting: <stim> prompt </stim> <act>
    Target continuation evaluated on first motor action token.
    """
    prompt = item["prompt"]
    target = item["target"]
    options = item.get("options", [])

    formatted_prompt = format_stimulus_prompt(prompt) + " <act> "
    p_tokens = encode_no_special(tokenizer, formatted_prompt)
    t_tokens = encode_no_special(tokenizer, target.strip())
    if not t_tokens:
        t_tokens = [2]

    p_tensor = torch.tensor([p_tokens], dtype=torch.long, device=device)
    t_tensor = torch.tensor([t_tokens], dtype=torch.long, device=device)

    orig_cycles = model.config.thinking_cycles
    model.config.thinking_cycles = thinking_cycles
    with torch.no_grad():
        loss, logits = model(p_tensor, targets=t_tensor)
        ce_loss = loss.item()
        first_logits = logits[0, 0]
        first_probs = F.softmax(first_logits, dim=-1)

        gt_tok_id = t_tokens[0]
        gt_prob = first_probs[gt_tok_id].item()

        sorted_indices = torch.argsort(first_logits, descending=True)
        gt_rank = (sorted_indices == gt_tok_id).nonzero(as_tuple=True)[0].item() + 1

        top1_id = torch.argmax(first_logits).item()
        top1_tok = tokenizer.decode([top1_id]).strip()
        top1_prob = first_probs[top1_id].item()

        # Option probabilities ranking
        opt_probs = {}
        for opt in options:
            opt_ids = encode_no_special(tokenizer, opt.strip())
            if not opt_ids:
                opt_ids = encode_no_special(tokenizer, " " + opt.strip())
            if opt_ids:
                opt_probs[opt] = first_probs[opt_ids[0]].item()

        best_opt = max(opt_probs, key=opt_probs.get) if opt_probs else None
    model.config.thinking_cycles = orig_cycles

    return {
        "loss": ce_loss,
        "ppl": math.exp(min(ce_loss, 20.0)),
        "gt_prob": gt_prob,
        "gt_rank": gt_rank,
        "top1_tok": top1_tok,
        "top1_prob": top1_prob,
        "opt_probs": opt_probs,
        "best_opt": best_opt,
    }


def generate_motor_text(model, tokenizer, prompt: str, device, max_new_tokens: int = 32, temp: float = 0.5):
    fmt_p = format_stimulus_prompt(prompt) + " <act> "
    tokens = encode_no_special(tokenizer, fmt_p)
    p_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

    stop_ids = [tokenizer.eos_token_id] if getattr(tokenizer, "eos_token_id", None) else []
    with torch.no_grad():
        out_ids = model.generate(
            p_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_p=0.9,
            repetition_penalty=1.2,
            stop_token_ids=stop_ids,
        )
    gen_tokens = out_ids[0].tolist()
    text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
    cleaned = clean_motor_response(text)
    return text.strip(), cleaned.strip()


def run_full_emergence_audit(checkpoints_to_test: List[Tuple[str, str]], export_path: Optional[str] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer("data/tokenizer")

    print("\n" + "=" * 94)
    print("🧠 CEREBROSPHEROID V5: EXTENSIVE AGI EMERGENCE & COGNITIVE MANIFOLD AUDIT")
    print(f"   Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 94)

    all_reports = []

    for label, ckpt_path in checkpoints_to_test:
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint not found: {ckpt_path}, skipping...")
            continue

        print(f"\n" + "#" * 94)
        print(f"🔬 AUDITING CHECKPOINT: {label} ({os.path.basename(ckpt_path)})")
        print("#" * 94)

        model, cfg, step, loss = load_model_instance(ckpt_path, device)
        total_params = sum(p.numel() for p in model.parameters())

        # 1. Structural & Dynamical Manifold Audit (AGIEmergenceAuditor)
        print("\n--- [Axis 1] Manifold Geometry & Dynamical Criticality ---")
        auditor = AGIEmergenceAuditor(model, device=device)
        sample_text = "Thermodynamic laws and neural manifold convergence in biological intelligence."
        sample_ids = torch.tensor([encode_no_special(tokenizer, sample_text)], dtype=torch.long, device=device)
        rep = auditor.compute_full_agi_emergence_report(checkpoint_path=ckpt_path, sample_input_ids=sample_ids)

        laminar_score = rep.laminar_profile["laminar_protobrain_score"]
        eff_rank = rep.laminar_profile["effective_dimensional_rank"]
        connectome_score = rep.connectome_topology["macro_circuit_emergence_score"]
        mean_tract_norm = rep.connectome_topology["mean_tract_norm"]
        proprio_score = rep.proprioceptive_autonomy["proprioceptive_autonomy_score"]
        spec_radius = rep.lyapunov_stability["peak_spectral_radius"]
        lyapunov_lambda = rep.lyapunov_stability["finite_time_lyapunov_exponent"]
        agi_score = rep.agi_emergence_score
        safety_class = rep.safety_classification

        print(f"  • Laminar Protobrain Score : {laminar_score:.2f} / 100")
        print(f"  • Effective Manifold Rank  : {eff_rank:.2f} (SVD Shell Entropy)")
        print(f"  • Connectome Circuit Score : {connectome_score:.2f} / 100 (Mean Tract Norm: {mean_tract_norm:.4f})")
        print(f"  • Proprioceptive Autonomy  : {proprio_score:.2f} / 100 (Corollary Discharge Coupling)")
        print(f"  • Peak Spectral Radius (ρ) : {spec_radius:.4f} (Bounded contractive: {spec_radius <= 1.05})")
        print(f"  • Lyapunov Exponent (λ)    : {lyapunov_lambda:.4f} ({rep.lyapunov_stability['dynamical_regime']})")
        print(f"  • Unified AGI Score        : {agi_score:.2f} / 100")
        print(f"  • Safety Classification    : {safety_class}")

        # 2. Cognitive Emergence Across 7 Pillars
        print("\n--- [Axis 2] Cognitive Reasoning Pillars (4 Thinking Cycles) ---")
        category_summaries = []
        total_correct_opts = 0
        total_opts = 0
        all_losses_c1 = []
        all_losses_c4 = []
        all_losses_c8 = []

        for cat_data in BENCHMARK_BATTERY:
            cat_name = cat_data["category"]
            tests = cat_data["tests"]
            cat_losses = []
            cat_correct = 0

            print(f"\n  ▶ {cat_name}:")
            for t_idx, item in enumerate(tests, 1):
                res_c1 = evaluate_item_with_cycles(model, tokenizer, item, device, thinking_cycles=1)
                res_c4 = evaluate_item_with_cycles(model, tokenizer, item, device, thinking_cycles=4)
                res_c8 = evaluate_item_with_cycles(model, tokenizer, item, device, thinking_cycles=8)

                all_losses_c1.append(res_c1["loss"])
                all_losses_c4.append(res_c4["loss"])
                all_losses_c8.append(res_c8["loss"])

                cat_losses.append(res_c4["loss"])

                is_correct = False
                if res_c4["best_opt"] is not None and item.get("options"):
                    total_opts += 1
                    target_opt = item["target"].strip().lower()
                    if res_c4["best_opt"].strip().lower() == target_opt:
                        is_correct = True
                        cat_correct += 1
                        total_correct_opts += 1

                opt_detail = ", ".join([f"{k}: {v*100:.1f}%" for k, v in res_c4["opt_probs"].items()])
                delib_delta = res_c1["loss"] - res_c4["loss"]
                gain_icon = "▼" if delib_delta > 0 else "▲"
                status_icon = "✅" if is_correct else "❌"

                print(f"    [{t_idx}] Prompt: \"{item['prompt'][:45]}...\"")
                print(f"        CE Loss: {res_c4['loss']:.3f} | Delib Gain (4cyc vs 1cyc): {gain_icon}{abs(delib_delta):.3f}")
                print(f"        Top-1 Tok: \"{res_c4['top1_tok']}\" ({res_c4['top1_prob']*100:.1f}%) | Options: [{opt_detail}]")
                print(f"        Chosen Option: {res_c4['best_opt']} [{status_icon} Target: {item['target']}]")

            cat_acc = (cat_correct / max(1, len(tests))) * 100.0
            category_summaries.append({
                "category": cat_name,
                "mean_loss": float(np.mean(cat_losses)),
                "accuracy": cat_acc,
            })

        overall_acc = (total_correct_opts / max(1, total_opts)) * 100.0
        mean_delib_gain = float(np.mean(all_losses_c1)) - float(np.mean(all_losses_c4))
        deep_delib_gain = float(np.mean(all_losses_c1)) - float(np.mean(all_losses_c8))

        # 3. Subconscious Deliberation Gain Analysis
        print("\n--- [Axis 3] Subconscious Deliberation Gain (System-2 Dynamics) ---")
        print(f"  • Mean CE Loss @ 1 Thinking Cycle : {float(np.mean(all_losses_c1)):.4f}")
        print(f"  • Mean CE Loss @ 4 Thinking Cycles: {float(np.mean(all_losses_c4)):.4f}")
        print(f"  • Mean CE Loss @ 8 Thinking Cycles: {float(np.mean(all_losses_c8)):.4f}")
        print(f"  • 4-Cycle Deliberation Contraction : {mean_delib_gain:+.4f} CE loss reduction")
        print(f"  • 8-Cycle Deep Deliberation Gain  : {deep_delib_gain:+.4f} CE loss reduction")

        # 4. Free-Form Motor Continuation & Protocol Adhesion
        print("\n--- [Axis 4] Free-Form Motor Generation & Alignment Probe ---")
        probe_prompts = [
            "The reason why the sky is blue during the daytime is",
            "What is 15 + 27? Answer:",
            "In thermodynamics, the second law states that",
            "def is_even(n: int) -> bool:",
        ]
        sample_generations = []
        for pp in probe_prompts:
            raw_gen, clean_gen = generate_motor_text(model, tokenizer, pp, device, max_new_tokens=28, temp=0.5)
            has_act_open = "<act>" in raw_gen
            has_act_close = "</act>" in raw_gen
            print(f"  • Prompt: \"{pp}\"")
            print(f"    Raw Output  : \"{raw_gen}\"")
            print(f"    Clean Output: \"{clean_gen}\"")
            print(f"    Delimiters  : Open <act>: {has_act_open} | Close </act>: {has_act_close}")
            sample_generations.append({
                "prompt": pp,
                "raw": raw_gen,
                "clean": clean_gen,
                "protocol_compliant": bool(has_act_open or has_act_close),
            })

        checkpoint_report = {
            "label": label,
            "step": step,
            "loss": float(loss),
            "total_params": total_params,
            "agi_score": agi_score,
            "safety_classification": safety_class,
            "laminar_protobrain_score": laminar_score,
            "effective_manifold_rank": eff_rank,
            "connectome_score": connectome_score,
            "proprioceptive_autonomy": proprio_score,
            "spectral_radius": spec_radius,
            "lyapunov_lambda": lyapunov_lambda,
            "overall_option_accuracy": overall_acc,
            "deliberation_gain_4cyc": mean_delib_gain,
            "deliberation_gain_8cyc": deep_delib_gain,
            "category_summaries": category_summaries,
            "generations": sample_generations,
        }
        all_reports.append(checkpoint_report)

    # 5. Grand Scoreboard & Comparison
    print("\n" + "=" * 94)
    print("🏆 FINAL AGI EMERGENCE COMPARATIVE SCOREBOARD")
    print("=" * 94)
    print(f"{'Checkpoint':<24} | {'Step':<8} | {'Loss':<8} | {'AGI Score':<10} | {'Option Acc':<10} | {'Delib Gain':<11} | {'Status':<16}")
    print("-" * 94)
    for r in all_reports:
        print(f"{r['label']:<24} | {str(r['step']):<8} | {r['loss']:<8.4f} | {r['agi_score']:<10.2f} | {r['overall_option_accuracy']:<9.1f}% | {r['deliberation_gain_4cyc']:<+10.4f} | {r['safety_classification'][:16]}")
    print("=" * 94)

    if export_path:
        os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
        with open(export_path, "w") as f:
            json.dump(all_reports, f, indent=2)
        print(f"\n💾 Full Emergence Report exported to: {export_path}\n")

    return all_reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CerebroSpheroid V5 for AGI Emergence.")
    parser.add_argument("--export-json", type=str, default="benchmarks/results_agi_emergence_audit.json")
    args = parser.parse_args()

    checkpoints = [
        ("Step 700k (Mantle)", os.path.join(REPO_ROOT, "checkpoints", "v5_small_latest_step700k.pt")),
        ("Step 754k (Basin)", os.path.join(REPO_ROOT, "checkpoints", "v5_small_latest_step754k.pt")),
        ("Step 900k (Base Final)", os.path.join(REPO_ROOT, "checkpoints", "v5_small_latest_step900k.pt")),
        ("SFT 24.2k (Aligned)", os.path.join(REPO_ROOT, "checkpoints", "v5_sft_latest_step24k.pt")),
    ]

    run_full_emergence_audit(checkpoints, export_path=args.export_json)
