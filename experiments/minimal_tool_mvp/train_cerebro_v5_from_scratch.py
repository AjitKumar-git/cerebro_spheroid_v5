"""
Train CerebroSpheroid V5 from Scratch on Minimal Tool Dataset
============================================================
Trains the native CerebroSpheroid V5 spherical shell architecture from scratch
(random initialization) on the exact same synthetic tool-augmented dataset.
Validates:
1. Whether CerebroSpheroid V5 learns to emit <tool_call> when needed and answer directly when not.
2. Tool schema validity and deterministic execution.
3. Turn-2 closed-loop feedback and final answer accuracy.
4. Direct testing on "How many days in a leap year?".
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5
from src.spheroid_optimizer import build_scmer_optimizer
from src.corpus import format_stimulus_prompt, parse_model_output
from experiments.minimal_tool_mvp.dataset_generator import TraceDatasetGenerator, ToolTrace
from experiments.minimal_tool_mvp.schema_validator import DeterministicToolExecutor, validate_tool_call
from experiments.minimal_tool_mvp.evaluate import clean_text_for_comparison


class SpheroidTrainingItem:
    def __init__(self, prompt: str, target: str, is_tool_call: bool = False):
        self.prompt = prompt
        self.target = target
        self.is_tool_call = is_tool_call


def prepare_spheroid_data(traces: List[ToolTrace]) -> List[SpheroidTrainingItem]:
    """Prepares stimulus-response training items for CerebroSpheroid V5."""
    items: List[SpheroidTrainingItem] = []

    for t in traces:
        if t.case in ("no_tool_needed", "tool_bad_choice"):
            # Case 1 & 3: Direct Answer
            p = f"Question: {t.user_request}\nAnswer:"
            tgt = f" {t.final_answer}"
            items.append(SpheroidTrainingItem(prompt=p, target=tgt, is_tool_call=False))

        else:
            # Case 2: Tool Needed
            call_str = json.dumps(t.tool_call, separators=(",", ":"))
            res_str = json.dumps(t.tool_result, separators=(",", ":"))

            # Turn 1: Emit tool call
            p1 = f"Question: {t.user_request}\nAnswer:"
            tgt1 = f" <tool_call> {call_str} </tool_call>"
            items.append(SpheroidTrainingItem(prompt=p1, target=tgt1, is_tool_call=True))

            # Turn 2: Feed back tool result and emit final answer
            p2 = (
                f"Question: {t.user_request}\nAnswer:\n"
                f"<tool_call> {call_str} </tool_call>\n"
                f"<tool_response> {res_str} </tool_response>\n"
                f"Final Answer:"
            )
            tgt2 = f" {t.final_answer}"
            items.append(SpheroidTrainingItem(prompt=p2, target=tgt2, is_tool_call=False))

    return items


def collate_spheroid_batch(
    batch: List[SpheroidTrainingItem], tokenizer: Any, pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    eos_id = getattr(tokenizer, "eos_token_id", 2) or 2
    p_ids_list = [tokenizer.encode(item.prompt, add_special_tokens=False) for item in batch]
    t_ids_list = [tokenizer.encode(item.target, add_special_tokens=False) + [eos_id] for item in batch]

    max_p = max(len(p) for p in p_ids_list)
    max_t = max(len(t) for t in t_ids_list)
    B = len(batch)

    prompts = torch.full((B, max_p), pad_id, dtype=torch.long)
    # Pad targets with -100 so cross_entropy ignores padding (no <unk> token training!)
    targets = torch.full((B, max_t), -100, dtype=torch.long)

    for i in range(B):
        prompts[i, :len(p_ids_list[i])] = torch.tensor(p_ids_list[i], dtype=torch.long)
        targets[i, :len(t_ids_list[i])] = torch.tensor(t_ids_list[i], dtype=torch.long)

    return prompts, targets


def train_cerebro_v5(
    num_train_traces: int = 1500,
    num_steps: int = 400,
    batch_size: int = 16,
    lr: float = 0.002,
    seed: int = 42,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training CerebroSpheroid V5 from Scratch ===")
    print(f"Device: {device} | Seed: {seed} | Steps: {num_steps} | Batch Size: {batch_size}")

    # 1. Load Tokenizer
    tok_dir = os.path.join(REPO_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    # 2. Generate Dataset
    print(f"\n[1/4] Generating synthetic traces...")
    gen = TraceDatasetGenerator(seed=seed)
    train_traces = gen.generate_dataset(num_train_traces)
    test_traces = gen.generate_dataset(150)

    train_items = prepare_spheroid_data(train_traces)
    print(f"Prepared {len(train_items)} training rollouts from {num_train_traces} traces.")

    # 3. Instantiate CerebroSpheroid V5 from Scratch (Random Init)
    print(f"\n[2/4] Initializing CerebroSpheroid V5 model from scratch...")
    cfg = ModelConfigV5(
        vocab_size=len(tokenizer),
        surface_dim=256,
        radial_dim=128,
        num_shells=8,
        num_shortcut_tracts=16,
        num_heads=4,
        thinking_cycles=2,
        phase_scale=2.0,
    )
    model = CerebroSpheroidV5(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Initialized CerebroSpheroid V5 architecture ({params:,} parameters).")

    optimizer = build_scmer_optimizer(
        model=model,
        lr=lr,
        scmer_lambda=0.005,
        momentum=0.95,
        weight_decay=0.01,
        spectral_clamp=1.0,
    )

    # 4. Training Loop
    print(f"\n[3/4] Training for {num_steps} steps...")
    model.train()
    start_time = time.time()
    random.seed(seed)

    for step in range(1, num_steps + 1):
        batch_items = random.sample(train_items, batch_size)
        prompts, targets = collate_spheroid_batch(batch_items, tokenizer, pad_id)
        prompts = prompts.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        loss, _ = model(prompts=prompts, targets=targets, deliberation_mode="teacher_forcing")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == 1 or step == num_steps:
            elapsed = time.time() - start_time
            print(f"[Step {step:03d}/{num_steps:03d}] Loss: {loss.item():.4f} | Speed: {step/elapsed:.2f} step/s", flush=True)

    print(f"Training completed in {time.time() - start_time:.2f}s.")

    # 5. Evaluation across the 3 cases
    print(f"\n[4/4] Evaluating CerebroSpheroid V5 on 150 held-out traces...")
    model.eval()
    executor = DeterministicToolExecutor()

    c1_c3_count = 0
    c1_c3_correct = 0
    c2_count = 0
    c2_tool_called = 0
    schema_valid_count = 0
    exec_success_count = 0
    c2_final_correct = 0

    for trace in test_traces:
        prompt = f"Question: {trace.user_request}\nAnswer:"
        with torch.no_grad():
            out_raw = model.generate_text(
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=48,
                temperature=0.0,
                device=device,
                return_parsed=False,
                execute_tools=False,
            )

        parsed = parse_model_output(out_raw)
        has_tool = len(parsed.get("tool_calls", [])) > 0

        if trace.case in ("no_tool_needed", "tool_bad_choice"):
            c1_c3_count += 1
            ans = parsed.get("answer", out_raw).strip()
            # Match check
            if (
                clean_text_for_comparison(ans) == clean_text_for_comparison(trace.final_answer)
                or trace.final_answer.strip().lower() in ans.lower()
            ):
                c1_c3_correct += 1
        else:
            c2_count += 1
            if has_tool:
                c2_tool_called += 1
                raw_call = parsed["tool_calls"][0]
                is_valid, err, parsed_dict = validate_tool_call(raw_call)
                if is_valid:
                    schema_valid_count += 1
                    exec_res = executor.execute(parsed_dict)
                    if exec_res.get("status") == "ok":
                        exec_success_count += 1
                    res_str = json.dumps(exec_res, separators=(",", ":"))
                else:
                    res_str = json.dumps({"status": "error", "message": err}, separators=(",", ":"))

                # Turn 2: Feed back result and generate final answer
                t2_prompt = (
                    f"Question: {trace.user_request}\nAnswer:\n"
                    f"<tool_call> {raw_call} </tool_call>\n"
                    f"<tool_response> {res_str} </tool_response>\n"
                    f"Final Answer:"
                )
                with torch.no_grad():
                    t2_out = model.generate_text(
                        prompt=t2_prompt,
                        tokenizer=tokenizer,
                        max_new_tokens=32,
                        temperature=0.0,
                        device=device,
                        return_parsed=False,
                        execute_tools=False,
                    )
                t2_clean = clean_text_for_comparison(t2_out)
                if clean_text_for_comparison(trace.final_answer) in t2_clean or trace.final_answer.strip().lower() in t2_out.lower():
                    c2_final_correct += 1

    print("\n" + "=" * 75)
    print("       CEREBRO-SPHEROID V5 (TRAINED FROM SCRATCH) AUDIT REPORT")
    print("=" * 75)
    print(f"Direct-Answer Accuracy (Case 1 & 3): {c1_c3_correct/max(1, c1_c3_count)*100:.2f}% ({c1_c3_correct}/{c1_c3_count})")
    print(f"Tool-Call Emission Rate (Case 2)   : {c2_tool_called/max(1, c2_count)*100:.2f}% ({c2_tool_called}/{c2_count})")
    print(f"Tool Schema Validity (Emitted)     : {schema_valid_count/max(1, c2_tool_called)*100:.2f}% ({schema_valid_count}/{c2_tool_called})")
    print(f"Tool Execution Success             : {exec_success_count/max(1, c2_tool_called)*100:.2f}% ({exec_success_count}/{c2_tool_called})")
    print(f"Turn 2 Post-Result Accuracy (Case2): {c2_final_correct/max(1, c2_count)*100:.2f}% ({c2_final_correct}/{c2_count})")
    print("=" * 75)

    # 6. Specific Leap Year and Benchmark Prompts Live Test
    print("\n" + "=" * 75)
    print("                     SPECIFIC PROMPT AUDIT")
    print("=" * 75)

    probe_queries = [
        "How many days in a leap year?",
        "How many days are in a leap year?",
        "What is the capital of France?",
        "Calculate 25 * 4.",
        "Compute 8 ^ 5.",
        "Do not call any tool: write 'mountain' in uppercase.",
    ]

    for q in probe_queries:
        print(f"\n[Prompt]: \"{q}\"")
        prompt_str = f"Question: {q}\nAnswer:"
        with torch.no_grad():
            t1_out = model.generate_text(
                prompt=prompt_str,
                tokenizer=tokenizer,
                max_new_tokens=48,
                temperature=0.0,
                device=device,
                return_parsed=False,
                execute_tools=False,
            )

        parsed_q = parse_model_output(t1_out)
        if parsed_q.get("tool_calls"):
            raw_call = parsed_q["tool_calls"][0]
            print(f"Turn 1 Emitted Action : <tool_call> {raw_call} </tool_call>")
            is_valid, _, parsed_d = validate_tool_call(raw_call)
            exec_res = executor.execute(parsed_d) if is_valid else {"status": "error"}
            res_str = json.dumps(exec_res, separators=(",", ":"))
            print(f"Executed Tool Result  : {res_str}")

            t2_p = (
                f"Question: {q}\nAnswer:\n"
                f"<tool_call> {raw_call} </tool_call>\n"
                f"<tool_response> {res_str} </tool_response>\n"
                f"Final Answer:"
            )
            with torch.no_grad():
                t2_out = model.generate_text(
                    prompt=t2_p,
                    tokenizer=tokenizer,
                    max_new_tokens=32,
                    temperature=0.0,
                    device=device,
                    return_parsed=False,
                    execute_tools=False,
                )
            print(f"Turn 2 Model Answer   : \"{t2_out.strip()}\"")
        else:
            print(f"Model Direct Answer   : \"{t1_out.strip()}\"")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CerebroSpheroid V5 from Scratch on Minimal Tool Dataset")
    parser.add_argument("--steps", type=int, default=400, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate for SCMER optimizer")
    parser.add_argument("--traces", type=int, default=1500, help="Number of training traces")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    train_cerebro_v5(
        num_train_traces=args.traces,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
