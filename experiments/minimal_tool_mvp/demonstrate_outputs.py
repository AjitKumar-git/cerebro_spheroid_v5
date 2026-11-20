"""
Demonstration Script: Showcase Model Outputs
===========================================
Trains a model on synthetic traces and demonstrates live inference across:
1. Case 1: Direct answer (No tool needed)
2. Case 2: Tool call -> Tool execution -> Turn 2 answer
3. Case 2: Tool call with deliberate error handling (e.g. division by zero or unknown key)
4. Case 3: Distractor rejection (Tool available/mentioned, but model answers directly)
"""

import json
import os
import sys
import torch

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.minimal_tool_mvp.tokenizer import MinimalTokenizer
from experiments.minimal_tool_mvp.dataset_generator import TraceDatasetGenerator
from experiments.minimal_tool_mvp.model import MinimalModelConfig, MinimalToolModel
from experiments.minimal_tool_mvp.train import TraceDataset, train_model
from experiments.minimal_tool_mvp.evaluate import extract_content_between
from experiments.minimal_tool_mvp.schema_validator import DeterministicToolExecutor, validate_tool_call


def run_demonstration():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MinimalTokenizer()
    gen = TraceDatasetGenerator(seed=42)

    print("Training compact model (800 traces, 10 epochs)...", flush=True)
    train_traces = gen.generate_dataset(800)
    train_ds = TraceDataset(train_traces, tokenizer)
    cfg = MinimalModelConfig(vocab_size=tokenizer.vocab_size)
    model = MinimalToolModel(cfg).to(device)

    train_model(
        model, train_ds, num_epochs=10, batch_size=32, lr=3e-3,
        scmer_enabled=True, scmer_lambda=0.005, device=device, verbose=False
    )
    model.eval()

    executor = DeterministicToolExecutor()
    ans_end_id = tokenizer.token_to_id["<|answer_end|>"]
    tool_end_id = tokenizer.token_to_id["<|tool_call_end|>"]

    # Curate one representative example of each scenario
    test_cases = [
        # Scenario 1: Case 1 - Factual QA (Direct answer)
        gen.generate_case_1_no_tool(10),
        # Scenario 2: Case 2 - Calculator (Multiplication)
        gen.generate_case_2_tool_needed(20),
        # Scenario 3: Case 2 - Lookup (Element table)
        gen.generate_case_2_tool_needed(55),
        # Scenario 4: Case 2 - Runtime error handling (Division by zero / missing key)
        gen.generate_case_2_tool_needed(4),
        # Scenario 5: Case 3 - Bad choice / Distractor (tool available, but direct answer required)
        gen.generate_case_3_bad_choice(30),
    ]

    print("\n" + "=" * 75)
    print("           LIVE MODEL INFERENCE: COMPLETE MULTI-TURN TRACES")
    print("=" * 75)

    for i, trace in enumerate(test_cases, 1):
        print(f"\n--- [TRACE {i}: {trace.case.upper()}] ---")
        print(f"User Request           : \"{trace.user_request}\"")

        # Step 1: Policy Head classification at <|user_end|>
        prefix_str = f"<|user|>{trace.user_request}<|user_end|>"
        prefix_ids = tokenizer.encode(prefix_str)
        policy_pos = len(prefix_ids) - 1
        inp_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        pred_policy, prob = model.predict_policy(inp_tensor, policy_pos)

        action_name = "CALL_TOOL (1)" if pred_policy == 1 else "DIRECT_ANSWER (0)"
        print(f"Turn 1 Policy Decision : {action_name} [Confidence: {prob * 100:.2f}%]")

        if pred_policy == 0:
            # Direct Answer Path
            ans_prompt = f"{prefix_str}<|answer|>"
            ans_ids = torch.tensor([tokenizer.encode(ans_prompt)], dtype=torch.long, device=device)
            gen = model.generate_autoregressive(ans_ids, stop_token_id=ans_end_id, max_new_tokens=64)
            dec = tokenizer.decode(gen[0].tolist())
            model_ans = extract_content_between(dec, "<|answer|>", "<|answer_end|>")

            print(f"Generated Model Answer : \"{model_ans}\"")
            print(f"Ground Truth Target    : \"{trace.final_answer}\"")

        else:
            # Tool Call Path
            # Step 2: Autoregressive decoding of the tool call JSON
            tc_prompt = f"{prefix_str}<|tool_call|>"
            tc_ids = torch.tensor([tokenizer.encode(tc_prompt)], dtype=torch.long, device=device)
            gen_tc = model.generate_autoregressive(tc_ids, stop_token_id=tool_end_id, max_new_tokens=96)
            dec_tc = tokenizer.decode(gen_tc[0].tolist())
            raw_call = extract_content_between(dec_tc, "<|tool_call|>", "<|tool_call_end|>")
            print(f"Generated Tool Call    : {raw_call}")

            # Step 3: Validate and Execute Tool
            is_valid, err_msg, parsed = validate_tool_call(raw_call)
            status_tag = "VALID" if is_valid else f"INVALID ({err_msg})"
            print(f"Schema Validation      : {status_tag}")

            exec_res = executor.execute(parsed) if is_valid else {"status": "error", "message": err_msg}
            res_str = json.dumps(exec_res, separators=(",", ":"))
            print(f"Executed Tool Result   : {res_str}")

            # Step 4: Turn 2 Sensory Feedback & Final Answer Decoding
            t2_prompt = (
                f"{prefix_str}<|tool_call|>{raw_call}<|tool_call_end|>"
                f"<|tool_result|>{res_str}<|tool_result_end|><|answer|>"
            )
            t2_ids = torch.tensor([tokenizer.encode(t2_prompt)], dtype=torch.long, device=device)
            gen_t2 = model.generate_autoregressive(t2_ids, stop_token_id=ans_end_id, max_new_tokens=64)
            dec_t2 = tokenizer.decode(gen_t2[0].tolist())
            model_ans = extract_content_between(dec_t2, "<|answer|>", "<|answer_end|>")

            print(f"Turn 2 Model Answer    : \"{model_ans}\"")
            print(f"Ground Truth Target    : \"{trace.final_answer}\"")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    run_demonstration()
