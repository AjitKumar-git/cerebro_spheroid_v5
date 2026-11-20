"""
Evaluation Suite for Minimal Tool MVP
====================================
Evaluates held-out test traces across:
- Direct-answer accuracy
- Correct tool-choice rate (policy accuracy)
- Tool-call schema validity
- Tool execution success
- Accuracy after receiving tool results
- Unnecessary-call and missed-call rates

Compares:
1. Direct-answer baseline (no tools)
2. Tool-enabled training WITHOUT SCMER
3. Tool-enabled training WITH SCMER
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from experiments.minimal_tool_mvp.dataset_generator import ToolTrace
from experiments.minimal_tool_mvp.model import MinimalToolModel
from experiments.minimal_tool_mvp.schema_validator import (
    DeterministicToolExecutor,
    validate_tool_call,
)
from experiments.minimal_tool_mvp.tokenizer import MinimalTokenizer


@dataclass
class EvaluationReport:
    system_name: str
    total_samples: int
    correct_tool_choice_rate: float
    unnecessary_call_rate: float
    missed_call_rate: float
    tool_call_schema_validity: float
    tool_execution_success: float
    accuracy_after_tool_results: float
    direct_answer_accuracy: float
    overall_accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_name": self.system_name,
            "total_samples": self.total_samples,
            "correct_tool_choice_rate": round(self.correct_tool_choice_rate * 100, 2),
            "unnecessary_call_rate": round(self.unnecessary_call_rate * 100, 2),
            "missed_call_rate": round(self.missed_call_rate * 100, 2),
            "tool_call_schema_validity": round(self.tool_call_schema_validity * 100, 2),
            "tool_execution_success": round(self.tool_execution_success * 100, 2),
            "accuracy_after_tool_results": round(self.accuracy_after_tool_results * 100, 2),
            "direct_answer_accuracy": round(self.direct_answer_accuracy * 100, 2),
            "overall_accuracy": round(self.overall_accuracy * 100, 2),
        }


def extract_content_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """Extracts content between start_tag and end_tag (or till end of string if end_tag missing)."""
    if start_tag in text:
        start_idx = text.find(start_tag) + len(start_tag)
        if end_tag in text:
            end_idx = text.find(end_tag, start_idx)
            return text[start_idx:end_idx].strip()
        return text[start_idx:].strip()
    return None


def clean_text_for_comparison(text: str) -> str:
    """Normalizes answers for robust exact-match comparison."""
    return re.sub(r"[^\w\s\.-]", "", text.lower()).strip()


def evaluate_system(
    system_name: str,
    model: MinimalToolModel,
    test_traces: List[ToolTrace],
    tokenizer: MinimalTokenizer,
    force_direct_baseline: bool = False,
    device: Optional[str] = None,
) -> EvaluationReport:
    """
    Evaluates a model over test traces, tracking all specified metrics.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    executor = DeterministicToolExecutor()

    total_samples = len(test_traces)
    policy_correct = 0
    unnecessary_calls = 0
    missed_calls = 0

    tool_call_attempts = 0
    tool_call_valid_schemas = 0
    tool_exec_successes = 0

    c2_count = 0
    c2_correct = 0

    direct_count = 0
    direct_correct = 0

    total_correct = 0

    ans_end_id = tokenizer.token_to_id["<|answer_end|>"]
    tool_end_id = tokenizer.token_to_id["<|tool_call_end|>"]

    for idx, trace in enumerate(test_traces):
        if (idx + 1) % 100 == 0 or idx == total_samples - 1:
            print(f"  [{system_name}] Evaluated {idx + 1}/{total_samples} traces...", flush=True)

        # Construct Turn 1 prefix
        prefix_str = f"<|user|>{trace.user_request}<|user_end|>"
        prefix_ids = tokenizer.encode(prefix_str)
        policy_pos = len(prefix_ids) - 1

        inp_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)

        if force_direct_baseline:
            pred_policy = 0
            policy_prob = 1.0
        else:
            pred_policy, policy_prob = model.predict_policy(inp_tensor, policy_pos)

        # Track policy choice
        gt_policy = trace.policy_target
        if pred_policy == gt_policy:
            policy_correct += 1

        if gt_policy == 0 and pred_policy == 1:
            unnecessary_calls += 1
        elif gt_policy == 1 and pred_policy == 0:
            missed_calls += 1

        # Action execution based on predicted policy
        generated_answer = ""

        if pred_policy == 0:
            # Direct answer path
            direct_count += 1
            ans_prompt = f"{prefix_str}<|answer|>"
            ans_prompt_ids = torch.tensor([tokenizer.encode(ans_prompt)], dtype=torch.long, device=device)
            gen_ids = model.generate_autoregressive(
                ans_prompt_ids, stop_token_id=ans_end_id, max_new_tokens=64, temperature=0.0
            )
            decoded = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=False)
            extracted_ans = extract_content_between(decoded, "<|answer|>", "<|answer_end|>")
            generated_answer = extracted_ans if extracted_ans is not None else decoded

            # Check correctness against ground truth final answer
            is_match = (
                clean_text_for_comparison(generated_answer) == clean_text_for_comparison(trace.final_answer)
                or trace.final_answer.strip().lower() in generated_answer.strip().lower()
            )
            if is_match:
                direct_correct += 1
                total_correct += 1

        else:
            # Tool call path (Turn 1: tool call generation)
            tool_call_attempts += 1
            tc_prompt = f"{prefix_str}<|tool_call|>"
            tc_prompt_ids = torch.tensor([tokenizer.encode(tc_prompt)], dtype=torch.long, device=device)
            gen_tc_ids = model.generate_autoregressive(
                tc_prompt_ids, stop_token_id=tool_end_id, max_new_tokens=96, temperature=0.0
            )
            decoded_tc = tokenizer.decode(gen_tc_ids[0].tolist(), skip_special_tokens=False)
            extracted_tc = extract_content_between(decoded_tc, "<|tool_call|>", "<|tool_call_end|>")

            is_valid_schema = False
            parsed_call = None
            exec_res = None

            if extracted_tc:
                is_valid, err, parsed = validate_tool_call(extracted_tc)
                if is_valid:
                    is_valid_schema = True
                    tool_call_valid_schemas += 1
                    parsed_call = parsed

            # Execute tool call
            if is_valid_schema and parsed_call is not None:
                exec_res = executor.execute(parsed_call)
                if exec_res.get("status") == "ok":
                    tool_exec_successes += 1
            else:
                exec_res = {"status": "error", "message": "Invalid or unparseable tool call"}

            # Turn 2: Sensory re-afference (feed tool result into context)
            c2_count += 1
            res_json = json.dumps(exec_res, separators=(",", ":"))
            t2_prompt = (
                f"{prefix_str}<|tool_call|>{extracted_tc or ''}<|tool_call_end|>"
                f"<|tool_result|>{res_json}<|tool_result_end|><|answer|>"
            )
            t2_prompt_ids = torch.tensor([tokenizer.encode(t2_prompt)], dtype=torch.long, device=device)
            gen_t2_ids = model.generate_autoregressive(
                t2_prompt_ids, stop_token_id=ans_end_id, max_new_tokens=64, temperature=0.0
            )
            decoded_t2 = tokenizer.decode(gen_t2_ids[0].tolist(), skip_special_tokens=False)
            extracted_ans = extract_content_between(decoded_t2, "<|answer|>", "<|answer_end|>")
            generated_answer = extracted_ans if extracted_ans is not None else decoded_t2

            is_match = (
                clean_text_for_comparison(generated_answer) == clean_text_for_comparison(trace.final_answer)
                or trace.final_answer.strip().lower() in generated_answer.strip().lower()
            )
            if is_match:
                c2_correct += 1
                total_correct += 1

    # Compute rates
    total_c1_c3 = sum(1 for t in test_traces if t.policy_target == 0)
    total_c2 = sum(1 for t in test_traces if t.policy_target == 1)

    unnecessary_rate = unnecessary_calls / max(1, total_c1_c3)
    missed_rate = missed_calls / max(1, total_c2)
    choice_rate = policy_correct / max(1, total_samples)

    schema_validity = tool_call_valid_schemas / max(1, tool_call_attempts) if tool_call_attempts > 0 else 0.0
    exec_success = tool_exec_successes / max(1, tool_call_attempts) if tool_call_attempts > 0 else 0.0
    c2_acc = c2_correct / max(1, c2_count) if c2_count > 0 else 0.0
    direct_acc = direct_correct / max(1, direct_count) if direct_count > 0 else 0.0
    overall_acc = total_correct / max(1, total_samples)

    return EvaluationReport(
        system_name=system_name,
        total_samples=total_samples,
        correct_tool_choice_rate=choice_rate,
        unnecessary_call_rate=unnecessary_rate,
        missed_call_rate=missed_rate,
        tool_call_schema_validity=schema_validity,
        tool_execution_success=exec_success,
        accuracy_after_tool_results=c2_acc,
        direct_answer_accuracy=direct_acc,
        overall_accuracy=overall_acc,
    )
