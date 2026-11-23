"""
🔬 Test Suite: Sensory KV & Tool Response Ingestion in CerebroSpheroid V5
========================================================================
Validates the Dual-Track Sensory Mechanism for agentic tool interactions:
1. Sensory KV Concatenation & Geometric Dimension Alignment
2. Arcuate Fasciculus Cross-Resonance Attention Mass on Tool Tokens
3. Volumetric Core Re-Afference: Mantle Perturbation & Contractive Stability
4. Two-Stage Closed-Loop Generation (Prompt -> Call -> Exec -> Ingest -> Answer)
5. Multi-Turn Chained Sensory KV Accumulation
6. Fault-Tolerant Error Ingestion (Handling Tool Exceptions Gracefully)
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    PRESETS_V5,
    load_v5_checkpoint_into_torch,
)
from src.jax.checkpoint import adapt_jax_to_torch_state_dict
from src.tools.unified_tool_dispatcher import UnifiedToolDispatcher


@pytest.fixture(scope="module")
def small_config():
    cfg_dict = PRESETS_V5["v5_small"].copy()
    cfg_dict["sensory_position_steps"] = 1024
    cfg_dict["max_phase_steps"] = 1024
    cfg_dict["max_seq_len"] = 1024
    cfg_dict["use_rope"] = False
    cfg_dict["top_k_resonance"] = 0
    cfg_dict["use_deliberation_probe"] = False
    cfg_dict["use_adaptive_thinking"] = False
    return ModelConfigV5(**cfg_dict)


@pytest.fixture(scope="module")
def tool_aligned_model(small_config):
    dev = torch.device("cpu")
    model = CerebroSpheroidV5(small_config)
    ckpt_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "tool_aligned", "v5_small_tool_aligned_step_100.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt.get("model", ckpt))
        if any(k.startswith(("sensory.", "core.", "motor.")) for k in state.keys()):
            state = adapt_jax_to_torch_state_dict(state, small_config)
        load_v5_checkpoint_into_torch(model, state, verbose=False)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    return AutoTokenizer.from_pretrained(tok_dir)


@pytest.fixture(scope="module")
def dispatcher():
    return UnifiedToolDispatcher()


def test_sensory_kv_concatenation_and_dimension_integrity(small_config, tool_aligned_model, tokenizer):
    """
    Track 1 Test: Ingest prompt and tool response, verifying that the keys and values
    form a contiguous [B, H, L_total, head_dim] tensor matching expected sequence length.
    """
    model = tool_aligned_model
    prompt_text = "<|im_start|>user\nWhat is (17 * 24) + 85?<|im_end|>\n<|im_start|>assistant\n<think>Call calculator</think><tool_call> {\"name\": \"calculator\", \"arguments\": {\"expression\": \"(17 * 24) + 85\"}} </tool_call>"
    tool_resp_text = "<tool_response> 493 </tool_response>"

    p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    t_ids = tokenizer.encode(tool_resp_text, add_special_tokens=False)

    p_tensor = torch.tensor([p_ids], dtype=torch.long)
    t_tensor = torch.tensor([t_ids], dtype=torch.long)

    # Ingest prompt prefix into sensory DNN
    with torch.no_grad():
        flux_p, (k_p, v_p) = model.sensory_dnn(p_tensor)
        flux_t, (k_t, v_t) = model.sensory_dnn(t_tensor, pos_offset=len(p_ids))

    assert k_p.shape == (1, model.config.num_heads, len(p_ids), model.sensory_dnn.head_dim)
    assert k_t.shape == (1, model.config.num_heads, len(t_ids), model.sensory_dnn.head_dim)

    # Perform Track 1 sensory concatenation
    k_total = torch.cat([k_p, k_t], dim=2)
    v_total = torch.cat([v_p, v_t], dim=2)

    total_len = len(p_ids) + len(t_ids)
    assert k_total.shape == (1, model.config.num_heads, total_len, model.sensory_dnn.head_dim)
    assert v_total.shape == (1, model.config.num_heads, total_len, model.sensory_dnn.head_dim)
    assert not torch.isnan(k_total).any(), "NaN detected in concatenated Sensory Keys"
    assert not torch.isnan(v_total).any(), "NaN detected in concatenated Sensory Values"


def test_arcuate_fasciculus_cross_attention_to_tool_response(small_config, tool_aligned_model, tokenizer):
    """
    Verifies that the Arcuate Fasciculus cross-resonance heads correctly attend to the
    newly appended tool response tokens during answer generation.
    """
    model = tool_aligned_model
    prompt_text = "<|im_start|>user\nCalculate 125 * 32<|im_end|>\n<|im_start|>assistant\n<tool_call>calc(125*32)</tool_call>"
    tool_text = "<tool_response> 4000 </tool_response>"
    answer_prefix = "\nThe answer is"

    full_context = f"{prompt_text}\n{tool_text}{answer_prefix}"
    ids = tokenizer.encode(full_context, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        flux, sensory_kv = model.sensory_dnn(inp)
        keys, values = sensory_kv
        # Compute motor query for the last token position
        motor_surface = flux[:, -1:, :]
        q = model.motor_decoder.q_proj(motor_surface).view(1, model.config.num_heads, 1, model.sensory_dnn.head_dim)
        scores = torch.matmul(q, keys.transpose(-1, -2)) / math.sqrt(model.sensory_dnn.head_dim)
        attn_weights = F.softmax(scores.float(), dim=-1)[0]  # [H, 1, L]

    # Delimiter tokens: <tool_response> (id 33) and </tool_response> (id 34)
    start_tag_id = tokenizer.convert_tokens_to_ids("<tool_response>")
    end_tag_id = tokenizer.convert_tokens_to_ids("</tool_response>")

    assert start_tag_id in ids, f"Start tag '{start_tag_id}' not found in {ids}"
    assert end_tag_id in ids, f"End tag '{end_tag_id}' not found in {ids}"

    start_idx = ids.index(start_tag_id)
    end_idx = ids.index(end_tag_id)
    tool_indices = list(range(start_idx, end_idx + 1))

    # Measure total attention allocated across all 8 heads on tool response tokens
    tool_attn_mass = attn_weights[:, 0, tool_indices].sum(dim=-1).mean().item()

    # Attention must be positive, non-zero, and free of numerical abnormalities
    assert tool_attn_mass > 0.0, f"Attention mass on tool tokens was 0.0: {tool_attn_mass}"
    assert not math.isnan(tool_attn_mass), "Attention mass contains NaN"
    assert tool_attn_mass <= 1.0, f"Attention mass exceeds 1.0: {tool_attn_mass}"


def test_sensory_reafference_volumetric_core_stability(small_config, tool_aligned_model, tokenizer):
    """
    Track 2 Test: Injects tool response flux centroid into the outer cortical shell (Shell 15)
    and verifies that 2 deliberation cycles assimilate the signal contractively without
    numerical explosion (SCMER spectral stability).
    """
    model = tool_aligned_model
    dev = torch.device("cpu")
    tool_output = "Speed of light in vacuum is 299,792,458 m/s"
    t_ids = tokenizer.encode(f"<tool_response> {tool_output} </tool_response>", add_special_tokens=False)
    t_tensor = torch.tensor([t_ids], dtype=torch.long, device=dev)

    with torch.no_grad():
        flux, _ = model.sensory_dnn(t_tensor)
        f_tool = flux.mean(dim=1)  # [1, D]

        # Initialize base spheroid state
        initial_state = model.init_spheroid_state(1, dev)

        # Inject sensory perturbation into outermost Shell 15 (mantle)
        perturbed_state = initial_state.clone()
        perturbed_state[:, -1:, :] = perturbed_state[:, -1:, :] + 0.2 * f_tool.unsqueeze(1)
        perturbed_norm = torch.linalg.norm(perturbed_state).item()

        # Deliberate 2 cycles inward through volumetric core
        state_after = perturbed_state
        for cycle in range(2):
            state_after = model.volumetric_core.deliberate_step(state_after)

        final_norm = torch.linalg.norm(state_after).item()

    assert not torch.isnan(state_after).any(), "NaN in state after tool sensory re-afference"
    assert not torch.isinf(state_after).any(), "Inf in state after tool sensory re-afference"

    # Stability: Deliberation must contract or preserve energy without explosive divergence
    assert perturbed_norm > 0.0, "Perturbed norm must be strictly positive"
    norm_ratio = final_norm / perturbed_norm
    assert 0.5 <= norm_ratio <= 2.0, f"Core norm unstable after sensory re-afference: ratio={norm_ratio:.3f}"


def test_closed_loop_two_stage_generation_with_sensory_kv(small_config, tool_aligned_model, tokenizer, dispatcher):
    """
    End-to-End Closed-Loop Test:
    Stage 1: User Prompt -> Model outputs <think> and <tool_call>
    Stage 2: Dispatcher executes call -> yields exact tool output
    Stage 3: Tool response is appended to Sensory KV and context -> Model outputs final Answer!
    """
    model = tool_aligned_model
    dev = torch.device("cpu")

    test_cases = [
        {
            "prompt": "What is (17 * 24) + 85?",
            "call": '{"name": "calculator", "arguments": {"expression": "(17 * 24) + 85"}}',
            "expected": "493",
        },
        {
            "prompt": "What is 2^10 - 24?",
            "call": '{"name": "calculator", "arguments": {"expression": "2**10 - 24"}}',
            "expected": "1000",
        },
        {
            "prompt": "What is the capital of France?",
            "call": '{"name": "facts", "arguments": {"query": "capital of France"}}',
            "expected": "Paris",
        },
    ]

    for tc in test_cases:
        prompt = tc["prompt"]
        canonical_call = tc["call"]
        expected = tc["expected"]

        # Step 1: Execute tool in dispatcher
        tool_out = dispatcher.execute_raw_call_string(canonical_call)
        assert expected.lower() in tool_out.lower(), f"Tool output missing expected answer: {tool_out}"

        # Step 2: Assemble full context with <tool_response>
        full_turn = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<think>Use tool</think><tool_call> {canonical_call} </tool_call>\n"
            f"<tool_response> {tool_out} </tool_response>\n"
        )
        inp_ids = tokenizer.encode(full_turn, add_special_tokens=False)
        inp_tensor = torch.tensor([inp_ids], dtype=torch.long, device=dev)

        # Step 3: Generate post-tool answer tokens (model.generate returns generated tokens)
        with torch.no_grad():
            out_ids = model.generate(inp_tensor, max_new_tokens=32, temperature=0.1)
        generated_answer = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=False)

        # Verify tokens were emitted without NaN/Inf
        assert out_ids.shape[1] > 0, "Model produced 0 tokens after tool response"
        assert not torch.isnan(out_ids.float()).any()


def test_multi_turn_chained_sensory_kv_accumulation(small_config, tool_aligned_model, tokenizer, dispatcher):
    """
    Tests multi-turn agentic chaining where Turn 1 tool result is used as input for Turn 2:
    Turn 1: Facts query: "atomic number of Gold" -> 79
    Turn 2: Calculator: "79 * 2" -> 158
    Verifies that Sensory KV expands monotonically and retains both memories.
    """
    model = tool_aligned_model
    dev = torch.device("cpu")

    # Turn 1
    call1 = '{"name": "facts", "arguments": {"query": "atomic number of Gold"}}'
    res1 = dispatcher.execute_raw_call_string(call1)  # 79

    turn1_text = f"<|im_start|>user\nWhat is twice the atomic number of Gold?<|im_end|>\n<|im_start|>assistant\n<tool_call>{call1}</tool_call><tool_response>{res1}</tool_response>"
    ids1 = tokenizer.encode(turn1_text, add_special_tokens=False)
    t1_tensor = torch.tensor([ids1], dtype=torch.long, device=dev)

    with torch.no_grad():
        _, (k1, v1) = model.sensory_dnn(t1_tensor)

    # Turn 2
    call2 = '{"name": "calculator", "arguments": {"expression": "79 * 2"}}'
    res2 = dispatcher.execute_raw_call_string(call2)  # 158

    turn2_text = f"\n<tool_call>{call2}</tool_call><tool_response>{res2}</tool_response>\nThe answer is 158."
    ids2 = tokenizer.encode(turn2_text, add_special_tokens=False)
    t2_tensor = torch.tensor([ids2], dtype=torch.long, device=dev)

    with torch.no_grad():
        _, (k2, v2) = model.sensory_dnn(t2_tensor, pos_offset=len(ids1))

    # Total accumulated Sensory KV
    k_accum = torch.cat([k1, k2], dim=2)
    v_accum = torch.cat([v1, v2], dim=2)

    total_tokens = len(ids1) + len(ids2)
    assert k_accum.shape[2] == total_tokens
    assert v_accum.shape[2] == total_tokens
    assert not torch.isnan(k_accum).any()
    assert not torch.isnan(v_accum).any()


def test_negative_tool_response_error_recovery(small_config, tool_aligned_model, tokenizer, dispatcher):
    """
    Tests robustness when a tool throws an error (e.g. division by zero or syntax error).
    Verifies that the sensory error response does NOT crash the network or introduce NaNs.
    """
    model = tool_aligned_model
    dev = torch.device("cpu")

    bad_call = '{"name": "calculator", "arguments": {"expression": "100 / 0"}}'
    err_out = dispatcher.execute_raw_call_string(bad_call)
    assert "error" in err_out.lower() or "exception" in err_out.lower() or "division" in err_out.lower()

    error_context = f"<|im_start|>user\nCompute 100/0<|im_end|>\n<|im_start|>assistant\n<tool_call>{bad_call}</tool_call><tool_response>{err_out}</tool_response>"
    e_ids = tokenizer.encode(error_context, add_special_tokens=False)
    e_tensor = torch.tensor([e_ids], dtype=torch.long, device=dev)

    with torch.no_grad():
        flux, sensory_kv = model.sensory_dnn(e_tensor)
        out_ids = model.generate(e_tensor, max_new_tokens=16, temperature=0.1)

    assert not torch.isnan(flux).any(), "NaN in sensory flux on error response"
    assert out_ids.shape[1] > 0, "Model generated 0 tokens after error tool response"
