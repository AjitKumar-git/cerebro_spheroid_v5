import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    PRESETS_V5,
)
from src.jax.v5_model import (
    ModelConfigV5 as JAXConfigV5,
    init_params_v5,
    generate_tri_stream_jax,
)
import jax
import jax.numpy as jnp


class DummyTokenizer:
    """Minimal mock tokenizer for testing generation without external tokenizers."""
    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [10, 20, 30]

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"tok_{t}" for t in token_ids)


def test_pytorch_dynamic_output_kv_append():
    """Verifies that each emitted token dynamically appends its Key and Value to the sensory KV buffer."""
    cfg = dict(PRESETS_V5["v5_small"])
    cfg["vocab_size"] = 256
    cfg["num_shells"] = 4
    cfg["surface_dim"] = 64
    cfg["num_heads"] = 4
    config = ModelConfigV5(**cfg)
    model = CerebroSpheroidV5(config).cpu().eval()

    B, L_prompt = 1, 6
    prompt_ids = torch.randint(3, 250, (B, L_prompt))
    max_new_tokens = 8

    out = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=20,
        eos_token_id=2,
    )

    num_generated = out.shape[1]
    assert num_generated > 0

    # Verify that model.current_causal_kv was populated
    assert hasattr(model, "current_causal_kv")
    causal_k, causal_v = model.current_causal_kv

    # Expected causal buffer length: 1 (silence slot 0) + num_generated
    expected_causal_len = 1 + num_generated
    assert causal_k.shape[2] == expected_causal_len
    assert causal_v.shape[2] == expected_causal_len

    # Verify that model.current_sensory_kv contains [prompt || causal]
    assert hasattr(model, "current_sensory_kv")
    full_k, full_v = model.current_sensory_kv
    expected_total_len = L_prompt + expected_causal_len
    assert full_k.shape[2] == expected_total_len
    assert full_v.shape[2] == expected_total_len

    # Verify that mask has correct dimensions and enables causal attention
    assert hasattr(model, "current_causal_mask")
    c_mask = model.current_causal_mask
    assert c_mask.shape == (B, expected_causal_len)
    assert not c_mask[0, 0].item()  # Slot 0 is silence (masked out)
    assert c_mask[0, 1:].all().item()  # All generated tokens are unmasked


def test_pytorch_phased_thinking_scratchpad_cross_referencing():
    """Verifies that phased generation emits thinking first and appends thoughts into the KV buffer before answering."""
    cfg = dict(PRESETS_V5["v5_small"])
    cfg["vocab_size"] = 256
    cfg["num_shells"] = 4
    cfg["surface_dim"] = 64
    cfg["num_heads"] = 4
    config = ModelConfigV5(**cfg)
    model = CerebroSpheroidV5(config).cpu().eval()

    tok = DummyTokenizer(vocab_size=256)

    # Force temporal decision router bias to trigger thinking and answering
    with torch.no_grad():
        model.motor_decoder.temporal_router.bias.copy_(torch.tensor([2.0, -2.0, 2.0]))

    res = model.generate_tri_stream(
        prompt="Calculate 25 * 4",
        tokenizer=tok,
        max_new_tokens=10,
        temperature=0.5,
        phased=True,
    )

    assert "thinking" in res
    assert "answer" in res
    assert "temporal_decisions" in res
    assert res["temporal_decisions"]["think"] is True
    assert res["temporal_decisions"]["answer"] is True


def test_jax_phased_thinking_scratchpad_cross_referencing():
    """Verifies that JAX tri-stream generation supports phased thinking scratchpad and dynamic KV buffer append."""
    cfg = dict(PRESETS_V5["v5_small"])
    cfg["vocab_size"] = 256
    cfg["num_shells"] = 4
    cfg["surface_dim"] = 64
    cfg["num_heads"] = 4
    config = JAXConfigV5(**cfg)

    key = jax.random.PRNGKey(123)
    params, buffers = init_params_v5(key, config)

    # Force temporal decision bias to trigger thinking and answering in JAX
    params["motor.temporal_router.bias"] = jnp.array([2.0, -2.0, 2.0])

    tok = DummyTokenizer(vocab_size=256)

    res = generate_tri_stream_jax(
        params=params,
        buffers=buffers,
        config=config,
        prompt="Solve math problem",
        tokenizer=tok,
        max_new_tokens=8,
        temperature=0.5,
        phased=True,
    )

    assert "thinking" in res
    assert "answer" in res
    assert "temporal_decisions" in res
    assert res["temporal_decisions"]["think"] is True
    assert res["temporal_decisions"]["answer"] is True
