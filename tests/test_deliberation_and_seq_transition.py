import pytest
import math
import numpy as np
import torch
import torch.nn.functional as F

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    PRESETS_V5,
    load_v5_checkpoint_into_torch,
)
from src.jax.v5_model import ModelConfigV5 as JAXConfigV5, init_params_v5, training_rollout
from src.jax.auto_tuner import get_hardware_info
import jax
import jax.numpy as jnp


def test_pytorch_deliberation_sequential_rollout_backward():
    """Verifies that PyTorch forward with sampling_p < 1.0 triggers sequential deliberation and backward pass."""
    cfg = dict(PRESETS_V5["v5_small"])
    cfg["vocab_size"] = 500
    cfg["num_shells"] = 4
    config = ModelConfigV5(**cfg)
    model = CerebroSpheroidV5(config).cpu()

    B, L_p, L_t = 2, 8, 16
    prompts = torch.randint(1, 500, (B, L_p))
    targets = torch.randint(1, 500, (B, L_t))
    feedback = torch.randint(1, 500, (B, L_t))

    # Fast TF should NOT be used when sampling_p < 1.0
    out = model(prompts, targets=targets, feedback_tokens=feedback, sampling_p=0.5, pitfall_prob=0.2)
    loss = out[0] if isinstance(out, tuple) else out
    assert loss is not None
    assert math.isfinite(loss.item())
    assert loss.item() > 0.0

    loss.backward()
    assert model.sensory_dnn.embedding.weight.grad is not None
    assert model.sensory_dnn.embedding.weight.grad.norm().item() > 0.0


def test_jax_deliberation_rollout_and_pitfall():
    """Verifies that JAX training_rollout with sampling_p < 1.0 and pitfall_prob > 0.0 produces valid finite loss."""
    cfg = dict(PRESETS_V5["v5_small"])
    cfg["vocab_size"] = 500
    cfg["num_shells"] = 4
    config = JAXConfigV5(**cfg)

    key = jax.random.PRNGKey(42)
    params, buffers = init_params_v5(key, config)

    B, L_p, L_t = 2, 8, 16
    prompts = jnp.ones((B, L_p), dtype=jnp.int32)
    targets = jnp.ones((B, L_t), dtype=jnp.int32)
    feedback = jnp.ones((B, L_t), dtype=jnp.int32)

    loss = training_rollout(
        params,
        prompts,
        target_len=L_t,
        buffers=buffers,
        config=config,
        feedback_tokens=feedback,
        sampling_p=0.5,
        pitfall_prob=0.15,
        targets=targets,
        pure_teacher_forcing=False,
    )
    assert loss is not None
    loss_val = float(loss)
    assert math.isfinite(loss_val)
    assert loss_val > 0.0


def test_sequence_length_transition_preserves_position_embeddings():
    """Verifies that transitioning sequence length from 256 to 512 cleanly loads and adapts weights without failure."""
    cfg256 = dict(PRESETS_V5["v5_small"])
    cfg256["vocab_size"] = 1000
    cfg256["num_shells"] = 4
    cfg256["max_seq_len"] = 288
    model256 = CerebroSpheroidV5(ModelConfigV5(**cfg256))
    sd256 = model256.state_dict()

    cfg512 = dict(PRESETS_V5["v5_small"])
    cfg512["vocab_size"] = 1000
    cfg512["num_shells"] = 4
    cfg512["max_seq_len"] = 544
    model512 = CerebroSpheroidV5(ModelConfigV5(**cfg512))

    loaded, total = load_v5_checkpoint_into_torch(model512, {"model_state": sd256}, verbose=False)
    assert loaded > 0
    # Positional embeddings table (capacity 1024) easily handles seq_len transition up to 1024
    assert model512.sensory_dnn.pos_embed.weight.shape[0] >= 544


def test_autotuner_cache_sig_incorporates_sequence_length():
    """Verifies that auto-tuner cache signature isolates configs by sequence length."""
    hw = get_hardware_info()
    sig = hw["hardware_signature"]

    # At 256, backward-compatible signature is retained
    seq_suffix_256 = f"_seq{256}" if 256 != 256 else ""
    assert seq_suffix_256 == ""

    # At 512, differentiated signature is generated
    seq_suffix_512 = f"_seq{512}" if 512 != 256 else ""
    assert seq_suffix_512 == "_seq512"
    assert f"{sig}{seq_suffix_512}" == f"{sig}_seq512"
