"""
CerebroSpheroid V5: Multimodal Parity Test Suite
=================================================
Verifies 1:1 mathematical and architectural parity between PyTorch and JAX:
1. ImagePatchStem (2D Conv patchify) numeric parity.
2. AudioWaveformStem (1D Conv cochlear mapping) numeric parity.
3. SwiGLU Perceptual Projector numeric parity.
4. Bidirectional weight adaptation (PyTorch state_dict <-> JAX params).
5. Multimodal sensory ingestion & forward execution in both engines.
6. Multimodal generation with strict <act> ... </act> boundaries.
"""

import math
import numpy as np
import pytest
import torch

import jax
import jax.numpy as jnp

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5 as TorchModelConfigV5,
    PRESETS_V5 as TORCH_PRESETS_V5,
    ImagePatchStem as TorchImagePatchStem,
    AudioWaveformStem as TorchAudioWaveformStem,
)
from src.jax.v5_model import (
    ModelConfigV5 as JaxModelConfigV5,
    PRESETS_V5 as JAX_PRESETS_V5,
    init_params_v5,
    encode_image_stem,
    encode_audio_stem,
    project_perceptual,
    training_rollout,
    generate_text_jax,
)
from src.jax.checkpoint import (
    adapt_jax_to_torch_state_dict,
    adapt_and_load_checkpoint_weights,
)
from src.corpus import build_tokenizer


def test_image_patch_stem_numeric_parity():
    """Verifies that PyTorch and JAX ImagePatchStem produce numerically identical outputs."""
    torch.manual_seed(42)
    patch_size = 16
    in_channels = 3
    visual_dim = 128
    B, H, W = 2, 224, 224

    torch_stem = TorchImagePatchStem(patch_size=patch_size, in_channels=in_channels, visual_dim=visual_dim)
    x_torch = torch.randn(B, in_channels, H, W)
    out_torch = torch_stem(x_torch)

    jax_params = {
        "image_stem.conv.weight": jnp.array(torch_stem.conv.weight.detach().numpy()),
        "image_stem.norm.weight": jnp.array(torch_stem.norm.weight.detach().numpy()),
    }
    x_jax = jnp.array(x_torch.numpy())
    out_jax = encode_image_stem(jax_params, x_jax, patch_size=patch_size)

    diff = np.max(np.abs(out_torch.detach().numpy() - np.array(out_jax)))
    assert diff < 1e-5, f"ImagePatchStem discrepancy: {diff:.2e}"


def test_audio_waveform_stem_numeric_parity():
    """Verifies that PyTorch and JAX AudioWaveformStem produce numerically identical outputs."""
    torch.manual_seed(42)
    kernel_size = 400
    stride = 160
    audio_dim = 128
    B, T_samples = 2, 16000

    torch_stem = TorchAudioWaveformStem(kernel_size=kernel_size, stride=stride, audio_dim=audio_dim)
    x_torch = torch.randn(B, T_samples)
    out_torch = torch_stem(x_torch)

    jax_params = {
        "audio_stem.conv.weight": jnp.array(torch_stem.conv.weight.detach().numpy()),
        "audio_stem.norm.weight": jnp.array(torch_stem.norm.weight.detach().numpy()),
    }
    x_jax = jnp.array(x_torch.numpy())
    out_jax = encode_audio_stem(jax_params, x_jax, kernel_size=kernel_size, stride=stride)

    diff = np.max(np.abs(out_torch.detach().numpy() - np.array(out_jax)))
    assert diff < 1e-4, f"AudioWaveformStem discrepancy: {diff:.2e}"


def test_perceptual_projector_numeric_parity():
    """Verifies that PyTorch and JAX Perceptual Projector produce numerically identical outputs."""
    torch.manual_seed(42)
    v_dim = 128
    s_dim = 256
    B, P = 2, 49

    cfg = TorchModelConfigV5(
        surface_dim=s_dim,
        radial_dim=64,
        num_shells=4,
        visual_dim=v_dim,
        enable_multimodal=True,
    )
    model = CerebroSpheroidV5(cfg)
    projector = model.perceptual_projector
    x_torch = torch.randn(B, P, v_dim)
    out_torch = projector(x_torch)

    jax_params = {
        "perceptual.norm_in.weight": jnp.array(projector[0].weight.detach().numpy()),
        "perceptual.linear.weight": jnp.array(projector[1].weight.detach().numpy()),
        "perceptual.swiglu.fc_gate.weight": jnp.array(projector[2].fc_gate.weight.detach().numpy()),
        "perceptual.swiglu.fc_val.weight": jnp.array(projector[2].fc_val.weight.detach().numpy()),
        "perceptual.swiglu.fc_down.weight": jnp.array(projector[2].fc_down.weight.detach().numpy()),
        "perceptual.norm_out.weight": jnp.array(projector[3].weight.detach().numpy()),
    }
    x_jax = jnp.array(x_torch.numpy())
    out_jax = project_perceptual(jax_params, x_jax)

    diff = np.max(np.abs(out_torch.detach().numpy() - np.array(out_jax)))
    assert diff < 1e-5, f"Perceptual Projector discrepancy: {diff:.2e}"


def test_bidirectional_checkpoint_adaptation():
    """Verifies that multimodal stem and projector weights adapt losslessly between PyTorch and JAX."""
    cfg_torch = TorchModelConfigV5(
        surface_dim=128,
        radial_dim=64,
        num_shells=4,
        num_shortcut_tracts=4,
        visual_dim=128,
        enable_multimodal=True,
    )
    torch_model = CerebroSpheroidV5(cfg_torch)
    state_dict = torch_model.state_dict()

    cfg_jax = JaxModelConfigV5(
        surface_dim=128,
        radial_dim=64,
        num_shells=4,
        num_shortcut_tracts=4,
        visual_dim=128,
        enable_multimodal=True,
    )
    key = jax.random.PRNGKey(42)
    jax_params, buffers = init_params_v5(key, cfg_jax)

    # 1. Load PyTorch state dict into JAX parameters
    num_loaded = adapt_and_load_checkpoint_weights(
        state_dict,
        jax_params,
        target_dtype=jnp.float32,
        config=cfg_jax,
    )
    assert num_loaded > 0

    # Verify multimodal keys are loaded into JAX
    multimodal_jax_keys = [
        "image_stem.conv.weight",
        "image_stem.norm.weight",
        "audio_stem.conv.weight",
        "audio_stem.norm.weight",
        "perceptual.norm_in.weight",
        "perceptual.linear.weight",
        "perceptual.swiglu.fc_gate.weight",
        "perceptual.swiglu.fc_val.weight",
        "perceptual.swiglu.fc_down.weight",
        "perceptual.norm_out.weight",
    ]
    for k in multimodal_jax_keys:
        assert k in jax_params, f"Missing {k} in JAX params"

    # 2. Convert JAX parameters back to PyTorch state dict
    adapted_torch_state = adapt_jax_to_torch_state_dict(jax_params, config=cfg_jax)
    multimodal_torch_keys = [
        "image_stem.conv.weight",
        "image_stem.norm.weight",
        "audio_stem.conv.weight",
        "audio_stem.norm.weight",
        "perceptual_projector.0.weight",
        "perceptual_projector.1.weight",
        "perceptual_projector.2.fc_gate.weight",
        "perceptual_projector.2.fc_val.weight",
        "perceptual_projector.2.fc_down.weight",
        "perceptual_projector.3.weight",
    ]
    for k in multimodal_torch_keys:
        assert k in adapted_torch_state, f"Missing {k} in adapted PyTorch state"
        diff = np.max(np.abs(state_dict[k].numpy() - adapted_torch_state[k].numpy()))
        assert diff < 1e-6, f"Weight corruption on {k}: diff={diff}"


def test_multimodal_forward_execution():
    """Verifies that multimodal forward pass runs cleanly with images, audio, and text in PyTorch and JAX."""
    cfg_torch = TorchModelConfigV5(
        vocab_size=1024,
        surface_dim=128,
        radial_dim=64,
        num_shells=4,
        num_shortcut_tracts=4,
        visual_dim=128,
        enable_multimodal=True,
    )
    torch_model = CerebroSpheroidV5(cfg_torch)

    cfg_jax = JaxModelConfigV5(
        vocab_size=1024,
        surface_dim=128,
        radial_dim=64,
        num_shells=4,
        num_shortcut_tracts=4,
        visual_dim=128,
        enable_multimodal=True,
    )
    key = jax.random.PRNGKey(42)
    jax_params, buffers = init_params_v5(key, cfg_jax)

    B = 2
    prompts_t = torch.randint(1, 1000, (B, 16))
    targets_t = torch.randint(1, 1000, (B, 16))
    images_t = torch.randn(B, 3, 224, 224)
    audio_t = torch.randn(B, 16000)

    # PyTorch Forward Pass
    loss_torch, logits_torch = torch_model(
        prompts=prompts_t,
        targets=targets_t,
        images=images_t,
        audio=audio_t,
    )
    assert loss_torch.item() > 0
    assert logits_torch.shape == (B, 16, 1024)

    # JAX Rollout Pass
    prompts_j = jnp.array(prompts_t.numpy())
    targets_j = jnp.array(targets_t.numpy())
    images_j = jnp.array(images_t.numpy())
    audio_j = jnp.array(audio_t.numpy())

    loss_jax = training_rollout(
        params=jax_params,
        prompts=prompts_j,
        target_len=16,
        buffers=buffers,
        config=cfg_jax,
        targets=targets_j,
        images=images_j,
        audio=audio_j,
        deliberation_mode="teacher_forcing",
    )
    assert not jnp.isnan(loss_jax)
    assert float(loss_jax) > 0


def test_multimodal_autoregressive_generation():
    """Verifies that multimodal generation works with strict <act> delimiters in PyTorch and JAX."""
    tokenizer = build_tokenizer()

    cfg_torch = TorchModelConfigV5(
        vocab_size=len(tokenizer),
        surface_dim=128,
        radial_dim=64,
        num_shells=4,
        num_shortcut_tracts=4,
        visual_dim=128,
        enable_multimodal=True,
    )
    torch_model = CerebroSpheroidV5(cfg_torch)

    cfg_jax = JaxModelConfigV5(
        vocab_size=len(tokenizer),
        surface_dim=128,
        radial_dim=64,
        num_shells=4,
        num_shortcut_tracts=4,
        visual_dim=128,
        enable_multimodal=True,
    )
    key = jax.random.PRNGKey(42)
    jax_params, buffers = init_params_v5(key, cfg_jax)

    images_t = torch.randn(1, 3, 224, 224)
    audio_t = torch.randn(1, 16000)

    # 1. PyTorch generate_text with multimodal inputs
    resp_torch = torch_model.generate_text(
        prompt="Describe this scene",
        tokenizer=tokenizer,
        max_new_tokens=16,
        strict_act=True,
        images=images_t,
        audio=audio_t,
    )
    assert isinstance(resp_torch, str)

    # 2. JAX generate_text_jax with multimodal inputs
    images_j = jnp.array(images_t.numpy())
    audio_j = jnp.array(audio_t.numpy())

    resp_jax = generate_text_jax(
        params=jax_params,
        buffers=buffers,
        config=cfg_jax,
        prompt="Describe this scene",
        tokenizer=tokenizer,
        max_new_tokens=16,
        strict_act=True,
        images=images_j,
        audio=audio_j,
    )
    assert isinstance(resp_jax, str)
