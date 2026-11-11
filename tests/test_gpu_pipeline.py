"""Tests for GPU training pipeline, JAX<->PyTorch checkpoint parity, optimizer exclusions, and Colab notebook validity."""

import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from src.spheroid_v5 import (
    CerebroSpheroidV5,
    ModelConfigV5,
    PRESETS_V5,
    infer_v5_config_from_state_dict,
    load_v5_checkpoint_into_torch,
)
from src.spheroid_optimizer import create_spheroid_param_groups, build_spheroid_optimizer


def test_preset_v5_small_matches_jax_tied_embeddings():
    cfg = ModelConfigV5(**PRESETS_V5["v5_small"])
    assert cfg.use_tied_embeddings is True
    assert cfg.surface_dim == 640
    assert cfg.radial_dim == 256
    assert cfg.num_shells == 16


def test_jax_to_pytorch_checkpoint_adaptation():
    from src.jax.v5_model import ModelConfigV5 as JAXConfig, init_params_v5
    import jax
    import numpy as np

    jcfg = JAXConfig(
        vocab_size=256,
        surface_dim=64,
        radial_dim=32,
        num_shells=4,
        num_shortcut_tracts=6,
        num_heads=4,
        max_phase_steps=64,
        thinking_cycles=2,
        top_k_tracts=4,
    )
    jparams, _ = init_params_v5(jax.random.PRNGKey(0), jcfg)
    raw_state = {k: torch.from_numpy(np.array(v)) for k, v in jparams.items()}
    # Simulate legacy checkpoint missing radial_up_out
    del raw_state["core.radial_up_out.weight"]

    ckpt_payload = {"model_state": raw_state, "step": 42, "loss": 5.5, "preset": "tpu_v5e_1"}
    inferred_cfg = infer_v5_config_from_state_dict(ckpt_payload)
    assert inferred_cfg.vocab_size == 256
    assert inferred_cfg.surface_dim == 64
    assert inferred_cfg.radial_dim == 32
    assert inferred_cfg.num_shells == 4
    assert inferred_cfg.num_shortcut_tracts == 6
    assert inferred_cfg.use_tied_embeddings is True

    model = CerebroSpheroidV5(inferred_cfg)
    loaded, total = load_v5_checkpoint_into_torch(model, ckpt_payload, verbose=False)
    assert loaded >= 40 and total > loaded

    # Verify 3D stacked intra_swiglu transpose & intra_norms split & radial_up_out fallback
    assert torch.allclose(
        model.volumetric_core.intra_shell_swiglu[1].fc_gate.weight,
        raw_state["core.intra_swiglu.w_gate"][1].T,
        atol=1e-6,
    )
    assert torch.allclose(
        model.volumetric_core.intra_shell_swiglu[2].fc_down.weight,
        raw_state["core.intra_swiglu.w_down"][2].T,
        atol=1e-6,
    )
    assert torch.allclose(
        model.volumetric_core.intra_norms[0].weight,
        raw_state["core.intra_norms.weight"][0],
        atol=1e-6,
    )
    assert torch.allclose(
        model.volumetric_core.radial_up_out.weight,
        raw_state["core.radial_up.weight"],
        atol=1e-6,
    )

    prompts = torch.randint(0, 256, (2, 8))
    targets = torch.randint(0, 256, (2, 6))
    loss, logits = model(prompts, targets=targets)
    assert torch.isfinite(loss)
    assert torch.isfinite(logits).all()


def test_real_step_1000_checkpoint_loads_into_pytorch_if_present():
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "v5_tpu_step_1000.pt")
    if not os.path.exists(ckpt_path):
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = infer_v5_config_from_state_dict(ckpt)
    assert cfg.surface_dim == 640 and cfg.num_shells == 16 and cfg.use_tied_embeddings is True
    model = CerebroSpheroidV5(cfg)
    loaded, _ = load_v5_checkpoint_into_torch(model, ckpt, verbose=False)
    assert loaded >= 100
    prompts = torch.tensor([[1, 25, 300, 42]], dtype=torch.long)
    targets = torch.tensor([[25, 300, 42, -100]], dtype=torch.long)
    with torch.no_grad():
        loss, logits = model(prompts, targets=targets)
    assert torch.isfinite(loss) and torch.isfinite(logits).all()


def test_untied_legacy_checkpoint_detection():
    cfg_untied = ModelConfigV5(
        vocab_size=128,
        surface_dim=64,
        radial_dim=32,
        num_shells=4,
        num_shortcut_tracts=4,
        num_heads=4,
        max_phase_steps=32,
        thinking_cycles=2,
        top_k_tracts=2,
        use_tied_embeddings=False,
    )
    src_model = CerebroSpheroidV5(cfg_untied)
    ckpt = {"model_state": src_model.state_dict(), "step": 10}
    inferred = infer_v5_config_from_state_dict(ckpt)
    assert inferred.use_tied_embeddings is False
    dst_model = CerebroSpheroidV5(inferred)
    loaded, total = load_v5_checkpoint_into_torch(dst_model, ckpt, verbose=False)
    assert loaded == total


def test_spheroid_optimizer_exclusions_and_half_precision():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    model = CerebroSpheroidV5(cfg)
    # Add a 2D stacked norm parameter to ensure it still routes to scalar_params (not orthogonalized)
    model.volumetric_core.stacked_norm_test = nn.Parameter(torch.ones(cfg.num_shells, cfg.surface_dim))

    groups = create_spheroid_param_groups(model, lr=1e-3)
    by_name = {g["name"]: g for g in groups}

    scalar_ids = {id(p) for p in by_name["scalar_gates"]["params"]}
    vocab_ids = {id(p) for p in by_name["embedding_head"]["params"]}
    matrix_ids = {id(p) for p in by_name.get("manifold_matrices", {}).get("params", [])}
    recurrent_ids = {id(p) for p in by_name.get("recurrent_manifold", {}).get("params", [])}

    assert id(model.volumetric_core.stacked_norm_test) in scalar_ids
    assert id(model.volumetric_core.stacked_norm_test) not in matrix_ids
    assert id(model.sensory_dnn.pos_embed.weight) in vocab_ids
    assert id(model.motor_decoder.phase_embed.weight) in vocab_ids
    assert id(model.motor_decoder.persona_embeddings) in vocab_ids
    assert id(model.sensory_dnn.pos_embed.weight) not in matrix_ids
    assert id(model.motor_decoder.phase_embed.weight) not in matrix_ids
    assert id(model.volumetric_core.w_radial_in) in recurrent_ids

    # Test bfloat16 optimizer step avoids ULP freeze & long prompt (> top_k_resonance) avoids -1e9 overflow
    bf_model = CerebroSpheroidV5(ModelConfigV5(**PRESETS_V5["v5_test"])).to(torch.bfloat16)
    opt = build_spheroid_optimizer(bf_model, lr=5e-4)
    init_emb = bf_model.sensory_dnn.embedding.weight.detach().clone()
    prompts = torch.randint(0, cfg.vocab_size, (2, 20))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    loss, _ = bf_model(prompts, targets=targets)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    for p in bf_model.parameters():
        assert torch.isfinite(p).all()
    assert not torch.equal(bf_model.sensory_dnn.embedding.weight.detach(), init_emb)

    # Test float16 top-k masking does not overflow half precision
    fp16_model = CerebroSpheroidV5(ModelConfigV5(**PRESETS_V5["v5_test"])).to(torch.float16)
    with torch.no_grad():
        loss16, logits16 = fp16_model(prompts, targets=targets)
        assert torch.isfinite(loss16) and torch.isfinite(logits16).all()


def test_colab_gpu_notebook_has_valid_cli_args():
    nb_path = os.path.join(os.path.dirname(__file__), "..", "colab", "CerebroSpheroid_V5_Colab_Training.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb["metadata"]["accelerator"] == "GPU"
    full_src = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert 'MODEL_PRESET = "v5_small"' in full_src
    assert "{MODEL_PRESET}" in full_src
    assert "{BATCH_SIZE}" in full_src
    assert "{GRAD_ACCUM}" in full_src
    assert "{TOTAL_WINDOW}" in full_src
    assert "--checkpoint {ckpt_loc}" in full_src

    # Verify no blank CLI arguments (e.g. '--model-preset  ') exist
    blank_flags = re.findall(r"--(?:model-preset|total-steps|batch-size|grad-accum|total-window|checkpoint)\s+(?:--|$|\n)", full_src)
    assert not blank_flags, f"Found blank CLI flags in GPU notebook: {blank_flags}"


def test_jax_gpu_dtype_resolution(monkeypatch):
    import jax.numpy as jnp
    from src.jax import auto_tuner

    monkeypatch.setattr(auto_tuner, "get_hardware_info", lambda: {"is_tpu": True, "is_gpu": False, "device_kind": "TPU v5 lite"})
    dt, _ = auto_tuner.resolve_jax_compute_dtype("auto")
    assert dt == jnp.bfloat16

    monkeypatch.setattr(auto_tuner, "get_hardware_info", lambda: {"is_tpu": False, "is_gpu": True, "device_kind": "NVIDIA L4"})
    dt, _ = auto_tuner.resolve_jax_compute_dtype("auto")
    assert dt == jnp.bfloat16

    monkeypatch.setattr(auto_tuner, "get_hardware_info", lambda: {"is_tpu": False, "is_gpu": True, "device_kind": "Tesla T4"})
    dt, _ = auto_tuner.resolve_jax_compute_dtype("auto")
    assert dt == jnp.float32
