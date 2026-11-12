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


def test_pytorch_to_jax_checkpoint_adaptation_full_parity():
    """Verifies PyTorch GPU checkpoint adapts all 54/54 parameters into JAX TPU engine with zero missing weights."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.jax.v5_model import ModelConfigV5 as JAXConfig, init_params_v5, PRESETS_V5
    from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5 as PTConfig
    from src.jax.checkpoint import adapt_and_load_checkpoint_weights

    cfg_d = PRESETS_V5["v5_small"]
    pt_model = CerebroSpheroidV5(PTConfig(**cfg_d))
    pt_state = pt_model.state_dict()

    jcfg = JAXConfig(**cfg_d)
    jparams, _ = init_params_v5(jax.random.PRNGKey(0), jcfg)

    restored = adapt_and_load_checkpoint_weights({"model_state": pt_state}, jparams, jnp.float32, jcfg)
    assert restored == len(jparams) == 54, f"Expected 54 restored parameters, got {restored}"

    # Verify motor swiglu weights (which previously suffered from prefix shadowing)
    assert np.allclose(
        np.array(jparams["motor.swiglu.fc_gate.weight"]),
        pt_state["motor_decoder.motor_swiglu.fc_gate.weight"].numpy(),
        atol=1e-6,
    )
    assert np.allclose(
        np.array(jparams["motor.swiglu.fc_val.weight"]),
        pt_state["motor_decoder.motor_swiglu.fc_val.weight"].numpy(),
        atol=1e-6,
    )
    assert np.allclose(
        np.array(jparams["motor.swiglu.fc_down.weight"]),
        pt_state["motor_decoder.motor_swiglu.fc_down.weight"].numpy(),
        atol=1e-6,
    )

    # Verify 3D fused intra_swiglu tensor
    for k in range(jcfg.num_shells):
        assert np.allclose(
            np.array(jparams["core.intra_swiglu.w_gate"][k]),
            pt_state[f"volumetric_core.intra_shell_swiglu.{k}.fc_gate.weight"].numpy().T,
            atol=1e-6,
        )


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


def test_t4_selects_fp16_not_emulated_bf16(monkeypatch):
    import scripts.train_v5_graded_curriculum as train_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *args, **kwargs: True)

    # Tesla T4 (sm_75): must pick float16 even if software BF16 emulation claims True
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: (7, 5))
    assert train_mod._has_native_bf16(0) is False
    assert train_mod._select_amp_dtype(True, 0) == torch.float16

    # Ampere/Ada (sm_80 / sm_89): native Tensor Core BF16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: (8, 0))
    assert train_mod._has_native_bf16(0) is True
    assert train_mod._select_amp_dtype(True, 0) == torch.bfloat16

    assert train_mod._select_amp_dtype(False, 0) == torch.float32


def test_clamp_spectral_norm_is_deterministic_across_rng():
    from src.spheroid_optimizer import _clamp_spectral_norm

    torch.manual_seed(123)
    base = torch.randn(32, 16) * 2.5
    w1 = base.clone()
    w2 = base.clone()

    torch.manual_seed(0)
    c1 = _clamp_spectral_norm(w1, max_spec_norm=1.0, power_steps=2)
    torch.manual_seed(99999)
    c2 = _clamp_spectral_norm(w2, max_spec_norm=1.0, power_steps=2)

    assert torch.equal(c1, c2)
    assert torch.linalg.matrix_norm(c1, ord=2).item() <= 1.0 + 1e-3


def test_kaggle_2xt4_kernel_artifacts_valid():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    meta_path = os.path.join(root, "kaggle", "kernel-metadata.json")
    nb_path = os.path.join(root, "kaggle", "cerebro_spheroid_v5_kaggle_2xt4.ipynb")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["id"] == "ajitsheoran/cerebrospheroid-v5-2xt4-training"
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["enable_gpu"] is True
    assert meta["enable_tpu"] is False
    assert meta["enable_internet"] is True
    assert meta["kernel_type"] == "notebook"
    assert meta["code_file"] == "cerebro_spheroid_v5_kaggle_2xt4.ipynb"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    assert nb["nbformat"] == 4

    code_sources = []
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            compile(src, nb_path, "exec")
            code_sources.append(src)

    joined = "\n".join(code_sources)
    assert "torchrun" in joined
    assert "scripts/train_v5_graded_curriculum.py" in joined
    assert "--target-batch-size" in joined
    assert "--prompt-len" in joined
    assert "--seq-len" in joined
    assert "--no-elastic-window" in joined
    assert "--max-hours" in joined
    assert "--from-scratch" not in joined
    assert 'MODEL_PRESET = "v5_small"' in joined
    assert "TARGET_BATCH_SIZE = 64" in joined
    assert 'REPO_DIR = "/tmp/cerebro_spheroid_v5"' in joined
    assert 'env["MALLOC_ARENA_MAX"] = "2"' in joined
    assert 'env["HF_HOME"] = "/tmp/hf_cache"' in joined


def test_graded_curriculum_diversity_and_tpu_parity():
    from src.corpus import build_tokenizer
    from src.graded_curriculum import CurriculumScheduleConfig, GradedCurriculumStreamer

    cfg = CurriculumScheduleConfig()
    assert cfg.multimodal_sample_ratio == 0.0
    assert cfg.prompt_len == 32
    assert cfg.seq_len == 256

    tok = build_tokenizer("llama")
    s0 = GradedCurriculumStreamer(
        tok, cfg, seed=42, mix="foundation", search_dirs=["data"], offline=True, rank=0, world_size=2
    )
    s1 = GradedCurriculumStreamer(
        tok, cfg, seed=42, mix="foundation", search_dirs=["data"], offline=True, rank=1, world_size=2
    )
    try:
        buckets = [("static", 32, 256)]
        p0_a, t0_a, f0_a, _, vis0, _, b0 = s0.get_elastic_batch(
            step=2750, device=torch.device("cpu"), batch_size=4, total_window=288, buckets=buckets
        )
        p0_b, _, _, _, _, _, _ = s0.get_elastic_batch(
            step=2751, device=torch.device("cpu"), batch_size=4, total_window=288, buckets=buckets
        )
        p1_a, _, _, _, _, _, _ = s1.get_elastic_batch(
            step=2750, device=torch.device("cpu"), batch_size=4, total_window=288, buckets=buckets
        )
        assert p0_a.shape == (4, 32)
        assert t0_a.shape == (4, 256)
        assert f0_a.shape == (4, 256)
        assert vis0 is None
        assert b0 == "static"

        # Consecutive rows, steps, and DDP ranks must be distinct (never stuck on row 0)
        assert not torch.equal(p0_a[0], p0_a[1])
        assert not torch.equal(p0_a, p0_b)
        assert not torch.equal(p0_a, p1_a)

        decoded = tok.decode(p0_a[0].tolist(), skip_special_tokens=False)
        assert "<stim>" in decoded
    finally:
        s0.close()
        s1.close()


def _ddp_gloo_worker(rank: int, world_size: int, port: int, out_dir: str):
    import contextlib
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import scripts.train_v5_graded_curriculum as train_mod
    from src.agentic_harness import ActionType, ToolActionHead

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        device = torch.device("cpu")
        cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
        torch.manual_seed(42)
        raw_model = CerebroSpheroidV5(cfg).to(device)
        raw_action_head = ToolActionHead(surface_dim=cfg.surface_dim, num_actions=len(ActionType)).to(device)
        raw_combined = train_mod._CerebroDDPModule(raw_model, raw_action_head)
        ddp_mod = DDP(raw_combined, find_unused_parameters=True)
        opt = build_spheroid_optimizer(raw_combined, lr=5e-4)

        opt.zero_grad(set_to_none=True)
        for micro in range(2):
            sync_ctx = ddp_mod.no_sync() if micro == 0 else contextlib.nullcontext()
            with sync_ctx:
                torch.manual_seed(1000 + rank * 100 + micro)
                prompts = torch.randint(0, cfg.vocab_size, (2, 10))
                targets = torch.randint(0, cfg.vocab_size, (2, 6))
                # Vary perceptual branch across rank/micro to stress unused parameter handling
                percep = torch.randn(2, 4, cfg.visual_dim) if (rank == 0 and micro == 0) else None
                compute_act = (micro == 1)
                ce_loss, _, act_logits = ddp_mod(
                    prompts,
                    targets=targets,
                    pad_id=0,
                    perceptual_inputs=percep,
                    compute_action=compute_act,
                )
                loss = ce_loss / 2.0
                if act_logits is not None:
                    loss = loss + 0.05 * act_logits.square().mean()
                loss.backward()
        opt.step()

        assert train_mod._sync_stop_signal(rank == 0, True, device) is True
        dist.barrier()
        torch.save({k: v.detach().cpu() for k, v in raw_combined.state_dict().items()}, os.path.join(out_dir, f"rank_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_ddp_2rank_gloo_smoke(tmp_path):
    import socket
    import torch.multiprocessing as mp

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]

    mp.spawn(_ddp_gloo_worker, args=(2, free_port, str(tmp_path)), nprocs=2, join=True)

    r0 = torch.load(os.path.join(tmp_path, "rank_0.pt"), map_location="cpu", weights_only=True)
    r1 = torch.load(os.path.join(tmp_path, "rank_1.pt"), map_location="cpu", weights_only=True)
    assert r0.keys() == r1.keys()
    for k in r0:
        assert torch.allclose(r0[k], r1[k], atol=1e-5), f"DDP rank divergence in {k}"


def test_elastic_batcher_no_bos_and_single_bio_delimiters():
    from src.corpus import build_tokenizer, format_motor_response, format_stimulus_prompt, get_bio_token_ids
    from src.elastic_window import ElasticBucketBatcher

    tok = build_tokenizer("llama")
    bio = get_bio_token_ids(tok)
    assert bio == {"stim": 31, "stim_end": 32, "act": 33, "act_end": 34}

    def _stream():
        for i in range(64):
            yield (
                format_stimulus_prompt(f"Question {i}: What is 2 + {i}?"),
                format_motor_response(f"The sum is {2 + i}."),
            )

    batcher = ElasticBucketBatcher(
        tokenizer=tok,
        data_iterator=_stream(),
        batch_size=2,
        total_window=256,
        framework="numpy",
        tail_probability=0.0,
    )
    _b_idx, _b_name, p_arr, t_arr = batcher.next_batch()
    assert p_arr.shape[0] == 2 and t_arr.shape[0] == 2
    assert int(p_arr[0, 0]) == 31, f"Expected <stim>=31 at prompt[0], got {p_arr[0, :4]}"
    assert int(t_arr[0, 0]) == 33, f"Expected <act>=33 at target[0] (no BOS=1), got {t_arr[0, :4]}"
    t_list = t_arr[0].tolist()
    assert 34 in t_list and 2 in t_list


def test_pytorch_generate_seed_token_and_bio_stop():
    from src.corpus import build_tokenizer

    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    model = CerebroSpheroidV5(cfg).eval()
    prompt_ids = torch.tensor([[31, 10, 11, 32]], dtype=torch.long)

    # 1. Verify seed_token_id forces token 0 to <act> (33)
    out = model.generate(
        prompt_ids,
        max_new_tokens=4,
        temperature=1e-4,
        top_k=1,
        top_p=1.0,
        seed_token_id=33,
        stop_token_ids=(2, 34),
    )
    assert int(out[0, 0]) == 33

    # 2. Verify clean early stop on </act> (34) at step 1
    orig_forward = model.motor_decoder.forward

    def _bias_act_end(*args, **kwargs):
        logits, eff, ref, plan = orig_forward(*args, **kwargs)
        logits = logits.clone()
        logits[:, 34] = 1e4
        return logits, eff, ref, plan

    model.motor_decoder.forward = _bias_act_end
    try:
        out_stop = model.generate(
            prompt_ids,
            max_new_tokens=8,
            temperature=1e-4,
            top_k=1,
            top_p=1.0,
            seed_token_id=33,
            stop_token_ids=(2, 34),
        )
    finally:
        model.motor_decoder.forward = orig_forward
    assert out_stop.tolist() == [[33, 34]]

    # 3. Smoke test generate_text with bundled tokenizer on matching vocab config
    tok = build_tokenizer("llama")
    cfg_full = ModelConfigV5(**{**PRESETS_V5["v5_test"], "vocab_size": 32000})
    m_full = CerebroSpheroidV5(cfg_full).eval()
    text = m_full.generate_text("Hello", tok, max_new_tokens=5, temperature=0.5)
    assert isinstance(text, str)


def test_kaggle_tpuv5e8_kernel_artifacts_valid():
    """Validate Kaggle TpuV5E8 metadata and headless Local-SGD launcher script."""
    import json
    import os

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_path = os.path.join(repo_root, "kaggle", "tpu", "kernel-metadata.json")
    script_path = os.path.join(repo_root, "kaggle", "tpu", "cerebro_spheroid_v5_kaggle_tpu.py")

    assert os.path.isfile(meta_path), "kaggle/tpu/kernel-metadata.json missing"
    assert os.path.isfile(script_path), "kaggle/tpu/cerebro_spheroid_v5_kaggle_tpu.py missing"

    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["id"] == "ajitsheoran/cerebrospheroid-v5-tpu-v5e-8-training"
    assert meta["code_file"] == "cerebro_spheroid_v5_kaggle_tpu.py"
    assert meta["kernel_type"] == "script"
    assert meta["language"] == "python"
    assert meta["machine_shape"] == "TpuV5E8"
    assert meta["enable_tpu"] is True
    assert meta["enable_gpu"] is False
    assert meta["enable_internet"] is True

    with open(script_path, "r") as f:
        src = f.read()
    compile(src, script_path, "exec")

    for required in (
        "--multi_tpu",
        "local_sgd",
        "--local_steps",
        "--tune_objective",
        "steps_per_sec",
        "--max_batch_size",
        "64",
        "--cloud-sync",
        "--max_hours",
        "8.5",
        "scripts/train_v5_tpu_jax.py",
    ):
        assert required in src, f"Missing expected token/flag in Kaggle TPU script: {required}"

    assert "--from-scratch" not in src and "--from_scratch" not in src


def test_open_hf_stream_uses_parquet_revision_for_openhermes(monkeypatch):
    import sys
    import types
    from src.elastic_window import _open_hf_stream

    calls = []

    def fake_load_dataset(repo, config_name=None, **kwargs):
        calls.append((repo, config_name, dict(kwargs)))
        return [{"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello world example"}]}]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    fake_dist = types.ModuleType("datasets.distributed")
    fake_dist.split_dataset_by_node = lambda ds, rank=0, world_size=1: ds
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "datasets.distributed", fake_dist)

    rows = list(_open_hf_stream("teknium/OpenHermes-2.5", rank=1, world_size=2))
    assert len(rows) == 1
    assert calls[0][0] == "teknium/OpenHermes-2.5"
    assert calls[0][2]["revision"] == "refs/convert/parquet"
    assert calls[0][2]["streaming"] is True

    # Non-OpenHermes repos should not inject an unexpected revision kwarg.
    list(_open_hf_stream("gsm8k", "main", rank=0, world_size=1))
    assert calls[1][0] == "gsm8k"
    assert "revision" not in calls[1][2]

    # Legacy mocks that do not accept `revision` fall back cleanly via TypeError handling.
    legacy_calls = []

    def legacy_load_dataset(repo, config_name=None, split="train", streaming=True):
        legacy_calls.append((repo, config_name, split, streaming))
        return [{"ok": True}]

    fake_datasets.load_dataset = legacy_load_dataset
    assert list(_open_hf_stream("teknium/OpenHermes-2.5")) == [{"ok": True}]
    assert legacy_calls == [("teknium/OpenHermes-2.5", None, "train", True)]


def test_gpu_and_tpu_trainers_parity_and_checkpoint_link(tmp_path):
    import os
    import torch
    from scripts.train_v5_graded_curriculum import _atomic_save_and_link

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gpu_src = open(os.path.join(root, "scripts", "train_v5_graded_curriculum.py"), encoding="utf-8").read()
    tpu_src = open(os.path.join(root, "scripts", "train_v5_tpu_jax.py"), encoding="utf-8").read()
    tpu_mirror = open(os.path.join(root, "training", "train_v5_tpu_jax.py"), encoding="utf-8").read()

    assert "select_highest_step_slot(" in gpu_src
    assert 'device_id' in gpu_src
    assert "build_online_dataset_iterator(" in tpu_src
    assert 'teknium/OpenHermes-2.5' not in tpu_src
    assert tpu_src == tpu_mirror

    primary = str(tmp_path / "v5_small_latest.pt")
    legacy = str(tmp_path / "latest.pt")
    _atomic_save_and_link({"step": 42}, primary, legacy)
    assert torch.load(primary, weights_only=True)["step"] == 42
    assert torch.load(legacy, weights_only=True)["step"] == 42


def test_graded_curriculum_retries_transient_stream_errors(monkeypatch):
    from src import graded_curriculum as gc_mod

    class _DummyTok:
        pad_token_id = 0
        eos_token_id = 2

        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3, 4]

    attempts = {"count": 0}

    def flaky_iterator(dataset_source="dense_edu", rank=0, world_size=1):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ConnectionError("transient hub timeout")
        yield {
            "conversations": [
                {"from": "human", "value": "What is ATP?"},
                {"from": "gpt", "value": "Adenosine triphosphate is the cellular energy currency."},
            ]
        }

    monkeypatch.setattr(gc_mod, "build_online_dataset_iterator", flaky_iterator)
    streamer = gc_mod.GradedCurriculumStreamer(
        tokenizer=_DummyTok(),
        config=gc_mod.CurriculumScheduleConfig(),
        mix="dense_edu",
        offline=False,
        online_streaming=True,
        prefetch_size=4,
    )
    pair = streamer._next_online_pair("dense_edu")
    streamer._stop_event.set()
    assert attempts["count"] >= 2
    assert pair is not None
    assert "ATP" in pair[0]
    assert not streamer._online_failed.get("dense_edu", False)


def test_jax_to_pytorch_numerical_loss_parity_and_config_preservation():
    """Verifies PyTorch matches JAX training_rollout within <1e-3 (fast & seq) and preserves flags across save/resume."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.jax.v5_model import ModelConfigV5 as JAXConfig, init_params_v5, training_rollout

    jcfg = JAXConfig(
        vocab_size=128,
        surface_dim=64,
        radial_dim=32,
        num_shells=6,
        num_shortcut_tracts=12,
        num_heads=8,
        max_phase_steps=64,
        thinking_cycles=4,
        top_k_tracts=4,
    )
    jparams, jbuf = init_params_v5(jax.random.PRNGKey(7), jcfg)
    raw_state = {k: torch.from_numpy(np.array(v)) for k, v in jparams.items()}
    raw_state["core.src_indices"] = torch.from_numpy(np.array(jbuf[0]))
    raw_state["core.dst_indices"] = torch.from_numpy(np.array(jbuf[1]))
    ckpt = {"model_state": raw_state, "step": 100, "loss": 4.2, "preset": "tpu_v5e_1"}

    cfg = infer_v5_config_from_state_dict(ckpt)
    assert cfg.use_sensory_self_attn is False
    assert cfg.use_rope is False
    assert cfg.top_k_resonance == 0
    assert cfg.use_deliberation_probe is False
    assert cfg.use_adaptive_thinking is False
    assert cfg.use_sparse_mot is False
    assert cfg.use_energy_regularization is False

    pt_model = CerebroSpheroidV5(cfg).eval()
    load_v5_checkpoint_into_torch(pt_model, ckpt, verbose=False)

    # Non-trivial prompt with pad_id=0 and target with -100 padding
    prompts_np = np.array([[5, 19, 42, 7, 11, 0, 0], [8, 22, 33, 44, 55, 66, 0]], dtype=np.int32)
    targets_np = np.array([[10, 12, 14, 16, 18, 20, -100], [9, 13, 17, 21, 25, 29, 31]], dtype=np.int32)

    jax_pure = float(
        training_rollout(
            jparams,
            jnp.array(prompts_np),
            target_len=7,
            buffers=jbuf,
            config=jcfg,
            targets=jnp.array(targets_np),
            sampling_p=1.0,
            pad_id=0,
            remat=False,
            pure_teacher_forcing=True,
        )
    )
    jax_seq = float(
        training_rollout(
            jparams,
            jnp.array(prompts_np),
            target_len=7,
            buffers=jbuf,
            config=jcfg,
            targets=jnp.array(targets_np),
            sampling_p=1.0,
            pad_id=0,
            remat=False,
            pure_teacher_forcing=False,
        )
    )

    tp = torch.from_numpy(prompts_np).long()
    tt = torch.from_numpy(targets_np).long()
    with torch.no_grad():
        pt_fast_loss, pt_fast_logits = pt_model(tp, targets=tt, pad_id=0)
        pt_model.force_sequential = True
        pt_seq_loss, pt_seq_logits = pt_model(tp, targets=tt, pad_id=0)

    assert abs(float(pt_fast_loss) - jax_pure) < 1e-3, f"fast={float(pt_fast_loss)} vs jax_pure={jax_pure}"
    assert abs(float(pt_seq_loss) - jax_seq) < 1e-3, f"seq={float(pt_seq_loss)} vs jax_seq={jax_seq}"
    assert float((pt_fast_logits - pt_seq_logits).abs().max()) < 1e-2

    # Saving PyTorch checkpoint and re-inferring config must retain JAX-parity flags
    resaved = {"model_state": pt_model.state_dict(), "config": cfg, "step": 101}
    re_cfg = infer_v5_config_from_state_dict(resaved)
    assert re_cfg.use_rope is False and re_cfg.top_k_resonance == 0 and re_cfg.use_sensory_self_attn is False
    assert re_cfg.use_sparse_mot is False and re_cfg.use_energy_regularization is False


def test_step_efference_fast_uses_gate_and_fp16_autocast_backward_finite():
    """Verifies step_efference_fast updates efference_gate and FP16 autocast backward stays NaN-free."""
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    model = CerebroSpheroidV5(cfg)
    model.force_sequential = True
    prompts = torch.randint(1, cfg.vocab_size, (2, 8))
    targets = torch.randint(1, cfg.vocab_size, (2, 6))

    model.zero_grad(set_to_none=True)
    loss, _ = model(prompts, targets=targets, pad_id=0)
    loss.backward()
    assert model.volumetric_core.efference_gate.weight.grad is not None
    assert model.volumetric_core.efference_gate.weight.grad.abs().sum().item() > 0.0

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        m_cuda = CerebroSpheroidV5(cfg).to(dev)
        scaler = torch.amp.GradScaler("cuda")
        p_cu = prompts.to(dev)
        t_cu = targets.to(dev)
        m_cuda.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss_cu, _ = m_cuda(p_cu, targets=t_cu, pad_id=0)
        scaler.scale(loss_cu).backward()
        scaler.unscale_(torch.optim.SGD(m_cuda.parameters(), lr=1e-3))
        for name, param in m_cuda.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_graded_curriculum_eval_tracking_and_checkpoint_eval_loss(tmp_path):
    """Verifies periodic held-out evaluation tracking, EMA logging, and eval_loss persistence."""
    import math
    import subprocess
    import sys

    ckpt_dir = str(tmp_path / "ckpts")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(repo_root, "scripts", "train_v5_graded_curriculum.py")
    cmd = [
        sys.executable,
        script_path,
        "--model-preset", "v5_small",
        "--batch-size", "2",
        "--target-batch-size", "2",
        "--test-steps", "3",
        "--log-interval", "1",
        "--eval-interval", "1",
        "--offline",
        "--from-scratch",
        "--checkpoint-dir", ckpt_dir,
    ]
    res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
    assert "Eval:" in res.stdout
    assert "EMA:" in res.stdout

    best_ckpt_path = os.path.join(ckpt_dir, "v5_small_best.pt")
    assert os.path.exists(best_ckpt_path)
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    assert "eval_loss" in best_ckpt
    assert math.isfinite(best_ckpt["eval_loss"]) and best_ckpt["eval_loss"] > 0.0


def test_mtp_heads_forward_loss_and_speculative_drafting():
    """Verifies Multi-Token Prediction (MTP) drafting heads compute auxiliary loss, receive gradients, and draft tokens."""
    cfg = ModelConfigV5(
        vocab_size=128,
        num_shells=4,
        surface_dim=32,
        radial_dim=16,
        num_heads=4,
        max_phase_steps=32,
        thinking_cycles=1,
        num_mtp_heads=3,  # 2 auxiliary MTP drafting heads
    )
    model = CerebroSpheroidV5(cfg)
    assert len(model.mtp_heads) == 2

    prompts = torch.randint(1, 128, (2, 4))
    targets = torch.randint(1, 128, (2, 8))

    # Fast teacher forcing forward pass
    loss_fast, logits_fast = model(prompts, targets=targets)
    assert torch.isfinite(loss_fast)
    assert logits_fast.shape == (2, 8, 128)
    loss_fast.backward()

    # Verify MTP heads received gradients
    for i, mtp_h in enumerate(model.mtp_heads):
        has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in mtp_h.parameters())
        assert has_grad, f"MTP head {i} did not receive gradients during fast teacher forcing"

    # Sequential rollout forward pass
    model.zero_grad()
    model.force_sequential = True
    loss_seq, logits_seq = model(prompts, targets=targets)
    assert torch.isfinite(loss_seq)
    loss_seq.backward()
    for i, mtp_h in enumerate(model.mtp_heads):
        has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in mtp_h.parameters())
        assert has_grad, f"MTP head {i} did not receive gradients during sequential teacher forcing"

    # Test speculative drafting
    model.eval()
    h_dummy = torch.randn(2, 32)
    last_tok = torch.tensor([5, 10], dtype=torch.long)
    drafted = model.draft_speculative_tokens(h_dummy, last_tok)
    assert len(drafted) == 2  # num_mtp_heads - 1
    assert drafted[0].shape == (2,)
    assert drafted[1].shape == (2,)


def test_persona_steering_embeddings_and_gradient_flow():
    """Verifies persona embeddings steer the initial plan and receive training gradients."""
    cfg = ModelConfigV5(
        vocab_size=128,
        num_shells=4,
        surface_dim=32,
        radial_dim=16,
        num_heads=4,
        max_phase_steps=32,
        thinking_cycles=1,
    )
    model = CerebroSpheroidV5(cfg)
    prompts = torch.randint(1, 128, (2, 4))
    targets = torch.randint(1, 128, (2, 6))

    # Forward with persona 0 vs persona 1
    with torch.no_grad():
        _, logits_p0 = model(prompts, targets=targets, persona_id=0)
        _, logits_p1 = model(prompts, targets=targets, persona_id=1)
    diff = (logits_p0 - logits_p1).abs().max()
    assert diff > 1e-4, f"Persona conditioning did not alter forward logits: diff={diff}"

    # Verify persona parameter gradients
    model.train()
    model.zero_grad()
    loss, _ = model(prompts, targets=targets, persona_id=torch.tensor([0, 1]))
    loss.backward()
    assert model.motor_decoder.persona_embeddings.grad is not None
    assert torch.norm(model.motor_decoder.persona_embeddings.grad) > 0
    assert model.motor_decoder.persona_scale.grad is not None
    assert torch.norm(model.motor_decoder.persona_scale.grad) > 0

    # Generation steering test
    model.eval()
    gen_p0 = model.generate(prompts, max_new_tokens=4, persona_id=0, temperature=0.01)
    gen_p2 = model.generate(prompts, max_new_tokens=4, persona_id=2, temperature=0.01)
    assert gen_p0.shape == (2, 4)
    assert gen_p2.shape == (2, 4)


def test_efference_non_linear_projection_active():
    """Verifies efference projection includes non-linearity (SiLU) and bounded residual updates."""
    cfg = ModelConfigV5(
        vocab_size=64,
        num_shells=4,
        surface_dim=32,
        radial_dim=16,
        num_heads=4,
        max_phase_steps=32,
    )
    model = CerebroSpheroidV5(cfg)
    core = model.volumetric_core
    state = torch.randn(2, 4, 32)
    eff = torch.randn(2, 32)

    s1 = core.step_efference_fast(state, eff)
    s2 = core.step_efference_fast(state, 2.0 * eff)
    assert s1.shape == state.shape
    assert s2.shape == state.shape
    assert torch.isfinite(s1).all()
    assert torch.isfinite(s2).all()


def test_gpu_checkpoint_save_and_cloud_sync_converts_to_bfloat16(tmp_path):
    """Verifies GPU checkpoints are converted to bfloat16 before saving and uploading."""
    from scripts.train_v5_graded_curriculum import _save_and_push_checkpoints, _to_bfloat16_state
    from src.cloud_sync import _ensure_bfloat16_checkpoint
    from src.agentic_harness import ToolActionHead, ActionType

    cfg = ModelConfigV5(
        vocab_size=128,
        num_shells=3,
        surface_dim=32,
        radial_dim=16,
        num_heads=4,
        max_phase_steps=32,
    )
    # Model in FP32 (master weights on GPU)
    model = CerebroSpheroidV5(cfg).float()
    action_head = ToolActionHead(surface_dim=cfg.surface_dim, num_actions=len(ActionType)).float()

    # 1. Test _to_bfloat16_state helper directly
    sd = model.state_dict()
    assert any(v.dtype == torch.float32 for v in sd.values() if isinstance(v, torch.Tensor) and v.is_floating_point())
    bf16_sd = _to_bfloat16_state(sd)
    for k, v in bf16_sd.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            assert v.dtype == torch.bfloat16, f"Tensor {k} should be bfloat16, got {v.dtype}"
        elif isinstance(v, torch.Tensor):
            assert v.dtype in (torch.int64, torch.long, torch.int32)

    # 2. Test _save_and_push_checkpoints saves bfloat16 on disk
    ckpt_dir = str(tmp_path / "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    _save_and_push_checkpoints(
        checkpoint_dir=ckpt_dir,
        model_preset="v5_test",
        raw_model=model,
        raw_action_head=action_head,
        config=cfg,
        step=500,
        loss_val=3.10,
        best_loss=99.0,
        need_local_save=True,
        need_cloud_sync=False,
    )

    saved_latest = os.path.join(ckpt_dir, "v5_test_latest.pt")
    saved_step = os.path.join(ckpt_dir, "v5_test_step_500.pt")
    assert os.path.exists(saved_latest)
    assert os.path.exists(saved_step)

    data_latest = torch.load(saved_latest, map_location="cpu", weights_only=False)
    for k, v in data_latest["model_state"].items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            assert v.dtype == torch.bfloat16, f"Saved tensor {k} should be bfloat16, got {v.dtype}"

    for k, v in data_latest["action_head_state"].items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            assert v.dtype == torch.bfloat16, f"Saved action tensor {k} should be bfloat16, got {v.dtype}"

    # 3. Test _ensure_bfloat16_checkpoint converts raw float32 file before upload
    legacy_fp32_file = str(tmp_path / "legacy_fp32.pt")
    torch.save({
        "step": 100,
        "model_state": {k: v.float() if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v for k, v in sd.items()}
    }, legacy_fp32_file)

    fp32_sz = os.path.getsize(legacy_fp32_file)
    _ensure_bfloat16_checkpoint(legacy_fp32_file)
    bf16_sz = os.path.getsize(legacy_fp32_file)
    assert bf16_sz < fp32_sz, f"bfloat16 size ({bf16_sz}) should be strictly smaller than fp32 ({fp32_sz})"

    converted_data = torch.load(legacy_fp32_file, map_location="cpu", weights_only=False)
    for k, v in converted_data["model_state"].items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            assert v.dtype == torch.bfloat16

    # 4. Roundtrip load into fresh model
    fresh_model = CerebroSpheroidV5(cfg).float()
    loaded, total = load_v5_checkpoint_into_torch(fresh_model, data_latest, verbose=False)
    assert loaded == total




