#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 CerebroSpheroid V5: Kaggle TPU v5e-8 / v3-8 High-Throughput Pretraining Launcher
================================================================================
Production-grade Cloud TPU runner for Kaggle TPU environments (TPU v5e-8 / TPU v3-8).
Executes pure JAX high-throughput training across all 8 TPU cores with:
  1. Linux Transparent Hugepage activation & PJRT environment initialization
  2. Dynamic Hugging Face Hub & Bioflo Cloud Sync authentication (website vault & Kaggle secrets)
  3. Automatic repository synchronization to latest origin/main & offline LLaMA tokenizer verification
  4. Local scratch storage with zero Google Drive dependency (checkpoints stream directly to bioflo.in)
  5. 100% loss-bearing contiguous pretraining curriculum (85% FineWeb 10BT + 15% The Stack v2 Code)
  6. Multi-TPU Local-SGD execution with periodic pmean parameter synchronization (+32 global steps/round)
  7. SCMER (Selective Cosine-Margin Embedding Regularization) anti-anisotropy regularizer
  8. Wall-clock guard (--max_hours 8.5) to safely flush checkpoints before Kaggle 9h runtime expiration
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from typing import List, Optional

# Out-of-band telemetry for instant progress tracking
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "cs5-tpu-ajitsheoran")


def publish(phase: str, **extra):
    """Publish live telemetry events to ntfy.sh so developer sees instant progress."""
    print(f"📡 [telemetry] {phase} {json.dumps(extra) if extra else ''}", flush=True)
    if not NTFY_TOPIC:
        return
    try:
        payload = {
            "topic": NTFY_TOPIC,
            "title": f"Cerebro-Kaggle-TPU {phase}",
            "message": json.dumps({"phase": phase, "time": time.time(), **extra}),
            "tags": ["tpu", "cerebro", "kaggle"],
        }
        req = urllib.request.Request(
            "https://ntfy.sh",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


# PyArrow compatibility shim for Python 3.12+ / pyarrow 19+
try:
    import pyarrow as pa
    if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
        pa.PyExtensionType = pa.ExtensionType
except Exception:
    pass

REPO_URL = "https://github.com/AjitKumar-git/cerebro_spheroid_v5.git"


def resolve_paths() -> tuple[str, str]:
    """Resolves working directory and checkpoint staging path based on environment."""
    if os.path.exists("/kaggle/working"):
        repo_dir = "/kaggle/working/cerebro_spheroid_v5"
        ckpt_dir = "/kaggle/working/checkpoints/v5_small_scratch"
    elif os.path.exists("/content"):
        repo_dir = "/content/cerebro_v5"
        ckpt_dir = "/content/checkpoints/v5_small_scratch"
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        repo_dir = base_dir
        ckpt_dir = os.path.join(base_dir, "checkpoints", "v5_small_scratch")

    return repo_dir, ckpt_dir


def configure_system_environment():
    """Sets system-level hugepages, TPU backend, and execution flags."""
    # 1. Enable Linux Transparent Hugepages
    os.system("sudo sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled' 2>/dev/null || true")

    # 2. Configure PJRT & PyTorch/XLA non-interference flags
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TORCH_XLA_DISABLE_DISTRIBUTED"] = "1"


def verify_tpu_devices():
    """Confirm JAX detects 8 TPU cores before launching training."""
    if os.environ.get("PJRT_DEVICE", "").upper() == "CPU":
        os.environ.pop("PJRT_DEVICE", None)

    import jax

    devices = jax.devices()
    backend = jax.default_backend().upper()
    print("=" * 80)
    print("⚡ CEREBROSPHEROID V5: KAGGLE TPU HIGH-THROUGHPUT ENGINE")
    print("=" * 80)
    print(f"JAX Version     : {jax.__version__}")
    print(f"JAX Backend     : {backend}")
    print(f"Active Devices  : {len(devices)} x {devices[0].device_kind if devices else 'none'}")
    for idx, dev in enumerate(devices):
        print(f"  - Core [{idx}]: {dev}")

    if "TPU" not in backend:
        print(f"⚠️ Warning: Expected TPU backend, got {backend!r}. Make sure Kaggle accelerator is set to TPU.", flush=True)

    publish("devices_verified", count=len(devices), backend=backend, kind=devices[0].device_kind if devices else "none")


def ensure_dependencies():
    """Installs required data streaming and model dependencies."""
    required = ["datasets", "transformers", "sentencepiece", "optax", "pyarrow", "huggingface_hub"]
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]

    # Verify datasets >= 3.0.0
    if "datasets" not in missing:
        try:
            import datasets as _ds
            major = int(str(getattr(_ds, "__version__", "0")).split(".")[0])
            if major < 3:
                missing.append("datasets>=3.0.0")
        except Exception:
            missing.append("datasets>=3.0.0")
    else:
        missing = [pkg if pkg != "datasets" else "datasets>=3.0.0" for pkg in missing]

    has_torch = importlib.util.find_spec("torch") is not None

    if missing or not has_torch:
        print(f"📦 Bootstrapping dependencies ({missing}, torch={has_torch})...", flush=True)
        publish("deps_installing", missing=missing)

        has_uv = False
        try:
            if shutil.which("uv") is None and importlib.util.find_spec("uv") is None:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=False)
            has_uv = shutil.which("uv") is not None or importlib.util.find_spec("uv") is not None
        except Exception:
            has_uv = False

        if missing:
            if has_uv:
                subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--system", "-q", *missing], check=False)
            else:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

        if not has_torch:
            print("📦 Installing CPU-only PyTorch for .pt checkpoint serialization...", flush=True)
            if has_uv:
                subprocess.run(
                    [
                        sys.executable, "-m", "uv", "pip", "install", "--system", "-q",
                        "--index-url", "https://download.pytorch.org/whl/cpu", "torch"
                    ],
                    check=False,
                )
            else:
                subprocess.run(
                    [
                        sys.executable, "-m", "pip", "install", "-q",
                        "--index-url", "https://download.pytorch.org/whl/cpu", "torch"
                    ],
                    check=True,
                )
        publish("deps_ready")
    else:
        print("✅ All required training dependencies verified.", flush=True)
        publish("deps_ready")


def sync_repository(repo_dir: str):
    """Clones or resets the repository to the latest origin/main."""
    if not os.path.exists(repo_dir):
        print(f"📦 Cloning repository {REPO_URL} -> {repo_dir} ...", flush=True)
        subprocess.run(["git", "clone", REPO_URL, repo_dir], check=True)
    else:
        print(f"🔄 Updating existing repository at {repo_dir} ...", flush=True)

    orig_cwd = os.getcwd()
    os.chdir(repo_dir)
    subprocess.run(["git", "fetch", "--all"], check=True)
    subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    print(f"✅ Repository synchronized at commit [{commit_hash}]", flush=True)

    tok_dir = os.path.join(repo_dir, "data", "tokenizer")
    if not os.path.isdir(tok_dir):
        raise FileNotFoundError(f"Offline LLaMA tokenizer directory missing at {tok_dir}")
    print(f"🔤 Offline LLaMA-32k tokenizer verified at {tok_dir}", flush=True)
    publish("repo_synced", commit=commit_hash)

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def authenticate_credentials(repo_dir: str):
    """Authenticates HuggingFace Hub and Bioflo Cloud Sync from secrets or website vault."""
    # 1. Check Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        hf_sec = secrets.get_secret("HF_TOKEN")
        if hf_sec:
            os.environ["HF_TOKEN"] = hf_sec
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_sec
            print("🔑 HF_TOKEN resolved from Kaggle Secrets.")
        bioflo_sec = secrets.get_secret("BIOFLO_SYNC_TOKEN")
        if bioflo_sec:
            os.environ["BIOFLO_SYNC_TOKEN"] = bioflo_sec
            os.environ["CEREBRO_SYNC_TOKEN"] = bioflo_sec
            print("🌐 BIOFLO_SYNC_TOKEN resolved from Kaggle Secrets.")
    except Exception:
        pass

    # 2. Dynamic website vault fallback
    try:
        from src.cloud_sync import get_sync_token
        from src.hf_utils import get_hf_token

        bioflo_tok = get_sync_token()
        if bioflo_tok:
            os.environ["BIOFLO_SYNC_TOKEN"] = bioflo_tok
            os.environ["CEREBRO_SYNC_TOKEN"] = bioflo_tok
            print("🌐 Bioflo Cloud Sync Token successfully authenticated.")

        hf_tok = get_hf_token()
        if hf_tok:
            os.environ["HF_TOKEN"] = hf_tok
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_tok
            from huggingface_hub import login
            login(token=hf_tok, add_to_git_credential=False)
            print("🔑 Hugging Face Hub successfully authenticated dynamically.")
    except Exception as e:
        print(f"ℹ️ Credential notice: {e}")


def launch_trainer(repo_dir: str, ckpt_dir: str, extra_args: Optional[List[str]] = None):
    """Launches the production Pure JAX trainer as an unbuffered subprocess."""
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        os.path.join(repo_dir, "scripts", "train_v5_tpu_jax.py"),
        "--preset", "v5_small",
        "--dataset_source", "pretrain",
        "--optimizer", "spheroid",
        "--multi_tpu", "local_sgd",
        "--local_steps", "4",
        "--auto-tune",
        "--tune_objective", "steps_per_sec",
        "--effective-batch-size", "64",
        "--remat", "auto",
        "--min_context_window", "512",
        "--context_stage_steps", "100000",
        "--total_window", "4096",
        "--sensory_position_steps", "4096",
        "--max_phase_steps", "4096",
        "--thinking_cycles", "4",
        "--lr", "2e-3",
        "--min_lr", "1e-5",
        "--lr_schedule", "warm_restart",
        "--deliberation_mode", "teacher_forcing",
        "--total-steps", "1000000",
        "--log-every", "10",
        "--eval_interval", "1000",
        "--save-interval", "250",
        "--periodic_every", "2000",
        "--cloud-sync",
        "--cloud_weights_only",
        "--cloud_url", "https://bioflo.in/api/cerebro/sync.php",
        "--checkpoint-dir", ckpt_dir,
        "--scratch-dir", ckpt_dir,
        "--max_hours", "8.5",
        "--scmer-lambda", "0.05",
        "--scmer-margin", "0.12",
        "--scmer-interval", "8",
        "--scmer-subsample", "512",
        "--scmer-pair-samples", "8192",
    ]

    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("PJRT_DEVICE", None)

    print("\n" + "=" * 80)
    print("🚀 Launching CerebroSpheroid V5 Small Multi-TPU Trainer:")
    print(" ".join(cmd))
    print("=" * 80 + "\n", flush=True)
    publish("trainer_started", preset="v5_small", multi_tpu="local_sgd")

    # Forward signals to child process cleanly
    proc = subprocess.Popen(cmd, env=env, cwd=repo_dir)

    def handle_signal(sig, frame):
        print(f"\n⚠️ Received signal {sig}, terminating trainer gracefully...", flush=True)
        proc.terminate()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    ret = proc.wait()
    publish("trainer_exited", returncode=ret)
    print(f"\n🏁 Trainer process completed with returncode: {ret}")
    sys.exit(ret)


def main():
    parser = argparse.ArgumentParser(description="Kaggle TPU CerebroSpheroid V5 Runner")
    parser.add_argument("--repo_dir", type=str, default=None, help="Repository directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Local scratch checkpoint directory")
    args, unknown = parser.parse_known_args()

    default_repo, default_ckpt = resolve_paths()
    repo_dir = args.repo_dir or default_repo
    ckpt_dir = args.checkpoint_dir or default_ckpt

    configure_system_environment()
    ensure_dependencies()
    verify_tpu_devices()
    sync_repository(repo_dir)
    authenticate_credentials(repo_dir)
    launch_trainer(repo_dir, ckpt_dir, extra_args=unknown)


if __name__ == "__main__":
    main()
