"""CerebroSpheroid V5: Kaggle TPU v5e-8 Multi-TPU Local-SGD Training Launcher.

Runs pure JAX training across all 8 TPU v5e cores using independent per-TPU
local optimizer steps (`--multi_tpu local_sgd --local_steps 4`) followed by
periodic `lax.pmean` weight & optimizer-state merging (+32 global steps/round),
resuming from the latest cloud checkpoint on bioflo.in.
"""

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

# Real-time out-of-band telemetry (reverse-engineered from kaggle-tpu-lab)
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "cs5-tpu-ajitsheoran")


def publish(phase: str, **extra):
    """Publish live telemetry events to ntfy.sh so developer sees instant progress."""
    print(f"📡 [telemetry] {phase} {json.dumps(extra) if extra else ''}", flush=True)
    if not NTFY_TOPIC:
        return
    try:
        payload = {
            "topic": NTFY_TOPIC,
            "title": f"Cerebro-TPU {phase}",
            "message": json.dumps({"phase": phase, "time": time.time(), **extra}),
            "tags": ["tpu", "cerebro"],
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
WORKDIR = "/kaggle/working/cerebro_spheroid_v5"
CHECKPOINT_DIR = "/kaggle/working/checkpoints/v5_small"


def verify_tpu_devices():
    """Confirm JAX sees TPU devices before spawning the trainer subprocess."""
    if os.environ.get("PJRT_DEVICE", "").upper() == "CPU":
        os.environ.pop("PJRT_DEVICE", None)

    import jax

    devices = jax.devices()
    backend = jax.default_backend()
    print(f"JAX Version  : {jax.__version__}")
    print(f"JAX Backend  : {backend}")
    print(f"TPU Devices  : {len(devices)} x {devices[0].device_kind if devices else 'none'}")
    for idx, dev in enumerate(devices):
        print(f"  - [{idx}] {dev}")
    if "tpu" not in backend.lower():
        print(f"⚠️ Warning: Expected TPU backend, got {backend!r}", flush=True)
    publish("devices_verified", count=len(devices), backend=backend, kind=devices[0].device_kind if devices else "none")


def sync_repository():
    """Clone or reset cerebro_spheroid_v5 to latest origin/main."""
    if not os.path.exists(WORKDIR):
        print(f"📦 Cloning {REPO_URL} -> {WORKDIR} ...", flush=True)
        subprocess.run(["git", "clone", REPO_URL, WORKDIR], check=True)
    else:
        print(f"🔄 Updating existing repository at {WORKDIR} ...", flush=True)

    os.chdir(WORKDIR)
    subprocess.run(["git", "fetch", "origin", "main"], check=True)
    subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    print(f"✅ Repository synchronized at commit [{commit_hash}]", flush=True)

    tok_dir = os.path.join(WORKDIR, "data", "tokenizer")
    if not os.path.isdir(tok_dir):
        raise FileNotFoundError(f"Offline LLaMA tokenizer directory missing at {tok_dir}")
    print(f"🔤 Offline LLaMA-32k tokenizer verified at {tok_dir}", flush=True)
    publish("repo_synced", commit=commit_hash)


def ensure_dependencies():
    """Install only missing pure-Python/data deps with high-speed uv package manager."""
    required = ["datasets", "pyarrow", "transformers", "sentencepiece", "optax"]
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]

    # Ensure datasets >= 3.0.0 for Modern HuggingFace parquet streaming
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
        print(f"📦 Bootstrapping dependencies ({missing}, torch={has_torch}) using uv...", flush=True)
        publish("deps_installing", missing=missing)
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
                        sys.executable,
                        "-m",
                        "uv",
                        "pip",
                        "install",
                        "--system",
                        "-q",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                        "torch",
                    ],
                    check=False,
                )
            else:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-q",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                        "torch",
                    ],
                    check=True,
                )
        publish("deps_ready")
    else:
        print("✅ All non-JAX data/tokenizer and torch dependencies already present.", flush=True)
        publish("deps_ready")


def main():
    print("=" * 80)
    print("🚀 CEREBRO-SPHEROID V5: KAGGLE TPU v5e-8 MULTI-TPU TRAINING LAUNCH")
    print("=" * 80, flush=True)

    ensure_dependencies()
    verify_tpu_devices()
    sync_repository()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "scripts/train_v5_tpu_jax.py",
        "--preset",
        "tpu_v5e_1",
        "--dataset_source",
        "dense_edu",
        "--optimizer",
        "spheroid",
        "--multi_tpu",
        "local_sgd",
        "--local_steps",
        "4",
        "--auto-tune",
        "--tune_objective",
        "steps_per_sec",
        "--max_batch_size",
        "64",
        "--remat",
        "auto",
        "--prompt_len",
        "32",
        "--seq-len",
        "256",
        "--lr",
        "0.0005",
        "--min_lr",
        "3e-5",
        "--lr_schedule",
        "warm_restart",
        "--deliberation_mode",
        "teacher_forcing",
        "--min_sampling_p",
        "1.0",
        "--total-steps",
        "1000000",
        "--log-every",
        "10",
        "--eval_interval",
        "1000",
        "--save-interval",
        "500",
        "--periodic_every",
        "2000",
        "--cloud-sync",
        "--cloud_save_hours",
        "0.25",
        "--max_hours",
        "8.5",
        "--checkpoint-dir",
        CHECKPOINT_DIR,
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("PJRT_DEVICE", None)

    print("\n🚀 Launching Multi-TPU Trainer:")
    print(" ".join(cmd))
    print("=" * 80, flush=True)
    publish("trainer_started", preset="tpu_v5e_1", multi_tpu="local_sgd")

    result = subprocess.run(cmd, env=env)
    publish("trainer_exited", returncode=result.returncode)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
