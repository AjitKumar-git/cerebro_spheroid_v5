"""
CerebroSpheroid V5 Cloud Synchronization Client.
================================================
Enables transparent, continuous checkpoint and metrics synchronization between
Google Colab, Local Workstations, and the centralized dashboard endpoint:
https://bioflo.in/api/cerebro/sync.php

Key Features:
1. Two-way live sync: Resumes checkpoints seamlessly across Colab and Local.
2. Chunked Streaming: Bypasses Cloudflare 100 MB limits by streaming 10 MB chunks.
3. Strict storage safety: Mirrored slots strictly bounded to latest.pt and best.pt.
4. Auto-resumption: Downloads with HTTP Range chunk verification and zip integrity validation.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple

# Prefer IPv4 to avoid hangs on misconfigured DNS64/NAT64 transitions
_orig_getaddrinfo = socket.getaddrinfo
def _preferred_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        if family == 0 or family == socket.AF_UNSPEC:
            res = _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
            if res:
                return res
    except Exception:
        pass
    return _orig_getaddrinfo(host, port, family, type, proto, flags)
socket.getaddrinfo = _preferred_getaddrinfo

def get_sync_token(token: Optional[str] = None) -> str:
    """Retrieves sync token from argument, environment variables, or local secret store."""
    if token:
        return token
    tok = os.environ.get("BIOFLO_SYNC_TOKEN", os.environ.get("CEREBRO_SYNC_TOKEN", ""))
    if tok:
        return tok
    for p in (os.path.expanduser("~/.bioflo_token"), os.path.expanduser("~/.cerebro_token")):
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    t = f.read().strip()
                    if t:
                        return t
            except Exception:
                pass
    return ""


DEFAULT_URL = os.environ.get("BIOFLO_SYNC_URL", "https://bioflo.in/api/cerebro/sync.php")
DEFAULT_TOKEN = get_sync_token()
TPU_PRESET_ALIASES = ("tpu_v5e_1", "tpu_v6e_1", "tpu_v6e", "tpu", "v5_tpu")


def normalize_model_name(model: Optional[str]) -> Optional[str]:
    """Normalizes model preset names into canonical bioflo.in tiers (v5_small, v5_compact, v5_base)."""
    if not model:
        return model
    m = str(model).strip()
    if m in TPU_PRESET_ALIASES:
        return "v5_small"
    if "compact" in m:
        return "v5_compact"
    if "base" in m:
        return "v5_base"
    return m

_IN_FLIGHT_UPLOADS: Dict[str, int] = {}
_IN_FLIGHT_UPLOAD_TIMES: Dict[str, float] = {}
_LAST_BEST_UPLOAD_TIME: float = 0.0
_BEST_COOLDOWN_SECONDS: float = 300.0  # 5 minutes minimum between best.pt uploads
_UPLOAD_LOCK = threading.Lock()
_ACTIVE_UPLOAD_THREADS: list = []


def detect_source_environment() -> str:
    """Detects whether current execution is in Google Colab, Kaggle, or Local."""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ or os.path.exists("/content"):
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


def wait_for_in_flight_uploads(timeout: float = 60.0):
    """Waits for any background cloud upload threads to complete."""
    global _ACTIVE_UPLOAD_THREADS
    with _UPLOAD_LOCK:
        threads = [t for t in _ACTIVE_UPLOAD_THREADS if t.is_alive()]
    for t in threads:
        print(f"⏳ [cloud_sync] Waiting for in-flight cloud upload to finish ({t.name})...")
        t.join(timeout=timeout)
    with _UPLOAD_LOCK:
        _ACTIVE_UPLOAD_THREADS = [t for t in _ACTIVE_UPLOAD_THREADS if t.is_alive()]


def check_remote_manifest(
    model: Optional[str] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Queries bioflo.in to retrieve checkpoint status, step numbers, and loss metrics for model tier."""
    model = normalize_model_name(model)
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN

    req_url = f"{sync_url}?action=status"
    if model:
        req_url += f"&model={model}"

    req = urllib.request.Request(
        req_url,
        headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroV5CloudSync/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"⚠️ [cloud_sync] Unable to reach sync hub ({sync_url}): {e}")
        return {"status": "offline", "has_latest": False, "has_best": False, "manifest": {}}


def select_highest_step_slot(
    manifest_resp: Dict[str, Any],
    model: Optional[str],
    target_type: str = "latest",
) -> Tuple[Optional[str], int, float]:
    """
    Pick whichever bioflo.in slot (named model vs root fallback) holds the highest step.
    Returns (slot_model_arg, step, loss).
    """
    norm_model = normalize_model_name(model)
    manifest = (manifest_resp or {}).get("manifest") or {}
    models_block = (manifest_resp or {}).get("models") or {}

    candidates = []
    lookup_models = [norm_model] if norm_model else []
    if norm_model and "compact" in norm_model and norm_model != "v5_compact":
        lookup_models.append("v5_compact")
    elif norm_model and "base" in norm_model and norm_model != "v5_base":
        lookup_models.append("v5_base")

    for m in lookup_models:
        named_entry = manifest.get(f"{m}_{target_type}")
        if not named_entry and m in models_block:
            named_entry = models_block[m].get(f"{target_type}_manifest")
        if isinstance(named_entry, dict) and int(named_entry.get("step", 0)) > 0:
            candidates.append((m, int(named_entry.get("step", 0)), float(named_entry.get("loss", float("inf")))))

    # Root slot fallback is strictly for v5_small or unassigned root models, never for compact/base tiers
    allow_root = (norm_model is None or norm_model in ("v5_small", "root", "") or norm_model in TPU_PRESET_ALIASES)
    if allow_root:
        root_entry = manifest.get(target_type)
        if isinstance(root_entry, dict) and int(root_entry.get("step", 0)) > 0:
            root_model = root_entry.get("model") or None
            candidates.append((root_model, int(root_entry.get("step", 0)), float(root_entry.get("loss", float("inf")))))

    if not candidates:
        return norm_model, 0, float("inf")
    return max(candidates, key=lambda item: item[1])


import uuid
def _download_range_chunk(url: str, token: str, start: int, end: int, fd: int, chunk_idx: int) -> int:
    headers = {
        "Authorization": f"Bearer {token}",
        "Range": f"bytes={start}-{end}",
        "User-Agent": "CerebroV5CloudSync/1.0",
    }
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
                expected = end - start + 1
                if len(data) != expected:
                    raise ValueError(f"Chunk {chunk_idx}: expected {expected} bytes, got {len(data)}")
                os.pwrite(fd, data, start)
                return len(data)
        except Exception as e:
            if attempt == 5:
                raise e
            time.sleep(1.0 * (1.5 ** attempt))
    return 0


def pull_checkpoint(
    local_dest: str,
    target_type: str = "latest",
    model: Optional[str] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
    show_progress: bool = True,
    num_workers: int = 4,
) -> bool:
    """
    Downloads latest.pt or best.pt for the specified model tier from bioflo.in.
    Returns True if successfully downloaded and integrity verified.
    """
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN
    model = normalize_model_name(model)

    os.makedirs(os.path.dirname(os.path.abspath(local_dest)), exist_ok=True)
    download_url = f"{sync_url}?action=download&type={target_type}&token={sync_token}"
    if model:
        download_url += f"&model={model}"
    tmp_dest = local_dest + ".part"

    try:
        model_str = f"[{model}] " if model else ""
        print(f"🔄 [cloud_sync] Downloading {model_str}{target_type}.pt from {sync_url}...")
        t0 = time.time()

        # Probe total size
        total_bytes = 0
        try:
            probe_req = urllib.request.Request(
                download_url,
                headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroV5CloudSync/1.0", "Range": "bytes=0-0"}
            )
            with urllib.request.urlopen(probe_req, timeout=15) as p_resp:
                cr = p_resp.headers.get("Content-Range", "")
                if "/" in cr:
                    total_bytes = int(cr.split("/")[-1])
        except Exception:
            pass

        if total_bytes > 5 * 1024 * 1024:
            chunk_size = 8 * 1024 * 1024  # 8 MB chunks to minimize TLS connection churn & avoid SSL reset drops
            ranges = []
            start = 0
            while start < total_bytes:
                end = min(start + chunk_size - 1, total_bytes - 1)
                ranges.append((start, end))
                start = end + 1

            fd = os.open(tmp_dest, os.O_CREAT | os.O_RDWR | os.O_TRUNC)
            try:
                os.ftruncate(fd, total_bytes)
                completed_bytes = 0
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(_download_range_chunk, download_url, sync_token, r[0], r[1], fd, idx)
                        for idx, r in enumerate(ranges)
                    ]
                    for fut in as_completed(futures):
                        completed_bytes += fut.result()
                        if show_progress:
                            pct = (completed_bytes / total_bytes) * 100
                            speed = (completed_bytes / 1e6) / max(time.time() - t0, 0.1)
                            print(f"\r  Downloading ({num_workers} threads): {completed_bytes / 1e6:.1f} / {total_bytes / 1e6:.1f} MB "
                                  f"({pct:.1f}%) | {speed:.1f} MB/s", end="", flush=True)
            finally:
                os.close(fd)
        else:
            downloaded = 0
            req = urllib.request.Request(download_url, headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroV5CloudSync/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_dest, "wb") as f:
                if total_bytes == 0:
                    total_bytes = int(resp.headers.get("Content-Length", 0))
                while True:
                    chunk = resp.read(2 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if show_progress and total_bytes > 0:
                        pct = (downloaded / total_bytes) * 100
                        speed = (downloaded / 1e6) / max(time.time() - t0, 0.1)
                        print(f"\r  Downloading: {downloaded / 1e6:.1f} / {total_bytes / 1e6:.1f} MB "
                              f"({pct:.1f}%) | {speed:.1f} MB/s", end="", flush=True)

        if os.path.exists(tmp_dest):
            actual_sz = os.path.getsize(tmp_dest)
            if total_bytes > 0 and actual_sz < total_bytes:
                print(f"\n❌ [cloud_sync] Incomplete download ({actual_sz:,} of {total_bytes:,} bytes).")
                if os.path.exists(tmp_dest):
                    os.remove(tmp_dest)
                return False

            if not zipfile.is_zipfile(tmp_dest):
                z_diag = ""
                try:
                    with zipfile.ZipFile(tmp_dest) as z_check:
                        z_check.testzip()
                except Exception as z_ex:
                    z_diag = f" ({z_ex})"
                print(f"\n⚠️ [cloud_sync] Checkpoint failed zip archive check{z_diag}. Discarding.")
                os.remove(tmp_dest)
                return False

            os.replace(tmp_dest, local_dest)
            dt = time.time() - t0
            sz_mb = os.path.getsize(local_dest) / 1e6
            print(f"\n✅ [cloud_sync] Restored {target_type}.pt ({sz_mb:.1f} MB) in {dt:.1f}s -> {local_dest}")
            return True
        return False
    except Exception as e:
        print(f"\n⚠️ [cloud_sync] Failed to pull {target_type} checkpoint: {e}")
        return False


def _ensure_bfloat16_checkpoint(file_path: str) -> None:
    """Ensures any PyTorch .pt checkpoint uploaded to cloud contains ONLY bare bfloat16 weights,
    stripping optimizer state ('opt_state'), master weights, and non-essential fluff.
    """
    if not file_path.endswith(".pt") or not os.path.exists(file_path):
        return
    try:
        import torch
    except ImportError:
        return

    try:
        try:
            ckpt = torch.load(file_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(file_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            return

        needs_rewrite = False

        # 1. Strip fluff: remove optimizer state if present
        if "opt_state" in ckpt:
            ckpt.pop("opt_state", None)
            needs_rewrite = True

        # 2. Strip any large non-essential objects in root
        allowed_root_keys = {
            "model_state", "state_dict", "params",
            "step", "loss", "eval_loss", "best_eval_loss",
            "model", "sync_model", "preset", "timestamp"
        }
        for k in list(ckpt.keys()):
            if k not in allowed_root_keys and not isinstance(ckpt[k], (int, float, str, bool)):
                if isinstance(ckpt[k], (dict, list)) and len(str(ckpt[k])) > 500:
                    ckpt.pop(k, None)
                    needs_rewrite = True

        # 3. Ensure all floating-point tensors are converted to bfloat16
        def _convert_dict_to_bf16(d: dict) -> Tuple[int, bool]:
            count = 0
            has_changes = False
            for k, v in list(d.items()):
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point() and v.dtype != torch.bfloat16:
                        d[k] = v.to(torch.bfloat16)
                        count += 1
                        has_changes = True
                elif isinstance(v, dict):
                    c, ch = _convert_dict_to_bf16(v)
                    count += c
                    if ch:
                        has_changes = True
            return count, has_changes

        target_keys = ["model_state", "state_dict", "params", "action_head_state", "action_head_state_dict"]
        is_nested = any(k in ckpt for k in target_keys)
        converted_count = 0

        if is_nested:
            for tk in target_keys:
                sub = ckpt.get(tk)
                if isinstance(sub, dict):
                    c, ch = _convert_dict_to_bf16(sub)
                    converted_count += c
                    if ch:
                        needs_rewrite = True
        else:
            c, ch = _convert_dict_to_bf16(ckpt)
            converted_count += c
            if ch:
                needs_rewrite = True

        if not needs_rewrite:
            return

        tmp_out = f"{file_path}.bare_bf16.{os.getpid()}"
        torch.save(ckpt, tmp_out)
        if zipfile.is_zipfile(tmp_out):
            os.replace(tmp_out, file_path)
            new_mb = os.path.getsize(file_path) / 1e6
            print(f"✅ [cloud_sync] Pruned fluff & converted {converted_count} tensors to bare bfloat16 ({new_mb:.1f} MB) in {os.path.basename(file_path)}")
        else:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
            print(f"⚠️ [cloud_sync] Bare bfloat16 conversion produced invalid zip archive; keeping original {file_path}")
    except Exception as e:
        print(f"⚠️ [cloud_sync] Warning during bare bfloat16 check: {e}")


_ensure_bare_bf16_checkpoint = _ensure_bfloat16_checkpoint


def _upload_single_chunk(
    upload_url: str,
    sync_token: str,
    upload_id: str,
    target_type: str,
    chunk_idx: int,
    total_chunks: int,
    step: int,
    loss: float,
    node_name: str,
    model: Optional[str],
    file_path: str,
    offset: int,
    length: int,
) -> int:
    """Reads a chunk slice at offset and uploads via multipart/form-data with idempotent retry."""
    with open(file_path, "rb") as f:
        f.seek(offset)
        chunk_data = f.read(length)

    boundary = f"----WebKitFormBoundary{os.urandom(8).hex()}"
    body = []

    form_fields = [
        ("action", "upload_chunk"),
        ("upload_id", upload_id),
        ("type", target_type),
        ("chunk_index", str(chunk_idx)),
        ("total_chunks", str(total_chunks)),
        ("step", str(step)),
        ("loss", f"{loss:.6f}"),
        ("source", node_name),
    ]
    if model:
        form_fields.append(("model", str(model)))

    for k, v in form_fields:
        body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8"))

    body.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"chunk\"; filename=\"chunk_{chunk_idx}.bin\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n".encode("utf-8")
    )
    body.append(chunk_data)
    body.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    payload = b"".join(body)
    req = urllib.request.Request(
        upload_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {sync_token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "CerebroV5CloudSync/1.0",
        },
        method="POST",
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                status = res_data.get("status")
                if status not in ("chunk_saved", "complete", "completed", "in_progress"):
                    raise RuntimeError(f"Server rejected chunk {chunk_idx}: {res_data}")
                return len(chunk_data)
        except urllib.error.HTTPError as he:
            err_detail = ""
            try:
                err_detail = f": {he.read().decode('utf-8', errors='ignore')}"
            except Exception:
                pass
            if attempt == 2:
                raise RuntimeError(f"HTTP {he.code} {he.reason}{err_detail}") from he
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(1.5 * (attempt + 1))
    return len(chunk_data)


def push_checkpoint(
    file_path: str,
    step: int,
    loss: float,
    target_type: str = "latest",
    model: Optional[str] = None,
    source_node: Optional[str] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
    chunk_size_mb: int = 35,
    num_workers: int = 4,
) -> bool:
    """
    Uploads a checkpoint to bioflo.in using parallel concurrent streaming for a specific model tier.
    Bypasses Cloudflare limits and ensures zero timeouts.
    """
    if not os.path.exists(file_path):
        print(f"⚠️ [cloud_sync] Cannot push: {file_path} not found.")
        return False

    # Ensure any float32 tensors are converted to bfloat16 before chunking and uploading
    _ensure_bfloat16_checkpoint(file_path)

    # Pre-upload archive verification: never push a corrupted or truncated archive
    if not zipfile.is_zipfile(file_path):
        print(f"⚠️ [cloud_sync] Pre-upload verification failed: {file_path} is not a valid zip archive. Aborting push.")
        if ".upload." in file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        return False

    model = normalize_model_name(model)
    slot_key = f"{model}_{target_type}" if model else target_type

    global _IN_FLIGHT_UPLOADS, _IN_FLIGHT_UPLOAD_TIMES, _LAST_BEST_UPLOAD_TIME
    with _UPLOAD_LOCK:
        now = time.time()
        if target_type == "best":
            elapsed = now - _LAST_BEST_UPLOAD_TIME
            if elapsed < _BEST_COOLDOWN_SECONDS and _IN_FLIGHT_UPLOADS.get(slot_key) != step:
                return False

        if slot_key in _IN_FLIGHT_UPLOADS and _IN_FLIGHT_UPLOADS[slot_key] != step:
            start_t = _IN_FLIGHT_UPLOAD_TIMES.get(slot_key, now)
            if now - start_t < 180.0:
                return False

        _IN_FLIGHT_UPLOADS[slot_key] = step
        _IN_FLIGHT_UPLOAD_TIMES[slot_key] = now
        if target_type == "best":
            _LAST_BEST_UPLOAD_TIME = now

    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN
    model = normalize_model_name(model)
    node_name = source_node or detect_source_environment()

    file_size = os.path.getsize(file_path)
    chunk_size = chunk_size_mb * 1024 * 1024
    total_chunks = max(1, (file_size + chunk_size - 1) // chunk_size)
    workers = min(num_workers, total_chunks)

    model_label = f"[{model}] " if model else ""
    print(f"\n🌐 [cloud_sync] Mirroring {model_label}{target_type}.pt (Step {step:,}, Loss {loss:.4f}) "
          f"to {sync_url} across {total_chunks} chunk(s) ({workers} worker threads)...")

    upload_id = uuid.uuid4().hex[:12]
    upload_url = f"{sync_url}?action=upload_chunk&token={sync_token}"
    t0 = time.time()
    try:
        chunk_ranges = []
        offset = 0
        for idx in range(total_chunks):
            length = min(chunk_size, file_size - offset)
            chunk_ranges.append((idx, offset, length))
            offset += length

        uploaded_bytes = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _upload_single_chunk,
                    upload_url,
                    sync_token,
                    upload_id,
                    target_type,
                    idx,
                    total_chunks,
                    step,
                    loss,
                    node_name,
                    model,
                    file_path,
                    off,
                    lg,
                )
                for idx, off, lg in chunk_ranges
            ]
            for fut in as_completed(futures):
                uploaded_bytes += fut.result()
                pct = (uploaded_bytes / file_size) * 100
                speed = (uploaded_bytes / 1e6) / max(time.time() - t0, 0.1)
                print(f"\r  Uploading ({workers} threads): {uploaded_bytes / 1e6:.1f} / {file_size / 1e6:.1f} MB "
                      f"({pct:.1f}%) | {speed:.1f} MB/s", end="", flush=True)

        dt = time.time() - t0
        sz_mb = file_size / 1e6
        speed_mb_s = sz_mb / max(dt, 0.01)
        print(f"\n✅ [cloud_sync] Successfully mirrored {model_label}{target_type}.pt ({sz_mb:.1f} MB in {dt:.1f}s | {speed_mb_s:.1f} MB/s) to bioflo.in")
        return True
    except Exception as e:
        print(f"\n⚠️ [cloud_sync] Upload failed for {model_label}{target_type}.pt: {e}")
        return False
    finally:
        with _UPLOAD_LOCK:
            _IN_FLIGHT_UPLOADS.pop(slot_key, None)
            _IN_FLIGHT_UPLOAD_TIMES.pop(slot_key, None)
        if ".upload." in file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass


def push_checkpoint_async(
    file_path: str,
    step: int,
    loss: float,
    target_type: str = "latest",
    model: Optional[str] = None,
    source_node: Optional[str] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
) -> threading.Thread:
    """Non-blocking background thread upload so training compute never stalls."""
    thread = threading.Thread(
        target=push_checkpoint,
        kwargs={
            "file_path": file_path,
            "step": step,
            "loss": loss,
            "target_type": target_type,
            "model": model,
            "source_node": source_node,
            "url": url,
            "token": token,
        },
        daemon=True,
        name=f"cloud_sync_upload_{model or 'default'}_{target_type}_{step}",
    )
    with _UPLOAD_LOCK:
        _ACTIVE_UPLOAD_THREADS.append(thread)
    thread.start()
    return thread


def notify_training_start(
    step: int = 0,
    model: Optional[str] = None,
    source_node: Optional[str] = None,
    hours: float = 8.5,
    message: Optional[str] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Notifies the central hub on bioflo.in that training has started."""
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN
    model = normalize_model_name(model)
    node_name = source_node or detect_source_environment()
    model_name = model or "v5_small"
    msg = message or f"🚀 CerebroSpheroid {model_name} training active on {node_name}"

    params = urllib.parse.urlencode({
        "action": "notify",
        "model": model_name,
        "source": node_name,
        "step": str(step),
        "hours": str(hours),
        "message": msg,
    })
    req = urllib.request.Request(
        f"{sync_url}?{params}",
        headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroV5CloudSync/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("success", False)
    except Exception as e:
        print(f"⚠️ [cloud_sync] Could not send start notification: {e}")
        return False


def fetch_remote_hardware_config(
    model: str,
    device_signature: str,
    url: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieves cached optimal batch size, remat, and throughput config for hardware from bioflo.in."""
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN
    clean_model = normalize_model_name(model) or model

    req_url = f"{sync_url}?action=get_config&model={clean_model}&device={urllib.parse.quote(device_signature)}"
    req = urllib.request.Request(
        req_url,
        headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroV5CloudSync/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") == "found" and "config" in data:
                return data["config"]
    except Exception as e:
        print(f"ℹ️ [cloud_sync] Could not retrieve cached hardware config from {sync_url}: {e}")
    return None


def push_remote_hardware_config(
    model: str,
    device_signature: str,
    config: Dict[str, Any],
    source_node: Optional[str] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Uploads benchmarked optimal hyperparameters to bioflo.in for zero-delay reuse."""
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN
    clean_model = normalize_model_name(model) or model
    source = source_node or detect_source_environment()

    payload = {
        "action": "save_config",
        "model": clean_model,
        "device": device_signature,
        "batch_size": int(config.get("batch_size", 16)),
        "remat": bool(config.get("remat", True)),
        "seq_len": int(config.get("seq_len", 256)),
        "dtype": str(config.get("dtype", "auto")),
        "tok_per_sec": float(config.get("tok_per_sec", 0.0)),
        "step_ms": float(config.get("step_ms", 0.0)),
        "vram_gb": float(config.get("vram_gb", 0.0)),
        "vram_utilization_pct": float(config.get("vram_utilization_pct", 0.0)),
        "source": source,
    }

    try:
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{sync_url}?action=save_config",
            data=req_data,
            headers={
                "Authorization": f"Bearer {sync_token}",
                "Content-Type": "application/json",
                "User-Agent": "CerebroV5CloudSync/1.0",
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return bool(res.get("success", False))
    except Exception as e:
        print(f"⚠️ [cloud_sync] Failed to save hardware config to {sync_url}: {e}")
        return False


def push_remote_loss_history(
    model: str,
    records: List[Any],
    url: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Uploads/syncs loss records over time to bioflo.in website data repository."""
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN
    clean_model = normalize_model_name(model) or model

    serialized = []
    for r in records:
        if hasattr(r, "to_dict"):
            serialized.append(r.to_dict())
        elif isinstance(r, dict):
            serialized.append(r)

    payload = {
        "action": "save_loss_history",
        "model": clean_model,
        "records": serialized,
    }

    try:
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{sync_url}?action=save_loss_history",
            data=req_data,
            headers={
                "Authorization": f"Bearer {sync_token}",
                "Content-Type": "application/json",
                "User-Agent": "CerebroV5CloudSync/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return bool(data.get("success", False))
    except Exception as e:
        print(f"ℹ️ [cloud_sync] Could not sync loss history to {sync_url}: {e}")
        return False


def fetch_remote_loss_history(
    model: str,
    url: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Retrieves recorded training loss progression over time from bioflo.in website data repository."""
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN
    clean_model = normalize_model_name(model) or model

    req_url = f"{sync_url}?action=get_loss_history&model={clean_model}&token={sync_token}"
    req = urllib.request.Request(
        req_url,
        headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroV5CloudSync/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") == "found" and "records" in data:
                return data["records"]
    except Exception as e:
        print(f"ℹ️ [cloud_sync] Could not retrieve remote loss history from {sync_url}: {e}")
    return None


def push_remote_token_tracker(
    model: str,
    cumulative_tokens: int,
    target_tokens: int = 200_000_000_000,
    tokens_by_dataset: Optional[Dict[str, int]] = None,
    effective_batch_size: int = 256,
    step: int = 0,
    loss: float = 0.0,
    tok_per_sec: float = 0.0,
    run_id: Optional[str] = None,
    force_reset: bool = False,
    url: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """
    Pushes real-time token ingestion counters and per-dataset breakdowns to bioflo.in,
    strictly linking token telemetry to both model tier and unique run ID.
    """
    clean_model = normalize_model_name(model) or model
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN

    payload = {
        "action": "save_tokens",
        "model": clean_model,
        "run_id": str(run_id) if run_id else "default",
        "cumulative_tokens": int(cumulative_tokens),
        "target_tokens": int(target_tokens),
        "tokens_by_dataset": tokens_by_dataset or {},
        "effective_batch_size": int(effective_batch_size),
        "step": int(step),
        "loss": float(loss),
        "tok_per_sec": float(tok_per_sec),
        "timestamp": time.time(),
        "source": detect_source_environment(),
        "force_reset": bool(force_reset),
    }

    try:
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{sync_url}?action=save_tokens",
            data=req_data,
            headers={
                "Authorization": f"Bearer {sync_token}",
                "Content-Type": "application/json",
                "User-Agent": "CerebroV5CloudSync/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return bool(data.get("success", False))
    except Exception:
        return False


def fetch_remote_token_tracker(
    model: str,
    run_id: Optional[str] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves current token ingestion metrics and per-dataset counts from bioflo.in for a specific model and run.
    """
    clean_model = normalize_model_name(model) or model
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN

    req_url = f"{sync_url}?action=get_tokens&model={clean_model}"
    if run_id:
        req_url += f"&run_id={urllib.parse.quote(str(run_id))}"
    req = urllib.request.Request(
        req_url,
        headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroV5CloudSync/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("data", {})
    except Exception:
        return {}


def clear_remote_tokens(
    model: Optional[str] = None,
    run_id: Optional[str] = None,
    clear_checkpoints: bool = False,
    url: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Clears remote token counts, loss history, and optionally checkpoints for a model and run on bioflo.in."""
    clean_model = normalize_model_name(model) if model else ""
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN

    req_url = f"{sync_url}?action=clear"
    if clean_model:
        req_url += f"&model={clean_model}"
    else:
        req_url += "&all=true"
    if run_id:
        req_url += f"&run_id={urllib.parse.quote(str(run_id))}"
    if clear_checkpoints:
        req_url += "&checkpoints=true"

    req = urllib.request.Request(
        req_url,
        headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroV5CloudSync/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"⚠️ [cloud_sync] Could not clear tokens on {sync_url}: {e}")
        return {"success": False, "error": str(e)}
