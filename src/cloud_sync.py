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

DEFAULT_URL = os.environ.get("BIOFLO_SYNC_URL", "https://bioflo.in/api/cerebro/sync.php")
DEFAULT_TOKEN = os.environ.get("BIOFLO_SYNC_TOKEN", "CerebroSpheroid_Sync_Secure_2026")

_IN_FLIGHT_UPLOADS: Dict[str, int] = {}
_IN_FLIGHT_UPLOAD_TIMES: Dict[str, float] = {}
_LAST_BEST_UPLOAD_TIME: float = 0.0
_BEST_COOLDOWN_SECONDS: float = 300.0  # 5 minutes minimum between best.pt uploads
_UPLOAD_LOCK = threading.Lock()
_ACTIVE_UPLOAD_THREADS: list = []


def detect_source_environment() -> str:
    """Detects whether current execution is in Google Colab or Local."""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ or os.path.exists("/content"):
        return "colab"
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


def check_remote_manifest(url: Optional[str] = None, token: Optional[str] = None) -> Dict[str, Any]:
    """Queries bioflo.in to retrieve checkpoint status, step numbers, and loss metrics."""
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN

    req = urllib.request.Request(
        f"{sync_url}?action=status",
        headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroV5CloudSync/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"⚠️ [cloud_sync] Unable to reach sync hub ({sync_url}): {e}")
        return {"status": "offline", "has_latest": False, "has_best": False, "manifest": {}}


def _download_range_chunk(url: str, token: str, start: int, end: int, filepath: str, chunk_idx: int) -> int:
    headers = {
        "Authorization": f"Bearer {token}",
        "Range": f"bytes={start}-{end}",
        "User-Agent": "CerebroV5CloudSync/1.0",
    }
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                expected = end - start + 1
                if len(data) != expected:
                    raise ValueError(f"Chunk {chunk_idx}: expected {expected} bytes, got {len(data)}")
                with open(filepath, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                return len(data)
        except Exception as e:
            if attempt == 4:
                raise e
            time.sleep(1.0 + attempt)
    return 0


def pull_checkpoint(
    local_dest: str,
    target_type: str = "latest",
    url: Optional[str] = None,
    token: Optional[str] = None,
    show_progress: bool = True,
    num_workers: int = 8,
) -> bool:
    """
    Downloads latest.pt or best.pt from bioflo.in with parallel range streaming.
    Returns True if successfully downloaded and integrity verified.
    """
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN

    os.makedirs(os.path.dirname(os.path.abspath(local_dest)), exist_ok=True)
    download_url = f"{sync_url}?action=download&type={target_type}"
    tmp_dest = local_dest + ".part"

    try:
        print(f"🔄 [cloud_sync] Downloading {target_type}.pt from {sync_url}...")
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
            chunk_size = 2 * 1024 * 1024  # 2 MB chunks
            ranges = []
            start = 0
            while start < total_bytes:
                end = min(start + chunk_size - 1, total_bytes - 1)
                ranges.append((start, end))
                start = end + 1

            with open(tmp_dest, "wb") as f:
                f.truncate(total_bytes)

            completed_bytes = 0
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(_download_range_chunk, download_url, sync_token, r[0], r[1], tmp_dest, idx)
                    for idx, r in enumerate(ranges)
                ]
                for fut in as_completed(futures):
                    completed_bytes += fut.result()
                    if show_progress:
                        pct = (completed_bytes / total_bytes) * 100
                        speed = (completed_bytes / 1e6) / max(time.time() - t0, 0.1)
                        print(f"\r  Downloading ({num_workers} threads): {completed_bytes / 1e6:.1f} / {total_bytes / 1e6:.1f} MB "
                              f"({pct:.1f}%) | {speed:.1f} MB/s", end="", flush=True)
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
                return False

            if not zipfile.is_zipfile(tmp_dest):
                print(f"\n⚠️ [cloud_sync] Checkpoint failed zip archive check. Discarding.")
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


def push_checkpoint(
    file_path: str,
    step: int,
    loss: float,
    target_type: str = "latest",
    source_node: Optional[str] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
    chunk_size_mb: int = 10,
) -> bool:
    """
    Uploads a checkpoint to bioflo.in using chunked streaming.
    Bypasses Cloudflare limits and ensures zero timeouts.
    """
    if not os.path.exists(file_path):
        print(f"⚠️ [cloud_sync] Cannot push: {file_path} not found.")
        return False

    global _IN_FLIGHT_UPLOADS, _IN_FLIGHT_UPLOAD_TIMES, _LAST_BEST_UPLOAD_TIME
    with _UPLOAD_LOCK:
        now = time.time()
        if target_type == "best":
            elapsed = now - _LAST_BEST_UPLOAD_TIME
            if elapsed < _BEST_COOLDOWN_SECONDS and _IN_FLIGHT_UPLOADS.get(target_type) != step:
                return False

        if target_type in _IN_FLIGHT_UPLOADS and _IN_FLIGHT_UPLOADS[target_type] != step:
            prev_step = _IN_FLIGHT_UPLOADS[target_type]
            start_t = _IN_FLIGHT_UPLOAD_TIMES.get(target_type, now)
            if now - start_t < 180.0:
                return False

        _IN_FLIGHT_UPLOADS[target_type] = step
        _IN_FLIGHT_UPLOAD_TIMES[target_type] = now
        if target_type == "best":
            _LAST_BEST_UPLOAD_TIME = now

    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN
    node_name = source_node or detect_source_environment()

    file_size = os.path.getsize(file_path)
    chunk_size = chunk_size_mb * 1024 * 1024
    total_chunks = max(1, (file_size + chunk_size - 1) // chunk_size)

    print(f"\n🌐 [cloud_sync] Mirroring {target_type}.pt (Step {step:,}, Loss {loss:.4f}) "
          f"to {sync_url} across {total_chunks} chunk(s)...")

    t0 = time.time()
    try:
        with open(file_path, "rb") as f:
            for idx in range(total_chunks):
                chunk_data = f.read(chunk_size)
                if not chunk_data:
                    break

                boundary = f"----WebKitFormBoundary{os.urandom(8).hex()}"
                body = []

                for k, v in [
                    ("action", "upload_chunk"),
                    ("type", target_type),
                    ("chunk_index", str(idx)),
                    ("total_chunks", str(total_chunks)),
                    ("step", str(step)),
                    ("loss", f"{loss:.6f}"),
                    ("source", node_name),
                ]:
                    body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8"))

                body.append(
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"chunk\"; filename=\"chunk_{idx}.bin\"\r\n"
                    f"Content-Type: application/octet-stream\r\n\r\n".encode("utf-8")
                )
                body.append(chunk_data)
                body.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))

                payload = b"".join(body)
                req = urllib.request.Request(
                    sync_url,
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
                            if res_data.get("status") not in ("chunk_saved", "complete"):
                                raise RuntimeError(f"Server rejected chunk: {res_data}")
                            break
                    except Exception as e:
                        if attempt == 2:
                            raise e
                        time.sleep(2.0)

        dt = time.time() - t0
        sz_mb = file_size / 1e6
        print(f"✅ [cloud_sync] Successfully mirrored {target_type}.pt ({sz_mb:.1f} MB in {dt:.1f}s) to bioflo.in")
        return True
    except Exception as e:
        print(f"⚠️ [cloud_sync] Upload failed for {target_type}.pt: {e}")
        return False
    finally:
        with _UPLOAD_LOCK:
            _IN_FLIGHT_UPLOADS.pop(target_type, None)
            _IN_FLIGHT_UPLOAD_TIMES.pop(target_type, None)


def push_checkpoint_async(
    file_path: str,
    step: int,
    loss: float,
    target_type: str = "latest",
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
            "source_node": source_node,
            "url": url,
            "token": token,
        },
        daemon=True,
        name=f"cloud_sync_upload_{target_type}_{step}",
    )
    with _UPLOAD_LOCK:
        _ACTIVE_UPLOAD_THREADS.append(thread)
    thread.start()
    return thread
