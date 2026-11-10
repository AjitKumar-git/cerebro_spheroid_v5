"""
CerebroSpheroid Cloud Synchronization Client.
Enables transparent, continuous checkpoint resumption across Google Colab, Kaggle, and Local Workstations.

Endpoints:
- Default Sync URL: https://bioflo.in/api/cerebro/sync.php
- Storage Policy: Strictly bounded to latest.pt and best.pt (capped at ~900 MB total server space).
- Chunked Upload: Bypasses Cloudflare 100 MB limits by streaming 10 MB chunks.
"""

import os
import sys
import time
import json
import socket
import threading
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Tuple

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
_LAST_BEST_UPLOAD_TIME: float = 0.0
_BEST_COOLDOWN_SECONDS: float = 300.0  # 5 minutes minimum between best.pt uploads
_UPLOAD_LOCK = threading.Lock()


def detect_source_environment() -> str:
    """Detects whether current execution is in Google Colab, Kaggle, or Local."""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ or os.path.exists("/content"):
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


def check_remote_manifest(url: Optional[str] = None, token: Optional[str] = None) -> Dict[str, Any]:
    """Queries bioflo.in to retrieve checkpoint status, step numbers, and loss metrics."""
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN

    req = urllib.request.Request(
        f"{sync_url}?action=status",
        headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroCloudSync/1.0"}
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
        "User-Agent": "CerebroCloudSync/1.0",
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
    Downloads latest.pt or best.pt from bioflo.in with high-speed multi-threaded
    HTTP Range streaming and automatic chunk integrity checks.
    Returns True if successfully downloaded and verified.
    """
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN

    os.makedirs(os.path.dirname(os.path.abspath(local_dest)), exist_ok=True)
    download_url = f"{sync_url}?action=download&type={target_type}"

    tmp_dest = local_dest + ".part"
    try:
        print(f"🔄 [cloud_sync] Downloading {target_type}.pt from {sync_url}...")
        t0 = time.time()

        # 1. Probe total size using HTTP Range
        total_bytes = 0
        try:
            probe_req = urllib.request.Request(
                download_url,
                headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroCloudSync/1.0", "Range": "bytes=0-0"}
            )
            with urllib.request.urlopen(probe_req, timeout=15) as p_resp:
                cr = p_resp.headers.get("Content-Range", "")
                if "/" in cr:
                    total_bytes = int(cr.split("/")[-1])
        except Exception:
            pass

        # Fast parallel download if server supports Range probing
        if total_bytes > 10 * 1024 * 1024:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            chunk_size = 2 * 1024 * 1024  # 2 MB chunks (reliable over varying network latencies)
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
            # Fallback sequential streaming for small files or non-range endpoints
            downloaded = 0
            req = urllib.request.Request(download_url, headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroCloudSync/1.0"})
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
                print(f"\n❌ [cloud_sync] Incomplete download ({actual_sz:,} of {total_bytes:,} bytes). Keeping .part for future resumption.")
                return False

            import zipfile
            if not zipfile.is_zipfile(tmp_dest):
                print(f"\n⚠️ [cloud_sync] Checkpoint failed zip archive integrity check. Discarding corrupt download.")
                os.remove(tmp_dest)
                return False

            os.replace(tmp_dest, local_dest)
            dt = time.time() - t0
            sz_mb = os.path.getsize(local_dest) / 1e6
            print(f"\n✅ [cloud_sync] Restored {target_type}.pt ({sz_mb:.1f} MB) in {dt:.1f}s -> {local_dest}")
            return True
        else:
            print(f"\n⚠️ [cloud_sync] Downloaded file not found.")
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
    chunk_size_mb: int = 10
) -> bool:
    """
    Uploads a checkpoint to bioflo.in using chunked streaming.
    Bypasses Cloudflare 100 MB limits and shared hosting PHP upload ceilings.
    Enforces a 5-minute wait between successive 'best.pt' uploads to minimize traffic.
    """
    if not os.path.exists(file_path):
        print(f"⚠️ [cloud_sync] Cannot push: {file_path} not found.")
        return False

    global _IN_FLIGHT_UPLOADS, _LAST_BEST_UPLOAD_TIME
    with _UPLOAD_LOCK:
        now = time.time()
        if target_type == "best":
            elapsed = now - _LAST_BEST_UPLOAD_TIME
            if elapsed < _BEST_COOLDOWN_SECONDS and _IN_FLIGHT_UPLOADS.get(target_type) != step:
                remaining = int(_BEST_COOLDOWN_SECONDS - elapsed)
                print(f"⏳ [cloud_sync] Best model cooldown active: skipping best.pt upload ({remaining}s remaining until next 5-min window).")
                return False

        if target_type in _IN_FLIGHT_UPLOADS and _IN_FLIGHT_UPLOADS[target_type] != step:
            prev_step = _IN_FLIGHT_UPLOADS[target_type]
            print(f"⏳ [cloud_sync] Upload in flight for {target_type}.pt (step {prev_step}): skipping duplicate step {step}.")
            return False

        _IN_FLIGHT_UPLOADS[target_type] = step
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

                # Multipart boundary assembly
                boundary = f"----WebKitFormBoundary{os.urandom(8).hex()}"
                body = []

                # Metadata fields
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

                # Chunk payload
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
                        "Content-Length": str(len(payload)),
                        "User-Agent": "CerebroCloudSync/1.0"
                    }
                )

                success = False
                for attempt in range(1, 6):
                    try:
                        with urllib.request.urlopen(req, timeout=240) as resp:
                            resp_json = json.loads(resp.read().decode("utf-8"))
                            if resp_json.get("success"):
                                success = True
                                break
                            else:
                                print(f"\n⚠️ [cloud_sync] Chunk {idx + 1}/{total_chunks} rejected (attempt {attempt}): {resp_json}")
                    except Exception as e:
                        print(f"\n⚠️ [cloud_sync] Chunk {idx + 1}/{total_chunks} failed (attempt {attempt}): {e}")
                    time.sleep(attempt * 2.0)

                # Explicit memory release
                del payload, body, chunk_data
                if not success:
                    print(f"\n❌ [cloud_sync] Aborted upload at chunk {idx + 1}/{total_chunks} after 3 attempts.")
                    return False
                pct = ((idx + 1) / total_chunks) * 100
                print(f"\r  Uploaded chunk {idx + 1}/{total_chunks} ({pct:.0f}%)", end="", flush=True)

        print()
        dt = time.time() - t0
        sz_mb = file_size / 1e6
        print(f"✅ [cloud_sync] Successfully uploaded {target_type}.pt ({sz_mb:.1f} MB) in {dt:.1f}s to bioflo.in!")
        if target_type == "best":
            _LAST_BEST_UPLOAD_TIME = time.time()
        return True
    finally:
        with _UPLOAD_LOCK:
            if _IN_FLIGHT_UPLOADS.get(target_type) == step:
                _IN_FLIGHT_UPLOADS.pop(target_type, None)


def push_checkpoint_async(
    file_path: str,
    step: int,
    loss: float,
    target_type: str = "latest",
    source_node: Optional[str] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
    chunk_size_mb: int = 10
):
    """Asynchronous background upload. 0 ms interruption to training loops."""
    global _LAST_BEST_UPLOAD_TIME
    with _UPLOAD_LOCK:
        now = time.time()
        if target_type == "best":
            elapsed = now - _LAST_BEST_UPLOAD_TIME
            if elapsed < _BEST_COOLDOWN_SECONDS:
                remaining = int(_BEST_COOLDOWN_SECONDS - elapsed)
                print(f"⏳ [cloud_sync] Best model cooldown active: skipping best.pt upload ({remaining}s remaining until next 5-min window).")
                return None

        if target_type in _IN_FLIGHT_UPLOADS:
            prev_step = _IN_FLIGHT_UPLOADS[target_type]
            print(f"⏳ [cloud_sync] Upload in flight for {target_type}.pt (step {prev_step}): skipping duplicate dispatch for step {step}.")
            return None

        _IN_FLIGHT_UPLOADS[target_type] = step
        if target_type == "best":
            _LAST_BEST_UPLOAD_TIME = now

    thread = threading.Thread(
        target=push_checkpoint,
        args=(file_path, step, loss, target_type, source_node, url, token, chunk_size_mb),
        daemon=True,
        name=f"CloudSync_{target_type}_{step}"
    )
    thread.start()
    return thread


def notify_training_start(
    source: str = "kaggle_tpu",
    step: int = 0,
    hours: float = 8.5,
    message: str = "",
    url: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """
    Sends an HTTP notification to bioflo.in indicating that a training session
    has commenced, which in turn records active_session and sends an alert.
    """
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN
    import urllib.parse
    payload = urllib.parse.urlencode({
        "action": "notify",
        "source": source,
        "step": str(step),
        "hours": str(hours),
        "message": message or f"CerebroSpheroid V4 training session started on {source} ({hours}h window).",
    }).encode("utf-8")

    req = urllib.request.Request(
        sync_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {sync_token}",
            "User-Agent": "CerebroCloudSync/1.0",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("success"):
                print(f"📢 [cloud_sync] Active training notice registered on bioflo.in ({source}, {hours}h).")
                return True
    except Exception as e:
        print(f"⚠️ [cloud_sync] Notice: Could not send start notification: {e}")
    return False


def auto_sync_checkpoints(
    local_dir: str = "checkpoints",
    url: Optional[str] = None,
    token: Optional[str] = None,
    resume_target: str = "latest"
) -> Optional[str]:
    """
    Checks remote status against local files.
    If remote checkpoint is newer, automatically downloads it.
    Can target either 'latest' (highest step) or 'best' (minimum validation loss).
    Returns path to the best available checkpoint file for resuming.
    """
    os.makedirs(local_dir, exist_ok=True)
    manifest_info = check_remote_manifest(url, token)
    remote_manifest = manifest_info.get("manifest", {})
    if not isinstance(remote_manifest, dict):
        remote_manifest = {}
    
    target_key = "best" if resume_target == "best" else "latest"
    remote_target = remote_manifest.get(target_key, {}) if isinstance(remote_manifest, dict) else {}
    remote_step = remote_target.get("step", 0) if isinstance(remote_target, dict) else 0

    # Inspect local checkpoint directory
    local_latest_step = -1
    local_latest_path = None

    import glob
    candidates = glob.glob(os.path.join(local_dir, "*.pt"))
    for cand in candidates:
        import re
        m = re.search(r"step_(\d+)", os.path.basename(cand))
        s = -1
        if m:
            s = int(m.group(1))
        else:
            base = cand[:-3] if cand.endswith(".pt") else cand
            for meta_cand in [base + ".meta.json", base + ".json", base.replace("_latest", "_state") + ".json", base.replace("_best", "_state") + ".json"]:
                if os.path.exists(meta_cand):
                    try:
                        with open(meta_cand, "r") as mf:
                            m_data = json.load(mf)
                            s = int(m_data.get("step", -1))
                            if s >= 0:
                                break
                    except Exception:
                        pass
        if s > local_latest_step:
            local_latest_step = s
            local_latest_path = cand

    print(f"🔍 [cloud_sync] Checkpoint Assessment ({target_key}): Local Step = {local_latest_step:,} | Remote Step = {remote_step:,}")

    target_local_file = os.path.join(local_dir, f"cerebro_spheroid_v4_{target_key}.pt")
    has_target = manifest_info.get(f"has_{target_key}", False)
    if (remote_step > local_latest_step or not os.path.exists(target_local_file)) and has_target:
        print(f"📥 [cloud_sync] Pulling {target_key} checkpoint from bioflo.in (Step {remote_step:,})...")
        success = pull_checkpoint(target_local_file, target_type=target_key, url=url, token=token)
        if success:
            loss_val = float(remote_target.get("loss", 7.34)) if isinstance(remote_target, dict) else 7.34
            meta_payload = {
                "step": remote_step,
                "loss": loss_val,
                "best": loss_val,
                "selected": loss_val,
                "source": remote_target.get("source", "bioflo"),
                "resumed_from_cloud": True,
            }
            for meta_ext in [".json", ".meta.json"]:
                try:
                    with open(target_local_file.replace(".pt", meta_ext), "w") as f:
                        json.dump(meta_payload, f, indent=2)
                except Exception:
                    pass
            return target_local_file

    if target_key == "best" and os.path.exists(target_local_file):
        return target_local_file

    return local_latest_path or (target_local_file if os.path.exists(target_local_file) else None)
