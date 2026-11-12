#!/usr/bin/env python3
"""CerebroSpheroid V5: Kaggle TPU v5e-8 Launcher & Live Telemetry Watcher.

Reverse-engineered from high-throughput Kaggle TPU runners (kaggle-tpu-lab):
- Direct Kaggle CLI kernel lifecycle management (push, status, watch, stop)
- Real-time out-of-band telemetry via ntfy.sh (bypasses Kaggle log buffering)
- Clean queue eviction for stale zombie sessions

Usage:
    python scripts/launch_kaggle_tpu.py push          # Push TPU kernel and watch boot
    python scripts/launch_kaggle_tpu.py status        # One-shot status of Kaggle TPU kernel
    python scripts/launch_kaggle_tpu.py watch         # Watch live queue and execution telemetry
    python scripts/launch_kaggle_tpu.py stop          # Terminate / evict TPU session
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TPU_DIR = REPO_ROOT / "kaggle" / "tpu"
KERNEL_SLUG = "ajitsheoran/cerebrospheroid-v5-tpu-v5e-8-training"
DEFAULT_TOPIC = "cs5-tpu-ajitsheoran"


def log(msg: str):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def kaggle_cmd(*args: str, capture: bool = True):
    cmd = [sys.executable, "-m", "kaggle", *args]
    return subprocess.run(cmd, capture_output=capture, text=True)


def check_auth():
    r = kaggle_cmd("kernels", "list", "-m", "--page-size", "1")
    if r.returncode != 0:
        sys.exit(
            "❌ Kaggle CLI is not authenticated or not installed.\n"
            "Ensure ~/.kaggle/kaggle.json exists and has valid credentials.\n"
            f"Output: {(r.stderr or r.stdout).strip()}"
        )


def get_kernel_status(slug: str = KERNEL_SLUG) -> str:
    r = kaggle_cmd("kernels", "status", slug)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r'"KernelWorkerStatus\.(\w+)"', out)
    if m:
        return m.group(1)
    if "RUNNING" in out:
        return "RUNNING"
    if "QUEUED" in out:
        return "QUEUED"
    if "COMPLETE" in out:
        return "COMPLETE"
    if "ERROR" in out:
        return "ERROR"
    return "UNKNOWN"


def read_ntfy_events(topic: str, since: int):
    try:
        url = f"https://ntfy.sh/{topic}/json?poll=1&since={since}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read().decode("utf-8")
    except Exception:
        return []

    events = []
    for line in body.splitlines():
        try:
            item = json.loads(line)
            if item.get("event") == "message":
                msg = json.loads(item.get("message", "{}"))
                events.append((item.get("time", int(time.time())), msg))
        except Exception:
            continue
    return events


def cmd_status(args):
    check_auth()
    st = get_kernel_status(KERNEL_SLUG)
    log(f"Kaggle TPU Kernel [{KERNEL_SLUG}]: {st}")
    if st == "QUEUED":
        log("⏳ Kernel is currently waiting in the Kaggle TPU v5e-8 queue lineup.")
    elif st == "RUNNING":
        log("⚡ Kernel is currently running on TPU v5e-8!")
    elif st in ("COMPLETE", "CANCEL_ACKNOWLEDGED", "CANCELACKNOWLEDGED"):
        log(f"⏹️ Kernel session is inactive ({st}).")
    elif st == "ERROR":
        log("⚠️ Kernel encountered an error.")


def cmd_stop(args):
    check_auth()
    log(f"Stopping/evicting Kaggle TPU session for {KERNEL_SLUG}...")
    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "delete", KERNEL_SLUG],
        input="yes\n",
        capture_output=True,
        text=True,
    )
    out = (r.stdout or "") + (r.stderr or "")
    log(f"Result: {out.strip() or 'Evicted'}")


def cmd_watch(args):
    check_auth()
    topic = args.topic or DEFAULT_TOPIC
    log(f"Watching {KERNEL_SLUG} | ntfy channel: https://ntfy.sh/{topic}")

    last_status = None
    since = int(time.time()) - 300

    try:
        while True:
            # Poll ntfy telemetry
            for ts, ev in read_ntfy_events(topic, since):
                since = max(since, ts)
                phase = ev.get("phase", "event")
                log(f"📡 [TPU VM Event] {phase.upper()}: {json.dumps({k: v for k, v in ev.items() if k != 'phase'})}")

            # Poll Kaggle status
            st = get_kernel_status(KERNEL_SLUG)
            if st != last_status:
                log(f"🔄 Kaggle Worker Status changed -> {st}")
                last_status = st
                if st in ("COMPLETE", "CANCEL_ACKNOWLEDGED", "CANCELACKNOWLEDGED", "ERROR"):
                    log(f"Session finished with {st}.")
                    break

            time.sleep(15)
    except KeyboardInterrupt:
        log("Detached watcher. The Kaggle TPU kernel continues in the background.")


def cmd_push(args):
    check_auth()
    if not TPU_DIR.is_dir():
        sys.exit(f"❌ TPU directory missing at {TPU_DIR}")

    log(f"📦 Pushing CerebroSpheroid V5 TPU kernel from {TPU_DIR} ...")
    r = kaggle_cmd("kernels", "push", "-p", str(TPU_DIR))
    out = (r.stdout or "") + (r.stderr or "")
    if "successfully pushed" not in out and "Kernel version" not in out:
        sys.exit(f"❌ Push failed:\n{out.strip()}")

    log(f"✅ {out.strip()}")
    log(f"🔗 Inspect on Kaggle: https://www.kaggle.com/code/{KERNEL_SLUG}")

    if not args.no_watch:
        cmd_watch(args)


def main():
    parser = argparse.ArgumentParser(description="CerebroSpheroid V5: Kaggle TPU v5e-8 Launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    push_p = subparsers.add_parser("push", help="Push TPU kernel to Kaggle")
    push_p.add_argument("--no-watch", action="store_true", help="Don't watch after pushing")
    push_p.add_argument("--topic", default=DEFAULT_TOPIC, help="ntfy.sh topic for telemetry")

    status_p = subparsers.add_parser("status", help="Check Kaggle TPU kernel status")

    watch_p = subparsers.add_parser("watch", help="Watch live queue & execution telemetry")
    watch_p.add_argument("--topic", default=DEFAULT_TOPIC, help="ntfy.sh topic for telemetry")

    stop_p = subparsers.add_parser("stop", help="Stop / evict Kaggle TPU kernel session")

    args = parser.parse_args()
    if args.command == "push":
        cmd_push(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "stop":
        cmd_stop(args)


if __name__ == "__main__":
    main()
