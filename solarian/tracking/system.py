"""What machine a run is on: the Overview of a run page."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def _command() -> str:
    # orig_argv keeps `-m package.module`, which is what reproduces the run.
    argv = getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv]
    return shlex.join(argv[1:])


def _cpu_counts() -> tuple[int | None, int | None]:
    logical = os.cpu_count()
    try:
        import psutil

        return psutil.cpu_count(logical=False), logical
    except Exception:  # noqa: BLE001 - missing, or blocked by a sandbox
        return logical, logical


def _gpus() -> tuple[int, str | None]:
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                return count, torch.cuda.get_device_name(0) if count else None
        except Exception:  # noqa: BLE001 - a broken CUDA install is not a tracking error
            pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip().splitlines()
            return len(out), out[0].strip() if out else None
        except (subprocess.SubprocessError, OSError):
            pass
    return 0, None


def _git() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    commit = run("rev-parse", "HEAD")
    if not commit:
        return {}
    return {
        "commit": commit,
        "remote": run("config", "--get", "remote.origin.url"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def system_info() -> dict[str, Any]:
    physical, logical = _cpu_counts()
    gpu_count, gpu_type = _gpus()
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "executable": sys.executable,
        "command": _command(),
        "cwd": os.getcwd(),
        "cpu_count": physical,
        "logical_cpu_count": logical,
        "gpu_count": gpu_count,
        "gpu_type": gpu_type,
        "solarian_version": _version("solarian"),
        "git": _git(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
