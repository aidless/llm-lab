"""Host diagnosis without consuming GPU or launching a model."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from typing import Any


def _which(name: str) -> str | None:
    return shutil.which(name)


def _module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def host_diagnostics() -> dict[str, Any]:
    commands = {
        name: _which(name)
        for name in (
            "docker",
            "kubectl",
            "kind",
            "minikube",
            "nvidia-smi",
            "vllm",
            "lmdeploy",
            "ollama",
        )
    }
    gpu: dict[str, Any]
    if commands["nvidia-smi"]:
        result = subprocess.run(  # noqa: S603
            [commands["nvidia-smi"], "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu = {
            "available": result.returncode == 0,
            "returncode": result.returncode,
            "devices": result.stdout.strip().splitlines(),
            "stderr": result.stderr.strip(),
        }
    else:
        gpu = {"available": False, "error": "nvidia-smi not found"}
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "commands": commands,
        "python_modules": {name: _module(name) for name in ("vllm", "lmdeploy", "httpx")},
        "gpu": gpu,
        "limitations": [
            "Command and module presence do not prove a backend can serve a model.",
            "GPU availability does not authorize spending or remote resource use.",
        ],
    }
