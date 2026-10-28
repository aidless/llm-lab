"""Managed serving process with bounded startup and log evidence."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import OpenAICompatibleClient, classify_provider_error
from .commands import build_serve_command, shell_join
from .models import InfraError, LaunchConfig, LaunchError


@dataclass
class ManagedServer:
    config: LaunchConfig
    log_dir: Path
    startup_timeout_seconds: float = 120.0
    poll_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        self.process: subprocess.Popen[bytes] | None = None
        self.command = build_serve_command(self.config)
        self.stdout_path = self.log_dir / "server.stdout.log"
        self.stderr_path = self.log_dir / "server.stderr.log"

    def start(self) -> dict[str, Any]:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stdout_handle = self.stdout_path.open("ab")
        stderr_handle = self.stderr_path.open("ab")
        try:
            self.process = subprocess.Popen(  # noqa: S603
                self.command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise LaunchError(f"backend executable not found: {self.command[0]}") from exc
        finally:
            stdout_handle.close()
            stderr_handle.close()
        return {
            "pid": self.process.pid,
            "command": shell_join(self.command),
            "stdout_log": str(self.stdout_path),
            "stderr_log": str(self.stderr_path),
        }

    def wait_ready(self) -> dict[str, Any]:
        if self.process is None:
            raise LaunchError("server has not been started")
        started = time.perf_counter()
        client = OpenAICompatibleClient(
            base_url=self.config.base_url,
            timeout_seconds=max(1.0, min(10.0, self.poll_interval_seconds * 5)),
        )
        try:
            while time.perf_counter() - started < self.startup_timeout_seconds:
                return_code = self.process.poll()
                if return_code is not None:
                    detail = self._log_tail()
                    error_class = classify_process_log(detail)
                    raise LaunchError(f"server exited during startup with code {return_code} ({error_class}): {detail}")
                try:
                    status, _latency = client.health()
                    if 200 <= status < 300:
                        models = client.models()
                        return {
                            "ready": True,
                            "health_status": status,
                            "models": list(models),
                            "startup_seconds": time.perf_counter() - started,
                        }
                except InfraError:
                    pass
                time.sleep(self.poll_interval_seconds)
        finally:
            client.close()
        detail = self._log_tail()
        raise LaunchError(
            f"server was not ready within {self.startup_timeout_seconds:g}s ({classify_process_log(detail)}): {detail}"
        )

    def stop(self) -> dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            return {"stopped": True, "return_code": None if self.process is None else self.process.returncode}
        assert self.process is not None
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        try:
            return_code = self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            return_code = self.process.wait(timeout=15)
        return {"stopped": True, "return_code": return_code}

    def _log_tail(self, limit: int = 4000) -> str:
        chunks: list[str] = []
        for path in (self.stderr_path, self.stdout_path):
            if path.exists():
                chunks.append(path.read_text(encoding="utf-8", errors="replace")[-limit:])
        return "\n".join(chunk for chunk in chunks if chunk).strip()


def classify_process_log(text: str) -> str:
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "out of memory",
            "outofmemoryerror",
            "cuda oom",
            "no available memory",
        )
    ):
        return "oom"
    if any(marker in lowered for marker in ("address already in use", "eaddrinuse")):
        return "port_in_use"
    if "model not found" in lowered or "does not exist" in lowered:
        return "model_not_found"
    if "permission denied" in lowered:
        return "permission_denied"
    return classify_provider_error(lowered)
