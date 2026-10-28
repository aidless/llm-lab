#!/usr/bin/env python3
"""Run a real localhost process/HTTP contract and write evidence."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from eval_infra_adapters.client import OpenAICompatibleClient  # noqa: E402
from eval_infra_adapters.commands import build_serve_command, shell_join  # noqa: E402
from eval_infra_adapters.diagnostics import host_diagnostics  # noqa: E402
from eval_infra_adapters.models import LaunchConfig  # noqa: E402
from eval_infra_adapters.process import ManagedServer  # noqa: E402


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_backend(
    backend: str,
    *,
    evidence: Path,
) -> dict[str, Any]:
    port = _free_port()
    config = LaunchConfig(
        backend=backend,  # type: ignore[arg-type]
        model="fixture-model",
        host="127.0.0.1",
        port=port,
        tensor_parallel_size=1,
    )
    log_dir = evidence / f"{backend}-server-logs"
    server = ManagedServer(
        config=config,
        log_dir=log_dir,
        startup_timeout_seconds=15,
        poll_interval_seconds=0.2,
    )
    server.command = [
        sys.executable,
        str(ROOT / "scripts" / "fake_openai_server.py"),
        "--port",
        str(port),
        "--backend",
        backend,
    ]
    started = time.perf_counter()
    process = server.start()
    ready = server.wait_ready()
    client = OpenAICompatibleClient(
        base_url=config.base_url,
        timeout_seconds=5,
    )
    try:
        probe = client.probe(
            backend=backend,
            model="fixture-model",
            prompt="Reply with OK.",
        )
    finally:
        client.close()
    stopped = server.stop()
    process = {
        key: value.replace(sys.executable, "<python>").replace(str(REPO_ROOT) + "/", "")
        if isinstance(value, str)
        else value
        for key, value in process.items()
    }
    process["stdout_log"] = f"{backend}-server-logs/server.stdout.log"
    process["stderr_log"] = f"{backend}-server-logs/server.stderr.log"
    return {
        "backend": backend,
        "generated_serve_command": build_serve_command(config),
        "generated_serve_command_shell": shell_join(build_serve_command(config)),
        "managed_process": {
            **process,
            **ready,
            **stopped,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "probe": probe.to_dict(),
        "logs": {
            key: str(value.relative_to(evidence))
            for key, value in {
                "stdout": server.stdout_path,
                "stderr": server.stderr_path,
            }.items()
        },
    }


def main() -> int:
    evidence = ROOT / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    environment = host_diagnostics()
    environment.update(
        {
            "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform_module": platform.platform(),
            "purpose": "localhost HTTP and managed-process contract smoke",
        }
    )
    (evidence / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results = [
        _probe_backend("vllm", evidence=evidence),
        _probe_backend("lmdeploy", evidence=evidence),
    ]
    report: dict[str, Any] = {
        "schema_version": "eval-infra.contract-smoke.v1",
        "ok": all(
            result["probe"]["ok"] and result["managed_process"]["ready"] and result["managed_process"]["stopped"]
            for result in results
        ),
        "backends": results,
        "boundary": (
            "本地安全评测，不是生产安全系统。 Local contract smoke, not a GPU "
            "serving benchmark or production security system."
        ),
        "limitations": [
            "This is a localhost OpenAI-compatible fixture, not a real vLLM or LMDeploy server.",
            "The host has no usable NVIDIA GPU and neither serving package was executed.",
            "Passing this smoke proves the adapter and process lifecycle only.",
        ],
    }
    (evidence / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Eval Infra Contract Smoke",
        "",
        f"- Overall: **{'PASS' if report['ok'] else 'FAIL'}**",
        f"- Backends exercised: **{', '.join(item['backend'] for item in results)}**",
        "- Claim boundary: localhost fixture + managed subprocess only.",
        "- 本地安全评测，不是生产安全系统。 Local contract smoke, not a GPU serving benchmark.",
        "",
        "## Backend results",
        "",
        "| Backend | Process ready | Probe | Generated command |",
        "|---|---:|---:|---|",
    ]
    for item in results:
        command = item["generated_serve_command_shell"].replace("|", "\\|")
        lines.append(
            f"| `{item['backend']}` | {item['managed_process']['ready']} | {item['probe']['ok']} | `{command}` |"
        )
    lines += [
        "",
        "## Non-claims",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
    ]
    (evidence / "report.md").write_text("\n".join(lines), encoding="utf-8")
    files = sorted(
        path
        for path in evidence.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "verification-result.json"}
    )
    (evidence / "SHA256SUMS").write_text(
        "\n".join(f"{_sha256(path)}  {path.relative_to(evidence).as_posix()}" for path in files) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
