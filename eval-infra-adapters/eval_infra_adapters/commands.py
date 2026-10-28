"""Deterministic command construction for supported serving backends."""

from __future__ import annotations

from .models import LaunchConfig


def build_serve_command(config: LaunchConfig) -> list[str]:
    """Return the documented current CLI command without executing it."""
    if config.backend == "vllm":
        command = [
            "vllm",
            "serve",
            config.model,
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--tensor-parallel-size",
            str(config.tensor_parallel_size),
        ]
        if config.served_model_name:
            command.extend(["--served-model-name", config.served_model_name])
    else:
        command = [
            "lmdeploy",
            "serve",
            "api_server",
            config.model,
            "--server-name",
            config.host,
            "--server-port",
            str(config.port),
            "--tp",
            str(config.tensor_parallel_size),
        ]
        if config.served_model_name:
            command.extend(["--model-name", config.served_model_name])

    command.extend(config.extra_args)
    return command


def shell_join(command: list[str]) -> str:
    """Render a readable command without invoking a shell."""
    import shlex

    return shlex.join(command)
