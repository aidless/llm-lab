"""Command-line interface for backend command, probe and diagnosis contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import OpenAICompatibleClient
from .commands import build_serve_command, shell_join
from .diagnostics import host_diagnostics
from .models import LaunchConfig
from .opencompass import export_opencompass_bundle


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def command_diagnose(_args: argparse.Namespace) -> int:
    _print(host_diagnostics())
    return 0


def command_serve(args: argparse.Namespace) -> int:
    config = _launch_config(args)
    _print(
        {
            "backend": config.backend,
            "command": build_serve_command(config),
            "shell": shell_join(build_serve_command(config)),
            "base_url": config.base_url,
            "openai_base_url": config.openai_base_url,
            "executed": False,
        }
    )
    return 0


def command_probe(args: argparse.Namespace) -> int:
    client = OpenAICompatibleClient(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=args.timeout,
    )
    try:
        result = client.probe(
            backend=args.backend,
            model=args.model or None,
            prompt=args.prompt,
        )
    finally:
        client.close()
    _print(result.to_dict())
    return 0 if result.ok else 2


def command_export_opencompass(args: argparse.Namespace) -> int:
    report = export_opencompass_bundle(
        Path(args.output_dir),
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        allow_non_loopback_endpoint=args.allow_non_loopback_endpoint,
    )
    _print(report)
    return 0


def _launch_config(args: argparse.Namespace) -> LaunchConfig:
    return LaunchConfig(
        backend=args.backend,
        model=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tp,
        served_model_name=args.served_model_name,
        extra_args=tuple(args.extra),
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="eval-infra")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("diagnose").set_defaults(func=command_diagnose)

    serve = sub.add_parser("serve-command")
    serve.add_argument("--backend", choices=("vllm", "lmdeploy"), required=True)
    serve.add_argument("--model", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--tp", type=int, default=1)
    serve.add_argument("--served-model-name", default=None)
    serve.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    serve.set_defaults(func=command_serve)

    probe = sub.add_parser("probe")
    probe.add_argument("--backend", choices=("vllm", "lmdeploy"), default="vllm")
    probe.add_argument("--base-url", required=True)
    probe.add_argument("--model", default="")
    probe.add_argument("--api-key", default="local")
    probe.add_argument("--timeout", type=float, default=30.0)
    probe.add_argument("--prompt", default="Reply with OK.")
    probe.set_defaults(func=command_probe)

    opencompass_export = sub.add_parser("opencompass-export")
    opencompass_export.add_argument(
        "--output-dir",
        default="opencompass-export",
    )
    opencompass_export.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
    )
    opencompass_export.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    opencompass_export.add_argument(
        "--api-key-env",
        default="EVAL_INFRA_OPENAI_API_KEY",
    )
    opencompass_export.add_argument(
        "--allow-non-loopback-endpoint",
        action="store_true",
    )
    opencompass_export.set_defaults(func=command_export_opencompass)
    return root


def entry() -> None:
    args = parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    entry()
