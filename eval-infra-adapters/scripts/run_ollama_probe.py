#!/usr/bin/env python3
"""Probe the real local Ollama OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_infra_adapters.client import OpenAICompatibleClient  # noqa: E402

BASE_URL = "http://127.0.0.1:11434"
MODEL = "llama3.2:1b"


def main() -> int:
    evidence = ROOT / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    client = OpenAICompatibleClient(
        base_url=BASE_URL,
        timeout_seconds=120,
        health_path="/api/version",
    )
    try:
        result = client.probe(
            backend="ollama-openai-compatible",
            model=MODEL,
            prompt="Reply with OK.",
        )
    finally:
        client.close()
    native: dict[str, Any] = {}
    try:
        with httpx.Client(timeout=10.0) as http:
            version = http.get(f"{BASE_URL}/api/version")
            version.raise_for_status()
            tags = http.get(f"{BASE_URL}/api/tags")
            tags.raise_for_status()
        models = tags.json().get("models", [])
        selected = next(
            (
                item
                for item in models
                if isinstance(item, dict) and str(item.get("name")) == MODEL
            ),
            {},
        )
        native = {
            "ollama_version": str(version.json().get("version", "unknown")),
            "model": MODEL,
            "model_digest": selected.get("digest"),
            "model_size_bytes": selected.get("size"),
            "model_details": selected.get("details"),
        }
    except (httpx.HTTPError, ValueError) as exc:
        native = {"error": f"{type(exc).__name__}: {exc}"}
    payload = {
        "schema_version": "eval-infra.ollama-probe.v1",
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ok": result.ok,
        "backend": "ollama-openai-compatible",
        "base_url": BASE_URL,
        "native": native,
        "probe": result.to_dict(),
        "execution_environment": "local CPU; not GPU/production throughput",
        "boundary": (
            "本地安全评测，不是生产安全系统。 This is a local CPU "
            "OpenAI-compatible contract probe, not a serving benchmark."
        ),
    }
    (evidence / "ollama-probe.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
