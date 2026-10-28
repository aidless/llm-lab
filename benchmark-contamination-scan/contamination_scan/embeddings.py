"""Local Ollama embeddings with cacheable vectors."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import httpx


class Embedder:
    def __init__(
        self,
        *,
        model: str = "nomic-embed-text:latest",
        base_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.client = httpx.Client(timeout=120, follow_redirects=False)

    def close(self) -> None:
        self.client.close()

    def encode(self, texts: list[str], *, batch_size: int = 16) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": batch,
                    "truncate": True,
                    "keep_alive": "10m",
                },
            )
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise ValueError("Ollama returned an invalid embeddings payload")
            vectors.extend([float(value) for value in vector] for vector in embeddings)
        return vectors


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def save_vectors(path: Path, vectors: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(vectors, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_vectors(path: Path) -> list[list[float]]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("embedding cache must be a list")
    return [[float(value) for value in row] for row in payload]
