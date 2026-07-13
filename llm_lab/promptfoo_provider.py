"""Promptfoo-style provider: YAML config, caching, retries — pure Python, no Node.js."""

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from llm_lab.pricing import estimate_cost as _estimate_cost
from llm_lab.worker import build_openai_client

_CACHE_DB = Path(os.getenv("PROMPTFOO_CACHE", str(Path(__file__).parent / ".promptfoo_cache.db")))
_MAX_RETRIES_DEFAULT = 3
_BASE_DELAY_DEFAULT = 1.0


def _cache_path() -> Path:
    return _CACHE_DB


def _ensure_cache():
    db = _cache_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            key TEXT PRIMARY KEY,
            response TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _cache_key(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}:::{prompt}".encode()).hexdigest()[:16]


def _cache_get(prompt: str, model: str) -> dict | None:
    try:
        conn = _ensure_cache()
        row = conn.execute("SELECT response FROM llm_cache WHERE key = ?", (_cache_key(prompt, model),)).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


def _cache_set(prompt: str, model: str, response: dict):
    try:
        conn = _ensure_cache()
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (key, response, created_at) VALUES (?, ?, ?)",
            (_cache_key(prompt, model), json.dumps(response), time.time()),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logging.warning("promptfoo cache write failed: %s", exc)


def _read_provider_config() -> dict:
    path = os.getenv("PROMPTFOO_CONFIG", "")
    if not path:
        return {}
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _build_client_from_config(cfg: dict[str, Any]):
    api_key: str = str(cfg.get("api_key") or os.getenv("LLM_API_KEY", "ollama"))
    base_url: str = str(cfg.get("base_url") or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    return build_openai_client(base_url, api_key)


def call_llm(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict:
    cfg = _read_provider_config()
    provider_id = cfg.get("id", "openai:chat")
    model = model or os.getenv("LLM_MODEL") or cfg.get("model", "gpt-4o")

    cached = _cache_get(prompt, model)
    if cached:
        cached["_from_cache"] = True
        return cached

    client = _build_client_from_config(cfg)
    last_error = None

    max_retries = int(os.getenv("PROMPTFOO_MAX_RETRIES", str(_MAX_RETRIES_DEFAULT)))
    base_delay = float(os.getenv("PROMPTFOO_BASE_DELAY", str(_BASE_DELAY_DEFAULT)))

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = resp.choices[0]
            usage = resp.usage
            tokens = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            result = {
                "output": choice.message.content or "",
                "model": model,
                "finish_reason": choice.finish_reason,
                "token_usage": tokens,
                "cost_usd": round(_estimate_cost(model, usage.prompt_tokens, usage.completion_tokens), 6),
                "provider": provider_id,
            }
            _cache_set(prompt, model, result)
            return result

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                time.sleep(delay)

    return {
        "output": f"[promptfoo/{provider_id}] LLM call failed after {max_retries} retries: {last_error}",
        "model": model,
        "finish_reason": "error",
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "cost_usd": 0.0,
        "provider": provider_id,
    }
