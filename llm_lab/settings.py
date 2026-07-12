"""Centralized settings via pydantic-settings — single source of truth for all config."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── LLM Provider ──────────────────────────────────────────────────────
    llm_provider: str = Field("openai", alias="LLM_PROVIDER")
    llm_base_url: str = Field("https://api.openai.com/v1", alias="LLM_BASE_URL")
    llm_model: str = Field("gpt-4o", alias="LLM_MODEL")
    llm_model_2: str = Field("gpt-4o-mini", alias="LLM_MODEL_2")
    llm_api_key: str = Field("", alias="LLM_API_KEY")

    # ── Second provider (multi-model) ──────────────────────────────────────
    llm_api_key_2: str | None = Field(None, alias="LLM_API_KEY_2")
    llm_base_url_2: str | None = Field(None, alias="LLM_BASE_URL_2")
    llm_model_2_override: str | None = Field(None, alias="LLM_MODEL_2_OVERRIDE")

    # ── Anthropic ──────────────────────────────────────────────────────────
    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")

    # ── Gemini ─────────────────────────────────────────────────────────────
    gemini_api_key: str | None = Field(None, alias="GEMINI_API_KEY")

    # ── DeepEval ───────────────────────────────────────────────────────────
    deepeval_enabled: bool = Field(False, alias="DEEPEVAL_ENABLED")
    deepeval_model: str | None = Field(None, alias="DEEPEVAL_MODEL")
    deepeval_threshold: float = Field(0.5, alias="DEEPEVAL_THRESHOLD")

    # ── Promptfoo ──────────────────────────────────────────────────────────
    promptfoo_max_retries: int = Field(3, alias="PROMPTFOO_MAX_RETRIES")
    promptfoo_base_delay: float = Field(1.0, alias="PROMPTFOO_BASE_DELAY")

    # ── Verifiers ──────────────────────────────────────────────────────────
    default_verifier: str = Field("deepeval", alias="DEFAULT_VERIFIER")

    @property
    def is_local(self) -> bool:
        return self.llm_provider.lower().strip() in {
            "ollama", "llamacpp", "vllm", "localai", "tgi", "promptfoo",
        }

    @property
    def is_anthropic(self) -> bool:
        return self.llm_provider.lower().strip() in {"anthropic", "claude"}

    @property
    def is_gemini(self) -> bool:
        return self.llm_provider.lower().strip() in {"gemini", "google"}

    def api_key_for(self, suffix: str = "") -> str:
        key = f"LLM_API_KEY{suffix}"
        if val := os.getenv(key):
            return val
        return self.llm_api_key

    def base_url_for(self, suffix: str = "") -> str:
        key = f"LLM_BASE_URL{suffix}"
        if val := os.getenv(key):
            return val
        return self.llm_base_url


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# ── Model Presets ───────────────────────────────────────────────────────
PRESETS: dict[str, dict[str, str]] = {
    "cheap": {"model": "gpt-4o-mini", "verifier": "structural"},
    "balanced": {"model": "gpt-4o", "verifier": "deepeval"},
    "best": {"model": "claude-sonnet-4-20250514", "verifier": "deepeval"},
    "quick": {"model": "gpt-4o-mini", "verifier": "structural"},
}


def resolve_preset(name: str | None) -> dict[str, str] | None:
    if name is None:
        return None
    p = PRESETS.get(name.lower())
    if p is None:
        raise KeyError(f"unknown preset: {name!r}. available: {list(PRESETS)}")
    return dict(p)
