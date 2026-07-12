"""Shared LLM token-pricing logic (single source of truth)."""

import os

_PRICE_PER_1K = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "deepseek-chat": (0.00014, 0.00028),
    "deepseek-reasoner": (0.00055, 0.00219),
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-haiku-3-5-20241022": (0.001, 0.005),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-haiku-20241022": (0.001, 0.005),
    "gemini-2.0-flash-001": (0.00015, 0.0006),
    "gemini-2.0-pro-001": (0.0025, 0.01),
}

_LOCAL_PROVIDERS = {"ollama", "llamacpp", "vllm", "localai", "tgi", "promptfoo"}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a model call.

    Local providers (ollama, vllm, ...) are free. Unknown remote models
    fall back to a flat $3 / 1M-token rate.
    """
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if provider in _LOCAL_PROVIDERS:
        return 0.0
    if model in _PRICE_PER_1K:
        p_in, p_out = _PRICE_PER_1K[model]
        return (prompt_tokens / 1000) * p_in + (completion_tokens / 1000) * p_out
    return 0.003 * (prompt_tokens + completion_tokens) / 1000
