import os
from typing import Any

from openai import OpenAI, OpenAIError

from llm_lab.pricing import _LOCAL_PROVIDERS
from llm_lab.pricing import estimate_cost as _estimate_cost

_LOCAL_DEFAULTS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "deepseek-r1:7b"},
    "llamacpp": {"base_url": "http://localhost:8080/v1", "model": "llama-3.1-8b-instruct"},
    "vllm": {"base_url": "http://localhost:8000/v1", "model": "meta-llama/Llama-3.1-8B-Instruct"},
    "localai": {"base_url": "http://localhost:8080/v1", "model": "llama-3.1-8b-instruct"},
    "tgi": {"base_url": "http://localhost:3000/v1", "model": "meta-llama/Llama-3.1-8B-Instruct"},
}


def _resolve_provider() -> str:
    return (os.getenv("LLM_PROVIDER") or "openai").strip().lower()


def _is_local(provider: str | None = None) -> bool:
    return (provider or _resolve_provider()) in _LOCAL_PROVIDERS


def _is_anthropic(provider: str | None = None) -> bool:
    return (provider or _resolve_provider()) in {"anthropic", "claude"}


def _is_gemini(provider: str | None = None) -> bool:
    return (provider or _resolve_provider()) in {"gemini", "google"}


def _build_client(provider_index: int = 0) -> tuple[OpenAI, str]:
    provider = _resolve_provider()
    suffix = "" if provider_index == 0 else f"_{provider_index + 1}"

    if _is_local(provider):
        provider_cfg = _LOCAL_DEFAULTS.get(provider, _LOCAL_DEFAULTS["ollama"])
        base_url = os.getenv(f"LLM_BASE_URL{suffix}") or provider_cfg["base_url"]
        api_key = os.getenv(f"LLM_API_KEY{suffix}") or "ollama"
        return OpenAI(api_key=api_key, base_url=base_url), provider

    api_key_raw = os.getenv(f"LLM_API_KEY{suffix}")
    api_key2 = api_key_raw if api_key_raw else os.getenv("LLM_API_KEY", "")
    base_url_raw = os.getenv(f"LLM_BASE_URL{suffix}")
    base_url2 = base_url_raw if base_url_raw else os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key2, base_url=base_url2), provider


def _get_default_model(provider: str) -> str:
    if provider in _LOCAL_DEFAULTS:
        return _LOCAL_DEFAULTS[provider]["model"]
    if _is_anthropic(provider):
        return "claude-3-5-sonnet-20241022"
    if _is_gemini(provider):
        return "gemini-2.0-flash-001"
    return "gpt-4o"


def _call_anthropic(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    try:
        from anthropic import Anthropic
    except ImportError:
        return {
            "output": "[anthropic] SDK not installed. Run: pip install anthropic",
            "model": model,
            "finish_reason": "error",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cost_usd": 0.0,
        }
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        output = getattr(resp.content[0], "text", "") if resp.content else ""
        tokens = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
        }
        cost = _estimate_cost(model, resp.usage.input_tokens, resp.usage.output_tokens)
        return {
            "output": output,
            "model": model,
            "finish_reason": resp.stop_reason or "stop",
            "token_usage": tokens,
            "cost_usd": round(cost, 6),
        }
    except Exception as e:
        return {
            "output": f"[anthropic] LLM call failed: {e}",
            "model": model,
            "finish_reason": "error",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cost_usd": 0.0,
        }


def _call_gemini(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {
            "output": "[gemini] SDK not installed. Run: pip install google-genai",
            "model": model,
            "finish_reason": "error",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cost_usd": 0.0,
        }
    try:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        output = resp.text or ""

        usage = resp.usage_metadata
        pt = getattr(usage, "prompt_token_count", 0) or 0
        ct = getattr(usage, "candidates_token_count", 0) or 0
        tokens = {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        }
        cost = _estimate_cost(model, pt, ct)
        candidates = resp.candidates
        if candidates and candidates[0].finish_reason is not None:
            fr = "stop" if getattr(candidates[0].finish_reason, "name", "OTHER") == "STOP" else "other"
        else:
            fr = "other"
        return {
            "output": output,
            "model": model,
            "finish_reason": fr,
            "token_usage": tokens,
            "cost_usd": round(cost, 6),
        }
    except Exception as e:
        return {
            "output": f"[gemini] LLM call failed: {e}",
            "model": model,
            "finish_reason": "error",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cost_usd": 0.0,
        }


def call_llm(
    prompt: str,
    model: str | None = None,
    provider_index: int = 0,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    provider = _resolve_provider()

    if provider == "promptfoo":
        from llm_lab import promptfoo_provider

        return promptfoo_provider.call_llm(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if _is_anthropic(provider):
        model = model or os.getenv("LLM_MODEL") or _get_default_model(provider)
        return _call_anthropic(prompt, model, temperature, max_tokens)

    if _is_gemini(provider):
        model = model or os.getenv("LLM_MODEL") or _get_default_model(provider)
        return _call_gemini(prompt, model, temperature, max_tokens)

    client, provider = _build_client(provider_index)
    model = model or os.getenv("LLM_MODEL") or _get_default_model(provider)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except OpenAIError as e:
        return {
            "output": f"[{provider}] LLM call failed: {e}",
            "model": model,
            "finish_reason": "error",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cost_usd": 0.0,
        }

    choice = resp.choices[0]
    usage = resp.usage
    if usage is None:
        return {
            "output": "", "model": model, "finish_reason": "error",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cost_usd": 0.0,
        }
    tokens: dict[str, int] = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    cost = _estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
    return {
        "output": choice.message.content or "",
        "model": model,
        "finish_reason": choice.finish_reason,
        "token_usage": tokens,
        "cost_usd": round(cost, 6),
    }
