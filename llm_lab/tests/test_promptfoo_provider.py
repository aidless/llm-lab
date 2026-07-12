"""Tests for promptfoo_provider.py — module-level functions for promptfoo-style LLM calls."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_lab import promptfoo_provider as pf


@pytest.fixture(autouse=True)
def _clean_cache_db(monkeypatch):
    tmp = Path("_test_promptfoo_cache.db")
    if tmp.exists():
        tmp.unlink()
    monkeypatch.setenv("PROMPTFOO_CACHE", str(tmp))
    yield
    if tmp.exists():
        tmp.unlink()


def test_cache_path_constant_module_level():
    path = pf._cache_path()
    assert ".promptfoo_cache" in str(path) or "_test_promptfoo_cache" in str(path)


def test_cache_path_default():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.delenv("PROMPTFOO_CACHE", raising=False)
    path = pf._cache_path()
    assert ".promptfoo_cache.db" in str(path)
    monkeypatch.undo()


def test_cache_key_deterministic():
    k1 = pf._cache_key("hello", "gpt-4o")
    k2 = pf._cache_key("hello", "gpt-4o")
    assert k1 == k2
    assert len(k1) == 16


def test_cache_key_differs_for_different_model():
    k1 = pf._cache_key("hello", "gpt-4o")
    k2 = pf._cache_key("hello", "gpt-4o-mini")
    assert k1 != k2


def test_cache_get_miss():
    assert pf._cache_get("nonexistent", "model") is None


def test_cache_set_and_get():
    response = {"output": "test response"}
    pf._cache_set("my prompt", "gpt-4o", response)
    cached = pf._cache_get("my prompt", "gpt-4o")
    assert cached == response


def test_cache_overwrite():
    pf._cache_set("key", "m", {"output": "first"})
    pf._cache_set("key", "m", {"output": "second"})
    cached = pf._cache_get("key", "m")
    assert cached["output"] == "second"


def test_cache_isolation_by_model():
    pf._cache_set("prompt", "model-a", {"output": "a"})
    pf._cache_set("prompt", "model-b", {"output": "b"})
    assert pf._cache_get("prompt", "model-a")["output"] == "a"
    assert pf._cache_get("prompt", "model-b")["output"] == "b"


def test_ensure_cache_creates_db():
    db = pf._cache_path()
    if db.exists():
        db.unlink()
    conn = pf._ensure_cache()
    assert db.exists()
    conn.close()


def test_read_provider_config_empty_by_default(monkeypatch):
    monkeypatch.delenv("PROMPTFOO_CONFIG", raising=False)
    assert pf._read_provider_config() == {}


def test_read_provider_config_missing_file(monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG", "/nonexistent/path.yaml")
    assert pf._read_provider_config() == {}


def test_read_provider_config_from_file(monkeypatch, tmp_path):
    cfg_file = tmp_path / "provider.yaml"
    cfg_file.write_text("id: openai:chat\nmodel: gpt-4o\n")
    monkeypatch.setenv("PROMPTFOO_CONFIG", str(cfg_file))
    result = pf._read_provider_config()
    assert result.get("id") == "openai:chat"
    assert result.get("model") == "gpt-4o"


def test_build_client_from_config_defaults():
    with patch.dict(os.environ, {"LLM_API_KEY": "test-key", "LLM_BASE_URL": "http://test:8000/v1"}):
        client = pf._build_client_from_config({})
        assert client.api_key == "test-key"
        assert str(client.base_url) == "http://test:8000/v1/"


def test_build_client_from_config_overrides():
    cfg = {"api_key": "cfg-key", "base_url": "http://cfg:8000/v1"}
    client = pf._build_client_from_config(cfg)
    assert client.api_key == "cfg-key"
    assert str(client.base_url) == "http://cfg:8000/v1/"


def test_estimate_cost_known_model(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    cost = pf._estimate_cost("gpt-4o", 1000, 500)
    assert cost == pytest.approx(1000 / 1000 * 0.0025 + 500 / 1000 * 0.01)


def test_estimate_cost_unknown_model(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    cost = pf._estimate_cost("unknown-model", 1000, 500)
    assert cost > 0


def test_estimate_cost_zero_tokens():
    cost = pf._estimate_cost("gpt-4o", 0, 0)
    assert cost == 0.0


def test_estimate_cost_local_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    cost = pf._estimate_cost("any-model", 1000, 500)
    assert cost == 0.0


def test_call_llm_returns_cache_on_hit(monkeypatch):
    pf._cache_set("cached prompt", "test-model", {"output": "from cache"})

    result = pf.call_llm("cached prompt", model="test-model")
    assert result["output"] == "from cache"
    assert result.get("_from_cache") is True


def test_call_llm_fallback_on_error(monkeypatch):
    monkeypatch.setenv("PROMPTFOO_MAX_RETRIES", "1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:0/v1")

    result = pf.call_llm("any prompt", model="gpt-4o")
    assert result["finish_reason"] == "error"
    assert "[promptfoo" in result["output"]


def test_call_llm_uses_default_model(monkeypatch):
    monkeypatch.setenv("PROMPTFOO_MAX_RETRIES", "1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:0/v1")

    result = pf.call_llm("test")
    assert result["model"] is not None


def test_call_llm_model_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("PROMPTFOO_MAX_RETRIES", "1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:0/v1")

    result = pf.call_llm("test")
    assert result["model"] == "env-model"


def test_call_llm_writes_cache_after_success(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:0/v1")
    monkeypatch.setenv("PROMPTFOO_MAX_RETRIES", "1")

    with patch.object(pf, "_build_client_from_config") as mock_build:
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_choice = MagicMock()
        mock_usage = MagicMock()
        mock_choice.message.content = "mock output"
        mock_choice.finish_reason = "stop"
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_chat.choices = [mock_choice]
        mock_chat.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_chat
        mock_build.return_value = mock_client

        cached_before = pf._cache_get("mock prompt", "mock-model")
        assert cached_before is None

        pf.call_llm("mock prompt", model="mock-model")

        cached_after = pf._cache_get("mock prompt", "mock-model")
        assert cached_after is not None
        assert cached_after["output"] == "mock output"


def test_call_llm_provider_from_config(monkeypatch, tmp_path):
    cfg = {"id": "custom:provider"}
    cfg_file = tmp_path / "prov.yaml"
    import yaml
    cfg_file.write_text(yaml.dump(cfg))
    monkeypatch.setenv("PROMPTFOO_CONFIG", str(cfg_file))
    monkeypatch.setenv("PROMPTFOO_MAX_RETRIES", "1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:0/v1")

    result = pf.call_llm("test")
    assert result["provider"] == "custom:provider"


# ── Cache exception + retry delay coverage ───────────────────────────────


def test_cache_get_exception():
    """promptfoo_provider.py 46-47: _cache_get returns None on any exception."""
    with patch("llm_lab.promptfoo_provider._ensure_cache", side_effect=Exception("db err")):
        result = pf._cache_get("test", "model")
    assert result is None


def test_cache_set_exception():
    """promptfoo_provider.py 59-60: _cache_set silently swallows exceptions."""
    with patch("llm_lab.promptfoo_provider._ensure_cache", side_effect=Exception("db err")):
        pf._cache_set("test", "model", {"output": "ok"})  # should not raise


def test_call_with_retries_delay(monkeypatch):
    """promptfoo_provider.py 157-158: retry loop calls time.sleep on failure."""
    import time as time_mod

    monkeypatch.setenv("PROMPTFOO_MAX_RETRIES", "2")
    monkeypatch.setenv("PROMPTFOO_BASE_DELAY", "0.01")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:0/v1")


    sleep_calls = []

    def _track_sleep(secs):
        sleep_calls.append(secs)

    with patch.object(time_mod, "sleep", side_effect=_track_sleep):
        result = pf.call_llm("retry-test")
    assert len(sleep_calls) >= 1
    assert result["finish_reason"] == "error"
