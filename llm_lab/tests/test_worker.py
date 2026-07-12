import os

from pytest import approx

from llm_lab import worker as wrk


def test_default_provider_is_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert wrk._resolve_provider() == "openai"


def test_local_provider_detection():
    os.environ["LLM_PROVIDER"] = "ollama"
    assert wrk._is_local() is True

    os.environ["LLM_PROVIDER"] = "openai"
    assert wrk._is_local() is False

    os.environ["LLM_PROVIDER"] = "vllm"
    assert wrk._is_local() is True

    os.environ.pop("LLM_PROVIDER", None)


def test_local_cost_is_zero():
    os.environ["LLM_PROVIDER"] = "ollama"
    assert wrk._estimate_cost("qwen2.5:7b", 100, 50) == 0.0
    assert wrk._estimate_cost("llama3.1:8b", 500, 200) == 0.0
    os.environ.pop("LLM_PROVIDER", None)


def test_remote_cost_uses_price_table():
    assert wrk._estimate_cost("gpt-4o", 1000, 500) == approx(0.0025 + 0.005)
    assert wrk._estimate_cost("deepseek-chat", 1000, 500) == approx(0.00014 + 0.00014)


def test_remote_cost_fallback():
    c = wrk._estimate_cost("unknown-model", 1000, 500)
    assert c == approx(0.003 * 1.5)


def test_build_client_local_defaults():
    os.environ["LLM_PROVIDER"] = "ollama"
    _, provider = wrk._build_client()
    assert provider == "ollama"
    os.environ.pop("LLM_PROVIDER", None)


def test_build_client_openai_defaults():
    os.environ["OPENAI_API_KEY"] = "sk-test"
    _, provider = wrk._build_client()
    assert provider == "openai"
    os.environ.pop("OPENAI_API_KEY", None)


def test_call_llm_local_without_server():
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_MODEL"] = "qwen2.5:7b"
    result = wrk.call_llm("hello")
    assert "LLM call failed" in result["output"]
    assert result["finish_reason"] == "error"
    assert result["cost_usd"] == 0.0
    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("LLM_MODEL", None)


def test_provider_resolve_case_insensitive():
    os.environ["LLM_PROVIDER"] = "OLLAMA"
    assert wrk._resolve_provider() == "ollama"
    os.environ.pop("LLM_PROVIDER", None)


def test_local_default_model_names():
    for provider, cfg in wrk._LOCAL_DEFAULTS.items():
        os.environ["LLM_PROVIDER"] = provider
        assert wrk._get_default_model(provider) == cfg["model"]
    os.environ.pop("LLM_PROVIDER", None)


def test_second_provider_index():
    os.environ["OPENAI_API_KEY"] = "sk-test"
    os.environ["LLM_API_KEY_2"] = "sk-test-2"
    os.environ["LLM_BASE_URL_2"] = "http://localhost:9999/v1"
    os.environ["LLM_MODEL_2"] = "test-model"
    client, _ = wrk._build_client(provider_index=1)
    assert str(client.base_url).rstrip("/") == "http://localhost:9999/v1"
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("LLM_API_KEY_2", None)
    os.environ.pop("LLM_BASE_URL_2", None)
    os.environ.pop("LLM_MODEL_2", None)


def test_provider_promptfoo_is_local():
    os.environ["LLM_PROVIDER"] = "promptfoo"
    assert wrk._is_local() is True
    os.environ.pop("LLM_PROVIDER", None)


def test_promptfoo_provider_cache_hit():
    from llm_lab import promptfoo_provider

    k = promptfoo_provider._cache_key("hello", "test-model")
    conn = promptfoo_provider._ensure_cache()
    conn.execute("DELETE FROM llm_cache WHERE key = ?", (k,))
    conn.commit()
    conn.execute(
        "INSERT INTO llm_cache (key, response, created_at) VALUES (?, ?, ?)",
        (k, '{"output":"cached reply","model":"test-model"}', 1.0),
    )
    conn.commit()
    conn.close()
    result = promptfoo_provider._cache_get("hello", "test-model")
    assert result is not None
    assert result["output"] == "cached reply"


def test_promptfoo_provider_cache_miss():
    from llm_lab import promptfoo_provider

    result = promptfoo_provider._cache_get("nonexistent-prompt", "nonexistent-model")
    assert result is None


def test_promptfoo_provider_retries_on_no_server():
    from llm_lab import promptfoo_provider

    os.environ["LLM_PROVIDER"] = "promptfoo"
    os.environ["LLM_BASE_URL"] = "http://localhost:19999/v1"
    os.environ["LLM_MODEL"] = "test-model"
    os.environ["PROMPTFOO_MAX_RETRIES"] = "1"
    os.environ["PROMPTFOO_BASE_DELAY"] = "0"
    unique_prompt = f"retry-test-{os.urandom(4).hex()}"
    result = promptfoo_provider.call_llm(unique_prompt, max_tokens=1)
    assert result.get("finish_reason") == "error", f"Expected error, got: {result}"
    assert "retries" in result["output"]
    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("LLM_BASE_URL", None)
    os.environ.pop("LLM_MODEL", None)
    os.environ.pop("PROMPTFOO_MAX_RETRIES", None)
    os.environ.pop("PROMPTFOO_BASE_DELAY", None)


# ── _call_anthropic ──────────────────────────────────────────────────────────


def test_call_anthropic_success(monkeypatch):
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    mock_content = MagicMock()
    mock_content.text = "Hello from Claude"
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 20
    mock_response.stop_reason = "end_turn"
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        result = wrk._call_anthropic("hello", "claude-sonnet-4-20250514")
    assert result["output"] == "Hello from Claude"
    assert result["finish_reason"] == "end_turn"
    assert result["token_usage"]["prompt_tokens"] == 10


def test_call_anthropic_api_error(monkeypatch):
    from unittest.mock import patch

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    with patch("anthropic.Anthropic", side_effect=Exception("API down")):
        result = wrk._call_anthropic("hello", "claude-sonnet-4-20250514")
    assert "LLM call failed" in result["output"]
    assert result["finish_reason"] == "error"
    assert result["cost_usd"] == 0.0


def test_call_anthropic_import_error(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("No module named anthropic")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)
    result = wrk._call_anthropic("hello", "claude-3-5-sonnet-20241022")
    assert "SDK not installed" in result["output"]


# ── _call_gemini ─────────────────────────────────────────────────────────


def test_call_gemini_success(monkeypatch):
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Gemini response"
    mock_resp.candidates = [MagicMock()]
    mock_resp.candidates[0].finish_reason = MagicMock()
    mock_resp.candidates[0].finish_reason.name = "STOP"
    mock_resp.usage_metadata = MagicMock()
    mock_resp.usage_metadata.prompt_token_count = 15
    mock_resp.usage_metadata.candidates_token_count = 25
    mock_client.models.generate_content.return_value = mock_resp

    with patch("google.genai.Client", return_value=mock_client):
        result = wrk._call_gemini("hello", "gemini-2.0-flash-001")
    assert result["output"] == "Gemini response"
    assert result["finish_reason"] == "stop"
    assert result["token_usage"]["prompt_tokens"] == 15


def test_call_gemini_api_error(monkeypatch):
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("Gemini down")

    with patch("google.genai.Client", return_value=mock_client):
        result = wrk._call_gemini("hello", "gemini-2.0-flash-001")
    assert "LLM call failed" in result["output"]
    assert result["finish_reason"] == "error"


def test_call_gemini_import_error(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "google.genai" or name == "google.genai.types":
            raise ImportError("No module named google.genai")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)
    result = wrk._call_gemini("hello", "gemini-2.0-flash-001")
    assert "SDK not installed" in result["output"]


# ── _get_default_model coverage ──────────────────────────────────────────


def test_get_default_model_anthropic():
    assert wrk._get_default_model("anthropic") == "claude-3-5-sonnet-20241022"


def test_get_default_model_gemini():
    assert wrk._get_default_model("gemini") == "gemini-2.0-flash-001"


# ── call_llm routing coverage ────────────────────────────────────────────


def test_call_llm_promptfoo(monkeypatch):
    from unittest.mock import patch

    monkeypatch.setenv("LLM_PROVIDER", "promptfoo")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:19999/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    with patch("llm_lab.promptfoo_provider.call_llm") as mock_pf:
        mock_pf.return_value = {"output": "pf ok", "finish_reason": "stop"}
        result = wrk.call_llm("hello", max_tokens=1)
    assert result["output"] == "pf ok"
    assert result["finish_reason"] == "stop"
    for k in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)


def test_call_llm_anthropic(monkeypatch):
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-20250514")

    mock_content = MagicMock()
    mock_content.text = "Hello from Claude"
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 20
    mock_response.stop_reason = "end_turn"
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        result = wrk.call_llm("hello")
    assert result["output"] == "Hello from Claude"
    assert result["finish_reason"] == "end_turn"
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)


def test_call_llm_gemini(monkeypatch):
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Gemini response"
    mock_resp.candidates = [MagicMock()]
    mock_resp.candidates[0].finish_reason = MagicMock()
    mock_resp.candidates[0].finish_reason.name = "STOP"
    mock_resp.usage_metadata = MagicMock()
    mock_resp.usage_metadata.prompt_token_count = 15
    mock_resp.usage_metadata.candidates_token_count = 25
    mock_client.models.generate_content.return_value = mock_resp

    with patch("google.genai.Client", return_value=mock_client):
        result = wrk.call_llm("hello")
    assert result["output"] == "Gemini response"
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_call_llm_openai_happy_path(monkeypatch):
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_choice = MagicMock()
    mock_choice.message.content = "hello world"
    mock_choice.finish_reason = "stop"
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 30
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("llm_lab.worker.OpenAI", return_value=mock_client):
        result = wrk.call_llm("hello")
    assert result["output"] == "hello world"
    assert result["finish_reason"] == "stop"
    assert result["token_usage"]["prompt_tokens"] == 10
    assert result["token_usage"]["total_tokens"] == 30
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_call_llm_usage_none(monkeypatch):
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = None
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("llm_lab.worker.OpenAI", return_value=mock_client):
        result = wrk.call_llm("hello")
    assert result["output"] == ""
    assert result["finish_reason"] == "error"
    assert result["token_usage"]["total_tokens"] == 0
    assert result["cost_usd"] == 0.0
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# ── _is_anthropic / _is_gemini ────────────────────────────────────────


def test_is_anthropic_true(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert wrk._is_anthropic() is True
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_is_anthropic_with_claude_alias(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    assert wrk._is_anthropic() is True
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_is_anthropic_false(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert wrk._is_anthropic() is False
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_is_gemini_true(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert wrk._is_gemini() is True
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_is_gemini_with_google_alias(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "google")
    assert wrk._is_gemini() is True
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_is_gemini_false(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert wrk._is_gemini() is False
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


# ── _build_client with local provider + LLM_BASE_URL override ────────


def test_build_client_local_base_url_override():
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_BASE_URL"] = "http://my-custom-ollama:11434/v1"
    client, _ = wrk._build_client()
    assert str(client.base_url).rstrip("/") == "http://my-custom-ollama:11434/v1"
    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("LLM_BASE_URL", None)
