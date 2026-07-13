"""Contract tests between ``worker.call_llm`` and ``promptfoo_provider.call_llm``.

ADR-0002 commits us to keeping the two LLM paths independent but consistent
on the user-visible response shape. This file pins the *contract* — the
shared field set and types — so the two paths can evolve without silently
diverging.

If you intentionally change the contract (add a field, drop one, change a
type), update both this test AND any consumer that depends on the old shape.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

# Shared fields both paths MUST return. The two paths are allowed to add
# extra fields (e.g., `_from_cache` on the promptfoo path), but they must
# never drop or rename these — those are the contract.
_REQUIRED_FIELDS = {"output", "model", "finish_reason", "token_usage", "cost_usd"}
_TOKEN_USAGE_FIELDS = {"prompt_tokens", "completion_tokens", "total_tokens"}


def _assert_contract(result: dict[str, Any], where: str) -> None:
    """Assert that ``result`` conforms to the shared LLM-response contract."""
    missing = _REQUIRED_FIELDS - set(result.keys())
    assert not missing, f"[{where}] missing required fields: {missing}; got {sorted(result.keys())}"

    assert isinstance(result["output"], str), f"[{where}] output must be str"
    assert isinstance(result["model"], str), f"[{where}] model must be str"
    assert isinstance(result["finish_reason"], str), f"[{where}] finish_reason must be str"

    tu = result["token_usage"]
    missing_tu = _TOKEN_USAGE_FIELDS - set(tu.keys())
    assert not missing_tu, f"[{where}] token_usage missing: {missing_tu}"
    for k in _TOKEN_USAGE_FIELDS:
        assert isinstance(tu[k], int), f"[{where}] token_usage.{k} must be int"
        assert tu[k] >= 0, f"[{where}] token_usage.{k} must be non-negative"

    cost = result["cost_usd"]
    assert isinstance(cost, (int, float)), f"[{where}] cost_usd must be numeric"
    assert cost >= 0, f"[{where}] cost_usd must be non-negative"


# ---------------------------------------------------------------------------
# worker.call_llm — OpenAI-compatible path
# ---------------------------------------------------------------------------


def test_worker_call_llm_response_shape(monkeypatch):
    """A successful OpenAI-compatible call returns the contract dict."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    fake_response = ChatCompletion(
        id="chatcmpl-fake",
        object="chat.completion",
        created=0,
        model="gpt-4o-mini",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12),
    )

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")

    with patch("llm_lab.worker._build_client") as mock_build:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_response
        mock_build.return_value = (mock_client, "openai")

        from llm_lab import worker

        result = worker.call_llm("hi")

    _assert_contract(result, "worker.openai")


def test_worker_call_llm_error_shape(monkeypatch):
    """When the call fails, the error path still returns a contract-conformant dict."""
    from openai import OpenAIError

    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with patch("llm_lab.worker._build_client") as mock_build:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = OpenAIError("boom")
        mock_build.return_value = (mock_client, "openai")

        from llm_lab import worker

        result = worker.call_llm("hi")

    _assert_contract(result, "worker.openai.error")
    assert result["finish_reason"] == "error"
    assert result["token_usage"]["total_tokens"] == 0
    assert result["cost_usd"] == 0.0
    assert "boom" in result["output"]


# ---------------------------------------------------------------------------
# promptfoo_provider.call_llm
# ---------------------------------------------------------------------------


def test_promptfoo_call_llm_response_shape(monkeypatch):
    """A successful promptfoo call returns the contract dict."""
    _expected_response = {
        "output": "pf hello",
        "model": "gpt-4o",
        "finish_reason": "stop",
        "token_usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        "cost_usd": 0.0002,
        "provider": "openai:chat",
        "_from_cache": False,
    }

    monkeypatch.setenv("PROMPTFOO_CONFIG", "")
    monkeypatch.setenv("LLM_API_KEY", "ollama")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:19999/v1")

    with patch("llm_lab.promptfoo_provider._ensure_cache") as mock_cache, \
         patch("llm_lab.promptfoo_provider._cache_get", return_value=None), \
         patch("llm_lab.promptfoo_provider._build_client_from_config") as mock_build:
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "pf hello"
        mock_choice.finish_reason = "stop"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 3
        mock_usage.completion_tokens = 4
        mock_usage.total_tokens = 7
        mock_response_obj = MagicMock()
        mock_response_obj.choices = [mock_choice]
        mock_response_obj.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_response_obj
        mock_build.return_value = mock_client
        # Pretend the cache table already exists; don't try to write to disk.
        mock_conn = MagicMock()
        mock_cache.return_value = mock_conn

        from llm_lab import promptfoo_provider

        result = promptfoo_provider.call_llm("hi")

    _assert_contract(result, "promptfoo")


# ---------------------------------------------------------------------------
# Cross-path consistency
# ---------------------------------------------------------------------------


def test_both_paths_emit_identical_required_field_set(monkeypatch):
    """No matter which path the call lands on, the contract field set is the same."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    fake_response = ChatCompletion(
        id="chatcmpl-fake",
        object="chat.completion",
        created=0,
        model="gpt-4o",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="x"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    # --- worker path ---
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    with patch("llm_lab.worker._build_client") as mock_build:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_response
        mock_build.return_value = (mock_client, "openai")
        from llm_lab import worker

        worker_result = worker.call_llm("x")
    worker_keys = set(worker_result.keys()) & _REQUIRED_FIELDS

    # --- promptfoo path ---
    monkeypatch.setenv("PROMPTFOO_CONFIG", "")
    with patch("llm_lab.promptfoo_provider._ensure_cache") as mock_cache, \
         patch("llm_lab.promptfoo_provider._cache_get", return_value=None), \
         patch("llm_lab.promptfoo_provider._build_client_from_config") as mock_build_pf:
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "x"
        mock_choice.finish_reason = "stop"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 1
        mock_usage.completion_tokens = 2
        mock_usage.total_tokens = 3
        mock_response_obj = MagicMock()
        mock_response_obj.choices = [mock_choice]
        mock_response_obj.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_response_obj
        mock_build_pf.return_value = mock_client
        mock_conn = MagicMock()
        mock_cache.return_value = mock_conn
        from llm_lab import promptfoo_provider

        pf_result = promptfoo_provider.call_llm("x")
    pf_keys = set(pf_result.keys()) & _REQUIRED_FIELDS

    assert worker_keys == pf_keys == _REQUIRED_FIELDS, (
        f"contract drift: worker={worker_keys}, promptfoo={pf_keys}, expected={_REQUIRED_FIELDS}"
    )