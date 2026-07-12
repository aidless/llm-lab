from unittest.mock import AsyncMock, patch

import pytest

from llm_lab.tests.helpers import make_verdict


@pytest.fixture
def mock_llm():
    """Patch worker.call_llm with a deterministic response."""
    with patch("llm_lab.worker.call_llm") as m:
        m.return_value = {
            "output": "mock output",
            "model": "gpt-4o",
            "finish_reason": "stop",
            "token_usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
            "cost_usd": 0.0001,
        }
        yield m


@pytest.fixture
def mock_verifier():
    """Patch verifier.get_verifier with a pass-verdict."""
    with patch("llm_lab.verifier.get_verifier") as m:
        m.return_value.verify.return_value = make_verdict("pass", "ok")
        yield m


@pytest.fixture
def mock_tracer():
    """Patch tracer.trace_call with a no-op AsyncMock."""
    with patch("llm_lab.tracer.trace_call", new=AsyncMock()) as m:
        yield m
