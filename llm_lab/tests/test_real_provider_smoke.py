"""Real-provider smoke test, gated on OPENAI_API_KEY.

Per ROADMAP-v0.10.0 (stability section): one cheap gpt-4o-mini call
against the live OpenAI API. The test skips when OPENAI_API_KEY is
missing, so the build never fails because a secret is absent — CI
simply doesn't exercise it until the secret is configured.

The error path in worker.py also returns the full 5-field contract
(zeroed token_usage, finish_reason="error"), so asserting the whole
contract is safe on both paths; success is asserted separately.
"""

import os

import pytest

from llm_lab.worker import call_llm

_CI_DUMMY_KEYS = {"sk-test", "sk-ant-test", "test"}


def _real_key() -> str:
    """Return OPENAI_API_KEY when it is a real key, else "".

    The repo's CI workflow sets dummy keys (sk-test etc.) so offline
    providers never hit the network; the smoke test must skip on those
    too, otherwise CI would try a live call with a fake key and fail.
    """
    key = os.getenv("OPENAI_API_KEY", "")
    return key if key and key not in _CI_DUMMY_KEYS else ""


pytestmark = pytest.mark.skipif(
    not _real_key(),
    reason="no real OPENAI_API_KEY (unset or CI dummy); real-provider smoke test skipped",
)


def test_real_gpt4o_mini_call_succeeds() -> None:
    result = call_llm(
        "Reply with exactly: OK",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=16,
    )
    assert isinstance(result["output"], str)
    assert result["output"].strip()
    assert result["model"] == "gpt-4o-mini"
    assert result["finish_reason"] == "stop"
    assert result["token_usage"]["total_tokens"] > 0
    assert result["cost_usd"] >= 0.0
