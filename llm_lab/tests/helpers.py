"""Helper functions shared across tests."""

from types import SimpleNamespace


def make_verdict(label: str = "pass", reason: str = "ok") -> SimpleNamespace:
    """Create a Verdict-like object without importing models."""
    return SimpleNamespace(label=label, reason=reason)
