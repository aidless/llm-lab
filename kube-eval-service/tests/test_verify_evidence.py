"""Regression tests for the NetworkPolicy evidence verifier."""

from __future__ import annotations

from verify_evidence import _verify_worker_network_policy


def _policy(*, policy_types: list[str], ingress: list[dict[str, object]] | None = None) -> dict[str, object]:
    spec: dict[str, object] = {"policyTypes": policy_types}
    if ingress is not None:
        spec["ingress"] = ingress
    return {"spec": spec}


def test_worker_policy_omitted_ingress_is_deny_all() -> None:
    errors: list[str] = []
    _verify_worker_network_policy(_policy(policy_types=["Ingress", "Egress"]), errors)
    assert errors == []


def test_worker_policy_empty_ingress_is_deny_all() -> None:
    errors: list[str] = []
    _verify_worker_network_policy(_policy(policy_types=["Ingress", "Egress"], ingress=[]), errors)
    assert errors == []


def test_worker_policy_requires_both_policy_types() -> None:
    errors: list[str] = []
    _verify_worker_network_policy(_policy(policy_types=["Egress"]), errors)
    assert "worker network policy does not restrict both ingress and egress" in errors


def test_worker_policy_rejects_nonempty_ingress() -> None:
    errors: list[str] = []
    _verify_worker_network_policy(
        _policy(
            policy_types=["Ingress", "Egress"],
            ingress=[{"from": [{"namespaceSelector": {}}]}],
        ),
        errors,
    )
    assert "worker network policy unexpectedly permits ingress" in errors
