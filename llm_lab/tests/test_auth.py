import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    import llm_lab.main as m

    monkeypatch.setattr(m, "_API_KEY", "test-secret")
    with TestClient(m.app) as c:
        yield c


def test_readonly_endpoint_requires_key(client):
    assert client.get("/history").status_code == 401
    assert client.get("/result/nope").status_code == 401
    assert client.get("/trace/nope").status_code == 401
    assert client.get("/compare/report/a/b").status_code == 401


def test_readonly_endpoint_allows_valid_key(client):
    headers = {"X-API-Key": "test-secret"}
    assert client.get("/history", headers=headers).status_code == 200
    # auth passes; unknown intent lookup fails with 404
    assert client.get("/result/nope", headers=headers).status_code == 404
    assert client.get("/trace/nope", headers=headers).status_code == 404


def test_bearer_auth_accepted(client):
    headers = {"Authorization": "Bearer test-secret"}
    assert client.get("/history", headers=headers).status_code == 200


def test_wrong_key_rejected(client):
    assert client.get("/history", headers={"X-API-Key": "wrong"}).status_code == 401


def test_export_endpoints_require_key(client):
    headers = {"X-API-Key": "test-secret"}
    assert client.get("/export/csv").status_code == 401
    assert client.get("/export/csv", headers=headers).status_code != 401
    assert client.get("/export/xlsx").status_code == 401
    assert client.get("/export/xlsx", headers=headers).status_code != 401


def test_status_and_templates_require_key(client):
    headers = {"X-API-Key": "test-secret"}
    # /status/{task_id} leaks run results
    assert client.get("/status/nope").status_code == 401
    assert client.get("/status/nope", headers=headers).status_code == 404
    # /templates lists template definitions
    assert client.get("/templates").status_code == 401
    assert client.get("/templates", headers=headers).status_code == 200


def test_security_headers_present():
    from llm_lab.main import app

    with TestClient(app) as c:
        resp = c.get("/")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert resp.headers.get("Referrer-Policy") == "no-referrer"


def test_invalid_path_id_rejected(client):
    # Path params carrying disallowed chars (whitespace, angle brackets, etc.)
    # are rejected by the route regex before reaching any handler/template.
    headers = {"X-API-Key": "test-secret"}
    assert client.get("/result/with%20space", headers=headers).status_code == 422
    assert client.get("/trace/with%20space", headers=headers).status_code == 422
    assert client.get("/status/with%20space", headers=headers).status_code == 422
    assert client.get("/compare/report/a%20b/c", headers=headers).status_code == 422
    # A clean uuid-shaped id still passes validation (lookup then 404).
    uid = "11111111-2222-3333-4444-555555555555"
    assert client.get(f"/result/{uid}", headers=headers).status_code == 404
