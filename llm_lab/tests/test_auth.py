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
