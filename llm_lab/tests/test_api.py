"""Tests for main.py API endpoints — uses conftest.py sys.path."""

from unittest.mock import AsyncMock, patch

import pytest

from llm_lab.tests.helpers import make_verdict


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from llm_lab.main import app

    with TestClient(app) as client:
        yield client


def test_submit_sync(client):
    mock_result = {
        "output": "mock",
        "model": "gpt-4o",
        "finish_reason": "stop",
        "token_usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
        "cost_usd": 0.0001,
    }
    mv = make_verdict("pass", "ok")

    with (
        patch("llm_lab.worker.call_llm", return_value=mock_result),
        patch("llm_lab.verifier.get_verifier") as mock_get_v,
        patch("llm_lab.tracer.trace_call", new=AsyncMock()),
    ):
        mock_get_v.return_value.verify.return_value = mv
        resp = client.post("/submit", json={"goal": "test goal"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["goal"] == "test goal"
        assert "intent_id" in data


def test_submit_async_returns_task_id(client):
    with patch("llm_lab.tracer.trace_call", new=AsyncMock()), patch("llm_lab.main.core.run_plan") as mock_run:
        mock_run.return_value = {"intent_id": "mock", "goal": "async test", "all_passed": True, "steps": 1}
        resp = client.post("/submit/async", json={"goal": "async test"})
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "queued"


def test_compare_sync(client):
    mock_result = {
        "output": "mock",
        "model": "gpt-4o",
        "finish_reason": "stop",
        "token_usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
        "cost_usd": 0.0001,
    }
    mv = make_verdict("pass", "ok")

    with (
        patch("llm_lab.worker.call_llm", return_value=mock_result),
        patch("llm_lab.verifier.get_verifier") as mock_get_v,
        patch("llm_lab.tracer.trace_call", new=AsyncMock()),
    ):
        mock_get_v.return_value.verify.return_value = mv
        resp = client.post("/compare", json={"goal": "compare test"})
        assert resp.status_code == 200
        data = resp.json()
        assert "model_a" in data
        assert "model_b" in data
        assert "summary" in data


def test_compare_async_returns_task_id(client):
    with patch("llm_lab.main.core.compare") as mock_cmp:
        mock_cmp.return_value = {"model_a": {}, "model_b": {}, "summary": {"winner": "tie"}}
        resp = client.post("/compare/async", json={"goal": "async compare test"})
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "queued"


def test_batch_sync(client):
    mock_result = {
        "output": "mock",
        "model": "gpt-4o",
        "finish_reason": "stop",
        "token_usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
        "cost_usd": 0.0001,
    }
    mv = make_verdict("pass", "ok")

    with (
        patch("llm_lab.worker.call_llm", return_value=mock_result),
        patch("llm_lab.verifier.get_verifier") as mock_get_v,
        patch("llm_lab.tracer.trace_call", new=AsyncMock()),
    ):
        mock_get_v.return_value.verify.return_value = mv
        resp = client.post("/batch", json={"goal": "batch test", "models": ["gpt-4o", "gpt-4o-mini"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["models"]) == 2


def test_batch_async_returns_task_id(client):
    with patch("llm_lab.main.core.batch") as mock_batch:
        mock_batch.return_value = {"goal": "async batch test", "count": 2}
        resp = client.post("/batch/async", json={"goal": "async batch test", "models": ["gpt-4o", "gpt-4o-mini"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "queued"


def test_batch_empty_models_errors(client):
    resp = client.post("/batch", json={"goal": "test", "models": []})
    assert resp.status_code == 422


def test_batch_too_many_models_errors(client):
    resp = client.post("/batch", json={"goal": "test", "models": [f"m{i}" for i in range(25)]})
    assert resp.status_code == 422


def test_get_templates(client):
    resp = client.get("/templates")
    assert resp.status_code == 200
    data = resp.json()
    assert "templates" in data
    assert len(data["templates"]) >= 5


def test_create_delete_custom_template(client):
    payload = {
        "template_id": "test-tmpl-123",
        "intent_keywords": ["testkeyword"],
        "steps": ["step1", "step2"],
    }
    resp = client.post("/templates", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "created"

    resp = client.get("/templates")
    found = any(t["template_id"] == "test-tmpl-123" for t in resp.json()["templates"])
    assert found

    resp = client.delete("/templates/test-tmpl-123")
    assert resp.status_code == 200


def test_delete_builtin_template_errors(client):
    resp = client.delete("/templates/eval-model")
    assert resp.status_code == 403


def test_create_duplicate_builtin_errors(client):
    payload = {
        "template_id": "eval-model",
        "intent_keywords": ["test"],
        "steps": ["step1"],
    }
    resp = client.post("/templates", json=payload)
    assert resp.status_code == 409


def test_get_status_not_found(client):
    resp = client.get("/status/nonexistent")
    assert resp.status_code == 404


def test_history_endpoint(client):
    with patch("llm_lab.db.list_intents", return_value=[]):
        resp = client.get("/history")
        assert resp.status_code == 200
        assert resp.json() == {"runs": []}


def test_export_csv_no_data(client):
    with patch("llm_lab.db.get_all_events", return_value=[]):
        resp = client.get("/export/csv")
        assert resp.status_code == 404


def test_export_csv_with_data(client):
    mock_rows = [
        {
            "id": 1,
            "intent_id": "abc123",
            "seq": 1,
            "timestamp": "2025-01-01 00:00:00",
            "action": "call",
            "model": "gpt-4o",
            "detail": "some detail",
            "cost_usd": 0.001,
        }
    ]
    with patch("llm_lab.db.get_all_events", return_value=mock_rows):
        resp = client.get("/export/csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        body = resp.text
        assert "intent_id" in body
        assert "abc123" in body
        assert "gpt-4o" in body


def test_export_json_not_found(client):
    with patch("llm_lab.tracer.get_trace", new=AsyncMock(return_value=[])):
        resp = client.get("/export/json/nonexistent")
        assert resp.status_code == 404


def test_web_ui_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.skipif(not __import__("importlib").util.find_spec("openpyxl"), reason="openpyxl not installed")
def test_export_xlsx_by_intent_not_found(client):
    with patch("llm_lab.tracer.get_trace", new=AsyncMock(return_value=[])):
        resp = client.get("/export/xlsx/nonexistent")
    assert resp.status_code == 404


@pytest.mark.skipif(not __import__("importlib").util.find_spec("openpyxl"), reason="openpyxl not installed")
def test_export_xlsx_by_intent_happy_path(client):
    events = [{"seq": 1, "output": "exported"}]
    with patch("llm_lab.tracer.get_trace", new=AsyncMock(return_value=events)):
        resp = client.get("/export/xlsx/test")
    assert resp.status_code == 200
    ct = resp.headers["content-type"]
    assert "openxmlformats-officedocument.spreadsheetml.sheet" in ct
    assert resp.content[:2] == b"PK"


@pytest.mark.skipif(not __import__("importlib").util.find_spec("openpyxl"), reason="openpyxl not installed")
def test_export_xlsx_all_no_data(client):
    with patch("llm_lab.db.get_all_events", return_value=[]):
        resp = client.get("/export/xlsx")
    assert resp.status_code == 404


@pytest.mark.skipif(not __import__("importlib").util.find_spec("openpyxl"), reason="openpyxl not installed")
def test_export_xlsx_all_with_data(client):
    mock_rows = [{"id": 1, "intent_id": "abc", "seq": 1, "timestamp": "2025-01-01", "action": "call", "model": "gpt-4o", "detail": "detail", "cost_usd": 0.001}]
    with patch("llm_lab.db.get_all_events", return_value=mock_rows):
        resp = client.get("/export/xlsx")
    assert resp.status_code == 200
    assert "openxmlformats-officedocument.spreadsheetml.sheet" in resp.headers["content-type"]
    assert resp.content[:2] == b"PK"


def test_export_xlsx_by_intent_runtime_error(client):
    """/export/xlsx/{id} when export.export_xlsx raises RuntimeError → 501."""
    with (
        patch("llm_lab.main.tracer.get_trace", new=AsyncMock(return_value=[{"seq": 1}])),
        patch("llm_lab.main.export.export_xlsx", side_effect=RuntimeError("openpyxl is required")),
    ):
        resp = client.get("/export/xlsx/test-err")
    assert resp.status_code == 501
    assert "openpyxl is required" in resp.text


def test_export_xlsx_all_runtime_error(client):
    """/export/xlsx when export.export_xlsx raises RuntimeError → 501."""
    with (
        patch("llm_lab.main.db.get_all_events", return_value=[{"id": 1}]),
        patch("llm_lab.main.export.export_xlsx", side_effect=RuntimeError("openpyxl is required")),
    ):
        resp = client.get("/export/xlsx")
    assert resp.status_code == 501
    assert "openpyxl is required" in resp.text


def test_async_submit_e2e(client):
    """Submit async, poll the status endpoint until done, verify result."""
    with patch("llm_lab.main.core.run_plan") as mock_run:
        mock_run.return_value = {
            "intent_id": "mock123",
            "goal": "async e2e",
            "model": "gpt-4o",
            "all_passed": True,
            "steps": 1,
            "steps_detail": [],
            "total_tokens": 10,
            "total_cost_usd": 0.0001,
        }
        resp = client.post("/submit/async", json={"goal": "async e2e"})
        task_id = resp.json()["task_id"]

        import time

        for _ in range(10):
            s = client.get(f"/status/{task_id}").json()
            if s["status"] == "done":
                assert s["result"]["goal"] == "async e2e"
                return
            time.sleep(0.05)
        pytest.fail("async task did not complete within polling window")


def test_async_batch_e2e(client):
    """Submit async batch, poll until done, verify result."""
    with patch("llm_lab.main.core.batch") as mock_batch:
        mock_batch.return_value = {
            "goal": "async batch e2e",
            "count": 1,
            "models": [{"model": "gpt-4o", "all_passed": True}],
        }
        resp = client.post("/batch/async", json={"goal": "async batch e2e", "models": ["gpt-4o"]})
        task_id = resp.json()["task_id"]

        import time

        for _ in range(10):
            s = client.get(f"/status/{task_id}").json()
            if s["status"] == "done":
                assert s["result"]["goal"] == "async batch e2e"
                return
            time.sleep(0.05)
        pytest.fail("async batch task did not complete within polling window")


# ── main.py uncovered-branch coverage ────────────────────────────────────


def test_submit_tracer_failure_does_not_break_response(client):
    """main.py lines 101-102: tracer.trace_call exception caught with warning."""
    from unittest.mock import patch

    from llm_lab.tests.helpers import make_verdict

    mock_result = {
        "output": "mock",
        "model": "gpt-4o",
        "finish_reason": "stop",
        "token_usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
        "cost_usd": 0.0001,
    }
    mv = make_verdict("pass", "ok")

    with (
        patch("llm_lab.worker.call_llm", return_value=mock_result),
        patch("llm_lab.verifier.get_verifier") as mock_get_v,
        patch("llm_lab.tracer.trace_call", side_effect=Exception("tracer down")),
    ):
        mock_get_v.return_value.verify.return_value = mv
        resp = client.post("/submit", json={"goal": "test"})
        assert resp.status_code == 200


def test_async_submit_exception_handling(client):
    """main.py lines 62-64: _run_task catches exceptions."""
    import time

    with patch("llm_lab.main.core.run_plan", side_effect=ValueError("async error")):
        resp = client.post("/submit/async", json={"goal": "test"})
        task_id = resp.json()["task_id"]

        for _ in range(10):
            s = client.get(f"/status/{task_id}").json()
            if s["status"] == "error":
                assert "async error" in s["error"]
                return
            time.sleep(0.05)
        pytest.fail("async task did not reach error state")


def test_history_endpoint_server_error(client):
    """main.py lines 45-46: _try catches db exceptions."""
    with patch("llm_lab.db.list_intents", side_effect=Exception("db is down")):
        resp = client.get("/history")
        assert resp.status_code == 500
        assert "db is down" in resp.json()["detail"]


def test_delete_template_not_found(client):
    """main.py lines 212-213: delete non-existent template -> 404."""
    resp = client.delete("/templates/nonexistent-template-xyz")
    assert resp.status_code == 404


def test_compare_report_not_found(client):
    """main.py lines 233-237: compare report with missing traces -> 404."""
    with patch("llm_lab.tracer.get_trace", new=AsyncMock(return_value=[])):
        resp = client.get("/compare/report/missing-a/missing-b")
        assert resp.status_code == 404


def test_get_result_not_found(client):
    """main.py lines 252-254: get_result for missing intent_id -> 404."""
    with patch("llm_lab.tracer.get_summary", new=AsyncMock(return_value={"events": 0})):
        resp = client.get("/result/nonexistent")
        assert resp.status_code == 404


def test_get_trace_not_found(client):
    """main.py lines 260-262: get_trace for missing intent_id -> 404."""
    with patch("llm_lab.tracer.get_trace", new=AsyncMock(return_value=[])):
        resp = client.get("/trace/nonexistent")
        assert resp.status_code == 404


# ── main.py happy-path return-line coverage ──────────────────────────────


def test_compare_report_happy_path(client):
    """main.py line 237: compare_report returns HTML when both traces found."""
    trace_data = [
        {"seq": 1, "action": "call", "model": "gpt-4o", "output": "ok", "verdict": {"label": "pass"}, "cost": 0.001}
    ]
    with patch("llm_lab.tracer.get_trace", new=AsyncMock(return_value=trace_data)):
        resp = client.get("/compare/report/a/b")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_get_result_happy_path(client):
    """main.py line 255: get_result returns summary when events > 0."""
    summary = {"intent_id": "test", "events": 1, "model": "gpt-4o", "all_passed": True}
    with patch("llm_lab.tracer.get_summary", new=AsyncMock(return_value=summary)):
        resp = client.get("/result/test")
    assert resp.status_code == 200
    assert resp.json()["intent_id"] == "test"


def test_get_trace_happy_path(client):
    """main.py line 263: get_trace returns events when not empty."""
    events = [{"seq": 1, "action": "call", "output": "ok"}]
    with patch("llm_lab.tracer.get_trace", new=AsyncMock(return_value=events)):
        resp = client.get("/trace/test")
    assert resp.status_code == 200
    assert resp.json()[0]["action"] == "call"


def test_export_json_happy_path(client):
    """main.py line 274: export_json returns PlainTextResponse when events exist."""
    events = [{"seq": 1, "output": "exported"}]
    with patch("llm_lab.tracer.get_trace", new=AsyncMock(return_value=events)):
        resp = client.get("/export/json/test")
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    assert "exported" in resp.text


def test_static_mount():
    """main.py line 81: static dir mount is registered when static/ exists."""
    import importlib
    import os

    import llm_lab.main

    os.makedirs("static", exist_ok=True)
    importlib.reload(llm_lab.main)
    routes = [r.path for r in llm_lab.main.app.routes]
    assert "/static" in routes or any("/static" in str(r.path) for r in llm_lab.main.app.routes)


def test_lifespan_shutdown_calls_tracer():
    """main.py lines 72-74: lifespan triggers init_db + tracer.shutdown."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from llm_lab.main import app

    with patch("llm_lab.tracer.shutdown") as mock_shutdown:
        with TestClient(app) as client:
            client.get("/")
        mock_shutdown.assert_called_once()
