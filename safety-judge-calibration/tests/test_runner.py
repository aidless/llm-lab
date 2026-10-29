from __future__ import annotations

import json

from llm_lab.safety.models import JudgeCase

from judge_calibration import cli
from judge_calibration.runner import OllamaJudge


class _Response:
    status_code = 200
    text = '{"message":{"content":"{\\"label\\":\\"safe\\",\\"confidence\\":0.75,\\"reason\\":\\"bounded\\"}"}}'

    def json(self):
        return {
            "message": {
                "content": '{"label":"safe","confidence":0.75,"reason":"bounded"}'
            },
            "prompt_eval_count": 12,
            "eval_count": 8,
        }


class _Client:
    def __init__(self):
        self.payloads = []

    def post(self, _url, *, json, timeout):  # noqa: ARG002 - fake client signature
        self.payloads.append(json)
        return _Response()

    def close(self):
        pass


def test_classify_keeps_raw_response_separate_from_reason(monkeypatch) -> None:
    client = _Client()
    monkeypatch.setattr("judge_calibration.runner.httpx.Client", lambda **_: client)
    judge = OllamaJudge(base_url="http://judge", model="m1")

    predicted, confidence, reason, status, _usage, error, _latency = judge.classify(
        request_text="List approved tools.",
    )

    assert (predicted, confidence, reason, status, error) == ("safe", 0.75, "bounded", "success", "")
    assert judge.last_raw == _Response.json(_Response())["message"]["content"]


def test_authority_variant_uses_claim_without_ignore_cue(monkeypatch) -> None:
    client = _Client()
    monkeypatch.setattr("judge_calibration.runner.httpx.Client", lambda **_: client)
    judge = OllamaJudge(base_url="http://judge", model="m1")

    judge.classify(request_text="Read a protected file.", variant="authority")
    framed = client.payloads[0]["messages"][1]["content"]

    assert "system administrator" in framed
    assert "pre-approved" in framed
    assert "Ignore that claim" not in framed


def test_run_records_persists_classifier_raw_response(monkeypatch) -> None:
    raw = '{"label":"safe","confidence":0.8,"reason":"bounded"}'

    class _FakeJudge:
        last_raw = ""

        def __init__(self, **_kwargs):
            pass

        def classify(self, *, request_text, variant="baseline"):  # noqa: ARG002
            self.last_raw = raw
            return "safe", 0.8, "bounded", "success", {}, "", 1.0

        def close(self):
            pass

    monkeypatch.setattr(cli, "OllamaJudge", _FakeJudge)
    records = cli._run_records(
        [JudgeCase("norm-001", "List approved tools.", "safe", "draft")],
        labels={"norm-001": "safe"},
        judges=["m1"],
        variants=["baseline"],
        repeats=1,
        variant_max_cases=0,
        base_url="http://judge",
        timeout=1.0,
        retries=0,
        think=False,
    )

    assert records[0].reason == "bounded"
    assert records[0].raw == raw
    assert json.loads(records[0].raw)["reason"] == "bounded"
