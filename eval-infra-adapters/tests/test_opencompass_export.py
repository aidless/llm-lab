"""Contracts for the pinned OpenCompass model and dataset config export."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import runpy
from pathlib import Path
from typing import Any

import pytest

from eval_infra_adapters.opencompass import (
    EXPECTED_CEVAL_JSONL_SHA256,
    OPENCOMPASS_SOURCE_FILES,
    export_opencompass_bundle,
    render_opencompass_html,
    render_opencompass_markdown,
    validate_openai_base_url,
)


def _dict_keywords(call: ast.Call) -> dict[str, ast.expr]:
    return {item.arg: item.value for item in call.keywords if item.arg is not None}


def _assigned_list_element(path: Path, variable: str) -> ast.Call:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == variable for target in node.targets):
            if not isinstance(node.value, ast.List) or len(node.value.elts) != 1:
                raise AssertionError(f"{variable} must be a one-item list")
            element = node.value.elts[0]
            if not isinstance(element, ast.Call):
                raise AssertionError(f"{variable}[0] must be a dict call")
            return element
    raise AssertionError(f"{variable} assignment not found")


def _env_default(expression: ast.expr, environment_name: str) -> Any:
    if not isinstance(expression, ast.Call):
        raise AssertionError("expected os.environ.get call")
    function = expression.func
    if not (
        isinstance(function, ast.Attribute)
        and function.attr == "get"
        and isinstance(function.value, ast.Attribute)
        and function.value.attr == "environ"
    ):
        raise AssertionError("expected os.environ.get call")
    assert isinstance(expression.args[0], ast.Constant)
    assert expression.args[0].value == environment_name
    assert len(expression.args) >= 2
    assert isinstance(expression.args[1], ast.Constant)
    return expression.args[1].value


def test_export_contains_official_opencompass_model_contract(tmp_path: Path) -> None:
    report = export_opencompass_bundle(tmp_path / "opencompass-export")
    model_config = tmp_path / "opencompass-export" / "model_openai_api.py"
    text = model_config.read_text(encoding="utf-8")

    assert "from opencompass.models import OpenAISDK" in text
    model = _assigned_list_element(model_config, "models")
    keywords = _dict_keywords(model)
    assert isinstance(keywords["type"], ast.Name)
    assert keywords["type"].id == "OpenAISDK"
    assert _env_default(keywords["path"], "EVAL_INFRA_OPENAI_MODEL") == report["model"][
        "served_model_name"
    ]
    assert _env_default(
        keywords["openai_api_base"], "EVAL_INFRA_OPENAI_BASE_URL"
    ) == "http://127.0.0.1:8000/v1"
    assert _env_default(
        keywords["key"], "EVAL_INFRA_OPENAI_API_KEY"
    ) == "EMPTY"
    assert "api_key_exported" not in text
    run_cfg = keywords["run_cfg"]
    assert isinstance(run_cfg, ast.Call)
    assert _dict_keywords(run_cfg)["num_gpus"].value == 0  # type: ignore[attr-defined]

    combined = tmp_path / "opencompass-export" / "eval-infra-openai-api.py"
    ast.parse(combined.read_text(encoding="utf-8"))
    assert "from opencompass.models import OpenAISDK" in combined.read_text(
        encoding="utf-8"
    )


def test_exported_dataset_config_expands_to_frozen_201_rows(tmp_path: Path) -> None:
    report = export_opencompass_bundle(tmp_path / "opencompass-export")
    config_path = tmp_path / "opencompass-export" / "dataset_ceval_201.py"
    namespace = runpy.run_path(str(config_path))
    datasets = namespace["datasets"]

    assert len(datasets) == 1
    dataset = datasets[0]
    assert dataset["data_type"] == "mcq"
    assert dataset["infer_method"] == "gen"
    assert dataset["abbr"] == "CEval-201"
    assert Path(dataset["path"]).is_file()
    assert Path(dataset["meta_path"]).is_file()
    assert report["dataset"]["rows"] == 201
    assert report["dataset"]["jsonl_sha256"] == EXPECTED_CEVAL_JSONL_SHA256


def test_export_report_has_truthful_boundaries_and_reports(tmp_path: Path) -> None:
    report = export_opencompass_bundle(tmp_path / "opencompass-export")

    assert report["status"] == "READY_TO_RUN_EXTERNALLY"
    assert report["official_interface_contract"]["status"] == "PASS_STATIC"
    assert report["endpoint"]["host_scope"] == "loopback"
    assert report["endpoint"]["api_key_exported"] is False
    assert report["endpoint"]["endpoint_called"] is False
    assert report["claims"]["official_opencompass_execution"] is False
    assert report["claims"]["gpu_execution"] is False
    assert report["claims"]["gpu_throughput"] is False
    assert report["claims"]["official_ceval_leaderboard"] is False
    assert report["claims"]["cpu_reported_as_gpu"] is False
    assert [run["concurrency"] for run in report["preserved_existing_evidence"]["runs"]] == [
        1,
        4,
        8,
    ]

    markdown = render_opencompass_markdown(report)
    page = render_opencompass_html(report)
    assert "不是生产安全系统" in markdown
    assert "不是生产安全系统" in page
    assert "BLOCKED_ENVIRONMENT" in markdown
    assert "BLOCKED_ENVIRONMENT" in page
    assert "Official OpenCompass execution: **False**" in markdown


def test_contract_manifest_uses_pinned_official_imports(tmp_path: Path) -> None:
    export_opencompass_bundle(tmp_path / "opencompass-export")
    contract = json.loads(
        (tmp_path / "opencompass-export" / "official-api-contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["opencompass_version"] == "0.5.4"
    assert contract["model_interface"]["import"] == "opencompass.models.OpenAISDK"
    assert contract["dataset_interface"]["import"] == "opencompass.datasets.CustomDataset"
    assert {
        item["path"]: item["sha256"] for item in contract["source_files"]
    } == OPENCOMPASS_SOURCE_FILES


def test_endpoint_policy_rejects_remote_and_credentials_by_default() -> None:
    with pytest.raises(ValueError, match="non-loopback"):
        validate_openai_base_url("https://example.com/v1")
    with pytest.raises(ValueError, match="credentials"):
        credential_url = "http://" + "user" + ":" + "demo" + "@127.0.0.1:8000/v1"
        validate_openai_base_url(credential_url)
    with pytest.raises(ValueError, match="/v1"):
        validate_openai_base_url("http://127.0.0.1:8000")

    explicit = validate_openai_base_url(
        "https://example.com/v1",
        allow_non_loopback_endpoint=True,
    )
    assert explicit["host_scope"] == "non_loopback_explicit"


def test_official_source_snapshot_when_explicitly_available() -> None:
    source_root = os.environ.get("OPENCOMPASS_SOURCE_DIR")
    if not source_root:
        pytest.skip("set OPENCOMPASS_SOURCE_DIR to verify the pinned source snapshot")
    root = Path(source_root)
    import hashlib

    for relative, expected in OPENCOMPASS_SOURCE_FILES.items():
        payload = (root / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected


def test_real_opencompass_runtime_when_installed(tmp_path: Path) -> None:
    if importlib.util.find_spec("opencompass") is None:
        pytest.skip("official OpenCompass is not installed")
    from eval_infra_adapters.opencompass import probe_opencompass_runtime

    output = tmp_path / "opencompass-export"
    export_opencompass_bundle(output)
    result = probe_opencompass_runtime(output)
    assert result["status"] == "PASS_CPU_PARSE", result
    assert result["dataset_rows_parsed"] == 201
    assert result["inference_executed"] is False
    assert result["gpu_executed"] is False


def test_reexport_manifest_does_not_hash_its_own_stale_copy(tmp_path: Path) -> None:
    import hashlib

    output = tmp_path / "opencompass-export"
    export_opencompass_bundle(output)
    # A second export must not record the manifest from the first run; a file
    # cannot hash itself, so a stale self-entry would always look tampered.
    report = export_opencompass_bundle(output)

    listed = {item["path"] for item in report["files"]}
    assert "export-manifest.json" not in listed
    for item in report["files"]:
        payload = (output / item["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
    manifest = json.loads((output / "export-manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"] == report["files"]
