"""Export and verify OpenCompass API-backed evaluation configs.

The exporter pins the public OpenCompass interface used here to release 0.5.4.
It never starts a model, calls an endpoint, or turns a local CPU check into a
GPU execution claim.
"""

from __future__ import annotations

import hashlib
import html
import importlib.util
import ipaddress
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .diagnostics import host_diagnostics

OPENCOMPASS_VERSION = "0.5.4"
OPENCOMPASS_RELEASE_URL = "https://github.com/open-compass/opencompass/releases/tag/0.5.4"
OPENCOMPASS_SOURCE_URL = (
    "https://codeload.github.com/open-compass/opencompass/tar.gz/refs/tags/0.5.4"
)
OPENCOMPASS_SOURCE_SHA256 = (
    "6584bde074465a28e75ca8543c4dfc4147629925e118027968064389d2919ee0"
)
OPENCOMPASS_DOCS = {
    "custom_dataset": "https://opencompass.readthedocs.io/en/stable/advanced_guides/custom_dataset.html",
    "api_acceleration": "https://opencompass.readthedocs.io/en/stable/advanced_guides/accelerator_intro.html",
}
OPENCOMPASS_SOURCE_FILES = {
    "opencompass/models/openai_api.py": (
        "64f00bbc0b6abfda8c6a438ec48d9546de08921d81b62b6bef1057988e2e7d6a"
    ),
    "opencompass/datasets/custom.py": (
        "d98f2a5e0b7d06c60ef32aec5598a4f44f5c3dbfd99a3c8a2673bdcdcbf631c2"
    ),
    "opencompass/utils/run.py": (
        "c060790b28c85665e3149663c8b59fdc54235225bc1210eef375d47a3b09067c"
    ),
}

EXPECTED_CEVAL_ROWS = 201
EXPECTED_CEVAL_JSONL_SHA256 = (
    "a76b144743ebdff4cb45908a6e7d23253c84e7bd1fc5321ade2eeaf09fc2a802"
)
EXPECTED_CEVAL_META_SHA256 = (
    "eef85425a5892db89bb1228b449e51eae0b98bf8cb825c04b5982404356575a3"
)
EXPECTED_TASKS_SHA256 = (
    "33fdba0a807b721548cdac4a6339443c6edcd4bff00e895f62b77a3361bb9657"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CEVAL_SOURCE = REPO_ROOT / "evalops" / "ceval-201" / "opencompass" / "ceval-201qa.jsonl"
DEFAULT_CEVAL_META_SOURCE = DEFAULT_CEVAL_SOURCE.with_name(
    DEFAULT_CEVAL_SOURCE.name + ".meta.json"
)
DEFAULT_TASKS_SOURCE = REPO_ROOT / "evalops" / "ceval-201" / "tasks-v1.jsonl"
DEFAULT_EVALOPS_SUMMARY = REPO_ROOT / "evalops" / "evidence-20261027" / "summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _json_load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_openai_base_url(
    base_url: str,
    *,
    allow_non_loopback_endpoint: bool = False,
) -> dict[str, Any]:
    """Validate an OpenAISDK base URL and return a redacted safety summary."""
    candidate = base_url.strip()
    if not candidate or any(character.isspace() for character in candidate):
        raise ValueError("base URL must be non-empty and contain no whitespace")
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base URL scheme must be http or https")
    if parsed.username or parsed.password:
        raise ValueError("base URL must not embed credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain a query string or fragment")
    if not parsed.hostname:
        raise ValueError("base URL must contain a hostname")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        raise ValueError("OpenAISDK base URL path must end with /v1")
    loopback = _is_loopback_host(parsed.hostname)
    if not loopback and not allow_non_loopback_endpoint:
        raise ValueError(
            "non-loopback endpoints require --allow-non-loopback-endpoint "
            "and a separate authorization decision"
        )
    normalized = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    return {
        "base_url": normalized,
        "host": parsed.hostname,
        "scheme": parsed.scheme,
        "host_scope": "loopback" if loopback else "non_loopback_explicit",
        "credentials_embedded": False,
    }


def validate_ceval_export(data_path: Path, meta_path: Path) -> dict[str, Any]:
    """Validate the frozen 201-row C-Eval quick-evaluation input."""
    data_hash = _sha256(data_path)
    meta_hash = _sha256(meta_path)
    if data_hash != EXPECTED_CEVAL_JSONL_SHA256:
        raise ValueError(f"unexpected C-Eval JSONL SHA-256: {data_hash}")
    if meta_hash != EXPECTED_CEVAL_META_SHA256:
        raise ValueError(f"unexpected C-Eval meta SHA-256: {meta_hash}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        data_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        rows.append(row)
    if len(rows) != EXPECTED_CEVAL_ROWS:
        raise ValueError(f"expected {EXPECTED_CEVAL_ROWS} C-Eval rows, got {len(rows)}")
    required = ("question", "A", "B", "C", "D", "answer")
    missing = [
        index
        for index, row in enumerate(rows, 1)
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required)
    ]
    if missing:
        raise ValueError(f"C-Eval rows with missing fields: {missing[:10]}")
    answers = sorted({str(row["answer"]) for row in rows})
    if answers != ["A", "B", "C", "D"]:
        raise ValueError(f"unexpected C-Eval answers: {answers}")

    meta = _json_load(meta_path)
    if meta.get("data_type") != "mcq" or meta.get("infer_method") != "gen":
        raise ValueError("C-Eval meta must declare mcq/gen")
    if meta.get("input_columns") != ["question", "A", "B", "C", "D"]:
        raise ValueError("C-Eval meta input columns do not match the frozen data")
    if meta.get("output_column") != "answer":
        raise ValueError("C-Eval meta output column must be answer")
    return {
        "rows": len(rows),
        "answers": answers,
        "jsonl_sha256": data_hash,
        "meta_sha256": meta_hash,
        "data_type": meta["data_type"],
        "infer_method": meta["infer_method"],
    }


def _model_config_text(base_url: str, model: str) -> str:
    model_default = json.dumps(model, ensure_ascii=True)
    base_url_default = json.dumps(base_url, ensure_ascii=True)
    return f'''"""OpenAI-compatible model config for OpenCompass {OPENCOMPASS_VERSION}.

Generated by eval-infra-adapters. The endpoint must already be running.
The API key is read from the environment and is never written to this file.
"""

from __future__ import annotations

import os

from opencompass.models import OpenAISDK

api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
    reserved_roles=[dict(role="SYSTEM", api_role="SYSTEM")],
)

models = [
    dict(
        type=OpenAISDK,
        abbr=os.environ.get("EVAL_INFRA_OPENAI_ABBR", "openai-compatible-api"),
        path=os.environ.get("EVAL_INFRA_OPENAI_MODEL", {model_default}),
        key=os.environ.get("EVAL_INFRA_OPENAI_API_KEY", "EMPTY"),
        openai_api_base=os.environ.get(
            "EVAL_INFRA_OPENAI_BASE_URL",
            {base_url_default},
        ),
        tokenizer_path=os.environ.get("EVAL_INFRA_OPENAI_TOKENIZER_PATH", "gpt-4"),
        meta_template=api_meta_template,
        query_per_second=1,
        rpm_verbose=False,
        max_seq_len=2048,
        max_out_len=8,
        batch_size=1,
        retry=2,
        temperature=0.0,
        run_cfg=dict(num_gpus=0),
    )
]
'''


def _dataset_config_text() -> str:
    return '''"""Frozen 201-row C-Eval quick-evaluation config for OpenCompass."""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "ceval-201qa.jsonl"
_DATA_PATH = Path(os.environ.get("EVAL_INFRA_CEVAL_201_PATH", str(_DEFAULT_DATA_PATH)))
_META_PATH = Path(os.environ.get("EVAL_INFRA_CEVAL_201_META_PATH", f"{_DATA_PATH}.meta.json"))

datasets = [
    dict(
        path=str(_DATA_PATH),
        meta_path=str(_META_PATH),
        data_type="mcq",
        infer_method="gen",
        abbr="CEval-201",
    )
]
'''


def _combined_config_text(base_url: str, model: str) -> str:
    model_config = _model_config_text(base_url, model)
    dataset_config = _dataset_config_text()
    model_body = model_config.split("from opencompass.models import OpenAISDK", 1)[1]
    dataset_body = dataset_config.split("from pathlib import Path", 1)[1]
    return f'''"""Ready-to-run OpenCompass {OPENCOMPASS_VERSION} config exported by eval-infra.

The model endpoint is external to OpenCompass. No GPU is requested here.
"""

from __future__ import annotations

import os
from pathlib import Path

from opencompass.models import OpenAISDK
{model_body}
{dataset_body}
'''


def _export_readme(base_url: str, model: str, api_key_env: str) -> str:
    return f"""# OpenCompass 0.5.4 Export

> 本地安全评测，不是生产安全系统。 Local contract export, not a GPU run or an
> official C-Eval leaderboard result.

This directory contains configs pinned to the public OpenCompass 0.5.4
`OpenAISDK` model interface and its `CustomDataset` quick-evaluation protocol.
The dataset is the frozen 201-row public C-Eval validation subset.

## Files

- `eval-infra-openai-api.py`: combined model and dataset config.
- `model_openai_api.py`: standalone model config.
- `dataset_ceval_201.py`: standalone dataset config.
- `data/ceval-201qa.jsonl`: 201 rows, SHA-256
  `{EXPECTED_CEVAL_JSONL_SHA256}`.
- `data/ceval-201qa.jsonl.meta.json`: `mcq` + `gen` metadata.
- `env.example`: non-secret runtime parameters.
- `official-api-contract.json`: pinned official source/interface snapshot.

## Run

The OpenAI-compatible endpoint must already be running. The default endpoint is
`{base_url}` and the default served model name is `{model}`.

```bash
export {api_key_env}=EMPTY
opencompass opencompass-export/eval-infra-openai-api.py \\
  --work-dir opencompass-export/runs/ceval-201 \\
  --debug
```

For a config-only check:

```bash
opencompass opencompass-export/eval-infra-openai-api.py --dry-run --debug
```

## Safety boundary

- Exporting and config checks do not start a model, call an endpoint, or use a GPU.
- The default endpoint is loopback-only. A non-loopback endpoint requires the
  explicit `--allow-non-loopback-endpoint` exporter flag and separate authorization.
- API credentials are read only from `{api_key_env}`; no secret is exported.
- This is a 201-row validation subset, not the official C-Eval test set.
- No external OpenCompass execution, GPU execution, throughput, production SLO,
  or leaderboard result is claimed by this export.
"""


def official_interface_contract(
    *,
    runtime_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the pinned OpenCompass API contract used by the exporter."""
    return {
        "schema_version": "eval-infra.opencompass-contract.v1",
        "opencompass_version": OPENCOMPASS_VERSION,
        "release_url": OPENCOMPASS_RELEASE_URL,
        "source_archive": {
            "url": OPENCOMPASS_SOURCE_URL,
            "sha256": OPENCOMPASS_SOURCE_SHA256,
        },
        "source_files": [
            {"path": path, "sha256": digest}
            for path, digest in OPENCOMPASS_SOURCE_FILES.items()
        ],
        "docs": OPENCOMPASS_DOCS,
        "model_interface": {
            "import": "opencompass.models.OpenAISDK",
            "config_type": "OpenAISDK",
            "openai_api_base_contract": "OpenAI SDK base URL ending in /v1",
            "required_export_keys": [
                "type",
                "path",
                "key",
                "openai_api_base",
                "max_seq_len",
                "max_out_len",
                "batch_size",
                "run_cfg",
            ],
        },
        "dataset_interface": {
            "import": "opencompass.datasets.CustomDataset",
            "config_type": "CustomDataset",
            "quick_config_keys": [
                "path",
                "meta_path",
                "data_type",
                "infer_method",
                "abbr",
            ],
            "config_expansion": (
                "opencompass.utils.run.try_fill_in_custom_cfgs converts dicts "
                "without type through make_custom_dataset_config"
            ),
        },
        "contract_validation": {
            "mode": "pinned_official_source_static",
            "status": "PASS_STATIC",
            "runtime_probe": runtime_probe
            or {
                "status": "NOT_RUN",
                "reason": "official OpenCompass runtime is not installed",
            },
        },
    }


def probe_opencompass_runtime(export_dir: Path) -> dict[str, Any]:
    """Run a small CPU-only contract when a real OpenCompass install exists."""
    if importlib.util.find_spec("opencompass") is None:
        return {
            "status": "NOT_RUN",
            "reason": "official OpenCompass runtime is not installed",
            "inference_executed": False,
            "gpu_executed": False,
        }
    try:
        from mmengine.config import Config  # type: ignore[import-not-found]
        from opencompass.datasets import CustomDataset  # type: ignore[import-not-found]
        from opencompass.models import OpenAISDK  # type: ignore[import-not-found]
        from opencompass.utils.run import try_fill_in_custom_cfgs  # type: ignore[import-not-found]

        config = Config.fromfile(
            str(export_dir / "eval-infra-openai-api.py"),
            format_python_code=False,
        )
        config = try_fill_in_custom_cfgs(config)
        dataset_cfg = config["datasets"][0]
        data_path = export_dir / "data" / "ceval-201qa.jsonl"
        rows = CustomDataset.load(str(data_path))
        return {
            "status": "PASS_CPU_PARSE",
            "opencompass_version": getattr(
                __import__("opencompass"), "__version__", "unknown"
            ),
            "dataset_type": f"{dataset_cfg['type'].__module__}.{dataset_cfg['type'].__name__}",
            "dataset_rows_parsed": len(rows),
            "model_type": f"{OpenAISDK.__module__}.{OpenAISDK.__name__}",
            "inference_executed": False,
            "gpu_executed": False,
            "claim_boundary": "CPU config/data parsing only; no endpoint or model call",
        }
    except Exception as exc:  # pragma: no cover - depends on an optional runtime
        return {
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "inference_executed": False,
            "gpu_executed": False,
        }


def _preserved_evidence(
    evalops_summary_path: Path,
    tasks_path: Path,
) -> dict[str, Any]:
    summary = _json_load(evalops_summary_path)
    runs = summary.get("ollama_concurrency_runs", [])
    selected = {
        int(run["concurrency"]): run
        for run in runs
        if isinstance(run, dict) and run.get("concurrency") in {1, 4, 8}
    }
    if sorted(selected) != [1, 4, 8]:
        raise ValueError("existing EvalOps summary lacks C=1/4/8 evidence")
    for concurrency, run in selected.items():
        if run.get("tasks") != EXPECTED_CEVAL_ROWS or run.get("success") != EXPECTED_CEVAL_ROWS:
            raise ValueError(f"existing C={concurrency} run is not a complete 201-task run")

    run_artifacts: list[dict[str, Any]] = []
    evidence_root = evalops_summary_path.parent
    for concurrency in (1, 4, 8):
        run_dir = evidence_root / f"ollama-201-c{concurrency}"
        for name in ("report.json", "run-meta.json"):
            path = run_dir / name
            if not path.exists():
                raise ValueError(f"missing preserved evidence file: {_repo_relative(path)}")
            run_artifacts.append(
                {
                    "path": _repo_relative(path),
                    "sha256": _sha256(path),
                    "concurrency": concurrency,
                }
            )
    if _sha256(tasks_path) != EXPECTED_TASKS_SHA256:
        raise ValueError("the frozen 201-task file changed")
    return {
        "status": "PRESERVED",
        "summary_path": _repo_relative(evalops_summary_path),
        "summary_sha256": _sha256(evalops_summary_path),
        "tasks_path": _repo_relative(tasks_path),
        "tasks_sha256": EXPECTED_TASKS_SHA256,
        "runs": [
            {
                "concurrency": concurrency,
                "tasks": selected[concurrency]["tasks"],
                "success": selected[concurrency]["success"],
                "correct": selected[concurrency]["correct"],
            }
            for concurrency in (1, 4, 8)
        ],
        "artifacts": run_artifacts,
    }


def export_opencompass_bundle(
    output_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    api_key_env: str = "EVAL_INFRA_OPENAI_API_KEY",
    allow_non_loopback_endpoint: bool = False,
    data_source: Path = DEFAULT_CEVAL_SOURCE,
    meta_source: Path = DEFAULT_CEVAL_META_SOURCE,
    tasks_source: Path = DEFAULT_TASKS_SOURCE,
    evalops_summary: Path = DEFAULT_EVALOPS_SUMMARY,
) -> dict[str, Any]:
    """Write a self-contained OpenCompass export and return its evidence report."""
    if not model.strip():
        raise ValueError("model must be non-empty")
    if not api_key_env.isidentifier():
        raise ValueError("api key environment variable name is invalid")
    endpoint = validate_openai_base_url(
        base_url,
        allow_non_loopback_endpoint=allow_non_loopback_endpoint,
    )
    dataset_contract = validate_ceval_export(data_source, meta_source)
    preserved = _preserved_evidence(evalops_summary, tasks_source)

    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    exported_data = data_dir / "ceval-201qa.jsonl"
    exported_meta = data_dir / "ceval-201qa.jsonl.meta.json"
    shutil.copyfile(data_source, exported_data)
    shutil.copyfile(meta_source, exported_meta)

    files = {
        "eval-infra-openai-api.py": _combined_config_text(
            endpoint["base_url"], model
        ),
        "model_openai_api.py": _model_config_text(endpoint["base_url"], model),
        "dataset_ceval_201.py": _dataset_config_text(),
        "env.example": (
            f"{api_key_env}=EMPTY\n"
            f"EVAL_INFRA_OPENAI_BASE_URL={endpoint['base_url']}\n"
            f"EVAL_INFRA_OPENAI_MODEL={model}\n"
            "EVAL_INFRA_OPENAI_TOKENIZER_PATH=gpt-4\n"
        ),
        "README.md": _export_readme(endpoint["base_url"], model, api_key_env),
    }
    for name, content in files.items():
        (output_dir / name).write_text(content.rstrip() + "\n", encoding="utf-8")

    runtime_probe = probe_opencompass_runtime(output_dir)
    diagnostics = host_diagnostics()
    gpu_available = bool(diagnostics.get("gpu", {}).get("available"))
    gpu_acceptance = {
        "status": "BLOCKED_ENVIRONMENT" if not gpu_available else "AVAILABLE_NOT_EXECUTED",
        "gpu_available": gpu_available,
        "device_details": diagnostics.get("gpu", {}).get("devices", []),
        "real_gpu_executed": False,
        "reason": (
            "nvidia-smi unavailable; no GPU experiment requested or executed"
            if not gpu_available
            else "GPU detected but this export never authorizes or launches a GPU job"
        ),
    }
    contract = official_interface_contract(runtime_probe=runtime_probe)
    contract_path = output_dir / "official-api-contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # A manifest cannot hash itself, so a manifest left by a previous run must
    # never enter the file list; otherwise the recorded hash is always stale.
    exported_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "export-manifest.json"
    )
    export_manifest = {
        "schema_version": "eval-infra.opencompass-export.v1",
        "opencompass_version": OPENCOMPASS_VERSION,
        "generated_at_utc": __import__("time").strftime(
            "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()
        ),
        "status": "READY_TO_RUN_EXTERNALLY",
        "official_interface_contract": {
            **contract["contract_validation"],
            "snapshot": {
                "version": contract["opencompass_version"],
                "release_url": contract["release_url"],
                "source_archive": contract["source_archive"],
                "source_files": contract["source_files"],
                "model_interface": contract["model_interface"],
                "dataset_interface": contract["dataset_interface"],
            },
        },
        "dataset": {
            **dataset_contract,
            "source": _repo_relative(data_source),
            "exported_path": _repo_relative(exported_data),
            "official_ceval_test_set": False,
        },
        "endpoint": {
            **endpoint,
            "api_key_env": api_key_env,
            "api_key_exported": False,
            "endpoint_called": False,
        },
        "model": {
            "served_model_name": model,
            "config_type": "opencompass.models.OpenAISDK",
            "gpu_requested": False,
        },
        "gpu_acceptance": gpu_acceptance,
        "preserved_existing_evidence": preserved,
        "files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in exported_paths
        ],
        "claims": {
            "opencompass_config_exported": True,
            "official_interface_static_contract": True,
            "official_opencompass_runtime_contract": runtime_probe["status"]
            == "PASS_CPU_PARSE",
            "official_opencompass_execution": False,
            "gpu_execution": False,
            "gpu_throughput": False,
            "official_ceval_leaderboard": False,
            "cpu_reported_as_gpu": False,
        },
        "limitations": [
            "The export validates config/data contracts but does not execute OpenCompass.",
            "No endpoint, model, GPU, latency, throughput or leaderboard result is produced.",
            "The 201-row public validation subset is not the official C-Eval test set.",
        ],
    }
    (output_dir / "export-manifest.json").write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return export_manifest


def render_opencompass_markdown(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    endpoint = report["endpoint"]
    contract = report["official_interface_contract"]
    gpu = report["gpu_acceptance"]
    preserved = report["preserved_existing_evidence"]
    runs = "\n".join(
        f"| {run['concurrency']} | {run['success']}/{run['tasks']} | {run['correct']} |"
        for run in preserved["runs"]
    )
    return f"""# OpenCompass Export Contract

> **本地安全评测，不是生产安全系统。 Local contract export, not a production security system.**

- OpenCompass API snapshot: **{report["opencompass_version"]}**
- Export status: **{report["status"]}**
- Official interface contract: **{contract["status"]}** (`{contract["mode"]}`)
- Runtime CPU contract: **{contract["runtime_probe"]["status"]}**
- Dataset: **{dataset["rows"]} rows**, `{dataset["data_type"]}` / `{dataset["infer_method"]}`
- Dataset SHA-256: `{dataset["jsonl_sha256"]}`
- Endpoint scope: **{endpoint["host_scope"]}**
- API key exported: **{endpoint["api_key_exported"]}**
- Official OpenCompass execution: **False**
- GPU execution / throughput: **False / False**
- GPU acceptance: **{gpu["status"]}** ({gpu["reason"]})

## Exported files

| File | SHA-256 |
|---|---|
""" + "\n".join(
        f"| `{item['path']}` | `{item['sha256']}` |" for item in report["files"]
    ) + f"""

## Preserved existing 201-row CPU evidence

Source: `{preserved["summary_path"]}`

| C | Success | Correct |
|---:|---:|---:|
{runs}

These are existing local CPU Ollama measurements. They are preserved and
re-hashed, not rerun by this offline export.

## Safety boundary

- The exporter does not start a model, call the endpoint, install OpenCompass,
  or request GPU resources.
- The default endpoint must be loopback. Remote endpoints require an explicit
  override and separate authorization.
- Credentials are read only from `{endpoint["api_key_env"]}` and are never exported.
- Static source inspection is not an OpenCompass execution or leaderboard result.
- The 201-row public validation subset is not the official C-Eval test set.

## Reproduce

```bash
cd eval-infra-adapters
make reproduce
```
"""


def render_opencompass_html(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    endpoint = report["endpoint"]
    contract = report["official_interface_contract"]
    gpu = report["gpu_acceptance"]
    preserved = report["preserved_existing_evidence"]
    file_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(item['path'])}</code></td>"
        f"<td><code>{html.escape(item['sha256'])}</code></td>"
        "</tr>"
        for item in report["files"]
    )
    run_rows = "".join(
        "<tr>"
        f"<td>{run['concurrency']}</td>"
        f"<td>{run['success']}/{run['tasks']}</td>"
        f"<td>{run['correct']}</td>"
        "</tr>"
        for run in preserved["runs"]
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenCompass Export Contract</title>
  <style>
    :root {{ --ink:#18231f; --paper:#f4f0e6; --accent:#b64b2b; --moss:#436653; }}
    body {{ margin:0; background:linear-gradient(135deg,#e8e0d0,#f8f5ed 55%,#dce7df);
      color:var(--ink); font:16px/1.55 Georgia, "Noto Serif SC", serif; }}
    main {{ max-width:1080px; margin:40px auto; padding:34px 42px; background:rgba(255,255,255,.82);
      border:1px solid #cabfa9; box-shadow:0 18px 55px rgba(38,42,35,.14); }}
    h1 {{ font-size:clamp(2rem,5vw,4rem); line-height:1; margin:.1em 0 .35em; color:var(--moss); }}
    h2 {{ margin-top:2em; border-bottom:2px solid var(--accent); padding-bottom:.25em; }}
    .warning {{ border-left:6px solid var(--accent); padding:12px 16px; background:#fff4ee; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }}
    .card {{ background:#fbfaf6; border:1px solid #d8cebc; padding:14px; }}
    code {{ overflow-wrap:anywhere; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; }}
    th,td {{ text-align:left; padding:9px 11px; border-bottom:1px solid #ddd4c5; vertical-align:top; }}
    th {{ background:#e6ede7; }}
  </style>
</head>
<body><main>
  <h1>OpenCompass Export Contract</h1>
  <p class="warning"><strong>本地安全评测，不是生产安全系统。</strong>
  Local contract export, not a GPU run, production security system, or leaderboard result.</p>
  <div class="grid">
    <div class="card"><strong>API snapshot</strong><br>{html.escape(report["opencompass_version"])}</div>
    <div class="card"><strong>Static contract</strong><br>{html.escape(contract["status"])}</div>
    <div class="card"><strong>Runtime CPU probe</strong><br>{html.escape(contract["runtime_probe"]["status"])}</div>
    <div class="card"><strong>GPU acceptance</strong><br>{html.escape(gpu["status"])}</div>
  </div>
  <h2>Dataset and endpoint</h2>
  <ul>
    <li>Rows: <strong>{dataset["rows"]}</strong>; protocol:
      <code>{html.escape(dataset["data_type"])}/{html.escape(dataset["infer_method"])}</code></li>
    <li>JSONL SHA-256: <code>{html.escape(dataset["jsonl_sha256"])}</code></li>
    <li>Endpoint scope: <strong>{html.escape(endpoint["host_scope"])}</strong>;
      API key exported: <strong>{endpoint["api_key_exported"]}</strong></li>
    <li>Official OpenCompass execution: <strong>False</strong>;
      GPU execution / throughput: <strong>False / False</strong></li>
  </ul>
  <h2>Exported files</h2>
  <table><thead><tr><th>File</th><th>SHA-256</th></tr></thead><tbody>{file_rows}</tbody></table>
  <h2>Preserved existing 201-row CPU evidence</h2>
  <p>Source: <code>{html.escape(preserved["summary_path"])}</code></p>
  <table><thead><tr><th>C</th><th>Success</th><th>Correct</th></tr></thead>
  <tbody>{run_rows}</tbody></table>
  <h2>Safety boundary</h2>
  <p>No endpoint call, model inference, OpenCompass execution, GPU allocation, or
  throughput measurement is performed by this export. Credentials remain in
  <code>{html.escape(endpoint["api_key_env"])}</code> and are never exported.</p>
  <p>The 201-row public validation subset is not the official C-Eval test set.</p>
</main></body></html>
"""
