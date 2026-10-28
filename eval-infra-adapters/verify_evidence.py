#!/usr/bin/env python3
"""Verify inf-runtime smoke evidence and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
EVALOPS_SUMMARY = REPO_ROOT / "evalops" / "evidence-20261027" / "summary.json"
EXCLUDED = {"SHA256SUMS", "verification-result.json"}
PRIVATE_PATH_RE = re.compile(r"(?:/home|/Users)/[^/\s\"']+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain an object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=str(ROOT / "evidence"))
    args = parser.parse_args()
    evidence = Path(args.evidence)
    errors: list[str] = []
    report = _read_json(evidence / "report.json")
    if report.get("ok") is not True:
        errors.append("contract smoke report is not PASS")
    backends = report.get("backends", [])
    if sorted(item.get("backend") for item in backends) != ["lmdeploy", "vllm"]:
        errors.append("expected vllm and lmdeploy contract results")
    for item in backends:
        process = item.get("managed_process") or {}
        probe = item.get("probe") or {}
        if process.get("ready") is not True or process.get("stopped") is not True:
            errors.append(f"{item.get('backend')}: process lifecycle failed")
        if probe.get("ok") is not True:
            errors.append(f"{item.get('backend')}: HTTP probe failed")
        if not item.get("generated_serve_command"):
            errors.append(f"{item.get('backend')}: serve command missing")

    probe = _read_json(evidence / "ollama-probe.json")
    if probe.get("ok") is not True or (probe.get("probe") or {}).get("ok") is not True:
        errors.append("real Ollama OpenAI-compatible probe did not pass")
    if probe.get("backend") != "ollama-openai-compatible":
        errors.append("Ollama probe backend label is missing")
    if not (probe.get("native") or {}).get("model_digest"):
        errors.append("Ollama probe lacks an immutable model digest")

    summary = _read_json(evidence / "integration-summary.json")
    if summary.get("contract_smoke", {}).get("source_sha256") != _sha256(evidence / "report.json"):
        errors.append("integration summary is not bound to the current contract report")
    if summary.get("real_ollama_probe", {}).get("source_sha256") != _sha256(evidence / "ollama-probe.json"):
        errors.append("integration summary is not bound to the current Ollama probe")
    if not EVALOPS_SUMMARY.exists():
        errors.append("EvalOps 201-task evidence summary is missing")
    elif summary.get("ollama_201", {}).get("source_sha256") != _sha256(EVALOPS_SUMMARY):
        errors.append("integration summary is not bound to the current EvalOps summary")
    runs = summary.get("ollama_201", {}).get("runs", [])
    if sorted(run.get("concurrency") for run in runs) != [1, 4, 8]:
        errors.append("Ollama C=1/4/8 evidence is incomplete")
    for run in runs:
        if run.get("tasks") != 201 or run.get("success") != 201:
            errors.append(f"Ollama C={run.get('concurrency')} did not complete 201 tasks")
        if float(run.get("p50_ms") or 0) <= 0 or float(run.get("p95_ms") or 0) <= 0:
            errors.append(f"Ollama C={run.get('concurrency')} latency evidence is invalid")
    gpu = summary.get("gpu_acceptance", {})
    if gpu.get("real_gpu_executed") is not False or gpu.get("status") != "BLOCKED_ENVIRONMENT":
        errors.append("GPU acceptance boundary is not explicit")
    claims = summary.get("claims", {})
    for key in ("real_vllm_execution", "real_lmdeploy_execution", "gpu_throughput", "production_slo"):
        if claims.get(key) is not False:
            errors.append(f"summary overclaims {key}")
    for name in ("report.md", "integration-summary.md"):
        text = (evidence / name).read_text(encoding="utf-8")
        if "不是生产安全系统" not in text:
            errors.append(f"{name} lacks the local-evaluation disclaimer")

    checksums = evidence / "SHA256SUMS"
    checked = 0
    listed_targets: set[str] = set()
    if not checksums.exists():
        errors.append("SHA256SUMS missing")
    else:
        for line_number, line in enumerate(checksums.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                errors.append(f"SHA256SUMS:{line_number} malformed")
                continue
            expected, relative = parts
            target = evidence / relative.strip()
            listed_targets.add(relative.strip())
            if not target.exists():
                errors.append(f"missing checksum target {relative}")
            elif _sha256(target) != expected:
                errors.append(f"checksum mismatch {relative}")
            else:
                checked += 1
        actual_targets = {
            path.relative_to(evidence).as_posix()
            for path in evidence.rglob("*")
            if path.is_file() and path.relative_to(evidence).as_posix() not in EXCLUDED
        }
        if listed_targets != actual_targets:
            errors.append("SHA256SUMS does not cover exactly the evidence files")

    for path in evidence.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt", ".log"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if PRIVATE_PATH_RE.search(text):
            errors.append(f"private workstation path leaked into {path.relative_to(evidence)}")

    result = {
        "schema_version": "eval-infra.verification.v1",
        "ok": not errors,
        "checksum_files_verified": checked,
        "checks": {
            "contract_backends": sorted(item.get("backend") for item in backends),
            "ollama_model": (probe.get("native") or {}).get("model"),
            "ollama_model_digest": (probe.get("native") or {}).get("model_digest"),
            "ollama_concurrency": sorted(run.get("concurrency") for run in runs),
            "gpu_acceptance": gpu.get("status"),
        },
        "errors": errors,
    }
    (evidence / "verification-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
