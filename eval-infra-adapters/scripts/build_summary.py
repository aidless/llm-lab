#!/usr/bin/env python3
"""Combine real Ollama evidence with the blocked GPU adapter boundary."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
EVALOPS_SUMMARY = REPO_ROOT / "evalops" / "evidence-20261027" / "summary.json"
EVIDENCE = ROOT / "evidence"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain an object")
    return payload


def _markdown(summary: dict[str, Any]) -> str:
    runs = summary["ollama_201"]["runs"]
    run_rows = "\n".join(
        f"| {run['concurrency']} | {run['success']}/201 | "
        f"{run['correct']} | {run['p50_ms']:.1f} | {run['p95_ms']:.1f} | "
        f"{run['throughput_tasks_per_second']:.3f} | {run['wall_seconds']:.1f} |"
        for run in runs
    )
    consistency_rows = "\n".join(
        f"| `{pair['pair']}` | {pair['agree']}/{pair['total']} | {pair['rate']:.2%} |"
        for pair in summary["ollama_201"]["repeat_consistency"]
    )
    opencompass = summary["opencompass_export"]
    return f"""# Eval Infra Adapter Evidence

> **本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.**

- Contract smoke: **{"PASS" if summary["contract_smoke"]["ok"] else "FAIL"}**
- Contract smoke identity: **{summary["contract_smoke"]["implementation"]}**,
  real vLLM/LMDeploy executed: **False**
- OpenCompass config export: **{opencompass["status"]}**
- OpenCompass official interface contract: **{opencompass["official_interface_contract"]["status"]}**
- OpenCompass runtime CPU probe: **{opencompass["official_interface_contract"]["runtime_probe"]["status"]}**
- Real Ollama OpenAI-compatible probe: **{"PASS" if summary["real_ollama_probe"]["ok"] else "FAIL"}**
- GPU acceptance: **{summary["gpu_acceptance"]["status"]}**
- vLLM / LMDeploy executed: **False**

## Existing 201-task Ollama run

Source: `{summary["ollama_201"]["source"]}`
Task-set SHA-256: `{summary["ollama_201"]["task_set_hash"]}`
Model: `{summary["ollama_201"]["model"]}`

| C | Success | Correct | p50 ms | p95 ms | Tasks/s | Wall s |
|---:|---:|---:|---:|---:|---:|---:|
{run_rows}

## Concurrency consistency

| Pair | Agreement | Rate |
|---|---:|---:|
{consistency_rows}

These are local CPU Ollama results, not GPU serving throughput, not a production SLO,
and not an official C-Eval leaderboard result.

## OpenCompass export

- Config: `{opencompass["combined_config"]}`
- Frozen C-Eval rows: **{opencompass["dataset"]["rows"]}**
- Dataset SHA-256: `{opencompass["dataset"]["jsonl_sha256"]}`
- Endpoint scope: **{opencompass["endpoint"]["host_scope"]}**
- API key exported: **{opencompass["endpoint"]["api_key_exported"]}**
- Official OpenCompass execution: **{opencompass["claims"]["official_opencompass_execution"]}**

The export is executable when a compatible endpoint and an official OpenCompass
installation are supplied separately. The offline contract verifies the pinned
imports/fields and the frozen 201-row input; it does not start a model or call an
endpoint. The fake OpenAI server remains a fixture, not a vLLM/LMDeploy execution.

## GPU blockers

- NVIDIA device available: **{summary["gpu_acceptance"]["gpu_available"]}**
- vLLM module installed: **{summary["gpu_acceptance"]["vllm_installed"]}**
- LMDeploy module installed: **{summary["gpu_acceptance"]["lmdeploy_installed"]}**
- Real C=1/4/8 GPU OOM and throughput experiment: **not executed**

The adapter command generation, process lifecycle, HTTP contract, error taxonomy,
OpenCompass config/data export and 201-task runner integration are tested. A real
GPU acceptance run remains blocked by the host environment and requires separately
authorized GPU resources.
"""


def main() -> int:
    contract = _json(EVIDENCE / "report.json")
    opencompass = _json(EVIDENCE / "opencompass-export.json")
    probe = _json(EVIDENCE / "ollama-probe.json")
    environment = _json(EVIDENCE / "environment.json")
    evalops = _json(EVALOPS_SUMMARY)
    runs = evalops["ollama_concurrency_runs"]
    selected = [run for run in runs if run["concurrency"] in {1, 4, 8}]
    if sorted(run["concurrency"] for run in selected) != [1, 4, 8]:
        raise SystemExit("EvalOps evidence does not contain C=1/4/8 Ollama runs")
    if any(run["tasks"] != 201 or run["success"] != 201 for run in selected):
        raise SystemExit("EvalOps Ollama runs are not complete 201-task executions")
    summary = {
        "schema_version": "eval-infra.integration-summary.v1",
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boundary": (
            "本地安全评测，不是生产安全系统。 Local CPU evidence only; "
            "no production security or GPU serving claim."
        ),
        "contract_smoke": {
            "ok": bool(contract.get("ok")),
            "backends": [str(item.get("backend")) for item in contract.get("backends", [])],
            "implementation": "deterministic_openai_compatible_fixture",
            "real_backends_executed": False,
            "source_sha256": _sha256(EVIDENCE / "report.json"),
        },
        "opencompass_export": {
            "status": opencompass["status"],
            "source_sha256": _sha256(EVIDENCE / "opencompass-export.json"),
            "combined_config": "opencompass-export/eval-infra-openai-api.py",
            "official_interface_contract": opencompass["official_interface_contract"],
            "dataset": opencompass["dataset"],
            "endpoint": opencompass["endpoint"],
            "claims": opencompass["claims"],
            "limitations": opencompass["limitations"],
        },
        "real_ollama_probe": {
            "ok": bool(probe.get("ok")),
            "model": probe.get("native", {}).get("model"),
            "model_digest": probe.get("native", {}).get("model_digest"),
            "source_sha256": _sha256(EVIDENCE / "ollama-probe.json"),
        },
        "ollama_201": {
            "source": "evalops/evidence-20261027/summary.json",
            "source_sha256": _sha256(EVALOPS_SUMMARY),
            "task_set_hash": evalops["dataset"]["tasks_sha256"],
            "model": selected[0]["model"],
            "runs": selected,
            "repeat_consistency": [
                evalops["repeat_consistency"]["c1_vs_c4"],
                evalops["repeat_consistency"]["c1_vs_c8"],
                evalops["repeat_consistency"]["c4_vs_c8"],
            ],
        },
        "gpu_acceptance": {
            "status": "BLOCKED_ENVIRONMENT",
            "gpu_available": bool(environment.get("gpu", {}).get("available")),
            "vllm_installed": bool(environment.get("python_modules", {}).get("vllm")),
            "lmdeploy_installed": bool(environment.get("python_modules", {}).get("lmdeploy")),
            "real_gpu_executed": False,
            "required_before_claim": [
                "record GPU model, VRAM, driver and backend version",
                "record C=1/4/8 wall time, p50/p95, retries, failures and result agreement",
                "capture startup logs and classify any OOM",
                "checksum the full evidence package",
            ],
        },
        "claims": {
            "openai_compatible_contract": True,
            "fake_openai_fixture": True,
            "real_vllm_execution": False,
            "real_lmdeploy_execution": False,
            "opencompass_config_exported": True,
            "official_opencompass_execution": False,
            "official_ceval_leaderboard": False,
            "real_local_ollama_probe": bool(probe.get("ok")),
            "gpu_throughput": False,
            "production_slo": False,
        },
        "limitations": [
            "The vLLM and LMDeploy contract smoke uses a deterministic localhost fixture.",
            "The 201-task C=1/4/8 measurements were collected with Ollama on CPU.",
            "The OpenCompass export is statically checked against a pinned official API; the external runner was not executed here.",
            "No GPU, multi-GPU, OOM, production latency or official leaderboard claim is made.",
        ],
    }
    (EVIDENCE / "integration-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (EVIDENCE / "integration-summary.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
