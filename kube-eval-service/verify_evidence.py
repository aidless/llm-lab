#!/usr/bin/env python3
"""Verify the local kind smoke evidence without trusting the report text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EVAL_NAMESPACE = "eval-lab"
SERVICE_NAME = "kube-eval-service"
IMAGE = "kube-eval-service:dev"
EXCLUDED = {"SHA256SUMS", "verification-result.json"}
PRIVATE_PATH_RE = re.compile(r"(?:/home|/Users)/[^/\s\"']+")
REQUIRED_NONEMPTY = {
    "boundary.md",
    "cluster-context.txt",
    "healthz.json",
    "readyz.json",
    "readyz-after-restart.json",
    "evaluate.json",
    "job-submit.json",
    "job-status.json",
    "job-idempotent.json",
    "job-before-restart.json",
    "job-after-worker-restart.json",
    "concurrency-smoke.json",
    "metrics.txt",
    "resources.txt",
    "deployment-live.json",
    "worker-deployment-live.json",
    "service-live.json",
    "configmap-live.json",
    "networkpolicy-live.json",
    "worker-networkpolicy-live.json",
    "pvc-live.json",
    "pdb-live.json",
    "pod-before.json",
    "pod-after.json",
    "pods-after.json",
    "worker-pod-before.json",
    "worker-pod-after.json",
    "worker-restart-request.txt",
    "events.txt",
    "restart-result.txt",
    "kind-clusters.txt",
    "kind-version.txt",
    "kubectl-version.json",
    "image-inspect.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain an object")
    return payload


def _read_json_list(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} does not contain a list")
    return payload


def _pod_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("kind") == "List" and isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]
    return [payload]


def _container(pod: dict[str, Any]) -> dict[str, Any] | None:
    containers = pod.get("spec", {}).get("containers", [])
    if not isinstance(containers, list):
        return None
    for container in containers:
        if isinstance(container, dict) and container.get("name") in {"service", "worker"}:
            return container
    return None


def _ready(pod: dict[str, Any]) -> bool:
    conditions = pod.get("status", {}).get("conditions", [])
    return any(
        item.get("type") == "Ready" and item.get("status") == "True" for item in conditions if isinstance(item, dict)
    )


def _policy_types(policy: dict[str, Any]) -> set[str]:
    values = policy.get("spec", {}).get("policyTypes", [])
    if not isinstance(values, list):
        return set()
    return {value for value in values if isinstance(value, str)}


def _verify_worker_network_policy(
    policy: dict[str, Any],
    errors: list[str],
) -> None:
    """An omitted or empty ingress list means deny-all ingress."""
    if not {"Ingress", "Egress"}.issubset(_policy_types(policy)):
        errors.append("worker network policy does not restrict both ingress and egress")
    spec = policy.get("spec", {})
    if not isinstance(spec, dict):
        errors.append("worker network policy spec is invalid")
        return
    ingress = spec.get("ingress", [])
    if ingress is None:
        return
    if not isinstance(ingress, list) or ingress:
        errors.append("worker network policy unexpectedly permits ingress")


def _pod_security(
    pod: dict[str, Any],
    *,
    label: str,
    errors: list[str],
) -> None:
    metadata = pod.get("metadata", {})
    if metadata.get("namespace") != EVAL_NAMESPACE:
        errors.append(f"{label} namespace mismatch")
    if pod.get("status", {}).get("phase") != "Running":
        errors.append(f"{label} is not Running")
    if not _ready(pod):
        errors.append(f"{label} has no Ready condition")
    pod_security = pod.get("spec", {}).get("securityContext", {})
    if pod_security.get("runAsNonRoot") is not True:
        errors.append(f"{label} does not enforce runAsNonRoot")
    if pod_security.get("runAsUser") != 10001:
        errors.append(f"{label} does not run as UID 10001")
    if pod_security.get("seccompProfile", {}).get("type") != "RuntimeDefault":
        errors.append(f"{label} does not use RuntimeDefault seccomp")
    if pod.get("spec", {}).get("automountServiceAccountToken") is not False:
        errors.append(f"{label} automounts a service-account token")
    container = _container(pod)
    if container is None:
        errors.append(f"{label} has no service container")
        return
    security = container.get("securityContext", {})
    if security.get("allowPrivilegeEscalation") is not False:
        errors.append(f"{label} allows privilege escalation")
    if security.get("readOnlyRootFilesystem") is not True:
        errors.append(f"{label} has a writable root filesystem")
    if security.get("capabilities", {}).get("drop") != ["ALL"]:
        errors.append(f"{label} does not drop all capabilities")
    resources = container.get("resources", {})
    if not resources.get("requests") or not resources.get("limits"):
        errors.append(f"{label} has no CPU/memory requests and limits")


def _verify_deployment(
    payload: dict[str, Any],
    errors: list[str],
) -> None:
    metadata = payload.get("metadata", {})
    if metadata.get("name") != SERVICE_NAME or metadata.get("namespace") != EVAL_NAMESPACE:
        errors.append("deployment identity mismatch")
    if payload.get("spec", {}).get("replicas") != 1:
        errors.append("deployment is not a single-replica smoke deployment")
    if payload.get("status", {}).get("readyReplicas") != 1:
        errors.append("deployment does not report one ready replica")
    template = payload.get("spec", {}).get("template", {}).get("spec", {})
    if template.get("automountServiceAccountToken") is not False:
        errors.append("deployment automounts a service-account token")
    pod_security = template.get("securityContext", {})
    if pod_security.get("runAsNonRoot") is not True:
        errors.append("deployment does not enforce runAsNonRoot")
    if pod_security.get("runAsUser") != 10001:
        errors.append("deployment does not run as UID 10001")
    containers = template.get("containers", [])
    container = next(
        (item for item in containers if isinstance(item, dict) and item.get("name") == "service"),
        None,
    )
    if container is None:
        errors.append("deployment has no service container")
        return
    if container.get("image") != IMAGE:
        errors.append("deployment image mismatch")
    if container.get("imagePullPolicy") != "Never":
        errors.append("deployment imagePullPolicy is not Never")
    if container.get("readinessProbe", {}).get("httpGet", {}).get("path") != "/readyz":
        errors.append("deployment readiness probe is not /readyz")
    if container.get("livenessProbe", {}).get("httpGet", {}).get("path") != "/healthz":
        errors.append("deployment liveness probe is not /healthz")
    security = container.get("securityContext", {})
    if security.get("allowPrivilegeEscalation") is not False:
        errors.append("deployment allows privilege escalation")
    if security.get("readOnlyRootFilesystem") is not True:
        errors.append("deployment has a writable root filesystem")
    if security.get("capabilities", {}).get("drop") != ["ALL"]:
        errors.append("deployment does not drop all capabilities")
    resources = container.get("resources", {})
    if not resources.get("requests") or not resources.get("limits"):
        errors.append("deployment has no resource requests and limits")


def _verify_manifests(
    *,
    service: dict[str, Any],
    configmap: dict[str, Any],
    network_policy: dict[str, Any],
    errors: list[str],
) -> None:
    if service.get("spec", {}).get("type") != "ClusterIP":
        errors.append("service is not ClusterIP")
    ports = service.get("spec", {}).get("ports", [])
    if not any(
        item.get("port") == 8080 and item.get("targetPort") == "http" for item in ports if isinstance(item, dict)
    ):
        errors.append("service port contract mismatch")
    data = configmap.get("data", {})
    if data.get("EVAL_MODE") != "deterministic":
        errors.append("configmap is not in deterministic mode")
    if data.get("EVAL_MAX_TASKS") != "500":
        errors.append("configmap task limit mismatch")
    if data.get("EVAL_MAX_CONCURRENCY") != "32":
        errors.append("configmap concurrency limit mismatch")
    if "Ingress" not in _policy_types(network_policy):
        errors.append("network policy does not declare Ingress")
    ingress = network_policy.get("spec", {}).get("ingress", [])
    if not ingress:
        errors.append("network policy has no ingress rule")
        return
    sources = ingress[0].get("from", [])
    namespaces = [
        item.get("namespaceSelector", {}).get("matchLabels", {}) for item in sources if isinstance(item, dict)
    ]
    if not any(labels.get("kubernetes.io/metadata.name") == EVAL_NAMESPACE for labels in namespaces):
        errors.append("network policy does not restrict ingress to eval-lab")


def _verify_runtime_evidence(
    evidence: Path,
    errors: list[str],
    checks: dict[str, Any],
) -> None:
    health = _read_json(evidence / "healthz.json")
    ready = _read_json(evidence / "readyz.json")
    ready_after = _read_json(evidence / "readyz-after-restart.json")
    if health.get("status") != "ok":
        errors.append("healthz is not ok")
    if ready.get("status") != "ready" or ready.get("mode") != "deterministic":
        errors.append("readyz is not deterministic-ready")
    if ready_after.get("status") != "ready":
        errors.append("readyz did not recover after pod replacement")

    evaluate = _read_json(evidence / "evaluate.json")
    summary = evaluate.get("summary", {})
    if evaluate.get("schema_version") != "kube-eval.response.v1":
        errors.append("evaluate response schema mismatch")
    if summary.get("tasks") != 4 or summary.get("correct") != 4:
        errors.append("small evaluate request did not complete four correct tasks")
    if summary.get("errors") != 0:
        errors.append("small evaluate request reports task errors")
    checks["small_request_tasks"] = summary.get("tasks")

    concurrency = _read_json(evidence / "concurrency-smoke.json")
    if concurrency.get("schema_version") != "kube-eval.concurrency-smoke.v1":
        errors.append("concurrency smoke schema mismatch")
    if concurrency.get("requests") != 24 or concurrency.get("max_workers") != 8:
        errors.append("concurrency smoke did not issue 24 requests with 8 workers")
    p50 = float(concurrency.get("p50_ms") or 0.0)
    p95 = float(concurrency.get("p95_ms") or 0.0)
    if p50 <= 0 or p95 <= 0 or p95 < p50:
        errors.append("concurrency latency metrics are invalid")
    checks["concurrency"] = {"requests": 24, "p50_ms": p50, "p95_ms": p95}

    metrics = (evidence / "metrics.txt").read_text(encoding="utf-8")
    request_match = re.search(r"^eval_service_requests_total (\d+)$", metrics, re.MULTILINE)
    error_match = re.search(r"^eval_service_errors_total (\d+)$", metrics, re.MULTILINE)
    if request_match is None or int(request_match.group(1)) < 25:
        errors.append("metrics do not account for the evaluation requests")
    if error_match is None or int(error_match.group(1)) != 0:
        errors.append("metrics report task errors")

    restart: dict[str, str] = {}
    for line in (evidence / "restart-result.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            restart[key] = value
    if not restart.get("deleted_pod") or restart.get("deleted_pod") == restart.get("replacement_pod"):
        errors.append("pod replacement evidence is missing")
    if restart.get("replacement_restart_count") != "0":
        errors.append("replacement pod reports restarts")
    if restart.get("job_id") is None or restart.get("restart_job_id") is None:
        errors.append("restart evidence does not identify durable jobs")
    if restart.get("restart_job_state") != "completed":
        errors.append("job did not recover after worker replacement")
    checks["restart"] = restart

    job_submit = _read_json(evidence / "job-submit.json")
    job_status = _read_json(evidence / "job-status.json")
    idempotent = _read_json(evidence / "job-idempotent.json")
    restart_job = _read_json(evidence / "job-after-worker-restart.json")
    if job_submit.get("existing") is not False:
        errors.append("first durable job submission unexpectedly reports existing=true")
    if job_status.get("state") != "completed":
        errors.append("durable worker job is not completed")
    if idempotent.get("existing") is not True:
        errors.append("idempotent resubmission created a duplicate job")
    if idempotent.get("job_id") != job_submit.get("job_id"):
        errors.append("idempotent resubmission returned a different job id")
    if restart_job.get("state") != "completed":
        errors.append("second durable job is not completed after worker restart")
    if restart_job.get("result", {}).get("summary", {}).get("correct") != 1:
        errors.append("recovered job did not return the expected result")
    checks["durable_jobs"] = {
        "job_id": job_submit.get("job_id"),
        "restart_job_id": restart.get("restart_job_id"),
        "idempotent": idempotent.get("existing"),
    }

    before = _read_json(evidence / "pod-before.json")
    after = _read_json(evidence / "pod-after.json")
    worker_before = _read_json(evidence / "worker-pod-before.json")
    worker_after = _read_json(evidence / "worker-pod-after.json")
    _pod_security(before, label="pod-before", errors=errors)
    _pod_security(after, label="pod-after", errors=errors)
    _pod_security(worker_before, label="worker-pod-before", errors=errors)
    _pod_security(worker_after, label="worker-pod-after", errors=errors)
    if worker_before.get("metadata", {}).get("uid") == worker_after.get("metadata", {}).get("uid"):
        errors.append("worker pod was not replaced")
    if before.get("metadata", {}).get("uid") == after.get("metadata", {}).get("uid"):
        errors.append("pod-before and pod-after have the same Kubernetes UID")

    _verify_deployment(
        _read_json(evidence / "deployment-live.json"),
        errors,
    )
    worker_deployment = _read_json(evidence / "worker-deployment-live.json")
    if worker_deployment.get("status", {}).get("readyReplicas") != 1:
        errors.append("worker deployment does not report one ready replica")
    _verify_manifests(
        service=_read_json(evidence / "service-live.json"),
        configmap=_read_json(evidence / "configmap-live.json"),
        network_policy=_read_json(evidence / "networkpolicy-live.json"),
        errors=errors,
    )
    worker_policy = _read_json(evidence / "worker-networkpolicy-live.json")
    _verify_worker_network_policy(worker_policy, errors)
    pvc = _read_json(evidence / "pvc-live.json")
    if pvc.get("status", {}).get("phase") != "Bound":
        errors.append("job database PVC is not bound")
    pdb = _read_json(evidence / "pdb-live.json")
    if pdb.get("spec", {}).get("minAvailable") != 0:
        errors.append("PDB evidence mismatch")

    clusters = (evidence / "kind-clusters.txt").read_text(encoding="utf-8").splitlines()
    if not clusters:
        errors.append("kind cluster evidence is empty")
    context: dict[str, str] = {}
    for line in (evidence / "cluster-context.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            context[key] = value
    cluster_name = context.get("cluster", "")
    if not cluster_name.startswith("eval-lab-") or len(cluster_name) <= len("eval-lab-"):
        errors.append("cluster name is not a unique eval-lab-* smoke cluster")
    if cluster_name not in clusters:
        errors.append("selected cluster is absent from kind cluster evidence")
    if context.get("namespace") != EVAL_NAMESPACE:
        errors.append("cluster context namespace mismatch")
    checks["cluster"] = cluster_name

    image_inspect = _read_json_list(evidence / "image-inspect.json")
    if not image_inspect or not isinstance(image_inspect[0], dict):
        errors.append("docker image inspection is empty")
    else:
        image = image_inspect[0]
        if image.get("Id") is None:
            errors.append("docker image has no immutable ID in the inspection")
        checks["image_id"] = image.get("Id")


def verify(evidence: Path) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, Any] = {}
    for name in sorted(REQUIRED_NONEMPTY):
        path = evidence / name
        if not path.exists():
            errors.append(f"missing required file: {name}")
            continue
        if path.stat().st_size == 0:
            errors.append(f"empty required file: {name}")
    optional_log = evidence / "port-forward.log"
    if not optional_log.exists():
        errors.append("missing port-forward.log")

    checksum_path = evidence / "SHA256SUMS"
    verified_files = 0
    listed_targets: set[str] = set()
    if not checksum_path.exists():
        errors.append("missing SHA256SUMS")
    else:
        for line_number, line in enumerate(
            checksum_path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                errors.append(f"SHA256SUMS:{line_number} malformed")
                continue
            expected_hash, relative = parts
            target = evidence / relative.strip()
            listed_targets.add(relative.strip())
            if not target.is_file():
                errors.append(f"SHA256SUMS target missing: {relative}")
                continue
            if _sha256(target) != expected_hash:
                errors.append(f"SHA256SUMS mismatch: {relative}")
                continue
            verified_files += 1
        actual_targets = {
            path.relative_to(evidence).as_posix()
            for path in evidence.rglob("*")
            if path.is_file() and path.relative_to(evidence).as_posix() not in EXCLUDED
        }
        if listed_targets != actual_targets:
            errors.append("SHA256SUMS does not cover exactly the evidence files")
    checks["checksum_files_verified"] = verified_files

    boundary = (evidence / "boundary.md").read_text(encoding="utf-8")
    if "不是生产安全系统" not in boundary:
        errors.append("boundary.md lacks the local-evaluation disclaimer")
    for path in evidence.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt", ".log"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if PRIVATE_PATH_RE.search(text):
            errors.append(f"private workstation path leaked into {path.relative_to(evidence)}")

    runtime_required = {
        "healthz.json",
        "readyz.json",
        "readyz-after-restart.json",
        "evaluate.json",
        "concurrency-smoke.json",
        "metrics.txt",
        "restart-result.txt",
        "pod-before.json",
        "pod-after.json",
        "deployment-live.json",
        "service-live.json",
        "configmap-live.json",
        "networkpolicy-live.json",
        "kind-clusters.txt",
        "image-inspect.json",
    }
    if not any(f"missing required file: {name}" in errors for name in runtime_required):
        _verify_runtime_evidence(evidence, errors, checks)

    result = {
        "schema_version": "kube-eval.verification.v1",
        "technical_ok": not errors,
        "acceptance_ok": not errors,
        "checks": checks,
        "errors": errors,
        "evidence_boundary": (
            "Local single-node kind smoke on CPU. This is not a production "
            "Kubernetes deployment, GPU serving benchmark, or SLO claim."
        ),
    }
    (evidence / "verification-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=str(ROOT / "evidence"))
    args = parser.parse_args()
    result = verify(Path(args.evidence))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["technical_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
