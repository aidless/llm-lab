#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
TOOLS="${KIND_TOOLS_DIR:-$REPO_ROOT/.tools/bin}"
CLUSTER="${KIND_CLUSTER_NAME:-eval-lab}"
KIND_NODE_IMAGE="${KIND_NODE_IMAGE:-}"
IMAGE="kube-eval-service:dev"
EVIDENCE="${EVIDENCE_DIR:-$ROOT/evidence}"
PORT_FORWARD_PORT="${PORT_FORWARD_PORT:-18080}"
BASE_IMAGE="${KUBE_EVAL_BASE_IMAGE:-python:3.12-slim}"
INSTALL_DEPENDENCIES="${KUBE_EVAL_INSTALL_DEPENDENCIES:-1}"

if [[ -e "$EVIDENCE" ]]; then
  if [[ ! -d "$EVIDENCE" ]]; then
    echo "evidence path exists and is not a directory: $EVIDENCE" >&2
    exit 1
  fi
  if find "$EVIDENCE" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "evidence directory is not empty: $EVIDENCE" >&2
    echo "choose a new EVIDENCE_DIR or archive the previous evidence" >&2
    exit 1
  fi
fi
mkdir -p "$TOOLS" "$EVIDENCE"

if [[ ! -x "$TOOLS/kind" ]]; then
  curl -fsSL --retry 3 -o "$TOOLS/kind" \
    https://kind.sigs.k8s.io/dl/v0.33.0/kind-linux-amd64
  chmod +x "$TOOLS/kind"
fi

if [[ ! -x "$TOOLS/kubectl" ]]; then
  KUBECTL_VERSION="$(curl -fsSL https://dl.k8s.io/release/stable.txt)"
  curl -fsSL --retry 3 -o "$TOOLS/kubectl" \
    "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
  chmod +x "$TOOLS/kubectl"
fi

export PATH="$TOOLS:$PATH"

if kind get clusters | grep -Fxq "$CLUSTER"; then
  echo "refusing to reuse an existing cluster: $CLUSTER" >&2
  echo "set KIND_CLUSTER_NAME to a new unique name" >&2
  exit 1
fi

create_args=(--name "$CLUSTER" --wait 120s)
if [[ -n "$KIND_NODE_IMAGE" ]]; then
  create_args+=(--image "$KIND_NODE_IMAGE")
fi
kind create cluster "${create_args[@]}"
kubectl config use-context "kind-$CLUSTER" >/dev/null

docker build \
  --build-arg "PYTHON_BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "INSTALL_DEPENDENCIES=$INSTALL_DEPENDENCIES" \
  -t "$IMAGE" "$ROOT"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
kind load docker-image --name "$CLUSTER" "$IMAGE"

kubectl apply -f "$ROOT/deploy/namespace.yaml"
kubectl apply -f "$ROOT/deploy/pvc.yaml"
kubectl apply -f "$ROOT/deploy/configmap.yaml"
kubectl apply -f "$ROOT/deploy/deployment.yaml"
kubectl apply -f "$ROOT/deploy/worker.yaml"
kubectl apply -f "$ROOT/deploy/service.yaml"
kubectl apply -f "$ROOT/deploy/networkpolicy.yaml"
kubectl apply -f "$ROOT/deploy/worker-networkpolicy.yaml"
kubectl apply -f "$ROOT/deploy/pdb.yaml"
kubectl -n eval-lab rollout status deployment/kube-eval-service --timeout=120s
kubectl -n eval-lab rollout status deployment/kube-eval-worker --timeout=120s

kubectl -n eval-lab port-forward service/kube-eval-service "${PORT_FORWARD_PORT}:8080" \
  >"$EVIDENCE/port-forward.log" 2>&1 &
PORT_FORWARD_PID=$!
cleanup() {
  kill "$PORT_FORWARD_PID" 2>/dev/null || true
}
trap cleanup EXIT

BASE_URL="http://127.0.0.1:${PORT_FORWARD_PORT}"
for _ in $(seq 1 30); do
  if curl -fsS "$BASE_URL/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl -fsS "$BASE_URL/healthz" >"$EVIDENCE/healthz.json"
curl -fsS "$BASE_URL/readyz" >"$EVIDENCE/readyz.json"

JOB_PAYLOAD='{"idempotency_key":"kind-smoke-job-v1","concurrency":2,"tasks":[{"id":"job-a","prompt":"2+2","gold":"4"},{"id":"job-b","prompt":"3+3","gold":"6"}]}'
curl -fsS -X POST "$BASE_URL/v1/jobs" \
  -H 'content-type: application/json' \
  -d "$JOB_PAYLOAD" >"$EVIDENCE/job-submit.json"
JOB_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$EVIDENCE/job-submit.json")"
for _ in $(seq 1 60); do
  curl -fsS "$BASE_URL/v1/jobs/$JOB_ID" >"$EVIDENCE/job-status.json"
  JOB_STATE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$EVIDENCE/job-status.json")"
  if [[ "$JOB_STATE" == "completed" || "$JOB_STATE" == "failed" ]]; then
    break
  fi
  sleep 1
done
if [[ "${JOB_STATE:-}" != "completed" ]]; then
  echo "durable worker job did not complete: ${JOB_STATE:-missing}" >&2
  exit 1
fi
curl -fsS -X POST "$BASE_URL/v1/jobs" \
  -H 'content-type: application/json' \
  -d "$JOB_PAYLOAD" >"$EVIDENCE/job-idempotent.json"
IDEMPOTENT_EXISTING="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["existing"])' "$EVIDENCE/job-idempotent.json")"
if [[ "$IDEMPOTENT_EXISTING" != "True" ]]; then
  echo "idempotent job resubmission created a new job" >&2
  exit 1
fi

printf 'cluster=%s\nnamespace=eval-lab\nport_forward=%s\nbase_image=%s\ninstall_dependencies=%s\nimage=%s\nimage_id=%s\ntask_delay_seconds=8\nlease_seconds=4\nmax_attempts=2\n' \
  "$CLUSTER" "$PORT_FORWARD_PORT" "$BASE_IMAGE" "$INSTALL_DEPENDENCIES" "$IMAGE" "$IMAGE_ID" \
  >"$EVIDENCE/cluster-context.txt"
cat >"$EVIDENCE/boundary.md" <<'MD'
# Evidence boundary

> **本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.**

This is a single-node kind smoke on a local CPU host. It verifies service
plumbing, bounded input, health checks, pod replacement and checksummed
evidence. It is not a production Kubernetes deployment, autoscaling study,
multi-tenant security proof, GPU serving benchmark or SLO claim. The local
single-node ReadWriteOnce PVC does not demonstrate safe multi-node storage.
NetworkPolicy objects are recorded and checked structurally, but this smoke
does not prove that the kind CNI enforces them. The image was built locally
from an available base image and was not published.
MD

kubectl -n eval-lab get deployment,service,pods,networkpolicy,pvc,pdb -o wide >"$EVIDENCE/resources.txt"
kubectl -n eval-lab get deployment kube-eval-service -o json >"$EVIDENCE/deployment-live.json"
kubectl -n eval-lab get deployment kube-eval-worker -o json >"$EVIDENCE/worker-deployment-live.json"
kubectl -n eval-lab get service kube-eval-service -o json >"$EVIDENCE/service-live.json"
kubectl -n eval-lab get configmap kube-eval-config -o json >"$EVIDENCE/configmap-live.json"
kubectl -n eval-lab get networkpolicy kube-eval-service -o json >"$EVIDENCE/networkpolicy-live.json"
kubectl -n eval-lab get networkpolicy kube-eval-worker -o json >"$EVIDENCE/worker-networkpolicy-live.json"
kubectl -n eval-lab get pvc kube-eval-data -o json >"$EVIDENCE/pvc-live.json"
kubectl -n eval-lab get pdb kube-eval-service -o json >"$EVIDENCE/pdb-live.json"
kubectl -n eval-lab get events --sort-by=.lastTimestamp >"$EVIDENCE/events.txt"

PAYLOAD='{"concurrency":4,"tasks":[{"id":"a","prompt":"2+2","gold":"4"},{"id":"b","prompt":"3+3","gold":"6"},{"id":"c","prompt":"4+4","gold":"8"},{"id":"d","prompt":"5+5","gold":"10"}]}'
curl -fsS -X POST "$BASE_URL/v1/evaluate" \
  -H 'content-type: application/json' \
  -d "$PAYLOAD" >"$EVIDENCE/evaluate.json"

python3 - "$EVIDENCE" "$PAYLOAD" "$PORT_FORWARD_PORT" <<'PY'
import concurrent.futures
import json
import pathlib
import statistics
import sys
import time
import urllib.request

evidence = pathlib.Path(sys.argv[1])
payload = sys.argv[2].encode()
port = sys.argv[3]

def one(index: int) -> float:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/evaluate",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=20) as response:
        body = json.load(response)
    if body["summary"]["errors"] != 0:
        raise RuntimeError(f"request {index} produced task errors")
    return (time.perf_counter() - started) * 1000.0

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    latencies = list(pool.map(one, range(24)))

result = {
    "schema_version": "kube-eval.concurrency-smoke.v1",
    "requests": len(latencies),
    "max_workers": 8,
    "p50_ms": statistics.median(latencies),
    "p95_ms": sorted(latencies)[int((len(latencies) - 1) * 0.95)],
    "min_ms": min(latencies),
    "max_ms": max(latencies),
}
(evidence / "concurrency-smoke.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

curl -fsS "$BASE_URL/metrics" >"$EVIDENCE/metrics.txt"

RUNNING_JOB_PAYLOAD='{"idempotency_key":"kind-restart-running-v1","concurrency":1,"tasks":[{"id":"restart-running","prompt":"4+4","gold":"8"}]}'
curl -fsS -X POST "$BASE_URL/v1/jobs" \
  -H 'content-type: application/json' \
  -d "$RUNNING_JOB_PAYLOAD" >"$EVIDENCE/restart-running-submit.json"
RUNNING_JOB_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$EVIDENCE/restart-running-submit.json")"

RUNNING_BEFORE_STATE=""
for _ in $(seq 1 30); do
  curl -fsS "$BASE_URL/v1/jobs/$RUNNING_JOB_ID" >"$EVIDENCE/job-before-restart.json"
  RUNNING_BEFORE_STATE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$EVIDENCE/job-before-restart.json")"
  if [[ "$RUNNING_BEFORE_STATE" == "running" ]]; then
    break
  fi
  if [[ "$RUNNING_BEFORE_STATE" == "completed" || "$RUNNING_BEFORE_STATE" == "failed" ]]; then
    echo "restart test job finished before worker interruption: $RUNNING_BEFORE_STATE" >&2
    exit 1
  fi
  sleep 1
done
if [[ "$RUNNING_BEFORE_STATE" != "running" ]]; then
  echo "restart test job did not enter running state: ${RUNNING_BEFORE_STATE:-missing}" >&2
  exit 1
fi

QUEUED_JOB_PAYLOAD='{"idempotency_key":"kind-restart-queued-v1","concurrency":1,"tasks":[{"id":"restart-queued","prompt":"5+5","gold":"10"}]}'
curl -fsS -X POST "$BASE_URL/v1/jobs" \
  -H 'content-type: application/json' \
  -d "$QUEUED_JOB_PAYLOAD" >"$EVIDENCE/restart-queued-submit.json"
QUEUED_JOB_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$EVIDENCE/restart-queued-submit.json")"
curl -fsS "$BASE_URL/v1/jobs/$QUEUED_JOB_ID" >"$EVIDENCE/queued-job-before-restart.json"
QUEUED_BEFORE_STATE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$EVIDENCE/queued-job-before-restart.json")"
if [[ "$QUEUED_BEFORE_STATE" != "queued" ]]; then
  echo "second job was not queued before worker interruption: $QUEUED_BEFORE_STATE" >&2
  exit 1
fi

WORKER_POD="$(kubectl -n eval-lab get pod -l app.kubernetes.io/name=kube-eval-worker -o jsonpath='{.items[0].metadata.name}')"
kubectl -n eval-lab get pod "$WORKER_POD" -o json >"$EVIDENCE/worker-pod-before.json"
{
  echo "command=kubectl -n eval-lab delete pod $WORKER_POD --grace-period=0 --force --wait=false"
  echo "worker_pod=$WORKER_POD"
  echo "running_job_id=$RUNNING_JOB_ID"
  echo "running_state=$RUNNING_BEFORE_STATE"
  echo "queued_job_id=$QUEUED_JOB_ID"
  echo "queued_state=$QUEUED_BEFORE_STATE"
} >"$EVIDENCE/worker-restart-request.txt"
kubectl -n eval-lab delete pod "$WORKER_POD" --grace-period=0 --force --wait=false \
  >>"$EVIDENCE/worker-restart-request.txt" 2>&1
kubectl -n eval-lab wait pod "$WORKER_POD" --for=delete --timeout=120s
for _ in $(seq 1 60); do
  AFTER_WORKER="$(kubectl -n eval-lab get pod -l app.kubernetes.io/name=kube-eval-worker \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -n "$AFTER_WORKER" && "$AFTER_WORKER" != "$WORKER_POD" ]]; then
    break
  fi
  sleep 1
done
if [[ -z "${AFTER_WORKER:-}" || "$AFTER_WORKER" == "$WORKER_POD" ]]; then
  echo "replacement worker pod was not created" >&2
  exit 1
fi
kubectl -n eval-lab wait pod "$AFTER_WORKER" --for=condition=Ready --timeout=120s
kubectl -n eval-lab get pod "$AFTER_WORKER" -o json >"$EVIDENCE/worker-pod-after.json"

for _ in $(seq 1 90); do
  curl -fsS "$BASE_URL/v1/jobs/$RUNNING_JOB_ID" >"$EVIDENCE/job-after-worker-restart.json"
  RUNNING_AFTER_STATE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$EVIDENCE/job-after-worker-restart.json")"
  curl -fsS "$BASE_URL/v1/jobs/$QUEUED_JOB_ID" >"$EVIDENCE/queued-job-after-worker-restart.json"
  QUEUED_AFTER_STATE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$EVIDENCE/queued-job-after-worker-restart.json")"
  if [[ "$RUNNING_AFTER_STATE" == "completed" || "$RUNNING_AFTER_STATE" == "failed" ]] \
    && [[ "$QUEUED_AFTER_STATE" == "completed" || "$QUEUED_AFTER_STATE" == "failed" ]]; then
    break
  fi
  sleep 1
done
if [[ "${RUNNING_AFTER_STATE:-}" != "completed" || "${QUEUED_AFTER_STATE:-}" != "completed" ]]; then
  echo "interrupted jobs did not recover after worker replacement: running=${RUNNING_AFTER_STATE:-missing} queued=${QUEUED_AFTER_STATE:-missing}" >&2
  exit 1
fi
RUNNING_AFTER_ATTEMPTS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["attempts"])' "$EVIDENCE/job-after-worker-restart.json")"
QUEUED_AFTER_ATTEMPTS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["attempts"])' "$EVIDENCE/queued-job-after-worker-restart.json")"

POD="$(kubectl -n eval-lab get pod -l app.kubernetes.io/name=kube-eval-service -o jsonpath='{.items[0].metadata.name}')"
kubectl -n eval-lab get pod "$POD" -o json >"$EVIDENCE/pod-before.json"
BEFORE_RESTARTS="$(kubectl -n eval-lab get pod "$POD" -o jsonpath='{.status.containerStatuses[0].restartCount}')"
kubectl -n eval-lab delete pod "$POD" --wait=false >"$EVIDENCE/restart-request.txt"
kubectl -n eval-lab wait pod "$POD" --for=delete --timeout=120s
for _ in $(seq 1 60); do
  AFTER_POD="$(kubectl -n eval-lab get pod -l app.kubernetes.io/name=kube-eval-service \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -n "$AFTER_POD" && "$AFTER_POD" != "$POD" ]]; then
    break
  fi
  sleep 1
done
if [[ -z "${AFTER_POD:-}" || "$AFTER_POD" == "$POD" ]]; then
  echo "replacement pod was not created" >&2
  exit 1
fi
kubectl -n eval-lab wait pod "$AFTER_POD" --for=condition=Ready --timeout=120s
kubectl -n eval-lab get pod "$AFTER_POD" -o json >"$EVIDENCE/pod-after.json"
kubectl -n eval-lab get pods -o json >"$EVIDENCE/pods-after.json"
AFTER_RESTARTS="$(kubectl -n eval-lab get pod "$AFTER_POD" -o jsonpath='{.status.containerStatuses[0].restartCount}')"

kill "$PORT_FORWARD_PID" 2>/dev/null || true
wait "$PORT_FORWARD_PID" 2>/dev/null || true
kubectl -n eval-lab port-forward service/kube-eval-service "${PORT_FORWARD_PORT}:8080" \
  >>"$EVIDENCE/port-forward.log" 2>&1 &
PORT_FORWARD_PID=$!

for _ in $(seq 1 30); do
  if curl -fsS "$BASE_URL/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "$BASE_URL/readyz" >"$EVIDENCE/readyz-after-restart.json"

{
  echo "deleted_pod=$POD"
  echo "replacement_pod=$AFTER_POD"
  echo "old_pod_restart_count=$BEFORE_RESTARTS"
  echo "replacement_restart_count=$AFTER_RESTARTS"
  echo "job_id=$JOB_ID"
  echo "restart_job_id=$RUNNING_JOB_ID"
  echo "restart_job_state=$RUNNING_AFTER_STATE"
  echo "queued_restart_job_id=$QUEUED_JOB_ID"
  echo "queued_restart_job_state=$QUEUED_AFTER_STATE"
  echo "running_before_state=$RUNNING_BEFORE_STATE"
  echo "queued_before_state=$QUEUED_BEFORE_STATE"
  echo "running_attempts_after_restart=$RUNNING_AFTER_ATTEMPTS"
  echo "queued_attempts_after_restart=$QUEUED_AFTER_ATTEMPTS"
  echo "worker_pod_before=$WORKER_POD"
  echo "worker_pod_after=$AFTER_WORKER"
  echo "worker_forced_delete=true"
} >"$EVIDENCE/restart-result.txt"

kind get clusters >"$EVIDENCE/kind-clusters.txt"
kind version >"$EVIDENCE/kind-version.txt"
kubectl version -o json >"$EVIDENCE/kubectl-version.json"
docker image inspect "$IMAGE" >"$EVIDENCE/image-inspect.json"
DOCKER_SERVER_VERSION="$(docker version --format '{{.Server.Version}}')"
KIND_VERSION="$(kind version)"
KUBECTL_CLIENT_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["clientVersion"]["gitVersion"])' "$EVIDENCE/kubectl-version.json")"
KUBECTL_SERVER_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("serverVersion",{}).get("gitVersion","unknown"))' "$EVIDENCE/kubectl-version.json")"
HOST_PLATFORM="$(uname -srm)"
python3 - "$EVIDENCE" "$CLUSTER" "$BASE_IMAGE" "$INSTALL_DEPENDENCIES" "$IMAGE" "$IMAGE_ID" \
  "$DOCKER_SERVER_VERSION" "$KIND_VERSION" "$KUBECTL_CLIENT_VERSION" "$KUBECTL_SERVER_VERSION" "$HOST_PLATFORM" <<'PY'
import json
import pathlib
import sys

evidence = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "kube-eval.runtime.v1",
    "local_smoke": True,
    "cluster": sys.argv[2],
    "namespace": "eval-lab",
    "base_image": sys.argv[3],
    "install_dependencies": sys.argv[4] == "1",
    "image": sys.argv[5],
    "image_id": sys.argv[6],
    "docker_server_version": sys.argv[7],
    "kind_version": sys.argv[8],
    "kubectl_client_version": sys.argv[9],
    "kubectl_server_version": sys.argv[10],
    "host_platform": sys.argv[11],
    "python_version": sys.version.split()[0],
}
(evidence / "runtime.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

python3 "$ROOT/scripts/write_evidence.py" --evidence "$EVIDENCE"
python3 "$ROOT/verify_evidence.py" --evidence "$EVIDENCE"

echo "Kubernetes smoke passed. Evidence: $EVIDENCE"
