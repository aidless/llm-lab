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

docker build -t "$IMAGE" "$ROOT"
kind load docker-image --name "$CLUSTER" "$IMAGE"

kubectl apply -f "$ROOT/deploy/namespace.yaml"
kubectl apply -f "$ROOT/deploy/configmap.yaml"
kubectl apply -f "$ROOT/deploy/deployment.yaml"
kubectl apply -f "$ROOT/deploy/service.yaml"
kubectl apply -f "$ROOT/deploy/networkpolicy.yaml"
kubectl -n eval-lab rollout status deployment/kube-eval-service --timeout=120s

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

printf 'cluster=%s\nnamespace=eval-lab\nport_forward=%s\n' "$CLUSTER" "$PORT_FORWARD_PORT" \
  >"$EVIDENCE/cluster-context.txt"
cat >"$EVIDENCE/boundary.md" <<'MD'
# Evidence boundary

> **本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.**

This is a single-node kind smoke on a local CPU host. It verifies service
plumbing, bounded input, health checks, pod replacement and checksummed
evidence. It is not a production Kubernetes deployment, autoscaling study,
multi-tenant security proof, GPU serving benchmark or SLO claim.
MD

kubectl -n eval-lab get deployment,service,pods,networkpolicy -o wide >"$EVIDENCE/resources.txt"
kubectl -n eval-lab get deployment kube-eval-service -o json >"$EVIDENCE/deployment-live.json"
kubectl -n eval-lab get service kube-eval-service -o json >"$EVIDENCE/service-live.json"
kubectl -n eval-lab get configmap kube-eval-config -o json >"$EVIDENCE/configmap-live.json"
kubectl -n eval-lab get networkpolicy kube-eval-service -o json >"$EVIDENCE/networkpolicy-live.json"
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
} >"$EVIDENCE/restart-result.txt"

kind get clusters >"$EVIDENCE/kind-clusters.txt"
kind version >"$EVIDENCE/kind-version.txt"
kubectl version -o json >"$EVIDENCE/kubectl-version.json"
docker image inspect "$IMAGE" >"$EVIDENCE/image-inspect.json"

python3 "$ROOT/scripts/write_evidence.py" --evidence "$EVIDENCE"
python3 "$ROOT/verify_evidence.py" --evidence "$EVIDENCE"

echo "Kubernetes smoke passed. Evidence: $EVIDENCE"
