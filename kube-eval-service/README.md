# Kube Eval Service

A small, security-conscious evaluation service for local Kubernetes acceptance.
It supports deterministic plumbing tests and a fixed OpenAI-compatible upstream;
requests cannot choose arbitrary upstream URLs.

> **本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.**

## Service contract

- `GET /healthz`: liveness;
- `GET /readyz`: deterministic readiness or a bounded upstream `/models` check;
- `GET /metrics`: Prometheus text metrics;
- `POST /v1/evaluate`: bounded task list and concurrency, per-task errors, JSON result.
- `POST /v1/jobs`: persist an evaluation request and enqueue it for workers;
- `GET /v1/jobs/{id}` and `POST /v1/jobs/{id}/cancel`: durable status and
  cooperative cancellation;
- optional bearer or `X-Eval-Token` authentication for the job endpoints.

Request-level limits:

- 1–500 tasks by default;
- concurrency 1–32;
- upstream URL only from `EVAL_UPSTREAM_BASE_URL`;
- API key only from environment, never echoed in responses.

## Local run

```bash
cd llm-lab
.venv/bin/pip install -e "kube-eval-service[dev]"
EVAL_MODE=deterministic .venv/bin/uvicorn kube_eval_service.app:app \
  --host 127.0.0.1 --port 8080
```

```bash
curl -fsS http://127.0.0.1:8080/healthz
curl -fsS -X POST http://127.0.0.1:8080/v1/evaluate \
  -H 'content-type: application/json' \
  -d '{"tasks":[{"id":"a","prompt":"2+2","gold":"4"}]}'
```

## kind smoke

The script downloads `kind` and `kubectl` into `llm-lab/.tools/bin`, builds a
local image, creates a cluster, deploys the API and worker, checks health, runs
24 concurrent requests, verifies idempotent job submission, force-deletes the
worker, and verifies that both an interrupted running job and a queued job
recover from the shared SQLite/PVC state. It also replaces the API pod and
checks readiness. The run writes checksummed evidence and exits non-zero if
`verify_evidence.py` fails.

```bash
cd kube-eval-service
CLUSTER="eval-lab-smoke-$(date +%s)"
KIND_CLUSTER_NAME="$CLUSTER" ./scripts/kind-smoke.sh
KIND_CLUSTER_NAME="$CLUSTER" make kind-clean
```

If Docker Hub is unavailable, point kind at an already-pulled node image:

```bash
KIND_NODE_IMAGE=kind-registry.example/kindest/node:v1.37.0 \
  ./scripts/kind-smoke.sh
```

This host may already have Docker but does not have `kind`, `kubectl`, or a GPU.
The smoke is a local CPU Kubernetes test, not a production cluster or GPU
serving benchmark.

After inspecting the evidence, remove the local cluster:

```bash
make kind-clean
```

## Evidence verification

The smoke writes 36 checksum-covered files plus
`evidence/verification-result.json`. Verify them independently:

```bash
(cd evidence && sha256sum -c SHA256SUMS)
../.venv/bin/python verify_evidence.py
```

For a self-contained rerun that does not overwrite checked-in evidence and
deletes only the cluster it created:

```bash
make reproduce
```

The verifier checks the live Deployment/Service/ConfigMap/NetworkPolicy,
non-root/read-only/dropped-capability settings, readiness after pod replacement,
the API and worker Deployments, the bound PVC and PDB, durable job idempotency
and post-worker-restart recovery, the 24-request concurrency smoke, Prometheus
counters, and the checksum manifest.

## Kubernetes boundary

The included manifests demonstrate:

- namespace-scoped API and worker Deployments plus a Service;
- a SQLite-backed durable queue with worker leases and bounded retries;
- readiness/liveness probes;
- non-root user, read-only root filesystem, dropped capabilities and seccomp;
- CPU/memory requests and limits;
- NetworkPolicy, PVC and PodDisruptionBudget;
- pod replacement/readiness after deletion.

They do not demonstrate multi-node scheduling, autoscaling, GPU allocation,
multi-tenancy, upgrades, backups or production incident response. The API and
worker share a `ReadWriteOnce` PVC on one kind node; this does not establish a
safe multi-node storage design. The kind CNI may not enforce NetworkPolicy, so
the evidence verifies the policy documents structurally rather than proving
runtime isolation.

## Tests

```bash
.venv/bin/python -m pytest -q kube-eval-service/tests
```
