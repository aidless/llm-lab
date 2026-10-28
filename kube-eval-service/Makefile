PYTHON ?= ../.venv/bin/python
KIND_CLUSTER_NAME ?=

.PHONY: install test run image kind-smoke kind-clean evidence verify reproduce

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q tests

run:
	EVAL_MODE=deterministic $(PYTHON) -m uvicorn kube_eval_service.app:app \
		--host 127.0.0.1 --port 8080

image:
	docker build -t kube-eval-service:dev .

kind-smoke:
	./scripts/kind-smoke.sh

kind-clean:
	@test -n "$(KIND_CLUSTER_NAME)" || (echo "Set KIND_CLUSTER_NAME to the exact smoke cluster" >&2; exit 1)
	PATH="../.tools/bin:$$PATH" kind delete cluster --name "$(KIND_CLUSTER_NAME)"

evidence:
	$(PYTHON) scripts/write_evidence.py

verify:
	$(PYTHON) verify_evidence.py

reproduce:
	$(PYTHON) -m pytest -q tests --timeout=30
	./scripts/reproduce.sh
