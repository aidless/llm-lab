PYTHON ?= ../.venv/bin/python
MODELS ?= qwen3:0.6b llama3.2:1b

.PHONY: install test dataset run stratified report review verify rerun reproduce

install:
	$(PYTHON) -m pip install -e ..
	$(PYTHON) -m pip install -e ".[dev]"

test:
	PYTHONPATH=.. $(PYTHON) -m pytest -q tests

dataset:
	$(PYTHON) -m agent_safety_eval build --output dataset --force
	$(PYTHON) -m agent_safety_eval verify-dataset --dataset dataset

run:
	$(PYTHON) -m agent_safety_eval run --models $(MODELS) --force

stratified:
	$(PYTHON) -m agent_safety_eval run \
		--stratified-smoke 12 --models $(MODELS) --force

report:
	$(PYTHON) -m agent_safety_eval report \
		$(if $(wildcard review/reviews-human.jsonl),--reviews review/reviews-human.jsonl,)

review:
	$(PYTHON) -m agent_safety_eval prepare-review

verify:
	$(PYTHON) verify_evidence.py --allow-pending-human

rerun:
	./scripts/rerun.sh

reproduce:
	$(PYTHON) -m agent_safety_eval report
	$(PYTHON) scripts/write_checksums.py
	$(PYTHON) verify_evidence.py --allow-pending-human
