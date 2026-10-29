PYTHON ?= ../.venv/bin/python
JUDGES ?= qwen2.5:3b-instruct-q4_K_M llama3.2:3b
BASE_URL ?= http://127.0.0.1:11434
VARIANTS ?= baseline,authority,verbose,order_prefix,order_suffix
REPEATS ?= 2
WORKERS ?= 6

.PHONY: test run report review ingest checksums verify reproduce

test:
	PYTHONPATH=.. $(PYTHON) -m pytest -q tests

run:
	$(PYTHON) -m judge_calibration run --judges $(JUDGES) \
		--base-url $(BASE_URL) --variants $(VARIANTS) --repeats $(REPEATS) \
		--workers $(WORKERS) --force

report:
	$(PYTHON) -m judge_calibration report \
		$(if $(wildcard review/judge-labels-human.jsonl),--human-labels review/judge-labels-human.jsonl,)

review:
	$(PYTHON) -m judge_calibration build-review

ingest:
	$(PYTHON) -m judge_calibration ingest-review --export review/judge-reviews-export.json

verify:
	$(PYTHON) verify_evidence.py --judges $(JUDGES) --allow-pending-human

checksums:
	$(PYTHON) scripts/write_checksums.py

reproduce:
	./scripts/reproduce.sh
