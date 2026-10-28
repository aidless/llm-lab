PYTHON ?= ../.venv/bin/python
JUDGES ?= qwen3:0.6b llama3.2:1b

.PHONY: test run report review ingest checksums verify reproduce

test:
	PYTHONPATH=.. $(PYTHON) -m pytest -q tests

run:
	$(PYTHON) -m judge_calibration run --judges $(JUDGES) --force

report:
	$(PYTHON) -m judge_calibration report \
		$(if $(wildcard review/judge-labels-human.jsonl),--human-labels review/judge-labels-human.jsonl,)

review:
	$(PYTHON) -m judge_calibration build-review

ingest:
	$(PYTHON) -m judge_calibration ingest-review --export review/judge-reviews-export.json

verify:
	$(PYTHON) verify_evidence.py --allow-pending-human

checksums:
	$(PYTHON) scripts/write_checksums.py

reproduce:
	./scripts/reproduce.sh
