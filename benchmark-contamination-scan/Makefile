PYTHON ?= ../.venv/bin/python

.PHONY: test fixture scan report gate gate-check checksums verify reproduce evalops-gated

test:
	PYTHONPATH=.. $(PYTHON) -m pytest -q tests

fixture:
	$(PYTHON) -m contamination_scan build-fixture

scan:
	$(PYTHON) -m contamination_scan scan

report:
	$(PYTHON) -m contamination_scan report

gate:
	$(PYTHON) -m contamination_scan gate

gate-check:
	$(PYTHON) scripts/assert_gate_fails_closed.py --expected-report evidence/report.json

checksums:
	$(PYTHON) scripts/write_checksums.py

verify:
	$(PYTHON) verify_evidence.py --allow-pending-human

reproduce:
	$(PYTHON) -m pytest -q tests
	@$(PYTHON) -m contamination_scan report; code=$$?; test $$code -eq 0 || test $$code -eq 2
	@$(PYTHON) -m contamination_scan gate; code=$$?; test $$code -eq 0 || test $$code -eq 2 || test $$code -eq 3
	$(PYTHON) scripts/write_checksums.py
	$(PYTHON) verify_evidence.py --allow-pending-human
	$(PYTHON) scripts/assert_gate_fails_closed.py --expected-report evidence/report.json

evalops-gated:
	$(PYTHON) -m contamination_scan gate
	$(PYTHON) scripts/evalops_gated.py
