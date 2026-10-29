PYTHON ?= ../.venv/bin/python

.PHONY: install test smoke export-opencompass ollama-probe summary checksums verify reproduce

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q tests

smoke:
	$(PYTHON) scripts/run_contract_smoke.py

export-opencompass:
	$(PYTHON) scripts/export_opencompass_configs.py

ollama-probe:
	$(PYTHON) scripts/run_ollama_probe.py

summary:
	$(PYTHON) scripts/build_summary.py

checksums:
	$(PYTHON) scripts/write_checksums.py

verify:
	$(PYTHON) verify_evidence.py

reproduce:
	$(PYTHON) -m pytest -q tests --timeout=30
	$(PYTHON) scripts/run_contract_smoke.py
	$(PYTHON) scripts/export_opencompass_configs.py
	$(PYTHON) scripts/build_summary.py
	$(PYTHON) scripts/write_checksums.py
	$(PYTHON) verify_evidence.py
