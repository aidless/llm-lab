# Reproducing Agent Safety evidence

## Install

```bash
cd llm-lab
python3 -m venv .venv
.venv/bin/pip install -e llm_lab
.venv/bin/pip install -e "agent-safety-eval[dev]"
```

## Full local run

```bash
cd agent-safety-eval
./scripts/reproduce.sh
```

The default run uses `qwen3:0.6b` and `llama3.2:1b`, policy off and on, for
96 tasks × 2 models × 2 policies = 384 trajectories.

The run remains `HOLD` until a human reviews:

- at least 100 records in total;
- all 96 task labels;
- every trajectory with an unsafe attempt or unsafe execution;
- no required trajectory is left flagged.

The rerun script archives prior evidence and review packets under
`.codex-evidence/` before creating a new package, so a rerun does not delete
pending human decisions.

## Stratified smoke

```bash
.venv/bin/agent-safety-eval run \
  --stratified-smoke 12 \
  --models qwen3:0.6b \
  --force
```

Smoke output is a service/plumbing test, not a safety conclusion.

## Verification

```bash
.venv/bin/python scripts/write_checksums.py
.venv/bin/python verify_evidence.py --allow-pending-human
```

`--allow-pending-human` permits technical verification while preserving
`acceptance_ok=false` and `human_gate_pending=true`.
