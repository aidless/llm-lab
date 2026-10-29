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

The default run uses `qwen2.5:3b-instruct-q4_K_M` and `llama3.2:3b`, with four
arms (`off`, `prompt`, `gate`, `on`), for 96 tasks × 2 models × 4 arms = 768
trajectories. Three task workers run concurrently by default with a 180-second
per-call timeout; `WORKERS=1` forces a serial replay and `OLLAMA_TIMEOUT`
overrides the timeout.

The run remains `HOLD` until a human reviews:

- at least 100 records in total;
- all 96 task labels;
- every trajectory with an unsafe attempt or unsafe execution;
- no required trajectory is left flagged.

The rerun script archives prior evidence and review packets under
`.codex-evidence/` before creating a new package, so a rerun does not delete
pending human decisions.

## Running against a remote Ollama host

The 768-trajectory sweep is CPU-bound and takes over an hour on a contended
16-core laptop. Both are supported:

```bash
# point the client at any Ollama instance; environment.json records the
# endpoint actually used plus each model's digest and quantization
OLLAMA_BASE_URL=http://127.0.0.1:11435 WORKERS=3 OLLAMA_TIMEOUT=180 \
  MODELS="qwen2.5:3b-instruct-q4_K_M llama3.2:3b" ./scripts/reproduce.sh
```

The checked-in evidence was produced on a 128-core CPU host with no GPU
(`nvidia-smi` absent, recorded as such in `environment.json`). The model
weights were rebuilt from byte-identical GGUF blobs, so the sizes match a
workstation install exactly; the Ollama manifest digest differs because the
template and licence layers are not byte-identical. `environment.json` records
the digests that were actually served. Nothing in this package needs a GPU;
nothing in it claims GPU behaviour.

## Stratified smoke

```bash
.venv/bin/agent-safety-eval run \
  --stratified-smoke 12 \
  --models qwen2.5:3b-instruct-q4_K_M \
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
