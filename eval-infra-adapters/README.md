# Eval Infra Adapters

GPU-free contract layer for using vLLM and LMDeploy behind the same
OpenAI-compatible evaluation runner, plus a pinned OpenCompass config export.

> **本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.**

## What works without a GPU

- deterministic `vllm serve` and `lmdeploy serve api_server` command generation;
- `/health`, `/v1/models` and `/v1/chat/completions` contract tests;
- model-not-found, authentication, server error and OOM classification;
- managed subprocess startup, bounded readiness wait, log capture and stop;
- executable OpenCompass 0.5.4 `OpenAISDK` + `CustomDataset` config export for
  the frozen 201-row C-Eval validation subset;
- static contract checks against the pinned official OpenCompass source,
  imports, config fields and quick-dataset expansion path;
- host diagnostics for Docker, Kubernetes tools, Python modules and NVIDIA GPUs.

These checks do not prove that a serving backend is installed, has enough VRAM,
that OpenCompass was executed, or that any result is an official leaderboard
score.

## Install

From the `llm-lab` monorepo:

```bash
cd llm-lab
python3 -m venv .venv
.venv/bin/pip install -e llm_lab
.venv/bin/pip install -e "eval-infra-adapters[dev]"
```

## Use

```bash
.venv/bin/eval-infra diagnose

.venv/bin/eval-infra serve-command \
  --backend vllm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --port 8000 \
  --tp 1

.venv/bin/eval-infra probe \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct

.venv/bin/eval-infra opencompass-export \
  --output-dir eval-infra-adapters/opencompass-export \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

For the evaluation half, use the same endpoint with `llm-lab evalops`:

```bash
.venv/bin/llm-lab evalops run evalops/ceval-201/tasks-v1.jsonl \
  --adapter vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --concurrency 1 \
  --output /tmp/vllm-c1
```

Replace `--adapter vllm` with `--adapter lmdeploy` for LMDeploy. The endpoint
must already be running.

## OpenCompass config export

The exporter pins the interface to official OpenCompass `0.5.4`. It writes:

```text
opencompass-export/
├── eval-infra-openai-api.py       # combined model + dataset config
├── model_openai_api.py            # standalone OpenAISDK config
├── dataset_ceval_201.py           # standalone CustomDataset quick config
├── data/ceval-201qa.jsonl         # frozen 201-row public validation subset
├── data/ceval-201qa.jsonl.meta.json
├── env.example                    # endpoint/model/key-variable names only
└── official-api-contract.json     # pinned official source/interface snapshot
```

Run it only after starting the endpoint and installing OpenCompass separately:

```bash
export EVAL_INFRA_OPENAI_API_KEY=EMPTY
opencompass eval-infra-adapters/opencompass-export/eval-infra-openai-api.py \
  --work-dir eval-infra-adapters/opencompass-export/runs/ceval-201 \
  --debug
```

The default endpoint is `http://127.0.0.1:8000/v1`. Non-loopback endpoints
require an explicit exporter flag and a separate authorization decision. The
API key is read from `EVAL_INFRA_OPENAI_API_KEY`; it is never written to the
config or evidence. The exporter does not call the endpoint or execute
OpenCompass.

The official contract is checked two ways:

- an always-offline static test against the pinned OpenCompass 0.5.4 source
  hashes, imports and exported field contract;
- an additional dynamic CPU config/data parse test only when a real OpenCompass
  installation is already present. It performs no inference.

## GPU acceptance boundary

A real backend experiment must record all of the following before it can be
called a GPU run:

- model revision and tokenizer revision;
- backend package version and launch command;
- GPU model, one-GPU VRAM, and tensor-parallel size;
- startup log, OOM or initialization failure;
- C=1/4/8 wall time, p50/p95, retries, failures and result consistency;
- immutable evidence checksums.

The current host has no usable NVIDIA GPU. This package therefore claims
offline contract, process-management and real-Ollama validation only.

The checked-in evidence combines:

- a real managed child-process HTTP smoke for both backend labels;
- a real OpenAI-compatible probe against local Ollama;
- a ready-to-run OpenCompass model/dataset config export with a pinned official
  interface contract and an explicit `BLOCKED_ENVIRONMENT` GPU preflight;
- the verified 201-task Ollama `C=1/4/8` CPU runs from
  `evalops/evidence-20261027/`;
- an explicit `BLOCKED_ENVIRONMENT` GPU section rather than a fabricated
  vLLM/LMDeploy throughput result.

## Tests

```bash
.venv/bin/python -m pytest -q eval-infra-adapters/tests
```

## Localhost process + HTTP smoke

```bash
cd eval-infra-adapters
make reproduce
```

The smoke starts a real child process and performs real localhost HTTP requests
against a deterministic OpenAI-compatible fixture for both backend labels. It
then stops the process. The report labels the fixture implementation, marks the
real backend as not executed, and records that the generated vLLM/LMDeploy
commands were not launched.

The default reproduction command is offline:

```bash
make reproduce
```

It runs tests, the fixture smoke, OpenCompass export, summary generation,
`SHA256SUMS`, and the independent verifier. It does not call Ollama or any model
endpoint. The checked-in real Ollama probe and existing 201-task `C=1/4/8`
evidence are preserved and re-bound by hash. To refresh the live Ollama probe
explicitly, use `make ollama-probe`.
