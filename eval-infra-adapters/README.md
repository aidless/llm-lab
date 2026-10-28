# Eval Infra Adapters

GPU-free contract layer for using vLLM and LMDeploy behind the same
OpenAI-compatible evaluation runner.

> **本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.**

## What works without a GPU

- deterministic `vllm serve` and `lmdeploy serve api_server` command generation;
- `/health`, `/v1/models` and `/v1/chat/completions` contract tests;
- model-not-found, authentication, server error and OOM classification;
- managed subprocess startup, bounded readiness wait, log capture and stop;
- host diagnostics for Docker, Kubernetes tools, Python modules and NVIDIA GPUs.

These checks do not prove that a serving backend is installed, has enough VRAM,
or meets a production latency target.

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
then stops the process. The reproduce command also probes real Ollama, combines
the existing 201-task evidence, writes `SHA256SUMS`, and verifies exact file
coverage and claim boundaries.
