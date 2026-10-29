# EvalOps GPU Simulation (vLLM/LMDeploy Adapter Contract)

> **Local evaluation, not a production system.**  
> **本地评测，不是生产系统。**

This package demonstrates the full end-to-end contract for using vLLM or LMDeploy
with llm-lab EvalOps on a CPU host. It does not execute real GPU serving workloads.
All metrics are placeholders and must be replaced by actual runs when GPU is available.

## Goal

Provide a verified adapter layer that:

- accepts `--adapter vllm|lmdeploy` on `llm-lab evalops run`,
- generates start commands for `vllm serve` / `lmdeploy serve api_server`,
- probes `/health` and `/v1/models`,
- executes against OpenAI-compatible `/v1/chat/completions`,
- classifies errors (model-not-found, auth, server error, OOM),
- records environment facts (GPU count, VRAM),
- produces JSON/Markdown/HTML reports with explicit non-claims.

The current result is **HOLD**: no real GPU experiment has been executed yet.

## Prerequisites

```bash
cd ../  # repo root
python3 -m venv .venv
.venv/bin/pip install -e "llm_lab[dev]" --quiet
```

Install backend modules (optional; we keep them disabled on CPU):

```bash
pip install vllm --quiet || echo "vllm skipped (no GPU)"
pip install lmdeploy --quiet || echo "lmdeploy skipped (no GPU)"
```

## Run simulation locally

Generate commands without running GPUs:

```bash
cd evalops-gpu-sim
../.venv/bin/python -m evalops_gpu_sim diagnose
../.venv/bin/python -m evalops_gpu_sim command vllm Qwen/Qwen2.5-0.5B-Instruct --port 8000 --tp 1
../.venv/bin/python -m evalops_gpu_sim command lmdeploy Qwen/Qwen2.5-0.5B-Instruct --port 8001 --tp 1
```

Execute a lightweight OpenAI-compatible smoke test against **Ollama only**:

```bash
../.venv/bin/python -m evalops_gpu_sim probe ollama \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen25-3b-16k:latest
```

Run an **EvalOps batch** with the same runner but CPU backend:

```bash
../.venv/bin/llm-lab evalops run evalops/ceval-201/tasks-v1.jsonl \
  --adapter ollama \
  --model qwen25-3b-16k:latest \
  --concurrency 4 \
  --output evidence/runs/ollama-c4-20261028
```

Create a **simulation report** summarizing expected GPU behavior (placeholder):

```bash
../.venv/bin/python -m evalops_gpu_sim simulate-ceval201 \
  --backend vllm \
  --output evidence/report-vllm-simulation.json
../.venv/bin/python -m evalops_gpu_sim summarize \
  --runs evidence/runs/ollama-c4-20261028 \
  --sim evidence/report-vllm-simulation.json \
  --output evidence/integration-summary.md
```

Write hashes:

```bash
../.venv/bin/python -m evalops_gpu_sim checksums
```

Verify:

```bash
../.venv/bin/python verify_evidence.py
```

Evidence artifacts live under `evidence/`. See the gate in `verify_evidence.py`.

## Gate (HOLD until GPU run)

Current decision: `HOLD`. The acceptance gates are:

- `real_gpu_run_executed`: false (requires NVIDIA device, vLLM/LMDeploy launch)
- `task_set_verified`: true (C-Eval 201 manifest hash)
- `adapter_contract_smoke`: true (Ollama HTTP probe passed)
- `environment_diagnosed`: true (host facts recorded)

It will become `PASS` only after a real GPU run fills:

```json
{
  "run_id": "...",
  "backend": "vllm|lmdeploy",
  "model_revision": "...",
  "gpu_count": 1,
  "vramp_gb": 16,
  "tp_size": 1,
  "success_rate": 1.0,
  "throughput_tasks_per_sec": 9.5,
  "p50_ms": 2100,
  "p95_ms": 4200,
  "result_agreement_with_ollama": 0.99
}
```

and pass consistency checks (agreement ≥0.98).

## What to change when GPU arrives

Replace the simulation placeholder with a real execution:

- Use a remote host or authorized notebook with GPU quota.
- Start `vllm serve` with desired model & TP size.
- Run the same C-Eval subset with concurrency 1/4/8.
- Record startup logs, OOM or success; export raw output.
- Recompute `integration-summary.md` from the new run and refresh hashes.

See `scripts/rerun-real.sh` (provided as template) once GPU access is granted.

## Explicit non-claims

- This is not a production guardrail, sandbox escape test, or compliance system.
- No real filesystem, shell, email or network action occurs beyond local Ollama.
- The simulated throughput numbers are estimates and must be replaced by measured values.
- CPU latency is not a serving SLO.
- Synthetic results do not prove benchmark leadership or frontier safety.

License: MIT. See the repository-level `LICENSE`.
