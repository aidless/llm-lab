# Eval Infra Adapter Evidence

> **本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.**

- Contract smoke: **PASS**
- Real Ollama OpenAI-compatible probe: **PASS**
- GPU acceptance: **BLOCKED_ENVIRONMENT**
- vLLM / LMDeploy executed: **False**

## Existing 201-task Ollama run

Source: `evalops/evidence-20261027/summary.json`
Task-set SHA-256: `33fdba0a807b721548cdac4a6339443c6edcd4bff00e895f62b77a3361bb9657`
Model: `qwen25-3b-16k:latest`

| C | Success | Correct | p50 ms | p95 ms | Tasks/s | Wall s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 201/201 | 108 | 1473.3 | 3638.3 | 0.579 | 347.2 |
| 4 | 201/201 | 107 | 1574.0 | 2815.8 | 2.414 | 83.3 |
| 8 | 201/201 | 108 | 1878.9 | 3458.4 | 3.986 | 50.4 |

## Concurrency consistency

| Pair | Agreement | Rate |
|---|---:|---:|
| `ollama-201-c1 vs ollama-201-c4` | 196/201 | 97.51% |
| `ollama-201-c1 vs ollama-201-c8` | 195/201 | 97.01% |
| `ollama-201-c4 vs ollama-201-c8` | 200/201 | 99.50% |

These are local CPU Ollama results, not GPU serving throughput, not a production SLO,
and not an official C-Eval leaderboard result.

## GPU blockers

- NVIDIA device available: **False**
- vLLM module installed: **False**
- LMDeploy module installed: **False**
- Real C=1/4/8 GPU OOM and throughput experiment: **not executed**

The adapter command generation, process lifecycle, HTTP contract, error taxonomy and
201-task runner integration are tested. A real GPU acceptance run remains blocked by
the host environment and requires separately authorized GPU resources.
