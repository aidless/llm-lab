# EvalOps evidence: frozen C-Eval subset and local Ollama runs

> **本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.**

This package turns `llm-lab`'s batch-evaluation path into an auditable local
result.  It combines a frozen 201-task C-Eval validation subset, a
content-hashed task manifest, raw per-task records, reproducible JSON/Markdown/
HTML reports, concurrency runs, failure injection, and a standalone verifier.

## Verified local result

Model: `qwen25-3b-16k:latest`
Backend: local Ollama `0.33.3` over `http://127.0.0.1:11434/v1`
Dataset: 201 public C-Eval validation questions across 10 subjects
Task SHA-256: `33fdba0a807b721548cdac4a6339443c6edcd4bff00e895f62b77a3361bb9657`

| Run | Success / parsed | Correct | Strict accuracy | Wall time | Throughput | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| C=1 | 201 / 201 | 108 | 53.73% | 347.25 s | 0.58 tasks/s | 1473 / 3638 ms |
| C=4 | 201 / 201 | 107 | 53.23% | 83.28 s | 2.41 tasks/s | 1574 / 2816 ms |
| C=8 | 201 / 201 | 108 | 53.73% | 50.42 s | 3.99 tasks/s | 1879 / 3458 ms |

The three runs are a concurrency and operations demonstration.  They are not
three independent benchmark samples.  Pairwise parsed-answer agreement was:

- C=1 vs C=4: 196 / 201 = 97.51%
- C=4 vs C=8: 200 / 201 = 99.50%
- C=1 vs C=8: 195 / 201 = 97.01%

This is repeat consistency, not bitwise determinism or a statistical
significance claim.  Temperature was zero, but Ollama execution still produced
some different answers.

## Operational checks

- Strict single-letter answer parsing separates provider failures, format
  failures, and evaluator failures.
- Config, task-set, and task-count hashes are checked before a resume.
- Retrying a failed task atomically replaces its old record instead of
  appending a duplicate.
- The deterministic fixture produced the same answer hash
  `77ace68bf48a4c309cf9ba5897c1d0238420cd0133cec9db8bb8be6a000781ed` at
  concurrency 1/4/8.  Its 46/201 result is intentionally meaningless as model
  quality; it proves plumbing only.
- A synthetic six-task run classified `timeout`, `transport`, `model_error`,
  and `parse_error` exactly once each.

## Contents

```text
dataset/                 # self-contained frozen task set, manifest, notice, export
ollama-201-c1/           # metadata, raw records, JSON/MD/HTML reports
ollama-201-c4/
ollama-201-c8/
ollama-smoke12-c4/
deterministic-c1/
deterministic-c4/
deterministic-c8/
fault-injection/         # task fixture plus synthetic failure records
environment.json         # host/model/runtime facts observed at capture time
summary.json             # machine-readable result and limitations
SHA256SUMS               # 52-file integrity manifest
verify_evidence.py       # recomputes hashes, counts, rates, and agreements
```

Run:

```bash
python3 verify_evidence.py
```

## Explicit non-claims

- This is not an official C-Eval leaderboard result.
- These local CPU timings are not a production SLO or GPU-serving benchmark.
- `vllm`, `opencompass`, paid APIs, GPU rental, human labels, customer traffic,
  and production concurrency were not used.
- `dataset/opencompass/` is a compatibility export only.  The external runner
  was not installed or executed.
- The fixture runs do not demonstrate model quality, user adoption, or
  benchmark leadership.

The subset retains the upstream dataset's `CC-BY-NC-SA-4.0` license.  See
`dataset/NOTICE.md`.  The bundled Ollama model reports the Qwen Research
License; model weights are not redistributed here.

## Portfolio use

The public portfolio publishes only the aggregate `summary.json` and the
concurrency-8 HTML report.  The full raw-record bundle remains local for
reproduction and review; the frozen C-Eval task set is not republished on the
static portfolio because of its non-commercial share-alike license.
