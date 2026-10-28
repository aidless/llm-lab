# EvalOps report

- Model: `qwen25-3b-16k:latest`
- Provider: `ollama`
- Tasks: 12
- Concurrency: 4
- Task-set SHA-256: `33fdba0a807b721548cdac4a6339443c6edcd4bff00e895f62b77a3361bb9657`
- Config SHA-256: `bcd0ffff2aefef706854f7451c4d1b2fe144938472a406286f3c312ba2e1c277`
- Answer SHA-256: `04f1e6233260f165e8c5f0bfc38bf7c651b01680d677894566404964455d6682`

## Outcome

| Metric | Value |
|---|---:|
| Successful provider responses | 12 |
| Parsed answers | 12 |
| Correct answers | 10 |
| Success rate | 100.00% |
| Parsed rate | 100.00% |
| Accuracy among parsed answers | 83.33% |
| Strict accuracy (all tasks) | 83.33% |
| Retry attempts | 0 |
| Wall time | 21.10s |
| Throughput | 0.569 tasks/s |

## Latency

| Set | p50 | p95 | Mean |
|---|---:|---:|---:|
| All final records | 6004.29 ms | 10644.01 ms | 6396.24 ms |
| Successful records | 6004.29 ms | 10644.01 ms | 6396.24 ms |

## Failures

No failures recorded.

## By subject

| Subject | Total | Success | Correct | Parsed accuracy | Strict accuracy |
|---|---:|---:|---:|---:|---:|
| college_programming | 12 | 12 | 10 | 83.33% | 83.33% |

## Limitations

- This is a local, non-production run.
- The public C-Eval subset is not the official C-Eval leaderboard.
- Parsed accuracy excludes parse/provider failures; strict accuracy includes them in the denominator.
- Deterministic adapters are fixtures, not model-quality evidence.
