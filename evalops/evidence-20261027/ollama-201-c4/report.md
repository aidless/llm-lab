# EvalOps report

- Model: `qwen25-3b-16k:latest`
- Provider: `ollama`
- Tasks: 201
- Concurrency: 4
- Task-set SHA-256: `33fdba0a807b721548cdac4a6339443c6edcd4bff00e895f62b77a3361bb9657`
- Config SHA-256: `bcd0ffff2aefef706854f7451c4d1b2fe144938472a406286f3c312ba2e1c277`
- Answer SHA-256: `e2e7ec77ce8fe2229710556471e0874f115b2eb52e98de202e7f3ba36add12ef`

## Outcome

| Metric | Value |
|---|---:|
| Successful provider responses | 201 |
| Parsed answers | 201 |
| Correct answers | 107 |
| Success rate | 100.00% |
| Parsed rate | 100.00% |
| Accuracy among parsed answers | 53.23% |
| Strict accuracy (all tasks) | 53.23% |
| Retry attempts | 0 |
| Wall time | 83.28s |
| Throughput | 2.414 tasks/s |

## Latency

| Set | p50 | p95 | Mean |
|---|---:|---:|---:|
| All final records | 1573.98 ms | 2815.81 ms | 1652.72 ms |
| Successful records | 1573.98 ms | 2815.81 ms | 1652.72 ms |

## Failures

No failures recorded.

## By subject

| Subject | Total | Success | Correct | Parsed accuracy | Strict accuracy |
|---|---:|---:|---:|---:|---:|
| college_physics | 19 | 19 | 7 | 36.84% | 36.84% |
| college_programming | 25 | 25 | 18 | 72.00% | 72.00% |
| computer_architecture | 21 | 21 | 14 | 66.67% | 66.67% |
| discrete_mathematics | 16 | 16 | 3 | 18.75% | 18.75% |
| high_school_mathematics | 18 | 18 | 9 | 50.00% | 50.00% |
| law | 24 | 24 | 14 | 58.33% | 58.33% |
| logic | 22 | 22 | 12 | 54.55% | 54.55% |
| middle_school_mathematics | 19 | 19 | 9 | 47.37% | 47.37% |
| operating_system | 19 | 19 | 12 | 63.16% | 63.16% |
| probability_and_statistics | 18 | 18 | 9 | 50.00% | 50.00% |

## Limitations

- This is a local, non-production run.
- The public C-Eval subset is not the official C-Eval leaderboard.
- Parsed accuracy excludes parse/provider failures; strict accuracy includes them in the denominator.
- Deterministic adapters are fixtures, not model-quality evidence.
