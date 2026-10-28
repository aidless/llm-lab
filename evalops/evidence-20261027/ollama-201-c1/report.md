# EvalOps report

- Model: `qwen25-3b-16k:latest`
- Provider: `ollama`
- Tasks: 201
- Concurrency: 1
- Task-set SHA-256: `33fdba0a807b721548cdac4a6339443c6edcd4bff00e895f62b77a3361bb9657`
- Config SHA-256: `39943495a2ffdfde3475da378a0eabbe72a888865c8f8eb1abff076d1a5a5778`
- Answer SHA-256: `6569fbbf44b50e363dc731324ef991124538a5f177f9bfb09a49da6ba82690fb`

## Outcome

| Metric | Value |
|---|---:|
| Successful provider responses | 201 |
| Parsed answers | 201 |
| Correct answers | 108 |
| Success rate | 100.00% |
| Parsed rate | 100.00% |
| Accuracy among parsed answers | 53.73% |
| Strict accuracy (all tasks) | 53.73% |
| Retry attempts | 0 |
| Wall time | 347.25s |
| Throughput | 0.579 tasks/s |

## Latency

| Set | p50 | p95 | Mean |
|---|---:|---:|---:|
| All final records | 1473.28 ms | 3638.26 ms | 1727.14 ms |
| Successful records | 1473.28 ms | 3638.26 ms | 1727.14 ms |

## Failures

No failures recorded.

## By subject

| Subject | Total | Success | Correct | Parsed accuracy | Strict accuracy |
|---|---:|---:|---:|---:|---:|
| college_physics | 19 | 19 | 7 | 36.84% | 36.84% |
| college_programming | 25 | 25 | 19 | 76.00% | 76.00% |
| computer_architecture | 21 | 21 | 14 | 66.67% | 66.67% |
| discrete_mathematics | 16 | 16 | 3 | 18.75% | 18.75% |
| high_school_mathematics | 18 | 18 | 10 | 55.56% | 55.56% |
| law | 24 | 24 | 14 | 58.33% | 58.33% |
| logic | 22 | 22 | 12 | 54.55% | 54.55% |
| middle_school_mathematics | 19 | 19 | 8 | 42.11% | 42.11% |
| operating_system | 19 | 19 | 12 | 63.16% | 63.16% |
| probability_and_statistics | 18 | 18 | 9 | 50.00% | 50.00% |

## Limitations

- This is a local, non-production run.
- The public C-Eval subset is not the official C-Eval leaderboard.
- Parsed accuracy excludes parse/provider failures; strict accuracy includes them in the denominator.
- Deterministic adapters are fixtures, not model-quality evidence.
