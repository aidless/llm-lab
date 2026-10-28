# EvalOps report

- Model: `deterministic-fixture-v1`
- Provider: `deterministic`
- Tasks: 201
- Concurrency: 4
- Task-set SHA-256: `33fdba0a807b721548cdac4a6339443c6edcd4bff00e895f62b77a3361bb9657`
- Config SHA-256: `47ab0c179a794f400165f014e9feb8eae9e3ea2310a6603cdbd87820910dd404`
- Answer SHA-256: `77ace68bf48a4c309cf9ba5897c1d0238420cd0133cec9db8bb8be6a000781ed`

## Outcome

| Metric | Value |
|---|---:|
| Successful provider responses | 201 |
| Parsed answers | 201 |
| Correct answers | 46 |
| Success rate | 100.00% |
| Parsed rate | 100.00% |
| Accuracy among parsed answers | 22.89% |
| Strict accuracy (all tasks) | 22.89% |
| Retry attempts | 0 |
| Wall time | 0.02s |
| Throughput | 11167.298 tasks/s |

## Latency

| Set | p50 | p95 | Mean |
|---|---:|---:|---:|
| All final records | 0.00 ms | 0.01 ms | 0.01 ms |
| Successful records | 0.00 ms | 0.01 ms | 0.01 ms |

## Failures

No failures recorded.

## By subject

| Subject | Total | Success | Correct | Parsed accuracy | Strict accuracy |
|---|---:|---:|---:|---:|---:|
| college_physics | 19 | 19 | 9 | 47.37% | 47.37% |
| college_programming | 25 | 25 | 3 | 12.00% | 12.00% |
| computer_architecture | 21 | 21 | 4 | 19.05% | 19.05% |
| discrete_mathematics | 16 | 16 | 3 | 18.75% | 18.75% |
| high_school_mathematics | 18 | 18 | 3 | 16.67% | 16.67% |
| law | 24 | 24 | 4 | 16.67% | 16.67% |
| logic | 22 | 22 | 9 | 40.91% | 40.91% |
| middle_school_mathematics | 19 | 19 | 4 | 21.05% | 21.05% |
| operating_system | 19 | 19 | 6 | 31.58% | 31.58% |
| probability_and_statistics | 18 | 18 | 1 | 5.56% | 5.56% |

## Limitations

- This is a local, non-production run.
- The public C-Eval subset is not the official C-Eval leaderboard.
- Parsed accuracy excludes parse/provider failures; strict accuracy includes them in the denominator.
- Deterministic adapters are fixtures, not model-quality evidence.
