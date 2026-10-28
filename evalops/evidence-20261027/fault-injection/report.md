# EvalOps report

- Model: `deterministic-fixture-v1`
- Provider: `fault-injection`
- Tasks: 6
- Concurrency: 2
- Task-set SHA-256: `6991f66ffa17295d0245491b886a97b855637c3fb28167a8a9541f7c0e42ef4e`
- Config SHA-256: `f7a5aa1421d9b570df6aaedcb998b235398d8f7aca56fb46e3c20906c21e4914`
- Answer SHA-256: `970a0e65f38caa2498281cd279cd7310b7c5e228415cb2617561bac92bce06bd`

## Outcome

| Metric | Value |
|---|---:|
| Successful provider responses | 2 |
| Parsed answers | 2 |
| Correct answers | 2 |
| Success rate | 33.33% |
| Parsed rate | 33.33% |
| Accuracy among parsed answers | 100.00% |
| Strict accuracy (all tasks) | 33.33% |
| Retry attempts | 0 |
| Wall time | 0.00s |
| Throughput | 3116.218 tasks/s |

## Latency

| Set | p50 | p95 | Mean |
|---|---:|---:|---:|
| All final records | 0.00 ms | 0.01 ms | 0.00 ms |
| Successful records | 0.00 ms | 0.00 ms | 0.00 ms |

## Failures

| Category | Count |
|---|---:|
| `model_error` | 1 |
| `parse_error` | 1 |
| `timeout` | 1 |
| `transport` | 1 |

## By subject

| Subject | Total | Success | Correct | Parsed accuracy | Strict accuracy |
|---|---:|---:|---:|---:|---:|
| college_programming | 6 | 2 | 2 | 100.00% | 33.33% |

## Limitations

- This is a local, non-production run.
- The public C-Eval subset is not the official C-Eval leaderboard.
- Parsed accuracy excludes parse/provider failures; strict accuracy includes them in the denominator.
- Deterministic adapters are fixtures, not model-quality evidence.
