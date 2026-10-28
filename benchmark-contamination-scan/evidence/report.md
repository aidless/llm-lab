# Benchmark Contamination Preflight

> Decision: **HOLD**
> **本地安全评测，不是生产安全系统。 / Local safety evaluation, not a production security system.**

- Frozen tasks: **201**
- Corpus documents: **240**
- N-gram threshold: `0.40`
- Semantic threshold: `0.90`

## Test metrics

| Metric | Value | 95% Wilson |
|---|---:|---:|
| Contamination rate | 25.7% | n/a |
| False positive rate | 1.4% | [0.2%, 7.6%] |
| False negative rate | 13.3% | [5.3%, 29.7%] |
| Precision | 96.3% | n/a |
| Recall | 86.7% | n/a |
| F1 | 0.912 | n/a |

## Gate

- `human_review_lt_40`

## EvalOps integration

```bash
python3 -m contamination_scan report
python3 -m contamination_scan gate --report evidence/report.json
```

A non-`PASS` decision exits non-zero and must stop an EvalOps run.

## Limitations

- The local corpus is a seeded audit fixture, not a model's pretraining corpus.
- Gold labels are known only because this fixture intentionally injects copies.
- Do not describe fixture metrics as evidence about any specific pretrained model.
- Threshold tuning uses the dev split only, but the fixture is synthetic.
- Embedding similarity is model-dependent and not a proof of memorization.
- A clean result cannot prove a model was not exposed to the benchmark.
