# Notice and boundary

The scanner implementation is original. No upstream contamination-detection
source code or benchmark rows are copied into this package.

## Upstream benchmark

The frozen 201-question task set is the existing llm-lab C-Eval validation
subset under `evalops/ceval-201/`. C-Eval remains subject to
`CC-BY-NC-SA-4.0`; see the upstream
[NOTICE](../evalops/ceval-201/NOTICE.md) and source revision before reuse.
This package does not change the benchmark's license or imply an official
C-Eval leaderboard result.

## Seeded corpus

`dataset/corpus-v1.jsonl` is a deterministic audit fixture. Sixty documents are
intentionally seeded from benchmark items so false positives and false
negatives can be measured. The fixture is not a model's pretraining snapshot.

## Design references

Public work on contamination detection informed the review structure only. Any
external implementation used in a future audit must be pinned, license-audited
and recorded here before its code or data is redistributed.

> **本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.**
