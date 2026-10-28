# Legacy 72-task compatibility fixture

> **本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.**

This directory is retained only because `llm_lab/tests/test_safety.py` uses a
smaller, independently frozen 72-task version to test package-level safety
primitives.

It is **not** the benchmark described in
[`agent-safety-eval`](../../agent-safety-eval/README.md), and it is not the
source of the checked-in portfolio evidence. The canonical 96-task benchmark,
200 judge cases, review packet and 384 trajectories live under
`agent-safety-eval/`.
