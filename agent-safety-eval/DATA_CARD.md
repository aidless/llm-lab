# Agent Safety Dataset Card

> **本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.**

## Dataset

- Version: `agent-safety-eval-96-v2`
- Tasks: 96 synthetic runtime-policy scenarios
- Judge cases: 200 request-level calibration cases
- Categories: prompt injection, unauthorized tool calls, dangerous file
  operations, data exfiltration, stop failures and tool-error recovery
- File format: canonical JSONL plus a SHA-256 manifest
- License: repository MIT license applies to the generated dataset

## Composition

Each category has 16 tasks. The task set includes 32 benign controls and 64
must-halt tasks. Tasks contain mock files, mock URLs, mock tool responses and
expected behavioral labels. The 200 judge cases are balanced by reference
label and cover prompt-injection, unauthorized-request and normal-workflow
slices.

## Collection Process

The dataset was authored as synthetic, non-operational text for this local
evaluation. No customer data, production credentials, private account data or
real malware is included. The apparent secret markers are deterministic test
fixtures and have no external value.

Labels are assistant-authored drafts. They remain outside the human ground
truth boundary until a reviewer identifier, UTC timestamp and explicit review
decision are recorded through the review workflow.

## Intended Use

- testing frozen-task hashes, trajectory capture and runtime policy plumbing;
- separating prompt-only, runtime-gate-only and combined policy effects on
  mock tools through the `off`, `prompt`, `gate`, `on` arms;
- calibrating local judges and review workflows;
- reproducing the checked-in local evidence.

## Out-of-Scope Use

- production guardrail certification;
- real sandbox-escape or malware testing;
- frontier-model safety ranking;
- regulatory compliance or field deployment decisions;
- treating assistant draft labels as human annotations.

## Known Limitations

- Tiny local CPU models are not representative of production agents.
- The task distribution is synthetic and intentionally small.
- Tool effects are implemented in memory; no real filesystem, shell, email or
  network execution occurs.
- The project cannot satisfy its human-review acceptance gate without an
  independent reviewer.
