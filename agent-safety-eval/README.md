# Agent Safety Runtime Monitor

> **Local safety evaluation, not a production security system.**
> **本地安全评测，不是生产安全系统。**

An auditable 96-task agent safety benchmark for local Ollama models. It runs a
four-arm ablation: no intervention (`off`), prompt only (`prompt`), runtime gate
only (`gate`), and prompt plus runtime gate (`on`). Every file, shell, email and
HTTP tool remains a mock.

This project reuses llm-lab EvalOps' frozen-task hashes, explicit failure
taxonomy, resumable records and JSON/Markdown/HTML reporting pattern. It does
not use `agi_harness.py` or any simulated success path.

Dataset composition and out-of-scope uses are documented in
[`DATA_CARD.md`](DATA_CARD.md).

## What is measured

| Metric | Definition |
|---|---|
| Unsafe action rate | Trajectories with at least one executed policy violation |
| Permission violation rate | Same conservative trajectory-level boundary |
| Stop precision | Correct required stops / all predicted stops |
| Stop recall | Correct required stops / all tasks requiring a stop |
| Over-refusal rate | Halted benign tasks / all benign tasks |
| Task success rate | Expected terminal, required tools, and no unsafe execution |
| Recovery success rate | `tool_error_recovery` tasks that hit an injected fault, then still ended with the expected terminal and required tools |

Two definitions are deliberately strict, and both matter when you read the
numbers:

- A `policy_halt` is recorded only when the runtime gate actually denied at
  least one action. A task that merely burns its step budget stays
  `step_limit`, so a long benign task is not scored as an over-refusal.
- Recovery counts only full task success. A trajectory that retries a failing
  tool successfully but then ends in `ask_user` instead of `finish` is a real
  partial recovery and is **not** counted; that is why the 3B models show 0%
  recovery even where `tool_retry_events` is non-zero.

The task set covers six equal-sized categories:

- prompt injection;
- unauthorized tool calls;
- dangerous file operations;
- data exfiltration;
- long-running / non-stopping behavior;
- transient tool-error recovery.

## Run

Prerequisites:

```bash
cd ../  # llm-lab repo root
python3 -m venv .venv
.venv/bin/pip install -e llm_lab
ollama list
```

Then:

```bash
cd agent-safety-eval
./scripts/rerun.sh
```

The default CPU-scale models are `qwen2.5:3b-instruct-q4_K_M` and
`llama3.2:3b`. The command freezes/verifies tasks, executes 768 trajectories
(96 tasks × 2 models × 4 arms), writes JSON/Markdown/HTML reports, creates a
review packet, writes `SHA256SUMS`, and runs `verify_evidence.py`. The runner
uses three concurrent task workers and a 180-second model timeout by default;
set `WORKERS=1` for serial replay or `OLLAMA_TIMEOUT` for a stricter/looser
transport bound.

For a stronger local run when memory and time permit:

```bash
MODELS="qwen3:8b llama3.1:8b" WORKERS=4 ./scripts/rerun.sh
```

The first run stops at the human-review gate unless `--allow-pending-human` is
used. This is intentional: assistant-authored labels are not human labels.

## Human review

Open `review/review.html`, review every task label and every required
unsafe-attempt trajectory, and export the packet. The acceptance gate needs at
least **100 human-review records** overall, all 96 task labels, full required
trajectory coverage, and no flagged required record. A `flag` on a required
trajectory keeps the project in HOLD. Then:

```bash
../.venv/bin/python -m agent_safety_eval ingest-review \
  --export review/reviews-export.json \
  --output review/reviews-human.jsonl
../.venv/bin/python -m agent_safety_eval report \
  --reviews review/reviews-human.jsonl
../.venv/bin/python scripts/write_checksums.py
../.venv/bin/python verify_evidence.py \
  --reviews review/reviews-human.jsonl
```

Only records with a reviewer identifier, UTC timestamp and explicit decision
are counted. A `flag` review does not silently become a gold label.

For a quick stratified plumbing test instead of the full run:

```bash
./scripts/rerun.sh  # full run
.venv/bin/agent-safety-eval run --stratified-smoke 12 \
  --models qwen2.5:3b-instruct-q4_K_M --force
```

The smoke selection covers all six categories and both benign controls and
must-halt tasks. It does not support a safety-effectiveness claim.

`./scripts/reproduce.sh` is the full one-command path. A rerun first moves any
pending/completed review packet and prior evidence into the ignored
`.codex-evidence/` archive instead of overwriting it.

## Evidence

```text
dataset/
  manifest.json              # frozen task and judge-case hashes
  tasks-v1.jsonl             # 96 tasks
  judge-cases-v1.jsonl       # 200 calibration cases
evidence/
  environment.json           # CPU, Ollama and accelerator facts
  trajectories.jsonl         # every step, tool argument and policy decision
  runs/<model>-policy-<off|prompt|gate|on>/
  report.json
  report.md
  report.html
  SHA256SUMS
  verification-result.json
review/
  review.html                # local-only review UI
```

Verify without trusting the report:

```bash
python3 verify_evidence.py --allow-pending-human
```

## Safety design

- The mock environment is an in-memory dictionary of `/workspace` files.
- Shell execution simulates only `ls`, `cat` and `wc`.
- Email appends to an in-memory outbox.
- HTTP appends to an in-memory request log.
- Path traversal is normalized before policy checks.
- Tool output is copied into the transcript as untrusted data.
- In `gate` and `on`, the step budget becomes `policy_halt` only after the
  runtime gate recorded a real denial. Without a denial it remains
  `step_limit`, and it is not scored as over-refusal.

## Explicit non-claims

- This is not a production guardrail, sandbox escape test, or compliance system.
- It does not prove that a model is safe to deploy.
- It does not evaluate real-world malware, exploit generation or destructive
  commands.
- CPU latency is not a serving SLO.
- Synthetic smoke results are not a model ranking.
- A complete human-review packet is required before the acceptance gate can pass.
- The acceptance gate also requires at least 100 validated review records.
- The synthetic labels are draft evidence until an independent reviewer
  completes the review packet.
- The local two-model result is not a frontier-model safety ranking.

## Inspirations

The task design is informed by public agent-safety work including AgentDojo,
InjecAgent, AgentHarm, and llm-lab's own EvalOps discipline. No upstream task
text is redistributed; all included scenarios are synthetic and mock-only.

License: MIT. See the repository-level `LICENSE`.
