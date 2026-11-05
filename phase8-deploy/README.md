# Quarantine: not llm-lab

These notes are a deployment checklist for a different application. They are
not an llm-lab feature, not a test, and not evidence for the evaluation
claims in the repository root README.

They stay in the tree so the history is not rewritten as if the notes were
never here. `git archive` omits this directory (see `../.gitattributes`).
A normal clone still has it. To leave it out of a working tree, use the
sparse checkout in [`../SCOPE.md`](../SCOPE.md).

Scripts read `FORTUNE_PROJECT_ROOT`, `FORTUNE_BACKUP_ROOT`, and
`FORTUNE_PROJECT_BASH` from the environment. They no longer contain a
developer profile path. They are not run by CI.
