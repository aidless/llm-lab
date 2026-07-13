# ADR-0009: CycloneDX SBOM auto-generation in CI

- **Status:** Accepted
- **Date:** 2026-07-13
- **Deciders:** @llm-lab/maintainers

## Context and problem statement

For the "eval tool you can show your security team" wedge, downstream
operators need to answer: "exactly what dependencies shipped in this
release?" A Software Bill of Materials (SBOM) is the standard answer.

We ship a Python package. The standard SBOM format for software supply
chain visibility is **CycloneDX** (also SPDX; we picked CycloneDX because
it has better Python tooling).

## Decision drivers

- **Zero new runtime deps.** `cyclonedx-bom` is a *build-time* dep, not
  shipped with the package itself.
- **Reproducible per-release.** The SBOM must be regenerated for every
  release tag, not hand-curated.
- **Standard format.** CycloneDX 1.5 JSON is what scanners (Trivy, Grype,
  Anchore) consume natively.

## Considered options

1. **CycloneDX via `cyclonedx-bom` CLI in CI** (chosen). One job, one
   command, JSON output uploaded as a release artifact.
2. **SPDX via `pip-tools`.** More universal but heavier tooling and SPDX
   Python support is less mature.
3. **Hand-curated `requirements.txt` as the SBOM.** Loses transitive
   information — exactly the data a security team needs.

## Decision outcome

**Option 1.**

A new `sbom` job in `.github/workflows/test.yml`:

```yaml
- cyclonedx-py environment \
    --output sbom.cdx.json \
    --spec-format 1.5 \
    --output-format JSON \
    --pyproject pyproject.toml
```

The resulting `sbom.cdx.json` is uploaded as the `sbom-cyclonedx` GitHub
Actions artifact and (on release tags) attached to the GitHub release.

### Consequences

**Positive:**

- Per-release SBOM is reproducible.
- Standard format — works with every modern scanner.
- Zero runtime dep cost.

**Negative:**

- `cyclonedx-bom` is itself a dependency. If its upstream is compromised,
  our SBOM could be inaccurate. Mitigated by pinning the version in
  CI; long-term, verify the SBOM hash against a second tool.

**Neutral:**

- SPDX fans can run `syft` or `spdx-tools` separately. We don't ship SPDX
  today; revisit if a buyer asks.

## Validation

- The `sbom` job runs on every PR and every push to `main`. If the
  command fails, CI fails.
- A spot-check: download the artifact, run `jq '.components | length'`
  to confirm the component list is non-empty.

## Links / references

- CycloneDX spec: <https://cyclonedx.org/specification/overview/>
- `cyclonedx-bom` docs: <https://github.com/CycloneDX/cyclonedx-python>