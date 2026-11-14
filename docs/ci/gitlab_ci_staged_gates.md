# TensorLBM GitLab CI Staged-Gates Design

## Overview

This document describes the staged-gate CI pipeline design for TensorLBM,
derived from a cross-validation (xval) study comparing TensorLBM's existing
GitHub Actions CI with the OSS lbmpy project's GitLab CI staged-gate ladder.

**Reference**: lssfau/lbmpy `.gitlab-ci.yml` (tag: `gitlab-ci-staged-gates`)

## XVAL Analysis Summary

The xval study evaluated three alternatives against TensorLBM's pytest corpus
(16,496 items):

| Alternative | Description | Guard Fraction (MR) | Verdict |
|---|---|---|---|
| ALT-0 | Keep GitHub Actions (baseline) | 99.4% | Baseline |
| ALT-1 | Transplant OSS ladder verbatim | 100% (leaks 100% slow tier) | Rejected |
| ALT-2 | Adapted ladder + marker vocabulary | 99.4% (0% slow-tier leak) | **Chosen** |

### Key Findings

**H1 (Functional Parity)**: HOLDS — TensorLBM's GitHub Actions CI achieves
guard_fraction within 15% of the OSS staged-gate ladder on every event
(mr, branch_push, main_push, schedule). The relative difference is <1%.

**H2 (Vocabulary Portability)**: HOLDS — The OSS marker expression
`-m "not longrun"` is NOT portable to TensorLBM. When applied verbatim, it
leaks 100% of TensorLBM's slow-tier tests into the fast gate (because
TensorLBM uses `slow`, not `longrun`). The adapted expression `-m "not slow"`
leaks 0%.

**Decision**: ALT-2 — transplant the structural/declarative elements that
survive the KPI test, while adapting the marker vocabulary.

## Staged-Gate Ladder Design

The `.gitlab-ci.yml` implements an 8-stage ladder:

```
quality → test → reporting → prerelease → integration → nightly → docs → deploy
  (4)      (5)      (1)         (1)          (0)          (1)      (1)     (1)
```

Numbers in parentheses show active job counts per stage.

### Stage Descriptions

| Stage | Purpose | Tier | Jobs |
|---|---|---|---|
| **quality** | Static analysis gates | blocking + advisory | lint, format-check, typecheck-core, typecheck-full (advisory) |
| **test** | Unit + fast integration tests | blocking | testsuite-fast, platform-test, repository-layout, architecture-audit, package-build |
| **reporting** | Coverage aggregation | blocking | coverage-report |
| **prerelease** | Pre-release dependency checks | advisory | prerelease-pytorch-nightly |
| **integration** | Full integration (declared-but-disabled) | — | *(commented out)* |
| **nightly** | Scheduled full-suite runs | conditional | nightly-full |
| **docs** | Documentation build | blocking | build-documentation |
| **deploy** | Pages deployment | conditional | pages |

### Gate-Tier Taxonomy

- **blocking**: Failure stops the pipeline (lint, test, coverage-report, etc.)
- **advisory**: Failure is informational; pipeline continues (typecheck-full, prerelease)
- **conditional**: Only runs on specific events (nightly on schedule, pages on default branch)
- **manual**: Requires explicit trigger (not currently used)

### Marker Vocabulary Adapter

The key ALT-2 adaptation:

| OSS (lbmpy) | TensorLBM (adapted) | Effect |
|---|---|---|
| `-m "not longrun"` | `-m "not slow"` | Fast gate excludes slow-tier tests |
| `longrun` marker | `slow` marker | Repository-local marker name |

### Declared-But-Disabled Gate Hygiene

Following the lbmpy pattern, the `integration` stage is declared but has no
active job (commented out). This documents the integration test surface without
running it on every pipeline, reducing CI cost while preserving the intent.

### DAG Dependencies (needs:)

- `coverage-report` needs `testsuite-fast` (coverage data flows from test to reporting)
- `pages` needs `coverage-report` and `build-documentation` (deploy aggregates both)

## Files

| Path | Purpose |
|---|---|
| `.gitlab-ci.yml` | The ALT-2 staged-gate pipeline |
| `xval/gitlab_ci_staged_gates/` | XVAL analysis framework |
| `xval/gitlab_ci_staged_gates/ci_spec.py` | CI definition normalizer |
| `xval/gitlab_ci_staged_gates/gate_sim.py` | Gate-ladder simulator |
| `xval/gitlab_ci_staged_gates/oss_refs/` | OSS reference files (content-addressed) |
| `xval/gitlab_ci_staged_gates/data/` | Corpus snapshot + measurements |
| `xval/gitlab_ci_staged_gates/predeclared.json` | Pre-declared experiment protocol |
| `xval/gitlab_ci_staged_gates/summary.json` | Analysis results |
| `tests/test_ci_staged_gates_xval.py` | Acceptance tests (18 tests) |

## Acceptance Tests

Run: `PYTHONPATH=. pytest tests/test_ci_staged_gates_xval.py -v`

The 18 tests verify:
1. OSS reference integrity (content-addressed, correct structure)
2. Marker expression semantics (pytest boolean rules)
3. Simulator fidelity (<15% error vs pytest --collect-only)
4. H1 functional parity (guard_fraction within 15%)
5. H2 vocabulary non-portability (verbatim leaks ≥15%, adapted leaks <15%)
6. Guard coverage on default events
7. `.gitlab-ci.yml` structural correctness (stages, tiers, markers, needs, etc.)
8. Parity between `.gitlab-ci.yml` and GitHub Actions CI
