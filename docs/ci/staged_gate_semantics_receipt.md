# Staged-gate ladder receipt

* source: `.gitlab-ci.yml` sha256:`9817144b95505349`
* stages: `quality`, `test`, `reporting`, `prerelease`, `integration`, `nightly`, `docs`, `deploy`
* jobs: 16  ·  blocking gates: 13  ·  advisory: lint-full, prerelease-pytorch-nightly, typecheck-full
* structural/artefact errors: **0**
* ladder claims (C1–C4): 67/67 verified

## Ladder inventory (generated from the CI definition)

8 declared stages · 16 jobs · 13 blocking · 3 advisory · 0 manual · 1 declared-but-empty (integration)

| stage | active jobs | tier mix | jobs (tier · events) |
|---|---|---|---|
| `quality` | 6 | advisory 2, blocking 4 | `format-check` (blocking · mr, branch_push, main_push)<br>`gate-semantics` (blocking · mr, branch_push, main_push)<br>`lint` (blocking · mr, branch_push, main_push)<br>`lint-full` (advisory · mr, branch_push, main_push)<br>`typecheck-core` (blocking · mr, branch_push, main_push)<br>`typecheck-full` (advisory · mr, branch_push, main_push) |
| `test` | 5 | blocking 5 | `architecture-audit` (blocking · mr, branch_push, main_push)<br>`package-build` (blocking · mr, branch_push, main_push)<br>`platform-test` (blocking · mr, branch_push, main_push)<br>`repository-layout` (blocking · mr, branch_push, main_push)<br>`testsuite-fast` (blocking · mr, branch_push, main_push) |
| `reporting` | 1 | blocking 1 | `coverage-report` (blocking · mr, branch_push, main_push) |
| `prerelease` | 1 | advisory 1 | `prerelease-pytorch-nightly` (advisory · mr, branch_push, main_push) |
| `integration` | 0 | — | *(declared, no active job)* |
| `nightly` | 1 | blocking 1 | `nightly-full` (blocking · schedule) |
| `docs` | 1 | blocking 1 | `build-documentation` (blocking · mr, branch_push, main_push) |
| `deploy` | 1 | blocking 1 | `pages` (blocking · main_push) |

## Gate power (downstream jobs pruned when this gate fails alone)

| event | architecture-audit | build-documentation | coverage-report | format-check | gate-semantics | lint | nightly-full | package-build | pages | platform-test | repository-layout | testsuite-fast | typecheck-core |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mr | 3 | 0 | 2 | 8 | 8 | 8 | – | 3 | – | 3 | 3 | 3 | 8 |
| branch_push | 3 | 0 | 2 | 8 | 8 | 8 | – | 3 | – | 3 | 3 | 3 | 8 |
| main_push | 4 | 1 | 3 | 9 | 9 | 9 | – | 4 | 0 | 4 | 4 | 4 | 9 |
| schedule | – | – | – | – | – | – | 0 | – | – | – | – | – | – |
