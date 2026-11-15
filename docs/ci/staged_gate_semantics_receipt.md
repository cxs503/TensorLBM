# Staged-gate ladder receipt

* source: `/srv/tensor/agent/work/m22498/.gitlab-ci.yml`
* stages: `quality`, `test`, `reporting`, `prerelease`, `integration`, `nightly`, `docs`, `deploy`
* jobs: 16  ·  blocking gates: 13  ·  advisory: lint-full, prerelease-pytorch-nightly, typecheck-full
* structural/artefact errors: **0**
* ladder claims (C1–C4): 67/67 verified

## Gate power (downstream jobs pruned when this gate fails alone)

| event | architecture-audit | build-documentation | coverage-report | format-check | gate-semantics | lint | nightly-full | package-build | pages | platform-test | repository-layout | testsuite-fast | typecheck-core |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mr | 3 | 0 | 2 | 8 | 8 | 8 | – | 3 | – | 3 | 3 | 3 | 8 |
| branch_push | 3 | 0 | 2 | 8 | 8 | 8 | – | 3 | – | 3 | 3 | 3 | 8 |
| main_push | 4 | 1 | 3 | 9 | 9 | 9 | – | 4 | 0 | 4 | 4 | 4 | 9 |
| schedule | – | – | – | – | – | – | 0 | – | – | – | – | – | – |
