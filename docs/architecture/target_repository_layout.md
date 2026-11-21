# TensorLBM target repository layout

This is the target and migration map for the repository root. The tracked
baseline list is from `bcdd5fa4`. The
[layout policy](repository_layout.md) describes the rules enforced today.

## Maintained deliverables

| Root | Ownership and acceptance |
| --- | --- |
| `src/` | The installable solver and supported public APIs. Package discovery selects `src/tensorlbm/`. |
| `tests/` | Automated software verification, including fixtures and regression rules used only by tests. |
| `benchmarks/` | Canonical entry for each benchmark. A case identifies its runnable command, configuration, reference data, acceptance criteria and evidence. |
| `examples/` | Maintained user-facing demonstrations, including examples which a guide runs directly. |
| `docs/` | Public guides, manuals, published reports and their document assets. Preserve published URLs. |
| `tools/` | Maintained repository-wide tooling for build, release, indexing and validation. Case-specific runners live with their case. |
| `development/` | Research iterations, audit working records and historical project material. Organize by topic or mission with provenance, affected benchmark and reproduction prerequisites. |
| `app/` (conditional) | A supported application can retain one product root with its backend, frontend and integration tests. Confirm the released product and deployment contract before consolidating. |
| `.github/` | GitHub CI and contributor automation. |

The benchmark registry is the discovery contract. A benchmark entry may link
to a retained audit, a published report or an immutable external dataset.
Preserve the distinction between archive verdicts and current acceptance. Record
the revision, reference, metric, threshold and result for each new acceptance
claim. Use a versioned external location for heavy output, with retrieval and
integrity verified before retiring a tracked copy.

## Existing first-level directory disposition

| Existing directory | Target | Constraint or decision |
| --- | --- | --- |
| `.audit/` | `development/audit/` | The [archive ledger](migrations/hidden-audit-root-2026-11-21.json) maps historical replay data, drivers and the standalone audit client. |
| `.github/` | Keep | Preserve workflow and branch-check names. |
| `TensorLBM/` | `development/legacy/TensorLBM/` | The full checkout is frozen and historical replay uses the archived solver. The [migration ledger](migrations/legacy-checkout-relocation-2026-11-21.json) records the exact tree and verified consumers. |
| `_scratch/` | `development/archived/scratch/` | The [archive ledger](migrations/scratch-root-retirement-2026-11-21.json) records the prior paths and the output paths for current runs. |
| `app/` | Keep if supported | Review product ownership, launch commands and deployment. |
| `artifacts/` | Split by consumer | Keep curated evidence with cases or audits. Use ignored output paths or external storage for new raw output. |
| `audit/` | `development/audit/` | Preserve mission identifiers, immutable evidence and links from published reports. |
| `audits/` | Merge with mission archive | #2837 stages consolidation under `audit/` before the wider archive move. |
| `benchmarks/` | Keep | One canonical entry per case. Preserve registered IDs and archived comparisons. |
| `cases/` | Split by purpose | The [migration ledger](migrations/case-root-retirement-2026-11-21.json) maps all case runners to their benchmark owners and records the two superseded case-pool wrappers. |
| `configs/` | `tests/fixtures/campaign_gates/`, `scripts/data/fsi/` | Acoustic campaign test fixture and FSI evaluator inputs. |
| `demos/` | `app/demos/`, `examples/` | Drag-echo Web assets stay with the application. The propeller example lives with other Python examples. |
| `docs/` | Keep | Published manual and report paths are compatibility contracts. |
| `evidence/` | Case data, document assets, development evidence and tools | The [migration ledger](migrations/evidence-root-retirement-2026-11-21.json) lists original blobs and updated replay commands. |
| `examples/` | Keep | Preserve documented commands and supported entry points. |
| `experiments/` | `development/experiments/` | The [migration ledger](migrations/experiment-root-retirement-2026-11-21.json) traces archived and exploratory assets. Maintained benchmark entry points remain indexed by case. |
| `figures/` | Case evidence, document assets, or development archives | The [migration ledger](migrations/figures-root-retirement-2026-11-21.json) preserves historical paths. Migrated commands use `outputs/figures/`. Unmigrated commands can still use the ignored local `figures/`. |
| `frontend-vue/` | `app/frontend-vue/` | Keep Vue source and its build with the serving app. The [migration ledger](migrations/vue-application-root-2026-11-21.json) records file provenance. |
| `models/` | Split by role | Judge supported library, model-unit CI, case inputs and historical weights separately. Preserve discoverability for model-unit jobs. |
| `notebooks/` | `examples/notebooks/` | Quickstart and its Colab entry. |
| `papers/` | `docs/`, `benchmarks/` or `development/` | Published material, supporting case assets and working drafts have separate owners. |
| `references/` | `scripts/data/air_layer_drag/` | Air-layer reference JSON shared by the campaign and literature checker. |
| `regression_rules/` | `tests/fixtures/regression_rules/` | Preserve the machine-readable contracts. |
| `runs/` | Historical evidence or external storage | Retain existing Git evidence until exact provenance, checksums, consumers and retrieval are verified. New raw output is ignored. |
| `scripts/` | `tools/`, owner case or `development/` | Separate maintained workflow tooling from case-specific and one-off scripts. |
| `src/` | Keep | Keep the installable package and its published import surface stable. |
| `teaching/` | `examples/teaching/` | Keep teaching commands and the companion comparison together. |
| `tests/` | Keep | Preserve test collection and the semantics of fixtures and rules. |
| `tmp/` | No tracked root | Reused paths are recorded by source revision. See the [m26816](migrations/m26816-root-retirement-2026-11-21.json) and [m26843](migrations/m26843-root-report-2026-11-21.json) ledgers for distinct versions of `tmp/report.json`. |
| `tools/` | Keep | Maintain shared tooling and repository policy together. |
| `xval/` | `benchmarks/` and `development/verification/` | Move by campaign. Link formal evidence from the canonical benchmark entry and preserve method, references and historical verdicts. |

## Root files and rollout

The [six-root migration ledger](migrations/bulk-root-retirement-2026-11-21.json)
records the paths, original Git objects, checksums and compatibility checks for
`configs/`, `demos/`, `notebooks/`, `references/`, `regression_rules/`
and `teaching/`.

Keep public entry files (`README.md`, `README.zh-CN.md`, `CONTRIBUTING.md`,
`CHANGELOG.md`, `LICENSE`, `CITATION.cff`, `AUTHORS`, `SECURITY.md`), contributor
configuration (`AGENTS.md`, `CODEOWNERS`, `.gitignore`, `.editorconfig`,
`.pre-commit-config.yaml`), automation (`.gitlab-ci.yml`, `.dockerignore`) and
build inputs (`pyproject.toml`, `requirements.txt`, `uv.lock`, `Makefile`,
`release.sh`) at their current paths. Audit package and lock-file authority
separately from this migration.

1. Pilot the archive with unchanged historical assets and a link from the
   associated case. Record old and new paths, source revision and checksums.
2. Route mixed roots by consumer, starting with `cases/`. Move low-coupling
   material in small batches. Retain aliases or stable public paths when in use.
3. Migrate active runners only after reviewing import, CLI, fixture and CI
   dependencies. Record runtime verification separately when needed.
4. Remove a retired root from the policy when its tracked contents are accounted
   for and its consumers are updated. Run directory, manual and registry checks
   on every batch. This pilot performs structural checks only.
