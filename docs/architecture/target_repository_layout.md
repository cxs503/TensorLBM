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
| `.audit/` | `development/audit/` | Audit import and command consumers first. |
| `.github/` | Keep | Preserve workflow and branch-check names. |
| `TensorLBM/` | `development/legacy/` | Frozen snapshot. Move only after old imports and execution paths have migrated or been documented as historical. |
| `_scratch/` | `development/archived/` | Check tracked assets and source references. Keep new local scratch ignored. |
| `app/` | Keep if supported | Review product ownership, launch commands and deployment. |
| `artifacts/` | Split by consumer | Keep curated evidence with cases or audits. Use ignored output paths or external storage for new raw output. |
| `audit/` | `development/audit/` | Preserve mission identifiers, immutable evidence and links from published reports. |
| `audits/` | Merge with mission archive | #2837 stages consolidation under `audit/` before the wider archive move. |
| `benchmarks/` | Keep | One canonical entry per case. Preserve registered IDs and archived comparisons. |
| `cases/` | Split by purpose | Benchmark cases to `benchmarks/`, supported demos to `examples/`, exploratory material to `development/`. Pilot includes m18115 and two case-pool runners with legacy CLI wrappers. |
| `configs/` | `tests/fixtures/campaign_gates/`, `scripts/data/fsi/` | Acoustic campaign test fixture and FSI evaluator inputs. |
| `demos/` | `app/demos/`, `examples/` | Drag-echo Web assets stay with the application. The propeller example lives with other Python examples. |
| `docs/` | Keep | Published manual and report paths are compatibility contracts. |
| `evidence/` | Split by purpose | Case acceptance evidence stays discoverable from its case. Review, investigation and supporting records may live in `development/`. |
| `examples/` | Keep | Preserve documented commands and supported entry points. |
| `experiments/` | `development/experiments/` | Promote any maintained cases before moving exploratory and archived work. |
| `figures/` | Split by owner | Published figures to their report or case. Keep plot indexes and generator paths together. #2839 stages the Sod figures. |
| `frontend-vue/` | Under `app/` if supported | Audit package manager, static output, deployment and CI paths. |
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
| `tmp/` | No tracked root | #2839 stages the publisher archive. Future local scratch should be ignored. |
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
