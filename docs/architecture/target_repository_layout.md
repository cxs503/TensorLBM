# TensorLBM target repository layout

This is the target and migration map for the repository root. The tracked
baseline list is from `bcdd5fa4`. The
[layout policy](repository_layout.md) describes the rules enforced today.

## Maintained deliverables

| Root | Ownership and acceptance |
| --- | --- |
| `src/` | `src/tensorlbm/` holds the installable solver implementations. `src/lbm/` contains compatibility imports for the relocated D3Q27 implementation ([migration ledger](migrations/d3q27-canonical-source-2026-11-23.json)). The source-authority gate limits this compatibility package to five known files. |
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
| `BB/` | `development/audit/drop_entry/BB/` | The [m26963 ledger](migrations/m26963-bb-root-retirement-2026-11-22.json) preserves the published URL. The [m26954](migrations/m26954-bb-root-retirement-2026-11-22.json), [m26952](migrations/m26952-bb-root-retirement-2026-11-22.json) and [m26998](migrations/m26998-bb-root-retirement-2026-11-22.json) ledgers record the internal work-card relocations. |
| `TensorLBM/` | `development/legacy/TensorLBM/` | The full checkout is frozen and historical replay uses the archived solver. The [migration ledger](migrations/legacy-checkout-relocation-2026-11-21.json) records the exact tree and verified consumers. |
| `_scratch/` | `development/archived/scratch/` | The [archive ledger](migrations/scratch-root-retirement-2026-11-21.json) records the prior paths and the output paths for current runs. |
| `app/` | Keep if supported | Review product ownership, launch commands and deployment. |
| `artifacts/` | Retire from the tracked tree | The [retirement ledger](migrations/artifacts-root-retirement-2026-11-22.json) maps the final 19 files to their owners. Existing scripts and CI can still write to the ignored local path. |
| `audit/` | `development/audit/` | The [migration ledger](migrations/audit-root-retirement-2026-11-21.json) records original Git objects, updated launch paths and the historical checkout. |
| `audits/` | `development/audit/` | The [original ledger](migrations/legacy-audit-root-2026-11-21.json) records the intermediate paths under `audit/`. |
| `benchmarks/` | Keep | One canonical entry per case. Preserve registered IDs and archived comparisons. |
| `cases/` | Split by purpose | The [migration ledger](migrations/case-root-retirement-2026-11-21.json) maps all case runners to their benchmark owners and records the two superseded case-pool wrappers. |
| `configs/` | `tests/fixtures/campaign_gates/`, `scripts/data/fsi/` | Acoustic campaign test fixture and FSI evaluator inputs. |
| `demos/` | `app/demos/`, `examples/` | Drag-echo Web assets stay with the application. The propeller example lives with other Python examples. |
| `drop_entry/` | `development/audit/drop_entry/` | Mission working notes and drafts are development records. The [m26734](migrations/m26734-drop-entry-reentry-2026-11-21.json), [m26737](migrations/m26737-drop-entry-reentry-2026-11-22.json), [m27075](migrations/m27075-drop-entry-reentry-2026-11-22.json), [m26892](migrations/m26892-drop-entry-reentry-2026-11-22.json), [m26887](migrations/m26887-drop-entry-reentry-2026-11-22.json), and [m26894](migrations/m26894-drop-entry-reentry-2026-11-22.json) re-entry ledgers preserve successive root regressions byte-for-byte while retaining published document copies where they exist. |
| `docs/` | Keep | Published manual and report paths are compatibility contracts. |
| `evidence/` | Case data, document assets, development evidence and tools | The [migration ledger](migrations/evidence-root-retirement-2026-11-21.json) lists original blobs and updated replay commands. |
| `examples/` | Keep | Preserve documented commands and supported entry points. |
| `experiments/` | `development/experiments/` | The [migration ledger](migrations/experiment-root-retirement-2026-11-21.json) traces archived and exploratory assets. Maintained benchmark entry points remain indexed by case. |
| `figures/` | Case evidence, document assets, or development archives | The [migration ledger](migrations/figures-root-retirement-2026-11-21.json) preserves historical paths. Migrated commands use `outputs/figures/`. Unmigrated commands can still use the ignored local `figures/`. |
| `frontend-vue/` | `app/frontend-vue/` | Keep Vue source and its build with the serving app. The [migration ledger](migrations/vue-application-root-2026-11-21.json) records file provenance. |
| `models/` | Split by role | The [model-unit input ledger](migrations/model-unit-inputs-2026-11-23.json) maps regression units to `benchmarks/model_units/`. Retain the published SUBOFF checkpoint at its existing path until its retrieval and runtime compatibility are verified for retirement. |
| `notebooks/` | `examples/notebooks/` | Quickstart and its Colab entry. |
| `papers/` | `docs/`, `benchmarks/` or `development/` | Retired by the [papers migration ledger](migrations/papers-root-retirement-2026-11-22.json): 35 tracked assets moved to SUBOFF benchmark evidence or published report ownership; active path consumers were updated and unresolved external evidence gaps remain explicit. |
| `raw/` | Retire | The [diagnostic duplicate ledger](migrations/drop-entry-raw-root-retirement-2026-11-23.json) maps the four byte-identical files back to their benchmark owners. |
| `references/` | `scripts/data/air_layer_drag/` | Air-layer reference JSON shared by the campaign and literature checker. |
| `regression_rules/` | `tests/fixtures/regression_rules/` | Preserve the machine-readable contracts. |
| `runs/` | `development/archived/historical_runs/` | The [migration ledger](migrations/run-root-retirement-2026-11-21.json) records the original Git tree and all nine campaign paths. Local run output stays ignored. |
| `scripts/` | `tools/`, owner case or `development/` | Separate maintained workflow tooling from case-specific and one-off scripts. The [m18430 ledger](migrations/m18430-audit-tools-2026-11-22.json) covers the report builder. The [CI and contract evidence ledger](migrations/ci-contract-evidence-2026-11-23.json) covers two retired audit-script directories. |
| `src/` | Keep | Keep the installable package and its published import surface stable. |
| `teaching/` | `examples/teaching/` | Keep teaching commands and the companion comparison together. |
| `tests/` | Keep | Preserve test collection and the semantics of fixtures and rules. |
| `tmp/` | No tracked root | Reused paths are recorded by source revision. See the [m26816](migrations/m26816-root-retirement-2026-11-21.json) and [m26843](migrations/m26843-root-report-2026-11-21.json) ledgers for distinct versions of `tmp/report.json`. |
| `tools/` | Keep | Maintain shared tooling and repository policy together. |
| `xval/` | `benchmarks/` and `development/verification/` | Move by campaign. Link formal evidence from the canonical benchmark entry and preserve method, references and historical verdicts. The [preregistration ledger](migrations/xval-preregistrations-2026-11-22.json) records the first two preregistration-only directory retirements. |

The [Ludwig placement ledger](migrations/ludwig-benchmark-placement-2026-11-23.json)
moves the registered case into `benchmarks/cross_validation/` with the same
`xval:ludwig_fe_stress` case ID. Use that directory for registered
cross-validations and keep historical supporting material in development.

The [EOS and FSI report ledger](migrations/eos-fsi-xval-reports-2026-11-23.json)
places the archived Maxwell stability and vivsim fixed-point evidence in their
benchmark directories.

## Second-level placement

| Area | Placement |
| --- | --- |
| Published documentation | Keep stable report and manual paths. Put static report assets in `docs/assets/<topic>/`, using topic names such as `maneuvering-m1984`. |
| Development notes | Put internal work cards and mission scratch notes under `development/audit/` or `development/verification/`. Check incoming links before moving published audit reports. |
| Historical process records | Put council deliberations and CI smoke receipts under `development/archived/` or `development/audit/ci/`. Keep one canonical copy of duplicate mission summaries. |
| Formal benchmarks | Keep stable case IDs and documented runner paths. Use descriptive case directory names for new cases and put curated evidence with the owning case. |
| Reference audits | Use `benchmarks/anchors/evidence/<topic>/` for curated anchor audits. Use ignored `outputs/` for newly generated material. |

The [second-level migration ledger](migrations/second-level-docs-benchmarks-2026-11-22.json)
records the original objects and compatibility checks for the CH6 card,
maneuvering report figures and W31 transcription evidence.

The [survey and Huang audit migration ledger](migrations/survey-and-huang-audit-layout-2026-11-22.json) records the SUBOFF report figures and the archived m17889 delivery snapshot. The [survey index](../surveys/README.md) links to the published surveys.

The [CMB5 migration ledger](migrations/cmb5-hydrorr-stability-2026-11-23.json)
groups the m27100 drop-entry histories, preregistration and tooling under the
drop-entry campaign. Its verdict and histories are discoverable from the
existing 3D drop-entry registry entry alongside the formal comparison inputs.

The [documentation process-directory ledger](migrations/docs-process-directories-2026-11-22.json)
retires `docs/council/`, `docs/history/` and `docs/smoke/`. It records the
unchanged archive objects and proves that the three removed history files are
byte-identical to their retained `docs/mission_summaries/` copies.

The [hotpatch records ledger](migrations/hotpatch-mission-records-2026-11-22.json)
maps `docs/m23777/` and `docs/m24569/` into
`development/audit/hotpatch-absorption/` and records the original checksums.

The [Blasius development-runner ledger](migrations/blasius-dev-runner-retirement-2026-11-22.json)
retires `benchmarks/blasius_flat_plate_dev/` in favour of the canonical
`benchmarks/pending/blasius_flat_plate/` case. The earlier B25 runner remains
byte-identical in the development experiment area and does not acquire a new
scientific verdict through the move.

The [AUDIT-RUNTEST-6 ledger](migrations/runtest-r6-audit-tools-2026-11-23.json)
retires `scripts/audit_runtest_r6/` and places its tools beside the archived
m14496 figure manifest. The published audit retains its existing URL.

The [YAML-to-XML pipeline migration ledger](migrations/yml-xml-pipeline-2026-11-23.json)
retires `scripts/xval_yml_xml_pipeline/`. The runnable driver, analytic
comparison and versioned replay are discoverable from its
[benchmark entry](../../benchmarks/xval_yml_xml_pipeline/README.md). The
historical report keeps its published path.

## Root files and rollout

The [SDAA script ledger](migrations/sdaa-script-directory-retirement-2026-11-24.json)
preserves the hardware scripts alongside their verification records and
identifies the maintained baseline launcher.

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
