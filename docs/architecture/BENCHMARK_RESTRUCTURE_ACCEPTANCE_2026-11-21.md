# Benchmark restructuring acceptance report, 2026-11-21

**Overall verdict: incomplete.** The checkpoint on main is `657473558e72cf19be86c8b17558356fe1dd1d0b`. Remote CI, remaining solver imports and case attribution are still open.

| PR | Disposition | Evidence |
| --- | --- | --- |
| [2803](https://github.com/cxs503/TensorLBM/pull/2803) | Merged at `1da5c586` | Structure audit and proposed migration order |
| [2804](https://github.com/cxs503/TensorLBM/pull/2804) | Merged at `2733d687` | Initial checker and tests; no workflow changes in the diff |
| [2805](https://github.com/cxs503/TensorLBM/pull/2805) | Initial commit merged at `543d63a0` by the repository merge process | Repair of relative manual links, integration into layout CI and six byte-identical audit record moves |
| [2806](https://github.com/cxs503/TensorLBM/pull/2806) | Merged | Registry, historical package guard, fresh Couette computation, and follow-up test fixes |
| [2807](https://github.com/cxs503/TensorLBM/pull/2807) | Merged | Published manual downloads and historical asset lookup |
| [2808](https://github.com/cxs503/TensorLBM/pull/2808) | Merged | Mission audit record migration and CI guard |
| [2809](https://github.com/cxs503/TensorLBM/pull/2809) | Merged | Validation matrix case, reference and evidence links |
| [2811](https://github.com/cxs503/TensorLBM/pull/2811) | Merged | Canonical wheel for cavity and square-duct runners, plus CPU field comparisons |
| [2812](https://github.com/cxs503/TensorLBM/pull/2812) | Merged | 80-file workspace archive with identical Git tree and recursive layout guard |
| [2813](https://github.com/cxs503/TensorLBM/pull/2813) | Merged | Registered Ludwig stress runner and dated comparison on six planar and two curved interfaces |

The three latest merged changes produced the same Git tree as the local integration run
(`bd17a9632e62b695af08f09d33054287c8304761`), with 71 passing tests, 2 optional
grid fixtures skipped and 327 passing subtests. GitHub returned zero check runs on
all ten listed PR heads. On the main checkpoint it also returned zero workflow runs
and zero commit statuses. The latest available main workflow was created on
2026-11-06. Repository Actions settings and runner availability were unverified.

The subsequent [standalone campaign inventory and replay](migrations/standalone-campaigns-2026-11-21.json)
links five cases and 229 previously unassigned files. As of this candidate, the registry
indexes all 3,557 tracked benchmark files, with 685 having no declared case link or
registered parent directory. The direct Ritter finite-volume campaign was rerun at
400, 800 and 1,600 cells and retains a different case ID from the validation-matrix CI case.
The cavity and square-duct packaging replays retain their numerical failures against their
case thresholds. Their input/output equivalence is recorded in the
[packaging ledger](migrations/canonical-cavity-and-duct-2026-11-21.json).

## Migration and compatibility

- [Drop-entry mapping](migrations/drop-entry-2026-11-21.json) enumerates the six
  old and new paths with their original Git blobs, byte sizes and base revision.
  Direct byte comparisons passed. Historical paths embedded in the drafts
  were preserved. Old branch-tip URLs for these six notes now require the
  destination path or the pinned historical revision. External users of those
  URLs could not be enumerated.
- Tutorial pages, PDFs and Word files retain their tracked paths and contents.
  Only the 35-row index changed. It now links relative to its actual directory.
- [Nested source comparison](migrations/nested-package-inventory-2026-11-21.json)
  records 142 historical files: 21 identical by relative path and blob, 34
  divergent and 87 nested-only. The root wheel built successfully and its
  entry point matches `src/tensorlbm/__init__.py`. Root packaging and CI
  declarations do not reference the nested package. Historical scripts and
  audits mention both paths, and external consumers were not verified. The
  proposed CI guard pins the nested source and packaging blobs for review.
- The registry discovers cases in the published catalogue and immediate
  lifecycle/case-pool directories. It exposes 175 records and preserves
  multi-variant tutorial links. Broader standalone campaigns, deep evidence
  directories and missing configuration/reference associations still need
  curation. Archive bucket names do not confer current acceptance.

## Verification

| Check | Local result |
| --- | --- |
| Manual checker, layout tests, report migration tests, registry tests | 3 + 14 + 7 + 2 passed |
| Catalogue and layout commands | 35 cases, 33 PDF and 33 Word links resolved; layout passed |
| Root wheel and tracked-source compatibility | Wheel built; source/package trees and manual download assets compared against `2733d687` |
| Selected numerical and contract tests | 109 passed, 1 failed; see convergence failure below |
| Remote Actions | Zero checks returned for the merged #2805 revision and initial #2806 candidate. Full remote CI is unverified |

The [2D Couette replay](../../benchmarks/verified/couette_2d/result_reproduction_2026-11-21.json)
records the solver tree, runner blob, CPU environment, command, raw profiles
and thresholds. For H=40 and H=80 the maximum masked relative errors were
0.0179632% and 0.0143644%. Both grids reached steady state at 10,000 steps.
Clean main and the migration candidate produced identical numerical output.
Elapsed time was excluded from equality. This replay passes the published
3% and decreasing-error gate for this case.

The zero-gradient-outlet Kovasznay test failed its H=64→128 monotonicity
criterion on the local CPU runtime. The u-velocity L2 relative error rose
from 0.0017053 to 0.0022321. Clean main reproduced the same error values.
The threshold, runner and numerical implementation
were preserved. This run cannot support a new Kovasznay acceptance verdict.

## Remaining acceptance gaps

- Obtain a remote Actions run for the exact candidate and verify the required
  jobs before merging the remaining batch.
- Investigate Kovasznay convergence using the original run configuration and
  environment before attributing the failure to a cause or revising claims.
- Inventory consumers of divergent or nested-only files before retirement of the
  nested package. Preserve Git provenance and public import/CLI compatibility.
- Audit individual consumers in `experiments/`, `xval/`, `benchmarks/xval/`,
  `audit/` and `audits/` before consolidating additional assets. Each move
  requires a source/destination and blob map, link and execution checks, and
  case-specific numerical evidence where a runner is affected.
