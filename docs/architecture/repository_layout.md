# Repository layout and artifact policy

This is the file-placement contract for contributors and coding agents. It
implements the low-risk part of the [refactoring plan](REPOSITORY_REFACTORING_PLAN.md),
not a new solver architecture. No source relocation is authorized by this document.

## Where new work belongs

| Area | Responsibility | Do not put here |
|---|---|---|
| Repository root | Project metadata, installation configuration, main README and contribution guide | New one-off Python scripts, figures, run logs or reports |
| `src/tensorlbm/` | Installable numerical library and supported library extensions | Web routers, frontend lifecycle or one-off campaign scripts |
| `app/`, `frontend-vue/` | Application backend, application tests, browser UI | Numerical kernels copied from the library |
| `tests/` | Automated library tests and small, documented reference fixtures | Unmarked long simulations or throughput claims on unsuitable hardware |
| `examples/` | Maintained runnable demonstrations of supported APIs | Unreviewed research campaigns |
| `benchmarks/` | Named benchmark runners, configuration, references, acceptance criteria and curated results | Unclassified dumps or verification claims based only on exit code |
| `experiments/<topic>/` | Exploratory runners with environment, command and limitations documented | Claims that an experiment is already a verified benchmark |
| `experiments/archived/` | Historically meaningful, explicitly unmaintained source | Unexplained deletion of the source of a published result |
| `scripts/` | Maintained simulation/data/plot/developer workflow entry points; may depend on TensorLBM | One-off scripts with personal machine paths |
| `tools/` | Repository maintenance and static governance tools; prefer Python stdlib | Simulation runners or solver implementations |
| `docs/` | User/developer documentation, architecture, reports and selected illustration assets | Large raw trajectories or generated duplicate document formats without a source policy |
| `evidence/<topic>/` | Small reproducible defect/audit bundles: README, reproducer, environment, result and short logs | Unbounded run dumps; evidence is not independent acceptance |
| `outputs/` | New local simulation outputs (ignored) | Files required to reproduce a clean-checkout test |
| `artifacts/runs/` | Local diagnostic/CI working outputs (ignored) | The sole copy of accepted, externally referenced evidence |

Existing `runs/`, `docs/evidence/`, `agent_work/`, `agent_memory/`, `demos/`,
`teaching/` and other historical paths are not being deleted or renamed. Do not
create additional parallel storage conventions. Keep benchmark evidence tied
to its stable benchmark ID; keep agent-private scratch output ignored. Select
one canonical location for each result and link to it rather than copying it.

## Source, evidence and raw output are different

Keep source/configuration, compact reference fixtures and acceptance definitions
in Git. Curated results and figures may be tracked when they have a clear consumer
and provenance. Raw fields, model weights and large trajectories should normally
be published as versioned release/data artifacts, not ordinary Git blobs.

Before externalizing an artifact, record:

- stable artifact/benchmark ID and source revision;
- generating command, parameters, environment, device and precision;
- immutable version/location, byte count and SHA-256;
- reference data and acceptance criteria, plus the scope of the actual result;
- retrieval instructions and a verified download, and all affected consumers.

Do not remove the in-repository copy until retrieval and consumers have been
validated. No history rewrite or repository split is part of this phase.
A runtime regression fixture may be an explicit reviewed exception; an ignored
file that tests silently rely on is not a valid fixture.

## Automated gate: prevent new accumulation

From any working directory:

```bash
python /path/to/TensorLBM/tools/check_repository_layout.py
```

From the repository root (no PyTorch or pytest required):

```bash
python tests/test_repository_layout.py -v
python tools/check_repository_layout.py
```

The gate checks Git-tracked paths plus untracked, non-ignored files. This catches
local additions before staging, and catches force-added ignored outputs in CI.
Ignored, untracked local runs are deliberately excluded. It does not modify files,
update Git's index, download data, or require the baseline commit's history.

Rules:

1. No new root files outside the explicit root metadata allowlist.
2. No new tracked files in `runs/`, `runs_prop/`, `outputs/`, `artifacts/runs/`,
   root `build/` or `dist/`, frontend `dist/`, or recognized cache/dependency directories.
3. No new model-weight files (`.pt`, `.pth`, `.ckpt`, `.onnx`, `.safetensors`),
   unless an exact-path exception is deliberately reviewed.
4. New files must be at most **5 MiB**. This is a repository maintenance threshold,
   not a scientific acceptance criterion. The two existing oversized files have
   individual byte ceilings, so their historical exceptions do not permit growth.

`repository_layout_policy.json` records the initial inventory at commit
`ad3d19ae6e5f55f02ac691f6719d4a5ae68f4cdc`: 11 canonical root files,
85 legacy root files, 392 generated-directory files, one model and two large files.
After the 2026-10-26 report/assets batch, seven root exceptions were retired;
78 legacy root paths remain. The migration ledger records the original hashes:
[report/assets batch](migrations/report-assets-2026-10-26.json).
Categories may overlap. Existing debt remains visible and usable, but exceptions
are exact paths, not directory exemptions. Historical exceptions are not evidence
that a file is correct, maintained or safe to remove.

Policy exceptions require a PR explanation: consumer, reason, owner/reviewer,
size/provenance where relevant, and retirement plan. Do not regenerate the baseline
to make a failed check green. Remove obsolete exceptions in the same PR that
migrates their files. The gate allows deletions without requiring placeholder files.

Limitations: this first gate does not detect every generated file in arbitrary
folders, enforce cumulative repository size, validate links or benchmark manifests,
prove import boundaries, or validate physics. CI is only authoritative for the
committed tree; local runs also see non-ignored untracked files. The independent
`repository-layout` CI job runs the checker and its dependency-free tests.

## Small-batch migration procedure

Use [migration_manifest.csv](migration_manifest.csv), not filename-prefix deletion.
Each proposed move must record old/new paths, purpose, reference scan,
compatibility, risk, owner and verification. `unassigned` means no reviewer has
accepted ownership, not that the implementing agent can approve its own change.

1. Search exact paths, basenames, imports, relative links, notebook/config strings,
   CI commands and known external consumers. No text matches does not prove no users.
2. Check relative output locations and hardcoded environment paths. Record source
   and data hashes; retain original benchmark status, including failures.
3. Move one coherent bundle with `git mv`; repair links and CLI/import compatibility
   as needed. Keep algorithm changes in a separate patch.
4. Run affected commands/tests and compare their collected tests and numerical
   results to the baseline. Update the manifest and remove obsolete exceptions.
5. Obtain independent review before calling the migration accepted.

Do not use `clean_repo.sh --apply` as a substitute for this procedure. Its glob
matching and index changes cannot establish provenance or compatibility.

## Next batches, in order

- **A: reports/assets.** Couette and Washburn reports have been relocated, with five
  byte-identical Washburn images deduplicated (implementation complete, independent
  review pending). Root report/image URLs are retired without redirects; update
  external links to the new paths or use the ledger's source-revision permalink.
  The Blasius report/three-image bundle remains pending: it references external
  `/nfs/...` data, so preserve those gaps rather than claiming reproducibility.
  `python tests/test_repository_report_migration.py -v` checks archived report-body
  hashes after undoing link rewrites/removing the migration note, asset/evidence
  hashes, local links, manifest statuses and retirement of root exceptions.
  It needs neither Git history nor a numerical runtime. Historical text is frozen
  for traceability; future scientific corrections should be explicit errata, not
  silent edits to these original reports.
- **B: standalone debug scripts.** Classify maintained vs archived, replace personal
  paths only when execution can be validated, and retain historical invocation notes.
- **C: evidence lifecycle.** Introduce a benchmark manifest/schema and artifact
  retrieval checks; review model consumers before externalizing weights.
- **D: tests/API/modules.** Map test categories, quarantine IDs and public import
  consumers before moving tests or flattening the legacy API into subpackages.
  No bulk source moves, tolerance changes or benchmark promotions in a layout patch.

### Updated inventory after PR #412 and compatibility repair

The 78 legacy root paths above describe the earlier report/assets batch, not the
current inventory. At `14f200c`, there are 20 tracked root files: 17 canonical
metadata files and three existing legacy scripts. The policy lists four legacy
root paths, one of which (`agent_memory_context_update.md`) no longer exists.
There remain 392 tracked historical `runs/` files and one model-weight exception.
The boundary/turbulence compatibility repair adds only source alias modules,
tests and documentation, not generated results. See the repair report for checks
and limitations.
