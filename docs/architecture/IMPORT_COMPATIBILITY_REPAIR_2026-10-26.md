# Boundary/turbulence extraction compatibility repair

Base revision: `14f200ce02c568b6175ed76e18ca5c3bf8be83fa` (#414).

## Scope

- Restore 31 old flat module paths using physical alias modules. Each replaces
  its `sys.modules` entry with the canonical relocated module, preserving object
  identity, private helpers and shared globals under either import order.
- Restore the old `tensorlbm.turbulence` LES symbols through lazy package
  delegation, preserving `__all__` and private helper access.
- Replace speculative relocated-package probing in the root `__getattr__`
  fallback with ordinary real-module lookup. A package name no longer resolves
  to its same-named inner implementation depending on attribute-access order.
- Repair import sorting in six affected source blocks and format the root export
  table. No solver formulas, acceptance thresholds, quarantine or CI requirements
  are changed. No further domain extraction is included.
- Report missing internal import targets in the architecture audit, with an
  explicit `--fail-on-unresolved` option and CLI regression tests.

## Verification

Local environment: Python 3.11.2, torch 2.2.2+cu121, NumPy 1.26.4,
Ruff 0.17.2, CPU, OMP/MKL one thread. This is not the remote CI environment.

- `tests/test_legacy_import_paths.py`: 33 passed; 30 module relocations in both
  fresh-interpreter import orders, LES symbol/private-helper identity, star
  exports, laziness and alias inventory. The 31st alias (`turbulence.py`) is
  represented by the package delegation rather than a flat shim file.
- Focused existing suite: 270 passed, 25 skipped (before adding the strict CLI
  test): `test_public_api_lazy_exports`, `test_audit_import_graph`,
  `test_turbulence_extensions`, `test_rans_common`,
  `test_rans_regression_equivalence`, `test_navier_slip_bc`,
  `test_wall_shear_profile`, `test_boundary_registry`, `test_sponge_layer`.
- `ruff check src tests examples benchmarks`: passed.
- `ruff format --check src tests examples benchmarks`: passed.

Audit after restoring aliases: 1113 discovered modules, 2987 edges, 11 SCCs,
1433 cross-layer edges, zero parse errors, 11 unresolved import statements.
Remaining findings: `tensorlbm.interactive_cylinder` (one),
`tensorlbm.advanced_collision_d3q19` (three), and seven relative imports in
byte-frozen historical test snapshots. These are visible baseline findings,
not claimed fixed. Alias wrappers add literal dynamic-import sites, so dynamic
site counts must not be compared as if they were unchanged solver dependencies.

## Limits and follow-up

No full library/platform suite, accelerator validation or new benchmark campaign
is claimed. Existing RANS equivalence and boundary physics checks provide bounded
regression evidence, not proof of all physical models. Do not enable global
strict unresolved enforcement until the remaining findings are classified.
Do not infer successful remote CI or protected-branch settings from local checks.
The older structure map is a pre-extraction snapshot, not the current topology.

Final combined rerun (including strict CLI regression): **304 passed, 25 skipped**
in 83 seconds. Both Ruff gates passed (1578 formatted files); all 12 layout
checker tests passed and the repository layout check passed (4725 candidate
paths). The test environment lacks FastAPI; the existing conftest explicitly
reported five ignored platform-dependent library test files. This focused run
is not a full-suite pass.
