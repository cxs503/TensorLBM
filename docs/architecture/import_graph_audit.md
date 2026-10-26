# Static Import Graph Audit

Run from the repository root with Python 3.11+:

```bash
python tools/audit_import_graph.py \
  --root src/tensorlbm=tensorlbm \
  --root app/backend=app.backend \
  --root tests=tests \
  --root app/tests=app.tests \
  --json-out runs/architecture/import-audit.json \
  --markdown-out docs/architecture/import-audit-latest.md
python -m unittest tests.test_audit_import_graph -v
```

The JSON report contains discovered modules, internal import edges, cross-layer edges, import cycles, dynamic import sites, and source parse/read errors. The scanner does not fail merely because cycles or dynamic imports exist: those are findings to review, not automatically defects. It exits nonzero when source files cannot be parsed or read.

## Interpretation rules

- A reported cycle is a strongly connected component in the statically resolved graph; verify whether it is exercised at runtime.
- Dynamic imports with unknown targets are unresolved evidence, not proof of a broken import.
- Cross-layer edges depend on the supplied root prefixes. For the intended boundary, scrutinize any tensorlbm-to-app.backend edge.
- The scanner does not inspect JavaScript/TypeScript imports, string references in docs/configuration, entry-point metadata, or third-party package resolution.
- Do not move or rename modules based only on this report. Compare public API use, runtime imports, tests, CI, docs, and benchmark references first.

## Missing internal targets (2026-10-26 compatibility repair)

Reports now include `unresolved_internal_imports`, with source module, line,
import kind and target, and a matching Markdown section. A target under a supplied
root prefix that is absent from the discovered modules/namespace prefixes is
reported rather than silently disappearing from edge statistics. External imports
are not checked. This is not symbol-level validation (`from pkg import symbol`)
and cannot validate runtime aliases or optional environment-dependent imports.

Use `--fail-on-unresolved` to make missing internal targets fail the command;
reports are still written before returning nonzero. The default remains advisory
because the current baseline contains 11 findings, including frozen legacy
snapshots intentionally loaded under a different package by their tests. These
are not suppressed and should be classified before enabling a mandatory global
strict gate. New relocation work must also pass the real-runtime compatibility
suite, not merely show fewer static edges:

```bash
PYTHONPATH=src pytest -q tests/test_legacy_import_paths.py
```

Physical legacy-path modules alias the canonical module in `sys.modules`, so
private helpers and mutable state are shared, not copied. The turbulence package
lazily delegates the old LES module's attributes (including private helpers) and
preserves its `__all__`. Bare imports remain lazy; first symbol access loads the
numerical implementation. These guarantees are distinct from numerical accuracy.
