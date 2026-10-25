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
