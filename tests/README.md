# Library tests

The default pytest path is `tests/`; platform tests run separately in `app/tests/`.
The [development workflow](../docs/development_workflow.md) lists exact commands
and environment limitations. Start with a focused module for a small change:

```bash
pytest -q tests/test_navier_slip_bc.py
pytest -q -m "not slow" --timeout=600
```

Classify new tests by their claim (unit, integration, physics, regression,
performance) and state hardware/precision requirements. Use registered markers;
`slow` is currently the shared configured timing marker. This guidance does not
claim that all legacy tests have been classified or that GPU markers already exist.

Keep small, necessary reference fixtures with documented provenance. Do not fetch
large data implicitly or depend on ignored local run outputs. Frozen legacy source
snapshots under `_legacy_snapshots/` are intentional regression references, not
cleanup candidates.

`ci_quarantine.txt` names exact pytest node IDs; `conftest.py` applies non-strict
xfail markers. A path/function move changes those IDs and must update the registry
and compare collection. Do not add quarantine entries merely to pass a relocation.
Report collection exclusions, skips, xfails and XPASS alongside passes/failures.

Dependency-free repository checks can run directly:

```bash
python tests/test_repository_layout.py -v
python tests/test_repository_report_migration.py -v
python tests/test_audit_import_graph.py
```

`test_repository_report_migration.py` protects the first report/asset relocation:
original report bodies (apart from documented link changes and a migration note),
canonical image hashes and archived numerical evidence must remain intact. These
are provenance checks, not an endorsement of the reports' physical conclusions.
