# M4671 — CI heal for PR #671 (branch agent/m4648)

Date: 2026-11-06 (UTC). Mission m4671, round 1.

## Situation at pickup

- PR #671 (head `agent/m4648` @ `4d597055`, base `main` @ `1ef76442`) showed
  **all 7 CI jobs `cancelled`** at the same instant (02:04:39Z):
  platform-test, test (3.11), test (3.12), architecture-audit,
  repository-layout, lint, package-build.
- A simultaneous cancellation of *every* job (including the fast,
  dependency-free ones) is a workflow-level cancellation, not a test
  failure: no job-level timeout (test jobs allow 180 min; pytest-timeout
  is 600 s) could produce that pattern.
- The PR was closed at 02:20:48Z with the standing ruling
  「修复 ≥2 轮未绿 → 关闭重开，原分支留档」. Branch `agent/m4648` retained.

## Local reproduction (worktree at PR head, then after sync)

Environment: Python 3.11.6, torch 2.7.1, pytest 9.2.0, `PYTHONPATH=src`.

| Gate (mirrors .github/workflows/ci.yml) | Result |
|---|---|
| `ruff check src tests examples benchmarks` | PASS |
| `ruff format --check src tests examples benchmarks` | PASS (1453 files) |
| `mypy --follow-imports=silent d2q9 d3q19 solver utils` | PASS |
| `scripts/check_platform_docs_consistency.py` | PASS |
| `tests/test_repository_layout.py` (12) + `test_repository_report_migration.py` (6) + `tools/check_repository_layout.py` | PASS |
| `tests/test_audit_import_graph.py` (5) + `tools/audit_import_graph.py` | PASS |
| `python -m build` (sdist + wheel) | PASS (local isolated-env build needs `--no-isolation`: the local loongnix pip mirror lacks `wheel`; GitHub CI uses real PyPI) |
| PR-touched tests: `test_control_volume_force.py` + `test_regress_wall_multi_gpu.py` | PASS — 20 passed, 13 xfailed (x2 runs) |

The PR diff touches only `tests/test_control_volume_force.py` (0-d tensor
`pytest.approx` fix), `tests/test_regress_wall_multi_gpu.py` (two
ImportError-expectation tests rewritten to assert the fixed behaviour
against the analytic D3Q19 stress-tensor value) and this docs file.
Both touched test files pass, so the PR diff introduces no new failures.

## Fix

The branch was one commit behind `main` (`4d597055` parented on `866438ac`,
while `main` had advanced to `1ef76442` via merged PR #667). Merged
`origin/main` into the branch (merge commit, no file changes) so the
re-opened PR runs CI on the current main and merges cleanly:

```
5f81e527 Merge remote-tracking branch 'origin/main' into agent/m4671
1ef76442 m4629: matured mission deliverables (#667)
4d597055 m4648: matured mission deliverables
```

Pushed back to the same branch `agent/m4648` (fast-forward).

## Caveats

- Full-suite local runs on this shared Loongson box are heavily
  contended (dozens of concurrent agent pytest processes); a complete
  `-m "not slow"` pass could not be captured within the mission budget.
  The PR-relevant subset and every static gate are fully verified above.
- The original cancellation cause (external governor/bot action) is not
  recoverable from the API (job logs of cancelled runs were purged).
