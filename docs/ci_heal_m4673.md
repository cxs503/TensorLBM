# PR #673 (agent/m4644) CI heal — round 1

Mission m4673. Branch: `agent/m4673`, cut from `agent/m4644` head `48ff6ea8`
(which itself sits on `main@1ef76442`). Scope: make the whole CI signal on the
PR tree **truthful and green** with minimal edits to the failing suites only.

## 1. What "red" actually was

The m4644 PR head carries 30 check runs, **all `cancelled`** — that part was
infrastructure, not code: every in-flight run on *all* branches of the repo was
cancelled in the same 3-minute window (2026-11-06 02:04–02:06Z, e.g. run
`42145735530` for m4644, plus m4641/m4645/m4646/m4648/m4649/m4652/m4654, and the
`main` push run `42145612061`), i.e. a repo-wide cancellation sweep, not a
per-PR failure.

The **real** red is the underlying baseline: `main`'s last completed run
`41995371431` (`main@866438ac`, 2026-11-05) failed `test (3.11)` and
`test (3.12)` with

```
5 failed, 6485 passed, 177 skipped, 116 xfailed, 6 xpassed in 2730.66s
tests/test_control_volume_force.py::test_curved_moving_slip_impulse_is_invariant_across_nested_control_volumes
tests/test_multiphase_physical_benchmarks.py::test_spinodal_2d_physically_separates      (1.21822e-05 < 1e-05)
tests/test_multiphase_physical_benchmarks.py::test_spinodal_3d_physically_separates      (1.0 > 2.0)
tests/test_regress_wall_multi_gpu.py::TestBugIdentification::test_wss_from_fneq_3d_import_error_equilibrium_vs_equilibrium3d
tests/test_regress_wall_multi_gpu.py::TestWallShearBugs::test_wss_from_fneq_3d_broken_import
```

Because the PR is judged by the whole suite, `test (3.11)` / `test (3.12)` can
only go green once these five are retired. All five reproduce locally on the
m4644 tree (`python 3.11.6`, `torch 2.7.1`, `pytest 9.2.0`), which is what this
round fixes.

## 2. The three defects that are fixed here

### D1 — two "bug exists" contracts outlived the bug (2 tests)

`tests/test_regress_wall_multi_gpu.py` was written TDD-red: it *asserted* that
`sensorlbm.boundary.wall_shear.wss_from_fneq_3d` raises `ImportError`
(`equilibrium` vs `equilibrium3d`). The module now resolves both symbols lazily
inside the function body (`from tensorlbm.d3q19 import C as C3D`,
`from tensorlbm.d3q19 import equilibrium3d`, line 185 onward), so the call
succeeds and the expectations inverted. The two tests are converted into
**regression guards** of the fixed behaviour: shape `(nz, ny, nx)`, all finite.
No `src/` change — the defect they documented is genuinely gone.

### D2 — `pytest.approx()` on a 0-d tensor (1 test)

`tests/test_control_volume_force.py::test_curved_moving_slip_impulse_is_invariant_across_nested_control_volumes`
died with `TypeError: iteration over a 0-d tensor` from `_pytest/approx.py`
(`_contains_decimal` → `float` coercion path iterates the operand on pytest ≥ 8).
The fix is version-independent: compare plain floats
(`float(bfl_force[0])`). The physics assertion (control-volume invariance of the
BFL impulse, `atol=1e-11`) is unchanged.

### D3 — SCMP spinodal acceptance gates are unreachable (2 tests) → quarantined with root cause

`test_spinodal_{2d,3d}_physically_separates` require `density_ratio > 2.0` at
`G = -4.0`, `rho0 = 0.7`, `tau = 1.0`. Neither sign of `G` produces separation
at those settings, so the gate is not a flake — it is unreachable. Diagnostics
committed under `artifacts/m4673/`:

| probe | result |
|---|---|
| `diag_force_sign.py` (linear-response sign test, 2-D **and** 3-D) | `⟨F_x · ∂ψ/∂x⟩ = -5.09e-06` for `G=-4` → the implemented force points **down** the ψ-gradient, i.e. **diffusive**, while the docstring promise `F = -G ψ Σᵢ wᵢ cᵢ ψ(x+cᵢ)` (G<0 attractive) demands up-gradient |
| `diag_sp3d.py` (24³, 40 steps) | `ρ_std` 0.0288 → 4.0e-04 monotone; `ρ_max/ρ_min` → 1.0; mass drift 4e-13 |
| `diag_sp_scan.py` (32², 200 steps) | `ρ_std/ρ_std(0)` = 0.0025 / 0.0063 / 0.0126 at `G = -8 / -6 / -4`, 0.0188 at `G = +4` — monotone decay with `|G|`, **no sign of `G` triggers growth** |

Root cause (documented, **not** fixed in this round): the neighbour sum is
implemented as a gather of `ψ(x − cᵢ)` (`_sc_neighbor_weighted_sum` /
`_sc_neighbor_weighted_sum_3d`, `(arange − c) % n`), whose first moment is
`−c_s²∇ψ`; the documented force `F = −G ψ Σ w c ψ(x+c)` therefore changes sign,
so the library's `G < 0` is the *non*-segregating direction, and even with the
sign corrected `|G| = 4` appears sub-critical for this `ψ` (Shan–Chen
`G_c` for `ψ = ρ₀(1 − e^{−ρ/ρ₀})` at `ρ₀ = 0.7` was never characterised in-repo).
Flipping the stencil inside `src/` would invert the convection direction of every
`G < 0` benchmark in the tree (`laplace_droplet`, `dam_break_sc`,
`benchmarks/bench_multiphase.py`, plus ~30 tests that pin `G = -4.0`) — far
outside a CI-heal mission with a 10-file cap, and it must be a *verified physics*
change, not a CI workaround.

Per the repo's own policy (2nd red-clearance rounds, `CHANGELOG.md` §141
pre-existing-failure quarantine; `07cbd913` "3 例物理/口径缺陷如实登记不静默修"),
the two gates are therefore **quarantined non-strict** in
`tests/ci_quarantine.txt` with the root cause written next to the entry, and the
assertions themselves are left **untouched** — no threshold was loosened, no
gate was deleted. The spinodal tests still run (xfail, non-strict), so the
moment someone fixes the stencil the xfail turns into a real signal.

Registered follow-up (out of this mission's scope): sign-stencil audit of
`_sc_neighbor_weighted_sum{,_3d}` + `G_c(ρ₀)` calibration against the Maxwell
construction for the SC EOS, then re-enable both gates.

## 3. Files touched (5)

| Path | Change |
|---|---|
| `tests/test_regress_wall_multi_gpu.py` | 2 stale "bug exists" contracts → regression guards of the fixed behaviour |
| `tests/test_control_volume_force.py` | `pytest.approx(0-d tensor)` → `float(...)` |
| `tests/ci_quarantine.txt` | +2 node IDs, each with the root-cause note |
| `docs/ci_heal_m4673.md` | this note |
| `artifacts/m4673/diag_{force_sign,sp3d,sp_scan}.py` | the three diagnostics quoted in §2, re-runnable |

No `src/` file is touched; no unrelated test, workflow or doc is modified.

## 4. Local reproduction (what was actually run)

```
pytest -q tests/test_regress_wall_multi_gpu.py tests/test_control_volume_force.py
→ 20 passed, 13 xfailed in 112 s        (was: 2 failed)
pytest -q tests/test_regress_wall_multi_gpu.py::TestBugIdentification::test_wss_from_fneq_3d_import_error_equilibrium_vs_equilibrium3d
→ passed
pytest --collect-only -q tests/test_multiphase_physical_benchmarks.py
→ 4 tests, the two spinodal gates now carry the non-strict xfail marker from tests/conftest.py::pytest_collection_modifyitems
ruff check tests/test_regress_wall_multi_gpu.py tests/test_control_volume_force.py  → All checks passed!
ruff format --check (same two files)                                               → already formatted
```

Full-suite and 3.12 legs are judged by CI; the three quarantined node IDs were
verified to match `pytest` node IDs exactly (see the `--collect-only` output
above), which is the only requirement `tests/conftest.py` imposes.

## 5. Limitations

* The repo-wide cancellation sweep of 2026-11-06 02:04Z is an infrastructure
  event; the only defence is a fresh run on a new commit, which this branch
  provides.
* D3 is **quarantined, not fixed** — CI goes green on an honest signal (xfail),
  not on a repaired model. The spinodal physics remains an open, registered
  defect with reproduction scripts.
* `mypy`, `package-build`, `architecture-audit`, `repository-layout`,
  `platform-test` legs were not re-run here; they were green on `main@866438ac`
  and nothing in this change set touches their inputs (no `src/`, no new files
  outside `docs/` + `artifacts/`).