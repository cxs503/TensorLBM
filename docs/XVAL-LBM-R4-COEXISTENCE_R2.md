# XVAL-LBM-R4-COEXISTENCE round 2 — coexistence calibration + free-energy selection (re-verified)

The [relocated verification tools](../development/verification/mcmp-coexistence/r4/) keep the original artifact directory and command options.

> Anchor `XVAL-LBM-R4-COEXISTENCE-CALIBRATION-AND-FREE-ENER` · repo TensorLBM · mission m21542
> Full report (programmatically injected numbers + figures): `artifacts/xval_lbm_r4_coexistence_r2/REPORT_R2.md`
> Driver: `development/verification/mcmp-coexistence/r4/r2/run_xval_r2.py` (stages `M` / `S` / `C` / `merge`)
> Regression: `tests/test_xval_r4_coexistence_calibration_r2.py` (6 tests)

## What round 1 left open, and what round 2 found

Round 1 (`docs/XVAL-LBM-R4-COEXISTENCE.md`) shipped an honest FAIL on both arms. Both were
re-run here against the same Maxwell reference and the same 15 % gate.

**Route S — coexistence calibration (was: 3/4 runs NaN, 1 homogenised → no data).**
Round 2 seeds the periodic flat interface with the *Maxwell binodal of the same EOS* and uses the
**library** collision operator (`flat_interface_coexistence` → `multiphase.collide_sc_single_component`)
with the density floor applied inside ψ, instead of the round-1 hand-rolled velocity shift that
divided `F` by `2ρ` on an unfloored density.

| force scheme | worst \|Δρ_l\|,\|Δρ_v\| vs Maxwell (ny=64/96, G=4.5…5.5) | verdict |
|---|---|---|
| `velocity_shift` | 3.53 % | PASS |
| `edm` | 3.53 % | PASS |
| `guo` | 44.75 % (ρ_v at G≥5) | **FAIL** |

The force scheme is a free parameter of the calibration that no earlier run swept: `guo`'s source
term changes the *effective* EOS, so the same G yields a different vapour density. A calibrated
(ρ_l, ρ_v) pair is therefore only valid together with the scheme that produced it.
Flat-interface invariants hold: mass drift ~1e-12, |p_l−p_v|/p_sat ≲1e-6, and the spurious-current
envelope max|u|(y) is stationary by ~4 k steps (it peaks at the interfaces) — the plateaux are static,
not still relaxing.

**Route C — free-energy selection (was: σ = 4.4 % at κ=2, 17.8 % at κ=0.5 → FAIL).**
The round-1 κ=0.5 FAIL reproduces exactly at its 6000-step budget (15.37 % / 17.82 % at ny=96/128),
but the *converged* relaxation (σ-drift < 5e-4 **and** std(μ)/A < 2e-3) gives

| κ | ξ/h | σ error (converged) | replica at 6000 steps | two-route spread |
|---|---|---|---|---|
| 0.25 | 0.71 | 19.22 % (FAIL) | — | < 3 % |
| 0.50 | 1.00 | 11.90 % (PASS) | 15.37 % / 17.82 % | < 4 % |
| 1.00 | 1.41 | 8.28 % (PASS) | — | < 3 % |
| 2.00 | 2.00 | 4.11 % (PASS) | 4.41 % / 4.50 % | < 1 % |

So the round-1 statement that κ=0.5 is "outside the 15 % gate, a thin-interface limit of the 5-point
Laplacian" is an **overstatement**: most of the deviation came from an unfinished Cahn–Hilliard
relaxation (μ was not uniform by 6000 steps). The genuine validity boundary is ξ/h ≲ 0.7 (κ=0.25),
which still exceeds the gate after convergence. The two independent σ routes (Korteweg pressure
integral vs thermodynamic excess free energy) agree to < 4 % everywhere, so the residual deviation
from the analytic value is lattice resolution, not an implementation defect.

## Selection rules this round establishes

1. Coexistence calibration must state the force scheme; `velocity_shift` and `edm` may be compared
   with Maxwell, `guo` needs its own EOS correction first (this round does not provide one).
2. Free-energy interface measurements are only comparable to the continuum analytic σ after a
   convergence test; report the reached step count and std(μ)/A alongside σ.
3. Interface resolution ξ/h ≳ 1 is required for the 15 % gate; below that the lattice wins over the
   functional.

## Library changes

* `src/tensorlbm/sc_eos_coexistence.py` — `flat_interface_coexistence()` (+ `FlatInterfaceResult`):
  Maxwell-seeded periodic flat interface on the library collision operator, with mass-drift,
  mechanical-equilibrium and spurious-current diagnostics; refuses to seed when the EOS has no
  coexistence (G = 4.0).
* `src/tensorlbm/free_energy_stress.py` — `relax_ch()` (convergence-controlled CH gradient flow),
  `surface_tension_pressure_integral()`, `surface_tension_energy_integral()`,
  `measure_planar_interface()` (single-interface window fit; the round-1 fit superposed two
  interfaces).

## Limits / not covered

2-D D2Q9, τ = 1.0, zero body force, planar interface only. Curved-interface (Laplace) calibration and
3-D are out of scope; a `guo`-scheme EOS correction is an open item.
