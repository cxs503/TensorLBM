# Compressible-shock family verification (campaign P19-N04)

Systematic verification surface for the compressible (Euler) branch of TensorLBM
in the **Mach 2–10** range.  It extends the single-point Sod / normal-shock
checks of `benchmarks/verified/sod_shock_tube` into a *continuous parameter
surface* with three families of flow physics:

| sub-suite | physics | cases |
|---|---|---|
| `normal_shock` | planar shock, Rankine–Hugoniot jump | Mach 2/3/5/8/10 × 400/800/1600 cells = 15 (+15 prescribed-jump) |
| `double_mach_reflection` | Mach stem / triple point / reflected shock / slip line / vortex roll-up | Mach 10, 30° incidence, 240×60 / 360×90 / 480×120 |
| `forward_facing_step` | detached bow shock, expansion fan, recompression shock | Mach 3, h = H/5, 150×50 / 225×75 / 300×100 |

Reference configurations: Woodward & Colella (1984), *J. Comput. Phys.* **54**, 115
(problems 1 and 2).

## What is verified, and against what

| quantity | reference | how it is measured |
|---|---|---|
| ρ₂/ρ₁, p₂/p₁, T₂/T₁ | exact Rankine–Hugoniot (γ = 1.4), shock Mach number **as a solver output** (driver state inverted from the shock-tube relations) | plateau average 3–20 cells behind the front |
| incident-shock angle (DMR) | exact 60° to the wall (prescribed initial data) | linear fit of the front above the triple point |
| Mach-stem height (DMR) | Richardson extrapolation of the three grids | intersection of the fitted Mach-stem and incident branches (sub-cell) |
| slip-line angle (DMR) | two independent detectors: density-jump/pressure-continuity ray scan **and** tangential-velocity (vortex-sheet) ray scan | see `family.slip_line_angle`, `family.slip_line_angle_velocity` |
| vortex core (DMR) | Richardson extrapolation, normalised by the channel height | pressure minimum with a quadratic sub-cell refinement |
| bow-shock / recompression-shock angle (FFS) | Richardson extrapolation (no closed-form inviscid solution exists) + oblique-shock polar inversion as an independent consistency check | linear fit of the leading front and of the internal `|∇p|` ridge |
| grid convergence | Richardson with observed order (`analyze_family.richardson`) | three grids, refinement ratio 1.5 (2 for the 1-D scan) |

The one place where classical theory is *not* usable is documented rather than
hidden: for Mach 10 / 30° the von Neumann three-shock closure has no
non-degenerate solution (`family.three_shock_closure`: the flow behind the
incident shock turns by 43.1° while a M₂ = 1.34 shock can deflect at most 7.7°),
so the slip-line angle is gated on the two-detector agreement instead.

## Current verified state (artifact `gates_verified.json`)

11/12 gates pass (`all_pass = False`).

| gate | quantity | measured | threshold | verdict |
|---|---|---|---|---|
| NS-M2..NS-M10 | ρ₂/ρ₁, p₂/p₁, T₂/T₁ vs exact R-H (production grid 1600) | ≤ 0.21% | 1% | PASS (5/5) |
| DMR-01 | incident-shock angle vs exact 60° | 0.29° | 1° | PASS |
| DMR-02 | Mach-stem height vs Richardson | 0.41% | 5% | PASS |
| DMR-03 | slip-line angle, two independent detectors | 1.75° | 3° | PASS |
| DMR-04 | vortex core vs Richardson (normalised by H) | 0.79% H | 5% | PASS |
| FFS-01 | detached bow-shock foot angle vs Richardson | 0.57° | 2° | PASS |
| FFS-02 | recompression-shock angle vs Richardson | 1.46° | 1° | **FAIL (honest)** |
| FFS-03 | bow-shock stand-off distance vs Richardson | 0.34% | 5% | PASS |

**Why FFS-02 fails (not hidden, not overwritten).** The internal recompression
shock is curved and sits in an unsteady separated flow, so its inclination is
not a clean grid-convergent scalar: the linear ridge fit gives
50.7° → 56.2° →
58.3° over the three grids (observed order
2.26, Richardson 59.79°,
fine/medium spread 2.18°), and the independent local-tangent
definition at y ≈ 0.41 gives
48.5° /
46.1° /
41.7°.
Both definitions miss the 1° criterion, so this is a resolution/physics limit at
the achieved grids (150×50 … 300×100), not a fitting-window artefact and not a
reason to relax the criterion.

Reference values of the finest grids (for quick orientation):

* DMR 480×120: triple point (2.736, 0.402),
  Mach-stem height 0.3974 H, slip line
  210.25° (density detector) vs
  212.00° (velocity detector),
  vortex core (2.609, 0.079), stem-foot speed
  13.19 c₁;
* FFS 300×100: bow-shock foot angle
  70.52°, upper-branch angle
  69.31°, stand-off 0.2590,
  post-shock density change RMS 2.53% between
  t = 3.8 and t = 4.0 (angle drift 0.10°).

## Layout

```
run_family.py        production runner (one process per grid, `--suite ns|--case dmr|ffs|--assemble`)
analyze_family.py    metrics, Richardson extrapolation, gates, figures
make_report.py       paper-style report builder (markdown + inline figures)
result.json          every measured number per case and per grid
gates_verified.json  gate table: criterion / reference / measured / threshold / pass
figures/*.png        density fields with detected structures, error and convergence charts
out/                 raw fields (npz, float32) + normal_shock.json produced by the runner
```

## Reproduce

```bash
python run_family.py --suite ns
python run_family.py --case dmr --grid 240x60     # and 360x90, 480x120 (parallel)
python run_family.py --case ffs --grid 150x50     # and 225x75, 300x100 (parallel)
python run_family.py --assemble                   # -> result.json, gates_verified.json, figures/
python -m pytest tests/test_compressible_shock_family_verified.py -q
```

Wall time on one 128-core host (single-threaded numpy solver): ≈ 60 s for the
1-D scan, 87 s / 327 s / 747 s for the three DMR grids, 174 s / 751 s / ≈1800 s
for the three FFS grids.

## Numerics

* compressible Euler, calorically perfect gas (γ = 1.4), no viscosity, no
  turbulence model, no real-gas effects;
* finite volume: MUSCL + minmod + SSP-RK2, CFL 0.4, HLLC flux
  (`compressible_shock.flux`, whose kinetic/KFVS closed form is verified
  separately against a velocity-space quadrature);
* Zhang–Shu-type positivity limiter; `n_admissibility_fixes = 0` for every 2-D
  case of this suite (no artificial fallback was triggered);
* the compressible branch is *not* the DdQq lattice-Boltzmann core: the link to
  the LBM lineage is the Boltzmann-level flux evaluation, documented per
  function in `src/tensorlbm/compressible_shock/flux.py`.

## Limitations (read before quoting a number)

1. **No analytic solution exists for the forward-facing step.** Its two angle
   gates are referenced to the grid-extrapolated value, i.e. they certify
   *convergence*, not agreement with a closed-form solution. The
   oblique-shock-polar residual at the recompression shock is reported as an
   independent check and stays above 3° at the coarse grids (shock transition
   smeared over 2–3 cells).
2. **The FFS separated region is unsteady.** Between t = 3.8 and t = 4.0 the
   field changes by a few percent RMS (concentrated in the separation vortex);
   the measured shock angles drift by O(0.1–1)°, so the angle gates are
   meaningful, but this is not a steady-state solution.
3. **Slip-line pressure continuity is only approximate numerically** (≈2–10% of
   the local pressure at the finest grid): the contact is smeared over 2–3
   cells. The corresponding check therefore uses a density-dominated
   criterion rather than an exact equality.
4. **Mach-stem post-shock pressure** deviates from the exact normal-shock value
   by ≈10% because the sampling window sits inside the low-pressure vortex
   region; it is reported as a diagnostic, not as a gate.
5. Data are 2-D inviscid; three-dimensional and side-wall effects, real-gas
   chemistry and viscous/turbulent physics are out of scope.