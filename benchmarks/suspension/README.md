# Suspension effective-viscosity benchmark (P46W3-N03)

LBM calibration of the effective-viscosity closure for crushed-ice / ice-crystal
suspensions:

```
Einstein (dilute)          eta_r = 1 + [eta] phi,                 [eta] = 2.5
Krieger-Dougherty (dense)  eta_r = (1 - phi/phi_max)**(-[eta] phi_max)
```

The *implementation* lives in [`src/tensorlbm/suspension_viscosity.py`](../../src/tensorlbm/suspension_viscosity.py);
this directory holds the measurement campaign that calibrates and judges it.

## Files

| file | role |
|---|---|
| `P46W3_N03_PREDECLARED_DESIGN.md` | pre-registered design: decision, hypotheses, metrics, baselines, thresholds, confounders, analysis, limits (written **before** any case ran, incl. two amendments made before the first result) |
| `run_lbm_calibration.py` | declares the case table (drag ladder + dilute/dense shear sweeps + negative controls) and runs it through `tensorlbm.ice_suspension_rheology.case_runner` → `artifacts/raw_cases.json` |
| `merge_raw.py` | merges the raw record files of the two execution groups into one set with a provenance block (no value is modified) |
| `probe_core_stress.py` | **exploratory** diagnostic: row-resolved stress, near-wall band vs cell core for the same suspension (explains the residual bias; not a predeclared criterion) |
| `fit_calibration.py` | turns the raw records into `artifacts/analysis.json` (both volume conventions, all acceptance criteria) and the figures in `artifacts/figures/` |
| `artifacts/` | raw records, analysis JSON, figures |
| `../../docs/reports/P46W3-N03_suspension_viscosity.md` | paper-grade report (physics, method, numbers, figures, limits, error verdict) |

## Reproduce

```bash
# 1) measurement (this is the expensive part -> run on the cluster / compute nodes)
python benchmarks/suspension/run_lbm_calibration.py --jobs 12 \
    --out benchmarks/suspension/artifacts/raw_cases.json

# 2) fit + figures + acceptance verdict (seconds)
PYTHONPATH=src python benchmarks/suspension/fit_calibration.py

# 3) tests
PYTHONPATH=src pytest tests/test_suspension_viscosity.py -q
```

## What is measured

Cell: `tensorlbm.ice_suspension_rheology.couette.ShearCell`, D3Q19 moving-wall
Couette, 48x48x57, `nu = 0.1`, `u_wall = 0.02`, 2600 steps (1300 warmup). The
relative viscosity is the **ratio of the same measurement with and without
particles at the same wall speed**,

```
eta_r = (tau_susp / tau_pure) * (gamma_dot_pure / gamma_dot_susp)
```

so a multiplicative bias of the momentum-exchange stress estimator cancels, and
the shear rates come from the *measured* velocity profile (linear fit, `R^2 > 0.99`
is a gate on every case).

Two volume-fraction conventions are carried in parallel, because that is the
scientific question of this campaign:

| convention | definition |
|---|---|
| `mask` | `phi = (solid lattice nodes) / (box nodes)` -- what the discretised sphere occupies |
| `hydrodynamic` | `phi_h = phi_mask (a_h/a_eff)^3`, with `a_h = F/(6 pi mu U K_HS)` inverted from a Stokes-drag run of the **same** discretised sphere in the **same** lattice (Haberman-Sayre wall factor `K_HS`) |

A third convention is carried as a diagnostic: `core_fraction`, i.e.
`phi_mask * nz/(nz - 2*wall_margin)`, because the particles are confined to the
wall-margin-free core and the suspension flow only "sees" that region.

A staircase sphere drags like a slightly smaller smooth sphere, so `a_h < a_eff`
and `phi_h < phi_mask`; the viscosity increment follows `phi_h`, hence a slope
`d eta_r / d phi_mask` that is biased low by exactly `(a_h/a_eff)^3`. The two
conventions share everything else (raw records, aggregation rule, thresholds).

## Acceptance criteria (predeclared, machine evaluated)

| id | criterion | threshold |
|---|---|---|
| a | `\|[eta]/2.5 - 1\|` from the dilute sweep `phi in [0.02, 0.10]`, evaluated with **both** the chord slope and the curvature-consistent one-parameter fit | <= 5 % |
| a' | dilute-fit intercept `\|b - 1\|` | <= 0.01 |
| b | divergence exponent `d ln eta_r / d ln(1-phi/phi_m)` vs `-[eta] phi_m` | <= 10 % |
| c | Krieger-Dougherty fit over the full measured range, `R^2` | >= 0.95 |
| d1 | pure-fluid negative control `eta_eff/nu - 1` | <= 0.5 % |
| d2 | smallest-fraction increment `(eta_r-1)/([eta] phi)` vs 1 | <= 20 % |

Why two estimators: the *chord* slope of a curved closed form is biased high --
over `[0.02, 0.05]` the Krieger-Dougherty chord slope is 2.92 (+16.8 %) and over
`[0.02, 0.10]` it is 3.28 (+31 %), so a perfect LBM would fail a 5 % criterion if
judged by a chord slope on a wide window. The curvature-consistent estimator
(`suspension_viscosity.fit_intrinsic_viscosity`, a weighted regression through
the origin on `ln eta_r = -[eta] phi_max ln(1-phi/phi_max)`) removes that bias
exactly. Both are reported; deviations are recorded as AMENDMENT-4 in the
predeclared design.

## Honest-scope notes

* frozen microstructures (particles advected kinematically, mask not re-meshed),
  monodisperse rigid spheres, `Re_p << 1`, wall-bounded shear cell;
* two microstructure seeds per point -- reported as mean +/- SEM, not averaged away;
* `a_h` and `phi_mask` share the same discretisation, so the correction is
  *consistent*, not independent evidence: the campaign can show that the two
  conventions differ by `(a_h/a_eff)^3` and that one of them reproduces the
  analytic limit, not that discretisation is the only possible mechanism.