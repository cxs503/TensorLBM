# Solver-in-the-loop closure calibration (B3)

*data → closure → solver*: identify a Reynolds-dependent Smagorinsky closure
from drag observations by differentiating through bounded LBM rollouts, and
evaluate it at held-out Reynolds numbers. This productises the manual loop
sketched by `examples/solver_in_the_loop.py` (A6) on top of the differentiable
bounded path (A6+ velocity inlet / zero-gradient outlet), module
`tensorlbm.autograd_calib`, example `examples/closure_calibration.py`.

## API

| Object | Role |
|---|---|
| `BoxCase` | bounded calibration domain: sphere at `cx=0.3·nx` in a uniform inflow, house relation `tau = 0.5 + 3·u_in·2r/Re` |
| `DragTarget(re, cd, weight)` | one drag observation |
| `bounded_drag(box, re, cs=…)` | one differentiable bounded rollout; windowed momentum-exchange C_D over `[window_start, steps)` |
| `synthetic_targets(box, re_values, closure)` | verification-mode observations with known ground truth |
| `calibrate(targets, box, kind=…)` | Adam through the rollouts; `kind="scalar"` (one C_s) or `"power"` (`C_s(Re)=c0·(Re/re_ref)^b`, log-space c0) |
| `evaluate(result, targets, box)` | per-Re predicted vs observed C_D |
| `cs_power(c0, b, re_ref)` | closure factory (truth or initial guess) |

Exports live in `tensorlbm` and `tensorlbm.api`.

## Identifiability (measured, 2026-08-22)

The closure is only calibratable from drag in a windowed-τ sense — the C_D
response to a 12× C_s sweep (0.02 → 0.25), fp64, 300 steps, window ≥ 200:

| grid | u_in | Re | τ | C_D spread |
|---|---|---|---|---|
| (6,8,18) r2 | 0.20 | 30 | 0.580 | **13.6%** |
| (6,8,18) r2 | 0.20 | 48 | 0.550 | **17.3%** |
| (10,14,30) r3 | 0.20 | 40 | 0.590 | **8.8%** |
| (10,14,30) r3 | 0.20 | 110 | 0.533 | **17.6%** |
| (6,8,14) r2 | 0.15 | 6 | 0.80 | 2.1% |

**τ ≤ 0.58: identifiable. τ ≥ 0.65: the response collapses to 2–7% and
calibration from drag is ill-posed** (viscous-dominated drag barely sees the
SGS term; the optimizer exploits solver noise instead). All tests and the
example default sit at τ ≤ 0.58.

## Parameterisation note (found the hard way)

The power closure is centred at `re_ref` = geometric mean of the training Re
(set inside `calibrate`). With a naive `re_ref = 1`, the intercept
`c0 = C_s(Re=1)` sits orders of magnitude from the physical `C_s ~ 0.1` range
— for a truth like `0.08·(Re/40)^-1.2` it is ≈ 6.7, outside the positivity
clamp `log c0 ≤ 0`, and the family silently *cannot represent* steeply falling
closures: the fit still matched C_D to ~1% (drag is insensitive), which is
exactly how the defect surfaced — a scalar baseline beat the power fit at the
train endpoints in `test_calibrate_power_tracks_re_dependence`.

## Verification results

CPU suite `tests/test_autograd_calib.py` (8 tests, ci-cpu torch 2.13 and GPU
venv torch 2.11): C_D-vs-C_s gradient matches central differences to < 1e-6
relative; constant truth recovered to < 5% (`test_calibrate_scalar_recovers_constant`);
power truth `0.08·(Re/40)^-1.2` at train Re {30, 48, 70} (τ 0.58 → 0.53) fits
to < 3%, held-out Re 58 to < 2%, and the constant closure given the same data
and budget is measurably worse at the endpoints it must compromise between.

GPU-scale demonstration (fp64, RTX 5090, (10,14,30) r3, u_in 0.20, 300 steps,
window >= 200; truth `cs_power(0.09, -1.0, re_ref=60)`, train Re {40, 70, 110}
at tau 0.590/0.551/0.533, held-out Re 90; 60 Adam iterations each):

| closure | identified | train C_D err | held-out Re 90 | endpoint errs (Re 40 / 110) |
|---|---|---|---|---|
| power | c0=0.0851 (at re_ref 67.5), b=-0.84 | 0.11 / 0.20 / 0.26% | **0.25%** | 0.11% / 0.26% |
| scalar | C_s=0.0824 | (compromises) | - | 1.65% / 1.43% |

~150 s per 60-iteration fit on one GPU. The drag-level recovery is tight;
*parameter-level* recovery is looser (b -0.84 vs truth -1.0) because
d ln C_D / d ln C_s ~ 0.1 — many closures fit drag almost equally well, which
is the same insensitivity that sets the identifiability regime.

## Honest caveats

- **Verification mode** (synthetic targets from the same solver) proves the
  machinery; with real campaign observations, Smagorinsky *model error* is
  absorbed into the identified constant — the closure reproduces the measured
  drag, not necessarily the true sub-grid physics.
- Calibration from drag requires the identifiable regime (above); at higher τ
  use a flow-field observable instead (the path is differentiable end to end).
- The bounded box keeps lateral planes periodic (A6++ adds free-slip walls on
  a separate branch); the drag window must exclude the initial transient.
