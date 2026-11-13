# PREREG — XVAL-LBM-MOVING-WALL-HALFWAY-BB-D2Q9-BGK

Predeclared **before** the production sweep was run (round 2, mission 18794).
Written after the pilot (one Couette config, one Stokes-2nd config, six-variant
ablation at δ=6) that only served to debug the driver; the pilot numbers are
quoted here as-is so the reader can see what was already known when the
thresholds were fixed.

## Claim under test

"TensorLBM's moving-wall half-way bounce-back closure for D2Q9 + BGK
(pre-streaming reflection of the pre-collision populations + Ladd momentum term
`2 w_q rho_w (c_q.u_w)/cs^2`, effective wall at the half link) is an acceptable
wall model: it reproduces wall-bounded analytic solutions to better than the
owner accuracy standard of 15%."

The closure is the one already shipped in
`benchmarks/verified/couette_2d/run.py::moving_wall_bounce_back` and in
`benchmarks/case_pool/couette_2d/run_case.py`; this study does not modify it.

## Reference cases (external anchors)

| case | analytic reference | why it discriminates |
|---|---|---|
| steady plane Couette | `u = U0 (y-0.5)/H_eff`, exact N-S solution for the given BCs | exact for a linear profile: measures the wall *position* convention and mass/momentum bookkeeping with zero modelling error |
| Stokes' 2nd problem (oscillating plate) | `u = U0 exp(-k(y-0.5)) cos(w t - k(y-0.5))`, `k = sqrt(w/2nu)` (Stokes 1851; Schlichting & Gersten ch. 5) | wall velocity is time dependent: tests the momentum term step-by-step, the phase (effective wall location) and the amplitude (wall slip) inside the viscous layer |

Primary case = Stokes 2nd problem, because a steady linear Couette profile is
reproduced *exactly* by any consistent half-way bounce-back (pilot:
max error 3.4e-10 % cf Umax), so it cannot separate closures.

## Metrics and predeclared thresholds

Reference configuration (primary): `tau = 0.8` (`nu = 0.1`), `delta_bl = 8`,
`gap = 6 delta_bl`, `U0 = 0.02` (`Ma = 0.035`), `nx = 4` (x-periodic),
`float64`, 6 oscillation periods, statistics over the last period.

| id | metric | threshold | verdict rule |
|---|---|---|---|
| M1 | Stokes-2nd max |du|/U0 over the resolved layer `eta <= 3 delta_bl` | **<= 15 %** | anchor acceptance (owner精度标准) |
| M2 | Stokes-2nd amplitude error at `eta = delta_bl` and `2 delta_bl` | <= 5 % | physical slip measure |
| M3 | Stokes-2nd phase error at `eta = delta_bl` and `2 delta_bl` | <= 5 deg | effective wall location measure |
| M4 | Couette pointwise max |du|/umax, `H = 32`, `tau = 0.8`, `U0 = 0.05` | <= 2 % | case-pool tolerance for CASE-COUETTE-2D |
| M5 | Couette fitted effective wall position `y_wall` | `|y_wall| <= 0.01` | half-way convention check |
| M6 | mass drift per case | <= 1e-6 % | wall closure must not create mass |
| M7 | ablation ranking at the reference config: `err(ladd_pre_local) < err(eq_wall_node) < err(static_bb)` | strict | the closure must be measurably better than the naive alternatives, otherwise the reference case does not discriminate |
| M8 | resolution: `err(delta_bl = 16) < err(delta_bl = 4)` | strict, plus fitted log-log slope reported | discretisation error must shrink with resolution; slope value reported as measured, no target |
| M9 | parameter sensitivity: error vs `tau in {0.55,0.6,0.8,1.2,1.8}` and vs `U0 in {0.005..0.1}` reported (no target) | – | envelope documentation |

A metric counts as PASS only if the machine-written number in
`artifacts/summary.json` satisfies the threshold; nothing is re-tuned after
seeing the result. If M1 fails the mission is an honest FAIL with the failing
stage recorded.

## Exclusions / masks (fixed now)

* Errors are always normalised by `U0`, never by the local velocity, so the
  error estimate cannot blow up in the far field (`u -> 0`).
* The far-field truncation floor `exp(-gap/delta_bl)` is reported separately; at
  `gap = 6 delta_bl` it is 0.248 % of `U0`. Rows below it are not used for M1.
* No result tuning, no calibration factor, no extrapolation in time or space.
* `float64` throughout; `float32` is out of scope for this study.