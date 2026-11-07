# maneuvering_zigzag — closed-loop zig-zag (overshoot) + IMO gate  ✅ VERIFIED

Status: **verified** (2026-11-07). The closed-loop helmsman logic added to
`tensorlbm.maneuvering` (`simulate_zigzag`) reproduces the exact piecewise-modal
solution (`analytic_zigzag`) to **1.2e-12** relative on every overshoot angle and
on every helm-reversal / heading-peak time, and the time-step convergence orders
come out as designed (Euler **p = 0.998**, RK4 **p = 4.02**).

## What is verified

| route | what it is |
|---|---|
| `simulate_zigzag` | event-driven integration of the same sway-yaw ODE as `simulate_turn`, with helm reversals («heading reached») and heading extrema («yaw rate = 0») located **inside** a step by bisection — no O(dt) switch-time quantisation |
| `analytic_zigzag` | exact piecewise-modal solution: each rudder segment is LTI, `q(t)`/`psi(t)` are two-exponential closed forms, switch and peak times are roots of those forms |

Also covered: `check_imo_msc137_zigzag` / `imo_msc137_zigzag_limits`
(IMO Res. MSC.137(76) annex 2.3 overshoot limits).

## Result (KVLCC2-class tanker, L=320 m, U=7.97 m/s, L/V = 40.2 s)

| quantity | 10°/10° | 20°/20° |
|---|---|---|
| first overshoot [deg] | 2.109197 | 4.218394 |
| second overshoot [deg] | 2.903124 | 5.806248 |
| helm reversal times [s] | 56.224 / 174.531 | 56.224 / 174.531 |
| heading peak times [s] | 74.738 / 196.430 | 74.738 / 196.430 |

| check | value | criterion |
|---|---|---|
| two-route max relative difference (6 quantities x 2 tests) | **1.2e-12** | < 1e-4 |
| Euler observed order (dt 1.0 → 0.125 s) | 0.998 | 0.8 … 1.3 |
| RK4 observed order (dt 0.5 → 0.0625 s) | 4.02 | ≳ 3.5 |
| RK4 finest relative error | 3.0e-12 | < 1e-5 |
| linear-model scale invariance 20°/20° = 2 × 10°/10° | 0.0 (exact) | < 1e-9 |
| port/starboard mirror symmetry | 0.0 (exact) | < 1e-9 |
| cross-flow (Y_vv/N_vv) case breaks the invariance | ratio 1.9226 vs 2.0 | 1.5 … 1.99 |
| IMO gate: nominal tanker | pass (first 4.22° ≤ 25°) | — |
| IMO gate: degraded course-keeper (N′_r × 0.3) | **fail** (first 32.8° > 25°) | — |

### Why 20°/20° is exactly twice 10°/10°

For a *linear* sway-yaw model with no rudder-rate limit, `psi(t) = delta·g(t)` with
`g` independent of `delta`.  Scaling the rudder and the ordered heading by the
same factor leaves every switch time unchanged and doubles every angle — so the
10/10 and 20/20 tests are literally the same solution.  The cross-flow case
(`Y_vv = |Y_v|/U`, `N_vv = |N_v|/U`) breaks it (ratio 1.92), as a real hull does.
This is pinned by tests so a future change to the helm logic cannot hide behind it.

## Honest boundary (recorded in `result.json`)

* The hydrodynamic derivatives are a **representative** KVLCC2-class linear set —
  not PMM/CFD-extracted.  **Absolute** overshoot angles for a specific ship are
  therefore WITHHELD; the claim is *internal verification* of the helmsman logic
  and of the criterion implementation (two independent routes + order of accuracy).
* No rudder-rate limit, no propeller/rudder slipstream, no heel/roll coupling, no
  free surface: this is a 3-DOF horizontal-plane linear model, not a hull-in-flow
  simulation.
* The yaw-damping degradation family used to show the IMO gate has teeth is a
  *sensitivity scan*, not a set of physical ships: members with `N′_r × scale ≤ 0.5`
  give `R/L < 1` and are directionally unstable.
* The IMO limits are transcribed from Res. MSC.137(76) annex 2.3 (10/10 → 10° if
  L/V < 10 s else 20°; 20/20 → 25° first / 40° second); combinations the
  resolution does not define (e.g. 20/10) return `applicable: False` rather than
  inventing a threshold.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/verified/maneuvering_zigzag/run.py   # ~2 min CPU, writes result.json + figures/
PYTHONPATH=src python -m pytest tests/test_maneuvering_zigzag.py -q   # 13 tests, ~35 s
```

Figures: `figures/fig1_zigzag_histories.png` (heading/rudder histories, analytical
vs RK4), `fig2_overshoot_convergence.png` (error vs dt, both schemes),
`fig3_imo_gate.png` (overshoot vs yaw damping with the IMO limits).