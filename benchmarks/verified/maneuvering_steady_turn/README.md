# maneuvering_steady_turn — steady turning response  ✅ VERIFIED

Status: **verified** (2026-10-05). The steady-turn closure reproduces the exact
fixed point of the sway-yaw maneuvering model to <0.02% on every level, with an
inter-level span below 0.001%.

## Result

| quantity | error | 
|---|---|
| steady turning radius R | **+0.009%** |
| drift angle beta        | **-0.011%** |
| two-level span          | **<0.001%** |
| independent algebraic route | 2.8e-14% |

* Euler transient (fine level): 0.124%, refinement ratio **4.09x** (first order — correct)
* RK4 transient: 1e-7% (fourth order — correct)
* Directionally stable; R/L = 2.01, beta = 15.3 deg, K' = 1.43 (all within the
  published tanker range)

## What is verified

`src/tensorlbm/maneuvering.py` (new, this cycle): linear / nonlinear-crossflow
(Abkowitz / Nomoto / Fossen) sway-yaw model, an exact `steady_turn` fixed point
(closed-form 2x2 for the linear case, bisection for crossflow), `simulate_turn`
time integration (semi-implicit Euler + RK4) and an analytic modal transient.

The benchmark runs two rudder angles (10/20 deg) x two dt (1.0/0.25 s) x two
integrators, against a KVLCC2-class tanker (L=320 m, U=7.97 m/s), and cross-checks
three independent routes (library fixed point == NumPy == Nomoto closed form).

## Honest boundary (recorded in result.json)

The hydrodynamic derivatives are **representative KVLCC2-class linear coefficients
synthesised by the caller** — not extracted from an LBM flow field and not from a
PMM experiment. There is no free surface, no rudder/propeller slipstream and no
higher-order nonlinearity. End-to-end fluid-maneuvering coupling remains a
structural block (free surface). The nonlinear crossflow path is recorded as a
diagnostic only.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/verified/maneuvering_steady_turn/run.py
```
Self-contained, deterministic, ~10 s on CPU, exit code 0.
