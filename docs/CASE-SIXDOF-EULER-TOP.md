# CASE-SIXDOF-EULER-TOP — torque-free rigid-body rotation vs analytic solution

Anchor: verification debt of the merged 6-DOF path. `tensorlbm/sixdof_capability_contract.py`
certifies 6-DOF only as `CONTRACT_TESTED` and declares `WITHHELD_NO_PHYSICS_VALIDATION`;
before this case no trajectory of the integrator had ever been compared with a known
solution. This case discharges the **torque-free rotation** sub-scope only (coupling /
added-mass / moving-body flow physics remain WITHHELD).

## 1. Case introduction
Rigid body with zero external moment, body frame, Euler equations
`I dω/dt = M − ω × (Iω)`; integrated by the library path
`tensorlbm.sixdof_common.rigid_body_step → tensorlbm.sixdof.step_sixdof`
(explicit/symplectic Euler + exponential-map quaternion update, dt in seconds,
gravity disabled so the configuration is genuinely torque-free).

| parameter | value |
|---|---|
| symmetric top inertia (I1, I2, I3) | (2.0, 2.0, 1.0) |
| initial body rate ω0 | (0.3, 0.0, 1.0) rad/s |
| analytic precession rate Ω = (I1−I3)ω3/I1 | 0.5 rad/s (period 12.5664 s) |
| tumble body inertia | (1.0, 2.0, 3.0), ω0 = (1e−3, 1, 1e−3) rad/s |
| analytic intermediate-axis rate | 0.577350 1/s |

## 2. Method
Three independent referees, none of which touches the integrator:
1. **Exact symmetric-top solution** (arbitrary amplitude, no linearisation):
   `ω1 = ω1_0 cos Ωt + ω2_0 sin Ωt`, `ω2 = −ω1_0 sin Ωt + ω2_0 cos Ωt`, `ω3 = ω3_0`.
2. **Exact first integrals**: `T = ½ω·Iω`, `|Iω|` (body), `|R(q) Iω|` (world, a constant
   vector under pure rotation), `|q| = 1`.
3. **Linear intermediate-axis rate** `λ = |ω2| √((I3−I2)(I2−I1)/(I1I3))`, used inside a
   stated amplitude band.

Same metric functions are applied to the analytic solution itself, so no number is fitted.
Falsification: the reference frequency is deliberately shifted by 0.05 rad/s (10% of Ω)
and the same gate must fail.

Reproduce: `python scripts/run_sixdof_rotation_validation.py`
(tests: `pytest tests/test_sixdof_rotation_validation.py`).

## 3. Results (programmatic, `artifacts/sixdof_rotation_validation/summary.json`)

Symmetric top, dt refinement at fixed t = 12 s:

| dt [s] | steps | rel L2 | rel inf | axial drift | period rel err | energy drift |
|---|---|---|---|---|---|---|
| 0.04 | 300 | 1.019e-02 | 1.774e-02 | 0.0e+00 | 1.33e-04 | 1.94e-02 |
| 0.02 | 600 | 5.035e-03 | 8.755e-03 | 0.0e+00 | 3.33e-05 | 9.43e-03 |
| 0.01 | 1200 | 2.503e-03 | 4.349e-03 | 0.0e+00 | 8.33e-06 | 4.65e-03 |
| 0.005 | 2400 | 1.248e-03 | 2.167e-03 | 0.0e+00 | 2.08e-06 | 2.31e-03 |


Observed convergence order from log-log fit: **1.010** (first-order, as expected
for explicit Euler on the gyroscopic term).

Single-run gate at dt = 0.02: rel L2 = 5.035e-03, rel inf = 8.755e-03,
axial-rate drift = 0.0e+00 (axial rate is exactly constant),
period recovered = 12.5668 s vs analytic 12.5664 s
(rel err 3.33e-05).

Invariant drift over the same 12 s window (dt = 0.02): energy 9.43e-03,
body |L| 8.15e-03, world |L| 8.15e-03,
quaternion norm deviation 2.22e-16.

Dzhanibekov (intermediate axis): fitted rate 0.6235 1/s vs linear theory 0.5774 1/s
→ rel err **8.000%** inside the band |ω1| ∈ [2e−3, 1e−2]
(937 samples, t ∈ [3.13, 5.00] s).

Falsification (reference frequency shifted 10%): rel L2 = 9.996e-02,
rel inf = 1.799e-01 — the gate rejects a wrong reference by a factor
20 over the accepted error.

## 4. Figures
`artifacts/sixdof_rotation_validation/figure.png` — (a) ω1 from the library path vs the exact
symmetric top, (b) log-log dt refinement with the fitted slope, (c) |ω1| growth for the
intermediate-axis case against the linear rate.

## 5. Caveats
- Explicit Euler on the gyroscopic term is **not** symplectic, so T and |Iω| drift secularly
  at O(dt) (2.3e−3 over 12 s at dt = 0.02). This is reported, not hidden; refinement reduces it.
- The Dzhanibekov fit is **semi-quantitative**: the exact solution is cnoidal, so the
  exponential band fit depends on the chosen amplitude window (bands 1e−3..5e−3 and
  2e−3..1e−2 differ by ~10%). Band and tolerance are stated (20%).
- Only the torque-free sub-scope is validated. Fluid coupling, added mass, moving-body drag
  and FSI energy exchange remain `WITHHELD_NO_PHYSICS_VALIDATION`.

## 6. Error judgement vs analytic reference
| quantity | reference | computed | rel. error | < 3%? |
|---|---|---|---|---|
| ω(t) trajectory, dt=0.02, 12 s | exact symmetric top | rel L2 5.035e-03 | 5.035e-03 | yes |
| precession period | 12.5664 s | 12.5668 s | 3.33e-05 | yes |
| axial rate ω3 | exactly constant | drift 0.0e+00 | 0.0e+00 | yes |
| intermediate-axis growth rate | 0.5774 1/s (linearised) | 0.6235 1/s | 8.00% | no — linearisation-only anchor, stated in caveats |
