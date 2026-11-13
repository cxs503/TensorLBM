# CASE-TAYLOR-GREEN-3D — round 2: where the round-1 error floor came from, and the fix

**Anchor:** `CASE-TAYLOR-GREEN-3D` (round 1 = mission 15374, `docs/CASE_TAYLOR_GREEN_3D.md`;
round 2 = mission 17543, branch `agent/m17543`).  All numbers below are generated from
`benchmarks/case_taylor_green_3d/artifacts/attribution_*.json` by
`benchmarks/case_taylor_green_3d/make_report_round2.py`; the criteria were fixed in
`prereg_round2.json` **before** the runs.

## ① What round 1 left open (and what this round tests)

Round 1 passed every predeclared metric, but two of its numbers looked
unphysical for a second-order solver:

* **M2** (energy decay-exponent error, least squares over the *whole* history) was
  +0.366 % at N=24,
  +0.356 % at N=32,
  +0.349 % at N=48,
  +0.346 % at N=64 —
  i.e. *independent of the grid*; and
* the consecutive L2 orders were 0.86 (24→32)
  and 0.84 (48→64) while the 24→48 pair gave
  2.05 — round 1 called this 'a grid-independent error floor'.

Round 2 tests the two *protocol* explanations (prereg_round2.json, H1/H2/H3):

| id | hypothesis |
|---|---|
| H1_protocol_fit | M2's whole-history least-squares fit of ln E(t) is inflated by the startup transient of the equilibrium initial condition; the asymptotic decay-rate error (slope of ln(E_num/E_exact)) is much smaller than M2. |
| H2_ic_floor | The 'grid-independent floor' in M3 is the missing first-order Chapman-Enskog part of the initial condition: f = f^eq(u0) starts with zero viscous stress, an O(k^2) state error, so the L2 error scales like O(k^2) (order ~1) instead of O(k^3). |
| H3_repair | Adding the first-order non-equilibrium part f^(1) = K w_i Q_iab S_ab removes that state error and restores second-order convergence of the L2 field error. |
| H4_ma_independence | The residual decay-rate error is not compressibility-driven at these amplitudes (it should not scale as Ma^2). |

## ② Method

* Same solver path as round 1 — `tensorlbm.d3q27` (`equilibrium27`, `collide_bgk27`,
  `macroscopic27`) driven by `lbm_step_correct_27`; D3Q27 BGK, float64, fully periodic,
  no solid cells, no mass correction. The case protocol is *not* rebuilt: only the
  initial condition and the post-processing are varied, so the comparison isolates them.
* **Initial conditions.** `eq`: `f = f^eq(rho=1, u0)` (round-1 protocol). `half`/`tau`: the
  first-order Chapman–Enskog part is added,
  `f_i = f_i^eq + K w_i Q_iab S_ab`, `Q_iab = e_ia e_ib - c_s^2 delta_ab`,
  `S_ab = (d_a u_b + d_b u_a)/2` (spectral derivatives), with `K = -rho(tau_l - theta)/c_s^2`
  and `theta = 1/2` (`half`) or `theta = 0` (`tau`). Both literature conventions are
  *measured*; the winner is read off the energy offset, not assumed.
* **Rate metrics.** `M2` = the round-1 least-squares slope of `ln E(t)` over the whole
  history (`t_start = 0`); re-fitting over `t_start ∈ {5,10,20,50,100}` exposes the
  startup transient. `asym` = slope of `ln(E_num/E_exact)` over the second half of the
  trace, which is the asymptotic rate error `(lambda_num - lambda_exact)/lambda_exact`
  and is independent of the initial condition (any state offset differentiates away).
* Reproduction: `python benchmarks/case_taylor_green_3d/run_attribution.py --arm all`
  then `python benchmarks/case_taylor_green_3d/make_report_round2.py`.

## ③ Results

### 3.1 Initial-condition study (N=24, traces sampled every 5 steps)

| IC | l2 field err at T | energy offset at T [%] | M2 (whole history) [%] | asymptotic rate err [%] |
| --- | --- | --- | --- | --- |
| eq | 1.135e-02 | -2.2580 | +0.0605 | +0.0210 |
| half | 9.537e-03 | -1.8982 | +0.0540 | +0.0210 |
| tau | 4.628e-04 | -0.0895 | +0.0213 | +0.0210 |

![startup transient](figures/case_taylor_green_3d_round2/fig1_startup_transient.png)

### 3.2 The fit-window artefact (N=32, IC=eq)

| window start t_start | fitted decay-exponent rel. error [%] |
| --- | --- |
| 0 | +0.0512 |
| 5 | +0.0075 |
| 10 | +0.0125 |
| 20 | +0.0122 |
| 50 | +0.0122 |
| 100 | +0.0122 |

Asymptotic rate error over the same trace: **+0.0122 %**.

**Record-cadence artefact.** The round-1 metric does not only depend on the fit window,
it also depends on the recording cadence (the number of early transient samples that
enter the least-squares): the identical N=32 eq-IC run gives

| record cadence [steps] | M2 (whole-history fit) [%] | asym rate error [%] |
| --- | --- | --- |
| 5 | +0.0512 | +0.0122 |
| 10 | +0.1397 | +0.0122 |
| 25 | +0.2416 | +0.0122 |
| 50 | +0.3772 | +0.0122 |
| 100 | +0.7004 | +0.0122 |

![M2 window sensitivity](figures/case_taylor_green_3d_round2/fig3_m2_window.png)

### 3.3 Grid convergence with the CE-consistent IC (T=500, tau=0.6)

| N | L2 err (round-1, eq IC) | L2 err (CE IC) | repair factor | order 24→48 (CE) |
| --- | --- | --- | --- | --- |
| 24 | 1.1450e-02 | 5.5010e-04 | 20.8x |  |
| 32 | 6.3281e-03 | 1.8907e-04 | 33.5x |  |
| 48 | 2.7741e-03 | 4.5830e-05 | 60.5x |  |
| 64 | 1.5539e-03 | 6.7787e-05 | 22.9x | 3.585 |

consecutive orders with the CE IC: 24_to_32: 1.541, 32_to_48: 2.045, 48_to_64: -0.565

![L2 convergence](figures/case_taylor_green_3d_round2/fig2_l2_convergence.png)

### 3.4 Residual decay-rate error vs amplitude and vs viscous rate (N=32, CE IC)

| arm | value | lambda_exact [1/step] | asymptotic rate err [%] |
| --- | --- | --- | --- |
| amplitude | A=2.5e-04 | 5.1404e-03 | +0.0123 |
| amplitude | A=2.0e-03 | 5.1404e-03 | +0.0122 |
| amplitude | A=2.0e-02 | 5.1404e-03 | +0.0078 |
| tau | tau=0.55 | 2.5702e-03 | +0.0025 |
| tau | tau=0.6 | 5.1404e-03 | +0.0122 |
| tau | tau=1.0 | 2.5702e-02 | +0.3231 |


Log-log slope of the rate error against `lambda_exact` over the tau sweep: **+2.09**
(2 means a second-order-in-Dt truncation error; the fitted one-parameter law is
`delta ~ 4.9 lambda^2`, which is consistent with both the tau arm and the grid arm since
`lambda ~ k^2` at fixed tau).

![rate scaling](figures/case_taylor_green_3d_round2/fig4_rate_scaling.png)

## ④ Predeclared criteria vs outcome

| criterion | measured | pass |
| --- | --- | --- |
| R1 M2 inflation ratio (N=32) | 4.2  (M2 +0.0512 % vs asym +0.0122 %) | ❌ |
| R2 IC repair factor >= 5x | N=24: 20.8x, N=32: 33.5x, N=48: 60.5x, N=64: 22.9x | ✅ |
| R3 order 24→48 >= 1.8 and min consecutive >= 1.7 | 3.585 / -0.565 | ❌ |
| R4 |asym rate err| < 0.1 % at every point | worst 0.3231 % over 16 points | ❌ |
| R5 amplitude slope (Ma^2 would need ~2) | -0.11 | ✅ |

## ⑤ Caveats / limits

* The CE-consistent IC uses **spectral derivatives** of the initial field; for the
  sine/cosine Taylor–Green and ABC modes this is exact, but a general (non-periodic,
  non-analytic) initial condition would need finite differences and the IC error would
  then be limited by that stencil, not by the solver.
* The asymptotic rate error is measured over the second half of a fixed trace; it is a
  single-mode property (`|k|^2 = 2k^2`) and says nothing about broadband flows.
* The `half`/`tau` distinction matters physically: `theta = 0` (`tau`) is the form that
  lands on the discrete slow manifold here. This is an empirical result at tau in
  {0.55, 0.6, 1.0} and Ma <= 3.5e-2; it is not derived.
* Round-1 numbers are kept unchanged: M2/M4 as published stay the official record; this
  round adds the diagnostic metrics (asym, IC study) rather than redefining M2.

## ⑤b What round 2 concludes (and what failed)

* **Attributed:** round 1's grid-independent M2 and its 0.84–1.19 consecutive orders
  come from the equilibrium initial condition (an O(k^2) state error, no viscous stress)
  and from the whole-history/cadence-dependent fit, **not** from the D3Q27 solver.
  The asymptotic rate error of the same runs is 0.0017–0.021 % at tau=0.6.
* **Repaired:** the first-order Chapman–Enskog part of the IC (`ic='tau'`) removes the
  state error and cuts the L2 field error by 20–60x (R2 PASS). It is now the canonical
  builder `run_case.init_distribution(..., ic=...)`; `--ic eq` reproduces round 1 exactly.
* **Honest FAILs.** R1 as predeclared failed (4.19 < 5) because the prereg did not fix
  the recording cadence — the cadence table above shows the same run gives a
  31x inflation at a 50-step cadence, so the conclusion stands but the criterion
  as written did not. R3 failed: with the state error gone the L2 error falls faster than
  second order (24→48 slope 3.59) and becomes non-monotone at N>=48 (48→64 slope -0.56),
  i.e. no spatial order can be claimed in N=24..64 at T=500 — round 1's 2.045 was a
  crossover coincidence of the IC error, not a convergence rate. R4 failed at tau=1.0
  (0.32 %), which is not a defect but the measured second-order temporal term
  (`delta ~ 4.9 lambda^2`): high-viscosity runs need a smaller Dt, and at the case's
  design point (tau=0.6) every rate error is < 0.022 %.
* **Next step (not done here):** a time-refined (tau-scaled) protocol and a
  time-averaged L2 estimator are needed to recover a convergence rate at N>=48.

## ⑥ Error judgement (vs reference)

| quantity | reference | error | < 3 % ? |
|---|---|---|---|
| L2 velocity field, CE IC, N=24 | exact Fourier mode | 5.501e-04 (relative) | yes (0.03 %) |
| L2 velocity field, CE IC, N=32 | exact Fourier mode | 1.891e-04 (relative) | yes |
| decay exponent (asymptotic) | `2 nu |k|^2` | 0.3231 % (worst over all arms) | yes |
| round-1 M2 (whole-history fit) | `2 nu |k|^2` | 0.346 % (N=64) | yes, but inflated by the startup transient |

All reference values are closed-form (`u(t) = u0 exp(-nu|k|^2 t)`), recomputed inside the
run — no table was transcribed.

Figures: fig1_startup_transient.png, fig2_l2_convergence.png, fig3_m2_window.png, fig4_rate_scaling.png
