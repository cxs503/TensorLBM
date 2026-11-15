# CASE-TAYLOR-GREEN-3D · round 4 — field-metric verification and gap closure
**Anchor**: `CASE-TAYLOR-GREEN-3D` (算例池·Taylor-Green 3D 衰减, acceptance: reference = 级数精确解, tolerance < 8 %).
Round 3 delivered the exact time-Taylor-series reference and met the 8 % band on the modal quantities, but left two of its own checks (`M8` grid ladder, `M9` Mach attribution) as `not_run` and made three claims that had no measurement behind them. Round 4 closes the two legs and re-verifies the field-level numbers. Two of the three round-3 claims do **not** survive; one of them was a reference-aliasing artifact that made the delivered field numbers look better than they are.
## 1  Case, method, and what changed
Same physical case as rounds 1–3: the classic z-dependent 3-D Taylor–Green vortex

```
u_x = A sin(kx) cos(ky) cos(kz),  u_y = -A cos(kx) sin(ky) cos(kz),  u_z = 0,
k = 2*pi/n,  Re = A n / nu,  T_end = A k t_end = 1
```

n = 32, A = 0.05, Re = 160.0, nu = 0.01, tau = 0.530, 102 steps, D3Q27 BGK, periodic, equilibrium initial condition (all float64). Reference: the exact time-Taylor series of the *same* incompressible NS problem (order 12, nfft 60), whose truncation at T=1 is 6.65e-06 (energy) / 1.80e-04 (enstrophy) — four to five orders of magnitude below every error quoted here, so the error is the solver's.
New/changed in round 4 (`benchmarks/case_taylor_green_3d/run_round4.py`, `make_round4_figures.py`, `tests/test_tg3d_round4.py`):

* `reproduce` — re-runs the round-3 case and records the *actual* reference truncation order/spectrum peak, and re-scores the same field against forced (Nyquist-violating) orders;
* `band` — band-projected field metrics (project both fields onto `|k|_inf <= k_max`, `k_max` = the reference's exact support), which is the only well-posed field comparison when the reference support is finite;
* `grid` — M8, fixed dimensionless problem on n = 24/32/48 with a **common** band;
* `mach` — M9, amplitude ladder on a 64x range of Ma²;
* artefact location moved from `runs/` (git-ignored, which is why round 3's JSONs are not in the repository) to `artifacts/case_taylor_green_3d/round4/`.
## 2  Case acceptance (unchanged metric, re-measured)
| quantity | LBM | exact series | rel. error | < 8 % | < 3 % |
|---|---|---|---|---|---|
| E(T=1)/E(0) | 0.767819 | 0.784576 | -2.136 % | yes | True |
| Ω(T=1)/Ω(0) | 0.825266 | 0.846760 | -2.538 % | yes | True |
| band velocity L2 (|k|∞≤4) | — | rms exact | +3.253 % | yes | False |
| band vorticity L2 (|k|∞≤4) | — | — | +6.491 % | yes | False |

Discrimination witness: single-mode viscous decay would give Ω(T=1)/Ω(0) = 0.789824, the exact nonlinear answer is 0.846760 (+7.21 % enstrophy *production*); the LBM reaches 0.825266, i.e. it recovers 62.2 % of that 3-D signature.
## 3  Findings
### F1 (material, record correction) — round 3's field numbers came from an aliased reference
Round 3 published `u L2 = 2.657 %`, `vorticity L2 = 4.600 %` and stated that *"on a 32³ lattice the field can only be sampled at order 4 (Nyquist)"*. Both are wrong: at T=1 the classic TG series truncated at order 4 has spectral support `|k|∞ = 20`, far above the Nyquist index 16 of a 32³ grid, so that reference field aliases. Forcing that order reproduces the delivered numbers to 1e-3 %-points (`[(3, 3.284, 6.647), (4, 2.657, 4.6), (5, 2.426, 3.461), (6, 2.314, 2.831), (8, 2.234, 2.128)]`), so the attribution is exact, not a guess. The Nyquist-safe truncation is order 3 (support 4), and the honest field errors at the very same run are **u L2 = +3.284 %** and **vorticity L2 = +6.647 %** — inside the 8 % case band, but 60 % (velocity) / 45 % (vorticity) worse than published. `TG3DSeries.velocity_field` now auto-lowers the truncation until the support fits below Nyquist, and `tests/test_tg3d_round4.py` pins both the guard and the corrected numbers.
### F2 (M8, as predeclared: FAIL) — the field error is not converging at the declared order
Grid ladder at fixed (A, Re, T_end), common band `|k|∞ ≤ 4` (the smallest exact reference support over the ladder, capped by the 2/3 rule):

| n | tau | steps | velocity L2 % | vorticity L2 % | energy % | enstrophy % |
|---|---|---|---|---|---|---|
| 24 | 0.5225 | 76 | +3.458 | +6.871 | -3.759 | -4.402 |
| 32 | 0.5300 | 102 | +3.253 | +6.491 | -2.136 | -2.538 |
| 48 | 0.5450 | 153 | +2.029 | +0.729 | -0.982 | -1.193 |

Observed order from the finest pair: **1.16** (per pair ['0.21', '1.16']) against the predeclared `>= 1.8` — **honest FAIL**. The error is dominated by a dissipation bias that decays slowly with refinement, not by a clean second-order spatial truncation. A first cut of this leg compared each grid at *its own* reference support (4 at n=24/32 vs 20 at n=48) and produced a meaningless 0.21; that design error is why the common band exists.
### F3 (M9, as predeclared: FAIL) — the −2 % offset is *not* a compressibility floor
| A | Ma | Ma² | nu | tau | steps | energy % | enstrophy % |
|---|---|---|---|---|---|---|---|
| 0.0125 | 0.0217 | 0.00047 | 0.00250 | 0.5075 | 407 | -2.078 | -2.445 |
| 0.025 | 0.0433 | 0.00187 | 0.00500 | 0.5150 | 204 | -2.089 | -2.465 |
| 0.05 | 0.0866 | 0.00750 | 0.01000 | 0.5300 | 102 | -2.136 | -2.538 |
| 0.1 | 0.1732 | 0.03000 | 0.02000 | 0.5600 | 51 | -2.088 | -2.639 |

Fit: slope **+0.068 %/Ma²**, intercept -2.099 %, R² = 0.0013. Over a 64× change of Ma² the offset moves by less than 0.06 %-points: it is **Ma-independent**, so round 3's claim that *"the −2 % offset is a compressibility floor"* is falsified, and so is its companion claim that it is *"grid independent"* (F2 shows it shrinking −3.76 → −0.98 % from n=24 to n=48). Caveat, stated because it limits the reading: at fixed Re the ladder co-varies nu (= A n / Re) and the step count with the amplitude, so the slope is an effective Ma/tau sensitivity; what it does rule out is any Ma² scaling of the size the claim needs.
### F4 (not run) — mechanism of the offset
The `nueff` leg (fit `nu_eff/nu` from the energy history → express the modal offset in viscosity units) is implemented in `run_round4.py` but **not run**: a single leg exceeds the control-plane per-call wall-time cap, and the batch job carrying it (see §6) had not been dispatched at report time. No mechanism claim is made. Round 3's compressibility attribution was the placeholder for exactly this measurement and is now withdrawn rather than replaced.
## 4  Verdict (predeclared checks, honest)
| check | value | threshold | pass |
|---|---|---|---|
| R1 energy rel err < 8 % | 2.136 | 8.0 | PASS |
| R2 enstrophy rel err < 8 % | 2.538 | 8.0 | PASS |
| R3 Nyquist-safe reference on n=32 grid | 4 | 16.0 | PASS |
| R4 vorticity L2 (Nyquist-safe) < 8 % | 6.647 | 8.0 | PASS |
| B1 band-projected velocity L2 < 8 % | 3.253 | 8.0 | PASS |
| B2 band-projected vorticity L2 < 8 % | 6.491 | 8.0 | PASS |
| B3 band-projected velocity L2 < 3 % | 3.253 | 3.0 | FAIL |
| M8 observed order >= 1.8 | 0.213 | 1.8 | FAIL |
| M8b modal energy offset spread across grids < 1 % | 2.776 | 1.0 | FAIL |
| M9a |slope| of energy err vs Ma^2 in [0.3, 30] %/Ma^2 | 0.06799 | None | FAIL |
| M9b |intercept| < 5 % (residual truncation at n=32) | 2.099 | 5.0 | PASS |

Overall: **FAIL** over 11 checks. The case-pool acceptance (modal error < 8 %, reference = exact series) is met with a margin of ~3× in the energy and enstrophy; the two attribution checks M8/M9 fail, and that failure is the round-4 result: the delivered numbers are right, the explanation attached to them in round 3 was not.
## 5  Caveats and scope
* Re = 160.0 and one tau per grid: this is a *verification* case (does the solver reproduce an exact solution), not a turbulence-statistics case. The compared window is T ∈ [0, 1]; beyond T ≈ 1.5 the order-12 series tail exceeds 1 %.
* The comparable field band is capped at `|k|∞ ≤ 4` by the coarse grids' reference truncation, so the field metrics say nothing about modes the exact reference cannot supply at n ≤ 32; the LBM's out-of-band energy fraction is reported instead (≤ 4.231 %).
* BGK only, equilibrium initial condition: the O(Ma²) acoustic transient is part of the measured error, never subtracted.
* Mass drift over the run: rho ∈ [0.997834, 1.002684].
## 6  Checks not run / limits of this round
| item | status |
|---|---|
| M9 `nueff` effective-viscosity fit (mechanism) | `not_run` (per-call wall-time cap; batch job pending) |
| independent batch replay of all legs (`run_round4.py --all`) | submitted as cj3831, queued — not waited for |
| round-3 `runs/*.json` provenance | unavailable: `runs/` is git-ignored, so round 3's raw numbers cannot be re-read, only regenerated |
| turbulence statistics / higher Re | out of scope |
## 7  Artefacts
| path | content |
|---|---|
| `benchmarks/case_taylor_green_3d/run_round4.py` | legs `reproduce`/`band`/`grid`/`mach`/`curves`/`nueff`/`verdict` |
| `benchmarks/case_taylor_green_3d/make_round4_figures.py` | JSON → figures + `summary.md` |
| `benchmarks/case_taylor_green_3d/publish_round4.py` | this document + platform report |
| `tests/test_tg3d_round4.py` | CI guards: Nyquist guard, corrected field numbers, M8/M9 windows |
| `artifacts/case_taylor_green_3d/round4/*.json` | every number in this document |
| `artifacts/case_taylor_green_3d/round4/figures/*.png` | figures ④ |
