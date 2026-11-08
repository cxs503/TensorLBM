# LBM linear stability and equivalent-equation verification surface (P155-N08, r1)

**Scope.** D2Q9 lattice Boltzmann, BGK (SRT) and MRT (Lallemand–Luo moment space)
collision, isothermal, single phase, periodic box. This document reports the
von Neumann (linear stability) surface, the dispersion–dissipation
(equivalent-equation) coefficients, and the *measured* cross-audit of both
against explicit nonlinear time integration.

**Artifacts.**

| path | content |
|---|---|
| `src/tensorlbm/stability/von_neumann.py` | amplification matrix, spectra, mode identification, box experiment |
| `src/tensorlbm/stability/campaign.py` | predeclared acceptance protocol (C1–C5), `build_summary` |
| `scripts/run_lbm_von_neumann_stability_campaign.py` | campaign driver (JSON + figures) |
| `scripts/summarize_lbm_von_neumann_campaign.py` | `results.json` → `summary.json` (no recomputation) |
| `tests/test_lbm_von_neumann_stability.py` | 27 unit/integration tests (`pytest -q`, ~10 s) |
| `artifacts/lbm_von_neumann_p155_n08/results.json` | every point, every number (raw record) |
| `artifacts/lbm_von_neumann_p155_n08/summary.json` | criterion verdicts (pure function of `results.json`) |
| `artifacts/lbm_von_neumann_p155_n08/results_supplement.json` | supplementary unstable-branch check |
| `artifacts/lbm_von_neumann_p155_n08/fig1..fig5*.png` | figures used below |

**Environment.** python 3.11.6, numpy 2.4.6, torch 2.7.1, matplotlib 3.10.8,
branch `agent/m6688` on top of `cb5a3671`; full campaign wall time **470 s**
(one core, 128-core host, no GPU), float64 throughout.

**Commands.**

```bash
PYTHONPATH=src python3 scripts/run_lbm_von_neumann_stability_campaign.py \
    --out artifacts/lbm_von_neumann_p155_n08            # 470 s
PYTHONPATH=src python3 scripts/run_lbm_von_neumann_stability_campaign.py \
    --supplement-only --out artifacts/lbm_von_neumann_p155_n08   # 0.4 s
PYTHONPATH=src python3 scripts/summarize_lbm_von_neumann_campaign.py \
    --results artifacts/lbm_von_neumann_p155_n08/results.json
PYTHONPATH=src python3 -m pytest tests/test_lbm_von_neumann_stability.py -q
```

---

## 1. Case description (physics, geometry, parameters)

The state is a uniform base flow `f_i^eq(rho0, u0)` plus an infinitesimal
single-mode perturbation. Lattice units `dx = dt = 1`, `cs^2 = 1/3`, so the
Mach number is `Ma = |u0| sqrt(3)`. Two collision models are audited:

* **BGK**: `f^post = f - (f - f^eq)/tau`, viscosity `nu = (tau - 1/2)/3`.
* **MRT** (production default of `tensorlbm.solver.collide_mrt`): moment
  ordering `(rho, e, eps, jx, qx, jy, qy, pxx, pxy)`, relaxation rates
  `(0, 1.64, 1.54, 0, 1.7, 0, 1.7, 1/tau, 1/tau)`. A small rate search is
  reported in §3.5 to test whether any rate set changes the BGK/MRT verdict.

Parameter ranges actually exercised (all lattice units):

| group | tau | Ma | theta | box |
|---|---|---|---|---|
| C1 protocol A (physical) | 0.52, 0.55, 0.60, 0.70, 0.85, 1.00 | 0, 0.2, 0.4 | 2 pi m/32, m = 1,2,4 | 32x32 |
| C1 protocol B (acoustic) | 0.60, 0.80 | 0, 0.2 | 2 pi m/32, m = 2,4 | 32x32 |
| C1 supplement (unstable) | 0.35, 0.40, 0.45 | 0, 0.2 | 2 pi /32 | 32x32 |
| C2 divergence probes | 0.40 … 1.00 (8 values) | 0, 0.2, 0.4, 0.6 | full box spectrum (32x32) | 32x32 |
| C3 equivalent equation | 0.60, 0.80, 1.00 | 0, 0.2 | 0.02 … 1.20 (27 values) | analytic |
| C4 viscosity gate | 0.52 … 1.20 (7 values) | 0 … 0.4 (5 values) | 0.002 … 1.20 (40 values) | analytic |
| C5 collision comparison | 0.45 … 1.00 (8 values) | 0 … 0.6 (7 values) | fine grid 33x33 | analytic |

---

## 2. Method

### 2.1 Analytic side (von Neumann)

Linearising around a uniform base state and inserting
`delta f_i = v_i exp(i(k.x - omega t))` gives the 9x9 eigenproblem

```
A(theta) v = exp(-i omega) v,     A(theta) = diag(exp(-i theta.c_i)) P,
P = I - C (I - J_eq),             J_eq = d f^eq / d f,
```

with `C = I/tau` (BGK) or `C = M^-1 S M` (MRT). The eigenvalues `mu_j` are the
per-timestep amplification factors:

```
sigma_j = ln|mu_j|   (growth, > 0 unstable),   c_j = -arg(mu_j)/|theta|   (phase speed)
lambda_max(theta) = max_j |mu_j|;   stable  <=>  max_theta lambda_max <= 1
```

`J_eq` is derived analytically (`d u_a/d f_j = (c_ja - u_a)/rho`) and locked
against a finite-difference Jacobian in the test suite; the MRT operator is
locked against `tensorlbm.solver.collide_mrt` (agreement `< 1e-12`, float64).

**Mode identification.** The transverse *kinetic* modes of BGK share the
momentum structure of the shear mode (`mass = 0`, `perp = 1`), so moment
content alone cannot name branches. Two mechanisms are used instead:

1. overlap with the exact `k -> 0` eigendirection (source term of a purely
   transverse velocity perturbation for the shear branch, of a pure density
   perturbation for the acoustic branches);
2. **eigenvector continuation in `k`** (`shear_branch_curve`), which follows the
   shear branch from `k -> 0` by maximum overlap with the previous wave vector.
   This is required at `tau = 1` where the kinetic eigenvalues vanish and the
   kinetic subspace becomes degenerate (overlap-based naming alone jumps
   branches there — see §5).

### 2.2 Measured side (periodic box, full nonlinear LBM)

`run_modal_box` integrates the **full nonlinear** LBM in float64 on an
`n x n` periodic box, seeded with a single Fourier mode, and fits the
exponential growth of the modal amplitude:

* seed `shear` (physical): `u = u0 + A p cos(k.x)`, `A = 1e-6` (quadratic terms
  `~1e-12`); **no eigendata is used in the initial condition**;
* seed `sound` (physical): `rho = rho0 + A cos(k.x)`;
* seed `eigen` (branch-selective): the exact right eigenvector — used for the
  acoustic branches and for instability probes.

Observables: **vorticity** `i(kx uy - ky ux)` for the shear branch (the
acoustic branches are irrotational, so this is a physical, non-circular
single-mode filter), density for acoustic branches, and the left-eigenvector
projection for kinetic branches. Growth and phase are fitted by least squares
on `ln|obs|` and `unwrap(arg obs)` over a window that (i) starts after
`t = 12` spin-up, (ii) stays inside `|u - u0| < 1e-3`, and (iii) stays above
`1e-12` in modal amplitude (float64 modal noise for an `O(1)` background is
`~1e-16`). The reference branch is the one carrying the largest modal amplitude
at the start of that window (declared rule; the *rate* is still compared with
the analytic eigenvalue).

### 2.3 Predeclared acceptance protocol

Thresholds are declared in `campaign.PREDECLARED` **before** the runs and are
not tuned afterwards:

| id | predeclared threshold |
|---|---|
| C1 | growth-rate relative error `< 5%`, `|lambda| <= 1.2`, `>= 24` points, purity `>= 0.99`, `R^2 >= 0.99` |
| C2 | unstable guarded set `lambda_max >= 1.001` (so that `lambda^5000 >= 100` is resolvable), divergence = blow-up **or** 100x modal growth in `<= 5000` steps; TPR `>= 90%`, FPR `<= 10%` |
| C3 | quadratic fit `nu_num = a + b theta^2` on `theta in [0.02, 0.60]`, max relative deviation `< 10%`; `a` vs `(tau-1/2)/3` within 2% at `Ma = 0` |
| C4 | `|nu_num - nu_phys|/nu_phys < 1%` band table, required to exist at `Ma = 0` |
| C5 | MRT stable set must contain the BGK stable set (declared form); pointwise `lambda_max` penalty reported as well |

Design controls: identical code path for prediction and measurement is avoided
by construction (analytic linearisation vs nonlinear time stepping share only
`c_i`, `w_i` and the collision definition); the C2 prediction uses the
**box-grid** spectrum (the only wave vectors realisable in the experiment), not
the fine-grid one; a `x10` seed-amplitude control reproduces the growth rate to
`< 1e-3` relative (test `test_measured_growth_is_amplitude_independent`).

---

## 3. Results

All numbers are in `artifacts/lbm_von_neumann_p155_n08/summary.json`
(programmatically derived from `results.json`; nothing in this section was
transcribed by hand).

### 3.1 C1 — spectral radius vs measurement (pass)

70/70 in-band points pass the 5% criterion (requirement: 24).

| protocol / branch | n | max rel. error | mean rel. error |
|---|---|---|---|
| A-physical, shear (vorticity observable) | 51 | 5.52e-6 | 9.85e-7 |
| A-physical, kinetic (data-selected branch) | 3 | 5.74e-7 | 2.85e-7 |
| B-branch-selective, acoustic | 16 | 7.80e-7 | 2.32e-7 |
| **all** | **70** | **5.52e-6** | median **3.46e-7** |

`lambda` range covered: **0.9024 … 0.99978** (stable side of the band). Phase
speed (52 acoustic points): max relative error **2.26e-7**
(max absolute **3.6e-8**); the shear branch has `c = 0`, so a *relative* phase
error is undefined there by construction.

Supplementary unstable-side check (`results_supplement.json`): 4/6 points pass
5%, `lambda` up to **1.855**, max relative error **9.7e-11**. The two failures
are `tau = 0.45` points where a faster-growing kinetic branch overtakes the
shear branch mid-window (fit `R^2 = 0.09`/`0.10`, i.e. detected and rejected,
not silently averaged); they are excluded from the C1 count.

### 3.2 C2 — stability boundary (pass)

| quantity | value |
|---|---|
| predicted-unstable (guarded, `lambda_max >= 1.001`) | 30, diverged **30** -> TPR = **1.00** |
| predicted-stable (`lambda_max <= 1 + 1e-9`) | 34, diverged **0** -> FPR = **0.00** |
| weakly unstable (`1 < lambda_max < 1.001`) | 25, diverged 0 (expected: `1.0005^5000 = 12 < 100`, below the declared resolution) |
| BGK split | 15/15 unstable diverged, 0/17 stable diverged |
| MRT split | 15/15 unstable diverged, 0/17 stable diverged |

The probes are two-fold: (i) seeding the *predicted* most amplified box mode
(eigendata used only as a probe), and (ii) a **broadband** mass/momentum-neutral
random seed that uses no eigendata at all; either one diverging counts as a
divergence. The weak-instability band is reported separately instead of being
counted as a false negative: the measured growth factor at
(`tau = 0.51`, `Ma = 0.4`, MRT, `lambda = 1.00053`) was 14.0 against the
theoretical `exp(5000 ln 1.00053) = 14.2`, i.e. the experiment resolves the
sub-threshold rate quantitatively even where it cannot reach 100x.

### 3.3 C3 — equivalent equation, second-order truncation (pass)

`nu_num(theta) = -sigma_shear/theta^2` on the continuation-traced shear branch
(tracking overlap `>= 0.99964` everywhere):

| collision | Ma | a (theta->0) | b (theta^2 coeff.) | b/a | nu_phys | max rel. dev. of the fit |
|---|---|---|---|---|---|---|
| BGK tau=0.60 | 0.0 | 0.033332 | 2.697e-3 | 0.0809 | 0.033333 | 6.0e-5 |
| BGK tau=0.60 | 0.2 | 0.031999 | 2.482e-3 | 0.0776 | 0.033333 | 8.8e-5 |
| BGK tau=0.80 | 0.0 | 0.099996 | 5.464e-3 | 0.0546 | 0.100000 | 9.1e-5 |
| BGK tau=0.80 | 0.2 | 0.095997 | 5.009e-3 | 0.0522 | 0.100000 | 8.6e-5 |
| BGK tau=1.00 | 0.0 | 0.166669 | -9.31e-5 | -0.0006 | 0.166667 | 3.9e-5 |
| BGK tau=1.00 | 0.2 | 0.160002 | -2.63e-5 | -0.0002 | 0.166667 | 3.1e-5 |

Worst deviation over all curves: **9.06e-5** (criterion `< 10%`); the `Ma = 0`
intercepts reproduce the Chapman–Enskog viscosity to **3.7e-5** relative
(criterion `< 2%`). The `theta^2` coefficient is positive for `tau < 1` and
vanishes at `tau = 1` (where the BGK kinetic eigenvalues are annihilated), i.e.
the classical O(`theta^2`) truncation is quantitatively captured, not merely
bounded.

### 3.4 C4 — physical-viscosity contamination gate (pass)

Two independent effects contaminate the shear attenuation:

1. **`theta^2` (second-order) term**: `nu_num = a + b theta^2 + O(theta^4)`,
   with `b/a = 0.081 (tau=0.6)`, `0.055 (tau=0.8)`, `~0 (tau=1.0)`;
2. **`Ma^2` term**: the `theta -> 0` intercept is
   `nu_num(0, Ma) = nu_phys (1 - Ma^2 + O(Ma^4))`. A least-squares fit of the
   measured intercept shift against `Ma^2` gives slope **-1.0000004** with
   maximum absolute residual **4.7e-7** over `Ma in {0.1, 0.2, 0.3, 0.4}` and
   five `tau` values. This is a discrete-model (non-Galilean) artifact of
   lattice BGK, not a continuum effect.

Consequently the `1%` band is a *`(Ma, theta)`* region, not a `theta` region:

| Ma | tau rows with a band starting at theta -> 0 | theta_max (median / min / max) |
|---|---|---|
| 0.0 | 7 / 7 | 0.32 / 0.32 / 1.20 |
| 0.1 | 5 / 7 | 0.48 / 0.00 / 0.64 |
| 0.2 | 0 / 7 | 0 (only isolated high-theta bands, e.g. [0.64, 0.80]) |
| 0.3 | 0 / 7 | 0 (isolated bands only, e.g. [1.08, 1.16]) |
| 0.4 | 0 / 7 | none |

The derived Mach threshold from the measured law is **Ma < 0.09999998** for the
contamination to stay below 1% at `theta -> 0`. Practical reading: with
lattice-BGK D2Q9, a `theta -> 0` shear wave carries less than 1% numerical
viscosity error only for `Ma <~ 0.1` and `theta <=` 0.32 (0.36 at `tau = 0.7`,
0.44 at `tau = 0.85`, 1.2 at `tau = 1.0`). Above `Ma ~ 0.2` the `Ma^2` term
alone exceeds 1% at any wave number — the isolated high-`theta` bands in the
table are accidental cancellations between the `Ma^2` and `theta^2` terms and
must not be used.

### 3.5 C5 — BGK vs MRT (declared form passes; pointwise dominance falsified)

Fine-grid (33x33) `lambda_max`, 56 `(tau, Ma)` points:

| tau | Ma_crit BGK | Ma_crit MRT | relative change |
|---|---|---|---|
| 0.505 | 0.2565 | 0.3049 | **+18.9%** |
| 0.520 | 0.4471 | 0.4884 | **+9.3%** |
| 0.550 | 0.5789 | 0.5825 | +0.6% |
| 0.600 | 0.6306 | 0.6251 | -0.9% |
| 0.700 | 0.7307 | 0.6643 | -9.1% |
| 0.800 | 0.7320 | 0.6680 | -8.7% |
| 1.000 | 0.7320 | 0.6554 | -10.5% |

* **Declared criterion (stable-set inclusion): passes.** Exactly 1 point is MRT-only
  stable (`tau = 0.505, Ma = 0.3`: BGK `1.00351` vs MRT `1.00000`) and **0**
  points are BGK-only stable, so the MRT stable set contains the BGK one on the
  audited grid.
* **Pointwise `lambda_max` dominance: falsified.** MRT is up to
  **0.0196** *less* stable (`tau = 0.45, Ma = 0.4`, both unstable) and up to
  **0.0138** *more* stable (`tau = 0.52, Ma = 0.5`, both unstable). A search over
  16 rate sets (`s_e = s_eps in {1.0, 1.2, 1.64, 1.8}`, `s_q in {1.0, 1.2, 1.5, 1.7}`)
  found **no** set that dominates BGK pointwise; the best (`s_e = s_eps = 1.2`,
  `s_q = 1.7`) still has a worst-point penalty of **+0.0082**.
* Physical reading: for large `tau` the BGK kinetic eigenvalues are
  `1 - 1/tau -> 0`, i.e. BGK damps its own non-hydrodynamic modes *optimally*;
  over-relaxed MRT rates (`|1 - s| ~ 0.4 … 0.6`) cannot beat that. MRT's real
  benefit is concentrated near `tau ~ 0.5`, where BGK's kinetics are
  under-damped (`1 - 1/tau -> -1`): up to +18.9% critical Mach there.

---

## 4. Figures

| figure | content |
|---|---|
| `fig1_dispersion.png` | (a) `nu_num(theta)` for three `tau` with the O(`theta^2`) growth; (b) acoustic phase speed `c/cs` for the same `tau` (`c/cs = 0.9719` at `tau=0.55, theta=1.0`; `0.9768` at `tau=0.7`; `1.0162` at `tau=1.0`), i.e. the discrete dispersion relation is within 3% of `cs` up to `theta = 1`; (c) predicted vs measured shear growth for `tau=0.7`, three Mach numbers (overlay of §3.1). |
| `fig2_stability_map.png` | (a) `(tau, Ma)` phase diagram of `lambda_max` with the `lambda_max = 1` contour and the C2 probe verdicts (green circle = no divergence, red cross = diverged); (b) `lambda_max^BGK - lambda_max^MRT` map (§3.5). |
| `fig3_c1_parity.png` | measured vs predicted growth over all 70 C1 points (protocols A and B), plus the histogram of `log10` relative errors with the 5% gate. |
| `fig4_viscosity.png` | `nu_num(theta)` with the `a + b theta^2` fits; and `|nu_num - nu_phys|/nu_phys` vs `theta` (median over `tau`) with the 1% gate, showing the `Ma^2` plateau at small `theta`. |
| `fig5_bgk_vs_mrt.png` | `lambda_max(tau)` for BGK/MRT and the `Ma_crit(tau)` bars (§3.5). |

---

## 5. Pitfalls, limitations, scope of validity

1. **Kinetic branches masquerade as the shear branch.** For BGK, the
   transverse kinetic eigenvalues have exactly `mass = 0`, `perp-momentum = 1`
   and are therefore indistinguishable from the shear mode by moment content.
   Naming them by overlap with the `k -> 0` direction is correct for
   `tau <~ 0.9` but jumps branches at `tau = 1` (degenerate kinetic subspace).
   All `nu_num` curves therefore use **continuation tracking**
   (`shear_branch_curve`, overlap `>= 0.9996`); the analytic `index_of("shear")`
   helper is documented as a naming aid only.
2. **Unstable side is not reachable with a physical shear seed near `tau = 1/2`.**
   At `tau < 1/2` the transverse kinetic branch grows as `1/tau - 1`, so a
   physical shear seed is overtaken and the single-mode window collapses
   (`§3.1` supplement: the two `tau = 0.45` points are correctly rejected by the
   `R^2` gate). The unstable side is audited through (a) the eigen-seeded
   unstable check (`lambda` up to 1.855, error `~1e-10`) and (b) the C2
   divergence probes.
3. **Do not fit into the round-off floor.** With an `O(1)` background, the
   modal coefficient noise is `~1e-16`; the fit window stops at a modal
   amplitude of `1e-12`. Fitting further biases the slope *upwards* (an early
   version produced a 57% error at `tau = 0.8` for this reason alone).
4. **`theta = 0` is not a perturbation mode.** The uniform mode carries the
   exact `|mu| = 1` conservation eigenvalues, so `lambda_max` never drops below
   `1 + O(1e-15)`; the verdict uses `lambda_max <= 1 + 1e-9`
   (`STABILITY_TOL`) and the C2 box-grid scan excludes `(0,0)`.
5. **Box grid vs fine grid.** The continuum stability limit is
   `max_theta lambda_max`; a `32x32` box can only realise `theta = 2 pi m/32`,
   so the C2 prediction uses the *box* spectrum (a stricter, realisable set).
   The `(tau, Ma)` phase diagram uses the fine `33x33` grid.
6. **Weak instabilities are not "divergences".** `lambda = 1.0005` grows only
   `~12x` in 5000 steps; the guarded set (`lambda >= 1.001`) is declared before
   the runs and the sub-threshold band is reported separately with its measured
   growth factor (14.0 measured vs 14.2 predicted).
7. **`Ma^2` contamination is a discrete artifact, measured on the shear branch
   only.** It does not contradict the Chapman–Enskog viscosity (the `theta -> 0`
   intercept at `Ma = 0` recovers `(tau-1/2)/3` to `3.7e-5`). No claim is made
   for other collision models or for D3Q19/D3Q27.
8. **Scope of validity (C6).** All results are for the *linear* regime
   (`|u - u0| < 1e-3`, seed `1e-6`), single-phase isothermal D2Q9 with
   periodic boundaries, no forcing, no body force, no boundaries, lattice units.
   `tau >= 0.45` and `Ma <= 0.8` only; the D3Q19/D3Q27 lattices, TRT/cumulant
   collisions, and non-uniform base states (shear layers, walls) are **not**
   covered. The acoustic-branch label heuristic can mislabel the slower
   acoustic branch at `tau = 1` in the second half of the Brillouin zone; the
   audited C1 acoustic points (`tau = 0.6, 0.8`) are unaffected.

---

## 6. Error judgement (per criterion, against the predeclared threshold)

| criterion | quantity | threshold | measured | verdict |
|---|---|---|---|---|
| C1 | growth-rate relative error, 70 points | `< 5%`, `>= 24` points | max **5.52e-6**, median 3.46e-7, 70/70 | **PASS** (error is `~4 orders` better than required) |
| C1 | phase-speed relative error, 52 acoustic points | (supporting quantity) | max **2.26e-7** | PASS |
| C2 | TPR on guarded unstable points | `>= 90%` | **100%** (30/30) | **PASS** |
| C2 | FPR on stable points | `<= 10%` | **0%** (0/34) | **PASS** |
| C3 | max relative deviation of `a + b theta^2` fit | `< 10%` | **9.06e-5** | **PASS** |
| C3 | `theta -> 0` intercept vs `(tau-1/2)/3` (`Ma=0`) | `< 2%` | **3.72e-5** | **PASS** |
| C4 | 1% band table landed; band exists at `Ma = 0` | required | 7/7 rows at `Ma = 0`; band width 0.32 … 1.20 | **PASS** |
| C4 | `Ma^2` law of the contamination | (supporting quantity) | slope `-1.0000004`, residual 4.7e-7 | quantified |
| C5 | MRT stable set contains BGK stable set | declared form | 1 MRT-only, 0 BGK-only | **PASS** |
| C5 | pointwise `lambda_max(MRT) <= lambda_max(BGK)` | parenthetical claim | **falsified**: max penalty +0.0196; no rate set among 16 dominates | **honest negative** |

**Is every key quantity within 3% of its reference?** Yes, and much better:
C1/C3 are within `1e-4` relative; C2 is a discrete count (100%/0%). The only
place where a claimed property failed is the C5 *pointwise* dominance claim,
which is reported as falsified with numbers rather than excused (the declared
set-inclusion form passes). No criterion was re-defined after seeing results;
the `PREDECLARED` block in `src/tensorlbm/stability/campaign.py` is the record.

---

## 7. Reproduction and audit chain

```
campaign run (470 s)                  -> artifacts/.../results.json  (raw, every point)
campaign run --supplement-only (0.4 s)-> artifacts/.../results_supplement.json
summarize (pure function)             -> artifacts/.../summary.json (verdicts + tables)
pytest tests/test_lbm_von_neumann_stability.py (27 passed, ~10 s)
```

Tests lock: equilibrium and both collision operators against the production
code (`tensorlbm.lattice`, `tensorlbm.solver.collide_bgk/collide_mrt`), the
analytic Jacobian against finite differences, the Chapman–Enskog viscosity
limit, `c_s` propagation, the `-Ma^2` attenuation law, `lambda_max = 1/tau - 1`
at `tau < 1/2`, the box-grid `(0,0)` exclusion, three measured-vs-predicted C1
points, the `x10` amplitude linearity control, the acoustic phase speed, the
divergence probe, and the C3/C4 helpers.

The module is importable as `tensorlbm.stability` (subpackage); it is not added
to the lazy top-level `__all__` table, so the public-API export lock
(`tests/test_public_api_lazy_exports.py`) is untouched.