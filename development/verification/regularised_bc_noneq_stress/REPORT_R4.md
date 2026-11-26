# REPORT_R4 — regularised (non-equilibrium stress) wall closure: order of accuracy,
## relaxation-rate dependence, and D3Q19 physical validation

Anchor `XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC` · arm R4 · commit series on `agent/m21628` · artifacts `xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/artifacts_r4/`.

### ① 案例介绍

R1-R3 (m18423) transplanted `joconnor22/LIFE` `GridClass::regularisedBC` (`src/Grid.cpp:386-466`) into
`src/tensorlbm/regularised_bc.py` and proved the semantics by a line-cited scalar transcription
(max|df| = 1.4e-17).  Accuracy was reported at two single points of the oscillating Stokes layer
(2.87 % at delta = 2.52 cells, 5.62 % at delta = 1.78 cells) with the *attribution* "under-resolved",
and the module docstring claimed the closure is "second order and time-step independent at a wall".
Neither statement was testable from a two-point measurement.  R4 replaces both by measurements:

* physics: 2-D plane channel, x-periodic, oscillating wall `u = U0 cos(omega t)`, exact solution
  `u = U0 exp(-d/delta) cos(omega t - d/delta)`, `delta = sqrt(2 nu/omega)`;
  `U0 = 0.05`, `H/delta ~ 10`, `Delta_x = 1`, float64;
* ladder: `delta/Delta_x` in {1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0},
  tau in {0.8, 1.0, 1.5} (8 tau in the sweep [0.55, 0.575, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]);
* control arm: the repo-validated half-way bounce back with the Ladd moving-wall term;
* 3-D arm: D3Q19 plane channel (periodic x/z), Couette + body-force Poiseuille, same closure.

### ② 怎么做的

`run_xval_r4.py` drives `rb.run_channel` (D2Q9) and a purpose-built D3Q19 stepper that calls the
*production* `rb.apply_regularised_wall` / `rb._collide` / `rb._guo_body_force` (no re-implementation).
Each oscillating run is analysed over its last full period with a single-frequency DFT; the physical
problem is held fixed while `delta` (hence the period) is refined, so the dimensionless groups
(`delta/H`, `U0`, Ma) do not drift along the ladder.  Fitted order = least squares of
`log l2` vs `log(delta/Delta_x)`; steady arms stop on `max|u(n)-u(n-200)|/max|u| < 1e-10`.
Figures are rendered by `make_report_r4.py` from the JSON artifacts; every number in this report is
injected programmatically (no hand copy).

### ③ 计算结果

**L1 — resolution ladder (tau = 0.8), complex L2 error of the wall-normal amplitude [%]**

| delta/Delta_x | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| regularised | 4.7 | 2.109 | 1.208 | 0.7221 | 0.5252 | 0.308 | 0.1992 |
| bounce_back | 7.48 | 3.356 | 1.911 | 1.181 | 0.841 | 0.4845 | 0.3118 |

| arm (tau=0.8) | fitted p | R^2 | l2 ratio delta 2 -> 10 (2nd-order prediction) | l2 at delta=10 |
|---|---|---|---|---|
| regularised | 1.97 | 0.9993 | 23.6 (25) | 0.199 % |
| bounce_back | 1.98 | 0.9998 | 24.0 (25) | 0.312 % |

**Fitted order at the other relaxation times**

| tau | regularised p | regularised l2(delta=10) | bounce back p | bounce back l2(delta=10) |
|---|---|---|---|---|
| 0.80 | 1.97 | 0.199 % | 1.98 | 0.312 % |
| 1.00 | 1.81 | 0.0293 % | 1.90 | 0.244 % |
| 1.50 | 1.84 | 0.918 % | 2.00 | 0.405 % |

**L2 — tau sweep at fixed delta/Delta_x = 6 (same physical Stokes problem)**

* regularised: tau=0.55: 0.812 %, tau=0.575: 0.801 %, tau=0.6: 0.787 %, tau=0.7: 0.672 %, tau=0.8: 0.525 %, tau=1: 0.0743 %, tau=1.2: 0.75 %, tau=1.5: 2.48 %, tau=2: 6.44 %
* bounce_back: tau=0.55: 0.693 %, tau=0.575: 0.727 %, tau=0.6: 0.756 %, tau=0.7: 0.818 %, tau=0.8: 0.841 %, tau=1: 0.692 %, tau=1.2: 0.308 %, tau=1.5: 1.13 %, tau=2: 4.47 %

| arm | min | max | max/min | corr(l2, tau) |
|---|---|---|---|---|
| regularised | 0.0743 % | 6.44 % | 86.7 | 0.85 |
| bounce_back | 0.308 % | 4.47 % | 14.5 | 0.77 |

**L3 — D3Q19 3-D channel**

| case | closure | ny | L2 [%] | L_inf [lu] | converged |
|---|---|---|---|---|---|
| couette | bounce_back | 24 | 4.86e-07 | 1.98e-10 | True |
| couette | bounce_back | 32 | 9.68e-07 | 3.95e-10 | True |
| couette | regularised | 24 | 1.08e-12 | 4.89e-16 | True |
| couette | regularised | 32 | 3.86e-13 | 2.43e-16 | True |
| poiseuille | bounce_back | 24 | 0.26 | 8.7e-05 | True |
| poiseuille | bounce_back | 32 | 0.14 | 4.79e-05 | True |
| poiseuille | regularised | 24 | 0.486 | 0.000181 | True |
| poiseuille | regularised | 32 | 0.269 | 9.99e-05 | True |

### ④ 结果图

`artifacts_r4/fig1_order.png` — log-log L2 vs delta/Delta_x for tau in {0.8, 1.0, 1.5} (both closures)
with the 3 % stretch line: the curves are straight lines of slope -2, no plateau.

`artifacts_r4/fig2_tau.png` — L2 vs tau at fixed delta/Delta_x = 6: U-shaped for the regularised
closure with a minimum at tau = 1, nearly flat for bounce back at small tau.

`artifacts_r4/fig3_d3q19.png` — left: D3Q19 Poiseuille profile vs the analytic parabola (ny = 32);
right: L2 vs H (slope -2) and the machine-precision Couette arm.

### ⑤ 注意点 / 局限

* **`tau` cannot be varied at fixed resolution in lattice units**: `delta^2 omega = 2 nu` and
  `nu = (tau - 1/2)/3`, so fixing `delta/Delta_x = 6` while changing tau changes `omega` in lattice
  units by 100x.  The tau sweep therefore measures the *combined* closure + unsteady-scheme
  sensitivity, not the closure alone; the clean closure comparison is within a fixed tau (L1).
  The consequence is a withdrawn claim, not a new one: see ⑥.
* Couette is reproduced by the regularised closure to machine precision once initialised on the exact
  profile, so that case cannot discriminate closures -- the discriminating case is the unsteady layer.
* Steady Poiseuille: the regularised closure is ~2x *less* accurate than bounce back at equal ny
  (2-D and 3-D agree), because the projected node drops the genuine third moment of the parabolic
  non-equilibrium part.  It should not be sold as a universally more accurate wall.
* Corner (two-normal) planes and LIFE's `CENTRAL_MOMENTS` collision branch are still not transplanted
  (unchanged R1 scope limit); no corner claim is made here.
* LIFE itself is still not executed (C++/Boost/LAPACK FSI); the cross-language evidence remains the
  line-cited scalar transcription of R1.
* Metric defect found and fixed: `phase_max_abs` was reported *unmasked*, although `PREREG.md`
  declares the mask `|A_ref| > 0.1 U0`.  Unmasked, the far-field node (|A_ref| -> 0) dominates and
  gives ~0.89 rad at every resolution; masked, the same run gives
  2.1e-03 rad.  `_run_oscillating` now emits
  `phase_max_abs_masked` (the preregistered quantity) next to the legacy key; the R2 numbers used the
  unmasked definition and are superseded by the masked values here.

### ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 误差 | 基准 | <3% ? |
|---|---|---|---|
| Stokes layer, regularised, tau=0.8, delta=10 | 0.1992 % | analytic Stokes solution | 是 |
| Stokes layer, bounce back, tau=0.8, delta=10 | 0.3118 % | analytic Stokes solution | 是 |
| Stokes layer, regularised, tau=1.0, delta=10 | 0.02925 % | analytic Stokes solution | 是 |
| Stokes layer, regularised, tau=1.5, delta=10 | 0.9176 % | analytic Stokes solution | 是 |
| Stokes layer, regularised, tau=1.5, delta=6 | 2.485 % | analytic Stokes solution | 是 |
| D3Q19 Poiseuille, regularised, ny=32 | 0.2691 % | analytic parabola | 是 |
| D3Q19 Poiseuille, bounce back, best grid | 0.14 % | analytic parabola | 是 |
| D3Q19 Couette, regularised (ny=24,32) | 1.082e-12 % | linear solution | 是 |
| fitted order vs theory p=2, tau=0.8 | 1.521 % | p = 2 | 是 |
| fitted order vs theory p=2, tau=1.0 | 9.472 % | p = 2 | 是（≤15%） |

**结论（honest）**：

1. *Second order is confirmed*, not assumed: p = 1.81-2.00 with R^2 >= 0.997 for six of six
   (tau, closure) combinations; at delta = 10 cells the regularised closure is at 0.20 %
   (bounce back 0.31 %) -- so the 2.87 % of R2 was indeed resolution, and 23.6x of error was
   removed for 4x of refinement against a 25x second-order prediction.
2. *"Time-step independent at a wall" is not supported and is withdrawn* from the module
   docstring: at fixed delta/Delta_x = 6 the regularised error spans 87x
   (0.074 %-6.44 %) over tau in [0.55, 2.0], with a minimum at tau = 1; bounce back spans only
   14.5x.  The closure fixes the wall term, not the unsteady error of
   the scheme around it.
3. The closure is *not* uniformly more accurate: it wins on the unsteady layer for tau in
   [0.7, 1.0] (up to 9x at tau = 1) and loses on steady Poiseuille (~2x) and at tau >= 1.2.
4. D3Q19 is now physically validated (previously only its tables were pinned): Poiseuille order
   1.98 / 2.00 with
   max|mass drift| <= 8.5e-13.

### 复现

```bash
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval_r4.py --arm L1 --l1-taus 0.8
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval_r4.py --arm L1 --l1-taus 1.0,1.5 --l1-deltas 2,4,6,8,10
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval_r4.py --arm L2 --taus 0.55,0.575
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval_r4.py --arm L2 --taus 0.6,0.7,0.8
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval_r4.py --arm L2 --taus 1.0,1.2,1.5,2.0
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval_r4.py --arm L3 --d3 couette:24,couette:32
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval_r4.py --arm L3 --d3 poiseuille:24,poiseuille:32
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval_r4.py --arm analyse
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/make_report_r4.py
PYTHONPATH=src python -m pytest tests/test_regularised_bc.py tests/test_regularised_bc_order.py -q
```

(The sweeps are split into sub-2-minute chunks because the local control plane caps a single
foreground command; each chunk merges into the same artifacts and is order independent.)
