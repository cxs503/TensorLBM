# XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE — canonical re-packaging and completed pre-declared ladder

Anchor `XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE`. Pre-registration: `xval/XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE/predeclared.json`
(mission 17677, written before any run). This page closes the defects the m17677/m17825 board audit recorded
(F1 shadow-tree packaging, F2 runner/artifact step mismatch, F3 missing docs page + no benchmark row,
F4 unused steady-state sampling, F5 truncated ladder) and reports the measured ladder.

## 1. Case, geometry, parameters

| item | value |
|---|---|
| lattice | D2Q9, BGK/SRT, float64, CPU torch 2.7.1 |
| relaxation / viscosity | tau = 0.8, nu = 0.1000 |
| forcing | Guo 2002 body force, uniform acceleration g along x |
| walls | half-way bounce-back; wall plane at 0.5 dx outside the outermost fluid row |
| channel cases | H = 16 (ny = 32) and H = 32 (ny = 64), phi = 0.5 slab |
| porous cases | random mask phi ~= 0.66 (identity control); square grains p = 0.90 / 0.75 (informational) |
| stop rule | steady criterion max|u(n)-u(n-200)|/max|u(n)| < 1e-09, cap 4000 steps |
| analytic anchor | plane Poiseuille `k_super = (H/ny) * H^2/12` (intrinsic `H^2/12` x slab porosity) |

## 2. Method

* Canonical implementation: `src/tensorlbm/porous/` (`FluidNodeTable`, `build_fluid_node_table`,
  `compressed_step`, `dense_step`, `permeability`) — ported out of the shadow tree
  `TensorLBM/src/tensorlbm/porous` (which was not importable from the repository root).
* Tests: `tests/test_porous_fluid_node_table.py` (12 tests, all passing in the canonical tree).
* Runner: `xval/XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE/run_xval.py`, every number in this
  page and in `results.json` is emitted by that script (no hand copying).
* Arms: A1 = compressed push streaming with the `(K,8)` neighbour table and in-place half-way
  bounce-back; A2 = dense pull streaming (`torch.roll`) + solid-mask bounce-back, identical
  collision and forcing; A3 = analytic closed form.

## 3. Results

| case | k_super (LBM) | k_super (analytic) | err % | steps | converged |
|---|---|---|---|---|---|
| C2a channel H=16 | 10.655000 | 10.666667 | -0.1094 | 5600 | True |
| C2b channel H=32 | 42.654604 | 42.666667 | -0.0283 | 12000 | False |

Profile error against the analytic parabola: H=16 `0.204%`
(u_max 0.031810 vs 0.031875),
H=32 `0.052%`.

Representation identity (C1, random porous mask, phi ~= 0.66):
`ux_maxabs_rel = 1.765e-12`, `ux_l2_rel = 8.511e-13`,
`k_rel_diff = 1.490e-13` (pre-declared bound 1e-8).

Forcing linearity (C3): admitted pair g = 5e-5 / 1e-4 -> `|dk|/k = 3.836e-15`,
`Ma_max = 0.0551`. The g = 2e-4 arm is reported but **not admitted**
(`Ma = 0.1101` > 0.1 pre-declared stability rule), even though its k deviation is
`5.504e-15`.

Porosity sensitivity (C4, informational — no independent reference claimed):
* p = 0.90: mask porosity 0.938, k_super = 1.15150, k_intrinsic = 1.22827, footprint(total) compressed/dense = 0.541
* p = 0.75: mask porosity 0.750, k_super = 0.25685, k_intrinsic = 0.34246, footprint(total) compressed/dense = 0.676

## 4. Figures

`figures/xval_compressed_node_table.png` — (a) velocity profile vs analytic parabola for both
resolutions; (b) G1 identity residuals against the pre-declared 1e-8 bound; (c) analytic-anchor error
versus channel width with the 3 % / 15 % bands.

## 5. Caveats / limitations

* The external OSS anchor (openLBMPM, CUDA) is still **not executed** — this environment has no GPU.
  The transplant is verified as *physics-neutral against the in-house dense control* and *against an
  analytic solution*, not against the OSS binary; that comparison remains WITHHELD.
* C2b hit the 4000-step cap without tripping the 1e-9 steady flag, although its KPI
  (k error -0.0283 %, profile maxrel 0.052 %) is already
  far inside the bands and its resolution delta against H=16 is `8.11e-04`.
  The flag is reported as `false`, not retuned.
* The int64 neighbour table is what the OSS recipe uses; at p = 0.90 the compressed footprint is
  `0.54x` the dense one (a **negative** result for the "saves memory"
  motivation at high porosity, kept verbatim).
* No Shan-Chen multiphase arm here: the transplant's wetting term is out of scope for this ladder.

## 6. Error verdict

| KPI | baseline | error | < 3 %? |
|---|---|---|---|
| k_super, H=16 | analytic `(H/ny)H^2/12` | 0.1094 % | **yes** |
| k_super, H=32 | analytic `(H/ny)H^2/12` | 0.0283 % | **yes** |
| u_x profile max rel, H=16 | analytic parabola | 0.204 % | **yes** |
| u_x profile max rel, H=32 | analytic parabola | 0.052 % | **yes** |
| A1 vs A2 identity | dense control | 1.77e-12 (rel) | **yes** (bound 1e-8) |
| porosity p = 0.90 / 0.75 k | none | n/a — no baseline, informational only | not applicable |
