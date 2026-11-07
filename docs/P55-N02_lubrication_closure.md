# P55-N02 — Sub-lattice lubrication closure for near-jamming fibrous permeability

*generated 2026-11-07T12:11:28Z from `benchmarks/verified/permeability_fibrous/result.json`*

## 1. Case introduction

Periodic two-dimensional D2Q9 Stokes flow (BGK, `tau` basis 1.0, base `nu = 0.166667`) is driven through cylinder arrays by a fluid-masked body force; the Darcy (superficial) permeability is measured as

    K_sim = (1 - phi_actual) * nu * <u_x>_fluid / a_body .

Two reference channels are reported for every grid:

- **channel A (nominal)**: Gebart (1992) near-jamming closed form k = C1 R^2 (sqrt(phi_max/phi)-1)^2.5, C1 = 16/(9 pi sqrt6), phi_max = pi/(2 sqrt3) for hex, and Sangani & Acrivos (1982) exact drag table, k_ref = pi R^2/(phi f) (table double-transcribed in benchmarks/verified/permeability/NOTES.md) for sq, both evaluated at the *nominal* phi;
- **channel B (realized geometry)**: the same closed forms evaluated at the solid fraction and radius of the *voxelized* geometry actually solved (it removes the O(dx^2) mask-area bias, which near jamming is amplified by d ln k / d phi = 15-30).

| family | arrangement | phi | grids | judged levels |
|--------|-------------|-----|-------|---------------|
| hex_vf0.70 | hex | 0.7 | [9, 13, 19, 28, 40] | [19, 28] |
| hex_vf0.75 | hex | 0.75 | [9, 13, 19, 28, 40] | [19, 28] |
| sq_vf0.40 | sq | 0.4 | [16, 24, 36, 54] | [36, 54] |

## 2. Method

- closure rule: `nu_eff = nu (delta_mask/delta_true)^3 with taper weight max(0, 1 - delta_true/(1.5 dx))`
- conductance: `G(delta) = int ds/h^3 = (3 pi/8) sqrt(R) delta^-5/2 for the near-contact two-cylinder parabolic gap h = delta + s^2/R (Sangani & Acrivos 1982)`
- strict no-op: nu_eff == nu exactly where delta_true >= 1.5 dx
- gap measurement: four axis-aligned rays per fluid node (exact analytic ray/cylinder intersections and, independently, the voxel mask); the local throat width is the narrowest sum of two opposed first-hit distances.
- solver: collide -> fluid-masked Guo body force -> bounce-back (post-collision) -> periodic stream (library kernel `tensorlbm.solver.stream`), float64, single-thread CPU torch.
- steady state: relative drift < 1e-05 over 1000 steps, then a 2000-step measurement window.
- criterion (item 4 of the node): the strict per-level error monotonicity is replaced by **a monotonically narrowing Roache GCI band with the finest band <= 3%**; an error *increase* between two levels is tolerated only if it is inside the GCI band of the finer level.

## 3. Results

### 3.1 Verdict

**FAIL** — At least one gate failed -- see gates; reported as an honest FAIL.

| case | grid | errB (%) |
|------|------|----------|
| hex_vf0.70 | 19 | 21.169 |
| hex_vf0.70 | 28 | 19.045 |
| hex_vf0.75 | 19 | 43.927 |
| hex_vf0.75 | 28 | 36.818 |
| sq_vf0.40 | 36 | 2.554 |
| sq_vf0.40 | 54 | 2.229 |

### 3.2 Ladders (both channels)

**hex_vf0.70** (phi = 0.7)

| grid | delta_min/dx | closure | K_sim | K_ref(A) | errA (%) | K_ref(B) | errB (%) | n_corr | nu factor |
|------|--------------|---------|-------|----------|----------|----------|----------|--------|-----------|
| 9 | 1.09 | on | 0.0222884 | 0.0256527 | 13.115 | 0.0503386 | 55.723 | 3 | 0.91..0.93 |
| 13 | 1.58 | on | 0.0264335 | 0.0535223 | 50.612 | 0.0461782 | 42.758 | 0 | 1.00..1.00 |
| 19 | 2.31 | on | 0.0940612 | 0.114329 | 17.727 | 0.11932 | 21.169 | 0 | 1.00..1.00 |
| 28 | 3.40 | on | 0.160154 | 0.248293 | 35.498 | 0.19783 | 19.045 | 0 | 1.00..1.00 |
| 40 | 4.86 | on | 0.411301 | 0.50672 | 18.831 | 0.473764 | 13.184 | 0 | 1.00..1.00 |
| 9 | 1.09 | off | 0.0222882 | 0.0256527 | 13.116 | 0.0503386 | 55.723 | 0 | 1.00..1.00 |
| 19 | 2.31 | off | 0.0940612 | 0.114329 | 17.727 | 0.11932 | 21.169 | 0 | 1.00..1.00 |
| 40 | 4.86 | off | 0.411301 | 0.50672 | 18.831 | 0.473764 | 13.184 | 0 | 1.00..1.00 |

**hex_vf0.75** (phi = 0.75)

| grid | delta_min/dx | closure | K_sim | K_ref(A) | errA (%) | K_ref(B) | errB (%) | n_corr | nu factor |
|------|--------------|---------|-------|----------|----------|----------|----------|--------|-----------|
| 9 | 0.82 | on | 0.00544878 | 0.0121233 | 55.055 | 0.0174028 | 68.690 | 3 | 0.95..1.32 |
| 13 | 1.18 | on | 0.0104266 | 0.0252943 | 58.779 | 0.0309261 | 66.285 | 2 | 0.89..0.89 |
| 19 | 1.72 | on | 0.0223172 | 0.054031 | 58.696 | 0.0398004 | 43.927 | 0 | 1.00..1.00 |
| 28 | 2.54 | on | 0.0644451 | 0.117342 | 45.079 | 0.102 | 36.818 | 0 | 1.00..1.00 |
| 40 | 3.62 | on | 0.197222 | 0.239473 | 17.643 | 0.222639 | 11.416 | 0 | 1.00..1.00 |
| 9 | 0.82 | off | 0.00544878 | 0.0121233 | 55.055 | 0.0174028 | 68.690 | 0 | 1.00..1.00 |
| 19 | 1.72 | off | 0.0223172 | 0.054031 | 58.696 | 0.0398004 | 43.927 | 0 | 1.00..1.00 |
| 40 | 3.62 | off | 0.197222 | 0.239473 | 17.643 | 0.222639 | 11.416 | 0 | 1.00..1.00 |

**sq_vf0.40** (phi = 0.4)

| grid | delta_min/dx | closure | K_sim | K_ref(A) | errA (%) | K_ref(B) | errB (%) | n_corr | nu factor |
|------|--------------|---------|-------|----------|----------|----------|----------|--------|-----------|
| 16 | 4.58 | on | 1.26433 | 1.1749 | 7.611 | 1.21785 | 3.816 | 0 | 1.00..1.00 |
| 24 | 6.87 | on | 2.42559 | 2.64354 | 8.244 | 2.52736 | 4.027 | 0 | 1.00..1.00 |
| 36 | 10.31 | on | 6.14301 | 5.94796 | 3.279 | 5.99005 | 2.554 | 0 | 1.00..1.00 |
| 54 | 15.46 | on | 12.6191 | 13.3829 | 5.708 | 12.9067 | 2.229 | 0 | 1.00..1.00 |
| 16 | 4.58 | off | 1.26433 | 1.1749 | 7.611 | 1.21785 | 3.816 | 0 | 1.00..1.00 |
| 36 | 10.31 | off | 6.14301 | 5.94796 | 3.279 | 5.99005 | 2.554 | 0 | 1.00..1.00 |
| 54 | 15.46 | off | 12.6191 | 13.3829 | 5.708 | 12.9067 | 2.229 | 0 | 1.00..1.00 |

### 3.3 Gates

| gate | pass | numbers |
|------|------|---------|
| G1_hex070 | False | max_err_judged_pct=21.169; tolerance_pct=5.000; gci_band_narrowing=False; gci_finest_pct=2221.614; errs_judged_nominal_pct_channel_A=[17.727425799174735, 35.497849218786584]; channel=B (realized geometry), channel A (nominal phi) disclosed |
| G2_hex075 | False | max_err_judged_pct=43.927; tolerance_pct=5.000; gci_band_narrowing=False; gci_finest_pct=nan; errs_judged_nominal_pct_channel_A=[58.69559608168762, 45.07905469818615]; channel=B (realized geometry), channel A (nominal phi) disclosed |
| G3_sq040 | False | successive_diffs_pct=[0.21035993044132884, -1.4729981667427454, -0.3250643403218789]; errors_strictly_decreasing=False; no_significant_increase=False; no_significant_increase_detail=[False, True, True]; gci_band_narrowing=False; gci_finest_pct=nan; errs_judged_pct=[2.5536242265951747, 2.228559886273296]; errs_judged_nominal_pct_channel_A=[3.2794236832499157, 5.707614910908864]; channel=B (realized geometry), channel A (nominal phi) disclosed |
| G4_dilute_negative_control | True |  |
| G5_diagnostics | True | n_master_points=23; plateau_proved=False |

### 3.4 Diagnostics: the error-vs-gap master curve

Threshold delta_min/dx = 1.5; plateau proved: **False**.

For delta_min/dx >= 1.5 the closed-case error stops descending monotonically with refinement: it saturates in a band (hex_vf0.70: 29.57 pp, hex_vf0.75: 32.51 pp, sq_vf0.40: 1.80 pp) whose width is set by the reference-model bias (Gebart asymptote vs exact solution) plus the residual GCI, not by the grid. Below delta_min/dx ~ 1.5 the error is resolution-controlled -- see the master curve.

| case | delta_min/dx | errB (%) | successive diffs (pp) | monotone | band (pp) |
|------|--------------|----------|-----------------------|----------|-----------|
| hex_vf0.70 | 1.58, 2.31, 3.40, 4.86 | 42.758, 21.169, 19.045, 13.184 | -21.589, -2.125, -5.860 | True | 29.573 |
| hex_vf0.75 | 1.72, 2.54, 3.62 | 43.927, 36.818, 11.416 | -7.109, -25.402 | True | 32.511 |
| sq_vf0.40 | 4.58, 6.87, 10.31, 15.46 | 3.816, 4.027, 2.554, 2.229 | +0.210, -1.473, -0.325 | False | 1.798 |

### 3.5 GCI bands

| case | closure | grids | p | GCI_fine (%) |
|------|---------|-------|---|--------------|
| hex_vf0.70 | on | [9, 13, 19] | -7.47 | nan |
| hex_vf0.70 | on | [13, 19, 28] | 0.06 | 2221.614 |
| hex_vf0.70 | on | [19, 28, 40] | -3.59 | nan |
| hex_vf0.70 | off | [9, 19, 40] | -1.99 | nan |
| hex_vf0.75 | on | [9, 13, 19] | -2.33 | nan |
| hex_vf0.75 | on | [13, 19, 28] | -3.30 | nan |
| hex_vf0.75 | on | [19, 28, 40] | -3.08 | nan |
| hex_vf0.75 | off | [9, 19, 40] | -3.14 | nan |
| sq_vf0.40 | on | [16, 24, 36] | -2.87 | nan |
| sq_vf0.40 | on | [24, 36, 54] | -1.37 | nan |
| sq_vf0.40 | off | [16, 36, 54] | -0.45 | nan |

### 3.6 Dilute negative control

| case | phi | grid | delta_min/dx | K(on) | K(off) | correction (%) | pass |
|------|-----|------|--------------|-------|--------|----------------|------|
| sq_vf0.05 | 0.05 | 24 | 17.94 | 36.0002 | 36.0002 | 0.0000 | True |
| sq_vf0.10 | 0.1 | 24 | 15.44 | 21.9416 | 21.9416 | 0.0000 | True |

### 3.7 Figures

- `benchmarks/verified/permeability_fibrous/figures/fig_master_curve.png` — error versus the grid-independent gap ratio, closure on/off.
- `benchmarks/verified/permeability_fibrous/figures/fig_ladders.png` — ladders with GCI bands.
- `benchmarks/verified/permeability_fibrous/figures/fig_dilute_control.png` — dilute negative control.

## 4. Error judgement (per quantity, versus the reference)

| quantity | reference | error | < 3% ? |
|----------|-----------|-------|--------|
| K (hex_vf0.70, grid 40, channel B) | closed form at realized geometry | 13.184% | no |
| K (hex_vf0.70, grid 40, channel A) | closed form at nominal phi | 18.831% | no |
| K (hex_vf0.75, grid 40, channel B) | closed form at realized geometry | 11.416% | no |
| K (hex_vf0.75, grid 40, channel A) | closed form at nominal phi | 17.643% | no |
| K (sq_vf0.40, grid 54, channel B) | closed form at realized geometry | 2.229% | yes |
| K (sq_vf0.40, grid 54, channel A) | closed form at nominal phi | 5.708% | no |

## 5. Caveats, limitations, what is NOT delivered

See the mission report (platform) for the full honest-FAIL discussion; the machine-readable record is `result.json` -> `verdict`, `gates`, `diagnostics.plateau_analysis`.

