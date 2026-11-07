# IN45-N07 — Air-lubrication drag reduction: hull-relative bubble curtain, Madavan-shape verification, air-flow optimum and hull curvature

*Repository:* TensorLBM · *branch:* `agent/m5688` · *new module:*
`src/tensorlbm/air_lubrication_frontier.py` · *tests:*
`tests/test_air_lubrication_frontier.py` (14 tests) · *driver:*
`experiments/air_lubrication_in45_n07/run_campaign.py` · *cluster scan:*
`experiments/air_lubrication_in45_n07/re_scan_worker.py` on loong64 (loongarch64).
Every number below is injected from `artifacts/in45_n07/campaign.json` and
`artifacts/in45_n07/re_scan.json`.

## 1 Case, geometry and parameters

Air lubrication (PAL) replaces part of the water in the near-wall layer of a ship
bottom by air. The engineering question is quantitative: **how much friction drag
does the bubble curtain remove, how thick is the curtain, what air flow pays for
itself, and how does hull curvature / injection position change the retention?**

| quantity | value | source |
|---|---|---|
| ship speed `U` | 5.14 m/s (10 kn) | operating point |
| boundary-layer thickness `delta` | 0.1 m | flat plate at the service Re |
| wetted area `S` | 5440 m^2 (100 m x 20 m x 6 m hull) | geometry |
| ship Re / ITTC-57 `Cf0` | 1.666e-03 | ITTC-57 |
| `lambda = mu_air/mu_water` | 0.0180 | 1.8e-5 / 1.0e-3 Pa s |
| injection depth / compressor efficiency | 6 m / 0.60 | PAL design point |

**Two-phase treatment.** A BGK lattice cannot carry the true viscosity ratio
(`tau -> 1/2`), so the air phase is resolved by a conservative drift-flux
transport of the void fraction `alpha(x,y)` in the boundary layer and the carrier
by a stratified-viscosity D2Q9 solver (verified separately in `air_lubrication`).
`lambda_lbm = 0.05` is the lattice-realisable ratio used for the C1/C7 lattice
cross-checks; the true `lambda` is used in the exact sandwich law.

**Hull-relative buoyancy (the defect this node fixes).** A ship bottom is a
*ceiling*: buoyancy presses a bubble onto the hull, so in the transport frame
(hull at `y = 0`, water at `y > 0`) the buoyant drift is `-V_b`. The earlier
P21-N05 campaign of this repository fed `+V_b` (away from the hull) and therefore
reported a non-wall-peaked void fraction with the Madavan comparison WITHHELD.
The test suite pins the correct sign with a negative control: reversed drift
drops `alpha_wall` by more than 3x and the inner-layer capture by 0.2 for
300 um bubbles.

## 2 How it was done

1. **Curtain transport.** `d alpha/dt + div(alpha u + alpha V_drift - D grad alpha) = 0`
   on a wall-normal domain of `2 delta` over a fetch of `L = 1000 delta`
   (dx = 10 delta cells, dy = delta/24), hull injection at superficial velocity
   `q_air`, wall-directed drift `V_b` from the Schiller-Naumann rise law
   (valid to `Re_b ~ 800`), wall-blocked turbulent dispersion
   `D = kappa u_tau y (1-y/delta)^2 / Sc_t` with `Sc_t = 1`. Because the cells are
   anisotropic, the kernel's wall-normal flux factor was corrected (see 5).
2. **Shape verification (C2).** The computed profile is fitted to the published
   Madavan et al. (1985) wall-peaked family `alpha = alpha_w exp(-y/delta_b)` with
   `delta_b/delta` in the reported 0.4-0.6 band; agreement is measured as the
   band-mean and worst-point relative deviation at matched `alpha_wall`.
3. **Drag reduction (C1).** The exact stratified half-channel solution for the
   computed `alpha(y)` gives `Delta C_f/C_f0 = 1 - G/G0` exactly at fixed flow
   rate (`tau_w = rho G H`); it is compared with the PAL empirical formula
   (same law on the exponential similarity profile) and with an independent D2Q9
   lattice solve at the same `lambda`.
4. **Energy (C4).** `P_saved = Delta C_f Cf0 0.5 rho U^3 S` against the isothermal
   compressor shaft power `P_air = (Q/eta) p_atm ln((p_atm + rho g h)/p_atm)`
   with `Q = q_air S`; the optimum of `P_saved - P_air` over `q_air` is located
   by a sweep plus a parabolic refinement in `log q_air`.
5. **Curvature (C5).** A light bubble in a curved streamline migrates toward the
   centre of curvature with `u_drift = u^2 kappa d_b^2 / (18 nu)`; on a
   concave-up bottom (`kappa = 1/R > 0`) this is *away* from the hull. Capture
   efficiency is the fraction of the air carried in the inner half of the
   boundary layer at the measurement window (`x = 0.85-0.95 L`).
6. **Re scan (C7).** 3 Re x 5 `alpha_wall` = 15 lattice cases on the loong64
   cluster (`Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38`, python 3.11.6),
   each with its own no-air baseline at the same Re.

## 3 Results

### 3.1 Verdicts

```json
{
  "C1_pal_formula_lt_20pct": true,
  "C1_lattice_lt_20pct": true,
  "C2_shape_lt_30pct": true,
  "C2_wall_peaked": true,
  "C2_band_any": true,
  "C3_residual_lt_1e8": true,
  "C4_net_positive": true,
  "C5_monotone_curvature": true,
  "C5_monotone_injection": true,
  "C7_available": true,
  "C7_lattice_lt_20pct": true
}
```

| criterion | target | measured | verdict |
|---|---|---|---|
| C1 PAL formula vs exact `Delta C_f` | < 20% | 4.98% | PASS |
| C1 lattice vs exact (Re=500) | < 20% | 0.27% | PASS |
| C2 shape vs Madavan family | < 30% | 9.3% (band-mean, in-band case) | PASS |
| C2 wall-peaked | all cases | True | PASS |
| C2 `delta_b/delta` in 0.4-0.6 | at least one `d_b` | 100um: 0.546 | PASS |
| C3 species-mass residual | < 1e-8 | 9.40e-15 | PASS |
| C4 net power at `Q_air_opt` | > 0 | +339.8 kW | PASS |
| C5 monotone in curvature | strict | True | PASS |
| C5 monotone in injection position | strict | True | PASS |
| C7 lattice vs exact, 15 cases | < 20% | 0.72% worst | PASS |

### 3.2 Void-fraction profile vs Madavan 1985 (C2)

`q_air = 1e-4 m/s`, fetch 1000 `delta`. `delta_b` is `int alpha dy / alpha_wall`;
the fitted value comes from the exponential family at matched `alpha_wall`.

| `d_b` [um] | `alpha_wall` | `delta_b/delta` | fitted | wall-peaked | L2 dev. | max dev. | in band |
|---|---|---|---|---|---|---|---|
| 30 | 0.0268 | 0.741 | 0.830 | True | 14.4% | 22.1% | no |
| 50 | 0.0287 | 0.696 | 0.768 | True | 13.4% | 20.6% | no |
| 100 | 0.0375 | 0.546 | 0.600 | True | 9.3% | 15.3% | yes |
| 200 | 0.0691 | 0.312 | 0.329 | True | 3.6% | 9.0% | no |
| 500 | 0.2113 | 0.116 | 0.107 | True | 3.6% | 17.8% | no |

The profile keeps a maximum at the hull for every bubble size, but the *layer
thickness* is a strong function of `d_b`: at 100 um the modelled curtain
sits at `delta_b/delta = 0.546`, inside the published band. Very
small bubbles are held in a thin near-wall layer by buoyancy, while millimetre
bubbles collapse to a sub-cell film (the model has no break-up/escape channel, so
that branch is reported WITHHELD for `Delta C_f`).

### 3.3 Drag reduction and the air-flow optimum (C1, C4)

Exact stratified `Delta C_f` of the computed curtain: `0.5270`;
the PAL empirical formula on the exponential profile gives
`0.5008` (4.98% difference).
At the reference design point (depth 6 m, eta 0.60):

| `q_air` [m/s] | `alpha_wall` | `delta_b/delta` | `Delta C_f/C_f0` | saved [kW] | compressor [kW] | net [kW] |
|---|---|---|---|---|---|---|
| 1.0e-05 | 0.0037 | 0.546 | 0.100 | 61.6 | 4.2 | +57.4 |
| 3.0e-05 | 0.0112 | 0.546 | 0.251 | 153.8 | 12.6 | +141.2 |
| 6.0e-05 | 0.0225 | 0.546 | 0.401 | 246.0 | 25.2 | +220.8 |
| 1.0e-04 | 0.0375 | 0.546 | 0.527 | 323.6 | 42.0 | +281.6 |
| 2.0e-04 | 0.0749 | 0.546 | 0.690 | 423.8 | 84.0 | +339.8 |
| 4.0e-04 | 0.1498 | 0.546 | 0.817 | 501.5 | 168.0 | +333.5 |
| 8.0e-04 | 0.2997 | 0.546 | 0.899 | 552.1 | 336.1 | +216.0 |

Optimum `q_air = 2.64e-04 m/s` (`Q_air = 1.44 m^3/s`),
net power `+339.8 kW` — a net saving.
Sensitivity of the optimum to the compressor depth / efficiency:

| depth [m] | eta | `q_opt` [m/s] | net at optimum [kW] | net > 0 |
|---|---|---|---|---|
| 6 | 0.60 | 2.0e-04 | +339.8 | yes |
| 6 | 0.75 | 4.0e-04 | +367.1 | yes |
| 4 | 0.75 | 4.0e-04 | +405.4 | yes |
| 3 | 0.75 | 4.0e-04 | +426.7 | yes |

### 3.4 Hull curvature and injection position (C5)

Capture efficiency (fraction of air inside `y < delta/2`), columns are
`x_inj/delta = 0, 50, 100, 150`:

| bottom | captures | Spearman | strictly increasing |
|---|---|---|---|
| flat | 0.661 | 0.696 | 0.777 | 0.958 | +1.00 | True |
| R=20 m | 0.650 | 0.687 | 0.771 | 0.957 | +1.00 | True |
| R=10 m | 0.639 | 0.677 | 0.765 | 0.956 | +1.00 | True |
| R=5 m | 0.615 | 0.658 | 0.752 | 0.954 | +1.00 | True |

Monotonicity in curvature at fixed injection position (flat first, then
increasing curvature):

| `x_inj/delta` | captures | strictly decreasing |
|---|---|---|
| 0 | 0.661 | 0.650 | 0.639 | 0.615 | True |
| 50 | 0.696 | 0.687 | 0.677 | 0.658 | True |
| 100 | 0.777 | 0.771 | 0.765 | 0.752 | True |
| 150 | 0.958 | 0.957 | 0.956 | 0.954 | True |

### 3.5 Re scan on the loong64 cluster (C7)

| Re | `alpha_wall` | lattice `Delta C_f` | exact | rel. error | momentum residual |
|---|---|---|---|---|---|
| 250 | 0.00 | 0.0000 | 0.0000 | 0.00% | 9.5e-11 |
| 250 | 0.05 | 0.3818 | 0.3811 | 0.18% | 1.4e-10 |
| 250 | 0.10 | 0.5527 | 0.5519 | 0.15% | 2.2e-10 |
| 250 | 0.20 | 0.7121 | 0.7112 | 0.13% | 3.5e-10 |
| 250 | 0.40 | 0.8321 | 0.8313 | 0.10% | 5.3e-10 |
| 500 | 0.00 | 0.0000 | 0.0000 | 0.00% | 1.6e-10 |
| 500 | 0.05 | 0.3821 | 0.3811 | 0.26% | 2.8e-10 |
| 500 | 0.10 | 0.5531 | 0.5519 | 0.22% | 4.6e-10 |
| 500 | 0.20 | 0.7124 | 0.7112 | 0.17% | 7.3e-10 |
| 500 | 0.40 | 0.8320 | 0.8313 | 0.09% | 1.1e-09 |
| 1000 | 0.00 | 0.0000 | 0.0000 | 0.00% | 3.9e-10 |
| 1000 | 0.05 | 0.3823 | 0.3811 | 0.32% | 5.8e-10 |
| 1000 | 0.10 | 0.5530 | 0.5519 | 0.20% | 7.6e-10 |
| 1000 | 0.20 | 0.7107 | 0.7112 | 0.07% | 1.0e-09 |
| 1000 | 0.40 | 0.8253 | 0.8313 | 0.72% | 1.4e-09 |

The no-air rows (`alpha_wall = 0`) return `Delta C_f = 0` to round-off
(3 cases), which is the built-in negative control of the scan.

### 3.6 Figures

Figures are embedded in the platform report: `alpha_profile` (void-fraction shape
vs the Madavan family), `air_optimum` (net power and `Delta C_f` vs `q_air`),
`curvature_capture` (capture vs injection position for four radii) and `re_scan`
(lattice-vs-exact parity, 15 cases).

## 4 Notes, limitations and honest gaps

1. **Void-fraction comparison is a shape/band comparison.** The digitised
   Madavan et al. (1985) point table is not available offline; only the published
   wall-peaked shape and the `delta_b/delta = 0.4-0.6` band are encoded. A
   point-by-point experiment comparison stays WITHHELD.
2. **Sandwich law is an upper bound.** `mu_mix` uses the Reuss (series) bound,
   correct for a *stratified film*. For a dispersed bubbly layer the harmonic
   average over-predicts the reduction (the Einstein-Roscoe bound would instead
   raise the viscosity); turbulence modification by bubbles is NOT modelled, so
   the bubbly-regime `Delta C_f` is theory-only.
3. **The drift-flux closure has no break-up, coalescence, or escape channel.**
   Millimetre bubbles are held at the hull and form a sub-cell film, which is
   why that branch is WITHHELD rather than claimed as drag reduction.
4. **`Sc_t = 1`** is the one closure constant; sensitivity to `Sc_t = 2-4` was
   checked during development (layer thickness changes by a few per cent, the
   monotonicity statements do not).
5. **Fixed-cell count, not grid-converged.** The curtain uses 100 x 48 cells over
   `1000 delta x 2 delta`; the layer is 12-15 cells thick for the reported
   `d_b`, so the profile is resolved but not grid-converged. A 2x refinement was
   not affordable within the node budget.
6. **Air-flow optimum is model-dependent through the retention closure.** The
   optimum is interior because `Delta C_f` saturates while the compressor power
   grows linearly, but its *value* depends on the sweep range and on the
   transport loss; the depth/efficiency sensitivity table gives the envelope.

## 5 Error assessment against references

| key quantity | reference | measured deviation | < 3%? |
|---|---|---|---|
| lattice `Delta C_f` vs exact stratified law (Re scan, 15 cases) | analytic solution of `d/dy(mu du/dy) = -G` | worst 0.72% | **yes** |
| `Delta C_f` of curtain vs PAL empirical formula | PAL limiting law on the exponential profile | 4.98% | **no** |
| species-mass budget residual | exact flux telescoping | 9.40e-15 (relative) | **yes** |
| `alpha(y)` shape vs Madavan family | published shape + band (no digitised points) | in-band case: L2 9.3%, max 15.3%; best shape overall 3.6% at 500 um | **no** — the reference is a shape family, not a digitised curve; 30% is the node target and it is met |
| `delta_b/delta` vs Madavan band | 0.4-0.6 | 0.546 at `d_b = 100` um | inside band; no 3% claim (band is a range) |
| net power at `Q_air_opt` | **no experimental benchmark available offline** | — | not applicable; declared, not claimed |

## 6 Code change adjacent to the physics

`VoidFractionTransport2D` divided the wall-normal fluxes and the wall-normal
divergence by `dx` (and used `dx` in the capacity limit), which is exact only for
square cells. The frontier case needs `dx = 10 dy`, where that factor makes the
vertical mixing 10x too fast. The kernel now takes `dy` (default `dy = dx`, so
all previous results and the 17 existing air-lubrication tests are unchanged) and
uses it in the y-fluxes, the y-divergence, the capacity limit and the mass
budget. A regression test diffuses a spike and checks the analytic second moment
`var = 2 D t`.
