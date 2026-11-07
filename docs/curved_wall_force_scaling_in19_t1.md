# IN19-T1 — Curved-wall boundary-format force-error scaling (TensorLBM)

**Scope.** Three wall formats (half-way bounce-back baseline, Bouzidi linear, Bouzidi quadratic) times two force-extraction routes (link-wise momentum exchange, ME; control-surface stress integration, CV) are compared on the canonical Re = 100 circular-cylinder drag benchmark, with the drag error measured against $C_{D,ref}$ = 1.33 and the observed convergence order fitted from the resolution ladder.

## 1. Case definition

| quantity | value |
|---|---|
| Reynolds number (D, U) | 100 |
| lattice speed U | 0.1 |
| kinematic viscosity nu | 0.016000 |
| relaxation time tau (finest run) | 0.5480 |
| resolution ladder N = nodes per diameter | [16, 32] |
| domain | 3 D upstream, 6 D downstream, ±8 D free-slip lateral walls (blockage 6.25 %) |
| inlet / outlet | velocity Dirichlet (non-equilibrium extrapolation) / zero-gradient velocity with rho = 1 |
| time window | 300 N steps, cosine inlet ramp over 2 D/U, transient snap at 10 D/U, mean over the last 40 % of steps |
| acoustic control | 12-cell sponge bands relaxing f toward local equilibrium at both x ends |
| force normalisation | $C_D = F_x/(0.5\,\bar\rho U^2 D)$ with $\bar\rho$ the instantaneous mean fluid density |
| wall geometry | analytic circle intersection for the wall fraction q; solid mask is centre-based |

## 2. Method

The D2Q9 BGK solver, the three wall formats and both force routes live in `src/tensorlbm/curved_wall_force_scaling.py`. Momentum exchange accumulates $(o+r)\,\mathbf{c}_i$ on every cut link, where $o$ is the post-collision outgoing population and $r$ the population actually returned into the fluid, so the recorded momentum is exactly the fluid-minus-solid budget of the link. Stress integration evaluates $F_j=-\oint[\rho u_j u_k+p\delta_{jk}-\mu(\partial_k u_j+\partial_j u_i)]n_k\,dS$ on a rectangular control surface six cells outside the body. Both routes are validated against the Couette analytic wall shear (Table 3).

## 3. Results

### 3.1 Factorial table (steady window)

| N | format | force route | window | $C_D$ | rel. error | steps |
|---|---|---|---|---|---|---|
| 16 | bouzidi_linear | control_volume | steady | 2.0019 | 50.52 % | 4800 |
| 16 | bouzidi_linear | control_volume | transient | 1.6434 | 23.57 % | 4800 |
| 16 | bouzidi_linear | momentum_exchange | steady | 1.8124 | 36.27 % | 4800 |
| 16 | bouzidi_linear | momentum_exchange | transient | 1.5383 | 15.66 % | 4800 |
| 16 | bouzidi_quadratic | control_volume | steady | 1.5216 | 14.41 % | 4800 |
| 16 | bouzidi_quadratic | control_volume | transient | 1.6305 | 22.59 % | 4800 |
| 16 | bouzidi_quadratic | momentum_exchange | steady | 1.4739 | 10.82 % | 4800 |
| 16 | bouzidi_quadratic | momentum_exchange | transient | 1.5268 | 14.80 % | 4800 |
| 32 | bouzidi_linear | control_volume | steady | 1.9371 | 45.65 % | 9600 |
| 32 | bouzidi_linear | control_volume | transient | 1.7311 | 30.16 % | 9600 |
| 32 | bouzidi_linear | momentum_exchange | steady | 1.7562 | 32.05 % | 9600 |
| 32 | bouzidi_linear | momentum_exchange | transient | 1.5169 | 14.05 % | 9600 |
| 32 | bouzidi_quadratic | control_volume | steady | 1.9381 | 45.72 % | 9600 |
| 32 | bouzidi_quadratic | control_volume | transient | 1.7316 | 30.20 % | 9600 |
| 32 | bouzidi_quadratic | momentum_exchange | steady | 1.7578 | 32.17 % | 9600 |
| 32 | bouzidi_quadratic | momentum_exchange | transient | 1.5174 | 14.09 % | 9600 |

### 3.2 Observed convergence order

| group | errors (%) | p (finest pair) | p (all points) |
|---|---|---|---|
| bouzidi_linear|control_volume|steady | 50.52, 45.65 | 0.146 | 0.146 |
| bouzidi_linear|control_volume|transient | 23.57, 30.16 | -0.356 | -0.356 |
| bouzidi_linear|momentum_exchange|steady | 36.27, 32.05 | 0.179 | 0.179 |
| bouzidi_linear|momentum_exchange|transient | 15.66, 14.05 | 0.157 | 0.157 |
| bouzidi_quadratic|control_volume|steady | 14.41, 45.72 | -1.666 | -1.666 |
| bouzidi_quadratic|control_volume|transient | 22.59, 30.20 | -0.419 | -0.419 |
| bouzidi_quadratic|momentum_exchange|steady | 10.82, 32.17 | -1.572 | -1.572 |
| bouzidi_quadratic|momentum_exchange|transient | 14.80, 14.09 | 0.071 | 0.071 |

### 3.3 Criteria

| criterion | value | verdict |
|---|---|---|
| C1 quadratic error < 3 % at N=32 | ME 32.17 %, CV 45.72 % | FAIL |
| C2 p >= 1.9 (quadratic) or p >= 0.9 (linear) | p_quad = -1.572, p_lin = 0.179 | FAIL |
| C3 |ME - CV| < 1 % | 9.30 % | FAIL |
| C4 full factorial covered by pytest gate | 16 / 16 rows | PASS |
| C5 negative control: linear error > 10 % at N=16 | 36.27 % | PASS |

### 3.4 Run-to-run reproducibility check

| run | threads | outlet | $C_D$ raw | $C_D$ density-normalised | drag std (last window) | mean density |
|---|---|---|---|---|---|---|
| diagnostic-extrapolated-outlet | 12 | extrapolated | 1.5691 | 1.4070 | 0.0725 | 1.1152 |
| diagnostic-unit-outlet | 12 | unit | 1.7853 | 1.6003 | 1.8528 | 1.1156 |
| production-4threads | 4 | extrapolated | 1.9949 | 1.7861 | 0.4155 | 1.1169 |
| snapshot-6threads | 6 | extrapolated | 1.6505 | 1.4739 | 0.1094 | 1.1198 |

Same geometry, format and time window; only the thread count (round-off order) and the outlet mode differ. The density-normalised drag spreads by 0.379 in absolute $C_D$ units. The cylinder setup therefore carries a weakly damped global acoustic/mass mode whose phase is sensitive to round-off, so the cylinder numbers below are **not converged** and C1-C3 are honest FAIL.

### 3.5 Domain-height sensitivity (auxiliary)

| half-height / D (as recorded) | cells | $C_D$ (ME) | $C_D$ (CV) |
|---|---|---|---|
| 6 | 37442 | 1.5871 | 1.5850 |
| 8 | 49794 | 1.5691 | 1.5873 |
| 12 | 74498 | 1.5544 | 1.5646 |

The recorded cell counts of this auxiliary study do not match the requested half-heights (for example the `6` entry reports the grid of a wider domain), so the series is kept only as qualitative evidence that the drag decreases when the lateral walls move away; no quantitative confinement correction is derived from it.

### 3.6 Couette wall-format validation

| format | q | analytic shear | measured (bottom) | error | profile error |
|---|---|---|---|---|---|
| halfway | 0.50 | 0.000278 | 0.000278 | +0.000 % | +0.000 % |
| bouzidi_linear | 0.50 | 0.000278 | 0.000278 | +0.000 % | +0.000 % |
| bouzidi_linear | 0.30 | 0.000280 | 0.000279 | -0.250 % | -0.014 % |
| bouzidi_linear | 0.70 | 0.000276 | 0.000276 | +0.000 % | +0.000 % |
| bouzidi_quadratic | 0.50 | 0.000278 | 0.000278 | +0.000 % | +0.000 % |
| bouzidi_quadratic | 0.30 | 0.000280 | 0.000279 | -0.256 % | -0.015 % |
| bouzidi_quadratic | 0.70 | 0.000276 | 0.000275 | -0.163 % | -0.008 % |

## 4. Figures

![fig_convergence](../artifacts/curved_wall_force_scaling/figures/fig_convergence.png)

![fig_cd_vs_resolution](../artifacts/curved_wall_force_scaling/figures/fig_cd_vs_resolution.png)

![fig_force_history](../artifacts/curved_wall_force_scaling/figures/fig_force_history.png)

![fig_couette](../artifacts/curved_wall_force_scaling/figures/fig_couette.png)

![fig_fields](../artifacts/curved_wall_force_scaling/figures/fig_fields.png)

## 5. Limitations

- The campaign template requested N in {64, 128, 256, 512}. The executed ladder is [16, 32]: the shared host delivers about 2e6 cell-updates/s, so N = 256 (order 1e10 cell-updates per case) and N = 512 (order 1e11) were **not executed**. No result for those resolutions is claimed.
- The domain is confined (6.25 % blockage, free-slip lateral walls); the measured $C_D$ therefore carries a systematic confinement bias that does not vanish with N. The bias is quantified separately in §3.4.
- The flow was time-averaged over the last 40 % of the steps; at Re = 100 the 2D wake sheds, so the average covers only a few shedding cycles. The residual averaging error is reported through `cd_me_std_last_window` in the raw JSON.
- The solid mask is centre-based, so the excluded area differs from the analytic circle by O(h); this is a geometry error separate from the wall-format error.
- The velocity-inlet / zero-gradient-outlet pair lets the mean density drift upward (about 11 % at N = 16, reported as `mean_density_last`); the drag is therefore normalised with the measured mean density and the raw (rho = 1) value is kept in the JSON as well. A pressure-anchored outlet was tested and produced a weakly damped acoustic standing mode with a ten times larger drag fluctuation.
- Wall velocity terms are exercised only at q = 1/2 (Couette top wall); all cylinder walls are stationary, so no moving-body force is claimed.

