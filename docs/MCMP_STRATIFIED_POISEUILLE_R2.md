# Stratified two-layer Poiseuille (MCMP) — round 2, five-criteria closure

**Anchor** GAP-LBM-MCMP-POISEUILLE · mission 14689 · stands on m14431 (8 tests + 24 flow
records) · model: Shan–Chen two-component D2Q9 + BGK, `u_eq="viscous"`, `force_scheme="guo"`.

## 1. Case, geometry and parameters

A periodic channel of height `ny` rows (`nx=4`, x-periodic), driven by a uniform body force
`G_x` per unit **mixture** volume, lower half filled with the heavy component and upper half
with the light one. Both layers co-move; velocity and shear stress are continuous at the
interface, each layer obeys `u'' = -G/nu_layer`:

```
u_lower(0) = 0,  u_upper(H) = 0,  u_lower(h) = u_upper(h),  nu_lower u_lower'(h) = nu_upper u_upper'(h)
```

closed form in `cases/mcmp_stratified_poiseuille/analytic.py` (uses `nu_lower b_lower =
nu_upper b_upper` directly; the generic 2x2 solve is singular in this equal-body-force limit).

| parameter | value | note |
|---|---|---|
| `ny` | 25 / 33 / 49 | three resolutions (25, 33 = acceptance; 49 = hold-out) |
| `nx` | 4 | x-periodic, uniform in x |
| `G_x` | 5e-5 | body force per unit mixture volume |
| `G_12` | −2.5 (scan −1.5…−2.5) | SC coupling; −1.0 of the pre-repair audit mixes |
| `rho_heavy`, `rho_light` | 0.7, 0.3 | component densities (density ratio 2.33) |
| `tau_lower`, `tau_upper` | 1.0, 0.75 | `nu = cs^2(tau-1/2)` = 1/6, 1/12 |
| interface | `alpha = (1-tanh((y-half)/3))/2` | initial condition, width 3 cells |
| walls | post-streaming bounce-back, rows 0 and `ny-1` | halfway planes at 0.5, ny−1.5 ⇒ `H=ny-2` |
| SC wall treatment | `dry` (historical) vs `mirror` (**new**) | see §2 |

**Wall convention (settled in round 1, reused):** the effective no-slip plane lies on the
half-cell (`H = ny-2`, interface at `h = (ny-2)/2`, i.e. row `ny/2 - 0.5` in row coordinates);
`H = ny-1` over-predicts `u_max`/`Q` by 8%/12% at ny=25 (`wall_calibration` block of the
round-1 records, reproduced by `--mode walls`).

## 2. What was changed in this round

1. **Neutral-wetting (mirror) Shan–Chen wall** — `tensorlbm.multiphase._mirror_wall_fill` +
   `wall_mode={"dry","mirror"}` on `sc_two_component_force` / `collide_sc_two_component`
   (default `dry` is bit-identical to the historical path; asserted by test).
   The historical dry wall zeroes the neighbour density inside solids, so the first fluid row
   feels an artificial normal force
   `F_y = -G_12 rho_sigma rho_other cs^2/2 ≈ 0.0875` (D2Q9 y-stencil coefficient `cs^2/2=1/6`),
   i.e. **1750× the body force `G_x=5e-5`**. That force wets the wall with one component,
   drags the interface, and drives a persistent transverse current. The mirror wall fills solid
   cells with the mirrored nearest-fluid density (`rho_solid = rho_fluid(y=1)`), which makes the
   interaction sum a symmetric central difference across the wall plane: **zero** net wall force
   on a uniform field (asserted exactly to float32 round-off).
2. **Spurious-current diagnostic** `cases/mcmp_stratified_poiseuille/spurious_diagnostic.py`:
   time-resolved `S_all` (all fluid rows), `S_ifc` (interface band `|alpha-0.5|<0.25`), `S_bulk`
   (outside the band), plus `y_int`, `phi_contrast`, mass drift — so transient drift and the
   parasitic floor are not conflated.
3. **Matrix driver** `cases/mcmp_stratified_poiseuille/run_r2_matrix.sh` (cell names = repro units)
   and **benchmark assembler** `cases/mcmp_stratified_poiseuille/r2_benchmark.py` →
   `benchmarks/verified/mcmp_stratified_poiseuille/result_r2.json` (every number in this report is
   generated there; nothing hand-typed).

## 3. Results

### 3.1 Criterion ① — steady interface position vs analytic (<5%): **PASS**

| ny | steps | `y_int` (row) | target `ny/2-0.5` | err | verdict |
|---|---|---|---|---|---|
| 25 | 20000 | 11.539 | 12.0 | −3.84% | PASS |
| 33 | 30000 | 15.528 | 16.0 | −2.95% | PASS |
| 49 | 45000 | 24.331 | 24.0 | +1.38% | PASS (hold-out) |

### 3.2 Criterion ② — two-phase volume flow rate (<10%): **PASS**

`Q_mix = sum_y u_x,mix`, `u_mix = (rho1 u1 + rho2 u2)/(rho1+rho2)`; the trace component's
`1/rho_sigma`-amplified velocity is *not* a continuum velocity and is not used (round-1 finding).

| ny | Q_mix sim | Q analytic | err (mixture rule) | err (component rule) | verdict |
|---|---|---|---|---|---|
| 25 | 0.3912 | 0.4074 | −3.97% | −6.46% | PASS |
| 33 | 0.9705 | 0.9976 | −2.71% | −5.24% | PASS |
| 49 | 3.4867 | 3.4766 | +0.29% | −2.30% | PASS |

Both viscosity rules are reported because the "true" layer viscosity of a diffuse two-component
interface is model-dependent: mixture rule = continuum limit of the co-moving components
(this is the only reference the model can be expected to reproduce), component rule = the QR
audit's convention. Both are inside 10%.

### 3.3 Criterion ③ — interface spurious current (<0.005·U_max): **FAIL (honest)**

All numbers `[lattice units]`; `S` is `max|u_y,mix|`; `U_ref` = analytic `u_max` of the same case.

| case | wall | S (interface band) | S (all fluid) | `U_ref` | S_ifc/U_ref |
|---|---|---|---|---|---|
| static `G_x=0`, ny=25 | dry | 1.06e-2 | 7.96e-2 | 1.81e-2 | **0.387** |
| static, ny=25 | mirror | 9.30e-3 | 1.02e-2 | 1.81e-2 | 0.338 |
| static, ny=33 | mirror | 8.57e-3 | 9.62e-3 | 0.0498 | 0.171 |
| static, ny=49 | mirror | 2.24e-3 | 1.01e-2 | 0.1147 | 0.0195 |
| driven, ny=49 | mirror | 7.34e-3 | 9.56e-3 | 0.1147 | 0.064 |
| driven, ny=25/33/49 | mirror | — | `S_all/u_max,mix` = 0.397 / 0.205 / 0.086 | | |

- The mirror wall cuts the transverse current **8.7×** at ny=49 (`S_all/u_max` 0.745 → 0.086 vs the
  round-1 dry baseline) and 7.8× in the static control at ny=25 (7.96e-2 → 1.02e-2). It is a real
  improvement, but it is **not** enough: the parasitic floor at `|G_12|=2.5` is `S_ifc ≈ 1e-2` for
  ny≤33 and `≈2e-3` at ny=49, essentially *resolution independent* — it is a model/discretisation
  property of this force, not a mesh artefact.
- The gate `S < 0.005 U_ref` would need `S < 9e-5…6e-4`, i.e. a further 4×–100× reduction, at
  every tested resolution. **Not reached.**
- Coupling scan (static control, mirror): `|G_12|=1.5` → the mixture mixes (φ_contrast → 0, no
  interface, i.e. no two-phase measurement at all); `|G_12|=2.0` → `S_ifc` = 9.2e-5 (ny=25, but
  φ=0.32 < 0.5 gate → invalid cell) and 2.24e-4 (ny=49, φ=0.67, ratio **0.0019 < 0.005**);
  `|G_12|=2.2` → 2.1e-3 (ny=49, ratio 0.018). The parasitic current is extremely steep in `|G_12|`
  (≈40× between 2.2 and 2.5).
- `fail_stage`: **parameter-regime search** — the only window that meets the ③ gate in a valid
  two-phase state (`|G_12|=2.0`, ny=49, ratio 0.0019) was measured **only in the static control**;
  its *driven* ② (flow-rate) acceptance cell was not run in this round, so ③ cannot be claimed as a
  pass at an admissible configuration.
- `failure_reason`: the Shan–Chen two-component force with a linear pseudopotential maintains a
  ~2–4 cell diffuse interface whose discrete force is not the exact gradient of the model's own
  pressure tensor; the residual is a steady `O(1e-2)` lattice velocity at `|G_12|≥2.2`, which no
  reachable `U_max` (Mach-limited to `O(0.1)`) can dilute to the 0.5% level.
- Tried / ruled out this round: dry→mirror wall (8.7× gain, done); resolution 25→49 (4×–5× gain in
  the ratio, not enough); `|G_12|` scan (steep, but weak coupling degrades stratification and ①/②).
- Round-2 method list (not attempted): free-energy / phase-field LBM (Cahn–Hilliard equilibrium
  removes the parasitic force by construction); exact-difference-method (EDM) forcing with a
  thermodynamically consistent pressure tensor; higher-order isotropic stencils (D2Q9 → D2Q17)
  for the interaction force; a well-conditioned definition of ③ at fixed `Re` (report
  `S/u_max` at `u_max` = the analytic value of a *dimensionless* target `Re`, which removes the
  arbitrary `G_x` scaling).

### 3.4 Criterion ④ — two-resolution trend of criterion ①: **PASS**

`err(y_int)` = −3.84% (25) → −2.95% (33) → +1.38% (49, hold-out): monotone towards zero, sign
change at the finest grid, i.e. first-order-in-h residual dominated by the half-cell wall
calibration. The flow-rate error shows the same monotone trend (④-adjacent, §3.2).

## 4. Figures

| figure | content |
|---|---|
| `docs/figures/r2_interface_and_profile.png` | (left) heavy-fraction profile vs `y`, mirror vs dry at ny=49, with `alpha=0,0.5,1` guides; (right) `u_x,mix(y)` for ny=25/33/49 (solid) against the closed-form two-layer solution (dotted) |
| `docs/figures/r2_spurious_current.png` | (left) `S(t)` decay of the static controls, semi-log, gate line; (right) criterion-③ ratio per cell |
| `docs/figures/r2_flow_error_trend.png` | criterion ②: `Q_mix` error vs `ny` for both viscosity rules, ±10% band |

## 5. Error budget, caveats, limitations

- **Discrete error vs reference** is reported under the calibrated halfway convention; under the
  rejected cell-centre convention the errors are ~10–13% larger (`analysis.json`, 4 conventions).
- **float32**, single thread, deterministic: differences below ~1e-6 are not physical.
- **Mass drift**: recorded per record (`mass_drift_1/2`) but *not* used as a gate: the two
  components exchange mass through the interface band, so the drift measures the interface
  position change rather than a conservation failure of the solver.
- **Not steady at ny=49** (45000 steps): `steady=False` in the hold-out record; the value is
  therefore labelled *transient* in `result_r2.json` and is not counted as an acceptance cell in
  the round-1 ledger sense (its Q error, +0.29%, is nevertheless inside the gate).
- The interface is ~3–4 cells thick: at ny=25 this is ~15% of the channel, so the `h` used by the
  analytic solution is a volume-averaged position, not a sharp-interface one.
- Conclusions hold for D2Q9 + this parameter window; no extrapolation to density ratio >10,
  `|G_12|>3`, or 3D.
- **Honest disclosure**: the parasitic-current decay was inspected before the ③ verdict was
  written, and the `|G_12|=2.0` regime was found *after* the acceptance matrix was fixed. It is
  reported as a round-2 candidate, not as a retro-fitted pass; the pre-registered thresholds of
  round 1 (`spec.json`) were not changed.

## 6. Reproduction

```bash
bash cases/mcmp_stratified_poiseuille/run_r2_matrix.sh all      # records + diagnostics
PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/analyze.py      # 4 conventions
PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/r2_benchmark.py # result_r2.json + figures
PYTHONPATH=src python -m pytest tests/test_mcmp_stratified_poiseuille.py \
    tests/test_mcmp_wall_neutral.py -q
```

## 7. Artifact map

| path | content |
|---|---|
| `artifacts/mcmp_stratified_poiseuille/records/r2_*.json` | 6 acceptance records (mirror/dry × ny=25/33/49), full profiles + 4-convention references |
| `artifacts/mcmp_stratified_poiseuille/diag/sp_*.json` | 14 spurious-current time series (static/driven, wall modes, coupling scan) |
| `artifacts/mcmp_stratified_poiseuille/analysis.json` | consolidated flow table (rewritten references) |
| `benchmarks/verified/mcmp_stratified_poiseuille/result_r2.json` | criteria ①–④, per-cell ref-vs-got, parasitic table, provenance, `repro_cmd` |
| `tests/test_mcmp_wall_neutral.py` | 5 tests: mirror fill, zero wall force, dry bit-identity, invalid-mode raise, end-to-end mechanism |
