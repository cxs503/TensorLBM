# R5 — Free-energy selection arm: criterion ③ head-to-head after the round-4 refutation

**Anchor** `XVAL-LBM-R4-COEXISTENCE-CALIBRATION-AND-FREE-ENER` ｜ **Repo** TensorLBM ｜
**Judgement** owner 15 % band (stretch 3 %) ｜ **Predecessor**
`docs/MCMP_STRATIFIED_POISEUILLE_R4.md` ｜ **Code**
`benchmarks/pending/free_energy_selection/{static_interface,selection}.py` ｜
**Tests** `tests/test_free_energy_selection_static_interface.py` (6 passed)

## 1 案例与物理背景

R4 calibrated the coexistence of the two-component Shan–Chen model (`G12 = −2.5`,
`ρ_A/ρ_B = 0.87373/0.13706`, `w_10_90 = 4.645` cells) and refuted the hypothesis that
criterion ③ — *static interface spurious current* `S_ifc < 0.005·u_max`, with
`S_ifc = max|u_y,mix|` inside the band `|α−1/2| < 1/4` and `α = (ρ₁−ρ₂)/(ρ₁+ρ₂)` — is
reachable inside the pseudopotential class: the plateau `S_ifc = 8.6–8.7e-3`
(ratio 0.077 vs gate 0.005) is a **bulk** motion (`S_bulk = S_all` in both pair-force
discretisations) driven by the wall treatment and the layer-viscosity asymmetry. R4 then
*selected* the free-energy binary LBM (Swift et al. PRL 78, 19 (1996); production kernel
`src/tensorlbm/phasefield/free_energy_pressure.py`) as the migration target — but never
measured it under the same metric. **That measurement is this round.**

Parameters of the free-energy arm (all lattice units):

| quantity | value | note |
|---|---|---|
| free energy | `f = A(−φ²/2 + φ⁴/4) + κ|∇φ|²/2` | `A = 0.1` |
| `κ` — standard arm | 0.2 | kernel default, `ξ = √(2κ/A) = 2` |
| `κ` — matched arm | 0.223456… | `ξ = 1.676` ⇒ `w_10_90 = 4.645` cells **= Shan–Chen coexistence width** |
| `τ_f`, `τ_φ`, `Γ` | 1.0, 2.0, 0.05 | `ν = 1/6`, `M = Γ(τ_φ−1/2)cs² = 0.025` |
| domain | `ny × nx = 64 × 4` (one arm `128 × 4`), `band = 32` | fully periodic, **no walls, no forcing** |
| seed | `φ = tanh(s/ξ)`, `s` = signed distance to the band edge | two smooth interfaces, 32 cells apart |
| analytic anchors | `ξ`, `w_10_90 = 2·atanh(0.8)·ξ`, `σ = (2/3)√(2κA)` | used only for comparison |
| steps / sampling | 20 000 / 100 (robustness arms 8 000) | float64 (primary) and float32 |

Pre-registered decision rules (fixed before running): **D1** `|Δw_10_90|/w < 15 %`;
**D2** `|Δσ|/σ < 15 %` (flat-interface part of prerequisite (b)); **D3**
`S_ifc/u_max < 0.005`; **D4** the free-energy `w_10_90` within 15 % of 4.645 cells, so
the head-to-head is not bought by a thinner interface. Selection = `free-energy` only if
D1–D4 hold for every arm.

## 2 方法与实现

1. `static_interface.py` — quiescent flat interface, `initial_condition` at local
   equilibrium, `collide_and_stream` (`free_energy_pressure.py`), no forcing, no walls.
   Per sample: band `S_ifc`, global `S_all`, out-of-band `S_bulk`, interface position
   from the `a = 1/2` crossing (localised to a monotone window around one edge),
   10-90 width via the R4 conversion, planar surface tension
   `σ = ½∮(P_NN − P_TT)dy` from the **measured** pressure tensor, φ/ρ mass drift.
2. **Geometry trap (recorded because it cost a diverged run):** a single monotone
   `tanh` across a periodic box leaves an infinitely sharp seam at the wrap-around; the
   run diverged in < 100 steps with `|u| ≈ 0.33`. The band configuration (two smooth
   interfaces, continuous across the seam) is stable, and
   `test_band_seed_is_periodic_consistent` pins it.
3. `selection.py` — consumes the free-energy records **and** the committed R4 record
   (`benchmarks/verified/mcmp_stratified_poiseuille/result_r4_probe.json`, sha256
   recorded in the decision JSON), evaluates D1–D4 with no hand-typed numbers, and emits
   `artifacts/free_energy_selection/selection.json`, `table.md` and the 3-panel figure.
4. Normalisers for D3: (i) the R4 analytic layered-Poiseuille `u_max = 0.11242`
   (same `G_x = 5e-5`, same `H = 47`), (ii) the free-energy model's own
   `u_max = G_x H²/(8ν) = 0.08284`. Both are reported; D3 must hold for both.

## 3 计算结果（程序化注入，`artifacts/free_energy_selection/table.md`）

| run | arm | dtype | w_10_90 (cells) | analytic | Δw | σ_band | analytic σ | Δσ | S_ifc | S_all | decay (dec/1k) | ratio (R4 u_max) | ratio (own ν) | D1/D2/D3/D4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matched | matched | float64 | 4.731 | 4.645 | +1.85% | 0.13580 | 0.14094 | -3.64% | 1.241e-09 | 5.161e-09 | 0.263 | 1.10e-08 | 1.50e-08 | P/P/P/P |
| matched_f32 | matched | float32 | 4.731 | 4.645 | +1.86% | 0.13579 | 0.14094 | -3.65% | 9.668e-07 | 9.668e-07 | 0.118 | 8.60e-06 | 1.17e-05 | P/P/P/P |
| matched_gamma050_f32 | matched | float32 | 4.731 | 4.645 | +1.85% | 0.13581 | 0.14094 | -3.64% | 1.564e-07 | 1.937e-07 | 0.390 | 1.39e-06 | 1.89e-06 | P/P/P/P |
| matched_ny128_f32 | matched | float32 | 4.732 | 4.645 | +1.87% | 0.13579 | 0.14094 | -3.65% | 8.725e-06 | 6.485e-05 | 0.186 | 7.76e-05 | 1.05e-04 | P/P/P/P |
| standard | standard | float64 | 4.430 | 4.394 | +0.80% | 0.12792 | 0.13333 | -4.06% | 1.262e-09 | 5.218e-09 | 0.264 | 1.12e-08 | 1.52e-08 | P/P/P/P |
| standard_f32 | standard | float32 | 4.430 | 4.394 | +0.82% | 0.12791 | 0.13333 | -4.07% | 3.018e-07 | 3.243e-07 | 0.144 | 2.68e-06 | 3.64e-06 | P/P/P/P |

Pseudopotential arm (committed R4 record, `sha256` in `selection.json`):

| arm | S_ifc | S_all | S_bulk | ratio vs u_max | gate |
|---|---|---|---|---|---|
| edm | 8.6235e-03 | 9.8392e-03 | 9.8392e-03 | 0.0767 | FAIL |
| standard | 8.6783e-03 | 9.6852e-03 | 9.6852e-03 | 0.0772 | FAIL |

Improvement of the interface-local residual against the best pseudopotential arm:

| run | S_ifc | best SC S_ifc | factor | decades |
|---|---|---|---|---|
| matched | 1.241e-09 | 8.6235e-03 | 6.95e+06× | 6.8 |
| matched_f32 | 9.668e-07 | 8.6235e-03 | 8.92e+03× | 4.0 |
| matched_gamma050_f32 | 1.564e-07 | 8.6235e-03 | 5.51e+04× | 4.7 |
| matched_ny128_f32 | 8.725e-06 | 8.6235e-03 | 9.88e+02× | 3.0 |
| standard | 1.262e-09 | 8.6235e-03 | 6.83e+06× | 6.8 |
| standard_f32 | 3.018e-07 | 8.6235e-03 | 2.86e+04× | 4.5 |

**Selection = free-energy** (primary run `matched_f32`: 8.9e3×, 4.0 decades). Alongside
the magnitude, the *shape* of the residual differs qualitatively: the Shan–Chen arms sit
on a non-decaying plateau (relative drift 1.0 % / 0.6 % over the last 20 % of a 30 000-step
run), whereas the free-energy residual keeps decaying at 0.12–0.39 decades per 1000 steps
and in float64 reaches the round-off floor `1.2e-9` — i.e. **there is no steady parasitic
current to plateau**. At 20 000 steps the head-to-head is 3.0 (worst arm, `ny = 128`,
float32) to 6.8 decades; every arm is inside the gate by ≥ 1.7 decades
(`ratio/gate ≥ 0.0017`).

## 4 图

`docs/figures/r5_free_energy_selection.png`
(a) `S_ifc(t)` (semi-log) for all six free-energy runs against the two Shan–Chen
plateaux and the gate `0.005·u_max`; (b) measured `w_10_90` vs the analytic
`2 atanh(0.8) ξ` vs the Shan–Chen coexistence width 4.645 cells (prerequisite (a) and
D4 at a glance); (c) criterion ③ head-to-head `S_ifc/u_max` per arm against the gate.
No field-contour panel is included on purpose: the run is quiescent by construction, so
the informative object is the residual *time series* and its ratio to the normaliser, not
a snapshot cloud. The measured interface profile, density and pressure fields are stored
per sample in the JSON artifacts.

## 5 注意点 / 局限

* **The wall/bulk failure mode of R4 is not tested here.** The free-energy arm is
  periodic and wall-free; the round-4 diagnosis (`S_bulk = S_all`, wall + viscosity
  asymmetry) cannot even be produced by this setup. What is measured is the
  *interface-local* residual, which is exactly the part criterion ③ is named after — the
  two-layer Poiseuille acceptance with walls is still open work (halfway bounce-back for
  `f`, no-flux condition for `φ`).
* **No density / viscosity contrast.** The arm runs at `ρ ≃ 1`, `τ_f = 1.0`; the
  stratification of the anchor (`ρ_A/ρ_B = 0.874/0.137`, `τ = 1.0/0.75`) is not carried
  by the free-energy model here. The D3 ratio uses analytic normalisers, not a measured
  driven free-energy flow (same convention as rounds 1–4).
* **The residual is a transient of the initialisation**, so its level depends on the
  horizon: the 8000-step robustness arms (`Γ = 0.5`, `ny = 128`) sit higher
  (`1.6e-7`, `8.7e-6`) than the 20 000-step primaries; the float32 arms hit the round-off
  floor (≈ 3e-7 ≈ 1e-6) and stop decaying. Reporting only the float64 value would
  overstate the model; both are in the table.
* **Diffuse-interface tension bias** −3.6 % / −4.1 % vs the continuum `σ`: the same sign
  and size as the 9-point isotropic-difference bias already documented for the droplet
  calibration (`docs/XVAL_LBM_FREE_ENERGY_LBM_ANALYTIC_YOUNG_LAPLACE_D.md`), i.e. it is a
  discretisation bias, not a model error.
* Prerequisite (b) of the round-4 migration plan (Laplace law over three radii) is **not**
  repeated here: it is carried by the committed droplet calibration above; this round
  only re-checks the flat-interface tension integral.

## 6 误差判定（vs 基准）

| key quantity | measured | benchmark | error | < 15 %? | < 3 %? |
|---|---|---|---|---|---|
| `w_10_90` (matched, f32/f64) | 4.731 cells | `2 atanh(0.8)·ξ = 4.645` (analytic) | **+1.85 %** | yes | yes |
| `w_10_90` (standard) | 4.430 cells | 4.394 (analytic) | **+0.82 %** | yes | yes |
| `ξ_fit` (matched) | 1.707 | 1.676 (analytic `√(2κ/A)`) | +1.85 % | yes | yes |
| `σ` (matched, flat interface) | 0.13580 | 0.14094 (analytic `(2/3)√(2κA)`) | **−3.64 %** | yes | no |
| `σ` (standard) | 0.12792 | 0.13333 (analytic) | −4.06 % | yes | no |
| `S_ifc/u_max` (free-energy, worst arm) | 7.8e-5 | gate 0.005 | −98.4 % | hequiv yes | yes |
| `S_ifc/u_max` (pseudopotential, R4) | 0.0767 | gate 0.005 | +1434 % | **no (FAIL)** | no |

**Is the key quantity < 3 %?** Width: **yes** (+0.8 to +1.9 %). Tension: **no**
(−3.6 to −4.1 %, inside the owner 15 % band, attributed to the discrete gradient
operator). Criterion ③: yes with 1.7–6.8 decades of margin for the free-energy arm; the
pseudopotential arm remains an honest FAIL and is not overwritten.

## 7 复现

```bash
PYTHONPATH=src:. python benchmarks/pending/free_energy_selection/static_interface.py --arm matched \
    --steps 20000 --out artifacts/free_energy_selection/fe_static_matched.json
PYTHONPATH=src:. python benchmarks/pending/free_energy_selection/static_interface.py --arm matched \
    --steps 20000 --float32 --out artifacts/free_energy_selection/fe_static_matched_f32.json
# … standard arm, Γ = 0.5 arm, ny = 128 arm (see artifacts/free_energy_selection/*.json)
PYTHONPATH=src:. python benchmarks/pending/free_energy_selection/selection.py     # table + figure + decision
PYTHONPATH=src:. python -m pytest tests/test_free_energy_selection_static_interface.py -q
```