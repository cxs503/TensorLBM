# XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION — error budget

Anchor: `XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION` (je-santos/MPLBM-UT `huang2007-contact-angle-inversion`: calibrate θ(f_w), then invert).

Subject under test: `src/tensorlbm/contact_angle_inversion/` — the m18090 deliverable, re-examined by m22445.

Round: **error budget** — is the residual error Huang-2007 physics / resolution transfer, or is it the angle *measurement* and the node spacing?

## ① What is being audited

| round | claim | status after this round |
|---|---|---|
| m18090 | calibrated inversion reaches 1.1–2.6 % at one configuration | reproducible, but the quoted error is a *total*, not a budget |
| m22445 | the residual loss on the production grid is **resolution transfer** | **contradicted**: two thirds of the same error is already present with no transfer |
| m22445 | the coarse θ(f_w) curve is non-monotone, so `invert()` is multi-valued | **re-diagnosed**: the curve is *insensitive* (identical masks for f_w = 0.60/0.62/0.64), not folded |

## ② Arm S — the measurement operator against analytic ground truth

`tensorlbm.contact_angle_inversion.synthetic.sessile_cap_field` renders a sessile cap whose interface is an exact circular arc meeting the wall at a prescribed angle, with the same diffuse `tanh` profile an SC interface has. The mask boundary of that field *is* the arc, so whatever `contact_angle_from_mask` returns differs from the truth by estimator error only — no LBM, no physics.

Cases: 375 (θ ∈ 30–150°, R ∈ [6.0, 8.0, 12.0, 15.0, 20.0], 3 sub-lattice phases); measured 255, rejected 120.

| cap radius R (cells) | bias at θ=90° (deg) | spread over phases (deg) | measurable θ band |
|---|---|---|---|
| 6 | not measurable | — | 95–150° |
| 8 | +3.05 | 4.13 | 80–150° |
| 12 | +1.89 | 3.59 | 65–150° |
| 15 | +1.34 | 3.08 | 60–150° |
| 20 | +1.23 | 1.29 | 50–150° |

Worst case: θ_true = 150°, R = 6, bias = +19.31° (12.87 %). Max |bias| over all measurable cases = **12.87 %** → inside the 15 % gate: **True**; inside the 3 % band: **False**.

Two consequences that matter more than the floor itself:

1. the bias is **θ-dependent and R-dependent** — +3.05° at R=8 versus +1.34° at R=15 for the same 90°·cap, so a coarse calibration carries a different offset than the production grid it is applied to;

2. the spread over sub-lattice phase (grid alignment) is up to 9.6° at R=8 and 4.3° at R=15: that is the repeatability of a *single* measurement at the calibration resolution, and it is the same order as the errors the previous rounds attributed to physics.


## ③ Arm D — the delivered calibration curve is a staircase

| f_w (extra nodes) | measured θ (deg) |
|---|---|
| 0.58 | 101.655 |
| 0.60 | 95.156 |
| 0.62 | 95.156 |
| 0.64 | 95.156 |
| 0.66 | 83.064 |
| 0.68 | 78.830 |
| 0.70 | 75.814 |
| 0.72 | nan |

The cause is visible in the masks themselves: only the apex row shrinks (15 → 5 pixels of liquid) while the seven rows the fit uses keep the identical half-widths 7, 7, 7, 7, 6, 5, 4 cells, and the apex row is excluded from the fit. The angle therefore steps in 6–12° jumps as the contact line crosses a lattice cell. This is also the true origin of the "non-monotone calibration curve" of m22445: θ = 95.16° at f_w = 0.60 and 95.86° at f_w = 0.65 are two plateaus a step apart, not a physical fold.


## ④ Error budget at θ_target = 90°

| case | f_w | θ_measured | error | error % |
|---|---|---|---|---|
| coarse grid, own calibration (no transfer) | 0.6647 | 83.06 | +6.94 | 7.71 % |
| production grid, same f_w (transfer) | 0.6647 | 80.07 | +9.93 | 11.03 % |

Total error on the production grid: **9.93°**. In-config term 6.94° (70 %), residual transfer term 2.99° (30 %).

Of the residual transfer term, 1.72° is already accounted for by the measured estimator-bias difference between the two grids (bias +3.05° at R=8 vs +1.34° at R=15). What remains (1.28°) is the only part that can be attributed to the change of the θ(f_w) map itself, i.e. to Huang-2007-type resolution dependence.

| curve / method | nodes held out | max LOO error | max LOO error in f_w∈[0.55,0.72] |
|---|---|---|---|
| delivered linear | 7 | 10.76 % | 10.76 % |
| delivered pchip | 7 | 11.65 % | 11.65 % |
| refined linear | 12 | 8.59 % | 8.59 % |
| refined pchip | 12 | 10.70 % | 10.70 % |

Leave-one-out means: drop one calibration node, rebuild the curve and forecast that node's angle (forward map only — never ``forward(invert())``, which is an identity). It needs no LBM re-run, so it isolates the interpolation term.

Node refinement is what closes the gap, and the mechanism is worth stating precisely: with the extra nodes the target 90° falls inside the *steep* segment f_w ∈ [0.64, 0.66] (95.16° → 83.06°), where cos θ is nearly linear and the inversion lands at f_w = 0.655. With the delivered 7 nodes it falls into the 0.60–0.65 plateau, whose chord is almost flat, so the returned f_w = 0.6647 is far to the right of where 90° actually lives. The plateau is therefore not benign: it is exactly the region where a sparse sweep returns a wrong wall fraction.


## ⑤ Arm E — end-to-end hold-out

| arm | θ_target | f_w | θ_measured | error | <15% ? |
|---|---|---|---|---|---|
| delivered | 90 | 0.6647 | 83.06 | 7.71 % | yes |
| delivered | 105 | 0.5734 | 105.22 | 0.21 % | yes |
| refined | 90 | 0.6550 | 90.65 | 0.72 % | yes |
| refined | 105 | 0.5717 | 105.22 | 0.21 % | yes |
| refined_linear | 90 | 0.6546 | 90.65 | 0.72 % | yes |
| refined_linear | 105 | 0.5717 | 105.22 | 0.21 % | yes |

Best end-to-end row: 0.21 %. Rows inside the 15 % gate: 6/6; inside 3 %: 5/6.


## ⑥ Explicit error verdict (vs the owner's 15 % / the 3 % question)

| key quantity | reference | error | < 15 % ? | < 3 % ? |
|---|---|---|---|---|
| estimator bias, R=8, θ=90° | analytic cap | 3.05° (3.39 %) | yes | **no** |
| estimator worst case | analytic cap | 12.87 % | yes | no |
| head-line inversion, θ=90, production grid | target angle | 11.03 % | yes | **no** |
| in-config term only | target angle | 7.71 % | yes | **no** |

**Answer on the 3 % question: no.** No configuration measured in this round reaches 3 % in a way that is reproducible: the estimator alone carries +3.05° (3.4 %) at R=8 and its phase spread reaches 4.3–9.6°, so a 3 % claim at that resolution would be inside the measurement noise. The 1.09–2.57 % of m18090 was one favourable draw on one configuration.


## ⑦ Notes, traps, validity range

- The synthetic ground truth is only as good as the geometry model: the cap is an exact arc with a symmetric `tanh` profile. A real SC interface near the wall is not exactly that, so Arm S gives the *discretisation and grid-alignment* part of the error, not the physics part. That is exactly the part the budget needed.

- `f_w = 0.72` on the coarse grid returns no angle at all (the cap dissolves); before this round `calibrate()` labelled such a point `status="ok"` with a NaN angle, so an unmeasured point could be counted as a calibration node. Fixed.

- A sub-cell (level-set) estimator was implemented and tested against the same analytic caps. It performed **worse** at the extremes (+4.4° at θ=60°, +13.3° at θ=140°, R=8) because the linear `rho = mid` crossing is not a better estimator of the diffuse-interface position on a 2.5-cell `tanh` width. It is kept in the tree as a falsified alternative, not used anywhere.

- Valid range of the delivered calibration: θ ≈ 75–153° on 64×32/R=8. Outside it `invert()` falls back (or returns NaN) and must not be read as accuracy.

- All LBM rows are single-run deterministic BGK/SC at τ=1.0, G=5.0, no gravity (Bo=0); nothing here says anything about hysteresis, dynamics or other viscosity ratios beyond the τ=1.2 spot-check of m22445.


## ⑧ Reproduce

```bash
cd benchmarks/xval_huang2007_contact_angle_inversion/estimator_budget
python run_xval.py synthetic --threads 4   # analytic caps, ~6 s
python run_xval.py dense     --threads 4   # 8 nodes, 64x32, ~104 s
python run_xval.py fine      --threads 4   # production grid: 7 nodes + 6 eval rows
python run_xval.py budget                   # leave-one-out + budget
python run_xval.py report                   # REPORT.md + figures
cd ../../..
pytest tests/test_contact_angle_estimator_budget.py -q
```

## Figures

**fig1_estimator_bias.png** — Estimator bias of contact_angle_from_mask against analytic caps.

![fig1_estimator_bias.png](figures/fig1_estimator_bias.png)

**fig2_calibration_staircase.png** — Measured theta(f_w) on the calibration grid and the production grid.

![fig2_calibration_staircase.png](figures/fig2_calibration_staircase.png)

**fig3_budget_theta90.png** — Error budget at theta_target = 90 deg.

![fig3_budget_theta90.png](figures/fig3_budget_theta90.png)

**fig4_leave_one_out.png** — Leave-one-out interpolation error of the calibration curve.

![fig4_leave_one_out.png](figures/fig4_leave_one_out.png)

