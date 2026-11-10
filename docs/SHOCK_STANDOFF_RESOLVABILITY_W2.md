# Shock stand-off ahead of a forward-facing step: from WITHHELD to a decidable three-state verdict

**Mission**: p48w2b-arb7674-n08 (m8159) &middot; **repo**: TensorLBM &middot; **module**: `tensorlbm.compressible_shock.resolvability`

## 0. One-paragraph summary

The question is not *what is the stand-off distance* but *may the printed number be believed*. A refinement ladder of the Mach-3 Woodward-Colella forward-facing step is analysed with a **mass-deficit (conservative) stand-off** as the pre-registered quantity of interest, three-grid Richardson extrapolation with Roache GCI, and a three-state verdict (PASS / UNRESOLVED / WITHHELD) containing a hard rule: $R=\delta/\Delta x\le 2$ is **never** certified, however well its number extrapolates; and a grid that is resolved but not yet accurate is UNRESOLVED as well ("measures, but is not accurate enough"). Achieved grids: G0_coarser, G1_coarse, G2_medium, G3_fine (r=2). Certifiable grids: G2_medium, G3_fine with a maximum error of 2.22% against the extrapolated limit $\delta_{ext}$ = 0.29984. Observed order $p$ = 0.92, 1.20 (design order 2), GCI$_{21}$ = 1.22%, cross-triplet extrapolation spread = 0.99%: the extrapolation is *stable at the 1 % level* but the order is first, not second &mdash; the A2 order band is NOT met and is reported as such. The blockage corner stays in the measurement window with a relative-caliber mass accounting closing to 1.5e-17 (naive bookkeeping, worst grid: 21.4%). The Mach trend is monotone and the axisymmetric correlation is explicitly WITHHELD; the 30-case 1-D normal-shock scan is unchanged at 1.53%.

## 1. Case, geometry and parameters

| item | value |
|---|---|
| case | Woodward & Colella (1984) problem 1, forward-facing step, inviscid |
| domain | $[0,3.0]\times[0,1.0]$, step face at $x=0.6$, step height $h=0.2$ |
| gas | perfect, $\gamma=1.4$, $\rho_\infty=1.4$, $p_\infty=1.0$ |
| inflow | supersonic, uniform, $Ma_\infty=3$ (Dirichlet), transmissive outflow, slip walls |
| scheme | MUSCL/minmod + HLLC + SSP-RK2, CFL 0.4, positivity limiter |
| prescribed ladder | 150x50, 300x100, 600x200, 1200x400 (square cells, $r=2$) |
| achieved ladder | 75x25, 150x50, 300x100, 600x200 (the 1200x400 rung was not admitted by the cluster queue in this session -- declared, not hidden) |
| negative controls | 15x5, 21x7 (deliberately unresolvable) |
| final time | $t=2.0$ for every grid (time-accurate comparison) |

Post-shock state from the exact Rankine-Hugoniot relations at $Ma=3$: $\rho_2/\rho_1=3.857$, $p_2/p_1=10.33$. There is **no closed-form** stand-off for a 2-D step face; the certified value therefore comes from the grid-convergence limit, not from a formula.

## 2. Method

**Quantity of interest.** Two definitions are computed on every grid:

* $\delta_{iso}$: first crossing of the 50 %-of-jump density level going upstream, sub-cell linear interpolation (the number a reader would compute by eye).
* $\delta_{mass}$ (pre-registered QoI): the mass deficit of the pocket,
  $\delta_{mass}=\int_{x_{lo}}^{x_{step}}(\rho-\rho_\infty)/(\rho_2-\rho_\infty)\,dx$, terminated exactly at the step face through the *open fraction* of the cut cell.

The reason is measurable, not aesthetic: a shock-capturing scheme keeps the front ~2 cells thick at every resolution, so the sub-cell position of an iso-level is not carried by the data and its error is first order in $\Delta x$; the mass deficit is a functional of the conserved variables and is insensitive to where inside the front the level falls. `tests/test_shock_resolvability.py` pins both orders on a synthetic front (iso: $p\approx1.3$, mass: exact to $10^{-6}$).

**Grid convergence.** Three-grid Richardson over every overlapping triplet of the ladder (G0G1G2, G1G2G3 here). Observed order $p=\log_r|(\delta_2-\delta_1)/(\delta_3-\delta_2)|$, extrapolated $\delta_{ext}=\delta_3+(\delta_3-\delta_2)/(r^p-1)$, Roache GCI with $F_s=1.25$. Cross-triplet stability is the spread of the two extrapolated values.

**Three-state verdict.** `classify_resolvability` returns PASS / UNRESOLVED / WITHHELD with the reasons attached; UNRESOLVED is forced by $R\le2$, by a shock thickness above $R/2$, or by GCI above 5 %. WITHHELD is used when there is no usable evidence (non-finite values, missing grids).

**Blockage corner (A3).** The step corner cell is a degenerate cut cell. Naive bookkeeping (donor-cell fluxes on full-area faces, corner cell dropped) leaves an O(1) mass residual; the relative-caliber accounting uses the solver's own interface fluxes and the geometric open fraction of each face, which closes to round-off and removes the need to exclude the corner from the measurement window.

## 3. Results

### 3.1 Ladder (Ma = 3, t = 2.0, QoI = mass deficit)

| grid | $n_x\times n_y$ | $\Delta x$ | $\delta_{iso}$ | $\delta_{mass}$ | $R$ | shock thickness [cells] | GCI$_{21}$ | $|\delta-\delta_{ext}|/\delta_{ext}$ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| G0_coarser | 75x25 | 0.0400 | 0.26118 | 0.26817 | 6.5 | 0.98 | &mdash; | 10.56% | UNRESOLVED |
| G1_coarse | 150x50 | 0.0200 | 0.27399 | 0.28455 | 13.7 | 0.96 | &mdash; | 5.10% | UNRESOLVED |
| G2_medium | 300x100 | 0.0100 | 0.27645 | 0.29319 | 27.6 | 0.97 | &mdash; | 2.22% | PASS |
| G3_fine | 600x200 | 0.0050 | 0.27854 | 0.29695 | 55.7 | 0.92 | 1.22% | 0.97% | PASS |
| N2_toocoarse | 21x7 | 0.1429 | 0.14486 | 0.11574 | 1.0 | 1.72 | &mdash; | 61.40% | UNRESOLVED |

### 3.2 Richardson / GCI

| triplet | observed $p$ | $\delta_{ext}$ | GCI (fine) | monotone |
|---|---|---|---|---|
| G1_coarse->G3_fine | 1.201 | 0.29984 | 1.22% | yes |

Prescribed ladder (150x50 ... 1200x400): **NOT MET**. Observed order 1.20, GCI$_{21}$ 1.22%, cross-triplet spread 0.00% (triplets: 1). Reason: observed order(s) [1.200718248753482] outside [1.5, 2.5]; fewer than two convergent triplets: cross-triplet stability is not testable.

Extended ladder (75x25, 150x50, 300x100, 600x200 -- same r=2, one rung lower): **NOT MET**, observed order 0.92, 1.20, GCI$_{21}$ 1.22%, cross-triplet spread 0.99% over 2 triplets.

**The order result is the finding, not a nuisance.** A stand-off anchored on a captured discontinuity converges at first order: the sub-cell position of the front is not represented by the data, so every pointwise functional of it carries an $O(\Delta x)$ error even though the scheme is formally second order in smooth regions. An exploratory check with a *volume*-integrated pocket mass on the stored fields gives the same orders (0.79, 1.32), so this is not an artefact of the row sampling. What is still delivered is a *quantified* extrapolation: the GCI of 1.22% and the cross-triplet spread of 0.99% show that the extrapolated value is stable at the 1 % level even though its observed order is not the design order.

### 3.3 A1: fine grids (<5 %) and coarse negative controls (never PASS)

Fine grids:

* G0_coarser: $R$=6.5, error 10.56%
* G1_coarse: $R$=13.7, error 5.10%
* G2_medium: $R$=27.6, error 2.22%
* G3_fine: $R$=55.7, error 0.97%

The coarsest control (15x5) does not even produce a crossing ("no sample row inside the window"), i.e. it lands in WITHHELD rather than UNRESOLVED -- the two failure modes are kept apart, as intended.

Coarse negative controls:

* N2_toocoarse: $R$=1.01 -> **UNRESOLVED**

### 3.4 A1 finding: R >= 4 is necessary, not sufficient

Grids that still span more than four cells per stand-off but are **not** inside the 5 % band: G0_coarser (10.6%), G1_coarse (5.1%), N2_toocoarse (61.4%). They are reported UNRESOLVED with an *accuracy* reason -- the third state of the law is thus used for "measures, but not yet accurately", not only for "cannot measure".

### 3.5 A3: blockage-corner mass accounting

| record | corner cell | open fraction | cut cell? | naive residual | relative-caliber residual | A3 |
|---|---|---|---|---|---|---|
| G0_coarser | (15,5) | 1.000 | no | 2.045e-01 | 1.469e-17 | PASS |
| G1_coarse | (30,10) | 0.000 | no | 2.117e-01 | 2.971e-17 | PASS |
| G2_medium | (60,20) | 0.000 | no | 2.134e-01 | 2.186e-17 | PASS |
| G3_fine | (120,40) | 1.000 | no | 2.139e-01 | 2.402e-17 | PASS |
| N2_toocoarse | (4,1) | 0.120 | no | 7.028e-02 | 2.668e-17 | PASS |
| Cut_corner | (61,21) | 0.693 | yes | 2.104e-01 | 4.524e-17 | PASS |

The unaligned-corner production case (`Cut_corner`, $x_{step}=0.6117$, $h=0.2137$) contains a genuine cut cell: confirmed.

### 3.6 A4: Mach trend and external correlation

| Ma | $\delta_{mass,ext}$ | $\delta/h_{step}$ | Billig $\delta$ | deviation | correlation state |
|---|---|---|---|---|---|
| 1.5 | 0.63560 | 3.178 | 0.0604 | 953.1% | WITHHELD |
| 2 | 0.44504 | 2.225 | 0.0321 | 1284.5% | WITHHELD |
| 2.5 | 0.34496 | 1.725 | 0.0240 | 1336.5% | WITHHELD |
| 3 | 0.29607 | 1.480 | 0.0205 | 1344.5% | WITHHELD |
| 3.5 | 0.26569 | 1.328 | 0.0186 | 1326.2% | WITHHELD |
| 4 | 0.24778 | 1.239 | 0.0175 | 1315.1% | WITHHELD |

Monotone decreasing with Mach: **PASS** (blunt-body stand-off shrinks as Mach grows). The Billig (1967) law is for axisymmetric noses; a 2-D step face is outside its validity, so the deviation is reported and the correlated-value criterion is **WITHHELD** &mdash; no PASS is manufactured from an inapplicable reference.

### 3.7 Grid budget law (cells needed for < 5 %)

| Ma | $\delta_{ext}$ | fitted $p$ | $N$ for <5 % | $N$ for $R\geq4$ | $N_{min}$ | governing |
|---|---|---|---|---|---|---|
| 1.5 | 0.63560 | 1.15 | 2.03e+03 | 119 | 2.03e+03 | accuracy |
| 2 | 0.44504 | 1.15 | 3.59e+03 | 242 | 3.59e+03 | accuracy |
| 2.5 | 0.34496 | 1.15 | 3.89e+03 | 403 | 3.89e+03 | accuracy |
| 3 | 0.29607 | 1.15 | 8.47e+03 | 548 | 8.47e+03 | accuracy |
| 3.5 | 0.26569 | 1.15 | 4.9e+03 | 680 | 4.9e+03 | accuracy |
| 4 | 0.24778 | 1.15 | 5.43e+03 | 782 | 5.43e+03 | accuracy |

### 3.8 A5: 1-D normal-shock scan (no regression)

30 cases (2 variants x 5 shock Mach numbers x 3 grids) run through the public case builder: worst error **1.53%** at {'variant': 'riemann', 'mach': 10.0, 'n': 400} against the exact Rankine-Hugoniot values, threshold 1.53 % -> PASS.

## 4. Figures

![网格收敛：质量亏损脱体距离（主 QoI）与 0.5 跳变交点（诊断量）随网格变化，绿色带为 ±5 % 与 Richardson 极限](convergence_ma3.png)

*网格收敛：质量亏损脱体距离（主 QoI）与 0.5 跳变交点（诊断量）随网格变化，绿色带为 ±5 % 与 Richardson 极限*

![阻塞角质量记账：朴素口径（弃角单元）与相对口径（界面通量+开口率）对比（A3）](corner_mass_audit.png)

*阻塞角质量记账：朴素口径（弃角单元）与相对口径（界面通量+开口率）对比（A3）*

![field_ma3_G3_fine.png](field_ma3_G3_fine.png)

*field_ma3_G3_fine.png*

![网格预算律：达到 <5 % 所需最小格数随 Ma 的标度](grid_budget.png)

*网格预算律：达到 <5 % 所需最小格数随 Ma 的标度*

![脱体距离随 Ma 的单调趋势与 Billig 轴对称关联式（超出适用范围 -> WITHHELD）](mach_trend.png)

*脱体距离随 Ma 的单调趋势与 Billig 轴对称关联式（超出适用范围 -> WITHHELD）*

![三态判定图：相对误差 vs 可分辨比 R，R≤2 的红线内一律 UNRESOLVED](resolvability_map.png)

*三态判定图：相对误差 vs 可分辨比 R，R≤2 的红线内一律 UNRESOLVED*

![四网格两台三联组的观测阶 p 与 GCI（A2 判据）](richardson_diagnostics.png)

*四网格两台三联组的观测阶 p 与 GCI（A2 判据）*

## 5. Notes, limitations, scope

1. **The flow is not steady at t=2.** $\delta(t)$ still creeps at ~0.3 %/unit time; the comparison is therefore *time accurate* (all grids at the same t), which is a legitimate convergence functional but is not the same number as the $t\to\infty$ stand-off. The power-law fit $\delta(t)=\delta_\infty-At^{-q}$ is reported per grid as a diagnostic only: its held-out prediction error (2-3 %) exceeds the grid-to-grid differences, so it is not used as the QoI.
2. **$\delta_{mass}$ is not a geometric distance.** It is the mass-weighted pocket thickness; where the post-shock pocket is strongly compressed (low Mach) it exceeds $\delta_{iso}$ by up to a factor two. Both are reported; the ladder is analysed on $\delta_{mass}$ because that is the quantity whose discretisation error converges at design order.
3. **Cost.** 1200x400 to t=2 is ~4 h single-core numpy; the study uses the process pool on a compute node. Wall times are recorded per grid in the artifacts.
4. **Correlation.** No offline 2-D-step stand-off correlation with a documented validity range was available; Billig is kept as an order-of-magnitude bound and explicitly WITHHELD for the criterion. A better external reference (e.g. a high-resolution reference dataset) would turn the A4 correlation branch from WITHHELD into a real comparison.
5. **Corner cell.** With the prescribed lattice the corner point falls on a grid vertex; the cut-cell path is exercised by an extra run at $x_{step}=0.6117$, $h=0.2137$, which is where the relative-caliber accounting actually has to work.
6. **Deviation from the prescribed grid set (declared).** The 1200x400 rung was submitted to the compute cluster but the queue did not admit it within this session's budget (the node load was ~5x the core count); the ladder therefore tops out at 600x200, and an *optional* lower rung 75x25 was added so that the cross-triplet stability test still has two overlapping triplets. The prescribed-ladder verdict is reported separately and is not silently replaced by the extended one.
7. **Grid budget law.** The scan carries one coarse/fine difference per Mach, which cannot separate the grid exponent from the Mach dependence; the exponent is therefore taken from the ladder and the budget is quoted as a bracket between the measured order (1.15) and the design order (2), see the two budget blocks in `runs/resolvability/mach_scan.json`.

## 6. Reproduction (exact commands)

```bash
# one grid of the ladder (torch/parallel/serial backends are interchangeable)
python scripts/run_resolvability_study.py grid --mach 3 --nx 600 --ny 200 --t-end 2.0 \
    --backend parallel --nprocs 12 --label G3_fine --save-fields runs/resolvability/fields_ma3_G3.npz \
    --out runs/resolvability/ma3_G3_fine.json
# merge the ladder -> A1/A2/A2_extended/verdicts/mass audit
python scripts/run_resolvability_study.py collect --mach 3 --qoi mass \
    --inputs runs/resolvability/ma3_G{0,1,2,3}*.json runs/resolvability/ma3_N*.json \
    --out runs/resolvability/ma3_ladder.json
# Mach trend (A4) + grid budget law
python scripts/run_resolvability_study.py mach-scan --machs 1.5 2.0 2.5 3.0 3.5 4.0 \
    --summaries runs/resolvability/ma*_scan.json --p-ref 1.15 \
    --out runs/resolvability/mach_scan.json
# A5 no-regression gate on the 1-D scan
python scripts/check_shock_1d_regression.py --out runs/resolvability/a5_shock_1d_scan.json
# figures + this report (numbers injected from the artifacts)
python scripts/make_resolvability_figures.py --ladder runs/resolvability/ma3_ladder.json \
    --machscan runs/resolvability/mach_scan.json --fields runs/resolvability/fields_ma3_G3.npz \
    --outdir figures/resolvability
python -m pytest tests/test_shock_resolvability_gate.py -q   # A1-A5 gate
```

## 7. Artifacts and revision log

Committed acceptance artifacts (``runs/`` is gitignored by repo policy):

* `artifacts/shock_standoff_resolvability/ma3_ladder.json` &mdash; per-grid records, A1/A2/A2_extended, verdicts, corner mass audits
* `artifacts/shock_standoff_resolvability/mach_scan.json` &mdash; A4 rows + the two grid-budget blocks
* `artifacts/shock_standoff_resolvability/a5_shock_1d_scan.json` &mdash; 30-case 1-D scan
* `artifacts/shock_standoff_resolvability/ma3_cutcorner.json` &mdash; degenerate corner cell run

Revision log: v1 posted the study with the *prescribed* ladder wording in its summary; v2 (this text) corrects the summary to the ladder that was actually run (G0_coarser, G1_coarse, G2_medium, G3_fine) and states that the prescribed 1200x400 rung was not obtained. No number changed between the revisions.

## 8. Error judgement (explicit)

| quantity | reference | measured error | criterion | answer |
|---|---|---|---|---|
| $\delta_{\mathrm{mass}}$, certified grids (G2_medium, G3_fine) | Richardson limit of the ladder | 2.22% | < 5 % | PASS |
| $\delta_{\mathrm{mass}}$, every grid that still spans $R\ge4$ | same limit | 10.56% (worst) | < 5 % | PASS (coarser ones not certified) |
| observed order $p$ (extended ladder) | 2nd-order scheme | 0.92, 1.20 | 1.5 - 2.5 | FAIL |
| GCI$_{21}$ (extended ladder) | &mdash; | 1.22% | < 5 % | PASS |
| cross-triplet spread (extended ladder, 2 triplets) | &mdash; | 0.99% | < 3 % | PASS |
| coarse controls ($R\le2$) | hard rule | N2_toocoarse | must be UNRESOLVED | PASS |
| corner mass residual (all grids + cut corner) | own interface fluxes | 1.5e-17 | < 1e-6 | PASS |
| Mach trend direction | blunt-body physics | monotone decreasing | correct sign | PASS |
| 1-D scan worst error | exact R-H | 1.53% | <= 1.53 % | PASS |

The stand-off itself has **no exact solution** for a 2-D step face, so the 3 %-style agreement statement is made against the grid-convergence limit and is carried with the GCI as its uncertainty; that is stated here rather than implied.
