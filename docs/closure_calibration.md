# Solver-in-the-loop closure calibration (B3)

*data → closure → solver*: identify a Reynolds-dependent Smagorinsky closure
from drag observations by differentiating through bounded LBM rollouts, and
evaluate it at held-out Reynolds numbers. This productises the manual loop
sketched by `examples/solver_in_the_loop.py` (A6) on top of the differentiable
bounded path (A6+ velocity inlet / zero-gradient outlet), module
`tensorlbm.autograd_calib`, example `examples/closure_calibration.py`.

## API

| Object | Role |
|---|---|
| `BoxCase` | bounded calibration domain: sphere at `cx=0.3·nx` in a uniform inflow, house relation `tau = 0.5 + 3·u_in·2r/Re` |
| `HullCase` | real-geometry domain: the *production* SUBOFF voxel mask (`suboff_n128` placement, production `tau = 0.5 + 3·u_in·L/Re`, free-stream lateral faces) |
| `DragTarget(re, cd, weight)` | one drag observation |
| `bounded_drag(box, re, cs=…)` | one differentiable bounded rollout; windowed momentum-exchange C_D over `[window_start, steps)` |
| `synthetic_targets(box, re_values, closure)` | verification-mode observations with known ground truth |
| `calibrate(targets, box, kind=…)` | Adam through the rollouts; `kind="scalar"` (one C_s) or `"power"` (`C_s(Re)=c0·(Re/re_ref)^b`, log-space c0) |
| `evaluate(result, targets, box)` | per-Re predicted vs observed C_D |
| `cs_power(c0, b, re_ref)` | closure factory (truth or initial guess) |
| `load_drag_history(path)` | read one `drag_history.json` sidecar (`tensorlbm.drag-history/v1`, schema-checked) |
| `windowed_cd(history, lo, hi, …)` | C_D of the samples in a step window — the convergence probe |
| `drag_targets_from_sidecars(…)` | campaign sidecars → `DragTarget` rows (tail mean, `2F/(ρ u² S_proj)`) |

Exports live in `tensorlbm` and `tensorlbm.api`. Both case classes work with
`bounded_drag`/`calibrate`/`evaluate` interchangeably (the mask, τ relation,
C_D reference area and lateral closure are case decisions, not arguments).

## Identifiability (measured, 2026-08-22)

The closure is only calibratable from drag in a windowed-τ sense — the C_D
response to a 12× C_s sweep (0.02 → 0.25), fp64, 300 steps, window ≥ 200:

| grid | u_in | Re | τ | C_D spread |
|---|---|---|---|---|
| (6,8,18) r2 | 0.20 | 30 | 0.580 | **13.6%** |
| (6,8,18) r2 | 0.20 | 48 | 0.550 | **17.3%** |
| (10,14,30) r3 | 0.20 | 40 | 0.590 | **8.8%** |
| (10,14,30) r3 | 0.20 | 110 | 0.533 | **17.6%** |
| (6,8,14) r2 | 0.15 | 6 | 0.80 | 2.1% |

**τ ≤ 0.58: identifiable. τ ≥ 0.65: the response collapses to 2–7% and
calibration from drag is ill-posed** (viscous-dominated drag barely sees the
SGS term; the optimizer exploits solver noise instead). All tests and the
example default sit at τ ≤ 0.58.

## Parameterisation note (found the hard way)

The power closure is centred at `re_ref` = geometric mean of the training Re
(set inside `calibrate`). With a naive `re_ref = 1`, the intercept
`c0 = C_s(Re=1)` sits orders of magnitude from the physical `C_s ~ 0.1` range
— for a truth like `0.08·(Re/40)^-1.2` it is ≈ 6.7, outside the positivity
clamp `log c0 ≤ 0`, and the family silently *cannot represent* steeply falling
closures: the fit still matched C_D to ~1% (drag is insensitive), which is
exactly how the defect surfaced — a scalar baseline beat the power fit at the
train endpoints in `test_calibrate_power_tracks_re_dependence`.

## Verification results

CPU suite `tests/test_autograd_calib.py` (8 tests, ci-cpu torch 2.13 and GPU
venv torch 2.11): C_D-vs-C_s gradient matches central differences to < 1e-6
relative; constant truth recovered to < 5% (`test_calibrate_scalar_recovers_constant`);
power truth `0.08·(Re/40)^-1.2` at train Re {30, 48, 70} (τ 0.58 → 0.53) fits
to < 3%, held-out Re 58 to < 2%, and the constant closure given the same data
and budget is measurably worse at the endpoints it must compromise between.

GPU-scale demonstration (fp64, RTX 5090, (10,14,30) r3, u_in 0.20, 300 steps,
window >= 200; truth `cs_power(0.09, -1.0, re_ref=60)`, train Re {40, 70, 110}
at tau 0.590/0.551/0.533, held-out Re 90; 60 Adam iterations each):

| closure | identified | train C_D err | held-out Re 90 | endpoint errs (Re 40 / 110) |
|---|---|---|---|---|
| power | c0=0.0851 (at re_ref 67.5), b=-0.84 | 0.11 / 0.20 / 0.26% | **0.25%** | 0.11% / 0.26% |
| scalar | C_s=0.0824 | (compromises) | - | 1.65% / 1.43% |

~150 s per 60-iteration fit on one GPU. The drag-level recovery is tight;
*parameter-level* recovery is looser (b -0.84 vs truth -1.0) because
d ln C_D / d ln C_s ~ 0.1 — many closures fit drag almost equally well, which
is the same insensitivity that sets the identifiability regime.

## Real observations (2026-08-22, B3-real)

`examples/closure_calibration_real.py` upgrades every ingredient of the loop
to the real campaign: geometry, observations and rollout length, against the
exact-drag dataset `/nfs/wangxi/datasets/scan_suboff_re_drag_20260821`
(24 log-spaced Re 50→800, `suboff_n128` bare hull, cumulant collision,
u_in = 0.1, 4000 steps; `docs/benchmarks/suboff_cd_re_20260821.md`).

**Geometry** — `HullCase` builds the *production* mask
(`suboff_cad.build_suboff_mask`, hull centred `cx = 0.35·nx`, `L = 0.6·nx`):
4093 solid cells at 64×64×128, matching the dataset sidecars cell for cell.
The τ relation is the production one (`tau = 0.5 + 23.04/Re` at u_in = 0.1)
and the lateral planes take free-stream faces (the campaign far-field
condition); the calibration rollout differs from the measured point only in
the collision model (Smagorinsky-BGK vs cumulant) and the outlet plane
(zero-gradient vs far-field).

**The C_D normalisation is the one place this task can silently go wrong**:
the data side divides by `S_proj` = the number of (y,z) columns containing
solid = **69 cells²** — not `π·r_max²` = 63.1 (a 9% bookkeeping bias, and
`C_D` errors of that size dwarf everything else in this study). `HullCase`
computes the projection of the very mask it simulates, so the two sides
agree by construction; `test_hull_case_production_conventions` pins 4093
cells / 69 columns to the dataset values.

**Rollout length from data, not guesswork** — the sidecars sample from step
25, so windowed C_D convergence is measurable directly on the campaign
histories (deviation of a 200-step window mean from the 4000-step tail mean
the report uses):

| window (steps) | worst deviation over all 24 Re |
|---|---|
| [400, 500) | +7.8% |
| [700, 800) | +1.7% |
| **[1000, 1200)** | **+0.52%** |
| [2000, 2400) | +0.1% |

Windowed C_D is converged (a slow monotone relaxation; within-window
fluctuation ≤ 0.1%, no resolved shedding) well before 4000 steps, so the
calibration rolls out **1200 steps with window [1000, 1200)** — 3.3× cheaper
for a ≤ 0.5% systematic offset, an order below the model mismatch below.

**Production-scale identifiability collapses** — the τ ≤ 0.58 rule from the
sphere study transfers as a *necessary* condition but not a sufficient one.
On the real hull the C_D response to C_s is an order of magnitude weaker
(12.5× C_s sweep, 0.02 → 0.25, window [1000, 1200)):

| Re | τ | C_s 0.02→0.25 C_D change | d ln C_D/d ln C_s |
|---|---|---|---|
| 305 | 0.576 | +2.5% | 0.010 |
| 437.8 | 0.553 | +3.9% | 0.015 |
| 800 | 0.529 | +7.8% | 0.030 |
| 148 (held-out, τ 0.66) | 0.656 | +1.0% | 0.004 |

At the same τ the sphere domain gave 13–18%: the production hull's drag is
pressure-dominated and barely reads the SGS term, so *parameter* recovery
from drag is effectively ill-posed at production scale even in the
"identifiable" τ band (drag-level fitting remains meaningful).

**Model mismatch is one-sided and out of the family's reach** — the campaign
drag sits *below* the calibration path even with the SGS term switched off:

| Re | campaign C_D | BGK floor (C_s → 0) | offset |
|---|---|---|---|
| 305 | 5.232 | 5.443 | +4.0% |
| 437.8 | 4.116 | 4.269 | +3.7% |
| 800 | 2.813 | 2.900 | +3.1% |
| 628.6 (held-out) | 3.265 | 3.375 | +3.4% |
| 148 (held-out) | 8.671 | 9.067 | +4.6% |

and any C_s > 0 only *adds* drag (the sweep table above). The cumulant-vs-BGK
difference (plus the outlet-plane treatment) is therefore **not absorbable
by the Smagorinsky closure**: the family is one-sided, the target is below
its zero-SGS limit, and the calibration is structurally infeasible — the
identified parameters slide toward the C_s → 0 edge of the family and the
loss stalls at the floor. That is the headline result of the real-data run,
not a failure of the optimiser: a control verification run at the same scale
(targets produced by the calibration solver itself, truth
`cs_power(0.10, −0.6, re_ref = 437.8)`) fits its train C_D to **0.01–0.09%**
and held-out Re 628.6 to **0.07%** in the same budget (identified
c0 = 0.0925 at re_ref 474.5, b = −0.686; parameter recovery loosens toward
the ends, C_s(800) 0.0696 → 0.0647, i.e. −7% — the elasticity 0.013 at
work), so the machinery works at production scale and the residual against
real data is model error.

**Calibration runs** (RTX 5090, fp32, 1200 steps, window [1000, 1200), train
Re {305, 437.8, 800}, held-out {628.6, 148}, 40 Adam iterations, lr 0.15;
artifacts in `/nfs/wangxi/runs/b3_real_20260822/`):

| model | identified | train err 305 / 437.8 / 800 | held-out 628.6 / 148 |
|---|---|---|---|
| power closure | c0 = 0.0162 (re_ref 474.5), b = +0.10 | 4.04 / 3.73 / 3.13% | 3.39 / 4.58% |
| scalar closure | C_s = 0.0162 | 4.04 / 3.73 / 3.13% | 3.39 / 4.58% |
| C_D = a·Re^b (no solver) | a = 205.6, b = −0.642 | 0.30 / 0.48 / 0.18% | **0.39** / 4.27% |

Both closure fits slide to the C_s → 0 edge (loss 0.103 → 0.0760 and stall,
exactly the floor Σ(BGK residual)² = 0.0760) and land *on* the BGK floor at
every Re — the Re-dependence the power closure is supposed to capture is
unidentifiable here (scalar and power fits are indistinguishable), and the
finite-difference sensitivity at the identified closure collapses to
d ln C_D/d ln C_s = 0.0002–0.0007 (dC_D/dC_s = 0.07–0.12). The identified
C_s ≈ 0.016 is a boundary artifact of a structurally infeasible fit, not a
physical constant.

**Reading** — the closure calibration is *worse* than a two-parameter
power-law fit to the same three training points (train err ≤ 0.5%, held-out
628.6 0.4%, held-out 148 4.3%), because its best attainable drag is the BGK
floor, +3.1–4.6% above the campaign. Solver-in-the-loop calibration against
this dataset needs either a collision model that brackets the campaign
(cumulant in the loop) or an observable the closure actually moves.

**Cost / memory** — per-step gradient checkpointing retains ~3 population
tensors per step (37 MiB/step at this scale → ~47 GiB for 1200 steps, OOM on
a 32 GB card). `HullCase` therefore checkpoints *blocks* of steps
(`checkpoint_block`, default 25) and reduces the window probes to a scalar
inside each block: peak ≈ 9 GiB for any rollout length, one training
iteration (3 rollouts, forward+backward) ≈ 42 s. Measured on the runs above:
power fit 1694 s and scalar fit 1698 s (40 iterations each), the
verification control 1250 s — ≈ 80 min of single-GPU work for the whole
study (the two long fits ran in parallel on two idle cards).

## Honest caveats

- **Verification mode** (synthetic targets from the same solver) proves the
  machinery; with real campaign observations, Smagorinsky *model error* is
  absorbed into the identified constant — the closure reproduces the measured
  drag, not necessarily the true sub-grid physics.
- Calibration from drag requires the identifiable regime (above); at higher τ
  use a flow-field observable instead (the path is differentiable end to end).
- The bounded box keeps lateral planes periodic (A6++ adds free-slip walls on
  a separate branch); the drag window must exclude the initial transient.

## Closure families (2026-08-23, B3-next): which axis moves the observable

The "Real observations" section closed on a prescription: either put *a
collision model that brackets the campaign* in the loop, or find *an
observable the closure actually moves*. The calibration API now exposes both
axes explicitly — `collision ∈ {bgk, mrt} × sgs ∈ {smagorinsky, wale}` on
`bounded_drag` / `synthetic_targets` / `calibrate` / `evaluate` (the four
kernels already existed in `turbulence.py`; `collide_mrt3d` and the MRT-SGS
kernels gained dtype-aware moment matrices so the fp64 calibration path runs
them — see the 2026-08-23 re-audit in `d3q19-d3q27-mrt-consistency-audit.md`).
`CalibResult` records the family it was identified under and `evaluate`
reuses it, so an evaluation cannot silently switch families.

### Experiment (HullCase n128, GPU, `runs/b3_famil_20260823/`)

**Stage 1 — collision axis (no-SGS floors, 1200-step windowed C_D):**

| Re | BGK floor | MRT floor | campaign (cumulant) | MRT gap to campaign |
|---|---|---|---|---|
| 305 | 5.443 | 5.295 | 5.232 | +1.2% |
| 437.8 | 4.269 | 4.166 | 4.116 | +1.2% |
| 800 | 2.900 | 2.850 | 2.813 | +1.3% |
| 628.6 (held-out) | 3.375 | 3.306 | 3.265 | +1.3% |
| 148 (held-out) | 9.067 | 8.785 | 8.671 | +1.3% |

The collision axis moves windowed C_D by **~2.7%** (BGK→MRT), uniformly
across the whole Re range, cutting the floor-to-campaign gap from 3–5% to
**+1.2–1.3%**. The campaign is now bracketed by the two no-SGS families
(MRT below, BGK above), and the remaining ~1.2% is the cumulant-vs-MRT
difference. Adding any SGS model on top only *raises* C_D (see stage 2),
so `(mrt, *)` families approach but cannot cross the campaign from above;
matching it exactly needs cumulant-in-the-loop (a differentiable cumulant
transform, substantially more work) or a residual model-error term.

**Stage 2 — SGS axis (WALE sweep at Re 437.8):**

| C_w | 0.10 | 0.20 | 0.30 | 0.45 | 0.60 |
|---|---|---|---|---|---|
| C_D (bgk+wale) | 4.196 | 4.196 | 4.198 | 4.201 | 4.205 |
| C_D (mrt+wale) | 4.166 | 4.167 | 4.168 | 4.171 | 4.174 |

Identifiability `d ln C_D / d ln C_w` at C_w=0.3: **0.0009 (bgk)**,
**0.0011 (mrt)** — an order of magnitude *weaker* than Smagorinsky's already
collapsed 0.004–0.03. WALE's near-wall vanishing design (nu_t → 0 at walls)
removes the only region where the sub-grid stress could act on a
pressure-dominated hull drag. **Changing the SGS family does not fix the
identifiability collapse; the drag observable itself barely reads any SGS
model at this resolution.**

### Conclusions

1. The **collision** axis is the identifiable lever on drag (~2.7%); the SGS
   axis is not (~0.1%), for either Smagorinsky or WALE.
2. The one-sided floor mismatch shrank from 3–5% to 1.2% via MRT; the rest
   is genuinely the campaign's cumulant collision, which the differentiable
   path does not yet implement.
3. Calibrating a closure constant against hull *drag* at n128 is therefore
   bounded whatever the SGS family: the next step that can change this
   conclusion is an **observable the closure moves** (wake profile,
   separation point, surface pressure distribution — all reachable through
   the same differentiable rollouts) rather than another closure family.


## Observable swap (2026-08-23, B3-next stage 3)

Instrument: `tensorlbm.autograd_calib.bounded_observables` (one no-grad
windowed rollout returning window-mean field observables at the production
probe phase) + `observable_response(a, b, ref)` (relative L2 per
observable).  Production hull `n128`, window `[1000, 1200)`, probe planes
x = 90/100/112 (library defaults derive (93, 103, 114) from the mask and
reproduce the same structure).  Full data: `obs_probe.json` /
`obs_analysis.json` (agent-of-record archive under
`runs/b3_obs_20260823`).

Metrics: `S_smag` = response per e-fold of `C_s` (central 0.05->0.2,
divided by ln 4); `G_sgs` = effect of switching SGS on at `C_s = 0.1` vs
the floor; `G_coll` = collision-family step (bgk -> mrt, both floors).

| Re | observable | S_smag | G_sgs | G_coll |
|---|---|---|---|---|
| 305 | **cd** (control) | 0.011 | 0.004 | 0.008 |
| 305 | press_profile | 0.005 | 0.002 | **0.062** |
| 305 | wake_deficit@100 | 0.008 | 0.003 | 0.006 |
| 305 | wake_cross@112 | 0.007 | 0.003 | 0.030 |
| 437.8 | **cd** (control) | 0.017 | 0.006 | 0.007 |
| 437.8 | press_profile | 0.007 | 0.003 | **0.052** |
| 437.8 | wake_deficit@100 | 0.014 | 0.005 | 0.006 |
| 437.8 | wake_cross@112 | 0.014 | 0.005 | 0.041 |
| 800 | **cd** (control) | 0.034 | 0.014 | 0.002 |
| 800 | press_profile | 0.013 | 0.005 | **0.083** |
| 800 | wake_deficit@100 | 0.030 | 0.012 | 0.006 |
| 800 | wake_cross@112 | 0.036 | 0.014 | 0.063 |

Scalar wake-deficit depth (`wake_min_ux`), bgk -> mrt floor: -1.1% (305),
-5.9% (437.8), -15.1% (800) — the largest collision response measured,
but a min-statistic (non-smooth; use the profile norms for gradients).

**Steadiness (robustness)**: splitting the window in halves, the pressure
response is identical in both (G_coll 0.0524 / 0.0525) while the
within-family cross-half drift is ~0.0016 — signal-to-drift ~33x, i.e. a
steady-state effect, not shedding phase.  The cross-flow field is the
opposite: cross-half drift 0.076 exceeds its response (~0.04) — phase
contaminated at 100-step averaging; usable only with long windows.

**WALE** (Re 437.8, C_w 0.15/0.3/0.45): every observable <= 0.0033
(press) — dead on all fields, not just drag, confirming the family-level
verdict of stage 2.

**Recirculation** is exactly zero at all three Re on the bare hull (no
separation): a separation-point observable needs sail/fin geometry or
higher Re — not available in this campaign regime.

### Measurement reconciliation (matched windows)

The stage-1 table above mixes BGK floors quoted from #224 with MRT
floors re-measured at the current default window.  Re-measuring the BGK
floor at the same window `[1000, 1200)` (both via `bounded_drag` and the
new instrument, agreeing to 7 digits) gives 5.340 / 4.195 / 2.857
(Re 305 / 437.8 / 800) instead of 5.443 / 4.269 / 2.900.  Consequences at
matched windows:

- the collision-family axis moves C_D by only ~0.8% (not ~2.7%);
- BGK-floor-to-campaign shrinks to +1.5-2.1% (MRT stays +1.2-1.3%);
- the remaining cumulant-vs-MRT gap is unchanged.

### Conclusions

1. **Surface pressure is the calibration observable**: the collision
   axis moves it 5.2-8.3% vs 0.2-0.8% for drag (7-37x amplification),
   steady across the window (33x SNR), and available in the campaign
   data (the center-plane snapshots carry `rho`, whose near-surface cut
   is the same profile).
2. Wake deficit inherits drag's blindness (G_coll ~0.6%, S_smag tracks
   drag's own); cross-flow reads the collision axis (3-6%) but is
   phase-noisy at practical window lengths.
3. **No observable rescues the SGS constant** at these Re/resolution
   (S_smag <= 0.036 everywhere, same order as drag's own response);
   WALE is dead on every field.  Identifying `C_s` needs higher Re or
   bluffer geometry, not a better observable on this campaign.
4. Next (stage 4): calibrate *continuous* MRT relaxation rates against
   the campaign pressure profiles (the profile Jacobian w.r.t. rates is
   the same autograd rollout away); a differentiable cumulant targets
   the same observable.

Tests: `tests/test_closure_observables.py` (7, CPU, ~10 s).
