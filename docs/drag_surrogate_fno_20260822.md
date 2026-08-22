# SUBOFF field-to-drag surrogate: plane snapshot → C_D (FNO encoder)

Date: 2026-08-22 (v1 pilot + v1.1 two-parameter addendum) · Module:
`tensorlbm.ai.drag_surrogate` · Run: `/nfs/wangxi/runs/fno_drag_20260822`

First closed AI4S loop over the B1 campaign: LBM Re sweep → exported fields +
exact drag labels → Fourier-encoder surrogate predicting C_D directly from a
flow-field snapshot — no Reynolds number input, no wake-survey post-processing.

> **v1.1 addendum (read first).** The pilot numbers below (0.41 % MAPE on the
> single-parameter sweep) do **not** transfer: on the Re × u_in grid the same
> model mispredicts by ~60 % MAPE, and retrained surrogates lose to the
> fitted power-law prior. See the v1.1 addendum section — the
> pilot's residual error reflected field *magnitude* (velocity scale) as a
> Re proxy, not shape understanding.

## Task

- **Input**: centre-plane snapshot `(3, ny=64, nx=128)` — channels `(ux, uy,
  rho)` at `z = nz/2` — from the B1 field campaign
  (`scan_suboff_re_20260821`, 24 points, 8 snapshots each, `(64, 64, 128)`).
- **Label**: exact C_D from the control-volume observer rerun
  (`scan_suboff_re_drag_20260821`, `tensorlbm.drag-history/v1` sidecars, tail
  mean over the last 25 % of samples), joined on Re. These are the *measured*
  values of the C_D–Re benchmark v1.1 — not wake-survey estimates.
- **Split**: the campaign's own point-level split (17/4/3 train/val/test,
  `dataset.json:split_points`). Normalisation statistics (per-channel and
  target) are fitted on the train split only.

## Model and protocol

FNO encoder (`FNODragRegressor`): lifting 1×1 conv → 4 Fourier layers
(`SpectralConv2d`, modes 12×12, width 32, GELU) → global spatial mean-pool →
MLP(128) → scalar. Target is `log10(C_D)` standardised by train statistics;
metrics are reported in raw C_D units. Adam (lr 1e-3, weight decay 1e-4), batch
32, ≤600 epochs, early stop on val loss (patience 80), seed 0, single RTX
5090, minutes per variant.

**Baselines** (fitted on train points only):

- `power_law`: least-squares `C_D = a·Re^b` — the physics prior, and a strong
  one on this sweep: fit gives `a ≈ 29.8`, `b = −0.7036`, matching the
  benchmark scaling `Re^−0.703`.
- `mean`: train-mean C_D (difficulty reference).

One snapshot per point per variant (snapshots from one point share a
trajectory and a label — pooling them inflates row counts without new
information; see the `late2` variant below).

## Results (test split, 3 held-out points)

| variant (input) | model | MAE | RMSE | R² | MAPE % |
|---|---|---|---|---|---|
| steady (step 4000) | **FNO** | **0.084** | 0.104 | 0.9996 | **0.74** |
| steady (step 4000) | power_law | 0.271 | 0.319 | 0.9966 | 2.52 |
| steady (step 4000) | mean | 4.648 | 5.628 | −0.07 | 42.0 |
| **early (step 500)** | **FNO** | **0.044** | 0.051 | 0.9999 | **0.41** |
| early (step 500) | power_law | 0.271 | 0.319 | 0.9966 | 2.52 |
| late2 (steps 3500+4000) | FNO | 0.380 | 0.585 | 0.9885 | 2.98 |

Val split (4 points) agrees: FNO MAPE 1.16 % (steady) / 1.06 % (early) vs
power_law 2.26 %.

Per-point test predictions (relative error):

| Re | C_D true | steady FNO | early FNO | power_law |
|---|---|---|---|---|
| 56.4 | 17.974 | −0.95 % | −0.45 % | −2.83 % |
| 212.4 | 6.709 | −0.68 % | −0.29 % | +2.42 % |
| 239.6 | 6.170 | −0.60 % | −0.50 % | +2.31 % |

## Findings

1. **The field knows the drag.** Both snapshot ages beat the power-law prior
   by 3–6× in MAE, with per-point errors consistently below 1 % (the power law
   is always ±2.3–2.8 % — it captures the trend but not point-specific
   deviation from it).
2. **The early field (step 500) is the best predictor.** 12.5 % of the
   4000-step trajectory already determines the final C_D to 0.4 % MAPE — the
   surrogate-acceleration story: run 1/8 of the steps, infer the rest. The
   transients (bow-wave width, boundary-layer development) encode the
   viscosity regime that fixes the steady drag.
3. **Pooling correlated snapshots hurts.** `late2` (two late snapshots per
   point) is worse than either single snapshot: rows double but information
   does not, and the loss weights points unevenly. Effective sample size is
   the number of points.
4. Best epochs: 571 (steady) / 541 (early) / 304 (late2); the first two sit
   near the epoch cap — a longer schedule might squeeze further, deliberately
   not tuned here.

## Caveats

- **3 test points, 1 seed, 1 geometry, 1 parameter.** The numbers are
  encouraging, not conclusive; with 24 points total this is a pilot, and the
  power-law prior is unusually strong on a pure Re sweep (it *is* the physics).
  The surrogate's edge should grow on multi-parameter / multi-geometry
  campaigns where such a closed-form prior does not exist.
- Single centre plane; 3-D encoders or multi-slice stacks are untested here.

## Usage

```python
from tensorlbm.ai.drag_surrogate import (
    PlaneSampleSpec, FNODragArch, DragTrainConfig, run_drag_surrogate_study,
)

summary = run_drag_surrogate_study(
    fields_dir="/nfs/wangxi/datasets/scan_suboff_re_20260821",
    drag_dir="/nfs/wangxi/datasets/scan_suboff_re_drag_20260821",
    out_dir="/nfs/wangxi/runs/fno_drag_20260822/early",
    spec=PlaneSampleSpec(steps=(500,)),
    arch=FNODragArch(),                    # modes ≤ (ny//2, nx//2+1)
    config=DragTrainConfig(device="cuda"),
)
```

Artefacts per variant under `/nfs/wangxi/runs/fno_drag_20260822/<variant>/`:
`model.pt` + `model.pt.json` (arch + normalisation), `metrics.json`
(schema `tensorlbm.drag-surrogate-study/v1`), `predictions.csv`.
Synthetic-campaign regression tests: `tests/test_drag_surrogate.py` (11 tests,
CPU and GPU venvs).

## Next steps

- Repeat on the planned `u_in` sweep and hull-variant campaigns — the regime
  where a closed-form prior disappears and surrogate value is the point.
- k-fold over the 24 points + multi-seed ensembles to tighten the error bars.
- Multi-plane / 3-D encoder; conditioning on geometry tokens for
  multi-hull generalisation.

## v1.1: Two-parameter generalization (Re × u_in grid, same day)

Campaign: 24-point full factorial (6 log Re levels × u_in ∈ {0.06, 0.085,
0.11, 0.14}) via `ScanPlan.generate` — the #201 doe fix exercised in
production, with level round-trip asserted at launch. Same observer, same
protocol; τ ∈ [0.517, 1.145], worst corner smoked first (no NaN). Dataset
`/nfs/wangxi/datasets/scan_suboff_re_uin_20260822` (~4 min on 8 GPUs).

### Finding 1 — u_in-similarity is violated at 3.7–8.3 %

At fixed Re, C_D falls monotonically with u_in (−6.2 % per e-fold; steepest at
Re=50). Incompressible similarity says ~0; the residual is the τ-dependence of
lattice errors (τ = 0.5 + 230.4·u/Re co-varies with u) plus a Ma² term at
u=0.14. A useful solver-fidelity fact in its own right, now on record.

### Finding 2 — the single-parameter model does not transfer

| u_in slice | 0.06 | 0.085 | 0.11 | 0.14 | all 24 |
|---|---|---|---|---|---|
| pilot model MAPE % | 128.6 | 41.5 | 18.9 | 49.3 | **59.6** |

The pilot model keyed on field magnitude (the velocity scale is a perfect Re
proxy at fixed u_in), not on wake shape. Scale-normalized inputs
(ux, uy → /u_in) fix the val split (3.67 % → 1.88 %) but not the test.

### Finding 3 — at 24 points the physics prior wins

| model (step-500 input) | val MAPE % | test MAPE % |
|---|---|---|
| power law in Re (2 params) | 1.93 | **2.57** |
| + log u_in correction (3 params) | 3.50 | 4.05 |
| FNO, raw inputs | 3.67 | 5.48 |
| FNO, u_in-normalized inputs | 1.88 | 5.37 |

The u_in effect (4–8 %) sits at or below the FNO noise floor for 17 training
points; residual correlation on test is effectively noise. Parametric
baselines dominate — and the 3-param fit does not beat the 2-param one
(small-N): data scale, not architecture, is the binding constraint.

### Reading

The pilot's 0.41 % was single-parameter luck; the honest state is: **the
loop works end-to-end (campaign → dataset → labels → study → retrain, ~30 min
including two retrainings), but surrogate value needs more data and a richer
parameter space** where closed-form priors do not exist. Next: LHS 32–48
points over (Re, u_in), scale-invariant inputs by default, then hull/geometry
parameters. API hardened along the way: labels now join per point
(`load_exact_cd_per_point` / `cd_by_point`) — Re-keyed joins collapse on
multi-parameter grids (regression-tested).
