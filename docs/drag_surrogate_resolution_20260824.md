# B4-g4: SUBOFF drag surrogate resolution axis — campaign + cross-resolution A/B (2026-08-24)

Branch `exp/b4-g4` (worktree `/nfs/wangxi/worktrees/b4_g4`, base `origin/main`
`ca9e0c4d`).  Question: how large is the grid-resolution error band of the
production drag chain, and does a resolution channel in the surrogate condition
vector give a single model cross-grid reach?

Grid convention: the TensorLBM production case `suboff_n128` builds the grid
`(nz, ny, nx) = (n/2, n/2, n)` with hull placement/length grid-relative
(`cx = 0.35 nx`, `L = 0.6 nx`), production `n = 128` (i.e. 64x64x128; the
"96x64x128" grid in the legacy SWLBM notes belongs to a different chain).
`n` below always means this integer `resolution`.

## 1. Campaign design

Two tiers, n=64 and n=160, same registered case and production chain as every
mother campaign (cumulant collision, `u_in = 0.1`, mass correction every 10
steps, `DragSurveySpec(margin=4, interval=25)`, `hull_type=with_sail`,
`sail_scale=fin_scale=1.0` — the mother reference geometry):

| tier | grid (nz,ny,nx) | hull L | vox (sail) | A_proj | steps | tau range |
|------|-----------------|--------|------------|--------|-------|-----------|
| n64  | 32x32x64        | 38.4   | 584 (+6)   | 23     | 2000  | [0.5188, 0.6908] |
| n160 | 80x80x160       | 96.0   | 7465 (+40) | 102    | 5000  | [0.5470, 0.9771] |

- DoE: 14-point LHS over log10(re) in [60, 700], seed 20260824, **identical
  for both tiers** (paired design: every tier point has a same-re sibling on
  the other tier; re = 60.37 … 613.27).
- Step budget = equal lattice convective time as the mother campaigns
  (steps = 4000 * n / 128 → 2000 / 5000); tau envelope is the case's own
  `CaseUnits.from_reference` (`tau = 0.5 + 3*u_in*0.6*n/re`), so the tiers
  sit inside the mother stability envelope.
- Datasets (all scripts archived inside each directory):
  - `/nfs/wangxi/datasets/scan_suboff_resolution_smoke_20260824` (preflight,
    4 pts @ 300 steps, per-point resolution override — chain viability at
    non-production grids, C_D within [0.5, 1.5] x mother curve).
  - `/nfs/wangxi/datasets/scan_suboff_resolution_n64_20260824` (14/14 done).
  - `/nfs/wangxi/datasets/scan_suboff_resolution_n160_20260824` (14/14 done).
- Source plumbing: the scan chain was already resolution-parametric
  (`resolution` is an integer constructor param of the case, coerced by
  `coerce_case_params`); **no solver or scan_runner change was needed** —
  tier campaigns pass `fixed_params={"resolution": 64|160}`.  The only src
  addition is the conditioning channel (`tensorlbm.ai.drag_cond`).
- Run scripts: `/nfs/wangxi/runs/b4_g4_20260824/{res_preflight,res_launch,
  res_validate,build_cache_g4,train_fno_g4}.py`, log `train_g4.log`, metrics
  `metrics_g4.json`, band `grid_band.json`.

## 2. Validation + grid-independence error band

All 28 tier points: completed at full step budget, forces finite, exact drag
sample counts (80 / 200), tail drift <= 5.7e-5 (bar 1e-3, v2 convention),
C_D(re) strictly decreasing.  Tier stored masks are bit-identical to the CAD
rebuild on each tier grid (28/28).

Error band vs the mother n=128 with_sail curve (`scan_suboff_hull_re_20260823`,
14 pts; reference = linear interpolation of log10 C_D over log10 re):

| tier | C_D/CD128 rel median | mean |MAE| | range | paired n160/n64 |
|------|----------------------|----------|-------|-----------------|
| n64  | **-12.0%**           | 11.7%    | [-15.3%, -6.8%] | +22.8% median |
| n160 | **+8.0%**            | 8.0%     | [+6.9%, +8.9%]  | (28.6% max)   |

Per-point offsets are smooth and monotone in re (n64: -15.3% at re=60 →
-6.8% at re=613; n160: +8.9% → +6.9%): the production 128-grid C_D sits
between the tiers — classic monotone grid convergence bracketing the 128
value.  **Grid-independence error band of the production chain: ~12% at
half resolution, ~8% at 1.25x resolution** (mesh resolution dominates C_D
error, consistent with the SWLBM-era finding; the sign is systematic —
coarse grids under-predict at low re).

## 3. A/B: resolution channel vs no channel

Corpus `cache_g4.npz` (266 pts) = mother v3 corpus (238, n=128, 6 datasets)
+ both tiers (14+14).  Tier planes enter the FNO canvas (64x128 centre
z-plane) via `torch F.interpolate` (bilinear antialias for flow/rho,
nearest for the solid mask); mother rows verbatim.  Everything else is the
v3 champion protocol (`CondFNODrag` width 32 x 4 layers, AdamW 1e-3/1e-4,
500 epochs patience 60, quota sampling over 8 dataset labels, force-tail
aux head lambda 0.1); FNO arms at model seeds 0/1/2.

Arms: `v3` = 8-ch condition (no resolution knowledge); `v3r` = 9-ch
`condition_v4` = v3 + `log10(n/128)` (`tensorlbm.ai.drag_cond.resolution_channel`;
production n encodes to exactly 0.0, first 8 columns bit-identical to v3).
Linear baselines: `power_re`, `power_geo` (v3 verbatim) and `power_geo_r`
(power_geo + the same resolution column).

MAPE (mean +/- std over 3 seeds; linear baselines deterministic):

| split (test) | v3 | v3r | power_geo | power_geo_r |
|--------------|-----|-----|-----------|-------------|
| random (54)  | 0.57+/-0.20 | 0.65+/-0.11 | 6.14 | **5.13** |
| random slice n64 / n160 | 0.19+/-0.07 / 1.23+/-0.58 | 0.27+/-0.08 / **0.24+/-0.13** | 15.4 / 8.5 | 2.3 / 3.9 |
| loro::n64 (14)  | 13.2+/-0.7 | 70.2+/-68.9 (seeds 167/15/28) | 16.2 | **3.27** |
| loro::n160 (14) | 6.90+/-0.33 | 5.16+/-3.14 (5.2/1.3/9.0) | 6.72 | **1.68** |
| loro::n128 (238)| diverged (1e3-1e57) | diverged | 12.7 | **13.9** |

Resolution-blind floor: a model that knows the mother curve but not the
grid cannot beat the systematic band of section 2 (n64 ~11.7%, n160 ~8.0%
MAPE).  `v3` lands exactly there (13.2 / 6.9) — it learned the n=128
mapping and applies it unchanged.

## 4. Findings

1. **In-corpus (random)**: both FNO arms are sub-1%; the resolution channel
   improves the fine-grid slice 5x (n160 1.23% -> 0.24%) and is otherwise
   neutral.  Cross-grid interpolation inside the training envelope is solved.
2. **Out-of-corpus (LORO)**: the resolution channel helps on upscale
   extrapolation on the mean (n160 5.2% vs 6.9%, best seed 1.3%) but is
   **unstable** (std 3.1; n64 fold seeds 15-167%).  The v3 FiLM conditioning
   extrapolates poorly in the channel it was never trained to span; the
   channel is necessary but not sufficient.
3. **The robust cross-grid predictor is linear**: `power_geo_r` (one extra
   log10(n/128) column) reaches 3.3% (n64) / 1.7% (n160) — it captures the
   systematic, nearly-linear-in-log-n C_D offset that section 2 measures
   directly.  The grid bias is a *parametric* effect the FNO does not learn
   from 14-point tiers.
4. **Tier-only training is degenerate** (loro::n128): 24 fit points, all
   with_sail at sail=fin=1, zero geometry variety — FNO val MAPE 0.01-0.2%
   (memorisation) while 21% of mother test points explode (median APE 19%,
   tail to 1e16).  Validation MAPE is blind to extrapolation failure here.

## 5. Anomalies / caveats

- `v3r` loro::n64 seed 0 read 167% (diverged); reported mean+/-std includes
  it.  Small-tier FNO folds are seed-sensitive — treated as a finding, not
  smoothed away.
- The n64 tier has 80 drag samples (2000 steps / 25); aux tail bins then
  average 2-3 samples per bin (mother: 20).  No adverse effect observed.
- A_proj quantisation at n64 (23 voxels) makes C_D normalisation itself
  grid-quantised; part of the n64 band width is A_proj, not flow physics.
- Mother with_sail reference uses log-re linear interpolation of log10 C_D
  (14 pts); interpolation error is inside the smooth-curve regime of the
  band but not separately quantified.

## 6. Pointer for the next step (owner decision)

L2/L3 roadmap: to make the FNO itself cross-grid, the evidence says: enrich
the tiers with geometry variety (the fam/v3 axes at n!=128) rather than more
re points, or hybridise — FNO at n=128 + the parametric `power_geo_r` grid
correction (log-linear in n) as a post-hoc calibrator; the campaign data
here already pins the correction amplitude (+8% @ n160 / -12% @ n64).
