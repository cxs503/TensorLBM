# Cross-hull generalisation: the geometry campaign (B1-4) + LHS40 error bars (B1-3)

Date: 2026-08-23 · Branch: `feat/hull-scan` · Dataset:
`/nfs/wangxi/datasets/scan_suboff_hull_re_20260823` (42 points) and
`scan_suboff_re_uin_lhs40_20260822` (40 points, re-analysis)

## The question

Every evaluation so far lived on the Re axis, where a two-parameter power law
`C_D = a·Re^b` is a strong prior (v1.1: it beat the raw-input FNO; v1.2: the
scale-invariant FNO edged it out 1.15 % vs 1.56 % on one split). Hull geometry
is the axis where **no such prior exists**: three real DARPA SUBOFF
configurations — `bare_hull`, `with_sail`, `full` — differ by appendage drag
that no 2-parameter law spans. If the field→C_D surrogate generalises across
hulls, that is its value case.

## Campaign

42 points: 3 hulls × 14 LHS-sampled Re ∈ [50, 800], grid 64×64×128, cumulant,
u_in = 0.10, τ(Re) = 0.5 + 23.04/Re ∈ [0.530, 0.958], 4 000 steps, snapshots
every 500. `hull_type` rides along as a **categorical sweep param** (string) —
this campaign is also what forced `_param_meta` (numerics → float, categoricals
verbatim) through plan IO, run metadata, and h5 attrs, so `scan_runner` now
supports non-numeric sweep axes.

## Result 0 — the geometry axis is (almost) not there at this resolution

Before reading any MAPE, the mask itself:

| hull | solid cells | Δ vs bare |
|---|---|---|
| bare_hull | 4 093 | — |
| with_sail | 4 121 | +28 cells (0.7 %) |
| full | 4 157 | +36 further (0.9 %) |

At 0.6·128 = 76.8-cell hull length the sail and stern appendages are
sub-voxel-thick features. Mean C_D per hull confirms it: 8.23 / 8.20 / 8.14 —
a ~1 % spread, i.e. the hulls are nearly drag-identical as voxelised, and the
Re trend dominates everything (R² ≥ 0.98 for all models below).

## Result 1 — campaign split (mixed hulls, 29/6/7)

| model | test MAPE |
|---|---|
| FNO (velocity-scaled) | 1.98 % |
| power law, global (2 params) | **0.88 %** |
| power law, per-hull (6 params) | 2.50 % |

The per-hull law is *worse* than the global one: three hulls × 14 points is
not enough to fit six parameters on ~1 %-separated families, so the extra
parameters absorb noise. The FNO lands in between.

## Result 2 — leave-one-hull-out (train on two hulls, test the third)

| held-out hull | FNO | power2 (2-hull fit) | donor-hull law |
|---|---|---|---|
| bare_hull | 5.92 % | **2.19 %** | 2.77 % |
| with_sail | 9.04 % | **2.82 %** | 3.29 % |
| full | 7.09 % | **1.55 %** | 1.71 % |

The physics prior transfers essentially for free *because there is nothing to
transfer* — the held-out hull's drag law is the training hulls' law to within
~2 %. The FNO is 3–6× worse: it can read hull identity off the input field and
spends capacity fitting each family's ~1 % offset, which is exactly the wrong
inductive bias when the families are near-degenerate.

## Reading

This is a **negative result with a diagnosed cause**, and the cause is
actionable: the experiment needs a geometry axis that survives voxelisation —
either n256 (appendages ≥ 2–3 cells), or exaggerated appendage scale, or hull
families that actually separate in C_D (sail-on-hull at higher Re where
appendage drag is form-dominated, not viscous-dominated). What the campaign
*did* establish:

1. the scan chain runs mixed categorical+numeric sweep axes end-to-end
   (`hull_type` in plan.json / status.json / h5 attrs, resume-safe);
2. 42 more production SUBOFF points across three real configurations;
3. a documented failure mode for the surrogate story: **wherever a low-order
   physics prior exists — even trivially, because the axis is degenerate —
   the surrogate must beat it, not just match it.** The value case has to be
   sought where the prior does not exist *and the simulation can represent
   the difference*.

## B1-3 statistical tightening: 5-fold × 3-seed on LHS40

v1.2's headline (FNO 1.15 % vs power3 1.56 % on one 28/6/6 split, one seed)
gets error bars: sorted-stratified 5-fold (sort by C_D, deal round-robin;
test = fold, val = next fold, train = rest), FNO seeds {0,1,2} per fold.

| model | test MAPE (mean ± std over 5 folds) | folds won |
|---|---|---|
| FNO (velocity-scaled, seed-averaged) | **1.63 % ± 0.90 %** | 4/5 vs power3 |
| power2 | 2.33 % ± 0.29 % | 0/5 |
| power3 | 1.95 % ± 0.23 % | 1/5 vs FNO |

Fold-by-fold (FNO vs power3, pp): +0.64, +0.84, **−0.81**, +0.71, +0.21.
The single losing fold (2) is also the FNO's least seed-stable one
(3.14 ± 0.61 across seeds {0,1,2}, one seed at 3.98) — an unstable
configuration, not a systematic loss. The v1.2 single-split margin
(1.15 vs 1.56) survives directionally; what it hid is the FNO's
fold-to-fold spread (±0.90 vs the power laws' ±0.25): the surrogate's
advantage is real on average and on 4/5 folds, but it pays for it in
variance. With five folds a paired sign test does not reach significance
(4/5, p ≈ 0.38 two-sided) — the claim to carry forward is "FNO ≥ power3
on this campaign, decisive on four of five folds", not a knockout.

## Reproduce

- campaign: `hull_campaign_launcher.py (repo root)` (launcher, assertions on
  per-hull counts and τ floor)
- analysis: `hull_campaign_analysis.py (repo root)` (split + LOHO + mask audit)
- k-fold: `drag_surrogate_kfold.py (repo root)`
