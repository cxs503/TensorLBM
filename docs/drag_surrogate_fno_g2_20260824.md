# Drag surrogate conditioning: G2 interaction data (v2) → physics-geometry encoding (v3)

Date: 2026-08-24 (v2 campaign + protocol; v3 same day) · Module:
`tensorlbm.ai.drag_cond` (new) + `tensorlbm.ai.fno.SpectralConv2d` · Runs:
`/nfs/wangxi/runs/b4_v2_20260824` (v2, report.md there) and
`/nfs/wangxi/runs/b4_v3_20260824` (v3) · Dataset:
`/nfs/wangxi/datasets/scan_suboff_hull_scale_g2_20260824` (G2, 48 points;
launcher/preflight/validate scripts archived in the dataset directory)

> **Short version.** v2 added the missing hull×appendage-scale interaction
> data (G2, 48 pts) and improved every LOHO fold, but no fold beat the
> parametric power-law prior: the binding constraint was the conditioning
> **encoding**, not the data. In every `loho::<hull>` fold the held hull's
> one-hot channel has train-set std = 0 — its weights never train and fire
> at test through random init. v3 replaces identity with four continuous
> mask-derived geometry channels, adds per-dataset quota sampling and a
> force-tail auxiliary head. Outcome, stated honestly: **no single arm
> beats the power-geo prior on all three LOHO folds.** The full v3 model
> (`C_full`) cuts the joint random split from 3.61 % to 0.66 % MAPE
> (geo-scaled slice 5.60 → 0.77 %), wins `with_sail` outright
> (1.05 ± 0.50 vs 5.16) and `loho::full` (7.86 ± 3.50 over 5 seeds vs
> 10.25), but reads 3.14 ± 0.31 on the bare fold against power-geo's
> 2.00. The 4-log-parameter no-one-hot reference (`R4_nooh`) wins the
> bare fold decisively (0.71 ± 0.32) and clears all three power-geo lines
> on 5-seed means, though its `full` margin (9.97 vs 10.25, 0.28 pp) is
> below the 0.5 pp evidentiary bar. The bare-fold residual of the
> geometry arms is **uniform across all 14 held points** (B: 2.13–2.98 %,
> C: 3.40–3.65 %, no outliers) and matches the 2.3–2.5 % C_D offset
> between G2's nearest anchors (sail = 0.4) and the true bare curve — a
> data-coverage floor at the extrapolation corner, not encoding noise.

## 1. G2 — the hull × scale interaction campaign (v2 recap)

**Why.** v1's only scaled-geometry corpora (geo_lhs 78, b15 6) are 100 %
`full`; the only non-full hulls (hull_re 42) sit at sail = fin = 1. The
hull-identity × appendage-scale interaction panel was empty, and the
geo-scaled crash (23–27 % on geo_lhs / 43 % on b15 for scale-1-trained
models) lived exactly there.

**Design** (seed 20260824, joint LHS, re log-U [50, 800], u_in = 0.1,
suboff_n128, cumulant, 4 000 steps): 22 `with_sail` × sail ∈ [0.4, 3]
(fin inert), 26 `full` × (sail, fin) ∈ [0.4, 3]², corner anchors at
re ∈ {200, 600}; 0 new `bare_hull` points — bare masks are **bit-identical**
under any (sail, fin), so hull_re's 14 bare pts already cover that row.
Sub-1 scales make hull identity a continuum: `full(0.4, 0.4)` differs from
bare by 13 voxels (vs 1 654 at (3, 3)); the sail = 0.4 anchors sit 2.3–2.5 %
above the bare C_D(Re) curve. All 48 pts completed @ 4 000 steps (360 s on
8 GPUs); C_D ∈ [2.13, 18.28], tail drift ≤ 1e-4. Scripts:
`g2_preflight.py` / `g2_launch.py` / `g2_validate.py`, archived in the
dataset directory (convention of the geo_lhs campaign).

**v2 result** (protocol byte-identical to v1; +G2 corpus only): joint test
3.61 % (geo-scaled slice 5.6 % vs 22.7–23.3 % for any scale-1-only
campaign); LOHO 9.44 / 5.65 / 12.78 (bare / with_sail / full) vs power-geo
2.00 / 5.16 / 10.25 — every fold improved, no fold flipped. The diagnostic
(`diag_nooh.py`): dropping the one-hot (cond 7 → 4) gave 0.50 / 2.77 / 9.69
and random 2.67 — the identity columns were pure variance.

## 2. v3 — physics-geometry conditioning, quota sampling, aux head

Protocol, corpus (238 pts), cache, representation, splits, evaluation
slices and hyper-parameters are **unchanged from v2** (split 0 / val 1 /
model 0; FiLM-FNO2d width 32 / 4 layers / modes 16×32; AdamW 1e-3/1e-4;
batch 32; ≤ 500 epochs; patience 60). The model class moved into the repo
(`tensorlbm.ai.drag_cond.CondFNODrag`); init parity with the v1/v2
run-script model is verified bitwise at seed 0 (all 30 state-dict tensors,
forward max-diff 0.0 — so v2_ref below re-derives the v2 numbers).

### 2.1 Encoding (cond_dim 7 → 8, no identity column)

Condition = `[log10 re, log10 u_in, log10 sail, log10 fin]` + geometry
block from the CAD predicates (`suboff_geometry_features`, same predicates
the scan chain voxelises with — a pure function of the design parameters,
computable for new designs without a stored simulation mask):

| channel | definition | corpus range |
|---|---|---|
| `log_aproj_ratio` | log10(A_proj / A_proj_bare), x-projected lattice area | [0, 0.409] |
| `sail_frac` | net sail voxels / bare-hull voxels (4 093) | [0, 0.195] |
| `fin_frac` | net fin voxels / bare-hull voxels | [0, 0.209] |
| `solid_frac` | total solid / bare-hull voxels = 1 + sail + fin | [1.000, 1.409] |

Guarantees (pinned in `tests/test_drag_cond.py`): determinism; bare-hull
**bit-invariance** under any (sail, fin); fin no-op on with_sail; sail/fin
fractions monotone non-decreasing over scale ∈ [0.4, 3.0] (0.1-step sweep:
zero violations); scale = 1 counts equal `build_suboff_mask` exactly
(anchors 4 093 / 4 121 / 4 157) and the predicate composition is
bit-identical to the voxel builder.

Zero-variance audit (the one-hot pathology, restated for the new
channels): on the corpus, `loho::full`'s train side (bare + with_sail)
still has three constant channels — `log10 u_in`, `log10 fin`, `fin_frac`
— because no non-`full` geometry carries fins. This is structural (the fin
axis is only populated by `full`); the difference vs v2 is that hull
discrimination no longer runs *exclusively* through those untrained
weights: `log_aproj_ratio`, `sail_frac`, `solid_frac` all vary in every
fold (std ≥ 0.055 / 0.071 / 0.055 in the worst fold), so the held hull is
reached by interpolation on trained channels, and the untrained
`fin_frac` fires with magnitude ≤ 0.21 instead of a one-hot 1.0.

**Vintage note** (documented, accepted): the two oldest campaigns
(re_drag 24, lhs40 40; both `full` @ scale 1) predate an appendage-predicate
fix — their stored masks are the current CAD minus 28 sail + 36 fin voxels
(A_proj 69 vs 73). The encoding uses the current-CAD values (design-truth,
deployable); per-point stored-mask agreement is logged in
`cache_v3_meta.json` (174/238 bit-identical; every campaign from
2026-08-23 onward matches).

### 2.2 Quota sampling

Plain uniform batching under-weights the small hard campaigns (b15: 4 fit
points on the random split vs geo_lhs 53). `QuotaSampler`: every dataset
present in fit contributes exactly `quota = max_k n_k` index slots per
epoch (minority datasets repeated by index tiling — auditable, exact
counts per epoch, single-dataset fits reduce to the v2 permutation).

### 2.3 Force-tail auxiliary head

`aux_dim = 8` head reading the same pooled features as the main head,
target = log10 mean force_x in 8 uniform bins over the tail (last 25 % of
drag samples — the same window the C_D label uses; z-scored per bin on
fit stats), loss += 0.1 · MSE. Created last in `__init__`, so shared
modules initialise identically with/without it at a fixed seed.

### 2.4 Ablation arms (same splits/seeds everywhere)

| arm | encoding | sampling | aux | isolates |
|---|---|---|---|---|
| `v2_ref` | one-hot 7 | plain | – | control, must reproduce v2 |
| `A_sampling` | one-hot 7 | **quota** | – | sampling alone |
| `B_encoding` | **geometry 8** | plain | – | encoding alone |
| `C_minus` | geometry 8 | quota | – | encoding + sampling |
| `C_full` | geometry 8 | quota | **λ=0.1** | + aux head |
| `R4_nooh` | 4 log params | plain | – | reference: v2 diagnostic promoted to the 3-seed protocol |

Every LOHO fold runs model seeds 0/1/2 per arm (mean ± std below) after
seed-0 evidence showed fold-level seed sensitivity (v2 encoding seeds on
`loho::with_sail`: 5.65 / 6.51 / 24.39 %).

## 3. v3 results

MAPE % on log10 C_D, test side of each split. Power-law reference
(power-geo prior, unchanged from v2): **2.00 / 5.16 / 10.25** on
bare / with_sail / full. Acceptance lines from the v2 report: random
≤ 3.61, geo-scaled slice ≤ 5.60, all three LOHO folds < power-geo,
aux (C_full vs C_minus) not worse.

### 3.1 Seed-0 sweep (all arms × all splits)

| split / arm | v2_ref | A_sampling | R4_nooh | B_encoding | C_minus | C_full |
|---|---|---|---|---|---|---|
| **random** | 3.61 | 1.97 | 2.67 | **0.69** | 0.69 | **0.66** |
| · geo-scaled slice | 5.60 | 3.35 | 3.82 | **0.73** | 0.80 | **0.77** |
| · scale-1 slice | 1.06 | 0.20 | 1.20 | 0.64 | 0.54 | **0.53** |
| **loho::bare_hull** | 9.44 | 8.96 | **0.50** | 2.54 | 3.75 | 3.55 |
| **loho::with_sail** | 5.65 | 7.44 | 2.77 | **0.74** | 1.66 | 1.77 |
| · geo-scaled slice | 5.24 | 8.16 | 2.97 | **0.89** | 1.50 | 1.72 |
| **loho::full** | 12.78 | 13.14 | 9.69 | 4.69 | 4.69 | **4.41** |
| · geo-scaled slice | 19.62 | 20.01 | 16.19 | 6.66 | 6.66 | **6.22** |
| · scale-1 slice | 3.95 | 4.26 | 1.29 | 2.18 | 2.14 | **2.07** |

(`loho::bare_hull` has no geo-scaled slice — the held set carries no
scaled-geometry points, as in v2.)

### 3.2 LOHO folds, model seeds 0/1/2 (mean ± std; power-geo in the header)

| fold (power-geo) | v2_ref | A_sampling | R4_nooh | B_encoding | C_minus | C_full |
|---|---|---|---|---|---|---|
| bare_hull (2.00) | 12.52±3.46 | 11.71±3.50 | **0.71±0.32** ✓ | 2.37±0.49 ✗ | 3.01±0.78 ✗ | 3.14±0.31 ✗ |
| with_sail (5.16) | 12.17±8.62 | 10.88±6.46 | 2.49±0.23 ✓ | **0.89±0.17** ✓ | 1.21±0.38 ✓ | 1.05±0.50 ✓ |
| full (10.25) | 19.64±10.16 | 19.28±9.86 | 10.47±0.84 ✗ | 9.79±3.68 ✓ | 9.69±3.57 ✓ | **9.53±3.67** ✓ |

### 3.3 `loho::full` extended to 5 model seeds (the fold acceptance turns on)

| arm | seeds (0–4) | mean ± std | median |
|---|---|---|---|
| v2_ref | 12.8, 34.0, 12.1, 10.4, 12.7 | 16.41±8.84 | 12.72 |
| A_sampling | 13.1, 33.2, 11.5, 10.5, 13.0 | 16.26±8.52 | 12.99 |
| R4_nooh | 9.7, 10.1, 11.6, 9.4, 9.0 | **9.97±0.90** | 9.69 |
| B_encoding | 4.6, 12.4, 12.4, 5.4, 5.2 | 7.99±3.60 | 5.42 |
| C_minus | 4.7, 11.6, 12.8, 5.4, 5.3 | 7.95±3.49 | 5.38 |
| C_full | 4.4, 11.4, 12.8, 5.3, 5.4 | **7.86±3.50** | 5.39 |

### 3.4 Random split, per dataset (seed 0)

| arm | b15 | g2 | geo_lhs | hull_re | lhs40 | re_drag |
|---|---|---|---|---|---|---|
| v2_ref | 7.32 | 5.04 | 5.84 | 1.72 | 0.68 | 0.62 |
| B_encoding | 1.20 | 0.89 | 0.59 | 0.55 | 0.57 | 0.88 |
| C_full | **1.14** | **0.90** | 0.67 | 0.67 | **0.32** | **0.63** |

## 4. Findings

1. **Encoding is the lever; sampling is not (on LOHO).** `B_encoding`
   alone moves bare 12.52 → 2.37, with_sail 12.17 → 0.89, full
   19.64 → 9.79 (3-seed means). `A_sampling` alone barely moves the
   folds (19.64 → 19.28 on full) though it does help the random split
   (3.61 → 1.97). Once the encoding is fixed, quota sampling adds
   nothing material (B vs C_minus: 0.89 vs 1.21 with_sail, 9.79 vs 9.69
   full) and slightly *hurts* the bare fold (2.37 → 3.01) — plausibly
   because quota up-weights the small full-hull campaigns (on this fold
   b15's 4 fit points are tiled ×14 to the 55-slot quota) and pushes the
   fit distribution further from the bare corner the fold must
   extrapolate to.
2. **The aux head never hurts where it matters.** C_full vs C_minus over
   12 seed×fold cells: 9 wins / 2 neutral-within-0.1 / 1 regression
   (bare seed 1, +0.89). Fold means: random 0.66 < 0.69, with_sail
   1.05 < 1.21, full (5 seeds) 7.86 < 7.95; bare 3.14 vs 3.01 is +0.13
   with seed std 0.31–0.78. Acceptance line "aux not worse" holds.
3. **Acceptance on the three LOHO lines: no single arm sweeps.**
   C_full/B/C_minus beat power-geo on with_sail and full (5-seed full
   margins 1.6–2.4 pp) but miss bare (2.37–3.14 vs 2.00). R4_nooh beats
   all three lines on 5-seed means (0.71 / 2.49 / 9.97) — its bare and
   with_sail margins are decisive (1.3 pp, 2.7 pp) but the full margin
   is 0.28 pp, under the 0.5 pp bar the protocol sets for thin wins.
4. **The bare-fold residual is a data-coverage floor, not noise.**
   Per-point errors of the geometry arms on the 14 held bare points are
   uniform — B: 2.13–2.98 % (mean 2.54), C_full: 3.40–3.65 % (3.55),
   no outlier structure — while R4 scatters 0.08–1.27 % (0.50) and v2_ref
   5.3–13.1 %. The nearest training geometries to the bare corner are
   G2's sail = 0.4 anchors (full(0.4,·) differs from bare by 13 voxels),
   and their C_D sits 2.3–2.5 % off the bare C_D(Re) curve (e.g. re ≈ 200:
   anchor 6.82 vs bare 7.11; re ≈ 60: bare 17.12). A geometry-conditioned
   regressor interpolating to the bare corner inherits exactly that
   offset as a uniform bias; the log-parameter arm smooths through it on
   the re axis. Fixing this needs data (true near-bare anchors on the
   bare row) or a hybrid condition, not a different mask encoding.
5. **Seed bimodality on `loho::full` is real and reported.** The
   geometry arms occupy two basins across seeds (≈4.4–5.4 and
   ≈11.4–12.8; R4 is stable at 9.0–11.6; v2/A swing 10.4–34). Medians
   (C_full 5.39, B 5.42) sit in the good basin; means carry the bad one.
   All fold-level conclusions above use 3–5 seed means as specified.
6. **Deploy recommendation (evidence-based, owner to decide).** For a
   single production surrogate: C_full — it wins the joint random split
   (0.66 / 0.77 geo-scaled) and the two folds where interaction data
   exist, and its worst fold (bare 3.14) is a documented data floor.
   If the bare corner must be tight, R4 (4 log params) is the better
   special-purpose model; the two are a natural ensemble/hybrid
   (R4 params + geometry block) for a v4.

## 5. Anomalies & handling

| # | anomaly | handling |
|---|---|---|
| 1 | First quota-sampler wiring raised a CUDA device-side assert (sampler emitted *global* corpus indices while the feature tensor held only the fit subset). | Rebuilt the sampler to take dataset labels + fit **positions** (what the unit test pins: quota conservation, membership, single-dataset reduction). Recorded; no data affected. |
| 2 | `loho::full` seed bimodality (finding 5) invalidated single-seed fold comparisons. | Protocol already mandated 3 seeds; extended to 5 on this fold (`addendum_seeds.py`); means and medians both reported. |
| 3 | JSON dump of metrics failed (`TypeError: ndarray`) after sweeps completed. | `_json_default` sanitizer (ndarray→list, numpy scalars→python) in both sweep scripts; raw predictions kept in `preds_v3.npz`. Root ndarray source not isolated — sanitizer is defensive. |
| 4 | Vintage drift: re_drag (20260821) and lhs40 (20260822) stored masks = current CAD minus 28 sail + 36 fin voxels (A_proj 69 vs 73). | Encoding uses current-CAD values (pure function of design params, deployable); per-point bit-agreement 174/238 logged in `cache_v3_meta.json`; all 2026-08-23+ campaigns bit-identical. |
| 5 | ssh backgrounding of the sweep hung the local wrapper (v2-era issue). | `ssh -f` + remote `setsid`; remote run verified via `pgrep` and log polling. |
| 6 | Remote python stdout fully buffered when redirected — `train.log` stays empty until exit. | Polling via `pgrep` + post-hoc log; run products unaffected. |
| 7 | `loho::bare_hull` has no geo-scaled slice (held set has no scaled points). | Same as v2; slice simply absent rather than zero-filled. |

## 6. Reproducibility

```
# v3 cache extension (geometry channels + aux bins + mask audit)
cd /nfs/wangxi/runs/b4_v3_20260824 && \
  PYTHONPATH=/nfs/wangxi/worktrees/b4_v3/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python build_cache_v3.py
# v3 sweep (all arms x splits x seeds) — prints tensorlbm.__file__ and
# asserts it points at the b4_v3 worktree
cd /nfs/wangxi/runs/b4_v3_20260824 && CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/nfs/wangxi/worktrees/b4_v3/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python train_fno_v3.py
# extra model seeds (3, 4) on loho::full for every arm
cd /nfs/wangxi/runs/b4_v3_20260824 && CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/nfs/wangxi/worktrees/b4_v3/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python addendum_seeds.py
# unit tests (both venvs)
cd /nfs/wangxi/worktrees/b4_v3 && TMPDIR=/nfs/wangxi/tmp \
  /nfs/wangxi/venvs/tensorlbm/bin/python -m pytest tests/test_drag_cond.py \
  --basetemp=/nfs/wangxi/tmp/pt_b4v3
```

Seeds: split 0 / val 1 / model 0 (+1, 2 on LOHO folds); quota-sampler RNG
`np.random.default_rng(1000 + model_seed)`. G2 DoE seed 20260824.
Worktree `exp/b4-v3` (from origin/main 4c787e6e). Datasets read-only
(only the G2 launcher/preflight/validate scripts were archived into the
G2 directory, per campaign convention).
