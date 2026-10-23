# Drag surrogate conditioning: G2 interaction data (v2) → physics-geometry encoding (v3) → G2b bare closure (v4)

Date: 2026-08-24 (v2 campaign + protocol; v3, v4 same day) · Module:
`tensorlbm.ai.drag_cond` (new) + `tensorlbm.ai.fno.SpectralConv2d` · Runs:
`/nfs/wangxi/runs/b4_v2_20260824` (v2, report.md there),
`/nfs/wangxi/runs/b4_v3_20260824` (v3) and `/nfs/wangxi/runs/b4_v4_20260824`
(v4) · Datasets: `scan_suboff_hull_scale_g2_20260824` (G2, 48 points) and
`scan_suboff_hull_scale_g2b_20260824` (G2b, 36 points; v4 —
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
>
> **v4 addendum.** The floor was data, and the data closed it. The sail
> scales down by similarity about the deck plane; below **s\* = 0.133**
> (capture threshold in (0.1333, 0.1335))
> the `with_sail` voxel mask is **bit-identical to the bare hull** (the
> sub-voxel bump still exists continuously but captures no voxel centre),
> so `with_sail` runs at sail ≤ 0.13 are *physically bare simulations*
> — verified bit-for-bit against the hull_re bare twins (160/160 force
> samples equal, C_D identical to 6 decimals). G2b (36 pts: a 1/1/2/3/5
> net-sail-voxel ladder at sail ∈ {0.15…0.40} × 6 re + 6 sub-s\* points,
> every re an exact hull_re bare twin) puts certified bare-equivalent
> anchors on the train side of `loho::bare_hull`. Result: the geometry
> arms go 2.37/3.14 → **2.02 ± 0.49 / 2.13 ± 0.59**, under the
> recomputed power-geo line (2.00 → 2.53 on the 274-pt corpus); the
> per-point bias halves (C_full +3.55 % → +1.80 %); `R4_nooh` collapses
> to **0.23 ± 0.18**; nothing regresses materially (random 0.56/0.65,
> full 7.48/8.42 vs 11.66). The R4×geometry log10-ensemble passes all
> three LOHO lines on fold means (0.92 / 3.74 / 7.95 vs 2.53 / 4.70 /
> 11.66) but not per-seed. One v3 statement is corrected: the sail=0.4
> anchors sit **2.3–2.6 % below** the bare C_D(Re) curve (an A_proj
> 69→71 normalisation effect), not above — the v3 over-prediction was
> extrapolation past the last anchor, not interpolation of a positive
> offset.

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

## 7. v4 — G2b: closing the bare-fold data floor (same day)

Run `/nfs/wangxi/runs/b4_v4_20260824` · dataset
`/nfs/wangxi/datasets/scan_suboff_hull_scale_g2b_20260824` (36 pts) ·
worktree `exp/b4-v4` (from origin/main ca9e0c4d; `scan_runner` /
`scan_drag` / `suboff_cad` byte-identical to the b16_scan tree G2 ran
on, verified by diff). Training/protocol byte-identical to v3 — the only
change is the corpus (238 → 274) and the arms actually rerun
(`B_encoding`, `C_full`, `R4_nooh` + prediction-level ensembles; the
238-pt corpus arm *is* v3).

### 7.1 s\* preflight: the sail-disappearance scale

The sail shrinks by similarity about `(x_center, centreplane, deck
plane)`. Sweeping `sail_scale` 0.40 → 0.02 (0.005 step + 0.001
refinement) on the production grid, the net sail voxel count is a
perfectly monotone staircase:

| sail_scale | ≤ 0.1333 | 0.134–0.290 | 0.295–0.320 | 0.325–0.385 | ≥ 0.390 |
|---|---|---|---|---|---|
| net sail voxels | **0** | 1 | 2 | 3 | 5 |
| A_proj | 69 | 70 | 70 | 70 | 71 |

**s\* = 0.133** at 0.001 sweep resolution (the voxel-capture threshold
itself lies in (0.1333, 0.1335) — 4-decimal probe): below it the
`with_sail` mask, A_proj and all four
geometry channels equal the bare hull exactly (solid 4093, `sail_frac`
0, `log_aproj_ratio` 0, `solid_frac` 1) — pinned by the new
`TestSubStarBareEquivalence` in `tests/test_drag_cond.py`. The
continuous predicate never becomes empty (sub-voxel probe at 1/8 lu:
bump volume 0.004 → 1.35 lu³, z-extent 0 → 1.5 lu over s = 0.02 → 0.4),
so the disappearance is **voxel quantisation, not geometric
submergence** — which is exactly what makes sub-s\* runs *physically
identical* to bare runs rather than merely similar.

### 7.2 G2b campaign and bit-identity certification

Design (36 pts, deterministic grid — no LHS): stratum M = 5 sail levels
{0.15, 0.20 (1 vox), 0.30 (2), 0.35 (3), 0.40 (5, the G2 anchor
geometry)} × 6 re log-spread over [60, 703.5]; stratum V = 2 sub-s\*
levels {0.13, 0.05} (both below s\* = 0.133 by ≥ 3.3e-3) × 3 re.
**Every re value is copied exactly
from a `hull_re` bare_hull point** (asserted at launch), so every G2b
point has a same-re bare twin — the ladder measures the physical
sail increment directly, and the sub-s\* points can be certified
against existing runs. Chain identical to G2 (suboff_n128, cumulant,
4 000 steps, `DragSurveySpec(margin=4, interval=25)`, u_in = 0.1, tau =
0.5 + 23.04/re ∈ [0.533, 0.884]); smoke 3 pts @300 steps first, then
the 6 sub-s\* points @4 000 **before** committing the remaining 30
(the task's verification gate).

**Bit-identity (the premise, certified)**: all 6 sub-s\* points match
their bare twins with mask XOR = 0, **160/160 force samples exactly
equal, max |ΔF| = 0.0, C_D identical to 6 decimals** (e.g. re 195.44:
7.111814 vs 7.111814). The solver is bit-deterministic across the
campaign vintages — a sub-s\* `with_sail` point *is* the bare point,
relabelled. These are certified free bare anchors on the train side of
`loho::bare_hull` (they carry `hull = with_sail`), at 3 of the 14 held
re values.

Validation: 36/36 completed @4 000 steps (226 s on 8 GPUs + 54 s
verify), max tail drift 5.1e-5, C_D ∈ [2.974, 17.121], all within
[0.5, 1.5]× of the bare twin. The measured ladder vs same-re bare
twins:

| sail (net vox) | C_D offset vs bare twin (re 60 → 703) |
|---|---|
| ≤ 0.13 (0) | +0.00 % (bit-identical) |
| 0.15–0.29 (1) | −1.38 % … −1.34 % |
| 0.30 (2) | −1.36 % … −1.31 % |
| 0.35 (3) | −1.34 % … −1.29 % |
| 0.40 (5) | −2.57 % … −2.30 % |

**Correction to v3 finding 4.** The nearest anchors sit *below* the
bare C_D(Re) curve — the +1 voxel already adds an A_proj column
(69 → 70, +1.45 %) while adding ~0.01 % force, and the 5-voxel anchor
adds +2.9 % area for +0.33 % force. v3's "2.3–2.5 % above" reading of
the anchor offset had the sign wrong (the underlying G2 numbers — e.g.
anchor 6.82 vs bare 7.11 at re ≈ 200 — already pointed down). The v3
uniform **over**-prediction (+2.1–3.0 %) of the bare fold was therefore
*extrapolation past the last anchor* (the with_sail family's C_D
decreases with sail near the corner), not interpolation of a positive
offset; both stories predict a uniform bias, which is why v3 could not
distinguish them from inside the corpus.

### 7.3 v4 results (274-pt corpus; power-geo lines refit on it)

Power-geo reference moves with the corpus: **2.00 → 2.53 / 5.16 → 4.70
/ 10.25 → 11.66** (bare / with_sail / full). MAPE %, mean ± std over
model seeds (bare and full folds: seeds 0–4; with_sail: 0–2; random:
seed 0):

| fold (power-geo) | B_encoding | C_full | R4_nooh | E2(R4+B) | E2(R4+C) |
|---|---|---|---|---|---|
| **random** (5.32) | 0.65 | **0.56** | 2.55 | 1.38* | 1.33 |
| **loho::bare_hull** (2.53) | 2.02 ± 0.49 ✓ | 2.13 ± 0.59 ✓ | **0.23 ± 0.18** ✓ | 0.92 ± 0.32 ✓ | 0.98 ± 0.32 ✓ |
| **loho::with_sail** (4.70) | 1.58 ± 0.29 ✓ | 1.39 ± 0.45 ✓ | 6.25 ± 1.92 ✗ | 3.74 ± 1.13 ✓ | 3.50 ± 1.44 ✓ |
| **loho::full** (11.66) | 7.48 ± 4.28 ✓ | 8.42 ± 4.85 ✓ | 10.64 ± 1.55 ✓ | 7.95 ± 3.56 ✓ | 8.19 ± 3.93 ✓ |

\* E2(R4+B) cells are recomputed post-hoc from `preds_v4.npz` with the
same log10-mean pairing; the in-sweep ensemble arm (`E2_r4_cfull` in
`metrics_v4.json`) pairs R4+C_full.

**Bare fold (the judgement).** v3: B 2.37 ± 0.60, C 3.14 ± 0.38 —
no geometry arm under the 2.00 line. v4: **B 2.02 ± 0.49, C_full 2.13
± 0.59, both under the recomputed 2.53 line** (B ties the old 2.00
line within one std). Per-point signed error (same 14 held points,
seed 0): B +2.54 % → +2.08 %, C_full **+3.55 % → +1.80 %** (range
+1.39…+2.27), R4 −0.50 % → −0.37 %. The residual is still uniform —
but it now sits *below* the 1-voxel anchor offset rather than above
the extrapolation edge, and it persists even at the 3 re values that
have bit-identical training twins (C_full: +1.39/+1.67/+2.20 at re
60/195/613), i.e. what remains is model bias at the (log sail = 0,
geometry = bare) input combination — training never shows bare
geometry with log sail > −0.89 — not a coverage hole.

**R4 collapse.** The 4-log arm goes 0.71 ± 0.40 → **0.23 ± 0.18**
(seeds 0.06–0.48): the ladder densifies log-sail → C_D along the exact
row the fold holds. Its with_sail weakness worsens (2.49 → 6.25, and
8.68 on the g2b slice alone): log-sail cannot represent the quantised
ladder, where sail 0.15/0.20/0.30 map to near-identical masks.

**Non-regression.** random: 0.65 / 0.56 (geo-scaled 0.67 / 0.65) vs
v3 0.66 / 0.77 — holds. full (5 seeds): 7.48 / 8.42 vs v3 7.86 / 9.79
(medians improve; bimodality persists, seeds 1–2 in the 10–14 basin).
with_sail: the fold's held set grew 36 → 72 (all G2b with_sail points
are held by definition), so like-for-like numbers are computed on the
v3-era subset: **B 0.89 → 0.74** (improves), C_full 1.05 → 1.77 (mild
regression; on the full 72-pt held set 1.39 ± 0.45, still 3.3 pp under
its line). The g2b slice is the hardest for every arm (B 2.1 %,
C 1.75 %, power-geo 4.23 %) — the near-bare corner is intrinsically
the regime that needs the geometry channels.

**Ensemble (prediction-level, log10 mean).** E2(R4+geo) passes all
three LOHO lines on fold means: 0.92 / 3.74 / 7.95 (R4+B) vs 2.53 /
4.70 / 11.66. Per-seed it does not sweep: with_sail seed 1 = 5.01 ✗,
full seed 2 = 11.96 ✗ (R4+B; R4+C fails 2 of 5 on full). The ensemble
averages away part of the geometry arms' seed bimodality but inherits
both partners' bad seeds, and R4's with_sail weakness caps the gain
there. Verdict: means pass, single-seed reliability does not — the
v3 deploy picture (geometry arm primary, R4 as bare-corner specialist)
is unchanged, now with both beating their lines on the closed corpus.

### 7.4 v4 anomalies & handling

| # | anomaly | handling |
|---|---|---|
| 1 | GitHub unreachable from the 5090 during setup (`Empty reply`); the `ca9e0c4d` objects present locally were an interrupted fetch's leftovers with a missing tree. | Retried fetches left the objects complete (fsck-clean); worktree `reset --hard ca9e0c4d` then succeeded. `gh-proxy.com` mirror confirmed as working fallback for future fetches. Recorded; no repo state touched beyond the mandated `fetch` + `worktree add`. |
| 2 | v3 §4-4 sign error (anchors below, not above, the bare C_D curve). | Corrected in §7.2 with the direct same-re twin ladder; v3 numbers unchanged (they measured |error|). |
| 3 | `ScanPlan` requires contiguous point indices 0..n-1, breaking naive subset launches with global ids. | Subsets carry local indices but global `point_id`/`run_id`; resume-by-point-id across phases verified (verify 6 pts → full 36 skipped them). |
| 4 | `ScanVariable` demands low < high; fin_scale is a pinned no-op on with_sail. | Removed from the plan's variables tuple (documented in `method`); per-point params still carry `fin_scale: 1.0`. |
| 5 | First preflight run printed a wrong "monotonicity VIOLATIONS" flag (test inequality reversed; the ladder itself is monotone). | Check direction fixed; preflight re-run clean and re-archived. No data affected. |
| 6 | `loho::with_sail` held set grows 36 → 72 with G2b (all with_sail held by definition), making raw v3→v4 fold deltas non-comparable. | Like-for-like evaluation on the v3-era subset added (`analyze_v4.py` §4); both readings reported. |
| 7 | Preflight refinement bug, caught by the new unit test: the 0.001 refinement seeded from `min(zeros)` (smallest zero scale) instead of `max(zeros)`, so it swept 0.021–0.024 and the reported s\* stayed the coarse-grid 0.130; and the summary line read the pre-refinement zero list. True s\* = 0.133 (threshold in (0.1333, 0.1335)). | Fixed both lines, re-ran (s\* = 0.133, ladder PASS, no nonzero xor below s\*), re-archived script + output into the dataset dir. **No campaign impact**: sub levels 0.13/0.05 remain below the true boundary by ≥ 3.3e-3, main levels ≥ 0.15 above it, and the bare-equivalence certification was always the empirical bit-identity of the runs, not the s\* label. |

### 7.5 v4 reproducibility

```
# s* preflight (mask-only, ~3 s)
cd /nfs/wangxi/runs/b4_v4_20260824 && PYTHONPATH=/nfs/wangxi/worktrees/b4_v4/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python g2b_preflight.py
# G2b campaign: smoke -> verify (bit-identity gate) -> all
cd /nfs/wangxi/runs/b4_v4_20260824 && PYTHONPATH=/nfs/wangxi/worktrees/b4_v4/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python g2b_launch.py --smoke
cd /nfs/wangxi/runs/b4_v4_20260824 && PYTHONPATH=/nfs/wangxi/worktrees/b4_v4/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python g2b_launch.py --subset verify
cd /nfs/wangxi/runs/b4_v4_20260824 && PYTHONPATH=/nfs/wangxi/worktrees/b4_v4/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python g2b_validate.py --subset verify
cd /nfs/wangxi/runs/b4_v4_20260824 && PYTHONPATH=/nfs/wangxi/worktrees/b4_v4/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python g2b_launch.py --subset all
cd /nfs/wangxi/runs/b4_v4_20260824 && PYTHONPATH=/nfs/wangxi/worktrees/b4_v4/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python g2b_validate.py
# cache (274 pts) + sweep + analysis
cd /nfs/wangxi/runs/b4_v4_20260824 && PYTHONPATH=/nfs/wangxi/worktrees/b4_v4/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python build_cache_v4.py
cd /nfs/wangxi/runs/b4_v4_20260824 && CUDA_VISIBLE_DEVICES=2 \
  PYTHONPATH=/nfs/wangxi/worktrees/b4_v4/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python train_fno_v4.py
cd /nfs/wangxi/runs/b4_v4_20260824 && /nfs/wangxi/venvs/tensorlbm/bin/python analyze_v4.py
# unit tests incl. the sub-s* ladder pin (both venvs)
cd /nfs/wangxi/worktrees/b4_v4 && TMPDIR=/nfs/wangxi/tmp \
  /nfs/wangxi/venvs/tensorlbm/bin/python -m pytest tests/test_drag_cond.py \
  --basetemp=/nfs/wangxi/tmp/pt_b4v4
```

Seeds as v3 (split 0 / val 1 / model 0–4 on bare+full, 0–2 on
with_sail); G2b is a deterministic grid (seed 20260824 recorded for
provenance only). Artifacts: `cache_v4.npz` / `cache_v4_meta.json`,
`metrics_v4.json` (67 rows), `preds_v4.npz`, `train_v4.log`,
`analyze_v4.py`. Datasets read-only; the three campaign scripts are
archived in the G2b dataset directory per convention.
