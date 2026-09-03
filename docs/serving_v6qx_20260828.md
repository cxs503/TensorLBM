# Drag-surrogate serving refresh — cond_v6 "v6qx" pool (2026-08-28)

- Status: **recommended serving configuration** for the drag surrogate
  (B4 line), written down from the wave-12/13 campaign verdicts.
- Provenance: PRs [#257](https://github.com/cxs503/TensorLBM/pull/257)
  (UQ temperature knob), [#258](https://github.com/cxs503/TensorLBM/pull/258)
  (v5 sail axial channel), [#260](https://github.com/cxs503/TensorLBM/pull/260)
  (two-stage SDF encoder), [#262](https://github.com/cxs503/TensorLBM/pull/262)
  (condition_v6). Campaign docs:
  [cond_v6_20260828.md](cond_v6_20260828.md),
  [sdf_two_stage_20260828.md](sdf_two_stage_20260828.md),
  [uq_temperature_serving_20260827.md](uq_temperature_serving_20260827.md),
  [v5_fam_20260828.md](v5_fam_20260828.md).
  Machine-readable evidence lives in the run directories listed in
  [§5](#5-provenance) (read-only).

## TL;DR

| setting | value |
|---|---|
| serving pool | **cond_v6 v6qx 10-seed ensemble (s0–s9) on the 382-row corpus** |
| `cond_version` | `"v6"` (13-column "v6qx" vector; see [§1](#1-recommended-pool)) |
| `uq_temperature` | **1.5** (`TENSORLBM_DRAG_UQ_TEMPERATURE=1.5`) |
| slender-class designs | **route to the two-stage SDF path — never the cond path** (see [§3](#3-serving-rules-hard-limits)) |
| out-of-support queries | **defended by the envelope guard, not by the uncertainty band** |
| new axis values | **k = 3 spread-logRe anchors into the SDF corpus** (see the recipe under §3) |

## 1. Recommended pool

The serving pool is the cond_v6 campaign winner: the 13-column "v6qx"
conditioning vector (v5's 9 columns + `[log10(sail_x_mult)²,
log10(l_over_d_mult), log10(nose_len_mult), log10(stern_len_mult)]`),
trained as a 10-seed ensemble on the 382-row corpus. `cond_version="v6"`
selects this path — `predict_design(..., cond_version="v6")` builds
`hullform_condition_rows_v6` rows `(N, 13)`, and `corpus_with_cond(...,
cond="v6")` / `retrain_ensemble(..., cond="v6")` are the matching corpus /
training entries (all defaults unchanged; PR #262).

Artifacts (read-only inputs; `cond_v6_20260828` run directory):

- checkpoints: `/nfs/wangxi/runs/cond_v6_20260828/ckpts/v6qx/al_aug_s0..s9.pt`
  (the directory also holds s10–s19 from a sibling session — the
  recommendation is the 10-seed pool, see [§4](#4-what-is-not-recommended));
- corpus: `/nfs/wangxi/runs/cond_v6_20260828/corpus382.npz`
  (`corpus_verify.json`: 382 rows; `sail_x_mult` spans 0.7–1.4 with
  333 rows at 1.0; `l_over_d_mult` = 1.0×326 + 1.10×28 + 1.20×28);
- evaluation evidence: `eval_v6qx.json`, `report.md` in the same directory,
  and `docs/cond_v6_20260828.md` (four-arm table).

Service wiring follows the existing path — no code change is required:
build the service with
`DragSurrogateService.from_checkpoints(..., uq_temperature=1.5)`
(see [§2](#2-uq_temperature--15); `src/tensorlbm/ai/inference_service.py`)
and serve queries with `predict_design(..., cond_version="v6")`
(`src/tensorlbm/ai/active_learning.py`).

Quality (10 seeds, 382-row corpus; from `eval_v6qx.json` /
`docs/cond_v6_20260828.md`):

| metric | value |
|---|---|
| trend ext ratio (median per-Re) | **0.992** — per-Re [0.984, 0.992, 1.007] |
| canon ratio (sail_x 1.0/1.3) | 0.905 |
| B-grid 5-point sweep ratio | 0.889 |
| test-55 MAPE | 0.511 % |
| fresh old12 / A / B MAPE | ≤ 0.14 % (0.136 / 0.044 / 0.085 %) |
| fresh-M MAPE | 0.515 % |
| negative members (ext sweep) | 0 |

B-grid gate-band annotation (2026-08-30): the regression-gate band applied
to the B-grid 5-point sweep ratio in the campaign gate tooling is
[0.90, 1.10], but the reference value in this table (0.889) itself sits
below the 0.90 floor — the band fails its own reference, so treat the
B-grid gate as **advisory** until the band is re-baselined (candidate: a
band centered on the reference value with a stated tolerance, e.g.
0.889 ± 0.05; decision pending). Evidence: of the four B-grid ratios
measured on this corpus family — reference 0.889 and the A-1 arms
0.857 / 0.926 / 0.914 — the current band excludes two (0.889, 0.857),
including the reference itself, while all four sit inside ref ± 0.05 =
[0.839, 0.939]. The band is annotated here, not changed.

### Trend-slope convention

Every slope in the tables above (and in the campaign docs) is measured in
**log10(C_D) / d(sail_x_mult)**; a "ratio" is the **median per-Re
pred/truth slope ratio**. The sail_x truth slope is **negative** — drag
decreases as the sail moves aft — so a ratio of 0.992 means the predicted
trend magnitude is within 0.8 % of truth, per Re, with the correct sign.

## 2. uq_temperature = 1.5

The #257 knob scales **only the reported ensemble std** (verdicts, member
min–max band and predictions are untouched; semantics audited in
[uq_temperature_serving_20260827.md](uq_temperature_serving_20260827.md)).
For this pool serve with

```bash
export TENSORLBM_DRAG_UQ_TEMPERATURE=1.5   # or uq_temperature=1.5 argument
```

so the #257 review band `mean ± 2.3·std·t` becomes **±3.45σ**.

Basis (wave-13 W2, 10-member pool s0–s9, `/nfs/wangxi/runs/v6_uq_20260828/`):
the implied 95 % multiplier `m` with `P(|err| ≤ m·std) = 0.95` is

| query class | m95 (×std) |
|---|---|
| test-55 (in-distribution) | 2.54 |
| fresh-M (M-anchor middles) | 3.37 |
| B-grid 5-pt sweep | 3.32 |
| labeled sail_x groups (old12/A/B) | ≤ 0.96 |
| slender 28 (l_over_d 1.30, out of support) | 14.9 — unreachable, see §3 |

The ±3.45σ band is therefore a **≥ 95 % band on every guard-passing query
class measured** (pooled excluding slender: 99.3 %, n = 139). The member
std is **informative in-distribution** (Spearman ρ(std, |err|) = +0.57 on
test-55) and **uninformative on bias-shaped classes** (fresh-M / B-grid
ρ ≈ 0) — the 1.5 setting is driven by those classes, whose misses sit in
the *low*-std half at t = 1.0.

Known cost (accepted): on test-55 the band widens to a 98.2 % band
(1 miss in 55) — ~3 pp over-conservative, the price of one knob for all
regimes. A per-class exact-95 % temperature does not exist: the per-regime
t95 spread among guard-passing classes is 0.18–1.47.

**Re-review 2026-09-03 (value unchanged).** A bundle-pool calibration
sweep (20 `ckpt_bundle` members, 406 corpus rows = 381 pool + 25 held;
see §5) measured on exactly this population: cov90 = 96.6 / 99.3 / 100 %
at t = 1.0 / 1.5 / 2.0 (mean 90 % band widths 0.118 / 0.177 / 0.236) and
rms-z 0.727 on held rows — the raw std **over-disperses on corpus rows**
(322 of 381 pool rows are trained-on). That is the in-envelope regime the
#251 audit already marks "keep T = 1.0", and it does **not** move the
deployment value: the 1.5 setting is driven by the fresh-M / B-grid
classes (bias-shaped, std uninformative, misses in the low-std half),
which the corpus sweep does not sample. Corpus-row coverage numbers must
not be used to argue a global temperature — recorded here to pre-empt
exactly that reading.

## 3. Serving rules (hard limits)

### Rule 1 — slender-class designs route to the SDF two-stage path

Slender-class designs — **l_over_d ≥ 1.10 beyond corpus support** (the
corpus carries the axis only to 1.20; the measured failure class is
**l_over_d 1.30**) — are a **constant multiplicative bias** on the v6qx
path, not a variance problem: error is a flat **−18.7 %** (per-row
pred/true ratio 0.813 ± 0.006), and covering 95 % of it would need
**±14.9σ** — no temperature fixes it (t ≈ 6.5 would inflate every honest
regime's band ~4.3× beyond its calibrated width). The defense is layered:
the envelope Mahalanobis guardrail **rejects these queries 28/28**, and
the correct serving answer is the **two-stage SDF path** (#260), which
holds **10.71 %** held-out MAPE on the slender class
(`/nfs/wangxi/runs/sdf_slender_20260828/`). Serving rule: slender-class
designs must route to the SDF two-stage path, never the cond path.

Refinement (2026-08-28 bias adjudication,
`/nfs/wangxi/runs/sdf_bias_20260828/`): the 10.71 % is a **support-hole
artifact, not a representation limit**. At the same axis value 1.30, with
partial training support (22 of 28 rows) the two-stage path reaches
**0.209 %**; the ×0.894 offset appears only when the axis loses all
training support, and a globally fitted lambda is a no-op (10.71 →
10.70 %). Operational consequence: for a new axis value, acquire a **k = 3
spread-logRe anchor set** (recipe below) into the SDF training corpus
instead of shipping any calibration layer: measured held-out error
collapses from 10.71 % to **0.411 %** (ts2) / **0.214 %** (ts4), 3-seed
medians, and the full acquisition step costs under 10 min on one 5090.

### Recipe — k = 3 anchor acquisition for a new axis value (2026-08-29)

Verified recipe (wave-15 W2, `/nfs/wangxi/runs/anchor_min_20260829/`,
24 cells x 2 arms x 3 seeds, every per-seed number independently
recomputed from `runs/<cell>/preds.npz`): to bring a **new axis value**
(e.g. a fresh l_over_d) into support, scan **3 anchor rows** at the
**min / geometric-mid / max of the intended query range in log10(Re)**
and add them to the SDF training corpus. Held-out MAPE at slender
l_over_d 1.30, median over seeds:

| anchors added | ts2 | ts4 | note |
|---|---|---|---|
| k = 0 | 10.776 % | 9.567 % | reproduces the support hole; lambda = 0.893 |
| k = 1 | 5.886-11.162 % | 5.155-12.196 % | by position; Re-slope stays wrong |
| k = 2 spread | 0.520 % | 0.390 % | [63.2, 654.2] |
| k = 2 adjacent | 3.574 % | 3.089 % | [190.2, 205.9] — span, not count |
| **k = 3 spread** | **0.411 %** | **0.214 %** | [63.2, 205.9, 654.2] = min/geo-mid/max |
| k = 6 spread | 0.396 % | 0.268 % | saturation from k >= 2 |
| k = 12 / k = 22 spread | 0.376 / 0.274 % | 0.307 / 0.254 % | diminishing |

Mechanism (two cliffs, both on **anchor log10-Re span**, not row count):

1. **offset cliff** — a single anchor (k = 1) does not restore the
   Re-response: the slope of log10(pred/true) vs log10(Re) stays at
   **+3 to +13 % per decade** (position- and seed-dependent) and
   lambda lands anywhere in 0.84-1.13 depending on anchor position
   (mid 0.99, low 1.05-1.13, high 0.84-0.92);
2. **slope cliff** — k >= 2 anchors with **>= 0.4 decade span** collapse
   the slope to -0.8 to +0.1 % per decade and lambda lands at
   **1.000 +/- 0.003** (k3-spread seeds 1.0031 / 0.9991 / 0.9994).

Random draws are acceptable **only if the span rule holds**: the single
failing k = 3 draw (span 0.32 decade) gives 1.152 %, while every draw
with span >= 0.64 decade lands at 0.37-0.51 %. Cost: 22.8 GPU-s per CFD
scan point (from the slender scan logs), so the whole acquisition step
— 3 scans plus corpus/training refresh — fits in ~7 min on one 5090.

For slender 1.30 specifically nothing new needs scanning: the 28 rows
already exist as held-out data, and the 3 anchor rows (Re 63.2 / 205.9 /
654.2, corpus indices 246 / 252 / 257) promote the **378-row hole pool**
(the 406-row ext corpus minus its 28 slender-1.30 rows) to a **381-row
production corpus** (fit 325 / val 56 / anchors 3).

10-seed production confirm (wave-16, `/nfs/wangxi/runs/anchor_promo_20260829/`,
protocol bit-identical to the k3-spread cell, seeds s0-s9, anchors and
corpus 9/9 bit-checked against the wave-14/15 caches): held-25 MAPE
median **0.237 %** ts2 [0.107, 0.805] / **0.309 %** ts4 [0.146, 0.923],
no seed above 0.93 %; per-seed held lambda in [0.9977, 1.0090];
in-support lambda pooled 0.99972 / 1.00025. The 3-seed wave-15 plateau
holds at 10 seeds. One disclosure: read per-seed on already-fit rows
the in-support lambda dips to 0.99785 on 5 of 20 arm-seed cells
(checkpoint wobble on fit rows; those same seeds hold 1.000 +/- 0.003
on the held-out 25), and the two 0.92 % ts4 seeds are flat
lambda ~ 1.008 tilts, not Re-ramps. The corpus file keeps the name
`corpus353.npz` for lineage but contains the honest 381 rows.
Selection is executable: `tensorlbm.ai.anchor_selection`
(`anchor_targets` / `validate_span` / `match_anchor_rows`).

### A-1 verdict — blunt (l_over_d 0.75) anchors on the cond path (2026-08-30)

Verified micro-experiment (wave-17 A-1,
`/nfs/wangxi/runs/blunt_anchor_20260830/` — 10 seeds per arm, v6qx pool
machinery on a main-7b406a10c worktree, anchors drawn with the PR #266
`tensorlbm.ai.anchor_selection` API):

- **The k = 3 spread recipe transfers FULLY to the blunt axis on the cond
  path.** The 382-row v6qx corpus plus 3 blunt (l_over_d 0.75) anchors at
  Re 63.04 / 198.04 / 647.92 (span 1.0119 decades, `validate_span` ok)
  collapses the blunt held-25 true-field ensemble MAPE **68.52 % → 0.496 %**
  across 10 seeds — every seed ≤ 1.14 %, per-seed lambda in
  [0.9932, 1.0087].
- **The span rule transfers only PARTIALLY on the cond path.** A
  0.0407-decade "narrow" anchor triple still collapses the cliff (held-25
  **3.507 %**): on this path the zero-support failure is a one-sided
  channel extrapolation, so ANY 3 rows at the axis value pull the channel
  inside support and kill the ×1/3 underprediction — anchor span then only
  grades the residual (~7×: spread 0.50 % vs narrow 3.51 %). k = 2
  collapses but is unstable: **1.181 %** ensemble with a 9.31 % seed
  outlier, and the canon gate breaks (1.631). Use k = 3.
- **Regression gates for the spread arm** (v6qx 382 reference): ext
  **0.970** PASS, canon **1.044** PASS, test-55 **0.504 %** PASS, B-grid
  **0.857** — the B-grid miss is measured against a band whose own
  reference (0.889) already sits below the 0.90 floor (see the B-grid
  annotation under the §1 quality table). Slender-1.30 held-out serving
  drifts +2–4 pp on this pool (18.68 % → 20.46–22.42 %), but
  slender-class queries route to the SDF path anyway (Rule 1).
- **Serving recommendation unchanged**: the cond main pool STAYS v6qx 382;
  blunt routes to the SDF 381-row pool (**0.1–0.2 %**, Rule-3 footnote
  update). The 385-row c385_spread variant is the VERIFIED BACKUP for
  single-path blunt support on cond, not the serving choice — it degrades
  ext 0.992 → 0.970, canon 0.905 → 1.044, B 0.889 → 0.857.
- **The B/M corpus needs no purchased intermediates**: 3 anchors already
  collapse blunt on the cond path (68.52 % → 0.496 %), and the SDF route
  answers blunt at 0.1–0.2 %.

### Rule 2 — out-of-support queries are defended by the guard, not the band

The member std is **blind to moderate out-of-label extrapolation**: at
sail_x 0.65 and 1.5 (beyond all corpus labels, which span 0.7–1.4) the
std stays at **0.9–1.8 %** relative — the same order as in-distribution
(0.49 %); the quadratic sail channel only forces member divergence at
sail_x 2.0 (std 18–63 %). The envelope guard **rejects all of these
queries**. Serving rule: extrapolation protection comes from the guard,
never from a widened uncertainty band. Constructibility boundary (from
`SuboffConfig`, wave-13 W2): `sail_x_mult` is valid only on
**≈ [0.624, 2.761]** (sail footprint window [1.667, 10.646] ft) —
**sail_x_mult = 0.6 is not a constructible hull** (CAD raises); the
nearest valid below-label probe is 0.65.

### Rule 3 — corpus hygiene: fam-axis fragments stay out of the serving corpus

Do **not** pool fam-axis fragment rows (blunt / long_nose 28-row caches)
into the serving corpus. Verified twice: v5 (wave-10) pooled fragment
corpus ext ratio **0.886** (`docs/v5_fam_20260828.md`); v6 (wave-13) the
438-row pool regressed to ext **1.363**, canon **1.570** vs the 382-row
pool's 0.992 / 0.905 (`/nfs/wangxi/runs/famall_v6_20260828/`). The
mechanism is zero-sail_x-contrast rows diluting the trend calibration —
explicit channels do not immunize against it. Blunt-class designs
(l_over_d 0.75) via the cond path are a **68.5 %** LOFO catastrophe;
route them to the SDF path (honest clean-stream base **7.92 % +/- 4.62**,
see the footnote) — in-support intermediates now exist in the 381-row SDF
production pool, which serves blunt 0.75 at **0.1–0.2 %** (footnote update
below).

Footnote (2026-08-29 path-guard correction, wave-15 W3,
`/nfs/wangxi/runs/sdf_axis_20260829/`): the two-stage trainer's
first-call path guard shifts model init for exactly one cell per
process, so the wave-10 campaign number 8.87 % (and the wave-11
5.02 +/- 0.76) rode a lucky shifted seed; the honest clean-stream
blunt base is 7.92 % +/- 4.62 with seed spread 5-17 %.

Update (2026-08-30 serving sanity, `/nfs/wangxi/runs/sdf_serve_sanity_20260830/`):
the routing decision is unchanged — blunt never serves on the cond path
(68.5 % LOFO) — but the SDF side is now measured, not estimated. Blunt
l_over_d 0.75 queried through the actual production SDF pool (381-row
corpus, 10-seed ensemble) is **in-support at 0.1–0.2 %** ensemble-median
APE (per-query ensemble-median ts2 0.014–0.133 %, mean 0.103 / max
0.133; ts4 0.012–0.237 %, mean 0.120 / max 0.237; worst single member
1.30 % ts2 / 0.830 % ts4), and
the whole serving path is bit-exact against the recorded predictions (max
rel dev 0.0 on all 20 arm-seed cells; isolated single-query fp noise
4.35e-5). Ops on one 5090: cold build 1.42 s, single query 18.5–18.6 ms
(median, n=30; ts2 18.57 / ts4 18.50), 381-row × 10-member batch 0.557 s, GPU peak 3.7 GB allocated / 4.7 GB
reserved; recommendation pin ts2. The **~8 %** above was the pre-pool
wave-10/11 extrapolation estimate (honest clean-stream base
**7.92 % +/- 4.62** — the lucky-seed correction earlier in this footnote),
a leave-blunt-out number from the 378-hole-pool lineage; with the 381-row
pool the blunt rows are in support, though any single-seed blunt number
from that older lineage still deserves suspicion.

## 4. What is NOT recommended

- **20-seed pool** — doubling s0–s9 to s0–s19 adds nothing material
  (wave-13 W2 sensitivity: test-55 m95 2.52 vs 2.54, pooled-main t95
  1.07 vs 1.14; the 1.5 band covers the 20-member pool with more
  margin). No extra seeds needed.
- **fam-axis fragment pooling** — banned from the serving corpus; see
  Rule 3 (ext 0.886 → 1.363 across the v5/v6 generations).
- **fresh-Re anchor rows** — banned from the serving corpus per wave-11
  W2 (`/nfs/wangxi/runs/freshre_corpus_20260828/`): held-out
  out-of-window median error went **13.9 % → 14.7 %** (no improvement,
  ~180× from the oracle reference) and the six anchors **flipped the
  v5 sail_x trend channel sign** (canon ratio 0.4345 → −2.825).
- **Temperature as a slender fix** — rejected; see Rule 1.

## 5. Provenance

| claim | evidence |
|---|---|
| v6qx pool + quality numbers | `/nfs/wangxi/runs/cond_v6_20260828/` (`report.md`, `eval_v6qx.json`, `corpus382.npz`, `ckpts/v6qx/`), `docs/cond_v6_20260828.md`, PR #262 |
| uq_temperature = 1.5 + coverage table | `/nfs/wangxi/runs/v6_uq_20260828/` (`report.md`, `uq_results.json`, `uq_analysis.json`), `docs/uq_temperature_serving_20260827.md` (knob semantics), PR #257 |
| corpus-side coverage re-review 2026-09-03, t = 1.5 kept | `/nfs/wangxi/runs/uq_quad3_evidence_20260901/` (`evidence.json`, `report.md`, `preds_406.npz`) |
| slender class (bias, guard, SDF route) | `/nfs/wangxi/runs/v6_uq_20260828/` (−18.7 %, ±14.9σ, 28/28), `/nfs/wangxi/runs/sdf_slender_20260828/` (10.71 %), `docs/sdf_two_stage_20260828.md`, PR #260 |
| extrapolation blindness + sail_x_mult interval | `/nfs/wangxi/runs/v6_uq_20260828/` §5 |
| fam-fragment corpus ban | `/nfs/wangxi/runs/famall_v6_20260828/` (ext 1.363, canon 1.570, blunt cond 68.5 %), `docs/v5_fam_20260828.md` (0.886) |
| fresh-Re anchor ban | `/nfs/wangxi/runs/freshre_corpus_20260828/` (`report.md`) |
| k = 3 anchor recipe (MAPE(k) table, span rule, two-cliff lambda) | `/nfs/wangxi/runs/anchor_min_20260829/` (`summary.json` 24 cells, `runs/<cell>/preds.npz` per-row), wave-15 W2 |
| 10-seed production confirm of the recipe (353-row corpus) | `/nfs/wangxi/runs/anchor_promo_20260829/` (`summary_promo.json`, `preds_promo.npz`), wave-16 W16-B |
| path-guard RNG correction (blunt honest base) | `/nfs/wangxi/runs/sdf_axis_20260829/` (`report.md`, `crossbatch_replicate/`), wave-15 W3 |
| blunt SDF serving measured (0.1–0.2 %, bit-exact path, ops) | `/nfs/wangxi/runs/sdf_serve_sanity_20260830/` (`report.md`, `identity_check.json`, `blunt_queries.json`, `ops_numbers.json`), maintenance wave A-2 |
| A-1 blunt × k = 3 cond-path anchors (full/partial transfer, gates) | `/nfs/wangxi/runs/blunt_anchor_20260830/` (`report.md`, `summary.json`, `preds.npz`), maintenance wave A-1 |

The temperature recommendation must be re-derived if the corpus changes
(fresh-M / B-grid middles are extrapolation for the current 382 rows;
their bias may shrink when those labels join the corpus).
