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
10.70 %). Operational consequence: for a new axis value, scan a small
anchor set (~6 rows) into the SDF training corpus instead of shipping any
calibration layer.

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
route them to the SDF path (**8.87 %**) until in-support intermediates
exist in the corpus.

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
| slender class (bias, guard, SDF route) | `/nfs/wangxi/runs/v6_uq_20260828/` (−18.7 %, ±14.9σ, 28/28), `/nfs/wangxi/runs/sdf_slender_20260828/` (10.71 %), `docs/sdf_two_stage_20260828.md`, PR #260 |
| extrapolation blindness + sail_x_mult interval | `/nfs/wangxi/runs/v6_uq_20260828/` §5 |
| fam-fragment corpus ban | `/nfs/wangxi/runs/famall_v6_20260828/` (ext 1.363, canon 1.570, blunt 68.5 % / SDF 8.87 %), `docs/v5_fam_20260828.md` (0.886) |
| fresh-Re anchor ban | `/nfs/wangxi/runs/freshre_corpus_20260828/` (`report.md`) |

The temperature recommendation must be re-derived if the corpus changes
(fresh-M / B-grid middles are extrapolation for the current 382 rows;
their bias may shrink when those labels join the corpus).
