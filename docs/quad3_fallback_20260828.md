# quad3 measured-curve serving fallback (`re_policy`)

- Date: 2026-08-28 (W3 wave 13)
- Branch `exp/quad3-fallback`, base `main` = `a3addebd` (#258 merge; #259/#260
  touch only `active_learning.py` / `sdf_two_stage.py` + their own tests/docs —
  no overlap with this PR)
- Module: `src/tensorlbm/ai/inference_service.py` (pure addition, opt-in,
  default OFF — the default call path is byte-identical, `info` included)
- Run directory: `/nfs/wangxi/runs/quad3_serve_20260828/`
- Tests: `tests/test_inference_service.py::TestQuad3Fallback`

## TL;DR

The drag surrogate serves ~0.5 % inside its Re window but ~14 % median error
outside it on unseen geometries. For a query whose design EXACTLY matches a
cached `(hull, sail, fin, u_in)` key with >= 3 cached Re rows, the measured
curve exists at serving time and is the ground truth — so serve it instead of
the network when the query Re is outside the corpus Re window. The
fresh-Re campaign (2026-08-27) adjudicated which cache-only extrapolator is
safe: **quad3** (exact quadratic through the 3 nearest cached levels in
log10-Re / log10-C_D space) is the only usable one, `global_lin` is banned.

## 1. Campaign adjudication (2026-08-27, real scans)

Artifacts: `/nfs/wangxi/runs/fresh_re_campaign_20260827/extrap_eval.json`
(+ `analyze_extrap.py`). 16 real-scan points across 4 B4-fam families,
distances up to 0.3 decades outside the family's cached Re edge; each
extrapolator uses ONLY the family's 28 cached rows.

| extrapolator | definition | at <= 0.1 dec (med / max) | full tested range | verdict |
|---|---|---|---|---|
| `global_lin` | OLS line over all 28 cached levels | 4.73 % / 5.61 % | 4.1 %–11.9 % | **banned** (systematic 4 %+ immediately outside the window) |
| `edge2` | exact line through the 2 edge levels | 0.099 % / 0.326 % | up to 2.4 % | second best, still curvature-limited |
| **`quad3`** | exact quadratic through the 3 nearest cached levels | **0.0038 % / 0.0138 %** | **<= 0.21 %** | **the only usable one** |
| `interp` (in-window recheck) | log-log interpolation | med 0.022 % | — | campaign-internal reference |

## 2. The exact quad3 definition (implemented verbatim)

From `analyze_extrap.py::predict` — the service reproduces this expression
bit-for-bit on the same rows (pinned by the real-data test to 1e-9):

```python
lr = np.log10(cached_re)   # ascending
lc = np.log10(cached_cd)
# "above" -> top-3 cached levels, "below" -> bottom-3; for an out-of-window
# query these ARE the 3 nearest in |log10 re| (what quad3_nearest3 selects)
c2, c1, c0 = np.polyfit(lr[slice], lc[slice], 2)
x = np.log10(re_query)
cd = float(10.0 ** (c2 * x * x + c1 * x + c0))
```

`quad3_nearest3(cached_re, cached_cd, re_query)` sorts the cache by Re,
selects the 3 rows nearest the query in `|log10 re|` (stable tie-break to
the lower Re), evaluates the selected triple in ascending order — the
campaign ordering — and returns `(cd, chosen_re)`; it returns `None` when
the nearest triple does not contain 3 DISTINCT `log10 Re` levels
(duplicate cached Re -> singular fit; the point keeps the network value,
nothing is silently interpolated).

## 3. API

```python
svc = DragSurrogateService(
    backend, guard,
    corpus_cache=fields,          # (N, 5, ny, nx)
    cache_re=re,                  # (N,)      row-aligned
    cache_designs=designs,        # (hull, sail, fin, u_in) per row
    cache_cd=cd,                  # (N,) measured C_D labels — NEW, optional
)
res = svc.predict(hull, sail, fin, re_grid, re_policy="quad3_fallback")
```

- `re_policy` (keyword-only, default `"network"`): `"network"` = the
  pre-flag behaviour byte-identical (no new `info` keys); `"quad3_fallback"`
  = opt-in measured-curve routing below. Unknown names raise `ValueError`
  (fail loud — no silent aliases).
- `cache_cd` is a new optional constructor argument (also passed through
  `DragSurrogateService.from_checkpoints`), row-aligned with `cache_re` /
`cache_designs`; `load_corpus_index` now also returns `CorpusIndex.cd` from
  the cache `cd` array, so the HTTP layer can wire it with one keyword.
  Without `cache_cd` the policy declines with
  `reason="no_measured_curve_cache"` and the network serves.

### Routing (per query point, first failure wins)

1. no measured curve attached -> `no_measured_curve_cache`;
2. every query Re inside the corpus window -> `re_inside_corpus_window`;
3. no cached row matches the design key -> `no_exact_design_match`;
4. fewer than 3 cached Re rows -> `insufficient_cached_rows`;
5. degenerate nearest-3 (duplicate cached log10-Re) -> that point keeps
   the network value (`declined_points`).

Applied points get `cd = quad3`, the network mean elsewhere. Mixed grids
are served per point; `info["re_policy"]["quad3_mask"]` records which.

### Result meta (`info["re_policy"]`)

```python
{
    "name": "quad3_fallback",
    "method": "quad3_fallback",        # or "network" when nothing applied
    "reason": None,                    # decline reason when method == network
    "corpus_re_window": [50.0, 800.0], # [min(cache_re), max(cache_re)]
    "n_cached_rows": 28,
    "cached_re": [...],                # the design's cached Re, ascending
    "nearest_cached_re": [[...3 Re...] | None per grid point],  # the chosen triple
    "quad3_mask": [True, ...],
    "n_quad3_points": 1,
    "loo_rel_rms": 0.0031,             # or None (see section 5)
    "std_source": "quad3_loo_relative_rms_times_cd",
    # when loo_rel_rms is None, std_source is instead one of
    #   "network_ensemble_std_loo_needs_4_cached_rows"
    #   "network_ensemble_std_loo_degenerate"  (then also
    #    quad3_loo_degenerate = True and quad3_loo_duplicate_re_rows =
    #    cached rows minus distinct log10-Re levels)
}
```

## 4. The Re window rule (exact)

The window is the **corpus** window of the service-attached cache:
`[min(cache_re), max(cache_re)]` over ALL rows — not the per-design cached
sweep, not the guard envelope. A point is **outside** iff
`re < min(cache_re)` or `re > max(cache_re)`; boundary values count as
inside (network path). Rationale: the surrogate's trust region is the
corpus support; a design's own cached sweep always lies inside it, so any
out-of-window query is also beyond the design's measured curve — exactly
the regime the campaign adjudicated. In-window exact-design queries keep
the network path (in-window the two agree to ~0.02 %; the win is outside).

## 5. std semantics (no fabricated uncertainty)

The reported ensemble `std` is absolute C_D everywhere, so the quad3 path
reports `std = cd_quad3 * quad3_loo_std(...)` where `quad3_loo_std` is the
RMS **relative** leave-one-out residual of the SAME estimator on the
design's cached curve: each cached row `j` is a pseudo-query served by
quad3 built on the remaining rows (nearest 3 to `re_j`), residual
`|cd_j - pred_j| / pred_j`, aggregated as `sqrt(mean(r_j^2))`.

- fewer than 4 cached rows, or duplicate cached log10-Re degenerating a
  leave-one-out pseudo-fit (the measured pool381 curve hits the latter:
  157 rows, 126 distinct Re) -> the point KEEPS the network ensemble std
  already computed for it — never `nan`, never a fabricated zero — with
  `std_source = "network_ensemble_std_loo_needs_4_cached_rows"` or
  `"network_ensemble_std_loo_degenerate"` plus, in the degenerate case,
  `quad3_loo_degenerate = True` and `quad3_loo_duplicate_re_rows` (cached
  rows minus distinct log10-Re levels). The kept std is the network-path
  std including its `uq_temperature` scaling — byte-identical to what the
  default (non-quad3) path serves at that point. The quad3 VALUE is
  untouched by this branch: prediction values and nearest-3 selection are
  byte-unchanged.
- history: before 2026-09-04 (#277) these points served `std = nan` with
  `std_source = "unavailable_fewer_than_4_cached_rows"` — and the
  degenerate case (duplicate Re) was mislabeled as the row-count case.
  The hardening replaced `nan` with the network std fallback and split the
  labels honestly; `tests/test_quad3_loo_hardening.py` pins the semantics
  (duplicate-Re curve -> network std bit-identical + flags; 3-row curve ->
  NaN-free with the row-count label; clean cache -> cd x LOO rms, no
  flags; default-off byte-identity).
- `lo`/`hi` (member min-max band) are `nan` at quad3 points — no ensemble
  exists for a measured curve; keeping the network band would attribute
  another estimator's spread to the value.
- `uq_temperature` does NOT rescale the quad3 std: that knob is calibrated
  on the deep-ensemble sigma of the network path (a different estimator).
  At the default `T = 1.0` nothing scales anywhere.

### Guard interaction (conservative)

The guard verdict never consumes `std` (`Guardrail.check` sees condition
rows only), so the std fallback branch cannot corrupt verdicts. An out-of-window Re still
flags `reject` on the `log10_re` envelope — the quad3 path does NOT soften
or suppress the verdict. The served pair is then (number, flag=reject,
method=quad3_fallback): honest "measured-curve extrapolation, outside trust
window", mirroring the service-wide semantics that a reject never suppresses
numbers and callers decide presentation.

## 6. When NOT to use

- **Unseen geometry** — no exact `(hull, sail, fin, u_in)` cache match:
  there is no measured curve; the network + guards own that regime (this is
  the ~14 % median out-of-window error case; the fallback declines with
  `no_exact_design_match`). The fallback is not a fix for generalisation.
- **In-window queries** — the surrogate is calibrated there (~0.5 %);
  the campaign's in-window interpolation recheck (med 0.022 %) confirms
  the two agree, so the network path stays.
- **Far extrapolation beyond the campaign's evidence** — quad3 was
  adjudicated to <= 0.3 decades outside the cached edge (<= 0.21 %); beyond
  that no cache-only predictor is validated. The guard's `reject` flag is
  the honest presentation.

## 7. Reproduction

```bash
# gates (CPU venv)
cd <repo checkout>
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= TMPDIR=/nfs/wangxi/tmp \
  /nfs/wangxi/venvs/ci-cpu/bin/pytest tests/test_inference_service.py \
  tests/test_quad3_loo_hardening.py -q \
  --basetemp=/nfs/wangxi/tmp/pt_quad3
# real-data quad3 reproduction (max |impl - stored campaign pred| over the
# 12 out-of-window campaign points; bar 1e-9)
/nfs/wangxi/venvs/ci-cpu/bin/python /nfs/wangxi/runs/quad3_serve_20260828/verify_realdata.py
```

The real-data test embeds the cached curves of three campaign families
(`fam_long_nose`, `fam_blunt`, `fam_slender`) and their stored quad3 preds
from `extrap_eval.json` (generated fixture, `fixture_block.py` in the run
directory) — the committed test is self-contained and does not read `/nfs`.

## 8. Provenance

- 2026-08-28: original campaign adjudication (2026-08-27 scans),
  implementation and this document; re_policy quad3_fallback ships
  default-OFF.
- 2026-09-04 (#277): LOO-degeneracy hardening. Duplicate-Re caches
  (pool381: 157 rows / 126 distinct) previously served `std = nan`
  mislabeled `unavailable_fewer_than_4_cached_rows`; the point now keeps
  the network ensemble std with honest `std_source` labels and a
  duplicate-row count. The fewer-than-4-rows case was fixed as the same
  defect class. Prediction values byte-unchanged; semantics pinned by
  `tests/test_quad3_loo_hardening.py`.
