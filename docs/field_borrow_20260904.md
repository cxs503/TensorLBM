# Field borrowing for new-geometry serving (`field_policy="field_borrow"`)

Date: 2026-09-04 · Branch: `exp/field-borrow-serving` · Evidence:
`/nfs/wangxi/runs/l2_e2e_validation_20260904/` (machine truth `e2e.json`,
driver `run_e2e.py`) and `/nfs/wangxi/runs/l2_field_sensitivity_20260904/`
(`study.json`).

## What the flag does

A new-geometry query — one that has an SDF but no cached LBM reference
field — previously could not be served at all: field resolution in
`DragSurrogateService` raises `BackendQueryError` for a design absent
from the attached cache. The opt-in flag changes exactly that:

```python
svc = DragSurrogateService(
    backend,                       # any ModelEnsembleBackend / PerMemberEnsembleBackend
    guard,
    field_provider=FieldProvider.from_corpus(corpus_view_dir),  # the pool
)
res = svc.predict(hull, sail, fin, re_grid, sdf=target_sdf,
                  field_policy="field_borrow")
```

- Default `field_policy="cache"` (`FIELD_POLICY_CACHE`) is the pre-flag
  behaviour, **byte-identical** including `info` (pinned by test).
- With `"field_borrow"` (`FIELD_POLICY_BORROW`), when — and only when —
  normal field resolution (caller `fields=` / `field_point` / attached
  corpus cache) would raise, the service borrows an in-manifold reference
  field from the pool via
  `tensorlbm.ai.field_borrow.borrow_serving_field`, keyed on the **query's
  own SDF** (`sdf=`). Cached designs keep their own cached field; a
  missing `sdf`/pool re-raises the original error — nothing is silently
  substituted.
- Two-stage (per-member) backends always take `sdf=` as a model input and
  serve the corpus-convention param cond
  (`[log10 re, log10 uin]` for ts2, plus `[log10 sail, log10 fin]` for
  ts4) instead of the 8-channel `condition_v3`; the guard verdict stays on
  `condition_v3` either way.

The composition lives in `src/tensorlbm/ai/field_borrow.py`
(`borrow_serving_field`, `param_cond_rows`, `borrow_conditioning`); the
service hook is `DragSurrogateService._serve_model_members`, called from
`predict` — deliberately far from `_quad3_fallback`.

## Evidence backing it (2026-09-04 e2e LODO campaign)

Composition under test: regenerated target SDF (SuboffConfig →
`build_suboff_mask` at PRODUCTION_GRID → `sdf_volume`) +
`FieldProvider(sdf_near)` borrowed field from a pool that excludes every
row of the target design + target cond, served through the frozen
pm20260831 10-member ensembles (ts2 primary).

- **SDF regeneration: 121/122 designs bit-exact** vs the stored corpus
  SDFs (the one miss is a vintage-offset design whose stored mask predates
  the current CAD; worst rel-L2 1.8 %).
- **New-geometry serving at oracle level**: ts2 macro MAPE
  **0.2149 % borrowed vs 0.2174 % oracle** (x0.99) **excluding the
  vintage design**; including it the ratio is 2.38, driven entirely by the
  non-bit-exact SDF on that one design (arm C — own fields + regenerated
  SDF — shows the same 2.38, isolating SDF regen as the cost, not
  borrowing).
- **Pure field borrowing is free**: arm D (borrowed field + stored SDF)
  vs arm B (own fields + stored SDF) = **x0.991 macro** over all five LODO
  designs — consistent with the field-sensitivity study (field swaps of
  any in-manifold strategy x0.998–x1.004 of baseline; the only field
  failure mode is a grossly out-of-manifold field, all-zeros probe x7.7
  ts2).
- **Guard behaviour**: all five LODO donors passed at rel-L2 0.026–0.132
  vs the 0.15 threshold; ensemble std ratio A/B median 1.0004, 2-sigma
  coverage 20/22 rows.

## The honest-provenance contract

Whenever borrowing happens, the response says so — a borrowed field is
never presented as a cached one:

```python
res.info["field_source"] == "field_borrow"
res.info["field_borrow"] == {
    "strategy": "sdf_near", "donor_index": 214, "donor_key": "dsi=... re=...",
    "distance": 0.0, "guard_ok": True,
    "guard_rel_l2": 0.054, "guard_threshold": 0.15,
    "pool_size": 406, "e2e": "/nfs/wangxi/runs/l2_e2e_validation_20260904",
}
```

If the in-manifold guard **fails** (`guard_ok=False`, i.e.
`||borrowed - pool_mean|| / ||pool_mean|| > threshold`), the service
**still serves** — flags set in `info["field_borrow"]` and a warning log
emitted (`tensorlbm.ai.field_borrow`). It does not raise and does not
silently swap strategy: out-of-manifold fields are the known failure mode
(x7.7 ts2 held MAPE), so the flag is the safety story, mirroring the
always-compute-and-attach semantics of the guard verdict.

## Caveats

- **Path consistency, not generator independence.** The SDF-regeneration
  leg of the evidence reused the corpus's own CAD parameterization
  (`suboff_cad` + `geom_encoder.sdf_volume`). A geometry from a different
  voxelizer has no measured guarantee; the guard and provenance flags are
  the mitigation.
- Field-swap invariance was measured with in-corpus donors; a borrowed
  field that passes the guard but resembles no corpus row is outside the
  study's support (the guard exists for exactly this).
- This flag composes with `re_policy` orthogonally (field source vs Re
  routing); both default OFF.
