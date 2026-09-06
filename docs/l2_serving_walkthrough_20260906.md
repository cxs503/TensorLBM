# L2 new-geometry serving walkthrough — STL in, drag curve out (2026-09-06)

- Status: **capstone demo of the merged L2 stack** — one runnable
  walkthrough that exercises every piece of the new-geometry serving line
  through the real service API.
- Provenance: PRs
  [#275](https://github.com/cxs503/TensorLBM/pull/275) (`field_provider`),
  [#276](https://github.com/cxs503/TensorLBM/pull/276) (`ckpt_bundle`
  serving pool), [#278](https://github.com/cxs503/TensorLBM/pull/278)
  (`inference_service` field-borrow hook),
  [#280](https://github.com/cxs503/TensorLBM/pull/280) (`geometry_stl`
  STL ingest/export). Validation backbone: the 2026-09-04 e2e LODO
  campaign (server path `/nfs/wangxi/runs/l2_e2e_validation_20260904/`,
  `e2e.json` + `report.md`).
- Machine evidence of THIS walkthrough (every number below comes from it):
  `/nfs/wangxi/runs/l2_walkthrough_20260906/` (`walkthrough.json`,
  `report.md` generated from the json, `run_walkthrough.py` driver,
  `design106.stl`, `sphere_oof.stl`).

## The story in one command

An external user brings a hull as an **STL mesh** and wants a drag curve
with honest uncertainty. The demo script
[`scripts/l2_serving_walkthrough.py`](../scripts/l2_serving_walkthrough.py)
does exactly that, end to end, through the public API:

```
python scripts/l2_serving_walkthrough.py            # 5090 defaults
python scripts/l2_serving_walkthrough.py --device cpu --out /tmp/walk/leg1.json
```

The path it walks (each step is the production module, not a re-assembly):

| step | module (PR) | what happens |
|---|---|---|
| 1. corpus | `ai.field_provider` (#275) | `FieldProvider.from_corpus(view_dir)` builds the 406-row retrieval pool from the four production npz files |
| 2. geometry out | `geometry_stl` (#280) | the design's occupancy mask is exported as a watertight **boxel STL** (`write_mask_stl`) |
| 3. geometry in | `geometry_stl` (#280) | `stl_to_sdf(stl, (64, 64, 128))` re-voxelises the STL and chains the corpus SDF path (exact EDT, clip, pool) — an STL-sourced hull meets the surrogate under the identical input contract |
| 4. pool | `ai.ckpt_bundle` (#276) | `load_bundle_pool(dir, arm="ts2")` -> `PerMemberEnsembleBackend.from_bundles` — the frozen pm20260831 10-seed ensemble |
| 5. retrieval | `ai.field_provider` (#275) | leave-one-design-out pool; the new geometry has no cached reference field |
| 6. serve | `ai.inference_service` (#278) | `DragSurrogateService.predict(..., sdf=stl_sdf, field_policy="field_borrow")` — field resolved by `sdf_near` borrowing, provenance + guard flags in the response |

## Evidence (2026-09-06 run, 5090, ts2)

### Leg (i) — truthful accuracy on an ALL-HELD design

Design 106 — the slender `l_over_d = 1.30` hull, the one corpus design
carrying all 25 held-out rows — served as if new (all 28 of its rows leave
the retrieval pool; pool = 378 rows), truth = its 5 target-row corpus
labels.

- **Geometry chain is lossless**: mask -> STL -> mask **bit-exact**;
  re-ingested SDF **bit-exact** vs the stored corpus SDF (max rel-L2
  0.0). The STL detour validates the external interface, not an
  approximation of it.
- **MAPE 0.1535 %** (max APE 0.4017 %) over the 5-point Re sweep
  66–591; 2-sigma coverage 5/5.
- **Provenance (verbatim `info["field_borrow"]`)**: strategy `sdf_near`,
  pool 378, donor = corpus row 378 (design 105, the `l_over_d = 1.2`
  neighbour), retrieval distance 2.981, field guard **pass** at rel-L2
  0.0876 vs the 0.15 threshold; cond-space guard `ok` (score 1.636).
- **Cross-check vs the e2e LODO A arm**: same donor row, per-row means
  agree to max rel 0.0 — the service composition reproduces the validated
  campaign bit-for-bit on this design.

### Leg (ii) — out-of-family probe (NO truth; mechanism only)

A volume-matched **sphere** (2449 voxels vs the hull's 2437) at the same
grid, through the same service and pool, described by the nearest
in-vocabulary descriptor (`bare_hull`, sail=fin=1):

- **Retrieval distance x10.9** (32.58 vs the in-family 2.98) — the
  geometry-side signal; the donor is still a corpus row (378).
- **Field guard passes** (rel-L2 0.0876) — it measures the *borrowed
  field* against the pool mean, and a corpus field is in-manifold by
  construction. Honest reading: the field guard is not a shape-family
  test; the retrieval distance is.
- **Cond-space guard `ok`** (score 1.683) — it sees only the CAD
  descriptor triple, so it cannot flag a foreign STL either.
- **The ensemble disagrees loudly**: std inflates x10–x27 vs the
  in-family case; the member min–max band widens from ~0.55–1.25 % to
  ~24–34 % of C_D. Served C_D falls to 0.44–0.57x the slender curve —
  plausible for a bluff body, but there is **no truth** for a sphere in
  this corpus: the flags and the sigma ARE the deliverable of this leg.

### Leg (iii) — latency and usability

| path | single query | 5-Re sweep |
|---|---:|---:|
| `field_policy="field_borrow"` | 77.9 ms | 78.8 ms |
| caller `fields=` (no retrieval) | 20.6 ms | 20.7 ms |

Cold construction (bundle discovery + pool + provider + guard fit +
service) = **2.3 s** once. Decomposition of the warm query: model work
20.2 ms, service facade (condition rows + guard) 0.04 ms, retrieval +
field guard 57.7 ms. Reference: the cached-fields SDF serving figure of
**18.5–18.6 ms/query** ([serving_v6qx_20260828.md](serving_v6qx_20260828.md),
ts2 18.57 ms median n=30). The caller-fields path here is the
like-for-like comparison; the residual ~2 ms is machine/session-level
(cudnn determinism pinning measured irrelevant), and `field_borrow` adds
the `sdf_near` retrieval — a chunked float64 L2 over the 378x32x32x64
pool on CPU numpy — charged **per predict call**, so a whole Re sweep
amortises it (sweep ≈ single query).

## Standing caveats

1. **In-family accuracy is oracle-level**: the all-held design serves at
   0.15 % MAPE, matching the e2e LODO campaign. This is NOT a
   voxelizer-independence result — the demo STL was exported from the
   corpus CAD mask (the round-trip guarantee is what makes the equality
   to the e2e meaningful).
2. **Out-of-family has NO truth.** Leg (ii) demonstrates the
   guard/provenance/std story — retrieval distance, sigma inflation,
   honest flags — never accuracy.
3. **The corpus is a SUBOFF-style family.** Surrogate, cond vocabulary
   and guard envelope are fit on it; a genuinely foreign hull must be
   read through the provenance block first.
4. **The cond-space guard is descriptor-blind for STL shapes**: it sees
   `(hull, sail, fin)`, not the query SDF. For STL-sourced geometry the
   retrieval distance and the ensemble std are the out-of-family signals
   (leg (ii) shows both firing).
5. **Borrowed fields are never silent**: every response carries
   `info["field_borrow"]` with strategy, donor, distance and guard
   numbers (the #275/#278 honest-serving contract); the default
   `field_policy="cache"` path stays byte-identical to the pre-flag
   service.
6. `uq_temperature = 1.5` is the serving pin carried over from
   [serving_v6qx_20260828.md](serving_v6qx_20260828.md); it scales the
   reported sigma only, never the guard or the member band.

## Provenance

| claim | evidence |
|---|---|
| this walkthrough (all three legs, all numbers above) | `/nfs/wangxi/runs/l2_walkthrough_20260906/` (`walkthrough.json`, `report.md`, `run_walkthrough.py`, `make_report.py`, `driver.log`, `design106.stl`, `sphere_oof.stl`) |
| LODO composition this demo reuses (pool rule, target rule, donor, oracle-level accuracy) | `/nfs/wangxi/runs/l2_e2e_validation_20260904/` (`e2e.json`, `run_e2e.py`), `docs/field_borrow_20260904.md` |
| frozen serving ensembles (10 seeds x ts2/ts4) | `/nfs/wangxi/runs/ckpt_bundle_pm20260831/` (`README.md`) |
| cached-fields latency reference (18.5–18.6 ms/query) | `/nfs/wangxi/runs/sdf_serve_sanity_20260830/` (`ops_numbers.json`), [serving_v6qx_20260828.md](serving_v6qx_20260828.md) |
