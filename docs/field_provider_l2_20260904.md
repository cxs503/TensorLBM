# Field provider v1 — retrieval-based reference fields for new-geometry serving (2026-09-04)

- Status: **L2 enabler, shipped as a library module** —
  `src/tensorlbm/ai/field_provider.py` (+ `tests/test_field_provider.py`).
  Pure numpy, CPU/CI-safe, additive (no existing file touched).
- Provenance: implements the provider-v1 policy recommended by the
  2026-09-04 field-sensitivity study
  (`/nfs/wangxi/runs/l2_field_sensitivity_20260904/` — machine truth
  `study.json`, verdict §7 of `report.md`; read-only).
- One-line summary: **a NEW geometry can be served by the two-stage drag
  surrogate without an LBM run** — retrieve (or average) an in-manifold
  reference field from the corpus, feed the target's OWN SDF, and let the
  guard certify in-manifold-ness.

## TL;DR

| input for a new geometry | source | measured cost (held MAPE) |
|---|---|---|
| SDF | **target's own** (voxelisation + SDF path, existing L2 deliverable) | — (this is the contract) |
| reference field | `FieldProvider.borrow(...)` — nearest donor or pool mean | x0.998 of baseline (invisible) |
| donor SDF | **never** — the API does not return one | **x57.9 (ts2) / x59.3 (ts4)** if you make this mistake |
| in-manifold check | `guard_ok` (rel-L2 to pool mean <= 0.15) | zeros probe fails at rel-L2 1.0 |

## 1. What problem this closes

The two-stage SDF drag surrogate consumes two geometry-side inputs: the
voxel SDF (through the frozen encoder) and a 5-channel reference
mid-plane field (through the body's channel normalisation). For a corpus
design both come from the LBM run that produced the row; for a NEW
geometry the field was therefore assumed to need a fresh LBM run. The
2026-09-04 study measured which of the two actually carries signal, and
the answer collapses the problem: the field only has to be *inside the
corpus manifold* — any in-corpus donor, a random donor, or the pool mean
all score identically — while the SDF is the input that must be the
target's own.

`FieldProvider` turns that verdict into the smallest API that cannot
misfire: retrieval by SDF- or cond-nearest neighbour, a mean fallback,
an in-manifold guard, and no way to hand back a donor SDF.

## 2. Evidence (copied from `study.json`)

Held-25 ensemble MAPE on the 406-row production corpus (381-row
in-support donor pool, 25 held-out rows; 10-seed v6qx serving pool,
counts=1, identical 75-row batch composition):

| field source for the held rows | ts2 arm MAPE (%) | ts4 arm MAPE (%) |
|---|---|---|
| own fields (baseline) | 0.21709127033512002 | 0.20952060997140265 |
| cond-nearest donor | 0.2166642340263256 (x0.9980329180987554) | 0.2094077066368616 (x0.9994611349472662) |
| sdf-nearest donor | 0.21401902935853648 (x0.9859, derived) | 0.2079510180243176 (x0.9925, derived) |
| random donor | 0.21786157482596688 (x1.0035, derived) | 0.2102765138120155 (x1.0036, derived) |
| corpus-mean fields | 0.2166891556513928 (x0.9981, derived) | 0.209360631849985 (x0.9992, derived) |
| all-zeros probe | 1.662701656894888 (x7.658998237599349) | 3.749149932100474 (x17.89394338157088) |

Field swaps of ANY in-corpus strategy move held MAPE by <= 0.2 % relative
— below the batch-composition fp control — and no donor-distance radius
limit is needed: the mismatch cost stays at baseline noise all the way
out to the full corpus field spread (field L2 up to
21.41758951575883 on ts2, ~1.9 median inter-corpus distances; p90 APE
0.418 % vs baseline p90 0.418 %).

The complementary arm — swap the SDF instead, keep the fields — is where
the signal lives:

| sdf source for the held rows | ts2 arm MAPE (%) | ts4 arm MAPE (%) |
|---|---|---|
| own SDF (baseline) | 0.21709127033512002 | 0.20952060997140265 |
| cond-nearest donor's SDF | 12.560543921936416 (**x57.9**) | 12.417900945730782 (**x59.3**) |

Conclusion encoded in this module: the frozen SDF encoder carries ALL
geometry signal; the reference field carries none in-corpus; the only
failure mode is out-of-manifold fields.

## 3. The contract

### 3.1 The target owns the SDF

`BorrowedField` has no `sdf` attribute at all — returning a donor SDF is
unrepresentable in the API. The caller must build the NEW geometry's SDF
through the existing voxelisation + SDF path and pass it verbatim to the
surrogate. The measured cost of getting this wrong is the x57.9/x59.3
row above. `borrow()`'s docstring and the returned `provenance` dict
both restate this contract.

### 3.2 Guard semantics

After retrieval the BORROWED field is checked for in-manifold-ness:

```
guard_rel_l2 = ||fields - mean(pool_fields)||_2 / ||mean(pool_fields)||_2   (float64)
guard_ok     = guard_rel_l2 <= guard_threshold                              (default 0.15)
```

The 0.15 default is the study's recommended operating point, calibrated
against the real corpus (re-measured 2026-09-04 on the production 406-row
pool with this module): corpus rows sit at rel-L2 0.024–0.159 with median
0.054, so a strict 0.15 flags a five-row `dsi=3/4/5` tail (max 0.15889;
the nearest threshold covering EVERY corpus row is 0.16), while the
all-zeros probe — the only observed failure mode, x7.7 (ts2) / x17.9
(ts4) — sits at rel-L2 exactly 1.000000, a >6x margin from the corpus
tail. It is a constructor argument (`FieldProvider(..., guard_threshold=)`)
so operators can tighten or loosen it without code changes. For the
`"mean"` strategy the guard is trivially satisfied (the borrowed field IS
the anchor; `guard_rel_l2 = 0.0`). A failing guard still returns the
result — refusing or escalating is the caller's policy:

```python
borrowed = provider.borrow(target_sdf=target_sdf)
if not borrowed.guard_ok:
    ...  # refuse the query, or log and fall back to borrowed = provider.borrow(strategy="mean")
```

### 3.3 Scope caveat

The invariance numbers were measured with IN-CORPUS donors (swaps among
406 production rows). "Any field works" is therefore a statement about
the corpus manifold, not about arbitrary arrays — which is exactly why
the guard exists. Distances are plain: full-array SDF-L2 for `sdf_near`,
plain Euclidean L2 over the supplied cond columns for `cond_near`
(z-score `pool_cond`/`target_cond` first if you want the study's z-space
metric; the corpus cond convention is
`[log10(re), log10(uin), log10(sail), log10(fin)]`, ts2 uses columns
0:2, ts4 uses 0:4).

## 4. API

### 4.1 Direct construction (arrays, no hardcoded paths)

```python
import numpy as np
from tensorlbm.ai.field_provider import FieldProvider

provider = FieldProvider(pool_fields, pool_sdfs=pool_sdfs, pool_cond=pool_cond)

borrowed = provider.borrow(target_sdf=target_sdf)          # -> "sdf_near" by default
borrowed = provider.borrow(target_cond=target_cond)        # -> "cond_near" when no SDF side
borrowed = provider.borrow(strategy="mean")                # donor_index=None, guard 0.0
borrowed.fields.shape                                      # (5, ny, nx) — a private copy
```

Strategy resolution defaults to `sdf_near` when the pool has SDFs and a
`target_sdf` is given, else `cond_near` when possible, else `mean`. An
EXPLICIT strategy whose inputs are missing raises `ValueError` with a
precise message instead of silently falling back. Ties break to the
lowest pool index; identical inputs give identical donors.

### 4.2 `from_corpus` — the production artifact (read-only)

`from_corpus(path)` accepts a directory holding the four production
files exactly as `ckpt_bundle_rehearsal_20260831/rehearsal.py`
`load_fam` assembles them (`cache_fam.npz` + `cache_ext56.npz` rows,
SDFs from `sdf_fam350.npz` + `sdf_ext2.npz` keys `d{dsi}`), or a single
`.npz` snapshot with key `x`/`fields` plus optional `sdf`/`cond`. The
production files live in four different run directories, so assemble a
symlink VIEW rather than copying:

```python
# view dir: cache_fam.npz -> /nfs/wangxi/runs/b4_fam_20260824/cache_fam.npz, etc.
provider = FieldProvider.from_corpus("/nfs/wangxi/tmp/corpus_view")
```

This materialises ~0.21 GB (105 MB fields + 107 MB SDFs, float32, 406
rows) and derives `cond` and per-row `donor_key` provenance itself. For
lazy access, `np.load(..., mmap_mode="r")` the members and use the plain
constructor. The study's donor pool was the 381-row in-support subset;
`from_corpus` returns all 406 rows and leaves subsetting to the caller.

### 4.3 End-to-end shape for a new geometry

```python
target_sdf = build_sdf(new_geometry)                        # existing voxelisation path
borrowed = provider.borrow(target_sdf=target_sdf)
assert borrowed.guard_ok
mean, std = backend.predict_batch(
    borrowed.fields[None], target_sdf[None], cond_rows, ...  # TARGET SDF + borrowed field
)
```

## 5. What v1 does NOT do

- **No learned field generator.** The study leaves no accuracy budget
  for one: the zero-information pool-mean field already scores x0.998 of
  baseline. A generator would only need to beat "in-manifold", which
  retrieval already guarantees.
- **No automatic SDF construction.** The caller supplies the target's
  own SDF from the existing voxelisation + SDF path (itself an L2
  deliverable). The provider will not build, borrow, or guess one.
- **No UQ recalibration.** Field swaps leave the ensemble std
  essentially unchanged (median ratio 1.000 on both arms), so the
  existing `uq_temperature=1.5` band is untouched.
- **No serving-path integration.** v1 is a library module; wiring it
  into `inference_service` field resolution is deliberately left for a
  follow-up (the additive constraint of this change set).
- **No out-of-manifold repair.** A failing guard is reported, not fixed.

## 6. Provenance

| item | value |
|---|---|
| study directory (read-only) | `/nfs/wangxi/runs/l2_field_sensitivity_20260904/` |
| machine truth | `study.json` (numbers in §2 copied from it verbatim) |
| verdict narrative | `report.md` §7 (provider-v1 policy, 2026-09-04) |
| corpus artifact convention | `/nfs/wangxi/runs/ckpt_bundle_rehearsal_20260831/rehearsal.py` (`load_fam`) |
| corpus rows | 406 production rows (350 fam + 56 ext); donor pool in the study = 381 in-support |
| module / tests | `src/tensorlbm/ai/field_provider.py`, `tests/test_field_provider.py` |
| key numbers | field invariance x0.998 (ts2 cond-near) · SDF sensitivity x57.9 (ts2) / x59.3 (ts4) · zeros probe x7.7 / x17.9 · guard anchor: zeros at rel-L2 1.0 to pool mean, corpus tail 0.159, threshold default 0.15 (0.16 covers every corpus row) |
