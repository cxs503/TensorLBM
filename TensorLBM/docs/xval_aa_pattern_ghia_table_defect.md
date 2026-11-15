# XVAL: the cavity acceptance metric was scoring against a scrambled Ghia Table II

**Anchor**: `XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE` (lanl/MF-LBM
`aa-pattern-fused-stream-collide`) · **Mission**: m22379 (verifier seat) ·
**Date**: 2026-11-16
**Disposition of this audit**: *defect found in the harness, fixed, re-scored*.

---

## 1. What was wrong

`docs/xval_aa_pattern_fused_stream_collide_acceptance.md` (m21739) left the
anchor with a visible anomaly and an explicit caveat:

> `L2 u_y = 4.80 %` at L = 32 … 4.59 % at L = 64 … 4.50 % at L = 128 —
> "`L2_v` does not fall with refinement … the run is not steady".

A profile error that is **flat in the grid** is not a discretisation error. It
was a harness error, in two layers:

**F1 (BLOCKING, evidence) — the reference table was a scrambled copy.**
`benchmarks/xval/aa_fused_stream_collide_ordering_audit.py` carried a hand-typed
copy of Ghia et al. (1982) instead of importing the provenance-bearing table
that this repository already owns
(`src/tensorlbm/verification/lid_driven_cavity/reference_data.py::GHIA_RE100`,
`provenance="BOUND"`, OCR of the paper, sha256 recorded in
`benchmarks/anchors/ghia1982_provenance.json`). Table I (`u`) was copied
correctly; **Table II (`v`) was not** — the values were attached to the wrong
`x/L` stations, and the copy also had 16 instead of 17 rows (station `0.8047`
missing). The acceptance doc claimed the two tables were "the same table"; they
were not.

**F2 (MAJOR, metric) — the centre line was sampled half a cell off.**
`_sample_profiles` took row/column `L // 2` of the *padded* array, i.e.
`y/H = 1/2 ± 1/(2L)`, not `y/H = 1/2`. On the horizontal centre line the normal
gradient of `v` is large near the top wall (`|∂v/∂y| ≈ 2–3` there), so the
offset alone contributed `≈ 8e-3 · U` at `L = 128` — an `O(1/L)` term that the
grid sweep reads as slow convergence.

Why the anomaly mattered: the flat ~4.5 % was attributed to the solver, which
kept the anchor's `< 15 %` acceptance "partial" for two missions and made the
`< 3 %` gate look unreachable for a D2Q9/BGK kernel on these grids.

## 2. Evidence that the retired table was the wrong one (not taste)

Physical invariants of Table II at mid height: the lid drives the flow in `+x`,
so fluid **rises on the left half** (`v > 0`) and **descends on the right half**
(`v < 0`), and `v = 0` at both side walls.

| station `x/H` | BOUND (used now) | retired copy | note |
|---|---|---|---|
| 0.8047 | **−0.24533** | (absent) | minimum of `v` |
| 0.8516 | −0.22445 (interp.) | −0.18866 | copy has no 0.8047 row |
| 0.9531 | −0.08864 | −0.03717 | 2.4× too small |
| 0.9609 | −0.07391 | **+0.00730** | sign flip |
| 0.9688 | −0.05906 | **+0.02526** | sign flip |
| 0.9766 | −0.05906 (interp.) | **+0.05248** | positive one cell from a wall |
| 1.0000 | 0.0 | 0.0 | wall |

The retired copy has `v` **positive** immediately next to the right wall and
non-monotone in the last three stations. Both are physically impossible for a
converged solution, and the fixed kernel reproduces the BOUND table instead
(§3). Independent check in the other direction: feeding an *exact* Ghia `v`
profile through the old metric scores **~4.4 %**, i.e. a perfect solver could
not pass `< 3 %` on that axis — pinned by
`tests/test_ghia_reference_single_source.py::test_exact_v_profile_fails_against_the_retired_table`.

## 3. Corrected numbers (program-generated, `artifacts/xval_aa_pattern/cavity_accept_corrected.json`)

Same kernels, same runs, re-scored: reference from the BOUND table, centre lines
interpolated onto `x/H = y/H = 1/2` exactly
(`benchmarks/xval/aa_fused_cavity_metric_recompute.py`). Percent of lid speed
`U = 0.1`; `L2` is `sqrt(mean((u/U − u_ghia)²))`.

| grid | source | `L2 u_x` | `L2 u_y` | `L2 u_x` interior | `L2 u_y` interior | `L2 u_y` vs retired table | floor `u` | floor `v` |
|---|---|---|---|---|---|---|---|---|
| L = 32 (step 9000, not steady) | field | 2.61 % | **1.05 %** | 0.60 % | 0.44 % | 4.44 % | 2.61 % | 0.95 % |
| L = 32 | stored profiles (old sampling) | 2.60 % | 1.25 % | 0.82 % | 0.73 % | 4.80 % | 2.61 % | 0.95 % |
| L = 64 (converged, 26 000) | stored profiles (old sampling) | 1.33 % | **0.77 %** | 0.48 % | 0.61 % | 4.59 % | 1.31 % | 0.47 % |
| L = 128 (converged, 49 000) | stored profiles (old sampling) | 0.70 % | 0.56 % | 0.34 % | 0.53 % | 4.50 % | 0.67 % | 0.25 % |
| L = 128 (converged) | **field, exact centre line** | 0.67 % | **0.38 %** | 0.24 % | 0.30 % | 4.40 % | 0.67 % | 0.25 % |

Reading the table:

* the `v` error **now converges**: 1.05 % → 0.77 % → 0.38 % (L = 32/64/128), and
  the retired-table column stays flat at 4.4–4.8 % on the *same data*. The
  plateau was the reference, not the kernel;
* `L2 u_x` sits exactly **on its metric floor** at every grid (2.61/1.31/0.67 % ≙
  floor 2.61/1.31/0.67 %), i.e. the vertical centre line is exact to the
  resolution of the comparison; the interior-`u` error (0.60 → 0.48 → 0.24 %) is
  the honest convergence signal there;
* both centre lines are **below 3 % at every grid**, and the `< 15 %` acceptance
  of the anchor is met with a factor 4–40 margin. The previous "`L2 u_y` 4.8 %,
  `< 3 %` not met" statement in
  `docs/xval_aa_pattern_fused_stream_collide_acceptance.md` §3.3/§4 is
  **withdrawn** — it was a harness artefact (F1), not a kernel result.

## 4. What changed in the tree

| path | change |
|---|---|
| `benchmarks/xval/aa_fused_stream_collide_ordering_audit.py` | `GHIA_U`/`GHIA_V` now built from `reference_data.GHIA_RE100` (single source of truth); scrambled copy kept as `GHIA_V_RETIRED_SCRAMBLED` for the regression test; `_sample_profiles` interpolates onto the exact centre lines; new `_sample_profiles_from_moments` |
| `benchmarks/xval/aa_fused_cavity_metric_recompute.py` | **new**: re-scores stored fields/profiles with both tables and both samplings, writes `cavity_accept_corrected.json` |
| `tests/test_ghia_reference_single_source.py` | **new**, 19 tests: table identity, physical invariants (both directions), exact-profile scoring, centre-line weights/sampling |
| `tests/gen_aa_fused_corrected_figures.py` | **new**: `fig7_ghia_table_defect.png`, `fig8_cavity_convergence_corrected.png` |
| `tests/gen_aa_fused_cavity_figures.py`, `tests/gen_aa_fused_ordering_figures.py` | dropped their own inline copies (they drew the scrambled Table II on published figures) |

Verification run locally:

```
pytest tests/test_ghia_reference_single_source.py -q          # 19 passed
pytest tests/test_aa_fused_cavity_acceptance.py \
       tests/test_aa_fused_stream_collide.py -q               # 21 passed (unchanged)
python benchmarks/xval/aa_fused_cavity_metric_recompute.py     # writes the corrected table
python tests/gen_aa_fused_corrected_figures.py                 # fig7, fig8
```

## 5. Notes, limits, honest gaps

1. **Provenance is inherited, not re-verified** — *corrected 2026-11-15: the
   witness IS present* at the repository root
   (`<repo>/benchmarks/verified/cavity/re5000/reference/ghia1982_tables_I_II.ocr.txt`,
   sha256 matches the pin), it had only been looked up relative to the
   `TensorLBM/` package directory.  The chain is now closed end-to-end in
   `docs/xval_aa_pattern_ghia_witness_rechain.md`
   (`benchmarks/xval/ghia_witness_rechain.py`): all 34 Re = 100 provenance cells
   are re-derived from the OCR stream, 0 sign disagreements against three
   independent transcriptions, and the acceptance numbers of §3 are reproduced
   from the witness-derived table alone (2.61 %/1.05 % at L = 32,
   0.67 %/0.38 % at L = 128).  Residual: the text layer cannot fix *signs*, so
   signs still rest on the independent copies. Two independent
   corroborations stand in: (i) the converged solver reproduces the BOUND table
   and violates the retired one; (ii) the retired table is unphysical next to
   the wall (§2). A future mission with the witness file should re-hash it.
2. **Not every grid was re-run with the corrected sampler.** Fresh runs with the
   fixed sampler were submitted to the scheduler as **cj4078 (L = 64)** and
   **cj4079 (L = 32)** and were still `queued` at write time (queue position
   ≈ 25). The corrected-sampler numbers above therefore come from the *stored*
   L = 128 field; L = 32/64 use the old (half-cell-offset) profiles with the
   corrected table, which is why their `v` errors are the conservative ones.
3. **L = 32 is not steady** (9 000 steps); it is quoted because it is the grid on
   which the old kernel "failed" at `L2_u = 0.247`.
4. Scope unchanged: D2Q9 / BGK, `tau` from `Re = 100, U = 0.1`, halfway
   bounce-back, 2 000-step lid ramp. Nothing here transfers to MRT/TRT or D3Q19.
5. The original doc's historical text is left intact (it documents how the error
   was introduced); the withdrawn claim is marked with a banner pointing here.

## 6. Artefacts

| path | content |
|---|---|
| `artifacts/xval_aa_pattern/cavity_accept_corrected.json` | both tables, per-grid scores (corrected / retired), floors, corrected profiles |
| `artifacts/xval_aa_pattern/fig7_ghia_table_defect.png` | `v` profile at L = 128 vs both tables + point-wise residual |
| `artifacts/xval_aa_pattern/fig8_cavity_convergence_corrected.png` | `L2(u_x)`, `L2(u_y)` vs `L` with floors and the 3 %/15 % gates |
| `artifacts/xval_aa_pattern/cavity_accept_field_L32.npz`, `…_L128.npz` | converged fields used by the recomputation |