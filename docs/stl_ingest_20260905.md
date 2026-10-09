# STL -> voxel -> SDF geometry ingestion for the ship-drag surrogate (2026-09-05)

Module: `src/tensorlbm/geometry_stl.py` (pure numpy — no new dependencies).
Tests: `tests/test_geometry_stl.py` (22 cases, ~8 s CPU, includes one
full PRODUCTION_GRID case).  Evidence:
`/nfs/wangxi/runs/stl_ingest_20260905/` (`ingest.json`, `report.md`,
`boxel_d0.stl`, `driver.log`).

## What it does

The L2 line consumed geometry as a boolean solid mask built from SUBOFF
CAD parameters; an external user brings a hull as an STL mesh.  This
module closes that seam with an export leg and an ingest leg, reusing
the existing machinery instead of duplicating it:

| Function | Purpose |
| --- | --- |
| `mask_to_stl(mask, *, binary=True) -> bytes` | Boxel (voxel-solid) STL writer: one quad — **2 triangles**, consistently outward-oriented — per **exposed** voxel face; watertight by construction. |
| `write_mask_stl(path, mask, *, binary=True) -> Path` | File-writing variant. |
| `stl_to_mask(source, shape, *, origin, spacing, axis, robust, require_watertight=True) -> bool array` | Re-voxelise any closed STL (path / raw bytes / mesh / triangle table; binary and ASCII auto-detected) via the vectorised Moller-Trumbore ray-parity caster of `tensorlbm.voxelize.mask_from_stl`. |
| `stl_to_sdf(source, shape, *, ..., clip, pool, device) -> float32 (D', H', W')` | Convenience chain `stl_to_mask` -> `tensorlbm.ai.geom_encoder.sdf_volume` — the **same** exact-EDT / +-8-voxel-clip / stride-2-pool chain the CAD corpus path uses (imported, not reimplemented). |

`tensorlbm.voxelize` (loader, `is_watertight`, ray-parity voxeliser,
2026-08-24) and `tensorlbm.ai.geom_encoder` (SDF chain) already existed
on main; `geometry_stl` adds only the boxel exporter, the bytes-level STL
entry point and the chain.  The CAD-param -> tessellated-STL ->
tolerance-based round trip lives separately in `tensorlbm.cad_stl`
(IoU/boundary-band metrics) — that path handles smooth meshes where
bit-exactness is impossible; this path handles masks and STL files.

## Conventions

- Masks are `(nz, ny, nx)` bool; mesh x on the last array axis (the
  `(Q, nz, ny, nx)` lattice layout); mesh axes `(x, y, z)`.
- `mask_to_stl` writes voxel `(iz, iy, ix)` as the unit box
  `[ix, ix+1] x [iy, iy+1] x [iz, iz+1]` (integer node grid).  With the
  default `origin=(0, 0, 0)`, `spacing=1` the caster samples voxel
  centres at `(i+0.5, j+0.5, k+0.5)` — exactly the box centres — so
  `stl_to_mask(mask_to_stl(m), m.shape) == m` **bit-exactly** on the
  same grid.  An external mesh must be placed on the grid first
  (`voxelize.place_on_grid`, canonical SUBOFF frame).
- Domain border counts as fluid: masks solid to the grid edge still emit
  their outer boundary faces.

## Round-trip evidence (2026-09-05 campaign)

8 corpus designs spanning the family — the e2e K=5 LODO set (mother
`with_sail`, blunt `l/d=0.75`, slender `l/d=1.3`, appendage corner
`sail=fin=3.0`, vintage stored-mask stress case) plus every hull type
and the global `l/d` extremes — at the production grid
`(nz, ny, nx) = (64, 64, 128)`:

| design | hull | tris | watertight | mask bit-eq | mask XOR | SDF bit-eq | SDF rel-L2 |
|---|---|---|---|---|---|---|---|
| 0 | bare_hull | 4988 | True | True | 0 | True | 0.000e+00 |
| 1 | full | 5048 | True | True | 0 | True | 0.000e+00 |
| 10 | full | 5292 | True | True | 0 | True | 0.000e+00 |
| 94 | full | 8388 | True | True | 0 | True | 0.000e+00 |
| 95 | with_sail | 5016 | True | True | 0 | True | 0.000e+00 |
| 100 | with_sail | 6100 | True | True | 0 | True | 0.000e+00 |
| 101 | with_sail | 5116 | True | True | 0 | True | 0.000e+00 |
| 106 | with_sail | 4144 | True | True | 0 | True | 0.000e+00 |

Decisive number — frozen-ensemble serving prediction shift through the
full new-geometry composition (FieldProvider `sdf_near` borrow from the
design-held-out pool + 10-member ts2/ts4 ensembles, bundles
`ckpt_bundle_pm20260831`): **0.000e+00** for every design, every target
row, ensemble mean AND member level; donors identical across legs.  With
bit-equal masks the STL leg is the CAD leg — the shift is exactly zero
by construction, and the campaign measures exactly that.  Wall clock per
design: STL export ~0.01 s, re-voxelisation 0.7-1.4 s, SDF ~0.2-0.3 s
(GPU; CPU-only also fast at these sizes).

## Hard limits

- **Resolution = the voxel grid.**  Anything below one voxel is
  invisible: the ingest path rasterises the mesh onto `(64, 64, 128)`
  (or whatever grid the caller passes).  Thin fins are already ~1 voxel
  on the production lattice — an STL with finer detail than the grid
  cannot express it, exactly as for the CAD path.
- **Watertight closed manifold required.**  `stl_to_mask` runs
  `voxelize.is_watertight` by default and raises on open / torn meshes:
  ray parity leaks on a non-closed surface (whole voxel columns flip
  solid/fluid across the breach; demonstrated in the tests).
  `require_watertight=False` exists for diagnosis only.  Known
  degenerate case: a mask whose solids touch only along edges/corners
  (checkerboard) has a *pinched* boundary — closed but not edge-manifold
  — rejected by `is_watertight` even though `mask_to_stl` wrote a valid
  closed surface and parity still round-trips it exactly; face-connected
  solids (the physical case) are unaffected.  None of the 8 corpus
  designs pinched.
- **Boxel bit-exactness is a same-grid property.**  Re-voxelising a
  boxel STL on a *coarser/finer* grid, a tessellated smooth mesh, or a
  rotated/scaled placement is a different operation with different
  error semantics — that is the `cad_stl` tolerance path, not this
  guarantee.

## What this does NOT prove

- **Out-of-family truth still does not exist.**  This campaign validates
  the *mechanism* (STL bytes -> mask -> SDF -> serving composition) on
  corpus designs; it injects no new ground truth.  Accuracy of
  new-geometry serving is the 2026-09-04 e2e LODO result, unchanged by
  this module.
- The zero serving shift is a **consistency** result (STL leg == CAD leg
  bit-for-bit), not an accuracy claim.
- An STL of a hull *outside* the SUBOFF family enters the encoder
  exactly the same way, but no LBM reference and no corpus-neighbour
  guarantee exist for it — the FieldProvider guard and UQ policy still
  apply, unchanged.
