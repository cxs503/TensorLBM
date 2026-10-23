# General voxelisation: arbitrary STL -> mask / SDF on the B4 grid (2026-08-24)

Module: `src/tensorlbm/voxelize.py` (numpy + stdlib only — no trimesh,
scipy or torch).  Tests: `tests/test_voxelize.py`.  Benchmark:
`benchmarks/b4_voxelize_bench.py`.

This closes the roadmap Phase-2 gap "通用体素化（STL→SDF）": geometry
coming out of arbitrary CAD software can now enter the drag-surrogate
stack.  It complements `tensorlbm.geometry_voxel` (torch/Triton, built
for BFL q-fields on GPU): `voxelize` is the CPU front-end whose output
is the boolean occupancy mask that the SDF encoder input contract
specifies.

## API contract

| Function | Purpose |
| --- | --- |
| `load_stl(path) -> StlMesh` | binary + ASCII STL reader, auto-detected. |
| `is_watertight(tris, *, weld_tol=1e-6)` | closed + orientation-consistent check. |
| `mask_from_stl(tris, shape, *, origin, spacing, axis=0, robust=True)` | ray-parity boolean mask. |
| `place_on_grid(tris, shape, *, scale=None, center_frac=(0.35,0.5,0.5), streamwise_frac=0.6)` | B4 canonical-frame placement. |
| `sdf_from_mask(mask, *, spacing=1.0, exact=None)` | signed distance seam (negative inside). |

`StlMesh` is a frozen dataclass `(vertices (T,3,3) float64, normals
(T,3) | None)`; every entry point accepts either an `StlMesh` or a raw
`(T,3,3)` array.

### STL loader

* Binary detection: exact size match `84 + 50*n_tri` wins even when the
  80-byte header starts with `solid` (common binary-exporter quirk).
* ASCII tolerated quirks: CRLF, trailing whitespace, `solid`/`endsolid`
  names, blank lines, empty solids (returns `T == 0`), per-facet
  missing `normal` lines.
* `normals` is `None` when any entry is non-finite or every normal is
  zero (as written by several exporters).
* Rejected with `ValueError` naming the **byte offset**: truncated
  binary (offset = end of last complete record), truncated ASCII
  (unterminated facet / malformed number / vertex with < 3 coords),
  files too short to be either format, empty files.

### Watertight semantics

`is_watertight` welds vertices on a `weld_tol` grid (STL repeats shared
vertices bit-identically), then requires: no degenerate faces, no
directed edge repeated, and every directed edge `(a, b)` present exactly
once with its opposite `(b, a)` — closed, edge-manifold and
consistently oriented.  Ray parity does **not** require it; the module
reports it because interiors of open meshes leak (whole wedges gain or
lose odd crossings).  Check it on every CAD import.

## Conventions (pinned by unit tests)

* Triangle tables `(T, 3, 3)` `[triangle, vertex, axis]`, mesh axes
  `(x, y, z)`.
* Masks are `(nz, ny, nx)` bool with mesh **x on the last array axis**,
  matching the lattice layout `(Q, nz, ny, nx)`; B4 streamwise = x.
* Cell-centre sampling: the sample of voxel `(iz, iy, ix)` is
  `origin + (ix + 0.5, iy + 0.5, iz + 0.5) * spacing` in mesh coords.
  Integer-aligned box faces therefore give an exact **half-open** voxel
  box (min faces in, max faces out) — the exact-mask test pins this.
* `axis` selects the mesh axis the ray travels along (default 0 = x,
  streamwise); the output layout is identical for every ray axis (all
  three agree on closed surfaces, tested).
* `place_on_grid` scales uniformly (aspect preserved) so the bbox
  streamwise (x) extent hits `streamwise_frac * nx`, then centres the
  bbox on `center_frac * (nx, ny, nz)`.  Defaults reproduce the SUBOFF
  convention: hull centred `cx = 0.35 * nx`, length `0.6 * nx`, nose at
  `0.05 * nx` upstream.  Returns `Placement(tris, origin=(0,0,0),
  spacing=1.0, scale, streamwise_extent)`; `tris` are in voxel-index
  coordinates and feed straight back into `mask_from_stl`.

## Robustness notes (ray parity)

One ray per voxel column; a voxel is inside iff an odd number of
triangle crossings lies strictly ahead of it (vectorised
Moller-Trumbore, column tiles of ~1e6 ray-triangle pairs, transverse
bbox culling; crossings beyond the domain still count).

* `robust=True` (default): deterministic sub-cell asymmetric ray-origin
  perturbation in the two transverse coordinates (+1.3e-4 / -3.7e-4
  cells, mirroring `geometry_voxel` / `preprocess_geo`) plus strict
  barycentric bounds `u > 0, v > 0, u+v < 1`.  This kills the classic
  grid-aligned failure: a crossing exactly on an edge shared by two
  triangles double-counts under the naive inclusive rule and flips the
  parity of every voxel behind it (one-column holes).  The
  degenerate-config test puts box face diagonals and shared vertices
  exactly on ray lines and pins the exact expected mask.
* A ray whose transverse coordinate lies exactly on a face is resolved
  to the perturbed side (documented in the test): such masks shift by
  at most one cell transversely.
* A crossing exactly on a voxel *sample* (ray-axis face aligned with
  the sample grid) resolves by fp rounding of the intersection
  parameter — deterministic per mesh but effectively arbitrary; place
  such faces between samples or accept a one-cell shift.
* Residual failure mode: a ray exactly on a mesh edge *after* the
  perturbation (adversarial coordinates only) drops one crossing; at
  most one column is affected.
* `robust=False` keeps the naive inclusive rule, exists only to
  demonstrate the failure mode.

## SDF seam

`sdf_from_mask` returns the signed distance, **negative inside**,
measured to the set of solid cells with at least one fluid 6-neighbour
(domain border counts as fluid).  Back-ends: exact brute force
(default for grids <= 32^3 cells, `exact=True` forces it; practical to
~64^3) or a two-pass 3-4-5 chamfer (<= ~11% worst-case relative error
on shallow diagonals, a few percent typical — the test compares it
against the brute-force distance).

This is deliberately **not** the boundary-restricted exact EDT that
lives in `geom_encoder` (PR #235): duplicating it here would create a
merge conflict.  Post-#235-merge, callers building the encoder input
should prefer `geom_encoder`; keep `sdf_from_mask` for standalone and
diagnostic use.

## Benchmarks

`PYTHONPATH=src python benchmarks/b4_voxelize_bench.py` (5090 server,
tensorlbm venv, CPU numpy).  One JSON line per config; the mask column
is `mask_from_stl` on the placed mesh, `robust=True`.

| mesh | tris | grid | watertight | mask_s | solid cells |
| --- | --- | --- | --- | --- | --- |
| icosphere subdiv 4 | 5120 | 32x32x64 | true | 0.184 | 27288 |
| icosphere subdiv 4 | 5120 | 64x64x128 | true | 0.648 | 218187 |
| prolate SUBOFF-like L/D 6 | 5120 | 32x32x64 | true | 0.235 | 3252 |
| prolate SUBOFF-like L/D 6 | 5120 | 64x64x128 | true | 0.908 | 26300 |

`is_watertight` ~0.011 s and `place_on_grid` ~0.001 s for both meshes —
all sub-second, i.e. negligible next to any downstream LBM step.

## Post-merge integration plan

1. **mask -> geom_encoder EDT -> SDFCondFNODrag** (after #235 merges):
   `load_stl` -> `is_watertight` sanity -> `place_on_grid` ->
   `mask_from_stl` -> `geom_encoder` boundary-restricted EDT (encoder
   input contract) -> SDFCondFNODrag (#234) inference.  `sdf_from_mask`
   stays as the dependency-free diagnostic path.
2. **Round-trip validation against the shape-family generator** (after
   #236 merges): export `suboff_cad` family masks to STL, re-import
   through this module and compare masks voxel-for-voxel — a
   generative-to-CAD-to-mask consistency gate for the surrogate
   training data.
3. Optional GPU path: meshes beyond ~10^5 triangles can route through
   `tensorlbm.geometry_voxel.voxelize_stl` (torch/Triton) with the same
   `(nz, ny, nx)` convention; this module remains the CPU reference.
