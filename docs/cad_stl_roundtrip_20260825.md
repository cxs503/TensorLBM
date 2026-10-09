# CAD STL round trip for the parametric SUBOFF — 2026-08-25

Closing the loop promised by `docs/voxelize_stl_20260824.md`: the same
analytic description `suboff_cad` turns into corpus masks is
tessellated, exported as a binary STL the way a CAD tool would, and
re-read through the generic `voxelize` intake. If the STL path lands
on the corpus masks cell-for-cell, then geometry authored in real CAD
software lands where the training data lives.

Module: `src/tensorlbm/cad_stl.py`. Tests: `tests/test_cad_stl.py`
(11 tests). Artifacts: `/nfs/wangxi/runs/cad_stl_20260825/`
(`gate.json`, `resolution_study.json`, `suboff_full.stl`,
`suboff_bare_hull.stl`).

## 1. Geometry conventions map

| quantity | `suboff_cad` (analytic) | `cad_stl` (tessellation) | `voxelize` (rasterisation) |
|---|---|---|---|
| streamwise axis | mesh `x`, bow tip `x=0`, stern tip `x=length` | same authored frame, all helpers reused (`_hull_radius_ft`, `_sail_half_thickness_np`, `_naca_thickness_np`, `_HULL_NODES_FT`/`_variant_nodes_ft` axial maps) | last array axis of `(nz, ny, nx)`; `place_on_grid` translates + optionally scales |
| cross axes | `y` lateral, `z` vertical, sail on `+z`, axis at `y=z=0` | identical rings in `(x, y, z)` | `(nz, ny, nx)` array layout, mesh `x` last |
| units | feet internally, `ft * (length / 14.291667)` to lattice units | lattice units, same conversion factor | lattice units, `spacing=1.0` |
| hull-form variants | `l_over_d_mult` / `nose_len_mult` / `stern_len_mult` / `sail_x_mult` piecewise-linear inverse-frame maps | the same `_pw_map_np(x_ft, _HULL_NODES_FT, _variant_nodes_ft(cfg))` applied to station abscissae; radius `r_over_l * length / l_over_d_mult` | n/a (sees only triangles) |
| placement | `build_suboff_mask(cx=0.35*nx, cy=ny/2, cz=nz/2, length=0.6*nx)` | `place_on_grid(hull, shape, scale=1.0)` on the hull (authored bbox is already symmetric in `y`/`z`, so this is an exact placement); the same translation offset is applied verbatim to sail/fins | `place_on_grid` bbox-centre placement |
| sampling | predicates evaluated at **integer lattice nodes** (torch `arange` meshgrid) | n/a | `mask_from_stl` samples **cell centres** `origin + (i+0.5)*spacing`; the round trip passes `origin=(-0.5,-0.5,-0.5)` so both paths evaluate the *same* points |
| composition | `mask | sail_mask | fins_mask` (overlapping closed solids, OR) | per-component tessellation; each closed component voxelised separately, OR'd | ray parity on a single soup of overlapping shells gives the XOR, not the union — hence the per-component path |

Sail/fins overlap the hull by construction (the DARPA sail predicate
grows from the hull axis `z > 0`, the fin roots are buried in the
stern taper), exactly as in `suboff_cad`; the round trip mirrors the
analytic `mask | sail | fins` composition one level up.

## 2. Tessellation and its closed-form checks

Every component is a closed, consistently oriented 2-manifold
(`voxelize.is_watertight` on each; the export report re-verifies the
reloaded soup):

* **hull** — surface of revolution over adaptive axial stations
  (knots + chord-deviation refinement, tolerance `chord_tol` in
  lattice units), bow/stern apex fans at the exact tips.
* **sail** — closed cross-section rings (3-segment DARPA half-width
  polynomial + semi-elliptical cap + bottom face at the axis plane
  `z=0`, matching the predicate) lofted axially, planar end caps.
* **fins** — closed NACA 0015-style rings (SUBOFF coefficients; sharp
  LE/TE as one shared edge) lofted along the span, planar root/tip
  caps; per-fin ring reversal where the mirror sign and the `y/z`
  span-axis swap jointly flip the shell winding.

| component | mesh volume (lu³) | analytic quadrature (lu³) | rel. dev. |
|---|---|---|---|
| hull (bare) | 3825.80 | 3836.23 (`suboff_statistics` displacement) | 0.27% |
| sail (closed section to axis plane) | 49.63 | 50.20 (`∫ [2·w·YTMP + π w²/4] dx`) | 1.1% |
| fins (all four) | 42.69 | 43.06 (`_fins_own_dimensions_ft`) | 0.9% |

Closed forms also pinned in tests: cylinder `π·9·10` and frustum
`63π` within 1%; a swept square ring gives the prism volume `20`
to 1e-9. Note `suboff_statistics` reports the *exposed above-deck*
sail proxy 23.14 lu³ (`h_body = YTMP − deck`), which is a different
solid than the closed section the predicate/mask actually covers; the
50.20 row above is the like-for-like quadrature.

Mother export (`hull_type="full"`, `length=76.8`, defaults
`chord_tol=0.02`, `n_circ=64`): 11616 triangles (hull 5376, sail
1824, fins 4416), 44 hull stations, watertight, reload-watertight,
volume 3918.12 lu³ (sum over components; the buried overlap is
double-counted by design), bbox `[0, −4.481, −4.481] … [76.8, 4.481,
8.397]`, 580884 bytes binary STL.

## 3. Tessellation resolution study

Mother hull, production grid 40×40×128, global IoU and boundary-band
disagreement (XOR fraction over analytic boundary cells) vs axial
station count. `uniform-N` = N stations per hull segment plus the
segment knots (total station count as listed); `adaptive-t` =
chord-deviation refinement with tolerance `t` lu.

| mode | hull stations | triangles | IoU | boundary disagr. | volume ratio | gate |
|---|---|---|---|---|---|---|
| uniform-16 | 19 | 8416 | 0.97979 | 5.306% | 0.97979 | FAIL |
| uniform-32 | 35 | 10464 | 0.98749 | 3.285% | 0.98749 | pass |
| uniform-64 | 67 | 14560 | 0.99519 | 1.263% | 0.99519 | pass |
| uniform-128 | 131 | 22752 | 0.99880 | 0.316% | 0.99880 | pass |
| uniform-256 | 259 | 39136 | 0.99952 | 0.126% | 0.99952 | pass |
| uniform-512 | 515 | 71904 | 0.99976 | 0.063% | 0.99976 | pass |
| adaptive-0.05 | 29 | 9696 | 0.99038 | 2.527% | 0.99038 | pass |
| **adaptive-0.02 (default)** | **44** | **11616** | **0.99423** | **1.516%** | **0.99423** | **pass** |
| adaptive-0.01 | 59 | 13536 | 0.99711 | 0.758% | 0.99711 | pass |
| adaptive-0.005 | 89 | 17376 | 0.99711 | 0.758% | 0.99711 | pass |

Reading: 16-station faceting genuinely fails the gate (IoU 0.980,
boundary 5.3%), so the gate is not vacuous. The default
`chord_tol=0.02` (44 stations, 11616 triangles) sits between
uniform-32 and uniform-64 in triangle count but delivers
uniform-64-class fidelity (IoU 0.99423 vs 0.99519, boundary 1.52% vs
1.26%) because its stations concentrate in the bow/stern tapers where
the radius curvature lives. Refinement past `tol=0.01` plateaus — the
remaining disagreement is voxelisation quantisation, not faceting.

## 4. Gate: all cases at the production 40×40×128 grid

Targets: global IoU > 0.98, boundary-band disagreement < 5% of
boundary cells. Volume ratio = STL-path solid cells / analytic solid
cells. `loc` = fraction of the XOR inside the 1-voxel boundary band
(1.0 = every disagreement sits on the tessellated surface); `iex` =
interior exact (no disagreement outside the band).

| case | hull_type | IoU | bnd disagr. | vol ratio | solids stl/ana | bnd cells | XOR | loc | iex | tris | mesh V lu³ | ana V lu³ | t (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mother | full | 0.9942 | 1.52% | 0.9942 | 4133/4157 | 1583 | 24 | 1.000 | T | 11616 | 3918.1 | 3836.2 | 0.93 |
| bare | bare_hull | 0.9941 | 1.56% | 0.9941 | 4069/4093 | 1538 | 24 | 1.000 | T | 5376 | 3825.8 | 3836.2 | 0.39 |
| with_sail | with_sail | 0.9942 | 1.54% | 0.9942 | 4097/4121 | 1559 | 24 | 1.000 | T | 7200 | 3875.4 | 3836.2 | 0.56 |
| slender (`l_over_d_mult=1.25`) | full | 0.9952 | 1.04% | 0.9952 | 2476/2488 | 1150 | 12 | 1.000 | T | 11232 | 2666.1 | 2584.2 | 0.89 |
| blunt (`l_over_d_mult=0.85`) | full | 0.9985 | 0.45% | 0.9985 | 5159/5167 | 1762 | 8 | 1.000 | T | 12384 | 5146.4 | 5063.5 | 0.97 |
| long_nose (`nose_len_mult=1.4`) | full | 0.9900 | 2.60% | 0.9900 | 3949/3989 | 1536 | 40 | 1.000 | T | 11616 | 3788.7 | 3709.0 | 0.90 |
| aft_sail (`sail_x_mult=1.12`) | full | 0.9942 | 1.52% | 0.9942 | 4129/4153 | 1580 | 24 | 1.000 | T | 11616 | 3918.1 | 3836.2 | 0.88 |

**All 7 cases pass both targets**; summed per-case wall time 5.52 s
(single process, CPU, well under the 2-minute budget). Every
disagreement cell lies in the 1-voxel boundary band — the interiors
agree exactly everywhere; the STL path merely quantises the surface
slightly inside/outside. Per-component IoU at the mother: hull 0.9941
(4069/4093), sail 1.000 (56/56), fins 1.000 (64/64) — the fins are
reproduced **cell-for-cell** on every case in the table.

## 5. What this means for real CAD intake

* A watertight tessellation of the DARPA description at
  ~64-circumferential × chord-tolerance-0.02 resolution reproduces
  the corpus masks to IoU 0.994 with zero interior error — the
  residual is a ~1.5% boundary-band skin. Real CAD geometry that
  arrives as a watertight STL can be rasterised onto the B4 canonical
  grid and trusted to land on the analytic masks to within one voxel
  of the surface.
* **Fin visibility (the PR #235 caveat) does not bite here.** The
  SDF-encoder study flagged barely-one-voxel-thin appendages as the
  fragile class; in this round trip the tessellated NACA slabs
  reproduce the fin cells exactly (IoU 1.000, 64/64 cells on the
  mother). The caveat concerned encoder-side fields, not
  rasterisation parity: `voxelize` resolves thin slabs fine when the
  triangles are watertight and the sampling points are aligned
  (`origin=-0.5`).
* Sharp trailing edges (NACA TE closed to exactly zero thickness, the
  hull tip apices) survive the ray-parity rasteriser as long as each
  component is a *closed* manifold; `is_watertight` on the reloaded
  soup is the cheap pre-flight check to demand of external CAD files.
* Non-convex/overlapping component arrangements (sail buried in the
  hull deck, fin roots in the stern taper) must be voxelised
  per-closed-shell and OR'd — a single merged soup flips parity in
  the overlap and yields the XOR. External CAD assemblies should be
  shipped as separate closed solids (or the intake must split them).

## 6. Reproduction

```bash
cd /nfs/wangxi/worktrees/cad_stl
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= TMPDIR=/nfs/wangxi/tmp
/nfs/wangxi/venvs/tensorlbm/bin/python -m pytest tests/test_cad_stl.py -q --basetemp=/nfs/wangxi/tmp/pt_cadstl   # 11 passed
/nfs/wangxi/venvs/tensorlbm/bin/python - <<'EOF'
from tensorlbm.cad_stl import run_roundtrip_gate
run_roundtrip_gate(out_json="/nfs/wangxi/runs/cad_stl_20260825/gate.json")
EOF
```

Gate cases are `DEFAULT_GATE_CASES` (the 7 above); a custom case list
is `run_roundtrip_gate(cases, out_json)` with
`{"name": ..., "hull_type": ..., "params": {...}}` entries.
