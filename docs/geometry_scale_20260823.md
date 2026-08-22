# Appendage scale: making the SUBOFF geometry axis real (B1-5)

Date: 2026-08-23 · Branch: `feat/geometry-scale` (base 2f75646) · A/B dataset:
`/nfs/wangxi/datasets/b15_ab_20260823` (6 points)

## The problem (from B1-4)

`docs/hull_geometry_campaign_20260823.md` showed the three-configuration
geometry axis is degenerate at production resolutions: the sail adds 28 solid
cells (0.7 % of the hull) at n128 and the same 0.7 % at n256 — refinement is
self-similar and cannot create the axis. Cause: `suboff_cad.py` builds the
sail and fins from the DARPA offset-table **foot constants**
(`_SAIL_ZMAX = 0.109375` ft, `_SAIL_YTMP = 1.507813` ft, `_SAIL_X1_START …
_SAIL_X3_END`, `_FIN_H/_FIN_R_INNER/_FIN_R_OUTER/_FIN_SWEEP_K/_FIN_SWEEP_C`),
which the sail half-width makes only 0.109 ft ≈ 0.59 lattice cells wide at
n128.

## Verification: SuboffConfig vs the CAD predicates

Read straight from the code (all of `src/tensorlbm/suboff_cad.py`):

1. **The `sail_*_frac` / `fin_*_frac` fields are dead config.** `_add_sail_mask`
   and `_add_fin_masks` receive `config` but never read it; the fields are
   consumed nowhere in `src/` or `tests/` (grep is empty outside the dataclass
   definition and its docstring). They are descriptive metadata of the DARPA
   proportions, not parameters of the generated solid. `bow_fraction` /
   `stern_fraction` / `stern_exponent` are in the same state — the profile
   uses the hard-coded Groves et al. ft polynomials.
2. **Two parallel implementations of the appendage geometry exist.** The voxel
   builders (`_add_sail_mask` / `_add_fin_masks`, used by `build_suboff_mask`)
   and the continuous point predicates (`suboff_sail_contains_points` /
   `suboff_fins_contain_points` / `suboff_appendages_contain_points`, used by
   `interpolated_bc_suboff.refine_q_suboff_appendages`) duplicate the same ft
   constants. They were verified to agree bitwise at cell centres
   (`mask == hull | predicate`, see the regression test), and the scale below
   is threaded through **both**, so the pair cannot drift apart on the new
   axis. The mesh/STL lofters (`_real_sail_triangles` / `_real_fin_triangles`
   via `suboff_mesh_data` / `export_suboff_stl` / `suboff_appendage_triangles`)
   are a third consumer and also take the scales.
3. Only `r_over_l` (auto-radius) plus the two new `*_scale` fields actually
   parameterise the geometry. This is now stated in the `SuboffConfig`
   docstring so nobody re-discovers it the hard way.

## The scale axis

`SuboffConfig.sail_scale: float = 1.0` and `fin_scale: float = 1.0` multiply
each appendage's **own** dimensions — length × height × width together —
about fixed DARPA anchors, implemented as an inverse similarity on the query
coordinates so the exact original predicates are reused:

- **sail**: axial stretch about the footprint centre `_SAIL_X_CENTER`
  (3.637 ft), vertical about the midbody deck plane `_SAIL_Z_DECK`
  (= R_max = 0.833 ft, so it always grows out of the hull face), width about
  the centreplane;
- **fin**: chord about the common trailing edge `x = _FIN_H` (13.146 ft),
  radial span about the root radius `_FIN_R_INNER` (0.075 ft, buried in the
  stern taper), NACA thickness × s.

Constraints, all pinned by `tests/test_suboff_appendage_scale.py` (31 tests):

- `scale = 1.0` is guarded to be the identity path — masks are **bit-identical
  with the pre-scale code** (verified against a base-2f75646 module import:
  `torch.equal` True for all three hull types; solid cells 4093 / 4121 / 4157
  at the n128 production grid);
- scaled masks nest (`mask(1) ⊆ mask(2) ⊆ mask(3)`) and solid counts are
  strictly monotone; the bare hull is untouched by any scale;
- every appendage connected component stays attached to the (dilated) hull at
  s = 3 (BFS check) — the sail axial centre and the fin trailing edge are
  pinned to ±1 cell;
- stats follow exact similarity laws: `*_own_volume_lu3(s) = s³·V(1)`,
  `*_own_wetted_area_lu2(s) = s²·A(1)` (analytic quadrature of the DARPA
  profiles), plus voxel-truth keys `bare_hull_solid_cells` /
  `appendage_solid_cells`;
- the `suboff_n128` case takes `sail_scale` / `fin_scale` constructor kwargs
  (→ `SuboffConfig` → mask, echoed in `metadata()`), so the scan chain sweeps
  them as plain numeric params (`coerce_case_params` passes floats through).

## Mask table (n128, production placement cx = 0.35·nx, L = 0.6·nx)

Bare hull 4 093 solid cells; percentages are appendage share of that.

| hull | (sail, fin) | solid | appendage cells | % of hull |
|---|---|---|---|---|
| with_sail | (1, —) | 4 121 | 28 | **0.68 %** (B1-4 baseline) |
| with_sail | (2, —) | 4 317 | 224 | 5.47 % |
| with_sail | (3, —) | 4 891 | 798 | 19.50 % |
| full | (1, 1) | 4 157 | 64 | 1.56 % |
| full | (1, 2) | 4 353 | 260 | 6.35 % |
| full | (1, 3) | 4 977 | 884 | 21.60 % |
| full | (2, 2) | 4 549 | 456 | 11.14 % |
| full | (3, 3) | 5 747 | 1 654 | 40.41 % |

Full-configuration grid (appendage % of hull, rows = sail_scale,
cols = fin_scale): s=1: 1.56 / 6.35 / 21.60 — s=2: 6.35 / 11.14 / 26.39 —
s=3: 20.38 / 25.16 / 40.41. The axis spans a ×26 range instead of the
degenerate 1.56 % of B1-4.

## C_D separation A/B (6 points, production chain)

`scale_ab_campaign_launcher.py` — full configuration, scale combos
(1,1)/(2,2)/(3,3) × Re {200, 600}, 64×64×128, cumulant, u_in = 0.1,
2 500 steps, `DragSurveySpec(margin=4, interval=25)`, same executor as B1-4.
Tail = last 25 % of the 100 drag samples; tail CV ≤ 0.1 % on every point
(fully steady). `C_D = 2·F_x_tail/(ρ·u_in²·A_proj)` with each point's own
projected frontal area (the B1-3/B1-4 surrogate label); `C_D@A1` keeps the
scale-1 area as a fixed reference.

| point | (sail, fin) | Re | F_x_tail | A_proj | C_D | C_D@A1 |
|---|---|---|---|---|---|---|
| p0000 | (1,1) | 200 | 2.5186 | 73 | 6.900 | 6.900 |
| p0002 | (2,2) | 200 | 3.1125 | 129 | 4.826 | 8.527 |
| p0004 | (3,3) | 200 | 4.4074 | 177 | 4.980 | 12.075 |
| p0001 | (1,1) | 600 | 1.2233 | 73 | 3.351 | 3.351 |
| p0003 | (2,2) | 600 | 1.6054 | 129 | 2.489 | 4.398 |
| p0005 | (3,3) | 600 | 2.3964 | 177 | 2.708 | 6.565 |

Separation read-out:

- raw tail force (fixed geometry, growing appendages): **+75 % (Re 200)** and
  **+96 % (Re 600)** from s=1 → s=3, strictly monotone — a far stronger
  geometry signal than B1-4's 8.23 vs 8.14 mean C_D across hulls;
- fixed-reference-area coefficient `C_D@A1` is monotone in scale at both Re
  (6.90 → 8.53 → 12.08 and 3.35 → 4.40 → 6.57);
- own-frontal-area `C_D` (the surrogate label) separates s=1 from s≥2 by
  **30–43 %** but saturates between s=2 and s=3 (the projected area grows
  ×2.4 while force grows ×1.75 at Re 200 and ×1.96 at Re 600): worth knowing
  before training — on this label the axis is step-like, not linear in scale.

## Scan recommendation

For the next surrogate campaign (B1-6 candidate): sweep `hull_type` (3) ×
`sail_scale` ∈ [1, 3] × `fin_scale` ∈ [1, 3] (LHS or a coarse grid, log- or
linear-uniform both defensible since the mask response is near-linear in s —
appendage cells 64/260/884 along either axis) × Re as in B1-4, with
`drag_survey` unchanged. Expect the C_D label to be dominated by the
s=1 → s≥2 transition; include intermediate values (e.g. s ∈ {1, 1.5, 2, 3})
if a smooth regression target matters. Everything flows through
`ScanPoint.params` already — no runner changes needed.

## Files

- `src/tensorlbm/suboff_cad.py` — scale fields, anchors, remap helpers,
  both predicates, voxel builders, stats quadrature, mesh/STL lofters
- `src/tensorlbm/cases/suboff.py` — `sail_scale` / `fin_scale` case kwargs
- `tests/test_suboff_appendage_scale.py` — 31 tests (bitwise scale-1 pins
  vs base 2f75646, monotonicity/nesting, similarity laws, attachment BFS,
  predicate/voxel agreement, case wiring)
- `scale_ab_campaign_launcher.py`, `scale_ab_analysis.py` — A/B run + analysis
