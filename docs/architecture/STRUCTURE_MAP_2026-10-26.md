# TensorLBM Structure Map & Incremental Split Plan

- Date: 2026-10-26
- Basis: `tools/audit_import_graph.py` over `src/tensorlbm`, `app/backend`, `tests`, `app/tests`
- Companion: `QUALITY_REVIEW_2026-10-26.md`, `REPOSITORY_REFACTORING_PLAN.md`

> This document is a **map and a plan**, not an authorization to bulk-move files.
> Every relocation must follow the safety rules in `REPOSITORY_REFACTORING_PLAN.md`:
> search imports/tests/CI/scripts/docs, keep public imports working via shims,
> and ship in small PRs.

## 1. Audit snapshot

| Metric | Value |
|---|---:|
| Discovered modules | 1,080 |
| Internal module edges | 2,955 |
| Dynamic import sites | 32 |
| Import cycles | 11 |
| Cross-layer edges (app → tensorlbm) | 1,431 |
| **Reverse edges (tensorlbm → app)** | **0** ✅ |
| Parse/read errors | 0 |

**Key positive finding:** the static dependency direction is **clean** — the
library never imports the application (`tensorlbm → app` edges = 0). The earlier
"porous boundary" concern (in `STRUCTURAL_AUDIT_2026-10-25.md`) was about
*runtime* `sys.path` injection in `app/backend/main.py`, not static imports.
That runtime coupling should still be removed (depend on the installed
`tensorlbm` package), but the core layering is fundamentally sound.

## 2. Import cycles (modularization debt)

11 cycles were found. The most significant:

1. `tensorlbm <-> backward_facing_step <-> cylinder_flow <-> d3q27_sphere_flow
   <-> dg_lbm <-> lid_driven_cavity <-> sphere_flow <-> turbulence_stats
   <-> turbulent_channel`
   → these are **example/solver entry modules** that reference each other
   loosely. They should not all live at the library root; several belong in
   `examples/` or a `tensorlbm.demos` package.
2. `tensorlbm.ai.inference_service <-> onnx_deploy <-> service_backends`
3. `tensorlbm.apps.* <-> ml.serving` (6-node cycle across AI/apps)
4. `tensorlbm.octree_boundary.{bfl,sharding,stepping}` (3-node)
5. `tensorlbm.octree_boundary.{geometry,qfield,topology}` (3-node)
6. `tensorlbm.phasefield.{evolution_stream_loop,phase_inventory_flux}`
7–11. smaller 2-node cycles (`drag_pressure/stl_geometry`, `geometry_voxel/
   voxel_accel`, `multipatch/surface_refinement`, plus the `backends` and
   `d3q27` self-cycles).

## 3. Highest fan-in flat modules (keep stable, do NOT rename)

These are already well-modularized and heavily depended upon — treat as stable
core; prefer thin wrappers over moves:

| Module | Incoming edges |
|---|---:|
| `tensorlbm.d3q19` | 306 |
| `tensorlbm.d3q27` | 139 |
| `tensorlbm.solver3d` | 117 |
| `tensorlbm.d2q9` | 94 |
| `tensorlbm.boundaries3d` | 81 |
| `tensorlbm.turbulence` | 57 |
| `tensorlbm.solver` | 55 |
| `tensorlbm.suboff_cad` | 53 |
| `tensorlbm.boundaries` | 37 |
| `tensorlbm.utils` | 34 |

## 4. Proposed subpackage regrouping of the 353 flat root modules

Group the remaining flat modules by physics domain (proposal only — each needs a
reference search + shim before execution):

| Proposed package | Candidate flat modules (by name pattern) |
|---|---|
| `tensorlbm.boundary` | `boundaries`, `boundaries3d`, `boundaries_d3q27`, `boundary_*`, `bouzidi*`, `far_field*`, `sponge*`, `rough_wall*`, `moving_wall*`, `zou_he*`, `wall_*` |
| `tensorlbm.collision` | `bgk*`, `mrt*`, `trt*`, `cumulant*`, `cascaded*`, `advanced_collision*`, `collision_*`, `chunked_collision`, `cg_*_collision` |
| `tensorlbm.amr` | `adaptive_*`, `amr_*`, `refinement`, `multipatch`, `surface_refinement` |
| `tensorlbm.multiphase` | `shan_chen*`, `color_gradient*`, `free_surface*`, `free_energy*`, `droplet*`, `spinodal*`, `phasefield*` (split from cycle) |
| `tensorlbm.thermal` | `thermal*`, `cht*`, `rayleigh*`, `boussinesq*`, `cavity_*` |
| `tensorlbm.marine` | `suboff_*`, `ship_*`, `propeller_*`, `wave_*`, `wigley*`, `kcs*`, `series60*`, `offshore_cad` |
| `tensorlbm.geometry` | `cad_*`, `stl_*`, `voxel*`, `geometry*`, `octree_*` (split from cycle) |
| `tensorlbm.turbulence` | `les_*`, `rans_*`, `k_epsilon*`, `komega*`, `smagorinsky*`, `wale*`, `vreman*`, `dynamic_*` |
| `tensorlbm.acoustics` | `acoustics*`, `fwh*`, `beamforming*` |
| `tensorlbm.demos` (or `examples/lib`) | `cylinder_flow`, `sphere_flow`, `lid_driven_cavity`, `turbulent_channel`, `backward_facing_step`, `dg_lbm`, `d3q27_sphere_flow`, `turbulence_stats` (breaks cycle #1) |

## 5. Incremental execution plan

Each step is one small PR:

1. **Thin-subpackage pattern (pilot):** create `tensorlbm/boundary/` with the
   boundary modules, add `tensorlbm/__init__.py` re-exports (compatibility
   shims) so all existing `from tensorlbm import boundaries3d` imports keep
   working, update intra-package imports, run `pytest` + layout checker.
2. Repeat per domain package from §4, highest-fan-in-stability first
   (`turbulence`, `marine`, `geometry`).
3. **Break cycle #1:** move the demo/example entry modules out of the library
   core into `examples/lib` (or `tensorlbm.demos`); update `examples/*.py`
   imports. This removes the largest cycle without touching numerics.
4. **Thin `__init__.py`:** once modules live in subpackages, replace the 2,903
   line lazy table with explicit re-exports from subpackages; keep a
   documented deprecation window for any legacy root names.
5. **Split巨石模块** (`aircraft_icing.py` 4,535; `postprocess.py` 3,542;
   `autograd_calib.py` 2,720; `triton_fused_obstacle.py` 2,662): split by
   responsibility within their own package, one file per PR.

## 6. First safe pilot recommendation

Start with **`tensorlbm.boundary`** (step 1): it is self-contained, high-value
(8+ boundary modules), and the shim approach is low-risk. Do **not** begin with
the demo modules (cycle #1) — that needs the example-import audit first.

> Status: map produced; no core modules relocated yet. Relocation is gated on
> per-cluster reference search + compatibility shims, per the refactoring plan.
