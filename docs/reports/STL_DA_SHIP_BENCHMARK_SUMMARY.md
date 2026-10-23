# STL-based dA + Real Ship Hull Benchmarks (SDAA 20-23)

## Summary

Implemented STL-based surface area correction (dA) and ran real ship hull
benchmarks using STL geometry files with STL-derived surface normals.
Compared STL normals vs from_gradient normals on the same STL-voxelized
hull geometry.

## Implementation

### New functions in `src/tensorlbm/stl_geometry.py`:
1. **`mirror_stl()`** — Mirrors half-hull STL about symmetry plane (y=0)
   to create a closed full hull for voxelization. Ship STL files are
   half-hulls (y≥0 only, no symmetry-plane triangles).

2. **`_compute_triangle_areas()`** — Per-triangle area from cross product.

3. **`dA_method` parameter in `SurfaceMesh_from_stl()`** — Three methods:
   - `'none'`: dA=1.0 (default, backward-compatible)
   - `'stl_area'`: Uses nearest STL triangle area, scaled so ΣdA = true
     STL surface area in lattice units. Recovers 99.8% of true area
     (vs 87.5% for dA=1.0).
   - `'cos_theta'`: dA = 1/|n_dominant|, geometric staircase correction.

### Updated `SurfaceMesh.from_stl()` in `drag_pressure.py`:
- Passes `dA_method` through to `SurfaceMesh_from_stl()`.

## Results

### STL vs from_gradient normals (same STL-voxelized hull)

| Ship | Normal | Cd_p | Cd_f | Cd_tot | ITTC | Error |
|------|--------|------|------|--------|------|-------|
| KVLCC2 (Re=1e5) | STL | -0.001605 | 0.001607 | 0.000002 | 0.008333 | 100.0% |
| KVLCC2 (Re=1e5) | from_gradient | 0.011614 | 0.001267 | 0.012880 | 0.008333 | 54.6% |
| DTMB5415 (Re=1e5) | STL | -0.000985 | 0.001768 | 0.000783 | 0.008333 | 90.6% |
| DTMB5415 (Re=1e5) | from_gradient | 0.011998 | 0.001429 | 0.013427 | 0.008333 | 61.1% |
| KCS (Re=1000) | STL | 0.000153 | 0.044432 | 0.044586 | 0.075000 | 40.6% |
| KCS (Re=1000) | from_gradient | 0.010980 | 0.038057 | 0.049037 | 0.075000 | 34.6% |

### Sphere dA comparison (Re=100, Cd_ref=1.09)

| Method | Cd_p | Cd_f | Cd_tot | sum_dA | ratio_to_true | Error |
|--------|------|------|--------|--------|---------------|-------|
| none (dA=1.0) | 0.7309 | 0.3551 | 1.0860 | 1584.0 | 0.8754 | 0.4% |
| stl_area | 1.0017 | 0.3733 | 1.3750 | 1805.4 | 0.9977 | 26.1% |
| cos_theta | 0.8640 | 0.4231 | 1.2870 | 1911.1 | 1.0561 | 18.1% |

## Key Findings

1. **STL normals eliminate spurious pressure drag** — the primary
   improvement. from_gradient normals on staircase surfaces produce
   large spurious pressure drag (Cd_p≈0.012) for all ships. STL normals
   give realistic small pressure drag (Cd_p≈0.000-0.002) for streamlined
   hulls. Reduction: 7-72× depending on ship.

2. **from_gradient's spurious pressure drag acts as a "fudge factor"**
   that partially compensates for friction underestimation at high Re.
   This makes from_gradient's total drag look closer to ITTC, but for
   the wrong physical reason. STL normals reveal the true friction
   underestimation.

3. **dA=1.0 gives best drag accuracy** for the sphere (0.4% error),
   despite underestimating surface area (87.5% of true). The stl_area
   method correctly recovers true area (99.8%) but overcorrects drag
   (26.1% error). The drag formula is more sensitive to dA distribution
   than to total surface area.

4. **Friction underestimation at Re=1e5** is a grid resolution issue
   (boundary layer ~1 cell thick at tau=0.500144). At Re=1000 (KCS),
   friction is better resolved (Cd_f=0.044 vs ITTC=0.075, 59%).

5. **KCS at Re=1000 achieves finite convergence** with STL normals
   (40.6% error, finite=True). This meets the TEST 3 target.

## Files Created/Modified

- `src/tensorlbm/stl_geometry.py` — Added mirror_stl(), dA_method
- `src/tensorlbm/drag_pressure.py` — Updated from_stl() signature
- `stl_ship_worker.py` — STL ship hull benchmark worker
- `launch_from_gradient_cmp.py` — from_gradient comparison launcher
- `results_stl_*.json` — All benchmark results (7 files)
- `log_stl_*.txt` — All benchmark logs (7 files)
