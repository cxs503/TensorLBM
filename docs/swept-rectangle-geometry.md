# Exact swept-rectangle geometry P0

This module supplies geometry for a future cut-cell finite-volume
D2Q9 solver. **It does not advance fluid populations or qualify hydrodynamic
loads.** The previous integer-mask and balanced-remap source and failed
artifacts remain separate.

## Coordinates and units

`SweptRectangleGeometry(nx, ny, dx_m, width_m, height_m, center_m,
velocity_m_s)` defines an axis-aligned solid rectangle translating at constant
velocity inside `[0,nx*dx] × [0,ny*dx]`. Cell indices are `[y,x]`; grid faces
are at integer multiples of `dx`. All geometry is CPU float64 in SI units.
Areas are two-dimensional volumes per unit extrusion thickness, in m².
**Normals point outward from liquid into solid:** left body edge +x, right
edge −x, bottom +y, top −y.

The body must remain strictly inside the domain at both step endpoints.
For linear translation that also guarantees containment throughout the step.
Periodic wrapping, rotation, nonrectangular bodies and multiple bodies are
unsupported and must not be inferred from this result.

## API

`at(time_s)` returns:

- `fluid_volume_m2[y,x]`: exact cell area minus rectangle intersection.
- `vertical_open_m[y,xface]`, `horizontal_open_m[yface,x]`: shared fluid face
  apertures, in m.
- `wall_facets`: cell, oriented normal, length and center.
- `wall_normal_length_m[y,x,2]` and the local closure residual.

`interval(time_start_s,dt_s)` returns the two endpoint geometries plus:

- `event_times_s`: every body edge crossing a Cartesian grid line.
- `vertical_open_integral_m_s`, `horizontal_open_integral_m_s`: exact
  space-time open face integrals.
- `wall_facets`: oriented facets with integrated length (m·s), first position
  moment (m²·s), owning cell, interval and wall velocity.
- `wall_normal_integral_m_s[y,x,2]`.
- `fluid_volume_change_m2`, `swept_volume_m2` and GCL residual, where
  `swept_volume = velocity · integrated outward fluid wall normal`.

There is no geometric state beyond configuration and clock; those reproduce
any interval. No fluid restart facility is claimed here.

## Exact integration and numerical handling

Between edge/grid crossings, face topology is fixed and each overlap length
is linear in time. Its endpoint trapezoid integrates that linear polynomial
exactly. Wall position times facet length is quadratic, integrated exactly
using three polynomial samples. The midpoint selects cell ownership; it is
**not used as an approximate one-sample space-time integral**.

The geometry must satisfy both cell identities:

```
Σ integrated open face normals + Σ integrated wall normals = 0
Vfluid(t1) − Vfluid(t0) = velocity · Σ integrated wall normals
```

Grid-aligned walls belong to the adjacent liquid cell. Shared Cartesian faces
coincident with the body edge are blocked, avoiding duplicate transport.
Only coordinates within a few floating-point ulps of an exact grid
coincidence are canonicalized; small cut-cell volume is not clamped to a
minimum. No population correction or density rescaling exists.

## Actual evidence

Five cases cover diagonal motion, reverse motion, axial motion, a static
aligned body, and spatial/time refinement. All use the same finite physical
domain; the refined grid doubles each dimension, halves dx and quarters dt.
Moving cases actually contain fluid cell birth/death events.

The declared GCL threshold is `1e-12 * dx²`, closure threshold
`1e-12 * dx * dt`, and total-area threshold `1e-12 * total fluid area`.
All cases pass. Maximum actual GCL residual is **1.44e−17 m²** and total
fluid area error is **2.23e−16 m²**. These are geometry results, not fluid
mass, energy or load convergence results.

The independent audit imports no geometry/solver code. It reconstructs cell
areas from raw configuration, face integrals using two-point Gauss on each
published event interval, wall-normal sums and first moments, and the local
GCL. Full raw geometry records are losslessly gzip-compressed with SHA256
binding; the readable summary remains JSON.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/swept_rectangle_geometry/run.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/swept_rectangle_geometry/audit.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_swept_rectangle_geometry.py
```

[Raw study and fixed gates](assets/swept-rectangle-geometry/study.json).
Next: extensive populations `Q_i=Vfluid*f_i`, shared finite-volume fluxes,
small-cell merging, actual moving-wall impulse and uniform co-moving
transport. A test that merely assigns `Qnew=Vnew*equilibrium` would not verify
transport and is not provided as fluid evidence.
