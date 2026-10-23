# Case study: SUBOFF hull geometry — metre vs lattice-unit mismatch

* **Module:** `tensorlbm.triton_fused_obstacle` — `src/tensorlbm/triton_fused_obstacle.py`
* **Function:** `create_suboff_obstacle_torch` (def at line 1853); fixed comparison at lines 1932–1938
* **Follow-up fix:** swap-at-solid second pass in the fused kernel `_fused_v2_kernel_xfar_les` (def at line 773, block at lines 880–892)
* **Original bug report:** `/tmp/suboff_test/BUG_REPORT_SUBOFF_unit_mismatch.md`

## Summary

`create_suboff_obstacle_torch` built a solid rectangular **prism** where it
should have built a slender SUBOFF **hull**. The cross-section test compared a
left-hand side in **square metres** against a right-hand side in **square
lattice units**, so the numeric value of the lattice-unit radius was
interpreted as a physical radius in metres. This inflated the effective hull
radius by a factor `1/dx` (≈ 23.5× at `n = 256`), the cross-section area by
`1/dx²` (≈ 585×), and produced ~2 million spurious solid cells: the "hull"
filled the entire cross-stream plane over its whole length.

Two downstream symptoms traced back to this single line:

* The measured "drag" was not drag at all — the flow saw a slab blocking the
  full cross-section, so the reported streamwise force was dominated by
  lateral deflection around a broad face.
* A persistent `F_y ≈ 18` anomaly at `n ≥ 256`: the prism's obstruction of the
  inlet plane broke the symmetry of the inflow aperture and injected a
  cross-stream momentum flux. `F_y ≠ 0` was a real geometric consequence of
  the bug, not a numerical transient.

The fix is a one-line unit correction in the geometry builder (compare both
sides in metres), plus a follow-up "swap-at-solid" second pass in the fused
Triton kernel that matches the PyTorch `bounce_back_cells_3d` semantics.

## Root cause

`create_suboff_obstacle_torch` builds the obstacle mask on a
`(nz, ny, nx)` meshgrid, with the SUBOFF long axis on LBM axis 0
(streamwise). Local coordinates are scaled to **metres** by `dx`
(metres per lattice unit), and the SUBOFF polynomials are parameterised in
**feet** from the nose:

```python
x_local = (X - cx) * dx                  # metres (SUBOFF axial coordinate)
y_local = (Y - cy) * dx                  # metres (cross-section)
z_local = (Z - cz) * dx                  # metres (cross-section)

ft_per_lx = 1.0 / 0.3048
x_ft = x_local * ft_per_lx               # feet, for the SUBOFF polynomials

# ... R assembled per segment (nose / cylinder / tail / end cap) ...
```

After assembling the segment radii `R`, the pre-fix code converted the radius
to lattice units and then compared it against metre-valued coordinates:

```python
R_lx = R / ft_per_lx / dx                # radius in LATTICE UNITS (a cell count)
# ...
hull = ((y_local ** 2 + z_local ** 2) < R_lx ** 2).to(torch.int8)
#       \------------ metres² ----------/     \---- lx² ----/
```

The left-hand side is a squared distance in metres; the right-hand side is the
square of a **cell count**. Python and PyTorch happily compare the raw floats,
so nothing flags the mismatch — the shape is silently wrong.

## Why it's wrong

The correct cross-section test can be written in either unit system, but both
sides must agree:

```text
lattice units:   ((y_local/dx)**2 + (z_local/dx)**2) < R_lx**2
metres:          (y_local**2 + z_local**2)          < R**2      (R in metres)
```

The buggy line drops the `/dx` on the left while keeping the lattice-unit
radius on the right. A point at physical distance `r` from the hull axis
passes the test whenever `r < R_lx` *numerically*, i.e. the effective radius
becomes `R_lx` **metres** — equivalently `R/dx²` cells instead of `R/dx`
cells. The inflation factor is exactly `1/dx`.

At `n = 256`, `dx = 0.0425 m/lx` and the SUBOFF cylinder radius is
`R = 0.254 m`:

| quantity                        | correct | buggy  | factor |
|---------------------------------|--------:|-------:|-------:|
| cross-section radius (cells)    |    5.97 | 140.4  | 23.5×  |
| cross-section area (cells²)     |     112 | 65536  | 585×   |

The buggy radius (140 cells) exceeds the half-domain (128 cells), so the
cross-section is clipped by the domain walls: the "hull" spans the full
`y`–`z` plane at every axial station, yielding a solid box
`117 × 256 × 103 ≈ 3.1M` cells instead of a slender body. Almost 2M of those
cells are spurious solid.

The streamwise force therefore acts on a slab perpendicular to the flow, and
the off-centre obstruction of the inlet plane breaks inflow symmetry —
explaining both the bogus drag magnitude and the `F_y ≈ 18` anomaly.

Other suspects were investigated and ruled out before the geometry was
isolated (details in the original bug report): the OPPOSITE lattice table
(verified on GPU with a sphere obstacle, `max |u| = 0` after 50 steps), the
Zou-He inlet direction sets (verified vectorised), and the `1e-4 sin(y)`
initial perturbation (zero effect on `F_y` — reproduced identically with a
uniform-equilibrium initial condition).

## The fix

### 1. Unit-consistent cross-section test (lines 1932–1938)

The comparison now uses `R` directly, so both sides are in metres.
`src/tensorlbm/triton_fused_obstacle.py`, in `create_suboff_obstacle_torch`:

```python
    # Convert R back to lattice units.
    R_lx = R / ft_per_lx / dx
    # NOTE: y_local, z_local are in METRES.  Compare against R (also in metres),
    # not R_lx (lattice units).  Comparing m^2 to lx^2 inflated the cross-section
    # radius by 1/dx ≈ 23.5x, turning the slender SUBOFF hull into a solid
    # prism filling the entire y-z plane.  See BUG_REPORT_SUBOFF_unit_mismatch.md.
    hull = ((y_local ** 2 + z_local ** 2) < R ** 2).to(torch.int8)
```

`R_lx` is retained (it is no longer used by the hull test) so the lattice-unit
radius stays available for callers that want a cell-count radius.

### 2. Swap-at-solid second pass in the fused kernel (lines 880–892)

A geometry fix alone was not enough: the Triton path applied wet-node
bounce-back only at fluid cells facing a solid source cell, while the PyTorch
reference `bounce_back_cells_3d` (`src/tensorlbm/boundaries3d.py:112`) also
swaps populations **inside every solid cell** after streaming. The fused
kernel `_fused_v2_kernel_xfar_les` gained a second pass to match that
semantics:

```python
    # === Swap-at-solid (matches PyTorch ``bounce_back_cells_3d``) ===
    # PyTorch's BB applies ``f[q, x_solid] = f[opp_q, x_solid]`` to
    # EVERY solid cell after streaming — interior solid cells included.
    # This zeros u at solid cells in the next collide, enforcing
    # no-slip at the fluid-solid interface.  We add it as a SECOND
    # pass over ``f_eff`` AFTER wet-node BB so the order is:
    #   fluid cell, src=solid: wet-node BB fires → use f_own_opp
    #   solid cell (any): swap-at-solid fires → use f_own_opp
    #   fluid cell, src=fluid: untouched (f_in)
    f_eff = tl.where(own_is_wall, f_own_opp, f_eff)
```

With `f[q] ↔ f[opp_q]` applied at all solid cells, the next collide step
yields `u = 0` there, enforcing no-slip at the fluid–solid interface exactly
as the PyTorch reference does. The pass deliberately sits **before** the Ladd
force block, so the momentum-exchange force still samples the post-stream,
pre-bounce-back state via `f_in`.

### Verification checklist

1. Cross-section inspection: the y/z span of the obstacle at a cylinder
   station should collapse to ~12 cells (a 6-cell-radius disc) at `n = 256`.
2. Force components: `F_x` and `F_y` should drop to ~0.01 at all `n`, and the
   streamwise force should scale linearly with `n` (frontal area ∝ `n` when
   `dx ∝ 1/n`).
3. Drag coefficient: re-run the production benchmark at `n = 256` and compare
   against the ITTC-1957 expectation for AFF-8 at the target Reynolds number.

### Lessons

* Carry the unit in the variable name (`R_m`, `R_lx`) whenever two unit
  systems meet in one expression; a bare float comparison hides the bug.
* Validate obstacle masks geometrically (span counts, cross-section slices,
  solid-cell totals) *before* trusting any force or drag number derived from
  them — a unit error in geometry construction is invisible to the kernels.
* When porting a solver to a new backend, diff the boundary-condition
  semantics cell-class by cell-class (fluid/solid, interior/interface); the
  wet-node-only bounce-back matched momentum exchange but missed the
  interior-solid swap that the reference implementation relied on.
