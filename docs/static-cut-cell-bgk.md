# Static cut-cell nonuniform finite-volume BGK

`CutCellBGK2D` advances **nonuniform** population states using local merged
control volumes, directional upwind face fluxes, stationary diffuse wall
fluxes and continuous-time BGK collision. It is a separate module: the
published integer-boundary, swept-rectangle geometry and uniform-moving
verification sources remain unchanged.

**Moving geometry is not implemented** and a nonzero rectangle velocity is
rejected. Passing the earlier uniform-moving proof does not supply a moving
nonuniform fluid solver. All physical qualification flags remain false.

## State, units and boundary conditions

Geometry is the existing exact stationary rectangle. Outer domain faces are
periodic, with one shared interface flux for each periodic pair. Inner wall
normals point from liquid into solid. Areas include the declared extrusion
thickness; populations are extensive mass:

```
Q_i,g [kg] = V_g [m³] f_i,g [kg/m³]
ρ_g = Σ_i Q_i,g / V_g
u_g = Σ_i c_i Q_i,g / Σ_i Q_i,g
```

The authoritative state is `group_Q[G,9]`. The `Q` property reconstructs cell
masses `[9,y,x]` from group density and exact cell volume. The constructor
accepts scalar or cell density, constant or cell velocity, or explicit
nonnegative population densities. Local velocity must remain within
`0.1 * kinetic_speed`. Solid cells have zero mass; no internal liquid or
population reservoir is retained.

## Small cells and actual transport

Fluid cells smaller than `merge_fraction * dx²` merge deterministically across
the largest shared open face, with volume and cell index tie-breaks. Union
find fixes the control-volume groups for all steps of the **static** geometry.
All group masses, volumes, exterior faces and body facets are aggregated;
internal group interfaces contribute no flux. Initial cell populations are
projected conservatively to the merged groups. The requested initial cell
state, mass/momentum changes and actual mechanical-energy mixing change are
published, so this initial smoothing is not hidden.

For each Cartesian interface and each direction, the donor population is
selected by the sign of `c_i·n`. The same flux is added and subtracted to
neighboring groups. For each actual wall facet, outgoing populations come
from that facet's local group. Incoming populations use stationary-wall
D2Q9 equilibrium weights, normalized by the outgoing relative mass rate.
Thus each facet has zero wall mass flux. Body impulse is the sum of the
actual population momentum fluxes; no measured-force adjustment exists.

The CFL limit is calculated from each group's volume and every direction's
outgoing Cartesian and wall capacity. A rejected timestep or state does not
advance populations, clock, history or step evidence. Negative populations,
zero fluid mass, invalid finite parameters and excess Mach speed fail closed.
There is no global density correction or rescaling.

## Collision and physical relaxation

Strang steps use half collision, full first-order upwind transport, half
collision. Collision is the positive exact relaxation mixture

```
Q* = exp(−h/τphysical) Q + (1−exp(−h/τphysical)) Veq(ρ,u)
```

The conserved collision moments are independently tested. `τphysical` is a
continuous kinetic relaxation time in seconds. Its hydrodynamic nominal
viscosity is `c_s² τphysical`, with `c_s² = kinetic_speed²/3`. **The usual
stream-and-collide lattice formula involving `(tau−0.5)` is not used.**
Upwind numerical diffusion and finite kinetic scale remain present and must
be measured; the nominal viscosity is not a wall-flow validation result.

## Evidence and failed convergence

Nine actual cases cover stationary equilibrium with cut cells, nonuniform
cut-cell waves, time/space refinement, periodic uniform coflow, three grids
of a nonuniform shear Fourier mode, and a same-grid time refinement. All
checkpoints continue bitwise. Small-cell cases genuinely exercise merging:
the coarse and fine geometries merge respectively 2 and 5 active cells.

For the nonuniform cut-cell case, peak force varies **0.903%** on refinement,
but total body impulse changes **about 4.46%**, failing the fixed **3%** gate.
The peak occurs at initialization, so its near-zero timestep sensitivity
alone is insufficient. The time-halved impulse changes approximately **0.24%**.
The complete dynamic grid gate remains failed; physical qualification remains
false.

The shear reference solves the **linearized continuous D2Q9 kinetic Fourier
system**, using a published 9×9 generator and matrix exponential. It is not
an exact nonlinear Navier–Stokes solution or a manufactured forcing test.
Amplitude is 1e−4 m/s to expose the linear kinetic limit. Fixed domain,
relaxation time and physical duration are used on all three grids. Errors
are **3.064%, 1.619%, 0.832%** on 16², 32², 64² grids. The finest passes the
fixed 3% comparison. A separately published semidiscrete upwind reference
isolates the temporal error: 32² time-halving reduces it **0.347% → 0.171%**.
The continuous-reference error increases slightly under that time refinement
because spatial and temporal errors had partially canceled; both are retained.

Per-stage kinetic/compressive energy and population entropy are recorded,
including initial merge smoothing. Algebraic energy-stage accounting is not
proof of a continuum energy law or an H theorem for polynomial equilibrium.

## Reproduce and raw API

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/static_cut_cell_bgk/run.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_cut_cell_bgk_2d.py
```

[Study and fixed gates](assets/static-cut-cell-bgk/study.json). Lossless
compressed raw cases include requested initial cell masses, fixed geometry
and group metadata, every before/after group state, half-collision and
transport states, actual Cartesian/wall fluxes, SI ledgers, source SHA256s
and bitwise restart evidence. Independent reconstruction can reproduce the
whole trajectory without importing this solver.

Checkpoints contain configuration, geometry/group SHA256, group populations,
clock, full history and initial projection metadata. The last derived face
flux is not required for subsequent evolution and is regenerated by a step.
Next work is dynamic group construction across old/new swept geometries and
continuous birth/death handling, followed by genuine moving nonuniform
transport and physically calibrated wall/ice response.
