# P2 boundary investigation: perimeter direct forcing

Status: **marker projection and conservative exchange verified; impermeability
failed; physical accuracy unqualified**. This isolated investigation does not
replace P1's bonded disk solver. Existing P1 sources and evidence are preserved.

## Algorithm and ownership

`perimeter_boundary_2d.py` constructs circle perimeter markers with spacing
1.5 fluid cells. It solves `(J M^-1 J^T + c I) impulse = v_wall - J u`,
where `M` is the actual fluid cell mass including current density and thickness.
The opposite impulse is the prescribed body's actuator reaction. SI `c` is an
inverse-mass compliance, not a calibrated physical material coefficient.
`c=0` is direct forcing, and positive `c` gives a finite implicit impedance.
The same bilinear `J` and `J^T` ensure force, moment and virtual work conservation.
No disk center drag is added in these experiments. No DEM damage is evolved.

The resulting grid impulse enters the actual float64 D2Q9 populations through
a zero-mass source. Real BGK collision and periodic streaming follow. Both
projection-time and post-stream velocity are measured; no diagnostic overwrites
or analytical velocities substitute for the evolved fluid field.

## Actual experiment

A 1 m square periodic single-phase domain, 1000 kg/m³ density, artificial
viscosity 0.02 m²/s, thickness 0.2 m, cylinder radius 0.125 m, initial liquid
velocity 0.02 m/s. The body is either fixed or translates at 0.01 m/s. All eight
cases run for 0.025 s. Grids are 32² and 64²; dt is 0.0005 and 0.000125 s,
respectively, to keep BGK tau fixed. This therefore measures combined fluid
space/time refinement, not an independently isolated spatial convergence study.
Compliance is 0 or 0.1 kg⁻¹. An independent perimeter probe uses spacing 0.375 dx.

The declared no-slip and local normal leakage gate is 0.0002 m/s (1% of initial
liquid speed), applied separately at each measurement stage. This is a local
normal-velocity diagnostic, not net enclosed-volume flux or fluid exclusion.

| Measurement (max over real histories) | Result |
|---|---|
| Zero-compliance marker slip immediately after projection | ≤ 5.66e−18 m/s, pass |
| Fixed-body off-marker normal slip, coarse → fine, zero compliance | 0.007128 → 0.004899 m/s, fail |
| Moving-body off-marker normal slip, coarse → fine, zero compliance | 0.003563 → 0.002449 m/s, fail |
| Post-stream marker slip, all cases | 0.00923–0.01942 m/s, fail |
| Compliance 0.1 marker slip | 0.000119–0.001131 m/s; only fine moving case passes |
| Linear momentum balance | ≤ 1.54e−11 kg m/s, pass |
| Relative mass drift | ≤ 2.10e−14, pass |
| Impulse / moment / virtual-work mapping error | ≤ 2.78e−17 N s / 6.94e−18 N m s / 2.17e−19 J |

Two grids and two compliances do not establish asymptotic convergence. Changing
compliance changes the boundary model. The exact marker projection does not
prevent velocity leakage between markers or after streaming. Interior fluid
remains present and contributes fictitious interior mass and reactions. Drag,
pressure traction, lift, impermeability, free surface, and physical icebreaking
are **not qualified**. Eight short transient cases are not steady drag cases.

## Reproduction and audit

Use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src` and:

```sh
python examples/perimeter_boundary/run.py
python examples/perimeter_boundary/audit.py
python -m pytest -q tests/test_perimeter_boundary_2d.py
```

Eight tests passed. Each committed raw case includes full final populations,
before/after projection velocity, cell masses, perimeter coordinates, impulses,
and every history entry. `study.json` binds cases and runtime sources by SHA256.
The audit reconstructs final projection, independent off-marker probes,
post-stream slip, momentum and reported history maxima; it also checks hashes.
It does not claim independent reconstruction of every historical timestep.

## Next implementation gate

Use link-based moving solid boundary / explicit fluid exclusion with a complete
mass and momentum ledger, then verify stationary and moving boundaries at the
end of the entire LBM step. Repeat independently varied grid, timestep, marker
spacing (if retained), domain and coefficient checks. Only after that gate should
this boundary replace P1's point coupling in the wet fracture benchmark.

## Implemented fixed-boundary correction

`fixed_disk_boundary_2d.bounce_step` provides a second, link-based route:
stationary solid-mask nodes retain **zero populations**, so no interior liquid
mass is evolved. Outgoing fluid populations that encounter a solid link are
returned through the opposite direction at the same fluid node (half-way
bounce-back). The body impulse is the sum of `2 f_out c` on every reflected
link, with the physical mass and velocity scales restored by the fixture.
The wall is represented by grid links and a staircase mask; it is not the
exact analytic circle. There is no moving-mask node creation or deletion.

Actual 32² and 64² fixtures run for the same 0.025 s at the same time/grid
scaling as above. Both pass zero interior populations, mass conservation and
fluid-plus-solid-impulse balance. Max mass error is 2.06e−14; max momentum
residual is 1.57e−11 kg m/s. The two grids' x impulses are 0.278232 and 0.403290 N s, a **44.95%**
change relative to the coarse result: force resolution convergence fails.
This is a sensitivity measurement, not evidence of converged cylinder drag. Rest-state and reaction
sign tests pass. The link construction blocks population transfer into solid
nodes, but does not certify off-grid circular skin geometry or physical drag.

```sh
python examples/perimeter_boundary/fixed_link.py
python examples/perimeter_boundary/audit_fixed_link.py
python -m pytest -q tests/test_perimeter_boundary_2d.py tests/test_fixed_disk_boundary_2d.py
```

Ten tests pass. Raw initial and final populations, the entire final pre-step
field, masks, reactions and histories are committed. The separate fixed-link
manifest binds them to the implemented sources. The audit reconstructs the
last complete BGK/bounce-back step bitwise and independently verifies fluid
mass, excluded interior and final total momentum balance. This fixed-boundary
route is available for subsequent P1 integration; **moving and breaking ice
remain unimplemented**, and the failing perimeter projection fixtures remain
as regression evidence rather than being relabelled as passing results.
