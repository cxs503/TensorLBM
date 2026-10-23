# Equilibrium-reference local moving boundary: diagnostic attempt

This independent prototype preserves the previous failed source and artifacts.
It is **not an accepted replacement** for the fluid–ice boundary.

## Two changes with discrete justification

1. Moving bounce-back adds a population correction `6 wi rho ci·u_wall`.
   The old implementation repaired each local correction as if it were a
   local excess of total fluid mass. Even homogeneous co-moving equilibrium
   has nonzero per-link corrections, although their closed-mask total is zero.
   That repair therefore destroyed equilibrium. The new repair subtracts the
   density-one equilibrium reference, correcting only
   `6 wi (rho−1) ci·u_wall` in the same cell's rest population. The sum of
   reference fluxes is zero on any closed periodic mask: opposite directed
   links occur in equal numbers. This conserves total mass and leaves the
   homogeneous co-moving bounce step unchanged. The initial density is one in
   this solver; no measured-force subtraction or common-speed special case is
   used. Insufficient positive local rest capacity fails atomically.
2. Simultaneously covered/exposed nodes within radius 4 move all nine actual
   populations directly, preserving mass and linear momentum. Remaining
   covered populations spread equally over bounded surviving liquid neighbors.
   Exposed nodes borrow actual populations proportionally from bounded donors,
   without re-equilibrating to the wall velocity. Thus the conversion itself
   preserves momentum, while density redistribution still perturbs pressure.

## Actual findings and failed acceptance

The no-conversion homogeneous moving equilibrium is now preserved to floating
point roundoff. The old module fails that same test. Direct population
conversion preserves linear momentum without imposing a force correction.

**Crossing-node co-moving flow still fails:** integer masks change the liquid
node count; a constant-density state and fixed total liquid mass cannot both
be preserved when that count changes. Bounded mass redistribution introduces
pressure gradients and artificial body impulse. The six-case raw study
includes ordinary crossing, co-moving crossing, wrap, an actually paired
small disk, and both moving/co-moving spatial refinement. Its 3% convergence
failure is retained. The ordinary moving-case impulse refinement difference is **4.0551%**,
compared with the old 7.6302%, but still fails 3%. Cross-grid co-moving
impulse is **−2.33626** lattice units (old **+0.41129**): this metric is worse.
The spatially refined co-moving impulse, converted to the coarse physical
scale, is **−2.66601**, so the artifact is not removed by refinement. Numerical
conservation is not sufficient evidence of physical improvement.

This scheme is not exact swept-volume integration. It does not qualify
angular momentum, interface energy or no-slip traction. A volume-fraction or
geometric swept-volume formulation is required before claiming that moving
co-equilibrium or continuum pressure response is solved. No global mass
rescaling, hidden reservoir, or manually reduced body impulse is present.

## Reproduce

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/balanced_local_moving_boundary/run.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/balanced_local_moving_boundary/audit.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_balanced_local_moving_boundary.py
```

[Actual study](assets/balanced-local-moving-boundary/study.json) binds raw
initial/final populations to module SHA256s, exact continuation restarts,
per-step wall/conversion momentum ledgers and independent raw-field audits.
The baseline grid is 40×32, disk radius 4, speed .02, tau .8 and 120 steps.
Diffusive refinement doubles dimensions/radius, halves lattice speed and
quadruples step count at unchanged tau, fixing physical domain, viscosity,
body speed and duration; numerical neighborhood remains four lattice cells.
