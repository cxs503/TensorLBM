# Flat-plate wall-model validation

The finite flat-plate benchmark isolates skin friction from the SUBOFF form
drag problem.  A one-cell-thick plate is placed inside an external-flow domain
so both wetted faces see the same free stream.  The production path uses:

- D3Q19 cumulant collision;
- physical viscosity in the wall law and a separately declared resolved
  collision Reynolds number;
- BFL impermeability with tangential slip;
- Guo wall traction with the `tau_w*A/V` force contract;
- optional wall-normal exchange-location sampling for grid-consistent stress
  input, without overwriting or assimilating populations;
- incoming-only non-equilibrium far-field reconstruction;
- outlet/transverse equilibrium-difference sponges, excluding the inlet; and
- an enclosing kinetic control-volume force observer.

The primary quantity is the two-sided integrated friction coefficient

`Cf = F_friction / (0.5 rho U^2 A_wet)`.

For a refinement sequence, hold the physical exchange height fixed.  Because
the CLI value is in lattice cells, this means scaling
`--stress-exchange-distance` with the plate resolution (for example 3 at
L=256 and 6 at L=512).  The sampler reconstructs the velocity by trilinear
interpolation at that wall-normal height; the wall law uses the same physical
distance, while the Guo source is still applied at boundary control volumes.
Its integrated fluid momentum change is unit-tested to equal the negative of
the reported wall traction.  A value of zero retains the legacy first-cell
input and is useful only as a declared sensitivity case.

The result records the minimum, time-mean and maximum exchange-location
`y+`, plus the largest fraction of requested boundary samples rejected for
leaving the domain or touching a solid interpolation value.  More than 1%
rejected samples fails the benchmark admission gate.  These fields are
applicability evidence: a log-law run must not be promoted merely because its
integrated force matches ITTC while its sampled `y+` lies outside the law's
valid region.
GPU reductions for this ledger are sampled every 50 steps by default
(`--wall-diagnostic-interval`); the cadence is checkpointed, and a BFL run
with no collected applicability samples fails closed.

ITTC-1957 is recorded as an engineering correlation, not an exact Navier–
Stokes solution.  Acceptance requires a stationary Cf history at multiple
plate resolutions and collision Reynolds sensitivities.  The control-volume
total force and BFL-link-plus-wall-stress total are reported separately; the
one-cell leading/trailing edges contribute form drag and are not included in
the friction coefficient.

Run the production-sized candidate with:

```bash
PYTHONPATH=src python examples/flat_plate_wall_model_validate.py \
  --device cuda:0 --nx 512 --ny 128 --nz 3 --plate-length 256 \
  --reynolds 1e6 --resolved-reynolds 1e5 \
  --stress-exchange-distance 3 \
  --steps 12000 --warmup-steps 6000 \
  --output results/flat-plate-re1e6-l256.json
```

No SUBOFF result is admitted by matching total resistance while this benchmark
shows a material, unresolved wall-friction bias.

## Admitted single-grid exchange candidate

The L256, collision-`Re=2e4`, Musker, exchange=3 run completed 30,000 steps
with a 15,000-step measurement window.  It produced `Cf=0.00459729` (1.924%
from ITTC), 0.283% eight-block range, 0.028% trend, 0.0286% independent-force
difference, zero positivity limiting and zero rejected exchange samples.
Mean exchange `y+` was 563 (range 533–808).  It passes the fail-closed
single-grid schema.  It is not yet a grid-converged wall-model validation;
the declared L256/L384/L512 sequence holds `exchange_distance/plate_length`
fixed at `3/256`.

## Multi-grid provenance gate

Checkpoint/result schema v3 records every variable needed to establish an
equivalent refinement sequence: domain shape, plate placement, lattice speed,
startup ramp, sponge width/strength, control-volume margin, collision
viscosity/LES constant, positivity policy, wall law, and diagnostic cadence.
Changing any of these invalidates restart identity.

`assess_flat_plate_convergence` accepts only three or more individually
admitted v3 records.  It verifies equal physics/numerics, invariant domain
proportions and invariant `exchange_distance/plate_length` before fitting the
observed order and extrapolated `Cf`.  Earlier v2 campaign files remain useful
development evidence but fail this formal provenance gate; they are never
silently promoted to a v3 grid-convergence claim.

The completed v2 development sequence is nevertheless numerically coherent:
L256/L384/L512 give `Cf=0.004597289`, `0.004657755`, and `0.004668967`.
The sequence is monotonic; a three-point fit gives observed order 3.739,
`Cf_infinity=0.004674770`, and 0.124% finest-grid discretisation distance.
The extrapolated value is about 0.272% below ITTC.  The stored evidence labels
this only a mathematical fit (`physical_validation=false`); the active v3
reruns must reproduce it before the case-specific convergence gate can pass.
