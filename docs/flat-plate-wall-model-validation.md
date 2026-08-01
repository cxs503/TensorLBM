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
