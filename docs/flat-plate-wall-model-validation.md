# Flat-plate wall-model validation

The finite flat-plate benchmark isolates skin friction from the SUBOFF form
drag problem.  A one-cell-thick plate is placed inside an external-flow domain
so both wetted faces see the same free stream.  The production path uses:

- D3Q19 cumulant collision;
- physical viscosity in the wall law and a separately declared resolved
  collision Reynolds number;
- BFL impermeability with tangential slip;
- Guo wall traction with the `tau_w*A/V` force contract;
- incoming-only non-equilibrium far-field reconstruction;
- outlet/transverse equilibrium-difference sponges, excluding the inlet; and
- an enclosing kinetic control-volume force observer.

The primary quantity is the two-sided integrated friction coefficient

`Cf = F_friction / (0.5 rho U^2 A_wet)`.

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
  --steps 12000 --warmup-steps 6000 \
  --output results/flat-plate-re1e6-l256.json
```

No SUBOFF result is admitted by matching total resistance while this benchmark
shows a material, unresolved wall-friction bias.
