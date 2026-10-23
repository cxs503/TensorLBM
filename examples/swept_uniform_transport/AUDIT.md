# Independent uniform-state transport audit

Run from the TensorLBM repository, without `PYTHONPATH`:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/swept_uniform_transport/audit.py
```

The auditor imports NumPy and the Python standard library, **no TensorLBM
solver, geometry or transport module**. It checks source SHA256 bindings and
compressed raw artifact hashes, then reconstructs every recorded step:

- Analytic rectangle–cell intersection areas and fixed total fluid area.
- Shared space-time Cartesian apertures using independent two-point Gauss
  integration on edge-crossing intervals.
- Wall facets, normal closure, first position moment and local GCL.
- Actual Cartesian directional population fluxes from `Q_before`.
- Diffuse moving-wall incoming normalization and actual wall momentum flux.
- `Q_before − Cartesian flux − wall flux`, compared to recorded `Q_after`.
- Explicit zero-volume roundoff cleanup, population/mass/momentum balances,
  raw density/velocity and uniform-state kinetic/compressive energy.
- State continuity, published history/summary metrics, and qualification flags.

The recorded solver-side restart is bitwise, checked during generation.
This auditor verifies all recorded population transitions and checkpoint
fields; it does not separately execute the TensorLBM restart implementation.

All four raw cases pass. Maximum independently reconstructed population
update discrepancy is **1.67e−16 kg**, momentum balance residual
**1.18e−14 N·s**, density relative error **6.62e−14**, and velocity error
**4.76e−14 m/s**. Net body impulse is at roundoff and no compressive energy
surge occurs in these uniform co-moving cases. Tiny signed compressive
energy values around 1e−15 J are floating-point cancellation.

**Scope remains restricted to uniform co-moving equilibrium traces.**
These results do not validate arbitrary-state transport, BGK collision,
small-cell stability, viscosity, nonuniform hydrodynamic traction or ice
coupling. The auditor requires both `physical_accuracy_qualified=false` and
`arbitrary_state_transport=false`; it rejects inflated qualification flags.

Machine-readable results:
[`audit-report.json`](../../docs/assets/swept-uniform-transport/audit-report.json).
