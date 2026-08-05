# D3Q19 CH adapter stream/boundary and flux contract R1

## Scope and production audit

`tensorlbm.multiphase3d.free_energy_step_3d` is the existing D3Q19
free-energy **collision** operation.  It returns post-collision `(f, g)` and
has no population streaming or boundary operation.  Its chemical-potential and
Korteweg-force finite differences explicitly use the existing `periodic`
operator policy.  R1 does not modify that production collision code.

The authoritative existing D3Q19 streaming convention is
`tensorlbm.solver3d.stream3d`: for a population `q` with `C[q] = (cx, cy, cz)`,
on storage `(19, z, y, x)`, pull streaming is

```python
out[q] = roll(inp[q], shifts=(cz, cy, cx), dims=(z, y, x))
```

R1 reproduces that convention only for its `periodic` adapter stream.

## R1 adapter API

`tensorlbm.phasefield.stream_boundary_contract` provides:

* `stream_d3q19_adapter(distribution, *, boundary)` for a floating point
  `(19, nz, ny, nx)` distribution;
* `PhaseBoundaryContract(boundary=...)`, an explicit reporting object.

Supported boundary policies are exactly:

* `periodic`: D3Q19 pull streaming with the authoritative shift convention;
* `no_flux`: link-wise exterior reflection.  A population whose push
  destination exits the domain is placed at its original cell in its D3Q19
  opposite direction.  It never wraps to the opposite face and total
  distribution inventory is retained by this streaming action.

There is no wetting/contact-angle model.  Unknown policies, including
`wetting`, fail closed with `ValueError`.

## Fail-closed reporting status

The adapter declares:

* `stage = "collision_then_adapter_stream"`;
* `physical = False`;
* `phase_flux_status = "withheld"` and `phase_flux = None`.

The adapter does not assign a continuum phase flux from population reflection
and does not upgrade the collision-only production CH path to a complete
physical simulation.  A future owner must define, implement, and validate a
boundary-flux observable before changing the withheld status.

## Verification

`tests/test_phasefield_stream_boundary_contract.py` verifies all 19 periodic
shifts against `tensorlbm.d3q19.C`, constant-field invariance, no-flux
no-wrap/link reflection and inventory retention, and boundary-policy rejection.
