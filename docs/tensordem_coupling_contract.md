# TensorLBM ↔ TensorDEM load-exchange contract (2-D first)

This document defines the boundary between the two solvers. It is an interface
contract and verification plan, **not a claim that a coupled time integrator
already exists**.

## Canonical conventions

- Coordinates: global Cartesian x/y shared by both solvers; positive y is
  whatever the case configuration declares and must not be silently flipped.
- Units at the boundary: positions in m, physical time in s, traction in Pa,
  effective surface area in m², and forces in N. Lattice units stay inside
  TensorLBM and are converted exactly once at the solver boundary.
- Force direction: every exchanged load is labelled by its recipient. For
  example, `fluid_on_ice` is the force applied by water to ice and is passed
  into DEM; `ice_on_fluid` is the opposite contribution used by the fluid
  solver. Never pass a force on the fluid into DEM without reversing its sign.
- Time: every load packet carries a physical timestamp and the interval over
  which it was averaged. The consumer must reject stale or mismatched packets;
  no hidden force accumulation across steps is allowed.
- Ownership: TensorDEM owns particle state, contacts, and bond failure.
  TensorLBM owns fluid state and fluid traction reconstruction. The coupling
  driver owns interpolation, area/weight assignment, subcycling, and logs.

## Minimal 2-D exchange packet

A coupling driver should serialize the following logical fields (JSON is an
example, not a required runtime dependency):

```json
{
  "schema_version": 1,
  "time_s": 0.01,
  "interval_start_s": 0.009,
  "interval_end_s": 0.01,
  "frame": "global_xy",
  "force_target": "ice_particles",
  "units": {"position": "m", "traction": "Pa", "area": "m^2", "force": "N"},
  "particle_ids": [10, 11],
  "tractions_pa": [[0.0, -120.0], [4.0, -90.0]],
  "area_weights_m2": [0.002, 0.002]
}
```

Particle IDs must map to the current DEM particle ordering; never assume IDs are
stable across remeshing/repacking unless explicitly guaranteed. Missing particles
and broken/removed particles need a declared policy. Reject packets with
non-finite values, negative weights, mismatched array lengths, wrong dimensions,
or timestamps outside the driver's allowed synchronization tolerance.

## Current helper boundary

- `integrate_surface_traction(...)` computes a hull wrench from caller-supplied
  traction samples and surface quadrature weights.
- `integrate_particle_tractions_2d(...)` multiplies already-mapped per-particle
  traction by the supplied effective area and returns an (N, 2) tuple of forces
  in N, suitable for conversion to a PyTorch tensor for
  `IceDEM.step(external_forces=...)`.
- The particle helper does **not** map LBM cells to particles, calculate wall
  stress, derive effective areas, or conserve moment automatically. The mapping
  algorithm and weights must be validated by the coupling driver.

## Required conservation checks before coupled production runs

1. **Resultant force:** compare the sum of particle forces against the fluid-side
   integrated force on the ice for the same body and time interval.
2. **Moment:** compare moments about the same global origin; matching net force
   alone is insufficient when force application points differ.
3. **Action/reaction:** verify equal-and-opposite fluid/ice exchange for the
   same interface interaction, with an explicit tolerance.
4. **No double counting:** fluid load on the hull, DEM ice-contact load on the
   hull, and fluid load on ice are separate ledger entries.
5. **Time alignment:** log fluid step, DEM substep count, packet timestamp,
   averaging interval, and any interpolation/extrapolation.
6. **Zero-load and uniform-load tests:** zero traction must produce exactly
   zero external forces; a uniform traction with known areas must reproduce
   the analytical resultant.
7. **2-D thickness:** effective areas must include the same out-of-plane
   thickness convention as TensorDEM's mass/contact model.

The current DEM external-load API accepts an (N, 2) force tensor in N on every
step and does not cache loads. The next milestone is a driver that constructs
these packets from actual solver state, then passes the conservation tests
above. Only after that should a coupled indentation or prescribed-speed
icebreaking benchmark be treated as an end-to-end result.
