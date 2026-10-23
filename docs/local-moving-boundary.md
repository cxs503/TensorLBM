# Local conservative moving-disk experiment

`LocalMovingDisk` is a separate float64 CPU D2Q9 prototype. The earlier
`moving_boundary_2d.py` and its published evidence remain unchanged.

## Algorithm and ownership

Each step uses the actual BGK collision and periodic moving-wall halfway
reflection. The wall reflection's per-cell mass increment is compensated by
adding/removing only rest populations in a periodic radius-4 neighborhood;
rest populations have zero momentum. Covered cells transfer all nine
populations to the nearest surviving fluid cell within that radius. Exposed
cells borrow proportional populations from surviving fluid donors within
that radius, then refill at the prescribed wall velocity. The resulting
change of fluid momentum is recorded as the conversion impulse on the body.
The wall impulse and conversion impulse are separate external actuator terms.
No whole-domain density rescaling or global reservoir exists.

The radius is an explicit numerical parameter, not a physical interaction
length. Donor capacity must suffice with positive populations. Missing
receivers, insufficient donor capacity, negative populations, or failed mass
conservation abort without changing the simulation state. Periodic distances
are used; transfers never cross more than the declared neighborhood radius.

## Limits and acceptance

This is integer node conversion, **not an exact geometric swept-volume
scheme**. Halfway bounce-back uses a staircase disk. Momentum conservation
here is discrete linear momentum bookkeeping; angular momentum and physical
traction have not been qualified. Finite-neighborhood density redistribution
introduces pressure artifacts. In particular, a common translating liquid
and disk should have negligible net hydrodynamic impulse; the published
co-moving case retains a nonzero artificial impulse and is not promoted as
physical validation.

Four full raw-population experiments cover crossing nodes, co-moving liquid,
periodic wrap, and diffusive grid/time refinement. The refinement halves dx,
quarters dt, halves lattice velocity, and keeps tau=.8: fixed physical
viscosity, domain, disk radius, speed, and duration. Conversion support remains
4 lattice cells and therefore changes physical width under refinement.
The 3% impulse convergence gate is fixed in advance; a failure is retained.
This is a numerical sensitivity study, not a qualified continuum result.

The raw artifacts contain initial/final distributions, complete per-step
ledgers, exact continuation restart checks, and both old/new module SHA256s.
The independent audit reconstructs mass and linear momentum directly from
populations and compares against accumulated wall/conversion impulses.
Physical qualification remains false regardless of conservation gates.

## Reproduce

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/local_moving_boundary/run.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/local_moving_boundary/audit.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_local_moving_boundary.py
```

Results: [study.json](assets/local-moving-boundary/study.json).
Next work: geometrically swept-volume local redistribution with uniform
co-moving-state preservation, then curved moving-link verification. The
current implementation deliberately exposes its failure before integrating
it as a physically accepted fluid–ice boundary.
