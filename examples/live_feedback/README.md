# Live fluid / rigid translation fixture

`MovingDiskFeedback` in `tensorlbm.feedback_moving_disk` receives the current SI
position, time, and held velocity on every call. It executes actual D2Q9 BGK and
streaming using the existing experimental moving disk. A current position or
clock mismatch is rejected before stepping. The body position after the step is
returned explicitly; the next body velocity may differ from the held velocity.

`impulse_on_body_Ns` is the sum of wall reflection and node conversion impulses.
`reservoir_impulse_on_fluid_Ns` is an external global density correction and must
not also be added to the body. DEM contact is excluded from this fluid interface.
The body work `J dot held_velocity` differs from the body kinetic energy increment
by the explicit velocity kick defect; both are published independently.

The 160-step actual free translation fixture uses a 5 kg body, initial velocity
(0, 0.03) m/s, 0.04 m radius, 0.01 m lattice spacing, 0.001 s timestep,
1000 kg/m3 liquid density and 0.2 m extrusion thickness. Two covered node events
occur. Maximum total momentum closure error is 1.05e-13 N s after subtracting the
external reservoir impulse. Midway restart reproduces each subsequent exchange
and all populations bitwise. Total macroscopic kinetic energy changes by
-0.0015089 J; this is an exposed numerical result, not an energy certification.

The boundary is a halfway voxel disk with a global mass reservoir. This fixture
has no locally conservative refill, calibrated drag, curved-grid convergence,
gravity, free surface, ice, contact or deformation. It verifies live motion
feedback plumbing, not the physical accuracy of wet icebreaking.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/live_feedback/run.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/live_feedback/audit.py
PYTHONPATH=src python -m pytest -q tests/test_feedback_moving_disk.py
```

Complete population checkpoint, per-step exchanges, body velocity, external
reservoir, kinetic ledger and source SHA256 hashes are in `evidence.json`.
