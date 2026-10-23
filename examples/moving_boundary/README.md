# Experimental moving excluded-fluid disk

This executable periodic D2Q9 BGK experiment moves the disk mask every step.
Fluid populations inside the disk are zero. Moving halfway bounce-back applies
the wall velocity; newly exposed nodes use adjacent fluid density and wall
velocity equilibrium. Disappearing nodes are removed explicitly.

Every step records wall momentum exchange, covered/exposed mass and momentum,
conversion impulse on the solid, and a **separate global mass reservoir**.
The reservoir rescales the remaining populations and changes their momentum.
It is a numerical correction, not evidence of local mass conservation or a
physically correct swept-volume model. Momentum closure includes its external
impulse. The numerical prescription follows the moving-wall reflection idea
in [Ladd's original work](https://doi.org/10.1017/S0022112094001771);
the exposed reservoir/refill policy is this experiment's implementation.

The uniform co-moving case crosses cell boundaries and retains uniform speed
to machine precision. Staircase area changes mean uniform density and exactly
zero solid force are not guaranteed. JSON restart retains every population,
configuration, geometry, clock and ledger; continuation is bitwise identical.

Eight actual cases publish full final populations and every step's ledger.
`moving`, `time-half`, `time-quarter`, `time-eighth` and `time-sixteenth`
have the same physical dx=0.01 m,
duration=1 s, radius=0.04 m, speed=0.03 m/s, density=1000 kg/m³,
extrusion thickness=0.01 m and viscosity=0.001 m²/s. Reducing dt changes both
lattice speed and tau to keep those physical parameters fixed. Independent
`motion-half`/`motion-quarter` cases keep tau unchanged and change Reynolds
number; they are motion sensitivity, not time convergence evidence.

At fixed grid, this also changes the SI lattice sound speed from 0.57735 m/s
to 9.23760 m/s (Mach 0.05196 to 0.00325). The study therefore measures
**combined time-step/compressibility sensitivity**, not isolated time error
for a fixed compressible equation of state.

The half-to-quarter step impulse change is approximately **5.83%**, exceeding
the declared **3%** threshold; quarter-to-eighth changes **3.58086%**, also
failing. Eighth-to-sixteenth changes **0.740271%**, passing the unchanged
3% threshold for the finest pair. The original failures remain in the audit.
The finest pair passing combined sensitivity does not qualify pure temporal
convergence or total physical accuracy.
Coarse node conversion creates approximately 44 lattice mass units of total
absolute reservoir exchange. Conservation bookkeeping does not remove this
local discretization error. Physical accuracy remains unqualified: no curved
interpolation, free surface, variable deformation, collision/contact or DEM
coupling is claimed.

Run with a PyTorch environment from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/moving_boundary/run_benchmark.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/moving_boundary/audit_evidence.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_moving_boundary_2d.py
```

The audit checks source/data hashes, independently reconstructs mass/momentum
and conversion ledgers, then replays real BGK steps to match the raw state.
