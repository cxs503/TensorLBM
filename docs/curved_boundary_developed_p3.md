# P3 long-window stationary circle experiment

This independent extension preserves the 0.15 s impulsive and 2 s smooth-drive
failures, their executable sources, and source hashes. It retains the same
stationary circle, viscosity, density, thickness, ramp, acceleration and dt.
No moving topology or DEM physics is added.

A 48-grid 10 s probe missed the unchanged 1% adjacent-window load drift gate:
**1.097437%**, with no earlier pass. Its raw record, source snapshot and separate
manifest remain in `docs/assets/curved-boundary/developed-probe.json`.
The complete follow-up uses **12 s** at n=32,48,64 with dt=.0005 s, plus n=64
at dt=.00025 s. All executed force samples are retained as a compact scalar
column; expanded diagnostic history is recorded every 100 steps. Every run
retains complete initial, before-last and final populations, source hashes,
cumulative impulses, extrema of every-step conservation/positivity checks,
and every non-overlapping .5 s mean force.

## Qualification

Two unchanged **1%** requirements are reported independently:

1. `window_drift_passed`: relative change between the last two .5 s mean
   reaction-force windows is below 1%. Its first passing time is published.
2. `net_acceleration_passed`: relative difference between mean applied drive
   force and mean solid reaction over the last .5 s is below 1%. Their difference
   is the actual mean rate of fluid linear momentum change.

`steady_qualified` requires both. It is never inferred from a slowly changing
force alone. The full velocity field and kinetic energy are not independently
certified as steady by these two aggregate criteria. No physical drag validation
or classical BFL accuracy certification is implied.

`qualify_developed.py` independently reconstructs exact applied drive impulse
from the SI acceleration ramp and actual fluid-node mass; reconstructs window
start/end momentum from cumulative forces; and checks end momentum against
stored raw populations. It does not change any simulation trajectory.
The source uses an explicit population force increment, not a certified
second-order body-force discretization. Changing dt also changes SI sound speed,
so timestep comparisons remain combined time/EOS sensitivities.

The mass repair remains local to f0, with no momentum increment. Center-mask
excluded volume errors remain. Neither moving disks nor wet DEM use this fixed
circle module. Earlier failed cases remain failed.

## Reproduce

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/curved_boundary/developed.py --duration 12
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/curved_boundary/qualify_developed.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/curved_boundary/audit_developed.py
```

The audit replays the final actual source/collision/stream/boundary step exactly,
checks actual final fluid mass/momentum, independently recomputes .5 s load
windows from every real force sample, checks first drift pass time, and
reconstructs the fluid-acceleration qualification. It preserves failed
qualification as evidence.

## Actual 12 s results

| n | dt (s) | Late mean reaction (N) | Load-window drift | Net acceleration / drive |
|---:|---:|---:|---:|---:|
| 32 | 0.0005 | 3.638928911 | 0.641766% | 4.650379% |
| 48 | 0.0005 | 3.624393725 | 0.649088% | 4.716988% |
| 64 | 0.0005 | 3.621572372 | 0.656260% | 4.788444% |
| 64 | 0.00025 | 3.622178817 | 0.654659% | 4.772500% |

The first passing **load-window drift** time is 10.5 s in all four runs.
Final 48→64 late-load sensitivity is **0.0778435%**, and finest dt/EOS
sensitivity **0.0167454%**. However, all four **fail the net acceleration gate**:
fluid still gains approximately 0.177–0.182 kg m/s each second. All final
`steady_qualified` values are therefore **false**. The force-window improvement
cannot be promoted to steady drag or full spatial convergence.

The largest recorded relative mass residual is **4.748e-12**, momentum residual
**2.007e-9 kg m/s** over 48,000 steps, and minimum fluid population **0.0274891**.
These extrema are accumulated at every step despite sparse expanded history.
Final raw replay, complete external impulse reconstruction, source/data hashes,
positivity, mass and momentum reconstructions all passed the independent audit.

The integrator's interim load-drift flag is only provisional. The required
qualification postprocess preserves its meaning under `window_drift_passed` and
publishes the two-gate `steady_qualified`; the final manifest binds both executed
simulation source and qualification source. Complete raw trajectories stay
unchanged during qualification. A longer run or steady nonlinear solve remains
future work; the current study deliberately stops at 12 s and retains failure.
