# P3 stationary curved boundary: conservative experimental interpolation

This extension preserves all older P1/P2 implementations and artifacts. It does
not enter the moving DEM coupling. Only a stationary circle in periodic liquid
is solved here; no free surface, ice fracture, buoyancy, or moving topology is
qualified.

## Method and accounting

The circle intersects each outgoing fluid-to-solid lattice link analytically at
fraction q. Using post-collision populations at fluid node x and its rear fluid
neighbour x-c, the reflected distribution is:

- q < 1/2: `r = 2q f_i*(x) + (1-2q) f_i*(x-c)`.
- q >= 1/2: `r = f_i*(x)/(2q) + (2q-1) f_opp*(x)/(2q)`.

This is the stationary interpolation form associated with Bouzidi, Firdaouss
and Lallemand, [Momentum transfer of a Boltzmann-lattice fluid on a curved
boundary (2001)](https://doi.org/10.1063/1.1399290). The DOI resolver was verified;
the publisher's full text was blocked by an HTTP 403 challenge during this task.
The existing repository BFL module was also inspected, but is not reused and its
`1/q` force convention is not adopted: such a factor would break the population
momentum budget in this solver.

Interpolation replaces an outgoing population o by r and therefore changes
mass by r-o. Here **the same fluid node's rest population f0 receives o-r**.
This has zero momentum. Multiple link corrections are added at each node.
Thus the solid reaction is `(o+r)c_i`, and global fluid-plus-solid momentum and
fluid mass can be independently checked against actual populations. No global
rescaling or unexplained sink is applied. Correction net and L1 amounts are
published every step. Solid interior populations remain exactly zero.

The rest correction is a new experimental modification. It can alter pressure
and can make populations negative for other parameters; this study monitors
positivity, not a general positivity theorem. Classical BFL second-order
accuracy is **not asserted for this modified scheme**. Geometry is analytic but
the excluded fluid volume is still a center-based solid mask, so area resolution
remains a source of error.

## Actual benchmark

Periodic 1 m x 1 m domain; stationary circle center (0.503, 0.497) m, radius
0.125 m; thickness 0.2 m; density 1000 kg/m3; deliberately viscous nu=0.02 m2/s;
initial fluid velocity (0.02, 0) m/s. These are a numerical fixture, not seawater.
The initial uniform moving liquid is incompatible with the stationary no-slip
wall and creates a startup acoustic transient. Duration is 0.15 s. No inlet,
external body force or artificial steady-state test is used.

Spatial grids **32, 48, 64** hold physical dt=0.0005 s fixed (300 steps), unlike
the previous diffusion-scaled dt study. The finest grid additionally uses
0.00025 s (600 steps). All three grids also run identical half-way staircase
bounce-back as a baseline. Actual JSON contains initial, before-last, final
populations and the full per-step history.

| Scheme | n | dt (s) | Integrated x reaction (N s) |
|---|---:|---:|---:|
| Curved interpolation + rest repair | 32 | 0.0005 | 0.5661801314 |
| Curved interpolation + rest repair | 48 | 0.0005 | 0.5720780396 |
| Curved interpolation + rest repair | 64 | 0.0005 | 0.6506211023 |
| Curved interpolation + rest repair | 64 | 0.00025 | 0.5453197590 |
| Staircase | 32 | 0.0005 | 0.5645491661 |
| Staircase | 48 | 0.0005 | 0.5827725782 |
| Staircase | 64 | 0.0005 | 0.6608285930 |

Maximum relative fluid mass error is **5.81e-14**; maximum momentum residual is
**2.38e-11 kg m/s**. Minimum actual fluid population is **0.0276709**, and all
solid populations are zero. These accounting checks pass.

Changing dt in this isothermal LBM also changes the SI sound speed
`c_s = dx/(sqrt(3) dt)` and thus the equation of state/compressibility. The dt
comparison is a coupled time/EOS sensitivity, not an isolated time truncation
error.

Curved interpolation's last spatial pair changes **13.7294%**, while finest
timestep halving changes **16.1847%**. **Load convergence fails.** The time
sensitivity is sufficiently large that a spatial convergence order cannot be
inferred from these results. This extension does not resolve the previous
44.95% fixed-disk load sensitivity or wet DEM sensitivity. Neither failure is
removed or renamed a pass.

Next: suppress initial acoustic incompatibility with a smoothly ramped drive
or independently solve a developed viscous reference; cross timestep refinement
on each grid before inferring spatial order; quantify mask fluid-volume errors
and verify wall slip/traction against an independent analytic or high-resolution
reference. Moving topology remains a separate conservation experiment.

## Reproduce and audit

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/curved_boundary/run.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/curved_boundary/audit.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_curved_boundary_2d.py
```

`docs/assets/curved-boundary/study.json` records source and raw-artifact SHA256.
The independent audit reconstructs mass and momentum from final distributions,
replays the real last collision/stream/boundary step exactly, verifies excluded
interior and positivity, and checks published sensitivity numbers. Twelve tests
cover both interpolation branches, exact intersections, randomized force/mass
budgets, resting equilibrium, invalid geometry and invalid populations.

## Follow-up repair experiment: smooth forcing and a longer window

The original impulsive-start failure above is retained. A second experiment
starts at rest, ramps a spatially uniform fluid acceleration smoothly with
`a(t)=0.02 sin²(pi min(t/0.25,1)/2)` m/s², then holds it constant. A D2Q9 source
`delta f_i=3 w_i c_ix a dt²/dx` is added only in fluid nodes: it has zero mass
and the prescribed momentum increment. This numerical population source is
explicitly accounted as external drive impulse; it is not a certified
second-order body-force scheme. Actual simulation duration is 2 s. The same
three spatial grids and finest dt/2 comparison are run, starting from zero
velocity and keeping old code and evidence intact.

| n | dt (s) | Mean x reaction, 1.75–2 s (N) | Adjacent-window drift |
|---:|---:|---:|---:|
| 32 | 0.0005 | 1.806992570 | 7.8904% |
| 48 | 0.0005 | 1.789154480 | 7.9703% |
| 64 | 0.0005 | 1.781233823 | 8.0027% |
| 64 | 0.00025 | 1.782983182 | 7.9958% |

The last spatial pair's late mean load difference improves to **0.442704%**,
and the coupled dt/EOS sensitivity to **0.098211%**. Maximum relative mass error
is **7.905e-13**, momentum accounting residual **3.345e-10 kg m/s**. Populations
stay positive (minimum 0.0276701). The longer smooth-start experiment removes
much of the original sensitivity at this observation window.

**All four runs fail the declared 1% steady drift gate**, comparing mean forces
in 1.5–1.75 s and 1.75–2 s. These are reproducible transient-window sensitivities,
not converged steady drag or certified physical accuracy. A single spatial pair
below 3% does not establish asymptotic order. Changing dt still changes physical
sound speed and the explicit force integration. Rest-mass repair and center-mask
volume remain unqualified contributors to spatial error.

`driven-study.json` and four new raw JSON files record the complete experiment.
The independent audit reconstructs total applied external impulse directly from
SI acceleration and fluid-node mass, reconstructs final momentum, replays the
last drive/collision/boundary step, and recomputes both late-window averages and
the failed steady gate.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/curved_boundary/driven.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python examples/curved_boundary/audit_driven.py
```
