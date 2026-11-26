# Fixed and insulated walls for D2Q5 diffusion

After scalar BGK collision, `scalar_stream_slab(post, west, east)` imposes
halfway walls with periodic transverse streaming. A wall may supply a scalar
or one value per transverse row. None gives zero normal transfer. Exchange
is incoming minus outgoing population per wall link. For lattice spacing
dx, unit heat capacity and unit thickness, multiply by dx² for heat and by
dx/dt for flux per wall area. Interior link transfers follow the same scaling
and use increasing-coordinate signs.

Run `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/oscillating_scalar_slab.py`
for a two-dimensional unit square with unit diffusivity, wet-wall spacing
dx/2, a 0.08 s driving period and equilibrium initial populations matching
the analytic cell-average temperature. West wall data are face averages of
cos(2πy) cos(ωt), evaluated at the temporal midpoint. With λ²=(2π)²+iω,
the exact complex field is cos(2πy) times sinh(λ(1−x))/sinh(λ) for a fixed
east wall or cosh(λ(1−x))/cosh(λ) for insulation. The flux comparison uses
face-averaged Fourier derivatives in both directions. All cell storage
changes are checked against signed neighbor and boundary transfers.

At 64 cells per direction the maximum temperature, x-flux and y-flux errors
over the two wall choices are 7.99e-4 K, 1.04e-2 W/m² and 6.01e-3 W/m².
The benchmark returns status 1: for each east wall, the first 8→16
refinement falls short of the factor-three reduction gate. Both subsequent
refinements pass and both finest runs meet the absolute targets. The
thermal attenuation length is about one cell on the coarsest grid. The
command reports all error reductions and keeps both failing rows.

### Discrete periodic response

`tensorlbm.scalar_harmonic.solve_harmonic_scalar_slab` solves the complex
frequency response of the BGK collision and halfway-wall streaming update.
Specify radians per lattice step and complex midpoint wall phasors. The
solver separates transverse Fourier components, solves the sparse linear
system along x and returns complex populations plus midpoint link transfers.
It runs on CPU with NumPy, SciPy and PyTorch; multiply transfers by dx/dt
for flux at unit capacity. For zero frequency with two insulated walls, pass a finite
`mean_temperature` to fix the spatial mean. Any volumetric source must
then sum to zero over the lattice. Both the streaming and harmonic operators
accept `wall_transfer=(west_in, east_in)` for specified inward transfers.
Use a scalar or one value per row and set that wall's temperature to `None`.
A physical inward heat flux density q corresponds to q·dt/(ρcp·dx) per link.
With two flux walls at zero frequency, the sum of both wall inputs and all
volume increments must vanish.

Run the oscillating slab command with `--periodic-start` to start the
time march from the real part of those populations. Every grid passes the
independent one-cycle time/frequency agreement check. On the fixed 64-cell
grid the maximum temperature and x-flux differences are 6.93e-15 K and
1.31e-13 W/m². The same two 8→16 continuum rate misses persist, so an
equilibrium-population startup transient does not explain them.

For grid-aligned material axes, `scalar_collide_diffusion` conserves the
scalar while independently relaxing the x and y antisymmetric moments.
The physical diffusivities are (τx−0.5)dx²/(3dt) and (τy−0.5)dx²/(3dt).
The even relaxation time is independent. The harmonic slab solver accepts
`tau_y` and `tau_even` as keywords and constructs its local operator from
the same collision update. Equal relaxation times recover stationary BGK.
