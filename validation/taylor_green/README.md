# Taylor-Green Vortex 2D Validation

## Overview

This validation suite tests the 2D Taylor-Green vortex (TGV) viscous decay against the analytical solution of the Navier-Stokes equations. The TGV is a canonical benchmark for incompressible flow solvers.

## Physical Background

The 2D Taylor-Green vortex has an exact analytical solution:

```
u(x,y,t) = -U₀ cos(kx) sin(ky) exp(-2νk²t)
v(x,y,t) =  U₀ sin(kx) cos(ky) exp(-2νk²t)
```

where:
- k = 2π/L is the wavenumber
- ν is the kinematic viscosity
- U₀ is the initial velocity amplitude

The volume-averaged kinetic energy decays exponentially:

```
E(t) = E(0) exp(-4νk²t)
```

## LBM Setup

- **Lattice**: D2Q9 with BGK collision operator
- **Boundaries**: Periodic in both x and y directions
- **Viscosity**: ν_lb = c_s²(τ - 0.5)Δt, where c_s² = 1/3
- **Initial condition**: TGV velocity field with U₀ << c_s (Ma < 0.1)

## Validation Criterion

The relative error in energy decay rate must be ≤ 8%:

```
|α_sim - α_ana| / α_ana ≤ 0.08
```

where α = 4νk² is the energy decay rate.

## Running the Validation

```bash
# Default parameters
python validation/taylor_green/tgv_validation.py

# Custom parameters
python validation/taylor_green/tgv_validation.py \
    --nx 64 \
    --tau 0.6 \
    --u0 0.05 \
    --n-steps 1000 \
    --output-dir validation/taylor_green/results
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--nx` | 32 | Grid resolution (square domain) |
| `--tau` | 0.6 | Relaxation time (> 0.5 for stability) |
| `--u0` | 0.05 | Initial velocity amplitude |
| `--n-steps` | 500 | Number of time steps |
| `--output-dir` | `results/` | Output directory |

## Outputs

- `results.json`: Complete simulation results and error metrics
- `energy_decay.png`: Energy decay comparison (simulation vs analytical)
- `energy_error.png`: Relative error over time
- `velocity_field.png`: Velocity magnitude snapshots at t=0 and t=t_max

## Expected Results

With default parameters (nx=32, τ=0.6, U₀=0.05, 500 steps):

- Decay rate error: ~0.005% (well within 8% criterion)
- Mean relative error: ~1.3%
- Maximum relative error: ~1.6%

## References

1. Taylor, G. I., & Green, A. E. (1937). Mechanism of the production of small eddies from large ones. Proceedings of the Royal Society of London. Series A-Mathematical and Physical Sciences, 158(895), 499-521.

2. Succi, S. (2001). The lattice Boltzmann equation for fluid dynamics and beyond. Oxford University Press.

## License

Part of TensorLBM validation suite.
