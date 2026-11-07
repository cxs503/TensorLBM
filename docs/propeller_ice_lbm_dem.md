# LBM-DEM Coupling for Propeller-Ice Interaction

## Mission m5529: Ice Zone Propulsion Performance Degradation Prediction

This module implements a coupled Lattice Boltzmann Method (LBM) and Discrete Element Method (DEM) framework for simulating propeller-ice interactions in ice-infested waters.

## Physical Background

Marine propellers operating in ice-covered waters face significant challenges:
- **Ice particle ingestion**: Ice particles are drawn into the propeller disk, causing impact loads
- **Thrust fluctuation**: Ice- propeller interaction causes unsteady thrust and torque
- **Performance degradation**: Ice accumulation and impact reduce propeller efficiency
- **Structural loads**: Impact forces can damage blades and shaft system

## Implementation

### Architecture

```
TensorLBM/
├── src/tensorlbm/
│   ├── dem/                      # DEM solver for ice particles
│   │   ├── __init__.py
│   │   ├── particle.py           # Particle dynamics (translation + rotation)
│   │   ├── contact.py            # Contact mechanics (linear spring-dashpot)
│   │   └── dem_solver.py         # DEM time integration
│   ├── ice_particle_ibm.py       # LBM-DEM coupling via IBM
│   └── propeller_ice_benchmark.py # Full benchmark
└── tests/
    ├── test_dem_particle.py      # Particle dynamics tests
    ├── test_dem_solver.py        # DEM solver tests
    ├── test_ice_particle_ibm.py  # Coupling tests
    └── test_propeller_ice_benchmark.py  # Benchmark tests
```

### Key Components

#### 1. DEM Solver (`dem/`)

Implements Newton's equations for spherical ice particles:

**Translation:**
```
m * dv/dt = F_gravity + F_contact + F_drag + F_fluid
```

**Rotation:**
```
I * dω/dt = T_contact + T_fluid
```

**Contact Model:** Linear spring-dashpot (Cundall & Strack 1979)
- Normal force: `F_n = k_n * δ - η_n * v_n`
- Tangential force: Coulomb friction limited

**Drag Models:**
- Stokes drag (Re << 1): `F_D = 6πμRv_rel`
- Schiller-Naumann (Re < 1000): `C_D = (24/Re)(1 + 0.15*Re^0.687)`

#### 2. LBM-DEM Coupling (`ice_particle_ibm.py`)

Uses Immersed Boundary Method (IBM) for fluid-particle interaction:

1. **Velocity interpolation**: Map Eulerian fluid velocity to Lagrangian particle surface markers
2. **Force computation**: Compute drag force from velocity difference
3. **Force spreading**: Spread reaction force back to fluid grid using IBM
4. **LBM forcing**: Apply body force to LBM distribution function via Guo scheme

**IBM Kernel:** 4-point Peskin kernel for smooth force distribution

#### 3. Propeller Model (`propeller_ice_benchmark.py`)

Actuator disk model for propeller (simplified approach):
- Thrust coefficient K_T from KP505 open-water data
- Torque coefficient K_Q from KP505 data
- Advance ratio J = U_in / (n * D)
- Body forces applied in disk region

### Verification Tests

#### 1. Single Particle Drag (Stokes Regime)

**Analytical solution:** `F_D = 6πμRv_rel`

**Test:** `test_drag_force_stokes_regime()`
- Particle Reynolds number Re_p < 1
- Error < 5% vs analytical

#### 2. Particle Settling Velocity

**Analytical solution:** `v_t = (ρ_p - ρ_f) * g * D² / (18μ)`

**Test:** `test_particle_settling_velocity_stokes()`
- Terminal velocity when gravity balances drag
- Error < 15% vs analytical (first-order time integration)

#### 3. Schiller-Naumann Drag

**Correlation:** `C_D = (24/Re)(1 + 0.15*Re^0.687)` for Re < 1000

**Test:** `test_drag_force_schiller_naumann()`
- Error < 5% vs correlation

### Benchmark Configuration

```python
PropellerIceConfig(
    # Domain
    nx=200, ny=100, nz=100,      # Grid size
    dx=0.01,                      # 1 cm grid spacing
    dt=1e-4,                      # Time step
    
    # Propeller
    propeller_diameter=48.0,      # 48 lu ≈ 0.48 m
    propeller_rpm=10.0,           # 10 rev/s
    propeller_position=(60, 50, 50),
    
    # Ice particles
    n_particles=10,               # Number of particles
    particle_radius_range=(0.05, 0.15),  # 5-15 cm
    particle_concentration=0.05,  # 5% volume fraction
    
    # Flow
    inflow_velocity=2.0,          # 2 m/s
    fluid_density=1025.0,         # Seawater
    fluid_viscosity=1.5e-6,       # m²/s
    
    # Simulation
    n_steps=5000,
    output_interval=100,
)
```

## Usage

### Basic Example

```python
from tensorlbm.propeller_ice_benchmark import (
    PropellerIceBenchmark,
    PropellerIceConfig,
)

# Configure benchmark
config = PropellerIceConfig(
    nx=100, ny=50, nz=50,
    n_particles=5,
    n_steps=1000,
)

# Run benchmark
benchmark = PropellerIceBenchmark(config)
results = benchmark.run()

# Access results
print(f"Mean thrust: {results['thrust_mean']:.2f} N")
print(f"Mean torque: {results['torque_mean']:.3f} N·m")
```

### DEM-Only Simulation

```python
from tensorlbm.dem import DEMSolver, DEMConfig, IceParticle
import torch

# Create particles
particles = [
    IceParticle(
        position=torch.tensor([0.0, 0.0, 1.0]),
        velocity=torch.zeros(3),
        angular_velocity=torch.zeros(3),
        radius=0.05,
    )
]

# Configure solver
config = DEMConfig(
    time_step=1e-4,
    gravity=torch.tensor([0.0, 0.0, -9.81]),
    drag_model="schiller_naumann",
)

# Run simulation
solver = DEMSolver(config)
solver.initialize(particles)
solver.run(n_steps=1000)
```

## Acceptance Criteria

### Verification Tests (All Must Pass)

1. ✅ Single particle drag coefficient vs Clift et al. (1978): error < 5%
2. ✅ Particle settling velocity vs Stokes correction: error < 15%
3. ✅ Schiller-Naumann drag correlation: error < 5%
4. ✅ Contact detection and force computation
5. ✅ Energy conservation in elastic collisions

### Benchmark Tests

1. ✅ Propeller thrust and torque computation
2. ✅ Particle ingestion tracking
3. ✅ Thrust fluctuation analysis
4. ✅ Parameter sweep capability

## Limitations and Future Work

### Current Limitations

1. **Simplified propeller model**: Actuator disk does not resolve blade geometry
2. **Spherical particles**: Real ice particles have irregular shapes
3. **No ice breaking**: Particles are rigid, no fragmentation model
4. **No free surface**: Single-phase flow only
5. **CPU-only**: GPU acceleration not yet implemented

### Future Enhancements

1. **Resolved blade geometry**: Use IBM with moving boundaries for actual blades
2. **Non-spherical particles**: Superquadric or polyhedral particle shapes
3. **Ice breaking model**: Bonded-particle DEM for ice fragmentation
4. **Free surface**: Coupling with free-surface LBM for wave effects
5. **GPU acceleration**: Port DEM and coupling to CUDA
6. **Full-scale validation**: Compare with model tests (e.g., Aker Arctic, HSVA)

## References

1. Cundall, P.A. & Strack, O.D.L. (1979). A discrete numerical model for granular assemblies. *Géotechnique* 29(4), 47-65.
2. Tsuji, Y. et al. (1993). Lagrangian numerical simulation of plug flow of cohesionless particles in a horizontal pipe. *Powder Technology* 71, 239-250.
3. Peskin, C.S. (1972). Flow patterns around heart valves: a numerical method. *J. Comput. Phys.* 10(2), 252-271.
4. Uhlmann, M. (2005). An immersed boundary method with direct forcing for the simulation of particulate flows. *J. Comput. Phys.* 209(2), 448-476.
5. Clift, R., Grace, J.R., & Weber, M.E. (1978). *Bubbles, Drops, and Particles*. Academic Press.
6. ITTC (2014). Recommended Procedures: Open Water Test. 7.5-02-03-02.1.

## Authors

- TensorLBM Development Team
- Mission m5529: Ice Zone Propulsion Performance Degradation Prediction

## License

Same as TensorLBM (see LICENSE file)
