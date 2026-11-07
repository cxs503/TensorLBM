#!/usr/bin/env python3
"""Demonstration script for LBM-DEM coupling and propeller-ice benchmark.

This script demonstrates:
1. Basic DEM simulation of ice particles
2. LBM-DEM coupling via IBM
3. Propeller-ice interaction benchmark
"""

import torch
from tensorlbm.dem import DEMSolver, DEMConfig, IceParticle
from tensorlbm.ice_particle_ibm import create_ice_particles
from tensorlbm.propeller_ice_benchmark import (
    PropellerIceBenchmark,
    PropellerIceConfig,
)


def demo_dem_settling():
    """Demonstrate particle settling in fluid."""
    print("\n" + "="*60)
    print("DEMO 1: Particle Settling in Fluid")
    print("="*60)
    
    # Create a single particle
    particle = IceParticle(
        position=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        velocity=torch.zeros(3, dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=0.01,  # 1 cm
        density=1100.0,  # Slightly denser than water
    )
    
    # Configure solver
    config = DEMConfig(
        time_step=1e-4,
        gravity=torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64),
        enable_wall_contacts=False,
        drag_model="stokes",
        fluid_density=1000.0,
        fluid_viscosity=1e-3,
    )
    
    solver = DEMSolver(config)
    solver.initialize([particle])
    
    # Run simulation
    fluid_vel = torch.zeros(3, dtype=torch.float64)
    n_steps = 1000
    
    print(f"Initial position: z = {particle.position[2].item():.3f} m")
    print(f"Initial velocity: v = {particle.velocity[2].item():.3f} m/s")
    
    for step in range(n_steps):
        solver.step(fluid_vel)  # Pass as (3,) vector, will be broadcast
        
        if step % 200 == 0:
            z = solver.system[0].position[2].item()
            v = solver.system[0].velocity[2].item()
            print(f"Step {step:4d}: z = {z:.4f} m, v = {v:.4f} m/s")
    
    print(f"\nFinal position: z = {solver.system[0].position[2].item():.3f} m")
    print(f"Final velocity: v = {solver.system[0].velocity[2].item():.3f} m/s")
    print("✓ Particle settling demonstration complete")


def demo_multiple_particles():
    """Demonstrate multiple particles with contacts."""
    print("\n" + "="*60)
    print("DEMO 2: Multiple Particles with Contacts")
    print("="*60)
    
    # Create multiple particles
    min_pos = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64)
    max_pos = torch.tensor([0.2, 0.2, 0.8], dtype=torch.float64)
    
    particles = create_ice_particles(
        n_particles=5,
        radius_range=(0.01, 0.02),
        position_range=(min_pos, max_pos),
        density=917.0,  # Ice
        seed=42,
    )
    
    # Configure solver
    config = DEMConfig(
        time_step=1e-4,
        gravity=torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64),
        domain_bounds=(
            torch.zeros(3, dtype=torch.float64),
            torch.tensor([0.3, 0.3, 1.0], dtype=torch.float64),
        ),
        enable_wall_contacts=True,
        drag_model="schiller_naumann",
        fluid_density=1000.0,
        fluid_viscosity=1e-3,
    )
    
    solver = DEMSolver(config)
    solver.initialize(particles)
    
    print(f"Created {len(particles)} ice particles")
    for i, p in enumerate(particles):
        print(f"  Particle {i}: r = {p.radius*100:.1f} cm, pos = ({p.position[0]:.3f}, {p.position[1]:.3f}, {p.position[2]:.3f})")
    
    # Run simulation
    fluid_vel = torch.zeros(3, dtype=torch.float64)
    n_steps = 500
    
    for step in range(n_steps):
        solver.step(fluid_vel)  # Pass as (3,) vector, will be broadcast
        
        if step % 100 == 0:
            ke = solver.system.total_kinetic_energy()
            print(f"Step {step:4d}: KE = {ke:.6f} J")
    
    print(f"\nFinal kinetic energy: {solver.system.total_kinetic_energy():.6f} J")
    print("✓ Multiple particles demonstration complete")


def demo_propeller_ice():
    """Demonstrate propeller-ice interaction benchmark."""
    print("\n" + "="*60)
    print("DEMO 3: Propeller-Ice Interaction Benchmark")
    print("="*60)
    
    # Configure benchmark (small for demo)
    config = PropellerIceConfig(
        nx=60,
        ny=40,
        nz=40,
        dx=0.01,
        dt=1e-4,
        propeller_diameter=30.0,
        propeller_rpm=10.0,
        n_particles=3,
        particle_radius_range=(0.02, 0.04),
        inflow_velocity=1.5,
        n_steps=100,
        output_interval=20,
        device="cpu",
    )
    
    print(f"Domain: {config.nx}×{config.ny}×{config.nz}")
    print(f"Propeller: D={config.propeller_diameter} lu, RPM={config.propeller_rpm}")
    print(f"Ice particles: {config.n_particles}")
    print(f"Steps: {config.n_steps}")
    
    # Run benchmark
    benchmark = PropellerIceBenchmark(config)
    results = benchmark.run()
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"Mean thrust: {results['thrust_mean']:.2f} N")
    print(f"Thrust RMS: {results['thrust_rms']:.2f} N")
    print(f"Mean torque: {results['torque_mean']:.3f} N·m")
    print(f"Elapsed time: {results['elapsed_time_s']:.2f} s")
    print("✓ Propeller-ice benchmark complete")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LBM-DEM Coupling Demonstration")
    print("Mission m5529: Ice Zone Propulsion Performance")
    print("="*60)
    
    # Run demonstrations
    demo_dem_settling()
    demo_multiple_particles()
    demo_propeller_ice()
    
    print("\n" + "="*60)
    print("All demonstrations complete!")
    print("="*60 + "\n")
