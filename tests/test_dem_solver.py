"""Tests for DEM contact mechanics and solver."""

import math
import pytest
import torch

from tensorlbm.dem import (
    DEMSolver,
    DEMConfig,
    IceParticle,
    LinearSpringDashpot,
)
from tensorlbm.dem.contact import (
    detect_particle_particle_contact,
    detect_particle_wall_contact,
)


def test_particle_particle_contact_detection():
    """Test particle-particle contact detection."""
    p1 = IceParticle(
        position=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        velocity=torch.zeros(3, dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=0.1,
    )
    
    p2 = IceParticle(
        position=torch.tensor([0.15, 0.0, 0.0], dtype=torch.float64),
        velocity=torch.zeros(3, dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=0.1,
    )
    
    in_contact, overlap, normal = detect_particle_particle_contact(p1, p2)
    
    assert in_contact
    assert abs(overlap - 0.05) < 1e-10  # 0.2 - 0.15 = 0.05
    assert abs(normal[0].item() - 1.0) < 1e-10  # Normal points in +x
    assert abs(normal[1].item()) < 1e-10
    assert abs(normal[2].item()) < 1e-10


def test_particle_particle_no_contact():
    """Test particles not in contact."""
    p1 = IceParticle(
        position=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        velocity=torch.zeros(3, dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=0.1,
    )
    
    p2 = IceParticle(
        position=torch.tensor([0.3, 0.0, 0.0], dtype=torch.float64),
        velocity=torch.zeros(3, dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=0.1,
    )
    
    in_contact, overlap, normal = detect_particle_particle_contact(p1, p2)
    
    assert not in_contact
    assert overlap == 0.0


def test_particle_wall_contact():
    """Test particle-wall contact detection."""
    particle = IceParticle(
        position=torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64),
        velocity=torch.zeros(3, dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=0.1,
    )
    
    wall_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    wall_normal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    
    in_contact, overlap, normal = detect_particle_wall_contact(
        particle, wall_pos, wall_normal
    )
    
    assert in_contact
    assert abs(overlap - 0.05) < 1e-10  # 0.1 - 0.05 = 0.05


def test_contact_force_computation():
    """Test linear spring-dashpot contact force."""
    model = LinearSpringDashpot(
        normal_stiffness=1e5,
        tangential_stiffness=1e4,
        normal_damping=1e2,
        friction_coefficient=0.3,
    )
    
    p1 = IceParticle(
        position=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        velocity=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=0.1,
    )
    
    p2 = IceParticle(
        position=torch.tensor([0.15, 0.0, 0.0], dtype=torch.float64),
        velocity=torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=0.1,
    )
    
    normal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    overlap = 0.05
    relative_velocity = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
    
    contact = model.compute_contact_force(p1, p2, normal, overlap, relative_velocity)
    
    # Normal force should be repulsive (positive x direction on p1)
    assert contact.normal_force[0].item() > 0
    # Overlap = 0.05, k_n = 1e5, so F_n ≈ 5000 N (minus damping)
    assert contact.normal_force[0].item() > 4000


def test_dem_solver_free_fall():
    """Test DEM solver with particle in free fall (no contacts)."""
    config = DEMConfig(
        time_step=1e-3,
        gravity=torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64),
        enable_wall_contacts=False,
        drag_model="none",
    )
    
    particle = IceParticle(
        position=torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64),
        velocity=torch.zeros(3, dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=0.1,
        mass=1.0,
    )
    
    solver = DEMSolver(config)
    solver.initialize([particle])
    
    # Run for 1 second
    n_steps = 1000
    for _ in range(n_steps):
        solver.step()
    
    # Check position: z = z0 - 0.5 * g * t^2
    t = 1.0
    expected_z = 10.0 - 0.5 * 9.81 * t**2
    actual_z = solver.system[0].position[2].item()
    
    assert abs(actual_z - expected_z) < 0.1  # Allow some numerical error
    
    # Check velocity: v = -g * t
    expected_vz = -9.81 * t
    actual_vz = solver.system[0].velocity[2].item()
    
    assert abs(actual_vz - expected_vz) < 0.1


def test_dem_solver_stokes_drag_short():
    """Test DEM solver with Stokes drag (short version for CI).
    
    For a particle in Stokes flow (Re << 1), the drag force is:
        F_D = 6 * π * μ * R * v_rel
    
    The particle velocity decays exponentially:
        v(t) = v0 * exp(-t / τ_p)
    
    where τ_p = m / (6 * π * μ * R) is the particle relaxation time.
    """
    mu = 1e-2  # Pa·s (higher viscosity for faster relaxation)
    R = 0.01  # m
    rho_p = 1000.0  # kg/m³
    m = rho_p * (4.0 / 3.0) * math.pi * R**3
    
    config = DEMConfig(
        time_step=1e-3,  # Larger time step
        gravity=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        enable_wall_contacts=False,
        drag_model="stokes",
        fluid_density=1000.0,
        fluid_viscosity=mu,
    )
    
    v0 = 1.0  # m/s
    particle = IceParticle(
        position=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        velocity=torch.tensor([v0, 0.0, 0.0], dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=R,
        mass=m,
    )
    
    solver = DEMSolver(config)
    solver.initialize([particle])
    
    # Fluid at rest
    fluid_vel = torch.zeros(3, dtype=torch.float64)
    
    # Particle relaxation time
    tau_p = m / (6.0 * math.pi * mu * R)
    
    # Run for 2 * tau_p (shorter simulation)
    t_final = 2.0 * tau_p
    n_steps = int(t_final / config.time_step)
    
    for _ in range(n_steps):
        solver.step(fluid_vel.unsqueeze(0))
    
    # Expected velocity: v(t) = v0 * exp(-t / tau_p)
    expected_v = v0 * math.exp(-t_final / tau_p)
    actual_v = solver.system[0].velocity[0].item()
    
    # Allow 20% error (first-order time integration)
    assert abs(actual_v - expected_v) / expected_v < 0.2


def test_dem_solver_energy_conservation():
    """Test that kinetic energy is conserved in elastic collision."""
    config = DEMConfig(
        time_step=1e-4,
        gravity=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        enable_wall_contacts=False,
        drag_model="none",
    )
    
    # Two particles approaching each other
    p1 = IceParticle(
        position=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        velocity=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=0.1,
        mass=1.0,
    )
    
    p2 = IceParticle(
        position=torch.tensor([0.15, 0.0, 0.0], dtype=torch.float64),
        velocity=torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=0.1,
        mass=1.0,
    )
    
    solver = DEMSolver(config)
    solver.initialize([p1, p2])
    
    # Initial kinetic energy
    ke_initial = solver.system.total_kinetic_energy()
    
    # Run for a few steps (collision should happen)
    for _ in range(100):
        solver.step()
    
    # Final kinetic energy (should be similar if elastic)
    ke_final = solver.system.total_kinetic_energy()
    
    # Allow some energy loss due to damping
    assert ke_final <= ke_initial
    # But not too much (elastic collision)
    assert ke_final > 0.5 * ke_initial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
