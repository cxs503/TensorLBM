"""Tests for LBM-DEM coupling via IBM.

Includes verification against analytical solutions:
1. Single particle drag coefficient vs Clift et al. (1978)
2. Particle settling velocity vs Stokes correction
"""

import math
import pytest
import torch

from tensorlbm.dem import DEMSolver, DEMConfig, IceParticle
from tensorlbm.ice_particle_ibm import (
    LBMDemCoupling,
    LBMDemCouplingConfig,
    create_ice_particles,
)


def test_create_ice_particles():
    """Test particle creation utility."""
    min_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    max_pos = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    
    particles = create_ice_particles(
        n_particles=5,
        radius_range=(0.05, 0.1),
        position_range=(min_pos, max_pos),
        density=917.0,
        seed=42,
    )
    
    assert len(particles) == 5
    
    for p in particles:
        assert 0.05 <= p.radius <= 0.1
        assert p.density == 917.0
        assert all(0.0 <= p.position[i] <= 1.0 for i in range(3))


def test_surface_marker_generation():
    """Test surface marker generation for IBM."""
    particle = IceParticle(
        position=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        velocity=torch.zeros(3, dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=0.1,
    )
    
    coupling = LBMDemCoupling(DEMSolver(DEMConfig()), LBMDemCouplingConfig())
    markers = coupling.generate_surface_markers(particle, n_markers=50)
    
    assert markers.shape[0] >= 40  # Approximately 50
    assert markers.shape[1] == 3
    
    # Check that markers are on the surface
    distances = torch.norm(markers - particle.position.unsqueeze(0), dim=1)
    assert torch.allclose(distances, torch.full_like(distances, 0.1), atol=1e-10)


def test_ibm_velocity_interpolation():
    """Test IBM velocity interpolation."""
    # Create a simple uniform flow field
    nx, ny, nz = 20, 20, 20
    ux = torch.ones((nz, ny, nx), dtype=torch.float32)
    uy = torch.zeros((nz, ny, nx), dtype=torch.float32)
    uz = torch.zeros((nz, ny, nx), dtype=torch.float32)
    
    # Markers at various positions
    markers = torch.tensor([
        [5.5, 10.0, 10.0],
        [10.0, 5.5, 10.0],
        [10.0, 10.0, 5.5],
    ], dtype=torch.float64)
    
    coupling = LBMDemCoupling(DEMSolver(DEMConfig()), LBMDemCouplingConfig())
    u_interp = coupling.interpolate_velocity_to_markers(ux, uy, uz, markers)
    
    # Should interpolate to uniform flow
    assert u_interp.shape == (3, 3)
    assert torch.allclose(u_interp[:, 0], torch.ones(3), atol=0.1)
    assert torch.allclose(u_interp[:, 1], torch.zeros(3), atol=0.1)
    assert torch.allclose(u_interp[:, 2], torch.zeros(3), atol=0.1)


def test_drag_force_stokes_regime():
    """Test drag force in Stokes regime (Re << 1).
    
    Analytical solution: F_D = 6 * π * μ * R * v_rel
    
    This is a verification test for the drag model in DEM solver.
    """
    mu = 1e-3  # Pa·s
    R = 0.01  # m
    v_rel = 0.01  # m/s (small for Stokes regime)
    
    # Particle Reynolds number
    Re_p = 1000.0 * v_rel * 2 * R / mu  # Should be << 1
    assert Re_p < 1.0, f"Re_p = {Re_p}, not in Stokes regime"
    
    # Analytical drag force
    F_analytical = 6.0 * math.pi * mu * R * v_rel
    
    # Create particle and compute drag
    config = DEMConfig(
        time_step=1e-4,
        drag_model="stokes",
        fluid_density=1000.0,
        fluid_viscosity=mu,
    )
    
    particle = IceParticle(
        position=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        velocity=torch.tensor([v_rel, 0.0, 0.0], dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=R,
    )
    
    solver = DEMSolver(config)
    solver.initialize([particle])
    
    fluid_vel = torch.zeros(3, dtype=torch.float64)
    F_computed = solver.compute_drag_force(particle, fluid_vel)
    
    # Should match analytical (within 5%)
    F_computed_x = F_computed[0].item()
    assert abs(F_computed_x - F_analytical) / F_analytical < 0.05


def test_drag_force_schiller_naumann():
    """Test drag force with Schiller-Naumann correlation.
    
    For Re_p < 1000:
        C_D = (24/Re_p) * (1 + 0.15 * Re_p^0.687)
        F_D = 0.5 * C_D * ρ_f * A * |v_rel| * v_rel
    """
    mu = 1e-3  # Pa·s
    rho_f = 1000.0  # kg/m³
    R = 0.01  # m
    v_rel = 0.1  # m/s
    
    # Particle Reynolds number
    Re_p = rho_f * v_rel * 2 * R / mu
    
    # Analytical drag coefficient
    C_D = (24.0 / Re_p) * (1.0 + 0.15 * Re_p**0.687)
    
    # Analytical drag force
    A = math.pi * R**2
    F_analytical = 0.5 * C_D * rho_f * A * v_rel**2
    
    # Create particle and compute drag
    config = DEMConfig(
        time_step=1e-4,
        drag_model="schiller_naumann",
        fluid_density=rho_f,
        fluid_viscosity=mu,
    )
    
    particle = IceParticle(
        position=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        velocity=torch.tensor([v_rel, 0.0, 0.0], dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=R,
    )
    
    solver = DEMSolver(config)
    solver.initialize([particle])
    
    fluid_vel = torch.zeros(3, dtype=torch.float64)
    F_computed = solver.compute_drag_force(particle, fluid_vel)
    
    # Should match analytical (within 5%)
    F_computed_x = F_computed[0].item()
    assert abs(F_computed_x - F_analytical) / F_analytical < 0.05


def test_particle_settling_velocity_stokes_short():
    """Test particle settling velocity in Stokes regime (short version).
    
    Terminal velocity when gravity balances drag:
        m * g = 6 * π * μ * R * v_t
        v_t = m * g / (6 * π * μ * R)
    
    With buoyancy correction:
        v_t = (ρ_p - ρ_f) * g * D² / (18 * μ)
    
    where D = 2R is the particle diameter.
    """
    rho_p = 1100.0  # kg/m³ (slightly denser than water)
    rho_f = 1000.0  # kg/m³
    mu = 5e-3  # Pa·s (higher viscosity for faster relaxation)
    R = 0.005  # m (5 mm)
    g = 9.81  # m/s²
    
    # Analytical terminal velocity
    D = 2 * R
    v_t_analytical = (rho_p - rho_f) * g * D**2 / (18 * mu)
    
    # Create particle
    m = rho_p * (4.0 / 3.0) * math.pi * R**3
    
    config = DEMConfig(
        time_step=5e-4,  # Larger time step
        gravity=torch.tensor([0.0, 0.0, -g], dtype=torch.float64),
        enable_wall_contacts=False,
        drag_model="stokes",
        fluid_density=rho_f,
        fluid_viscosity=mu,
    )
    
    particle = IceParticle(
        position=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        velocity=torch.zeros(3, dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=R,
        mass=m,
    )
    
    solver = DEMSolver(config)
    solver.initialize([particle])
    
    # Fluid at rest
    fluid_vel = torch.zeros(3, dtype=torch.float64)
    
    # Run until terminal velocity (approx 3 * relaxation time)
    tau_p = m / (6.0 * math.pi * mu * R)
    t_final = 5.0 * tau_p
    n_steps = int(t_final / config.time_step)
    
    for _ in range(n_steps):
        solver.step(fluid_vel.unsqueeze(0))
    
    # Check terminal velocity
    v_t_computed = abs(solver.system[0].velocity[2].item())
    
    # Should match analytical (within 15% due to first-order integration)
    assert abs(v_t_computed - v_t_analytical) / v_t_analytical < 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
