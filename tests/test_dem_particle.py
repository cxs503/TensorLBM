"""Tests for DEM particle dynamics."""

import math
import pytest
import torch

from tensorlbm.dem.particle import IceParticle, ParticleSystem


def test_particle_creation():
    """Test basic particle creation."""
    pos = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    vel = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    omega = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
    
    particle = IceParticle(
        position=pos,
        velocity=vel,
        angular_velocity=omega,
        radius=0.1,
        density=917.0,
    )
    
    assert particle.radius == 0.1
    assert particle.density == 917.0
    assert particle.mass is not None
    assert particle.moment_of_inertia is not None
    
    # Check mass computation
    expected_volume = (4.0 / 3.0) * math.pi * 0.1**3
    expected_mass = 917.0 * expected_volume
    assert abs(particle.mass - expected_mass) < 1e-10
    
    # Check moment of inertia (solid sphere)
    expected_I = 0.4 * expected_mass * 0.1**2
    assert abs(particle.moment_of_inertia - expected_I) < 1e-10


def test_particle_kinetic_energy():
    """Test kinetic energy computation."""
    pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    omega = torch.tensor([0.0, 10.0, 0.0], dtype=torch.float64)
    
    particle = IceParticle(
        position=pos,
        velocity=vel,
        angular_velocity=omega,
        radius=0.1,
        mass=1.0,
        moment_of_inertia=0.004,  # 0.4 * 1.0 * 0.1^2
    )
    
    ke = particle.kinetic_energy()
    
    # Translational: 0.5 * 1.0 * 1.0^2 = 0.5
    # Rotational: 0.5 * 0.004 * 10.0^2 = 0.2
    expected_ke = 0.5 + 0.2
    assert abs(ke - expected_ke) < 1e-10


def test_particle_momentum():
    """Test momentum computation."""
    pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    vel = torch.tensor([2.0, 3.0, 0.0], dtype=torch.float64)
    omega = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64)
    
    particle = IceParticle(
        position=pos,
        velocity=vel,
        angular_velocity=omega,
        radius=0.1,
        mass=2.0,
        moment_of_inertia=0.008,
    )
    
    momentum = particle.momentum()
    expected_momentum = 2.0 * vel
    torch.testing.assert_close(momentum, expected_momentum)
    
    angular_momentum = particle.angular_momentum()
    expected_angular = 0.008 * omega
    torch.testing.assert_close(angular_momentum, expected_angular)


def test_particle_system():
    """Test particle system."""
    particles = []
    for i in range(3):
        pos = torch.tensor([float(i), 0.0, 0.0], dtype=torch.float64)
        vel = torch.tensor([0.1 * i, 0.0, 0.0], dtype=torch.float64)
        omega = torch.zeros(3, dtype=torch.float64)
        
        particle = IceParticle(
            position=pos,
            velocity=vel,
            angular_velocity=omega,
            radius=0.1,
            mass=1.0,
        )
        particles.append(particle)
    
    system = ParticleSystem(particles)
    
    assert len(system) == 3
    assert system[0].radius == 0.1
    
    # Test positions tensor
    positions = system.positions_tensor()
    assert positions.shape == (3, 3)
    assert positions[0, 0].item() == 0.0
    assert positions[1, 0].item() == 1.0
    assert positions[2, 0].item() == 2.0
    
    # Test total momentum
    total_momentum = system.total_momentum()
    expected = torch.tensor([0.3, 0.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(total_momentum, expected)


def test_particle_volume_and_area():
    """Test volume and cross-section area."""
    particle = IceParticle(
        position=torch.zeros(3, dtype=torch.float64),
        velocity=torch.zeros(3, dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=0.5,
    )
    
    expected_volume = (4.0 / 3.0) * math.pi * 0.5**3
    expected_area = math.pi * 0.5**2
    
    assert abs(particle.volume - expected_volume) < 1e-10
    assert abs(particle.cross_section_area - expected_area) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
