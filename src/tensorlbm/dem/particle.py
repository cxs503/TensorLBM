"""Ice particle dynamics for DEM simulations.

Implements Newton's equations for translation and rotation of spherical
ice particles in a fluid medium.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class IceParticle:
    """Single spherical ice particle.
    
    Parameters
    ----------
    position : torch.Tensor
        Position vector (x, y, z) in meters, shape (3,).
    velocity : torch.Tensor
        Velocity vector (vx, vy, vz) in m/s, shape (3,).
    angular_velocity : torch.Tensor
        Angular velocity vector (wx, wy, wz) in rad/s, shape (3,).
    radius : float
        Particle radius in meters.
    density : float
        Particle density in kg/m³ (ice ~917 kg/m³).
    mass : float, optional
        Particle mass in kg. If None, computed from density and radius.
    moment_of_inertia : float, optional
        Moment of inertia in kg·m². If None, computed for solid sphere.
    """
    position: torch.Tensor
    velocity: torch.Tensor
    angular_velocity: torch.Tensor
    radius: float
    density: float = 917.0  # Ice density kg/m³
    mass: Optional[float] = None
    moment_of_inertia: Optional[float] = None
    
    def __post_init__(self):
        if self.mass is None:
            volume = (4.0 / 3.0) * math.pi * self.radius**3
            self.mass = self.density * volume
        if self.moment_of_inertia is None:
            # Solid sphere: I = (2/5) * m * r²
            self.moment_of_inertia = 0.4 * self.mass * self.radius**2
        
        # Ensure tensors are on same device
        device = self.position.device
        self.velocity = self.velocity.to(device)
        self.angular_velocity = self.angular_velocity.to(device)
    
    @property
    def volume(self) -> float:
        """Particle volume in m³."""
        return (4.0 / 3.0) * math.pi * self.radius**3
    
    @property
    def cross_section_area(self) -> float:
        """Cross-sectional area in m²."""
        return math.pi * self.radius**2
    
    def kinetic_energy(self) -> float:
        """Total kinetic energy (translational + rotational) in Joules."""
        trans_ke = 0.5 * self.mass * torch.sum(self.velocity**2).item()
        rot_ke = 0.5 * self.moment_of_inertia * torch.sum(self.angular_velocity**2).item()
        return trans_ke + rot_ke
    
    def momentum(self) -> torch.Tensor:
        """Linear momentum in kg·m/s."""
        return self.mass * self.velocity
    
    def angular_momentum(self) -> torch.Tensor:
        """Angular momentum in kg·m²/s."""
        return self.moment_of_inertia * self.angular_velocity


@dataclass
class ParticleSystem:
    """Collection of ice particles.
    
    Parameters
    ----------
    particles : list of IceParticle
        List of particles in the system.
    gravity : torch.Tensor
        Gravitational acceleration vector (gx, gy, gz) in m/s².
        Default is (0, 0, -9.81) for Earth gravity in -z direction.
    """
    particles: list[IceParticle]
    gravity: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0, 0.0, -9.81]))
    
    def __len__(self) -> int:
        return len(self.particles)
    
    def __getitem__(self, idx: int) -> IceParticle:
        return self.particles[idx]
    
    def total_kinetic_energy(self) -> float:
        """Total kinetic energy of all particles in Joules."""
        return sum(p.kinetic_energy() for p in self.particles)
    
    def total_momentum(self) -> torch.Tensor:
        """Total linear momentum of all particles in kg·m/s."""
        if len(self.particles) == 0:
            return torch.zeros(3)
        return sum(p.momentum() for p in self.particles)
    
    def positions_tensor(self) -> torch.Tensor:
        """Stack all particle positions into (N, 3) tensor."""
        if len(self.particles) == 0:
            return torch.empty((0, 3))
        return torch.stack([p.position for p in self.particles])
    
    def velocities_tensor(self) -> torch.Tensor:
        """Stack all particle velocities into (N, 3) tensor."""
        if len(self.particles) == 0:
            return torch.empty((0, 3))
        return torch.stack([p.velocity for p in self.particles])
    
    def radii_tensor(self) -> torch.Tensor:
        """Stack all particle radii into (N,) tensor."""
        if len(self.particles) == 0:
            return torch.empty(0)
        return torch.tensor([p.radius for p in self.particles])
