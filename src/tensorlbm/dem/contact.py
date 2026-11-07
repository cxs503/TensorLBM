"""Contact mechanics models for DEM simulations.

Implements linear spring-dashpot model for particle-particle and
particle-wall contacts.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .particle import IceParticle


@dataclass
class ContactInfo:
    """Information about a contact between two particles or particle-wall.
    
    Attributes
    ----------
    normal_vector : torch.Tensor
        Contact normal vector (unit vector from particle 1 to particle 2).
    overlap : float
        Overlap distance in meters (positive = overlapping).
    contact_point : torch.Tensor
        Contact point position in meters.
    normal_force : torch.Tensor
        Normal contact force vector in Newtons.
    tangential_force : torch.Tensor
        Tangential contact force vector in Newtons.
    total_force : torch.Tensor
        Total contact force vector in Newtons.
    """
    normal_vector: torch.Tensor
    overlap: float
    contact_point: torch.Tensor
    normal_force: torch.Tensor
    tangential_force: torch.Tensor
    total_force: torch.Tensor


class ContactModel(ABC):
    """Abstract base class for contact models."""
    
    @abstractmethod
    def compute_contact_force(
        self,
        p1: IceParticle,
        p2: Optional[IceParticle],
        normal: torch.Tensor,
        overlap: float,
        relative_velocity: torch.Tensor,
    ) -> ContactInfo:
        """Compute contact force between two particles or particle-wall.
        
        Parameters
        ----------
        p1 : IceParticle
            First particle.
        p2 : IceParticle or None
            Second particle (None for wall contact).
        normal : torch.Tensor
            Contact normal vector (unit vector).
        overlap : float
            Overlap distance in meters.
        relative_velocity : torch.Tensor
            Relative velocity at contact point in m/s.
        
        Returns
        -------
        ContactInfo
            Contact information including forces.
        """
        pass


class LinearSpringDashpot(ContactModel):
    """Linear spring-dashpot contact model.
    
    The normal force is:
        F_n = k_n * δ - η_n * v_n
    
    where:
        k_n = normal stiffness (N/m)
        δ = overlap (m)
        η_n = normal damping coefficient (N·s/m)
        v_n = normal relative velocity (m/s)
    
    The tangential force is:
        F_t = min(k_t * s_t, μ * |F_n|)
    
    where:
        k_t = tangential stiffness (N/m)
        s_t = tangential displacement (m)
        μ = friction coefficient
        |F_n| = magnitude of normal force
    
    Parameters
    ----------
    normal_stiffness : float
        Normal stiffness k_n in N/m.
    tangential_stiffness : float
        Tangential stiffness k_t in N/m.
    normal_damping : float
        Normal damping coefficient η_n in N·s/m.
    tangential_damping : float
        Tangential damping coefficient η_t in N·s/m.
    friction_coefficient : float
        Friction coefficient μ (dimensionless).
    """
    
    def __init__(
        self,
        normal_stiffness: float = 1e5,
        tangential_stiffness: float = 1e4,
        normal_damping: float = 1e2,
        tangential_damping: float = 1e1,
        friction_coefficient: float = 0.3,
    ):
        self.normal_stiffness = normal_stiffness
        self.tangential_stiffness = tangential_stiffness
        self.normal_damping = normal_damping
        self.tangential_damping = tangential_damping
        self.friction_coefficient = friction_coefficient
        
        # Store tangential displacement history (simplified: reset each step)
        self._tangential_displacement: dict[tuple[int, int], torch.Tensor] = {}
    
    def compute_contact_force(
        self,
        p1: IceParticle,
        p2: Optional[IceParticle],
        normal: torch.Tensor,
        overlap: float,
        relative_velocity: torch.Tensor,
    ) -> ContactInfo:
        """Compute linear spring-dashpot contact force."""
        device = p1.position.device
        
        # Normal components
        v_n = torch.dot(relative_velocity, normal)
        normal_velocity = v_n * normal
        tangential_velocity = relative_velocity - normal_velocity
        
        # Normal force: F_n = k_n * δ - η_n * v_n
        # Positive overlap = compression = repulsive force
        f_n_magnitude = self.normal_stiffness * overlap - self.normal_damping * v_n.item()
        f_n_magnitude = max(0.0, f_n_magnitude)  # No tension
        normal_force = f_n_magnitude * normal
        
        # Tangential force (simplified: no history tracking for now)
        # F_t = -k_t * s_t - η_t * v_t, limited by Coulomb friction
        tangential_force = -self.tangential_stiffness * torch.zeros_like(tangential_velocity) \
                          - self.tangential_damping * tangential_velocity
        
        # Coulomb friction limit
        f_t_magnitude = torch.norm(tangential_force).item()
        f_n_magnitude_val = torch.norm(normal_force).item()
        max_friction = self.friction_coefficient * f_n_magnitude_val
        
        if f_t_magnitude > max_friction and f_t_magnitude > 1e-10:
            tangential_force = tangential_force * (max_friction / f_t_magnitude)
        
        # Total force
        total_force = normal_force + tangential_force
        
        # Contact point (midpoint for equal-sized particles)
        if p2 is not None:
            contact_point = p1.position + normal * (p1.radius - 0.5 * overlap)
        else:
            # Wall contact
            contact_point = p1.position - normal * p1.radius
        
        return ContactInfo(
            normal_vector=normal,
            overlap=overlap,
            contact_point=contact_point,
            normal_force=normal_force,
            tangential_force=tangential_force,
            total_force=total_force,
        )
    
    def reset_tangential_history(self):
        """Reset tangential displacement history (call each time step)."""
        self._tangential_displacement.clear()


def detect_particle_particle_contact(
    p1: IceParticle,
    p2: IceParticle,
) -> Tuple[bool, float, torch.Tensor]:
    """Detect contact between two spherical particles.
    
    Parameters
    ----------
    p1, p2 : IceParticle
        Two particles to check.
    
    Returns
    -------
    in_contact : bool
        True if particles are in contact.
    overlap : float
        Overlap distance in meters (positive = overlapping).
    normal : torch.Tensor
        Contact normal vector (unit vector from p1 to p2).
    """
    diff = p2.position - p1.position
    distance = torch.norm(diff).item()
    sum_radii = p1.radius + p2.radius
    
    if distance < sum_radii:
        overlap = sum_radii - distance
        normal = diff / max(distance, 1e-10)
        return True, overlap, normal
    else:
        return False, 0.0, torch.zeros_like(diff)


def detect_particle_wall_contact(
    particle: IceParticle,
    wall_position: torch.Tensor,
    wall_normal: torch.Tensor,
) -> Tuple[bool, float, torch.Tensor]:
    """Detect contact between particle and a flat wall.
    
    Parameters
    ----------
    particle : IceParticle
        Particle to check.
    wall_position : torch.Tensor
        Point on the wall in meters.
    wall_normal : torch.Tensor
        Wall normal vector (unit vector pointing into fluid domain).
    
    Returns
    -------
    in_contact : bool
        True if particle is in contact with wall.
    overlap : float
        Overlap distance in meters (positive = penetrating).
    normal : torch.Tensor
        Contact normal vector (same as wall_normal).
    """
    # Distance from particle center to wall
    diff = particle.position - wall_position
    distance = torch.dot(diff, wall_normal).item()
    
    if distance < particle.radius:
        overlap = particle.radius - distance
        return True, overlap, wall_normal
    else:
        return False, 0.0, wall_normal
