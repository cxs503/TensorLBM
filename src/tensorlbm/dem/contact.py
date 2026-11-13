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




# ---------------------------------------------------------------------------
# BondedDEM — 3-D breakable-ice (cohesive bond) leg for the T5 flagship
# ---------------------------------------------------------------------------
# GAP-LBM-T5-ICE-COUPLING requires the DEM leg to *fail/break* ice (R3), not
# merely to repel.  LinearSpringDashpot above is normal-only (f_n = max(0, ...),
# no tension), so a bonded ice cover can never fracture.  BondedDEM adds a
# cohesive bond between every pair of particles in contact at initialisation:
#   * the bond carries *tension* (F = k_b * (d - rest)) as well as compression,
#   * it ruptures (irreversibly) when the axial or shear strain exceeds a
#     threshold, or when |F| exceeds the bond strength,
#   * a ruptured bond releases fracture energy (booked for the energy ledger),
#   * after rupture the pair reverts to the no-tension LinearSpringDashpot law.
# It operates on particle *arrays* by index (the same layout IceDEM3D uses), so
# it drops straight into the three_way DEM leg.


@dataclass
class Bond:
    """A cohesive bond between particles ``i`` and ``j`` (``i < j``)."""

    i: int
    j: int
    rest_length: float
    stiffness: float
    strength: float
    breaking_strain: float
    shear_breaking_strain: float
    broken: bool = False


class BondedDEM:
    """Breakable-ice DEM: cohesive bonds + no-tension fallback after rupture.

    Parameters
    ----------
    positions : torch.Tensor
        ``(n, 3)`` particle centres (m).
    velocities : torch.Tensor
        ``(n, 3)`` particle velocities (m/s).
    radius : float
        Particle radius (m) — all particles equal.
    bond_stiffness : float
        Bond stiffness k_b (N/m).
    bond_strength : float
        Maximum bond force (N).
    breaking_strain : float
        Axial strain rupture threshold (dimensionless).
    shear_breaking_strain : float
        Shear strain rupture threshold (dimensionless).
    normal_damping : float
        Dashpot on the normal relative velocity (N·s/m).
    """

    def __init__(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        radius: float,
        bond_stiffness: float = 1e5,
        bond_strength: float = 1e3,
        breaking_strain: float = 0.015,
        shear_breaking_strain: float = 0.03,
        normal_damping: float = 1e2,
    ):
        self.positions = positions.double()
        self.velocities = velocities.double()
        self.radius = float(radius)
        self.bond_stiffness = bond_stiffness
        self.bond_strength = bond_strength
        self.breaking_strain = breaking_strain
        self.shear_breaking_strain = shear_breaking_strain
        self.normal_damping = normal_damping
        self.bonds: list[Bond] = []
        self.fracture_energy_released: float = 0.0
        self.n_broken: int = 0
        self._form_bonds()

    def _form_bonds(self) -> None:
        """Bond every pair of particles in contact at initialisation."""
        n = self.positions.shape[0]
        for i in range(n - 1):
            for j in range(i + 1, n):
                d = torch.norm(self.positions[j] - self.positions[i]).item()
                if d <= 2.0 * self.radius:
                    self.bonds.append(
                        Bond(
                            i=i, j=j, rest_length=2.0 * self.radius,
                            stiffness=self.bond_stiffness,
                            strength=self.bond_strength,
                            breaking_strain=self.breaking_strain,
                            shear_breaking_strain=self.shear_breaking_strain,
                        )
                    )

    def step(self) -> torch.Tensor:
        """Advance one bond step; return the ``(n, 3)`` net cohesive force.

        Intact bonds apply tension/compression ``F = k_b * (d - rest)`` plus a
        normal dashpot.  A bond ruptures when ``|d - rest|/rest > breaking_strain``
        or ``|v_t|/rest > shear_breaking_strain`` or ``|F| > strength``; on rupture
        the pair reverts to the no-tension ``LinearSpringDashpot`` and the released
        elastic energy is booked into :attr:`fracture_energy_released`.
        """
        n = self.positions.shape[0]
        net = torch.zeros((n, 3), dtype=self.positions.dtype, device=self.positions.device)
        for b in self.bonds:
            if b.broken:
                continue
            p_i = self.positions[b.i]
            p_j = self.positions[b.j]
            v_i = self.velocities[b.i]
            v_j = self.velocities[b.j]
            diff = p_j - p_i
            d = torch.norm(diff).item()
            if d < 1e-12:
                continue
            n_hat = diff / d
            axial = (d - b.rest_length) / b.rest_length
            v_rel = v_i - v_j
            v_n = float(torch.dot(v_rel, n_hat))
            v_t = torch.linalg.norm(v_rel - v_n * n_hat).item()
            shear = v_t / b.rest_length
            f_mag = self.bond_stiffness * (d - b.rest_length)
            # rupture criterion
            if (abs(axial) > self.breaking_strain
                    or shear > self.shear_breaking_strain
                    or abs(f_mag) > self.bond_strength):
                b.broken = True
                self.n_broken += 1
                self.fracture_energy_released += (
                    0.5 * self.bond_stiffness * (d - b.rest_length) ** 2
                )
                # fallback: no-tension repulsion if still overlapping
                if d < b.rest_length:
                    f_fb = self.bond_stiffness * (b.rest_length - d) - self.normal_damping * v_n
                    net[b.i] += max(0.0, f_fb) * n_hat
                    net[b.j] -= max(0.0, f_fb) * n_hat
                continue
            # intact bond: tension (d > rest) or compression (d < rest)
            f = (f_mag - self.normal_damping * v_n) * n_hat
            net[b.i] += f
            net[b.j] -= f
        return net

    @property
    def n_intact(self) -> int:
        return sum(1 for b in self.bonds if not b.broken)
