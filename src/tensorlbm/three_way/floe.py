"""Buoyant ice-floe DEM body used in the three-way coupling scenario.

A 2D (plane) circular ice floe integrated with the same linear
spring-dashpot contact law as :mod:`tensorlbm.dem` (Cundall–Strack), a
hydrostatic buoyancy law against a prescribed **still-water line** (rigid-lid
single-phase flume: the free surface is not resolved, see the limitation notes
in ``docs/THREE_WAY_LBM_DEM_FSI.md``), gravity, and the IBM reaction force
coming from the fluid.

The body keeps a 2D :class:`~tensorlbm.dem.IceParticle` mirror so that contact
forces are produced by the repository-owned contact model rather than by a
re-implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from ..dem.contact import LinearSpringDashpot
from ..dem.particle import IceParticle


def submerged_area(radius: float, d: float) -> float:
    """Area of a disc of ``radius`` below the chord at height ``d``.

    ``d`` is measured from the disc centre and positive upwards; the chord is
    the still-water line.
    """
    u = max(-1.0, min(1.0, d / radius))
    return radius * radius * (math.pi / 2.0 + math.asin(u) + u * math.sqrt(max(0.0, 1.0 - u * u)))


def draft_for_ratio(radius: float, ratio: float) -> float:
    """Distance ``d`` (positive up) of the centre below the water line."""
    lo, hi = -0.999, 0.999
    target = ratio * math.pi * radius * radius
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if submerged_area(radius, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class FloeConfig:
    """Floe and contact parameters (lattice units unless stated)."""

    radius: float = 5.0
    density_ratio: float = 0.60  # rho_ice / rho_water
    depth: float = 1.0  # out-of-plane thickness
    waterline: float = 34.0
    gravity: float = 2.0e-4
    normal_stiffness: float = 2.0
    tangential_stiffness: float = 1.0
    normal_damping: float = 2.0
    tangential_damping: float = 0.5
    friction_coefficient: float = 0.3
    seed: int = 0


class Floe2D:
    """A buoyant, contact-bearing disc (DEM rigid body, 2D)."""

    def __init__(self, cfg: FloeConfig, position: Tuple[float, float]):
        self.cfg = cfg
        self.pos = torch.tensor(position, dtype=torch.float64)
        self.vel = torch.zeros(2, dtype=torch.float64)
        self.theta = 0.0
        self.omega = 0.0
        self.area = math.pi * cfg.radius ** 2
        self.mass = cfg.density_ratio * self.area * cfg.depth
        self.inertia = 0.5 * self.mass * cfg.radius ** 2
        self.contact = LinearSpringDashpot(
            normal_stiffness=cfg.normal_stiffness,
            tangential_stiffness=cfg.tangential_stiffness,
            normal_damping=cfg.normal_damping,
            tangential_damping=cfg.tangential_damping,
            friction_coefficient=cfg.friction_coefficient,
        )
        self._mirror = IceParticle(
            position=self.pos.clone(),
            velocity=self.vel.clone(),
            angular_velocity=torch.zeros(2, dtype=torch.float64),
            radius=cfg.radius,
            density=cfg.density_ratio,
            mass=self.mass,
            moment_of_inertia=self.inertia,
        )
        #: accumulated diagnostics
        self.diagnostics = {
            "buoyancy_work": 0.0,
            "contact_work": 0.0,
            "ibm_work": 0.0,
            "contact_energy_loss": 0.0,
        }

    # -- geometry ---------------------------------------------------------
    @property
    def radius(self) -> float:
        return self.cfg.radius

    def submerged_area(self) -> float:
        return submerged_area(self.cfg.radius, self.cfg.waterline - float(self.pos[1].item()))

    def potential_energy(self) -> float:
        """Gravitational + hydrostatic potential energy of the floe.

        ``PE = m g y_c + rho_w g ∫_0^d A_sub(s) ds`` with ``d = y_w - y_c``;
        differentiating gives ``F_y = -m g + rho_w g A_sub(d)``, i.e. exactly
        the weight plus buoyancy applied in :meth:`integrate`.
        """
        r = self.cfg.radius
        d = self.cfg.waterline - float(self.pos[1].item())
        n = 40
        h = d / n
        acc = 0.0
        for i in range(n + 1):
            w = 1 if i in (0, n) else (4 if i % 2 else 2)
            acc += w * submerged_area(r, i * h)
        acc *= h / 3.0  # Simpson
        y_c = float(self.pos[1].item())
        return self.mass * self.cfg.gravity * y_c + acc * self.cfg.gravity

    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * float(self.vel[0] ** 2 + self.vel[1] ** 2) + 0.5 * self.inertia * self.omega ** 2

    # -- contact ----------------------------------------------------------
    def _sync_mirror(self) -> None:
        self._mirror.position = self.pos
        self._mirror.velocity = self.vel
        self._mirror.angular_velocity = torch.stack([torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)])

    def contact_with_wall(
        self, wall_point: torch.Tensor, wall_normal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Contact force / torque from a flat wall (normal points into fluid)."""
        self._sync_mirror()
        dist = float(torch.dot(self.pos - wall_point, wall_normal))
        if dist >= self.cfg.radius:
            return torch.zeros(2, dtype=torch.float64), torch.zeros((), dtype=torch.float64)
        overlap = self.cfg.radius - dist
        v_rel = self.vel
        info = self.contact.compute_contact_force(self._mirror, None, wall_normal, overlap, v_rel)
        r_vec = -wall_normal * self.cfg.radius
        torque = r_vec[0] * info.total_force[1] - r_vec[1] * info.total_force[0]
        return info.total_force, torque

    def contact_with_segment(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        deflection_fn=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Contact of the floe with a (possibly deformed) line segment.

        The distance is the true 2D distance from the floe centre to the
        (deformed) segment, so the floe can also graze the tip.  The segment
        points are ordered bottom ``a`` → top ``b`` and the structure surface
        is the concave (left, ``-x``) side.

        ``deflection_fn(y)`` returns the horizontal displacement of the
        structure at height ``y`` — this is how the FSI deformation feeds back
        into the DEM contact geometry.  Returns ``(force, torque, y_contact)``
        with zero force when there is no contact.
        """
        self._sync_mirror()
        zero = torch.zeros(2, dtype=torch.float64)
        zero_t = torch.zeros((), dtype=torch.float64)
        xc, yc = float(self.pos[0].item()), float(self.pos[1].item())
        y_lo, y_hi = sorted((float(a[1]), float(b[1])))
        y_contact = min(max(yc, y_lo), y_hi)
        wall_x = float(a[0]) + (deflection_fn(y_contact) if deflection_fn is not None else 0.0)
        dx = xc - wall_x
        dy = yc - y_contact
        dist = math.hypot(dx, dy)
        if dist >= self.cfg.radius:
            return zero, zero_t, y_contact
        if dist < 1e-12:
            normal = torch.tensor([-1.0, 0.0], dtype=torch.float64)
        else:
            normal = torch.tensor([dx / dist, dy / dist], dtype=torch.float64)
        overlap = self.cfg.radius - dist
        info = self.contact.compute_contact_force(self._mirror, None, normal, overlap, self.vel)
        r_vec = torch.tensor([wall_x - xc, y_contact - yc], dtype=torch.float64)
        force = info.total_force
        torque = r_vec[0] * force[1] - r_vec[1] * force[0]
        return force, torque, y_contact

    # -- buoyancy / weight ------------------------------------------------
    def weight_force(self) -> torch.Tensor:
        return torch.tensor([0.0, -self.mass * self.cfg.gravity], dtype=torch.float64)

    def buoyancy_force(self) -> torch.Tensor:
        return torch.tensor([0.0, self.submerged_area() * self.cfg.gravity], dtype=torch.float64)

    # -- integration ------------------------------------------------------
    def integrate(
        self,
        force: torch.Tensor,
        torque: torch.Tensor,
        dt: float,
        external_force: Optional[torch.Tensor] = None,
        external_torque: Optional[torch.Tensor] = None,
    ) -> None:
        """Semi-implicit Euler step (same scheme as :class:`DEMSolver`)."""
        total = force + self.weight_force() + self.buoyancy_force()
        if external_force is not None:
            total = total + external_force
        total_torque = torque + (external_torque if external_torque is not None else 0.0)
        self.vel = self.vel + total / self.mass * dt
        self.pos = self.pos + self.vel * dt
        self.omega = float(self.omega + float(total_torque) / self.inertia * dt)
        self.theta = self.theta + self.omega * dt

    def support_height(self, deflection_fn=None) -> float:
        """Lowest point of the floe (for contact bookkeeping)."""
        self._sync_mirror()
        return float(self.pos[1].item()) - self.cfg.radius