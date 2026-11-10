"""Modal Euler-Bernoulli beam for the FSI leg of the three-way coupling.

The structure is a vertical cantilever (the DEM-FSI target: ice-floe impacts
land on its wetted face near the still-water line).  Deflection is horizontal,
``w(y, t) = sum_n q_n(t) phi_n(y)``, with the exact cantilever eigenfunctions.
Bending stress follows from the curvature,

    ``sigma(y, t) = E * (h/2) * w''(y, t)``,

so the mid-span and root whipping stress time histories are direct modal
outputs.

Loads:

* distributed hydrodynamic pressure on the wetted face → ``Q_n = ∫ p phi_n dy``,
* discrete DEM contact impulse → ``Q_n += F phi_n(y_c)``,
* the structural deformation ``w(y_c)`` is fed back into the DEM contact
  geometry (see :meth:`tensorlbm.three_way.floe.Floe2D.contact_with_segment`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch

#: first five cantilever eigenvalues beta_n * L
BETA_L = (1.875104068711961, 4.694091132974174, 7.854757438237613, 10.995540734875467, 14.137168391046470)


def _sigma_n(beta: float) -> float:
    return (math.cosh(beta) + math.cos(beta)) / (math.sinh(beta) + math.sin(beta))


def mode_shape(n: int, xi: torch.Tensor) -> torch.Tensor:
    """Cantilever eigenfunction ``phi_n`` at normalised ``xi = y/L``."""
    beta = BETA_L[n]
    s = _sigma_n(beta)
    return (
        torch.cosh(beta * xi)
        - torch.cos(beta * xi)
        - s * (torch.sinh(beta * xi) - torch.sin(beta * xi))
    )


def mode_curvature(n: int, xi: torch.Tensor, length: float) -> torch.Tensor:
    """``d^2 phi_n / dy^2`` (units 1/length^2)."""
    beta = BETA_L[n]
    s = _sigma_n(beta)
    return (beta / length) ** 2 * (
        torch.cosh(beta * xi)
        + torch.cos(beta * xi)
        - s * (torch.sinh(beta * xi) + torch.sin(beta * xi))
    )


@dataclass
class BeamConfig:
    """Structural parameters in lattice units."""

    youngs_modulus: float = 2.0e5
    density: float = 15.0
    length: float = 34.0
    thickness: float = 2.0  # in-plane bending depth (x-width)
    depth: float = 1.0  # out-of-plane thickness
    damping_ratio: float = 0.02
    n_modes: int = 2
    x_base: float = 70.0  # x of the clamped face
    y_base: float = 0.0  # clamped end height, beam spans y_base..y_base+length
    yield_stress: float = 300.0


class CantileverBeam:
    """Modal cantilever beam (Euler-Bernoulli) with explicit time stepping."""

    def __init__(self, cfg: BeamConfig):
        self.cfg = cfg
        self.area = cfg.thickness * cfg.depth
        self.I = cfg.depth * cfg.thickness ** 3 / 12.0
        n = cfg.n_modes
        self.q = torch.zeros(n, dtype=torch.float64)
        self.qdot = torch.zeros(n, dtype=torch.float64)
        # modal mass / stiffness from numerical quadrature of the eigenfunctions
        xi = torch.linspace(0.0, 1.0, 401, dtype=torch.float64)
        self._xi = xi
        self.modal_mass = torch.zeros(n, dtype=torch.float64)
        self.modal_stiffness = torch.zeros(n, dtype=torch.float64)
        self.omega = torch.zeros(n, dtype=torch.float64)
        for i in range(n):
            phi = mode_shape(i, xi)
            phipp = mode_curvature(i, xi, cfg.length)
            self.modal_mass[i] = cfg.density * self.area * torch.trapz(phi * phi, xi) * cfg.length
            self.modal_stiffness[i] = cfg.youngs_modulus * self.I * torch.trapz(phipp * phipp, xi) * cfg.length
            self.omega[i] = math.sqrt(float(self.modal_stiffness[i] / self.modal_mass[i]))
        self.enable_feedback = True

    # -- loads ------------------------------------------------------------
    def projected_generalized_force(self, y: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
        """``Q_n = ∫ f(y) phi_n(y) dy`` from discrete loads ``load`` at ``y``."""
        xi = (y - self.cfg.y_base) / self.cfg.length
        xi = xi.clamp(0.0, 1.0)
        out = torch.zeros(self.cfg.n_modes, dtype=torch.float64)
        for i in range(self.cfg.n_modes):
            out[i] = (load * mode_shape(i, xi)).sum()
        return out

    def generalized_force_from_pressure(
        self, y_cells: torch.Tensor, pressure: torch.Tensor, dy: float = 1.0
    ) -> torch.Tensor:
        """Modal force from a pressure distribution on the wetted face."""
        return self.projected_generalized_force(y_cells, pressure * dy)

    # -- time integration -------------------------------------------------
    def step(self, generalized_force: torch.Tensor, dt: float) -> None:
        """One damped modal step (semi-implicit Euler, sub-stepped for stability).

        The sub-step count keeps ``omega_max * dt_sub`` below 0.05, so higher
        modes cannot destabilise the explicit integrator.
        """
        if not self.enable_feedback:
            # negative-control path: no structural response at all
            self.q.zero_()
            self.qdot.zero_()
            return
        wmax = float(self.omega.max())
        n_sub = max(1, int(math.ceil(wmax * dt / 0.05)))
        h = dt / n_sub
        for _ in range(n_sub):
            for i in range(self.cfg.n_modes):
                wn = float(self.omega[i])
                zeta = self.cfg.damping_ratio
                # q'' + 2 zeta omega q' + omega^2 q = Q / M
                acc = (
                    float(generalized_force[i]) / float(self.modal_mass[i])
                    - 2.0 * zeta * wn * float(self.qdot[i])
                    - wn * wn * float(self.q[i])
                )
                self.qdot[i] = self.qdot[i] + acc * h
                self.q[i] = self.q[i] + float(self.qdot[i]) * h

    # -- outputs ----------------------------------------------------------
    def deflection(self, y: float) -> float:
        xi = torch.tensor([(y - self.cfg.y_base) / self.cfg.length], dtype=torch.float64).clamp(0.0, 1.0)
        w = 0.0
        for i in range(self.cfg.n_modes):
            w += float(self.q[i]) * float(mode_shape(i, xi)[0])
        return w

    def curvature(self, y: float) -> float:
        xi = torch.tensor([(y - self.cfg.y_base) / self.cfg.length], dtype=torch.float64).clamp(0.0, 1.0)
        k = 0.0
        for i in range(self.cfg.n_modes):
            k += float(self.q[i]) * float(mode_curvature(i, xi, self.cfg.length)[0])
        return k

    def stress(self, y: float) -> float:
        """Bending stress at height ``y`` (``E * h/2 * w''``)."""
        return self.cfg.youngs_modulus * (self.cfg.thickness / 2.0) * self.curvature(y)

    def strain_energy(self) -> float:
        return float(0.5 * (self.modal_stiffness * self.q * self.q).sum())

    def kinetic_energy(self) -> float:
        return float(0.5 * (self.modal_mass * self.qdot * self.qdot).sum())

    def total_energy(self) -> float:
        return self.strain_energy() + self.kinetic_energy()


def beam_modal_force_reference(
    y_cells: torch.Tensor,
    pressure: torch.Tensor,
    cfg: BeamConfig,
    mode: int = 0,
) -> float:
    """Reference modal force by direct quadrature (used by gate (b))."""
    xi = ((y_cells - cfg.y_base) / cfg.length).clamp(0.0, 1.0)
    phi = mode_shape(mode, xi)
    return float((pressure * phi).sum().item())