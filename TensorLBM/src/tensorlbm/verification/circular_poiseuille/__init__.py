"""Circular Poiseuille flow verification for TensorLBM.

Analytical solution for fully developed laminar flow in a circular pipe:
    u(r) = u_max * (1 - (r/R)^2)
    u_max = (dp/dx) * R^2 / (4 * mu)
    Q = pi * R^4 * (dp/dx) / (8 * mu)

This case validates curved boundary treatments (Inamuro wet-node vs standard bounce-back)
against the analytical parabolic profile.

Reference: White, F.M. (2011). "Fluid Mechanics", 7th ed., McGraw-Hill.
"""

from .solver import CircularPoiseuilleConfig, CircularPoiseuilleSolver, run_circular_poiseuille
from .analytical import (
    poiseuille_velocity_profile,
    poiseuille_max_velocity,
    poiseuille_flow_rate,
)

__all__ = [
    "CircularPoiseuilleConfig",
    "CircularPoiseuilleSolver",
    "run_circular_poiseuille",
    "poiseuille_velocity_profile",
    "poiseuille_max_velocity",
    "poiseuille_flow_rate",
]
