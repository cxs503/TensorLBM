"""Discrete Element Method (DEM) for ice particle simulations.

This module provides a simplified DEM solver for spherical ice particles
with translation and rotation dynamics, contact mechanics, and coupling
to LBM fluid solvers via immersed boundary method (IBM).

References
----------
- Cundall & Strack (1979). A discrete numerical model for granular assemblies.
  Géotechnique 29(4), 47-65.
- Tsuji et al. (1993). Lagrangian numerical simulation of plug flow of
  cohesionless particles in a horizontal pipe. Powder Technology 71, 239-250.
"""

from .particle import IceParticle, ParticleSystem
from .contact import ContactModel, LinearSpringDashpot
from .dem_solver import DEMSolver, DEMConfig

__all__ = [
    "IceParticle",
    "ParticleSystem",
    "ContactModel",
    "LinearSpringDashpot",
    "DEMSolver",
    "DEMConfig",
]
