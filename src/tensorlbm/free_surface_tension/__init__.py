"""Free-surface surface tension (CSF) model with machine-checkable physics anchors.

Public API
----------
* :mod:`.csf` -- geometry operators (normal, ``delta_s``, curvature), the CSF
  force ``f = sigma kappa grad(phi)``, the tangential Marangoni force and the
  analytic anchors (Young-Laplace, Rayleigh period).
* :mod:`.solver` -- a compact LBGK + Guo-forcing solver used to demonstrate the
  anchors (``CapillarySolver``).
* :mod:`.verification` -- the acceptance-case drivers (Young-Laplace 2-D/3-D,
  droplet oscillation, Marangoni sweep, spurious-current convergence, mass
  conservation).

See ``docs/FREE_SURFACE_SURFACE_TENSION_CSF.md`` for the report.
"""

from __future__ import annotations

from .csf import (
    csf_force,
    curvature,
    divergence,
    gradient,
    interface_delta,
    interface_normal,
    laplace_pressure,
    marangoni_force,
    rayleigh_period,
    tanh_droplet_phi,
    tanh_ellipse_phi,
    tanh_flat_phi,
    tanh_sphere_phi,
    total_force,
)
from .lattice import CS2, Lattice, get_lattice
from .solver import CapillarySolver, SolverConfig

__all__ = [
    "CS2",
    "CapillarySolver",
    "Lattice",
    "SolverConfig",
    "csf_force",
    "curvature",
    "divergence",
    "get_lattice",
    "gradient",
    "interface_delta",
    "interface_normal",
    "laplace_pressure",
    "marangoni_force",
    "rayleigh_period",
    "tanh_droplet_phi",
    "tanh_ellipse_phi",
    "tanh_flat_phi",
    "tanh_sphere_phi",
    "total_force",
]