"""Mesh-orientation and Galilean invariance verification surface.

Public entry points
-------------------
``D2Q9Solver``          minimal D2Q9 solver with SDF-interpolated bounce-back
``tilted_channel``      plane channel at arbitrary inclination (true SDF)
``circle``              circular cylinder (rotationally symmetric body)
``poiseuille_case``     criterion C1 (observed order on inclined walls)
``cylinder_case``       criteria C2/C3/C6 (Cd / St orientation spread)
``taylor_green_case``   criteria C4/C5 (Galilean residual and Ma^2 law)
``evaluate_criteria``   turns measured numbers into pass/fail (C1..C7)

Design rule: geometry is described *analytically* and rotated by ``theta``;
the lattice never changes.  Because the physical problem is rotation
invariant, every ``theta`` describes the same flow and the spread of the
measured quantities over ``theta`` is a direct measure of lattice anisotropy
plus boundary-treatment error.
"""

from .acceptance import evaluate_criteria
from .cases import (
    cylinder_case,
    galilean_superposition_residual,
    observed_order,
    poiseuille_case,
    taylor_green_case,
    tg_exact,
)
from .geometry import (
    Geometry2D,
    channel_boundary_cells,
    circle,
    poiseuille_profile,
    rotated_unit,
    tilted_channel,
)
from .solver import CS2, CS, D2Q9Solver, VelocityDirichlet, equilibrium

__all__ = [
    "CS",
    "CS2",
    "D2Q9Solver",
    "Geometry2D",
    "VelocityDirichlet",
    "channel_boundary_cells",
    "circle",
    "cylinder_case",
    "equilibrium",
    "evaluate_criteria",
    "galilean_superposition_residual",
    "observed_order",
    "poiseuille_case",
    "poiseuille_profile",
    "rotated_unit",
    "taylor_green_case",
    "tg_exact",
    "tilted_channel",
]