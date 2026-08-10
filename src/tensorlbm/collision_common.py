"""Common collision and LBM-utility re-export module.

This module provides a single import surface for all collision operators
and LBM utility functions (equilibrium, macroscopic, streaming, mass
correction) so that **worker scripts never need to import directly** from
``d3q19``, ``solver3d``, ``turbulence``, or ``rans_ke``.

Workers should import collision functions and utilities from here::

    from tensorlbm.collision_common import (
        equilibrium3d,
        macroscopic3d,
        collide_mrt3d,
        collide_bgk3d,
        collide_smagorinsky_mrt3d,
        correct_mass3d,
        stream3d,
    )

Collision model selection guide (pass as ``collide_fn`` to
:func:`tensorlbm.lbm_step_correct.lbm_step_correct`):

================  ================================================
Model             collide_fn
================  ================================================
BGK               ``collide_bgk3d``
MRT               ``collide_mrt3d``
MRT + Smagorinsky ``collide_smagorinsky_mrt3d``
RANS k-epsilon    ``collide_rans_ke`` (with ``ke_solver`` kwarg)
================  ================================================

For BFL and wall-function treatments, use the ``wall_treatment`` parameter
of :func:`lbm_step_correct` (``'bfl'`` or ``'wf'``) rather than a different
collision function.
"""

from __future__ import annotations

# ------------------------------------------------------------------ #
#  Lattice utilities (from d3q19)                                    #
# ------------------------------------------------------------------ #
from .d3q19 import equilibrium3d, macroscopic3d  # noqa: F401

# ------------------------------------------------------------------ #
#  Core collision + streaming + mass correction (from solver3d)       #
# ------------------------------------------------------------------ #
from .solver3d import (  # noqa: F401
    collide_bgk3d,
    collide_mrt3d,
    collide_trt3d,
    collide_rlbm3d,
    correct_mass3d,
    stream3d,
)

# ------------------------------------------------------------------ #
#  Smagorinsky LES collisions (from turbulence)                      #
# ------------------------------------------------------------------ #
from .turbulence import (  # noqa: F401
    collide_smagorinsky_bgk3d,
    collide_smagorinsky_mrt3d,
)

# ------------------------------------------------------------------ #
#  RANS k-epsilon collision (from rans_ke)                           #
# ------------------------------------------------------------------ #
try:
    from .rans_ke import KESolver, collide_rans_ke, C_MU, C_E1, C_E2  # noqa: F401
except ImportError:  # pragma: no cover
    pass

# ------------------------------------------------------------------ #
#  RANS common collisions (from rans_common)                          #
# ------------------------------------------------------------------ #
try:
    from .rans_common import collide_rans_bgk3d, collide_rans_mrt3d  # noqa: F401
except ImportError:  # pragma: no cover
    pass


__all__ = [
    # Lattice utilities
    "equilibrium3d",
    "macroscopic3d",
    # Core collisions
    "collide_bgk3d",
    "collide_mrt3d",
    "collide_trt3d",
    "collide_rlbm3d",
    # Streaming + mass correction
    "stream3d",
    "correct_mass3d",
    # Smagorinsky LES
    "collide_smagorinsky_bgk3d",
    "collide_smagorinsky_mrt3d",
    # RANS k-epsilon
    "collide_rans_ke",
    "KESolver",
    "C_MU",
    "C_E1",
    "C_E2",
    # RANS common
    "collide_rans_bgk3d",
    "collide_rans_mrt3d",
]
