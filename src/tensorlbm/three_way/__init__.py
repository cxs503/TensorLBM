"""Three-way LBM–DEM–FSI coupling for TensorLBM.

Modules
-------
``core2d``   compact D2Q9 solver with a Guo force field and non-equilibrium
             stress / dissipation diagnostics,
``ibm2d``    immersed-boundary interpolation, spreading and traction integrals,
``floe``     buoyant ice-floe DEM body (spring-dashpot contacts from ``dem``),
``beam``     modal Euler-Bernoulli cantilever (FSI leg),
``scenario`` end-to-end floe-slam / whipping scenario,
``gates``    the machine-checkable acceptance gates (a)–(h).
"""

from .beam import BeamConfig, CantileverBeam
from .core2d import Flow2D
from .floe import Floe2D, FloeConfig, draft_for_ratio, submerged_area
from .scenario import ThreeWayConfig, ThreeWaySimulation

__all__ = [
    "BeamConfig",
    "CantileverBeam",
    "Flow2D",
    "Floe2D",
    "FloeConfig",
    "submerged_area",
    "draft_for_ratio",
    "ThreeWayConfig",
    "ThreeWaySimulation",
]
