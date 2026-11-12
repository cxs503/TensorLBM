"""Three-way LBM–DEM–FSI coupling for TensorLBM.

Modules
-------
``core2d``   compact D2Q9 solver with a Guo force field and non-equilibrium
             stress / dissipation diagnostics,
``ibm2d``    immersed-boundary interpolation, spreading and traction integrals,
``floe``     buoyant ice-floe DEM body (spring-dashpot contacts from ``dem``),
``beam``     modal Euler-Bernoulli cantilever (FSI leg),
``scenario`` end-to-end floe-slam / whipping scenario,
``gates``    the machine-checkable acceptance gates (a)–(h),
``interface3d``  3-D surface/particle load-exchange interface (Q1 hex, Hermite beam),
``coupled3d``    end-to-end 3-D three-way chain (LBM+IBM water, beam FEM, DEM ice).
"""

from .beam import BeamConfig, CantileverBeam
from .coupled3d import (
    Coupled3DConfig,
    Fluid3DIBM,
    IceDEM3D,
    clift_cd,
    hull_girder_from_contact_loads,
    run_coupled3d,
    schiller_naumann_cd,
    tip_deflection_reference,
)
from .interface3d import HexGrid3D, Surface3D, exchange_cycle, project_to_beam
from .core2d import Flow2D
from .floe import Floe2D, FloeConfig, draft_for_ratio, submerged_area
from .scenario import ThreeWayConfig, ThreeWaySimulation

__all__ = [
    "Coupled3DConfig",
    "Fluid3DIBM",
    "HexGrid3D",
    "IceDEM3D",
    "Surface3D",
    "clift_cd",
    "exchange_cycle",
    "hull_girder_from_contact_loads",
    "project_to_beam",
    "run_coupled3d",
    "schiller_naumann_cd",
    "tip_deflection_reference",
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
