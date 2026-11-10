"""Compressible (shock-capturing) kinetic solver family for TensorLBM.

TensorLBM's lattice-BGK cores are weakly compressible (isothermal DdQq lattices,
Ma <~ 0.3).  The compressible branch lives here and is built on the *kinetic*
(Boltzmann-level) closure rather than on the isothermal equilibrium: the Euler
flux of the solver is the first moment of a local Maxwellian, and the numerical
interface flux is a kinetic/Godunov-type flux split.  Diagnostics are the exact
Rankine-Hugoniot jump relations, so the module carries its own reference
solutions instead of relying on visual plausibility.

Public pieces
-------------
``rankine_hugoniot``
    Exact normal / oblique / moving shock relations, shock-tube initial data
    inversion, Prandtl-Meyer function (gamma=1.4 by default, general gamma).
``flux``
    Numerical fluxes: HLLC (production), Rusanov/LLF (cross-check), and the
    closed-form kinetic flux-vector split (KFVS, gamma-corrected) with a direct
    velocity-space quadrature reference used to verify the closed form.
``solver``
    1-D and 2-D finite-volume MUSCL/SSP-RK solvers on uniform grids.
``cases``
    The three verification cases: normal-shock Mach scan (2..10),
    double Mach reflection (oblique Mach-10 shock at 30 deg to the wall) and the
    Mach-3 forward-facing step.
``analysis``
    Machine-checkable metrics: shock tracking, plateau extraction, Mach-stem /
    triple-point detection, shock stand-off distance, grid-convergence index.
``resolvability``
    Three-state verdict (PASS / UNRESOLVED / WITHHELD) for a stand-off
    measurement: the ``R = delta/dx`` resolvability law, Richardson order /GCI
    from grid triplets, the blockage-corner mass-flux audit and the grid budget
    law.  ``fast2d`` (torch mirror) and ``par2d`` (y-slab process pool) are the
    verified accelerators used for the 1200x400 grid.

Scope and limits (read before trusting a number)
------------------------------------------------
* The solver integrates the compressible Euler equations with a kinetic flux;
  viscous (Navier-Stokes) terms and real-gas effects are out of scope.
* ``gamma`` is fixed (perfect gas, default 1.4); no chemistry/vibration.
* The compressible branch is *not* the DdQq LBM core: continuity with the LBM
  lineage is the Boltzmann-level flux evaluation, documented per function.
"""

from . import (  # noqa: F401
    analysis,
    cases,
    family,
    flux,
    rankine_hugoniot,
    resolvability,
    solver,
    standoff,
)
from .rankine_hugoniot import (  # noqa: F401
    GAMMA,
    moving_shock,
    normal_shock,
    oblique_shock,
    shock_tube_driver,
)
from .solver import EulerSolver1D, EulerSolver2D  # noqa: F401

__all__ = [
    "GAMMA",
    "normal_shock",
    "moving_shock",
    "oblique_shock",
    "shock_tube_driver",
    "EulerSolver1D",
    "EulerSolver2D",
    "flux",
    "solver",
    "cases",
    "analysis",
    "family",
    "standoff",
    "resolvability",
]
