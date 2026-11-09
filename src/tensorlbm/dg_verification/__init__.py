"""DG convection/diffusion verification ladder (campaign node IN70-N01).

This sub-package is the *verification face* of the discontinuous-Galerkin
effort in TensorLBM.  It is deliberately independent of the LBM coupling code
(``tensorlbm.dg_advection`` / ``dg_lbm``): the kernels here solve textbook
conservation laws (scalar advection-diffusion, 1-D Euler) on periodic Cartesian
grids so the discretisation itself can be checked against analytic /
manufactured / exact-Riemann references with measurable convergence orders.

Modules
-------
``element``    nodal Gauss-Lobatto reference element: mass/stiffness matrices,
               face selection matrices and quadrature (1-D building blocks).
``limiter``    Legendre modal <-> nodal transforms and the Cockburn-Shu TVB /
               minmod slope limiter (conservative: the cell mean is untouched).
``scalar``     2-D scalar advection (variable coefficient, upwind / Rusanov
               flux) and LDG diffusion, SSP-RK3 time stepping, mass ledger.
``euler1d``    1-D compressible Euler (ideal gas `gamma` and isothermal
               `p = rho c_s^2`) DG with a local Lax-Friedrichs flux, plus the
               exact Riemann solvers used as the non-circular reference.
``benchmarks`` the five acceptance benchmarks of IN70-N01 (rotating cone, MMS,
               Sod shock tube + LBM cross-check, square-wave TVD, mass ledger).
"""

from __future__ import annotations

from .element import RefElement, lobatto_nodes, ref_element
from .limiter import cell_means, minmod, modal_to_nodal, nodal_to_modal, tvb_limit
from .scalar import (
    advection_rhs_2d,
    cell_mean_2d,
    convection_diffusion_rhs_2d,
    diffusion_rhs_2d,
    integral_2d,
    l2_error_vs,
    mass_ratio_drift,
    node_coords_2d,
    ssp_rk3,
)
from .stability import operator_spectra_1d, spectral_report, stable_dt
from .euler1d import (
    EulerDG1D,
    exact_isothermal_riemann,
    exact_sod,
    rarefaction_tail,
    shock_position,
)

__all__ = [
    "RefElement",
    "lobatto_nodes",
    "ref_element",
    "minmod",
    "nodal_to_modal",
    "modal_to_nodal",
    "tvb_limit",
    "cell_means",
    "advection_rhs_2d",
    "diffusion_rhs_2d",
    "convection_diffusion_rhs_2d",
    "ssp_rk3",
    "integral_2d",
    "cell_mean_2d",
    "mass_ratio_drift",
    "node_coords_2d",
    "l2_error_vs",
    "EulerDG1D",
    "exact_sod",
    "exact_isothermal_riemann",
    "rarefaction_tail",
    "shock_position",
    "stable_dt",
    "operator_spectra_1d",
    "spectral_report",
]