"""Linear stability and dispersion-dissipation analysis for lattice Boltzmann.

This subpackage turns the "empirical tuning" of LBM relaxation parameters into a
*measurable* property: given a collision model (BGK / MRT) and its parameters
(``tau``, base velocity ``u0`` = ``Ma * cs``) and a wave vector, it predicts the
per-timestep growth of every Fourier mode of the linearised system (von Neumann
analysis) and measures that growth with an independent, explicitly integrated
periodic-box experiment.  The two sides are independently implemented and are
cross-audited point by point (see :mod:`tensorlbm.stability.von_neumann`).

Public surface
--------------
:class:`Collision`          collision-model descriptor (kind / tau / MRT rates)
:func:`equilibrium`         D2Q9 equilibrium (NumPy, float64)
:func:`equilibrium_jacobian` ``dfeq/df`` at a uniform base state
:func:`amplification_matrix` von Neumann amplification matrix ``A(theta)``
:class:`Spectrum`            eigenvalues / growth rates / phase speeds / labels
:func:`spectrum`             eigen-decomposition of ``A(theta)``
:func:`run_modal_box`        periodic-box single-mode experiment (measured side)
:func:`numerical_viscosity`  ``nu_num(theta)`` from the shear-mode attenuation
"""

from __future__ import annotations

from .channel_shear import (
    ChannelFlow,
    LatticeModeSpectrum,
    channel_mode_operator,
    lattice_mode_spectrum,
    local_jacobian,
    stokes_damping_benchmark,
    time_march_growth_rate,
)
from .orr_sommerfeld import OSSpectrum, os_growth_rate, os_spectrum, orszag_benchmark_check
from .von_neumann import (
    CVEL,
    CX,
    CY,
    MOMENT_ORDER,
    Collision,
    Spectrum,
    W,
    amplification_matrix,
    collision_jacobian,
    equilibrium,
    equilibrium_jacobian,
    numerical_viscosity,
    run_modal_box,
    spectral_radius,
    spectrum,
    stability_point,
)

__all__ = [
    "CVEL",
    "ChannelFlow",
    "LatticeModeSpectrum",
    "OSSpectrum",
    "channel_mode_operator",
    "lattice_mode_spectrum",
    "local_jacobian",
    "os_growth_rate",
    "os_spectrum",
    "orszag_benchmark_check",
    "stokes_damping_benchmark",
    "time_march_growth_rate",
    "CX",
    "CY",
    "MOMENT_ORDER",
    "W",
    "Collision",
    "Spectrum",
    "amplification_matrix",
    "collision_jacobian",
    "equilibrium",
    "equilibrium_jacobian",
    "numerical_viscosity",
    "run_modal_box",
    "spectral_radius",
    "spectrum",
    "stability_point",
]
