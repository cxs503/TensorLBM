"""Multiphase / non-Newtonian frontier validation surface (campaign p40w2-n02).

Three coupled validation legs, each with an *independent* reference:

(a) :mod:`tensorlbm.multiphase_frontier.viv` -- vortex-induced vibration (VIV)
    lock-in of an elastically mounted circular cylinder.  The solver is a
    two-relaxation-time (TRT) D2Q9 lattice-Boltzmann fluid with halfway
    bounce-back on a *fixed* staircase cylinder, plus a 1-DOF spring-mass
    damper.  The relative cylinder motion is handled by the **accelerating
    frame** transformation: the far-field velocity carries ``-dx_s/dt`` and a
    uniform pseudo-acceleration ``-d2x_s/dt2`` is applied to the fluid.  This
    removes the moving-mesh / fresh-node problem that produced the inflated
    loads recorded for ``fsi_viv_ibm`` (``Cd = 32.3`` vs ``1.34`` at
    ``Re = 100``), because the geometry is fixed and the momentum-exchange
    force is free of staircase aliasing.

(b) :mod:`tensorlbm.multiphase_frontier.cavitation_dynamics` -- bubble
    dynamics kernel: Rayleigh-Plesset (with gas content and surface tension)
    integrated against the exact Rayleigh collapse time / Minnaert frequency,
    and the Schnerr-Sauer vapour source term checked for consistency with the
    same bubble radius evolution.

(c) :mod:`tensorlbm.multiphase_frontier.poiseuille` -- generalised Newtonian
    pressure-driven flow for power-law / Carreau-Yasuda / Cross fluids, LBM
    (channel) and radial finite-volume (pipe) against the exact quadrature
    solution of the same constitutive law.

Every module returns plain dict/np.ndarray data so that the driver script can
inject numbers into the report without any hand copying.
"""

from __future__ import annotations

from .rheology import (
    CarreauYasuda,
    Cross,
    PowerLaw,
    make_model,
    poiseuille_channel_profile,
    poiseuille_pipe_profile,
)
from .poiseuille import (
    channel_finite_volume,
    channel_lbm_generalized_newtonian,
    pipe_finite_volume,
    profile_error,
)
from .cavitation_dynamics import (
    RayleighPlesset,
    minnaert_frequency,
    rayleigh_collapse_time,
    schnerr_sauer_source,
)
from .viv import (
    VIVFrameSolver,
    VivResult,
    elastic_mount_response,
    dominant_frequency,
    free_decay_verlet,
    period_from_peaks,
    stationary_strouhal,
)

__all__ = [
    "PowerLaw",
    "CarreauYasuda",
    "Cross",
    "make_model",
    "poiseuille_channel_profile",
    "poiseuille_pipe_profile",
    "channel_lbm_generalized_newtonian",
    "channel_finite_volume",
    "pipe_finite_volume",
    "profile_error",
    "RayleighPlesset",
    "minnaert_frequency",
    "rayleigh_collapse_time",
    "schnerr_sauer_source",
    "VIVFrameSolver",
    "VivResult",
    "elastic_mount_response",
    "dominant_frequency",
    "free_decay_verlet",
    "period_from_peaks",
    "stationary_strouhal",
]