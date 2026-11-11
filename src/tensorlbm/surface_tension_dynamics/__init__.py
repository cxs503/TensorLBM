"""Surface-tension *dynamics* verification surface (capillary waves / Marangoni / coalescence).

Why a second module next to :mod:`tensorlbm.free_surface_tension`
------------------------------------------------------------------
``free_surface_tension`` established the **statics** of the CSF operator
(Young-Laplace pressure jumps, curvature accuracy, parasitic currents) plus one
linear Marangoni anchor.  A free-surface solver used for ship hydrodynamics,
air layers, bubbles and ice/water fronts lives or dies on the **dynamics** of the
same coefficient: the frequency with which the numerical interface answers a
curvature perturbation, the tangential (Marangoni) coupling of that interface to
the bulk, and the capillary time scale of a topological event (drop
coalescence).  Those three quantities are what this package makes
machine-decidable:

* :mod:`.capillary_wave` -- standing capillary wave on a flat interface,
  ``omega(k)`` vs ``omega^2 = sigma k^3 / (rho_l + rho_g)`` and the viscous decay
  ``gamma = 2 nu k^2``, with a zero-surface-tension negative control.
* :mod:`.marangoni` -- interfacial shear driven channel flow, the whole
  velocity *profile* against the exact Stokes two-layer (Couette) solution.
* :mod:`.coalescence` -- two-dimensional drop pair merging: neck growth law,
  regime classification (Ohnesorge) and the dimensionless coalescence time.

The time stepping is the existing
:class:`tensorlbm.free_surface_tension.solver.CapillarySolver` (LBGK + Guo
forcing + CSF force); nothing here re-implements a solver, only initial
conditions and measurement chains.  See
``docs/SURFACE_TENSION_DYNAMICS_P152W1_N01.md`` for the report.
"""

from __future__ import annotations

from .capillary_wave import (
    WaveConfig,
    damped_fit_windows,
    interface_sharpness,
    run_dispersion_scan,
    run_standing_wave,
    standing_wave_phi,
    zero_crossing_period,
)
from .coalescence import (
    CoalescenceConfig,
    two_drop_phi,
    run_coalescence,
)
from .marangoni import (
    MarangoniConfig,
    analytic_couette_profile,
    run_marangoni_channel,
)

__all__ = [
    "CoalescenceConfig",
    "MarangoniConfig",
    "WaveConfig",
    "analytic_couette_profile",
    "damped_fit_windows",
    "interface_sharpness",
    "run_coalescence",
    "run_dispersion_scan",
    "run_marangoni_channel",
    "run_standing_wave",
    "standing_wave_phi",
    "two_drop_phi",
    "zero_crossing_period",
]