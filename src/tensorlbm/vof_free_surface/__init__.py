"""Conservative VOF free-surface solver and its three-tier verification ladder.

The package provides

* :class:`VOFConfig` / :class:`VOFChannelFlow` - a staggered (MAC) incompressible
  two-phase solver with CICSAM interface transport, variable-density pressure
  projection and continuum-surface-force surface tension;
* :mod:`tensorlbm.vof_free_surface.ladder` - the L1 (Ritter dam break),
  L2 (second-order Stokes wave) and L3 (Lamb capillary-wave damping) anchors.

Every metric returned by the ladder is computed from the simulated fields;
no value is hard-coded.
"""

from .solver import VOFChannelFlow, VOFConfig, still_water_level
from .ladder import (
    capillary_damping_lamb,
    run_capillary_damping,
    run_dam_break_ritter,
    run_stokes2_wave,
    stokes2_free_surface,
)

__all__ = [
    "VOFConfig",
    "VOFChannelFlow",
    "capillary_damping_lamb",
    "run_capillary_damping",
    "run_dam_break_ritter",
    "run_stokes2_wave",
    "still_water_level",
    "stokes2_free_surface",
]