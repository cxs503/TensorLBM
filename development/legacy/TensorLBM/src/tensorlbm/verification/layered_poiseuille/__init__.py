"""Layered plane-Poiseuille flow: exact solution, layered D2Q9 solver, TCLB-port metrics.

Cross-validation case ``XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET`` against the
CFD-GO/TCLB ``developedFlow`` / ``updateMyGlobals`` recipe
(``models/multiphase/d3q27_pf_velocity``).
"""

from .analytic import (LayeredSolution, fit_effective_height, layered_profile,
                       parabolic_profile, pipe_profile, poiseuille_force_from_peak,
                       poiseuille_peak_from_force, solve_layered,
                       wall_normal_coordinates)
from .metrics import (Globals, layer_fluxes, peak_metrics, profile_errors,
                      track_interface_y, update_my_globals)
from .solver import ChannelConfig, LayeredChannel

__all__ = [
    "LayeredSolution", "solve_layered", "parabolic_profile", "pipe_profile",
    "poiseuille_peak_from_force", "poiseuille_force_from_peak",
    "wall_normal_coordinates", "fit_effective_height", "layered_profile",
    "ChannelConfig", "LayeredChannel",
    "Globals", "update_my_globals", "track_interface_y", "profile_errors",
    "layer_fluxes", "peak_metrics",
]
