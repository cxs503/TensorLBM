"""Non-Newtonian rheology verification surface (TensorLBM P40W4-N04).

Three generalised-Newtonian closures (power law, Carreau–Yasuda, Cross) plus a
Papanastasiou-regularised Herschel–Bulkley (ice-slurry) closure, driven through
a variable-viscosity D2Q9 plane-channel solver, each checked against an
independent one-dimensional anchor of the *same* constitutive law:

* closed-form Ostwald–de Waele profile (power law),
* constitutive inversion + quadrature (Carreau–Yasuda, Cross, Herschel–Bulkley),
* the rheology-independent stress field ``τ = ρ a |y|`` for the plug boundary.

Modules
-------
:mod:`~tensorlbm.nonnewtonian_frontier.closures`   constitutive laws
:mod:`~tensorlbm.nonnewtonian_frontier.reference`  1-D anchors
:mod:`~tensorlbm.nonnewtonian_frontier.channel`    variable-viscosity D2Q9 runner
:mod:`~tensorlbm.nonnewtonian_frontier.plug`       yield-surface detectors
"""

from .channel import (
    bounce_back_prestream_2d,
    channel_geometry,
    l2_relative_error,
    run_channel,
    shear_rate_profile,
    solution_residual,
    wall_nonequilibrium_2d,
)
from .closures import (
    Closure,
    carreau_yasuda,
    cross,
    herschel_bulkley,
    newtonian,
    power_law,
)
from .plug import (
    plug_half_width_fit,
    plug_half_width_ideal_from_closure,
    plug_half_width_stress,
    plug_half_width_velocity,
    plug_report,
    symmetrise,
)
from .reference import (
    channel_reference,
    gamma_of_tau,
    plug_half_width_ideal,
    power_law_velocity,
    power_law_wall_shear_rate,
)

__all__ = [
    "Closure",
    "newtonian",
    "power_law",
    "carreau_yasuda",
    "cross",
    "herschel_bulkley",
    "channel_reference",
    "gamma_of_tau",
    "power_law_velocity",
    "power_law_wall_shear_rate",
    "plug_half_width_ideal",
    "run_channel",
    "channel_geometry",
    "shear_rate_profile",
    "l2_relative_error",
    "plug_report",
    "plug_half_width_fit",
    "plug_half_width_velocity",
    "plug_half_width_stress",
    "plug_half_width_ideal_from_closure",
    "symmetrise",
    "wall_nonequilibrium_2d",
    "bounce_back_prestream_2d",
    "solution_residual",
]