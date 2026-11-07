"""Conjugate-heat-transfer / DDF thermal validation package (IN28-N03).

Public surface::

    from tensorlbm.conjugate_thermal import (
        DDF2D, Conduction2D,            # solver core
        cavity_natural_convection,      # case (b)
        cylinder_forced_convection,     # case (a)
        conjugate_channel,              # cases (d) + (e)
        rayleigh_growth_rate,           # case (c)
    )
"""

from .ddf2d import (
    C5,
    C9,
    OPP5,
    OPP9,
    W5,
    W9,
    Conduction2D,
    DDF2D,
    bilinear_sample,
    lagrange_derivative_weights,
    shifted,
)
from .cases import (
    cavity_natural_convection,
    channel_nusselt_fully_developed,
    churchill_bernstein,
    conjugate_channel,
    cylinder_forced_convection,
    de_vahl_davis_nu,
    nu_cylinder_polar,
    rayleigh_growth_rate,
)

__all__ = [
    "C5", "C9", "W5", "W9", "OPP5", "OPP9",
    "DDF2D", "Conduction2D", "bilinear_sample", "lagrange_derivative_weights",
    "shifted",
    "cavity_natural_convection", "cylinder_forced_convection",
    "conjugate_channel", "rayleigh_growth_rate",
    "nu_cylinder_polar", "churchill_bernstein", "de_vahl_davis_nu",
    "channel_nusselt_fully_developed",
]
