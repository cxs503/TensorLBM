"""Environment field driving interface for ice navigation and marine applications.

This module provides unified input interfaces for:
- Wind field (wind speed/direction profiles)
- Current field (current speed/direction)
- Temperature field

These fields are essential for ice navigation simulations, ship performance
analysis, and environmental loading assessments.

Physical Background
-------------------
1. Wind Profile: Atmospheric boundary layer follows logarithmic law
   u(z) = (u*/κ) * ln(z/z0)
   where u* is friction velocity, κ≈0.41 is von Kármán constant, z0 is roughness length.

2. Current Profile: Ocean currents can be uniform or follow power law
   u(z) = U_ref * (z/z_ref)^(1/n)
   For ice navigation, uniform profile is often assumed for simplicity.

3. Temperature Field: Conservative advection-diffusion
   ∂T/∂t + u·∇T = α∇²T + Q
   Conservative formulation ensures energy conservation.

References
----------
- Garratt, J.R. (1994). The Atmospheric Boundary Layer. Cambridge University Press.
- Dyer, A.J. (1974). A review of flux-profile relationships. Boundary-Layer Meteorology.
- API RP 2PS (2013). Planning, Designing, and Constructing Fixed Offshore Platforms.
"""

from .wind_profile import WindProfile, LogLawWindProfile, PowerLawWindProfile
from .current_profile import CurrentProfile, UniformCurrentProfile, PowerLawCurrentProfile
from .temperature_field import TemperatureField, ConservativeTemperatureField

__all__ = [
    "WindProfile",
    "LogLawWindProfile",
    "PowerLawWindProfile",
    "CurrentProfile",
    "UniformCurrentProfile",
    "PowerLawCurrentProfile",
    "TemperatureField",
    "ConservativeTemperatureField",
]

__version__ = "1.0.0"
