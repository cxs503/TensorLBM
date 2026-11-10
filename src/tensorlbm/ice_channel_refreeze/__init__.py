"""Ice-channel refreezing dynamics (thermal LBM + phase change + forecast).

Modules
-------
``constants``     bulk properties of sea ice / sea water (single frozen source)
``surface``       atmospheric boundary condition ``q_s = h_tot (T_s - T_eff)``
``stefan``        exact Neumann solution, Robin closed form, mixed-layer ODE
``column``        1-D enthalpy lattice-Boltzmann refreezing column
``channel_flow``  2-D D2Q9 + D2Q5 wind-driven circulation / heat transport
``window``        ``T_window(W, h_0, T_air, U_wind)`` lookup table and query
``anchors``       external validation anchors with honest provenance/sha256

Typical use::

    from tensorlbm.ice_channel_refreeze import (
        AtmosphericForcing, RefreezingColumn, simulate_channel, build_table,
    )

    flow = simulate_channel(W=100.0, H=20.0, u_wind=5.0, n_steps=2000)
    col = RefreezingColumn(forcing=AtmosphericForcing(t_air=-20.0, u_wind=5.0),
                           q_ocean=flow.q_ocean_mean)
    col.set_open_water()
    history = col.run_to(10 * 86400.0, sample_every=200)
"""

from .anchors import (  # noqa: F401
    ANCHORS,
    LEWIS_1993_WINDOW,
    ODDEN_1998_THICKNESS,
    WEEKS_ACKLEY_GROWTH_CONSTANT,
    WEEKS_ACKLEY_LEAD_OCEAN_FLUX,
    anchor_table,
)
from .channel_flow import (  # noqa: F401
    ChannelFlowResult,
    physical_ocean_flux,
    simulate_channel,
    validate_flow_solver,
    validate_scalar_solver,
)
from .column import MixedLayerOcean, RefreezingColumn  # noqa: F401
from .constants import DEFAULT_PROPERTIES, IceWaterProperties, freezing_point_C  # noqa: F401
from .stefan import (  # noqa: F401
    GrowthResult,
    degree_day_thickness,
    equilibrium_thickness,
    integrate_growth,
    neumann_lambda,
    neumann_temperature_profile,
    neumann_thickness,
    thickness_with_surface_resistance,
)
from .surface import (  # noqa: F401
    AtmosphericForcing,
    drag_law_current,
    effective_air_temperature,
    ocean_heat_flux,
    sensible_heat_transfer_coefficient,
    surface_heat_flux,
    wind_stress,
)
from .window import (  # noqa: F401
    WindowTable,
    build_table,
    interpolation_error,
    time_to_thickness,
)

__all__ = [
    "ANCHORS",
    "LEWIS_1993_WINDOW",
    "ODDEN_1998_THICKNESS",
    "WEEKS_ACKLEY_GROWTH_CONSTANT",
    "WEEKS_ACKLEY_LEAD_OCEAN_FLUX",
    "anchor_table",
    "ChannelFlowResult",
    "physical_ocean_flux",
    "simulate_channel",
    "validate_flow_solver",
    "validate_scalar_solver",
    "MixedLayerOcean",
    "RefreezingColumn",
    "DEFAULT_PROPERTIES",
    "IceWaterProperties",
    "freezing_point_C",
    "GrowthResult",
    "degree_day_thickness",
    "equilibrium_thickness",
    "integrate_growth",
    "neumann_lambda",
    "neumann_thickness",
    "neumann_temperature_profile",
    "thickness_with_surface_resistance",
    "AtmosphericForcing",
    "drag_law_current",
    "effective_air_temperature",
    "ocean_heat_flux",
    "sensible_heat_transfer_coefficient",
    "surface_heat_flux",
    "wind_stress",
    "WindowTable",
    "build_table",
    "interpolation_error",
    "time_to_thickness",
]