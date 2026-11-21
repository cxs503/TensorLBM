"""Ice-cover hydroelastic breakup verification package (p338w3-n03).

Components
----------
analytical        : elastic-plate rigidity / flexural-gravity dispersion /
                    critical-velocity criteria (prescribed k^2 form + first-principles form)
lbm_channel       : D2Q9 channel LBM (no-slip Poiseuille verification + wavy free-slip
                    cover pressure response)
bonded_cover      : bonded-particle (lattice DEM) ice cover with breakable bonds,
                    foundation springs, membrane load and transient crack detection
coupled           : LBM-calibrated pressure -> DEM load coupling and critical-velocity sweep
"""

from .analytical import (  # noqa: F401
    CoverProperties,
    plate_rigidity,
    characteristic_length,
    flexural_gravity_omega,
    phase_speed,
    critical_velocity_k2,
    u_cr_k2,
    u_cr_flexural_gravity,
    steady_divergence_velocity,
    mode_wavenumbers,
    momentum_flux_compression,
    crack_moment_capacity,
    critical_mode,
    antinode_positions,
)
from .lbm_channel import (  # noqa: F401
    poiseuille_channel,
    wavy_cover_pressure_gain,
)
from .bonded_cover import BondedCover  # noqa: F401
from .coupled import IceCoverBreakupStudy  # noqa: F401