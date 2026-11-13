"""Two-phase wetting verification ladder (IN32-L01).

Physics: gas-liquid-solid wetting on top of the D2Q9 CAC phase-field solver
(:mod:`tensorlbm.cac_lbm2d`).  Four analytic anchors plus two gates:

* Young-Laplace pressure jump  ``dp = sigma / R``          (2D drop)
* Young's law static angle     ``sigma_lg cos(theta) = sigma_sg - sigma_sl``
* Jurin capillary rise         ``h = 2 sigma cos(theta) / (drho g w)``
* Lamb/Rayleigh drop oscillation ``omega^2 = m(m^2-1) sigma / ((rho_l+rho_g) R^3)``
* Cox-Voinov dynamic angle     ``theta_d^3 = theta_e^3 + 9 Ca ln(L/Ls)``
* spurious-current gate        ``max|u| < 1e-4 U_ref``

Measurement operators live in :mod:`tensorlbm.wetting.metrics`; the campaign
driver is ``benchmarks/in32_l01_wetting_ladder`` and the acceptance report is
``docs/IN32_L01_WETTING_LADDER.md``.
"""

from . import metrics
from .metrics import (
    contact_angle_circle_fit,
    cox_voinov_fit,
    damped_sine_fit,
    flat_interface_tension,
    laplace_drop,
    meniscus_height,
    radial_extents,
    spherical_cap_inversion,
)

__all__ = [
    "metrics",
    "contact_angle_circle_fit",
    "cox_voinov_fit",
    "damped_sine_fit",
    "flat_interface_tension",
    "laplace_drop",
    "meniscus_height",
    "radial_extents",
    "spherical_cap_inversion",
]

from .wall_order_parameter_2d import (
    WallOrderWettingConfig,
    analytic_contact_angle,
    contact_angle_from_field,
    free_energy_parameters,
    initialise_drop_on_wall,
    interface_points,
    macroscopics,
    relax_to_equilibrium,
    model_surface_tension,
    run_to_equilibrium,
    step as wall_order_step,
)

__all__ = list(__all__) + [
    "WallOrderWettingConfig",
    "analytic_contact_angle",
    "contact_angle_from_field",
    "free_energy_parameters",
    "initialise_drop_on_wall",
    "interface_points",
    "macroscopics",
    "relax_to_equilibrium",
    "model_surface_tension",
    "run_to_equilibrium",
    "wall_order_step",
]
