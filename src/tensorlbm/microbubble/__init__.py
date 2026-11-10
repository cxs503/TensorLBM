"""Microbubble drag reduction: two-phase LBM + void transport + net-power ledger.

Campaign node **IN62-N04** (TensorLBM).  The module closes the chain

    gas injection  ->  wall void fraction alpha_w  ->  void profile alpha(y)
    ->  friction drag reduction DR  ->  net power P_net = P_saved - P_comp

Modules
-------
``loglaw``
    The *analytic anchor*: modified log law
    ``u+ = (1/kappa_eff) ln(y+ + b+) + B_eff`` with
    ``kappa_eff = kappa sqrt(1-alpha)`` derived from a density-weighted mixing
    length (kept intercept from sublayer continuity) and a drag law solved at
    matched outer velocity.  Gives ``Cf(alpha_w)`` in closed form.  No constant
    is fitted.
``transport``
    The bubble transport equation ``d alpha/dt + div(alpha u) = div(D grad alpha)``
    solved three ways -- closed form, finite volume (structural mass ledger) and
    D2Q9 advection-diffusion LBM.
``lbm``
    Euler-Euler two-phase mixture LBM of the channel with a bubbly wall layer
    (matched throughput), plus the exact two-layer solution it is graded against.
``boundary_layer``
    Mixing-length boundary-layer solver that turns ``alpha(y)`` into ``Cf``
    numerically -- the independent (non-log-law) route to ``DR(alpha_w)``.
``power``
    Net-power ledger: saved drag power minus compressor power, with the gas flow
    taken from the transport solve instead of a tuned coefficient.
``campaign`` / ``figures``
    Driver and plotting for ``artifacts/microbubble_in62_n04``.
"""

from .loglaw import (
    B_INTERCEPT,
    KAPPA,
    LogLawParams,
    b_eff,
    cf_analytic,
    cf_clean_correlation,
    coverage_drag_reduction,
    drag_reduction,
    drag_reduction_layer,
    drag_reduction_loglaw,
    dr_table,
    log_layer_average,
    solve_r,
    kappa_eff,
    u_plus,
    u_tau_ratio,
)
from .transport import (
    TransportConfig,
    TransportFV,
    TransportLBM,
    alpha_profile,
    alpha_profile_analytic,
    analytic_supply,
    buffer_layer_alpha,
    mass_ledger,
)
from .lbm import (
    ChannelConfig,
    MixtureChannelLBM,
    exact_two_layer_channel,
    mixture_viscosity,
    two_layer_drag_reduction,
)
from .boundary_layer import (
    BLParams,
    boundary_layer,
    log_law_check,
    drag_reduction_re_theta,
    solve_u_tau,
)
from .power import (
    CompressorSpec,
    gas_flow_rate,
    isothermal_work,
    ledger_scan,
    multistage_work,
    net_power,
    profitability_threshold,
    wall_shear,
)

__all__ = [
    "KAPPA", "B_INTERCEPT", "LogLawParams",
    "kappa_eff", "b_eff", "u_plus", "u_tau_ratio", "drag_reduction",
    "drag_reduction_loglaw", "drag_reduction_layer", "log_layer_average", "solve_r",
    "coverage_drag_reduction", "cf_analytic",
    "cf_clean_correlation", "dr_table",
    "TransportConfig", "TransportFV", "TransportLBM",
    "alpha_profile", "alpha_profile_analytic", "analytic_supply",
    "buffer_layer_alpha",
    "mass_ledger",
    "ChannelConfig", "MixtureChannelLBM", "mixture_viscosity",
    "exact_two_layer_channel", "two_layer_drag_reduction",
    "BLParams", "boundary_layer", "solve_u_tau", "drag_reduction_re_theta",
    "log_law_check",
    "CompressorSpec", "isothermal_work", "multistage_work", "wall_shear",
    "gas_flow_rate", "net_power", "ledger_scan", "profitability_threshold",
]