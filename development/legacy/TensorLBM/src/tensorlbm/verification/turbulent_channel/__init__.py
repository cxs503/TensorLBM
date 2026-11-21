"""Turbulent channel flow verification for TensorLBM.

Re_tau = 180 / 395 / 590 channel flow with D3Q19 MRT + Smagorinsky SGS
+ van Driest damping, validated against MKM1999 / Moser2021 DNS.
"""

from .channel_solver import (
    ChannelConfig,
    ChannelSolver,
    run_channel,
    van_driest_damping,
    log_law_u_plus,
    reichardt_u_plus,
)
from .statistics import (
    StatsAccumulator,
    compute_profiles,
    compute_energy_spectrum,
    compute_sgs_stress,
)
from .reference_data import (
    DNSReference,
    get_dns_reference,
    MKM1999_DATA,
    MOSER2021_DATA,
)
from .verification import (
    VerificationResult,
    run_single_case,
    run_grid_convergence,
    run_full_campaign,
)

__all__ = [
    "ChannelConfig",
    "ChannelSolver",
    "run_channel",
    "van_driest_damping",
    "log_law_u_plus",
    "reichardt_u_plus",
    "StatsAccumulator",
    "compute_profiles",
    "compute_energy_spectrum",
    "compute_sgs_stress",
    "DNSReference",
    "get_dns_reference",
    "MKM1999_DATA",
    "MOSER2021_DATA",
    "VerificationResult",
    "run_single_case",
    "run_grid_convergence",
    "run_full_campaign",
]
