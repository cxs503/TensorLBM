"""Verification case: Zou-He channel inlet/outlet reconstruction (D2Q9)."""

from .solver import (
    ChannelCase,
    analytic_profile,
    run_channel,
    equilibrium_d2q9,
)

__all__ = ["ChannelCase", "analytic_profile", "run_channel", "equilibrium_d2q9"]