"""Cold, observer-only inventory measurements for D3Q19 free-surface stages.

The solver has an independent tracked ``mass`` field, while the physically
represented liquid inventory combines population density in LIQUID cells with
fill/rho content in INTERFACE cells.  These measurements deliberately expose
that distinction; they do not feed any value back into the solver.
"""
from __future__ import annotations

from typing import Final

import torch


GAS: Final = 0
LIQUID: Final = 1
INTERFACE: Final = 2

CANONICAL_STAGE_ORDER: Final = (
    "before_collision",
    "after_collision_and_forcing",
    "after_stream_and_gas_zero",
    "after_abb",
    "after_wall_boundary",
    "after_mass_exchange",
    "after_topology_redistribution",
    "after_topology_clamp",
    "after_topology_conversion",
    "after_topology_halo_isolation_boundary",
)


def inventory_measurement(
    f: torch.Tensor,
    fill: torch.Tensor,
    flags: torch.Tensor,
    mass: torch.Tensor,
    *,
    rho_liquid: float,
) -> dict[str, float]:
    """Measure distinct state representations without inferring conservation.

    ``fill_rho_inventory`` is the fill-field representation over tracked
    liquid/interface cells. ``population_density_inventory`` is the raw
    population-density representation over those same cells. The named
    ``total_liquid_inventory`` is the production physical-inventory convention:
    population density for LIQUID plus fill/rho for INTERFACE.
    """
    rho = f.sum(dim=0)
    liquid = flags == LIQUID
    interface = flags == INTERFACE
    represented = liquid | interface
    fill_rho = torch.where(represented, fill * rho_liquid, torch.zeros_like(fill)).sum()
    population = torch.where(represented, rho, torch.zeros_like(rho)).sum()
    total = (
        torch.where(liquid, rho, torch.zeros_like(rho)).sum()
        + torch.where(interface, fill * rho_liquid, torch.zeros_like(fill)).sum()
    )
    return {
        "tracked_independent_mass": float(mass.sum()),
        "fill_rho_inventory": float(fill_rho),
        "population_density_inventory": float(population),
        "total_liquid_inventory": float(total),
    }


def inventory_stage_deltas(
    stages: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Return consecutive actual-state deltas for a chronological stage map."""
    previous: dict[str, float] | None = None
    result: dict[str, dict[str, float]] = {}
    for name, measurement in stages.items():
        if previous is not None:
            result[name] = {
                key: float(measurement[key] - previous[key]) for key in measurement
            }
        previous = measurement
    return result
