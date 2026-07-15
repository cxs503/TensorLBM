"""Detached, fail-closed topology commit for D3Q19 free-surface states.

The module owns only the cold conversion path.  It deliberately receives flag
values and precomputed redistribution increments from the hot solver so it has
no dependency on :mod:`free_surface_lbm` and cannot alter collision, streaming,
ABB, or mass-exchange arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .core.d3q19_stencil import D3Q19_MOVING_Q, all_moving_neighbor_masks, assert_no_direct_phase_links
from .d3q19 import equilibrium3d


class TopologyTransactionError(ValueError):
    """A staged topology candidate violated the solver's terminal contract."""


@dataclass(frozen=True)
class TopologyTransactionPlan:
    """Immutable handle for a fully detached staged conversion candidate."""

    f: torch.Tensor
    fill: torch.Tensor
    flags: torch.Tensor
    mass: torch.Tensor
    mass_after_redistribution: float
    mass_after_clamp: float
    mass_after_conversion: float
    mass_after_isolation: float
    conversion_evidence: dict[str, object] | None
    gas_flag: int
    liquid_flag: int
    interface_flag: int
    solid_flag: int
    solid_mask: torch.Tensor


def _init_new(
    f: torch.Tensor, flags: torch.Tensor, mask: torch.Tensor, rho_init: float,
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor, liquid_flag: int, interface_flag: int,
) -> torch.Tensor:
    active = (flags == liquid_flag) | (flags == interface_flag)
    neighbours = torch.stack(all_moving_neighbor_masks(active)).to(f.dtype)
    count = neighbours.sum(dim=0).clamp(min=1)
    # pull-source rolls are generated through the canonical stencil helper.
    from .core.d3q19_stencil import roll_from_pull_source
    ux_mean = (torch.stack([roll_from_pull_source(ux, q) for q in D3Q19_MOVING_Q]) * neighbours).sum(dim=0) / count
    uy_mean = (torch.stack([roll_from_pull_source(uy, q) for q in D3Q19_MOVING_Q]) * neighbours).sum(dim=0) / count
    uz_mean = (torch.stack([roll_from_pull_source(uz, q) for q in D3Q19_MOVING_Q]) * neighbours).sum(dim=0) / count
    feq = equilibrium3d(torch.full_like(ux, float(rho_init)), ux_mean, uy_mean, uz_mean)
    return torch.where(mask.unsqueeze(0), feq, f)


def _validate_candidate(
    f: torch.Tensor, fill: torch.Tensor, flags: torch.Tensor, mass: torch.Tensor,
    solid_mask: torch.Tensor, gas_flag: int, liquid_flag: int, interface_flag: int, solid_flag: int,
) -> None:
    fields = {"f": f, "fill": fill, "mass": mass}
    for name, field in fields.items():
        if not bool(torch.isfinite(field).all()):
            raise TopologyTransactionError(f"topology candidate has non-finite {name}")
    legal = (flags == gas_flag) | (flags == liquid_flag) | (flags == interface_flag) | (flags == solid_flag)
    if not bool(legal.all()):
        raise TopologyTransactionError("topology candidate has an invalid flag value")
    if not bool((flags[solid_mask] == solid_flag).all()):
        raise TopologyTransactionError("topology candidate violates solid flag consistency")
    if bool((fill < 0).any()) or bool((fill > 1).any()):
        raise TopologyTransactionError("topology candidate fill is outside [0, 1]")
    if bool((mass < 0).any()):
        raise TopologyTransactionError("topology candidate has negative mass")
    try:
        assert_no_direct_phase_links(flags, liquid_flag, gas_flag, "direct LIQUID-GAS D3Q19")
    except ValueError as error:
        raise TopologyTransactionError(str(error)) from error


def build_topology_transaction(
    f: torch.Tensor, fill: torch.Tensor, flags: torch.Tensor, mass: torch.Tensor, *,
    to_iface: torch.Tensor, to_liq: torch.Tensor, to_gas: torch.Tensor, recv_new: torch.Tensor,
    redistribution_increment: torch.Tensor, rho_liquid: float, rho_gas: float, solid_mask: torch.Tensor,
    gas_flag: int, liquid_flag: int, interface_flag: int, solid_flag: int,
    ux: torch.Tensor | None = None, uy: torch.Tensor | None = None, uz: torch.Tensor | None = None,
    capture_evidence: bool = False,
    redistribution_link_evidence: tuple[dict[str, object], ...] = (),
) -> TopologyTransactionPlan:
    """Build a detached candidate in the legacy conversion/halo/cleanup order."""
    if ux is None or uy is None or uz is None:
        raise TopologyTransactionError("topology transaction requires pre-conversion velocity fields")
    cf, cfill, cflags, cmass = (value.clone() for value in (f, fill, flags, mass))
    gas_mask = cflags == gas_flag
    cf = _init_new(cf, cflags, to_iface, rho_gas, ux, uy, uz, liquid_flag, interface_flag)
    cflags = torch.where(to_iface, torch.full_like(cflags, interface_flag), cflags)

    mass_before_redistribution = cmass.clone() if capture_evidence else None
    cmass = cmass + redistribution_increment
    mass_after_redistribution = float(cmass.sum())
    cmass = cmass.clamp(0.0, rho_liquid)
    mass_after_clamp = float(cmass.sum())
    mass_before_conversion = cmass.clone() if capture_evidence else None
    fill_before_conversion = cfill.clone() if capture_evidence else None
    flags_before_conversion = cflags.clone() if capture_evidence else None
    f_before_conversion = cf.clone() if capture_evidence else None

    cflags = torch.where(to_liq, torch.full_like(cflags, liquid_flag), cflags)
    cfill = torch.where(to_liq, torch.ones_like(cfill), cfill)
    cmass = torch.where(to_liq, torch.full_like(cmass, rho_liquid), cmass)
    cflags = torch.where(to_gas, torch.full_like(cflags, gas_flag), cflags)
    cfill = torch.where(to_gas, torch.zeros_like(cfill), cfill)
    cmass = torch.where(to_gas, torch.zeros_like(cmass), cmass)
    cf = torch.where(to_gas.unsqueeze(0), torch.zeros_like(cf), cf)
    mass_after_conversion = float(cmass.sum())
    # Evidence attributes conversion itself, not the subsequent envelope halo.
    # Preserve the legacy post-conversion/pre-halo observation boundary.
    flags_after_conversion = cflags.clone() if capture_evidence else None
    fill_after_conversion = cfill.clone() if capture_evidence else None
    mass_after_conversion_field = cmass.clone() if capture_evidence else None
    f_after_conversion = cf.clone() if capture_evidence else None

    shifted_flags = torch.stack(all_moving_neighbor_masks(cflags))
    is_neighbor = ((shifted_flags == liquid_flag) | (shifted_flags == interface_flag)).any(dim=0)
    to_i = (((gas_mask | to_gas) & is_neighbor & ~solid_mask) | recv_new)
    cf = _init_new(cf, cflags, to_i, rho_gas, ux, uy, uz, liquid_flag, interface_flag)
    cflags = torch.where(to_i, torch.full_like(cflags, interface_flag), cflags)
    cfill = torch.where(to_i & ~recv_new, torch.zeros_like(cfill), cfill)
    cmass = torch.where(to_i & ~recv_new, torch.zeros_like(cmass), cmass)
    interface_mask = cflags == interface_flag
    has_neighbor = ((torch.stack(all_moving_neighbor_masks(cflags)) == liquid_flag) | (torch.stack(all_moving_neighbor_masks(cflags)) == interface_flag)).any(dim=0)
    isolated = interface_mask & ~has_neighbor & ~solid_mask
    cflags = torch.where(isolated, torch.full_like(cflags, gas_flag), cflags)
    cfill = torch.where(isolated, torch.zeros_like(cfill), cfill)
    cmass = torch.where(isolated, torch.zeros_like(cmass), cmass)
    cf = torch.where(isolated.unsqueeze(0), torch.zeros_like(cf), cf)
    cflags = torch.where(solid_mask, torch.full_like(cflags, solid_flag), cflags)
    mass_after_isolation = float(cmass.sum())

    evidence = None
    if capture_evidence:
        assert mass_before_redistribution is not None and mass_before_conversion is not None
        assert fill_before_conversion is not None and flags_before_conversion is not None and f_before_conversion is not None
        assert flags_after_conversion is not None and fill_after_conversion is not None
        assert mass_after_conversion_field is not None and f_after_conversion is not None
        conversion_delta = mass_after_conversion_field - mass_before_conversion
        cells = []
        for cell in torch.nonzero((to_liq | to_gas) & (conversion_delta != 0.0), as_tuple=False).tolist():
            z, y, x = (int(value) for value in cell)
            cells.append({"cell": (z, y, x), "flag_before": int(flags_before_conversion[z, y, x]), "flag_after": int(flags_after_conversion[z, y, x]), "mass_before": float(mass_before_conversion[z, y, x]), "mass_after": float(mass_after_conversion_field[z, y, x]), "mass_delta": float(conversion_delta[z, y, x]), "fill_before": float(fill_before_conversion[z, y, x]), "fill_after": float(fill_after_conversion[z, y, x]), "f_before": tuple(float(value) for value in f_before_conversion[:, z, y, x]), "f_after": tuple(float(value) for value in f_after_conversion[:, z, y, x]), "population_before": float(f_before_conversion[:, z, y, x].sum()), "population_after": float(f_after_conversion[:, z, y, x].sum()), "event_id": "conversion", "operator": "conversion"})
        links = []
        for raw_link in redistribution_link_evidence:
            donor = raw_link["donor"]
            receiver = raw_link["receiver"]
            dz, dy, dx = donor  # type: ignore[misc]
            rz, ry, rx = receiver  # type: ignore[misc]
            links.append({
                **raw_link,
                "donor_mass_before_redistribution": float(mass_before_redistribution[dz, dy, dx]),
                "receiver_flag_before": int(flags_before_conversion[rz, ry, rx]),
                "receiver_flag_after": int(flags_after_conversion[rz, ry, rx]),
                "receiver_fill_before": float(fill_before_conversion[rz, ry, rx]),
                "receiver_fill_after": float(fill_after_conversion[rz, ry, rx]),
                "receiver_mass_before": float(mass_before_redistribution[rz, ry, rx]),
                "receiver_mass_after": float(mass_after_conversion_field[rz, ry, rx]),
                "receiver_f_before": tuple(float(value) for value in f_before_conversion[:, rz, ry, rx]),
                "receiver_f_after": tuple(float(value) for value in f_after_conversion[:, rz, ry, rx]),
            })
        evidence = {
            "snapshot_kind": "actual_sparse_pre_redistribution_to_post_conversion",
            "conversion_cells": tuple(cells),
            "redistribution_links": tuple(links),
            "conversion_cell_delta_sum": float(sum(cell["mass_delta"] for cell in cells)),
            "conversion_tensor_delta_sum": float(conversion_delta.sum()),
            "redistribution_link_delta_sum": float(sum(float(link["mass_delta"]) for link in links)),
        }
    _validate_candidate(cf, cfill, cflags, cmass, solid_mask, gas_flag, liquid_flag, interface_flag, solid_flag)
    return TopologyTransactionPlan(cf, cfill, cflags, cmass, mass_after_redistribution, mass_after_clamp, mass_after_conversion, mass_after_isolation, evidence, gas_flag, liquid_flag, interface_flag, solid_flag, solid_mask.clone())


def commit_topology_transaction(
    plan: TopologyTransactionPlan, *, candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate then return detached committed state; inputs are never mutated."""
    f, fill, flags, mass = candidate if candidate is not None else (plan.f, plan.fill, plan.flags, plan.mass)
    _validate_candidate(f, fill, flags, mass, plan.solid_mask if solid_mask is None else solid_mask, plan.gas_flag, plan.liquid_flag, plan.interface_flag, plan.solid_flag)
    return f.clone(), fill.clone(), flags.clone(), mass.clone()
