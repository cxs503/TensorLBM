"""Diagnostic-only CH inventory and adapter-stream boundary crossing ledger.

The ledger observes the states returned by the collision → adapter-stream loop.
It does not define a continuum phase flux, infer a collision contribution, or
make a total-phase-conservation claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from ..d3q19 import C
from .stream_boundary_contract import BoundaryPolicy

if TYPE_CHECKING:
    from .evolution_stream_loop import FreeEnergyAdapterStreamLoopResult

ADAPTER_STREAM_DIAGNOSTIC_ONLY = "diagnostic_only"


@dataclass(frozen=True)
class AdapterStreamBoundaryCrossing:
    """Population-``g`` crossing ledger for one adapter streaming operation."""

    outgoing_g: float
    incoming_g: float
    net_g: float
    status: str
    scope: str


@dataclass(frozen=True)
class PhaseInventoryFluxStep:
    """One returned loop state plus the preceding adapter-stream crossing terms."""

    step: int
    phi_integral: float
    g_sum: float
    phi_integral_change: float
    g_sum_change: float
    stream_boundary_outgoing_g: float
    stream_boundary_incoming_g: float
    stream_boundary_net_g: float
    stream_boundary_crossing_status: str
    stream_boundary_scope: str


@dataclass(frozen=True)
class PhaseInventoryFluxDiagnostic:
    """Fail-closed inventory/crossing diagnostic over an actual loop result."""

    boundary: BoundaryPolicy
    steps: tuple[PhaseInventoryFluxStep, ...]
    status: str = ADAPTER_STREAM_DIAGNOSTIC_ONLY
    physical: bool = False
    physical_phase_flux: None = None
    collision_contribution: None = None
    scope: str = (
        "Inventory changes and boundary terms are adapter-stream diagnostics only; "
        "they do not state total phase conservation or a physical phase flux."
    )


def adapter_stream_boundary_crossing(
    g: torch.Tensor, *, boundary: BoundaryPolicy
) -> AdapterStreamBoundaryCrossing:
    """Report adapter-stream link crossing terms for the phase distribution ``g``.

    For periodic streaming, each outgoing exterior link is paired with its
    periodic re-entry, so the *net adapter-stream transfer* is structurally
    zero.  This is not a statement about collision or total phase conservation.
    For no-flux streaming, exterior links are reflected and boundary crossing is
    structurally zero; collision and physical-flux terms remain undefined.
    """
    if boundary == "no_flux":
        return AdapterStreamBoundaryCrossing(
            outgoing_g=0.0,
            incoming_g=0.0,
            net_g=0.0,
            status="no_flux_reflection_zero_crossing",
            scope=(
                "Adapter-stream link reflection has zero boundary crossing; this "
                "does not infer collision contribution or physical phase flux."
            ),
        )
    if boundary != "periodic":
        raise ValueError("boundary must be either 'periodic' or 'no_flux'")
    if not isinstance(g, torch.Tensor) or g.ndim != 4 or g.shape[0] != 19:
        raise ValueError("g must have shape (19, nz, ny, nx)")

    nz, ny, nx = g.shape[1:]
    outgoing = g.new_zeros(())
    for q, (cx, cy, cz) in enumerate(C.tolist()):
        # A diagonal link crossing two faces remains one outgoing D3Q19 link.
        z_slice = slice(max(0, -cz), nz - max(0, cz))
        y_slice = slice(max(0, -cy), ny - max(0, cy))
        x_slice = slice(max(0, -cx), nx - max(0, cx))
        inside = torch.zeros((nz, ny, nx), dtype=torch.bool, device=g.device)
        inside[z_slice, y_slice, x_slice] = True
        outgoing = outgoing + g[q][~inside].sum()
    outgoing_value = float(outgoing.item())
    return AdapterStreamBoundaryCrossing(
        outgoing_g=outgoing_value,
        incoming_g=outgoing_value,
        net_g=0.0,
        status="periodic_transfer_net_zero",
        scope=(
            "Periodic adapter-stream transfer has structurally zero net boundary "
            "term only; it is not a total phase conservation claim."
        ),
    )


def diagnose_adapter_stream_phase_inventory_flux(
    result: "FreeEnergyAdapterStreamLoopResult",
) -> PhaseInventoryFluxDiagnostic:
    """Build the diagnostic from the loop's returned ``step_states`` and ledger."""
    from .evolution_stream_loop import FreeEnergyAdapterStreamLoopResult

    if not isinstance(result, FreeEnergyAdapterStreamLoopResult):
        raise TypeError("result must be a FreeEnergyAdapterStreamLoopResult")
    if len(result.diagnostics) != len(result.step_states) + 1:
        raise ValueError("loop result diagnostics must include the initial state")

    samples: list[PhaseInventoryFluxStep] = []
    previous_phi = result.diagnostics[0].phi_integral
    previous_g = result.diagnostics[0].g_sum
    initial = result.diagnostics[0]
    samples.append(
        PhaseInventoryFluxStep(
            step=0, phi_integral=previous_phi, g_sum=previous_g,
            phi_integral_change=0.0, g_sum_change=0.0,
            stream_boundary_outgoing_g=0.0, stream_boundary_incoming_g=0.0,
            stream_boundary_net_g=0.0, stream_boundary_crossing_status="initial_no_stream",
            stream_boundary_scope="Initial returned-loop inventory; no adapter stream precedes it.",
        )
    )
    for state, ledger in zip(result.step_states, result.diagnostics[1:]):
        g_sum = float(state.g.sum().item())
        phi_integral = g_sum
        samples.append(
            PhaseInventoryFluxStep(
                step=ledger.step, phi_integral=phi_integral, g_sum=g_sum,
                phi_integral_change=phi_integral - previous_phi,
                g_sum_change=g_sum - previous_g,
                stream_boundary_outgoing_g=ledger.stream_boundary_outgoing_g,
                stream_boundary_incoming_g=ledger.stream_boundary_incoming_g,
                stream_boundary_net_g=ledger.stream_boundary_net_g,
                stream_boundary_crossing_status=ledger.stream_boundary_crossing_status,
                stream_boundary_scope=ledger.stream_boundary_scope,
            )
        )
        previous_phi, previous_g = phi_integral, g_sum
    return PhaseInventoryFluxDiagnostic(boundary=result.boundary, steps=tuple(samples))


__all__ = [
    "ADAPTER_STREAM_DIAGNOSTIC_ONLY", "AdapterStreamBoundaryCrossing",
    "PhaseInventoryFluxDiagnostic", "PhaseInventoryFluxStep",
    "adapter_stream_boundary_crossing", "diagnose_adapter_stream_phase_inventory_flux",
]
