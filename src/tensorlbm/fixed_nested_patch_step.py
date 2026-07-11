"""Pure one-step orchestration for a fixed 2:1 nested D3Q27 patch.

This reference composes only the fixed schedule, planar population exchange,
and temporal reflux ledger helpers.  It neither allocates volumetric patches
nor performs collision, streaming, solver mutation, adaptive refinement, or
SUBOFF integration.  Inputs are source/receiver face snapshots owned by the
caller; returned tensors and corrections are likewise unapplied snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeVar, cast

import torch

from .d3q27_temporal_reflux import (
    D3Q27InterfaceFluxPacket,
    D3Q27TemporalRefluxResult,
    reflux_d3q27_2to1,
)
from .fixed_nested_interface import (
    reconstruct_coarse_incoming_from_fine_d3q27,
    reconstruct_fine_incoming_from_coarse_d3q27,
)
from .fixed_nested_patch_schedule import FixedNestedPatchScheduleD3Q27

FaceKey = tuple[int, int]
_T = TypeVar("_T")


@dataclass(frozen=True)
class FinePatchSubstepResultD3Q27:
    """One scheduled fine half-step's reconstructed fine incoming faces."""

    index: int
    time_start: float
    time_end: float
    incoming_faces: dict[FaceKey, torch.Tensor]


@dataclass(frozen=True)
class FixedNestedPatchStepResultD3Q27:
    """Unapplied exchange/reflux outputs of exactly one coarse patch step."""

    coarse_incoming_faces: dict[FaceKey, torch.Tensor]
    fine_substeps: tuple[FinePatchSubstepResultD3Q27, FinePatchSubstepResultD3Q27]
    reflux_by_face: dict[FaceKey, D3Q27TemporalRefluxResult]


def _face_keys(schedule: FixedNestedPatchScheduleD3Q27) -> tuple[FaceKey, ...]:
    return tuple((face.axis, face.side) for face in schedule.interface_faces)


def _require_schedule_faces(name: str, values: Mapping[FaceKey, _T], keys: tuple[FaceKey, ...]) -> None:
    if not isinstance(values, Mapping) or set(values) != set(keys):
        raise ValueError(f"{name} must contain exactly the six schedule interface faces")


def _require_pair(name: str, value: object) -> tuple[object, object]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} entries must be two ordered fine-substep values")
    return value


def run_fixed_nested_patch_step_d3q27(
    schedule: FixedNestedPatchScheduleD3Q27,
    *,
    coarse_outgoing_faces: Mapping[FaceKey, torch.Tensor],
    fine_outgoing_faces: Mapping[FaceKey, tuple[torch.Tensor, torch.Tensor]],
    coarse_receivers: Mapping[FaceKey, torch.Tensor],
    fine_receivers: Mapping[FaceKey, tuple[torch.Tensor, torch.Tensor]],
    coarse_flux_packets: Mapping[FaceKey, D3Q27InterfaceFluxPacket],
    fine_flux_packets: Mapping[FaceKey, tuple[D3Q27InterfaceFluxPacket, D3Q27InterfaceFluxPacket]],
) -> FixedNestedPatchStepResultD3Q27:
    """Wire one coarse step, its two fine substeps, and six reflux ledgers.

    The six face mappings must exactly match ``schedule.interface_faces`` using
    ``(axis, side)`` keys.  Each fine tuple is ordered ``(substep_0, substep_1)``.
    Coarse->fine reconstruction occurs for each fine substep; fine->coarse
    reconstruction receives both fine source snapshots together, then each
    face's full-step coarse packet and two fine packets are refluxed.  No input
    tensor or packet is modified.
    """
    if not isinstance(schedule, FixedNestedPatchScheduleD3Q27):
        raise TypeError("schedule must be a FixedNestedPatchScheduleD3Q27")
    keys = _face_keys(schedule)
    for name, values in (
        ("coarse_outgoing_faces", coarse_outgoing_faces),
        ("fine_outgoing_faces", fine_outgoing_faces),
        ("coarse_receivers", coarse_receivers),
        ("fine_receivers", fine_receivers),
        ("coarse_flux_packets", coarse_flux_packets),
        ("fine_flux_packets", fine_flux_packets),
    ):
        _require_schedule_faces(name, values, keys)

    coarse_incoming: dict[FaceKey, torch.Tensor] = {}
    fine_incoming = ({}, {})
    reflux_by_face: dict[FaceKey, D3Q27TemporalRefluxResult] = {}
    for face in schedule.interface_faces:
        key = (face.axis, face.side)
        fine_sources = _require_pair("fine_outgoing_faces", fine_outgoing_faces[key])
        fine_targets = _require_pair("fine_receivers", fine_receivers[key])
        fine_packets = _require_pair("fine_flux_packets", fine_flux_packets[key])
        if not all(isinstance(value, torch.Tensor) for value in (*fine_sources, *fine_targets)):
            raise TypeError("fine face entries must be torch.Tensor instances")
        if not all(isinstance(value, D3Q27InterfaceFluxPacket) for value in fine_packets):
            raise TypeError("fine flux entries must be D3Q27InterfaceFluxPacket instances")

        coarse_source = coarse_outgoing_faces[key]
        coarse_target = coarse_receivers[key]
        if not isinstance(coarse_source, torch.Tensor) or not isinstance(coarse_target, torch.Tensor):
            raise TypeError("coarse face entries must be torch.Tensor instances")
        fine_source_pair = cast(tuple[torch.Tensor, torch.Tensor], fine_sources)
        fine_target_pair = cast(tuple[torch.Tensor, torch.Tensor], fine_targets)
        fine_packet_pair = cast(
            tuple[D3Q27InterfaceFluxPacket, D3Q27InterfaceFluxPacket], fine_packets
        )
        coarse_incoming[key] = reconstruct_coarse_incoming_from_fine_d3q27(
            torch.stack(fine_source_pair), coarse_target, face.normal
        )
        for substep in range(2):
            fine_incoming[substep][key] = reconstruct_fine_incoming_from_coarse_d3q27(
                coarse_source, fine_target_pair[substep], face.normal
            )
        reflux_by_face[key] = reflux_d3q27_2to1(coarse_flux_packets[key], fine_packet_pair)

    substeps = schedule.fine_substeps
    return FixedNestedPatchStepResultD3Q27(
        coarse_incoming_faces=coarse_incoming,
        fine_substeps=(
            FinePatchSubstepResultD3Q27(substeps[0].index, substeps[0].time_start, substeps[0].time_end, fine_incoming[0]),
            FinePatchSubstepResultD3Q27(substeps[1].index, substeps[1].time_start, substeps[1].time_end, fine_incoming[1]),
        ),
        reflux_by_face=reflux_by_face,
    )


__all__ = [
    "FaceKey",
    "FinePatchSubstepResultD3Q27",
    "FixedNestedPatchStepResultD3Q27",
    "run_fixed_nested_patch_step_d3q27",
]
