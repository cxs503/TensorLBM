"""TDD coverage for the pure one-step fixed-nested D3Q27 patch reference."""
from __future__ import annotations

import torch

from tensorlbm.d3q27 import C, equilibrium27
from tensorlbm.d3q27_temporal_reflux import D3Q27InterfaceFluxPacket
from tensorlbm.fixed_nested_patch_schedule import CellExtent3D, FixedNestedPatchScheduleD3Q27
from tensorlbm.fixed_nested_patch_step import run_fixed_nested_patch_step_d3q27


FaceKey = tuple[int, int]


def _schedule() -> FixedNestedPatchScheduleD3Q27:
    return FixedNestedPatchScheduleD3Q27(
        CellExtent3D((0, 0, 0), (8, 8, 8)),
        CellExtent3D((4, 4, 4), (12, 12, 12)),
    )


def _equilibrium_face(shape: tuple[int, int], rho: float = 1.03) -> torch.Tensor:
    rho_field = torch.full((1, *shape), rho, dtype=torch.float64)
    return equilibrium27(
        rho_field,
        torch.full_like(rho_field, 0.021),
        torch.full_like(rho_field, -0.013),
        torch.full_like(rho_field, 0.007),
    ).squeeze(1)


def _faces(schedule: FixedNestedPatchScheduleD3Q27) -> tuple[FaceKey, ...]:
    return tuple((face.axis, face.side) for face in schedule.interface_faces)


def _uniform_inputs(schedule: FixedNestedPatchScheduleD3Q27) -> dict[str, object]:
    coarse_faces: dict[FaceKey, torch.Tensor] = {}
    fine_faces: dict[FaceKey, tuple[torch.Tensor, torch.Tensor]] = {}
    coarse_receivers: dict[FaceKey, torch.Tensor] = {}
    fine_receivers: dict[FaceKey, tuple[torch.Tensor, torch.Tensor]] = {}
    coarse_packets: dict[FaceKey, D3Q27InterfaceFluxPacket] = {}
    fine_packets: dict[FaceKey, tuple[D3Q27InterfaceFluxPacket, D3Q27InterfaceFluxPacket]] = {}
    for face in schedule.interface_faces:
        key = (face.axis, face.side)
        coarse_shape = tuple(size for axis, size in enumerate(face.coarse_face_extent.shape) if axis != face.axis)
        fine_shape = tuple(size for axis, size in enumerate(face.fine_face_extent.shape) if axis != face.axis)
        coarse = _equilibrium_face(coarse_shape)
        fine = _equilibrium_face(fine_shape)
        coarse_faces[key] = coarse
        fine_faces[key] = (fine, fine)
        coarse_receivers[key] = coarse
        fine_receivers[key] = (fine, fine)
        # These are common-orientation, full-face integrated packets.  Their
        # values are intentionally independent of the exchange tensors.
        packet = coarse[:, 0, 0].clone()
        coarse_packets[key] = D3Q27InterfaceFluxPacket(packet, substep=None)
        fine_packets[key] = (
            D3Q27InterfaceFluxPacket(packet / 2.0, substep=0),
            D3Q27InterfaceFluxPacket(packet / 2.0, substep=1),
        )
    return {
        "coarse_outgoing_faces": coarse_faces,
        "fine_outgoing_faces": fine_faces,
        "coarse_receivers": coarse_receivers,
        "fine_receivers": fine_receivers,
        "coarse_flux_packets": coarse_packets,
        "fine_flux_packets": fine_packets,
    }


def test_closed_uniform_equilibrium_is_exact_through_one_coarse_and_two_fine_substeps() -> None:
    schedule = _schedule()
    inputs = _uniform_inputs(schedule)

    result = run_fixed_nested_patch_step_d3q27(schedule, **inputs)

    assert tuple(substep.index for substep in result.fine_substeps) == (0, 1)
    assert tuple(substep.time_start for substep in result.fine_substeps) == (0.0, 0.5)
    assert tuple(substep.time_end for substep in result.fine_substeps) == (0.5, 1.0)
    for key in _faces(schedule):
        assert torch.equal(result.coarse_incoming_faces[key], inputs["coarse_receivers"][key])
        assert torch.equal(result.fine_substeps[0].incoming_faces[key], inputs["fine_receivers"][key][0])
        assert torch.equal(result.fine_substeps[1].incoming_faces[key], inputs["fine_receivers"][key][1])
        reflux = result.reflux_by_face[key]
        assert torch.equal(reflux.mismatch, torch.zeros(27, dtype=torch.float64))
        assert torch.equal(reflux.coarse_correction, torch.zeros(27, dtype=torch.float64))


def test_nonuniform_packets_are_refluxed_to_a_conservative_interface_ledger() -> None:
    schedule = _schedule()
    inputs = _uniform_inputs(schedule)
    coarse_packets = inputs["coarse_flux_packets"]
    fine_packets = inputs["fine_flux_packets"]
    assert isinstance(coarse_packets, dict)
    assert isinstance(fine_packets, dict)
    for ordinal, key in enumerate(_faces(schedule), start=1):
        coarse = torch.arange(27, dtype=torch.float64) * (0.01 * ordinal)
        fine_0 = torch.flip(coarse, dims=(0,)) * 0.2
        fine_1 = coarse * -0.35
        coarse_packets[key] = D3Q27InterfaceFluxPacket(coarse, substep=None)
        fine_packets[key] = (
            D3Q27InterfaceFluxPacket(fine_0, substep=0),
            D3Q27InterfaceFluxPacket(fine_1, substep=1),
        )

    result = run_fixed_nested_patch_step_d3q27(schedule, **inputs)

    total_owner_correction = torch.zeros(27, dtype=torch.float64)
    for key, reflux in result.reflux_by_face.items():
        corrected_fine = reflux.corrected_fine_fluxes[0] + reflux.corrected_fine_fluxes[1]
        assert torch.allclose(reflux.corrected_coarse_flux, corrected_fine, rtol=0.0, atol=1e-14)
        assert torch.allclose(
            (reflux.corrected_coarse_flux - corrected_fine).sum(), torch.tensor(0.0, dtype=torch.float64), rtol=0.0, atol=1e-14
        )
        assert torch.allclose(
            ((reflux.corrected_coarse_flux - corrected_fine)[:, None] * C.to(dtype=torch.float64)).sum(dim=0),
            torch.zeros(3, dtype=torch.float64), rtol=0.0, atol=1e-14,
        )
        total_owner_correction += reflux.coarse_correction + sum(reflux.fine_corrections)
    assert torch.equal(total_owner_correction, torch.zeros_like(total_owner_correction))


def test_missing_interface_face_is_rejected_before_any_exchange() -> None:
    schedule = _schedule()
    inputs = _uniform_inputs(schedule)
    inputs["coarse_outgoing_faces"].pop((0, -1))

    try:
        run_fixed_nested_patch_step_d3q27(schedule, **inputs)
    except ValueError as error:
        assert "exactly the six schedule interface faces" in str(error)
    else:
        raise AssertionError("incomplete face ledger must be rejected")
