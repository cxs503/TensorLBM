"""Fixed-nested D3Q27 planar patch-interface population reconstruction.

This deliberately narrow module only reconstructs one planar receiving face at
a fixed 2:1 coarse/fine interface.  It has no patch discovery, ghost-layer
allocation, collision, streaming, temporal scheduling, or reflux/AMR policy;
it is not a full AMR implementation.

The interface normal ``n`` is a signed Cartesian unit vector that always points
from the *coarse* patch to the *fine* patch.  A population with ``c_i . n > 0``
therefore crosses coarse -> fine, while one with ``c_i . n < 0`` crosses fine
-> coarse.  Each function returns a new receiver-face tensor and changes only
those incoming directions; all other directions retain the supplied receiver
values.  The caller owns source faces and must perform its own stream/collision
bookkeeping.  The returned receiver face is owned by the receiving patch.

A coarse face cell covers four fine face cells.  Coarse -> fine uses
piecewise-constant injection.  Fine -> coarse averages the four fine face
cells and the two fine substeps corresponding to one coarse step.  With
``V_c/V_f = 8`` and ``dt_c/dt_f = 2``, these choices preserve volume-time
weighted mass and all population momentum components crossing the plane.
"""
from __future__ import annotations

import torch

from .d3q27 import C

_Q = 27
_SUBSTEPS = 2


def _validate_normal(normal: tuple[int, int, int]) -> torch.Tensor:
    """Return a CPU D3Q27-compatible signed Cartesian unit normal."""
    if not isinstance(normal, tuple) or len(normal) != 3:
        raise ValueError("normal must be a three-component signed Cartesian unit tuple")
    if any(not isinstance(value, int) for value in normal) or sum(value * value for value in normal) != 1:
        raise ValueError("normal must be a three-component signed Cartesian unit tuple")
    return torch.tensor(normal, dtype=C.dtype)


def _validate_face(face: torch.Tensor, *, name: str) -> None:
    if not isinstance(face, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if face.ndim != 3 or face.shape[0] != _Q:
        raise ValueError(f"{name} must have shape (27, tangential_0, tangential_1)")
    if not face.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


def _validate_matching_receivers(source: torch.Tensor, receiver: torch.Tensor, *, source_name: str) -> None:
    _validate_face(source, name=source_name)
    _validate_face(receiver, name="receiver")
    if source.dtype != receiver.dtype or source.device != receiver.device:
        raise ValueError(f"{source_name} and receiver must have the same dtype and device")


def reconstruct_fine_incoming_from_coarse_d3q27(
    coarse_face: torch.Tensor,
    fine_receiver: torch.Tensor,
    normal: tuple[int, int, int],
) -> torch.Tensor:
    """Fill coarse->fine incoming D3Q27 populations on one fine receiver face.

    ``coarse_face`` is source-owned with shape ``(27, nc0, nc1)``.
    ``fine_receiver`` is receiver-owned with shape ``(27, 2*nc0, 2*nc1)``.
    ``normal`` points from coarse to fine.  Only directions ``c_i . normal >
    0`` are written: they leave coarse and are incoming to fine.  Every coarse
    population is copied to its 2x2 fine-face children; directions not crossing
    coarse->fine are retained verbatim from ``fine_receiver``.  Neither input
    is mutated.

    This is the spatial part of a fixed 2:1 interface exchange.  For a
    coarse-step/fine-step ratio of two, four injected fine face values per fine
    step carry the same volume-time weighted packet as one coarse face value.
    """
    _validate_matching_receivers(coarse_face, fine_receiver, source_name="coarse_face")
    normal_tensor = _validate_normal(normal).to(device=coarse_face.device)
    nc0, nc1 = coarse_face.shape[1:]
    if fine_receiver.shape[1:] != (2 * nc0, 2 * nc1):
        raise ValueError("fine_receiver must have tangential shape (2*nc0, 2*nc1)")

    crossing = (C.to(coarse_face.device) @ normal_tensor) > 0
    injected = coarse_face.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    return torch.where(crossing[:, None, None], injected, fine_receiver)


def reconstruct_coarse_incoming_from_fine_d3q27(
    fine_substep_faces: torch.Tensor,
    coarse_receiver: torch.Tensor,
    normal: tuple[int, int, int],
) -> torch.Tensor:
    """Fill fine->coarse incoming D3Q27 populations on one coarse receiver face.

    ``fine_substep_faces`` is source-owned with shape
    ``(2, 27, 2*nc0, 2*nc1)``: its leading axis is the two ordered fine
    substeps inside one coarse step. ``coarse_receiver`` is receiver-owned
    with shape ``(27, nc0, nc1)``. ``normal`` points from coarse to fine. Only
    directions ``c_i . normal < 0`` are written: they leave fine and are
    incoming to coarse.  They are reconstructed by the arithmetic mean over
    the two fine substeps and each 2x2 fine-face footprint. Directions not
    crossing fine->coarse remain verbatim from ``coarse_receiver``. Neither
    input is mutated.

    The two-substep mean is exact for a temporally uniform incoming packet and
    makes the volume-time transfer convention explicit; no interpolation or
    asynchronous patch scheduling is supplied here.
    """
    if not isinstance(fine_substep_faces, torch.Tensor):
        raise TypeError("fine_substep_faces must be a torch.Tensor")
    if fine_substep_faces.ndim != 4 or fine_substep_faces.shape[:2] != (_SUBSTEPS, _Q):
        raise ValueError("fine_substep_faces must have shape (2, 27, tangential_0, tangential_1)")
    if not fine_substep_faces.is_floating_point():
        raise TypeError("fine_substep_faces must have a floating-point dtype")
    _validate_face(coarse_receiver, name="coarse_receiver")
    if fine_substep_faces.dtype != coarse_receiver.dtype or fine_substep_faces.device != coarse_receiver.device:
        raise ValueError("fine_substep_faces and coarse_receiver must have the same dtype and device")
    normal_tensor = _validate_normal(normal).to(device=coarse_receiver.device)
    nc0, nc1 = coarse_receiver.shape[1:]
    if fine_substep_faces.shape[2:] != (2 * nc0, 2 * nc1):
        raise ValueError("fine_substep_faces must have tangential shape (2*nc0, 2*nc1)")

    restricted = fine_substep_faces.reshape(_SUBSTEPS, _Q, nc0, 2, nc1, 2).mean(dim=(0, 3, 5))
    crossing = (C.to(coarse_receiver.device) @ normal_tensor) < 0
    return torch.where(crossing[:, None, None], restricted, coarse_receiver)


__all__ = [
    "reconstruct_coarse_incoming_from_fine_d3q27",
    "reconstruct_fine_incoming_from_coarse_d3q27",
]
