"""Exact lattice-link intersections with triangulated CAD surfaces.

The hot LBM boundary kernel consumes a fractional link distance ``q``.  This
module is a cold preprocessing component: it refines selected fluid-to-solid
links by exact Moller-Trumbore segment/triangle intersections, using a simple
integer-cell spatial index to avoid testing every CAD triangle against every
boundary link.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class TriangleLinkIntersectionDiagnostics:
    target_links: int
    resolved_links: int
    missing_links: int
    minimum_q: float | None
    maximum_q: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _lattice_velocities(q: int) -> torch.Tensor:
    if q == 19:
        from .d3q19 import C
    elif q == 27:
        from .d3q27 import C
    else:
        raise ValueError("triangle link intersections support D3Q19 or D3Q27")
    return C.cpu()


def _triangle_spatial_index(
    triangles: np.ndarray,
) -> dict[tuple[int, int, int], tuple[int, ...]]:
    bins: dict[tuple[int, int, int], list[int]] = {}
    lower = np.floor(triangles.min(axis=1)).astype(np.int64)
    upper = np.floor(triangles.max(axis=1)).astype(np.int64)
    for triangle_index, (lo, hi) in enumerate(zip(lower, upper, strict=True)):
        for ix in range(int(lo[0]), int(hi[0]) + 1):
            for iy in range(int(lo[1]), int(hi[1]) + 1):
                for iz in range(int(lo[2]), int(hi[2]) + 1):
                    bins.setdefault((ix, iy, iz), []).append(triangle_index)
    return {key: tuple(values) for key, values in bins.items()}


def _segment_triangle_q(
    origin: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    *,
    tolerance: float,
) -> float | None:
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, pvec)
    nonparallel = np.abs(determinant) > tolerance
    if not np.any(nonparallel):
        return None
    inverse = np.zeros_like(determinant)
    inverse[nonparallel] = 1.0 / determinant[nonparallel]
    tvec = origin - triangles[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inverse
    qvec = np.cross(tvec, edge1)
    v = (
        np.einsum(
            "ij,ij->i",
            np.broadcast_to(direction, qvec.shape),
            qvec,
        )
        * inverse
    )
    fraction = np.einsum("ij,ij->i", edge2, qvec) * inverse
    valid = (
        nonparallel
        & (u >= -tolerance)
        & (v >= -tolerance)
        & (u + v <= 1.0 + tolerance)
        & (fraction >= -tolerance)
        & (fraction <= 1.0 + tolerance)
    )
    if not np.any(valid):
        return None
    return float(np.clip(fraction[valid].min(), 0.0, 1.0))


def refine_bfl_q_with_triangles(
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    target_solid: torch.Tensor | None = None,
    require_complete: bool = True,
    tolerance: float = 1.0e-9,
) -> tuple[torch.Tensor, TriangleLinkIntersectionDiagnostics]:
    """Refine selected BFL links with exact CAD segment intersections.

    ``target_solid`` can restrict refinement to links whose destination cell
    belongs to a particular component, such as SUBOFF sail/fins.  The input
    ``q_field`` is not mutated.  With ``require_complete=True`` any selected
    voxel link lacking a CAD intersection fails closed.
    """
    if (
        fluid_boundary_mask.ndim != 4
        or fluid_boundary_mask.dtype is not torch.bool
        or q_field.shape != fluid_boundary_mask.shape
        or not q_field.is_floating_point()
        or q_field.device != fluid_boundary_mask.device
    ):
        raise ValueError("mask and q_field must be matching Q-by-3D tensors")
    spatial_shape = fluid_boundary_mask.shape[1:]
    if target_solid is not None and (
        target_solid.shape != spatial_shape
        or target_solid.dtype is not torch.bool
        or target_solid.device != fluid_boundary_mask.device
    ):
        raise ValueError("target_solid must be a matching 3-D bool tensor")
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not vertices.is_floating_point()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or faces.dtype not in {torch.int32, torch.int64}
    ):
        raise ValueError("vertices/faces must have shapes (N,3)/(M,3)")
    if faces.numel() == 0:
        raise ValueError("at least one triangle is required")
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[0]:
        raise ValueError("triangle face index is out of bounds")

    velocities = _lattice_velocities(fluid_boundary_mask.shape[0])
    vertices_np = vertices.detach().cpu().to(torch.float64).numpy()
    faces_np = faces.detach().cpu().to(torch.int64).numpy()
    triangles = vertices_np[faces_np]
    spatial_index = _triangle_spatial_index(triangles)
    refined = q_field.clone()
    target_count = resolved_count = 0
    resolved_values: list[float] = []

    for lattice_direction in range(1, fluid_boundary_mask.shape[0]):
        cx, cy, cz = (int(value) for value in velocities[lattice_direction].tolist())
        selected = fluid_boundary_mask[lattice_direction]
        if target_solid is not None:
            target_neighbor = torch.roll(
                target_solid,
                shifts=(-cz, -cy, -cx),
                dims=(0, 1, 2),
            )
            selected = selected & target_neighbor
        indices_zyx = selected.nonzero(as_tuple=False).cpu().numpy()
        if not len(indices_zyx):
            continue
        target_count += len(indices_zyx)
        direction = np.asarray((cx, cy, cz), dtype=np.float64)
        direction_values: list[float] = []
        direction_indices: list[tuple[int, int, int]] = []
        for z, y, x in indices_zyx:
            origin = np.asarray((x, y, z), dtype=np.float64)
            destination = origin + direction
            keys = {
                tuple(np.floor(origin).astype(np.int64)),
                tuple(np.floor(destination).astype(np.int64)),
            }
            candidates = sorted(
                {triangle_index for key in keys for triangle_index in spatial_index.get(key, ())}
            )
            fraction = (
                _segment_triangle_q(
                    origin,
                    direction,
                    triangles[candidates],
                    tolerance=tolerance,
                )
                if candidates
                else None
            )
            if fraction is None:
                continue
            direction_values.append(fraction)
            direction_indices.append((int(z), int(y), int(x)))
        if direction_values:
            index_tensor = torch.tensor(
                direction_indices,
                device=q_field.device,
                dtype=torch.long,
            )
            refined[
                lattice_direction,
                index_tensor[:, 0],
                index_tensor[:, 1],
                index_tensor[:, 2],
            ] = torch.tensor(
                direction_values,
                device=q_field.device,
                dtype=q_field.dtype,
            )
            resolved_count += len(direction_values)
            resolved_values.extend(direction_values)

    diagnostics = TriangleLinkIntersectionDiagnostics(
        target_links=target_count,
        resolved_links=resolved_count,
        missing_links=target_count - resolved_count,
        minimum_q=min(resolved_values) if resolved_values else None,
        maximum_q=max(resolved_values) if resolved_values else None,
    )
    if require_complete and diagnostics.missing_links:
        raise ValueError(
            "triangulated CAD did not intersect "
            f"{diagnostics.missing_links} of {diagnostics.target_links} "
            "selected boundary links",
        )
    return refined, diagnostics


__all__ = [
    "TriangleLinkIntersectionDiagnostics",
    "refine_bfl_q_with_triangles",
]
