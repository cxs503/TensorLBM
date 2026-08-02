"""Solid-aware pressure-gradient sampling at wall exchange nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class WallTangentialPressureGradientSamples:
    """Tangential-gradient magnitudes aligned with active wall nodes."""

    vector: torch.Tensor
    magnitude: torch.Tensor
    valid: torch.Tensor
    requested_nodes: int
    valid_nodes: int

    @property
    def rejected_fraction(self) -> float:
        if not self.requested_nodes:
            return 0.0
        return (self.requested_nodes - self.valid_nodes) / self.requested_nodes


@dataclass(frozen=True)
class WallPressureGradientAggregate:
    """Count-weighted aggregate without inventing combined quantiles."""

    observations: int
    requested_sample_exposures: int
    valid_sample_exposures: int
    rejected_fraction: float
    minimum: float | None
    mean: float | None
    maximum: float | None
    le_one_sample_exposures: int
    gt_ten_sample_exposures: int
    fraction_le_one: float
    fraction_gt_ten: float
    gradient_scheme: str

    def to_dict(self) -> dict[str, float | int | str | None]:
        return dict(vars(self))


def aggregate_wall_pressure_gradient_summaries(
    summaries: Sequence[Mapping[str, float | int | str | None]],
) -> WallPressureGradientAggregate:
    """Aggregate repeated gradient summaries using exact sample counts."""
    if not summaries:
        raise ValueError("at least one pressure-gradient summary is required")
    scheme = str(summaries[0]["gradient_scheme"])
    if any(str(item["gradient_scheme"]) != scheme for item in summaries[1:]):
        raise ValueError("cannot aggregate different pressure-gradient schemes")

    def count(item: Mapping, single: str, aggregate: str) -> int:
        return int(item[single] if single in item else item[aggregate])

    requested = sum(
        count(item, "requested_samples", "requested_sample_exposures") for item in summaries
    )
    valid = sum(count(item, "valid_samples", "valid_sample_exposures") for item in summaries)
    le_one = sum(count(item, "le_one_samples", "le_one_sample_exposures") for item in summaries)
    gt_ten = sum(count(item, "gt_ten_samples", "gt_ten_sample_exposures") for item in summaries)
    weighted_means = [
        (
            count(item, "valid_samples", "valid_sample_exposures"),
            float(item["mean"]),
        )
        for item in summaries
        if count(item, "valid_samples", "valid_sample_exposures") and item["mean"] is not None
    ]
    minima = [float(item["minimum"]) for item in summaries if item["minimum"] is not None]
    maxima = [float(item["maximum"]) for item in summaries if item["maximum"] is not None]
    return WallPressureGradientAggregate(
        observations=sum(int(item.get("observations", 1)) for item in summaries),
        requested_sample_exposures=requested,
        valid_sample_exposures=valid,
        rejected_fraction=(requested - valid) / requested if requested else 0.0,
        minimum=min(minima) if minima else None,
        mean=(
            sum(count_value * mean for count_value, mean in weighted_means) / valid
            if valid
            else None
        ),
        maximum=max(maxima) if maxima else None,
        le_one_sample_exposures=le_one,
        gt_ten_sample_exposures=gt_ten,
        fraction_le_one=le_one / valid if valid else 0.0,
        fraction_gt_ten=gt_ten / valid if valid else 0.0,
        gradient_scheme=scheme,
    )


def sample_wall_tangential_pressure_gradient(
    pressure: torch.Tensor,
    solid: torch.Tensor,
    active: torch.Tensor,
    normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> WallTangentialPressureGradientSamples:
    """Fit a local 3-D pressure gradient using only fluid neighbours.

    A weighted least-squares fit over the 26-cell Moore neighbourhood avoids
    reading arbitrary populations inside a curved solid, which can strongly
    contaminate Cartesian central differences at a boundary node.  The result
    is projected onto the local wall tangent plane.  Rank-deficient and
    non-finite samples fail closed through the returned validity mask.
    """
    if pressure.ndim != 3 or not pressure.is_floating_point():
        raise ValueError("pressure must be a floating 3-D tensor")
    if solid.shape != pressure.shape or solid.dtype is not torch.bool:
        raise ValueError("solid must be bool with the pressure shape")
    if active.shape != pressure.shape or active.dtype is not torch.bool:
        raise ValueError("active must be bool with the pressure shape")
    if any(component.shape != pressure.shape for component in normals):
        raise ValueError("wall normals must share the pressure shape")
    devices = {pressure.device, solid.device, active.device}
    devices.update(component.device for component in normals)
    if len(devices) != 1:
        raise ValueError("pressure and wall geometry tensors must share a device")

    indices = active.nonzero(as_tuple=False)
    requested = int(indices.shape[0])
    vectors = torch.zeros(
        (requested, 3),
        dtype=pressure.dtype,
        device=pressure.device,
    )
    magnitudes = torch.zeros(requested, dtype=pressure.dtype, device=pressure.device)
    valid = torch.zeros(requested, dtype=torch.bool, device=pressure.device)
    if not requested:
        return WallTangentialPressureGradientSamples(
            vectors,
            magnitudes,
            valid,
            requested,
            0,
        )

    nz, ny, nx = pressure.shape
    interior = (
        (indices[:, 0] > 0)
        & (indices[:, 0] < nz - 1)
        & (indices[:, 1] > 0)
        & (indices[:, 1] < ny - 1)
        & (indices[:, 2] > 0)
        & (indices[:, 2] < nx - 1)
    )
    selected = interior.nonzero(as_tuple=False).flatten()
    if not int(selected.numel()):
        return WallTangentialPressureGradientSamples(
            vectors,
            magnitudes,
            valid,
            requested,
            0,
        )
    centers = indices[selected]
    offsets_zyx = torch.tensor(
        [
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dz, dy, dx) != (0, 0, 0)
        ],
        device=pressure.device,
        dtype=torch.long,
    )
    neighbours = centers[:, None, :] + offsets_zyx[None, :, :]
    z, y, x = neighbours.unbind(dim=2)
    neighbour_pressure = pressure[z, y, x]
    center_pressure = pressure[centers[:, 0], centers[:, 1], centers[:, 2]]
    usable = ~solid[z, y, x]
    usable &= torch.isfinite(neighbour_pressure)
    usable &= torch.isfinite(center_pressure)[:, None]

    design = offsets_zyx[:, (2, 1, 0)].to(dtype=pressure.dtype)
    inverse_distance_squared = design.square().sum(dim=1).reciprocal()
    weights = usable.to(dtype=pressure.dtype) * inverse_distance_squared
    matrix = torch.einsum("no,oi,oj->nij", weights, design, design)
    delta = neighbour_pressure - center_pressure[:, None]
    right_hand_side = torch.einsum("no,oi,no->ni", weights, design, delta)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    scale = eigenvalues[:, -1].clamp_min(torch.finfo(pressure.dtype).eps)
    solvable = (
        (eigenvalues[:, 0] > 1.0e-6 * scale)
        & torch.isfinite(matrix).all(dim=(1, 2))
        & torch.isfinite(right_hand_side).all(dim=1)
    )
    local_positions = solvable.nonzero(as_tuple=False).flatten()
    if int(local_positions.numel()):
        gradient = torch.linalg.solve(
            matrix[local_positions],
            right_hand_side[local_positions].unsqueeze(2),
        ).squeeze(2)
        active_centers = centers[local_positions]
        normal = torch.stack(
            [
                component[
                    active_centers[:, 0],
                    active_centers[:, 1],
                    active_centers[:, 2],
                ]
                for component in normals
            ],
            dim=1,
        )
        normal_norm = torch.linalg.vector_norm(normal, dim=1)
        finite_normal = torch.isfinite(normal).all(dim=1) & (normal_norm > 1.0e-8)
        normal = normal / normal_norm[:, None].clamp_min(1.0e-8)
        tangential = gradient - (gradient * normal).sum(dim=1)[:, None] * normal
        finite_gradient = torch.isfinite(tangential).all(dim=1)
        accepted = finite_normal & finite_gradient
        global_positions = selected[local_positions]
        accepted_positions = global_positions[accepted]
        vectors[accepted_positions] = tangential[accepted]
        magnitudes[accepted_positions] = torch.linalg.vector_norm(
            tangential[accepted],
            dim=1,
        )
        valid[accepted_positions] = True

    return WallTangentialPressureGradientSamples(
        vector=vectors,
        magnitude=magnitudes,
        valid=valid,
        requested_nodes=requested,
        valid_nodes=int(valid.sum().item()),
    )


__all__ = [
    "WallPressureGradientAggregate",
    "WallTangentialPressureGradientSamples",
    "aggregate_wall_pressure_gradient_summaries",
    "sample_wall_tangential_pressure_gradient",
]
