"""Axial contribution profiles for wall-model shear force diagnostics."""

from __future__ import annotations

import torch


def summarize_axial_wall_shear(
    axial_coordinate: torch.Tensor,
    shear_force_x: torch.Tensor,
    area_weight: torch.Tensor,
    y_plus: torch.Tensor,
    tangential_speed: torch.Tensor,
    friction_velocity: torch.Tensor,
    *,
    bins: int = 20,
) -> list[dict[str, float | int]]:
    """Bin signed x-shear, wall area and ``y+`` along a body's x extent."""
    fields = (
        axial_coordinate,
        shear_force_x,
        area_weight,
        y_plus,
        tangential_speed,
        friction_velocity,
    )
    if bins < 1:
        raise ValueError("bins must be positive")
    if any(field.ndim != 1 for field in fields):
        raise ValueError("axial wall-shear inputs must be one-dimensional")
    if len({field.numel() for field in fields}) != 1:
        raise ValueError("axial wall-shear inputs must have equal lengths")
    if not axial_coordinate.numel():
        return []
    finite = torch.ones_like(axial_coordinate, dtype=torch.bool)
    for field in fields:
        finite &= torch.isfinite(field)
    finite &= area_weight >= 0.0
    coordinate = axial_coordinate[finite].to(dtype=torch.float64)
    shear = shear_force_x[finite].to(dtype=torch.float64)
    area = area_weight[finite].to(dtype=torch.float64)
    local_y_plus = y_plus[finite].to(dtype=torch.float64)
    local_tangential_speed = tangential_speed[finite].to(dtype=torch.float64)
    local_friction_velocity = friction_velocity[finite].to(dtype=torch.float64)
    if not coordinate.numel():
        return []
    lower = coordinate.min()
    span = (coordinate.max() - lower).clamp_min(1.0)
    normalized = (coordinate - lower) / span
    bin_index = torch.clamp((normalized * bins).to(torch.long), max=bins - 1)
    total_signed = shear.sum()
    total_absolute = shear.abs().sum()
    tiny = torch.finfo(torch.float64).tiny
    result: list[dict[str, float | int]] = []
    for index in range(bins):
        selected = bin_index == index
        count = int(selected.sum().item())
        if not count:
            continue
        bin_shear = shear[selected]
        bin_area = area[selected]
        bin_y_plus = local_y_plus[selected]
        bin_tangential_speed = local_tangential_speed[selected]
        bin_friction_velocity = local_friction_velocity[selected]
        signed_sum = bin_shear.sum()
        absolute_sum = bin_shear.abs().sum()
        y_plus_quantiles = torch.quantile(
            bin_y_plus,
            torch.tensor((0.5, 0.95), device=coordinate.device, dtype=torch.float64),
        )
        tangential_speed_quantiles = torch.quantile(
            bin_tangential_speed,
            torch.tensor((0.5, 0.95), device=coordinate.device, dtype=torch.float64),
        )
        friction_velocity_quantiles = torch.quantile(
            bin_friction_velocity,
            torch.tensor((0.5, 0.95), device=coordinate.device, dtype=torch.float64),
        )
        area_sum = bin_area.sum()
        result.append(
            {
                "bin": index,
                "normalized_x_lower": index / bins,
                "normalized_x_upper": (index + 1) / bins,
                "sample_count": count,
                "area_sum_lu2": float(area_sum.item()),
                "signed_shear_x_sum_lu": float(signed_sum.item()),
                "absolute_shear_x_sum_lu": float(absolute_sum.item()),
                "signed_shear_x_fraction": float(
                    (signed_sum / total_signed.abs().clamp_min(tiny)).item(),
                ),
                "absolute_shear_x_fraction": float(
                    (absolute_sum / total_absolute.clamp_min(tiny)).item(),
                ),
                "mean_signed_shear_x_per_area_lu": float(
                    (signed_sum / area_sum.clamp_min(tiny)).item(),
                ),
                "median_y_plus": float(y_plus_quantiles[0].item()),
                "percentile95_y_plus": float(y_plus_quantiles[1].item()),
                "median_tangential_speed_lu": float(
                    tangential_speed_quantiles[0].item(),
                ),
                "percentile95_tangential_speed_lu": float(
                    tangential_speed_quantiles[1].item(),
                ),
                "median_friction_velocity_lu": float(
                    friction_velocity_quantiles[0].item(),
                ),
                "percentile95_friction_velocity_lu": float(
                    friction_velocity_quantiles[1].item(),
                ),
            }
        )
    return result


__all__ = ["summarize_axial_wall_shear"]
