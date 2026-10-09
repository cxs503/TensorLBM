"""Two-point log-slope diagnostics for resolved wall-parallel velocity."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TwoPointLogSlopeSummary:
    requested_samples: int
    valid_samples: int
    rejected_fraction: float
    minimum_friction_velocity: float | None
    median_friction_velocity: float | None
    mean_friction_velocity: float | None
    percentile95_friction_velocity: float | None
    maximum_friction_velocity: float | None


def estimate_two_point_log_slope_friction_velocity(
    inner_speed: torch.Tensor,
    outer_speed: torch.Tensor,
    inner_distance: torch.Tensor,
    outer_distance: torch.Tensor,
    *,
    kappa: float = 0.41,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate ``u_tau = kappa Δu / log(y_outer/y_inner)`` sample-wise."""
    fields = (inner_speed, outer_speed, inner_distance, outer_distance)
    if any(field.ndim != 1 or not field.is_floating_point() for field in fields):
        raise ValueError("two-point wall inputs must be floating 1-D tensors")
    if len({field.numel() for field in fields}) != 1:
        raise ValueError("two-point wall inputs must have equal lengths")
    if not 0.0 < kappa < 1.0:
        raise ValueError("kappa must lie in (0,1)")
    valid = torch.ones_like(inner_speed, dtype=torch.bool)
    for field in fields:
        valid &= torch.isfinite(field)
    valid &= inner_speed >= 0.0
    valid &= outer_speed > inner_speed
    valid &= inner_distance > 0.0
    valid &= outer_distance > inner_distance
    ratio = outer_distance / inner_distance.clamp_min(
        torch.finfo(inner_distance.dtype).tiny,
    )
    denominator = torch.log(ratio)
    valid &= torch.isfinite(denominator) & (denominator > 0.0)
    friction_velocity = torch.zeros_like(inner_speed)
    friction_velocity[valid] = (
        kappa * (outer_speed[valid] - inner_speed[valid]) / denominator[valid]
    )
    return friction_velocity, valid


def summarize_two_point_log_slope(
    friction_velocity: torch.Tensor,
    valid: torch.Tensor,
) -> TwoPointLogSlopeSummary:
    """Summarize finite positive two-point estimates without filling rejects."""
    if friction_velocity.ndim != 1 or not friction_velocity.is_floating_point():
        raise ValueError("friction_velocity must be a floating 1-D tensor")
    if valid.shape != friction_velocity.shape or valid.dtype is not torch.bool:
        raise ValueError("valid must be bool with the friction-velocity shape")
    requested = friction_velocity.numel()
    accepted = friction_velocity[valid & torch.isfinite(friction_velocity)]
    count = accepted.numel()
    if not count:
        return TwoPointLogSlopeSummary(
            requested,
            0,
            1.0 if requested else 0.0,
            None,
            None,
            None,
            None,
            None,
        )
    quantiles = torch.quantile(
        accepted.to(dtype=torch.float64),
        torch.tensor(
            (0.5, 0.95),
            device=accepted.device,
            dtype=torch.float64,
        ),
    )
    return TwoPointLogSlopeSummary(
        requested_samples=requested,
        valid_samples=count,
        rejected_fraction=(requested - count) / requested,
        minimum_friction_velocity=float(accepted.min().item()),
        median_friction_velocity=float(quantiles[0].item()),
        mean_friction_velocity=float(accepted.mean().item()),
        percentile95_friction_velocity=float(quantiles[1].item()),
        maximum_friction_velocity=float(accepted.max().item()),
    )


__all__ = [
    "TwoPointLogSlopeSummary",
    "estimate_two_point_log_slope_friction_velocity",
    "summarize_two_point_log_slope",
]
