"""Diagnostic-only decomposition audit for direct CFD resistance."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ResistanceComponentAudit:
    total_resistance: float
    pressure_resistance: float
    wall_shear_resistance: float
    experimental_total: float
    component_sum: float
    component_sum_vs_total_pct: float
    total_reference_bias_pct: float
    total_reference_error_pct: float
    friction_reference: float | None
    wall_shear_vs_friction_reference_pct: float | None
    inferred_experimental_residual: float | None
    pressure_over_inferred_residual: float | None

    def to_dict(self) -> dict[str, float | None | str]:
        return vars(self) | {
            "scope": "diagnostic_only_not_a_cfd_correction",
        }


def audit_resistance_components(
    *,
    total_resistance: float,
    pressure_resistance: float,
    wall_shear_resistance: float,
    experimental_total: float,
    friction_reference: float | None = None,
) -> ResistanceComponentAudit:
    """Compare direct force components without modifying any CFD observable.

    ``friction_reference`` may be an ITTC line or another independently
    declared scale.  The experimental residual is only reported when that
    scale is below the measured total; it is never subtracted from CFD.
    """
    values = (
        total_resistance,
        pressure_resistance,
        wall_shear_resistance,
        experimental_total,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("resistance values must be finite")
    if experimental_total <= 0.0:
        raise ValueError("experimental total must be positive")
    if friction_reference is not None and (
        not math.isfinite(friction_reference) or friction_reference < 0.0
    ):
        raise ValueError("friction reference must be finite and non-negative")

    component_sum = pressure_resistance + wall_shear_resistance
    inferred_residual = None
    shear_scale_error = None
    pressure_over_residual = None
    if friction_reference is not None:
        shear_scale_error = (
            wall_shear_resistance / max(abs(friction_reference), 1.0e-30) - 1.0
        ) * 100.0
        candidate_residual = experimental_total - friction_reference
        if candidate_residual > 0.0:
            inferred_residual = candidate_residual
            pressure_over_residual = pressure_resistance / candidate_residual
    total_reference_bias_pct = (total_resistance / experimental_total - 1.0) * 100.0
    return ResistanceComponentAudit(
        total_resistance=total_resistance,
        pressure_resistance=pressure_resistance,
        wall_shear_resistance=wall_shear_resistance,
        experimental_total=experimental_total,
        component_sum=component_sum,
        component_sum_vs_total_pct=(
            (component_sum / max(abs(total_resistance), 1.0e-30) - 1.0) * 100.0
        ),
        total_reference_bias_pct=total_reference_bias_pct,
        total_reference_error_pct=abs(total_reference_bias_pct),
        friction_reference=friction_reference,
        wall_shear_vs_friction_reference_pct=shear_scale_error,
        inferred_experimental_residual=inferred_residual,
        pressure_over_inferred_residual=pressure_over_residual,
    )


__all__ = ["ResistanceComponentAudit", "audit_resistance_components"]
