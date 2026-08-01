"""Fail-closed lateral-domain convergence for the Re=100 cylinder."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


_IDENTITY_FIELDS = (
    "schema_version", "radius", "center_x_fraction", "reynolds",
    "lattice_speed", "collision_model", "warmup_steps", "ramp_steps",
    "sponge_width", "sponge_strength", "sponge_inlet", "cv_margin",
    "far_field_mode", "periodic_axes", "link_force_frame", "steps",
    "report_interval", "statistics_window_steps_resolved",
    "minimum_shedding_cycles",
)


def _monotonic(values: Sequence[float]) -> bool:
    differences = [
        right - left for left, right in zip(values, values[1:], strict=False)
    ]
    return all(value >= 0.0 for value in differences) or all(
        value <= 0.0 for value in differences
    )


def _relative_change(previous: float, finest: float) -> float:
    return abs(finest - previous) / max(abs(finest), 1.0e-30) * 100.0


def _reference_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-30) * 100.0


def assess_cylinder_domain_convergence(
    records: Sequence[dict[str, object]],
    *,
    maximum_finest_domain_change_pct: float = 1.0,
    maximum_reference_error_pct: float = 5.0,
) -> dict[str, object]:
    """Assess direct 20D/30D/40D-style lateral blockage convergence.

    No empirical blockage correction or assumed asymptotic order is used.
    Admission requires identical resolution/time/numerics, monotonically
    expanding lateral domains, monotonic drag and shedding sequences, a small
    change between the two widest domains, and direct agreement of the widest
    domain with both reference observables.
    """
    if len(records) < 3:
        raise ValueError("cylinder domain convergence requires at least three records")
    if maximum_finest_domain_change_pct <= 0.0:
        raise ValueError("maximum finest domain change must be positive")
    if maximum_reference_error_pct <= 0.0:
        raise ValueError("maximum reference error must be positive")

    parsed = []
    schema_valid = True
    source_quality = True
    for record in records:
        schema_valid &= record.get("schema") == "tensorlbm-cylinder-bfl-control-volume-v4"
        configuration = record.get("configuration")
        result = record.get("result")
        acceptance = record.get("acceptance")
        if not isinstance(configuration, dict) or not isinstance(result, dict):
            raise ValueError("each record needs configuration and result mappings")
        if not isinstance(acceptance, dict):
            raise ValueError("each record needs an acceptance mapping")
        shape = configuration.get("shape_zyx")
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError("each record needs a three-dimensional shape_zyx")
        radius = float(configuration["radius"])
        lateral_width_diameters = float(shape[1]) / (2.0 * radius)
        parsed.append((
            lateral_width_diameters,
            float(result["cd_control_volume"]),
            float(result["strouhal"]),
            configuration,
            result,
        ))
        source_quality &= acceptance.get("numerical_quality_admitted") is True
    parsed.sort(key=lambda item: item[0])
    widths = [item[0] for item in parsed]
    if len(set(widths)) != len(widths) or any(width <= 0.0 for width in widths):
        raise ValueError("lateral domain widths must be unique and positive")

    baseline = parsed[0][3]
    required_present = all(
        field in configuration
        for *_, configuration, _ in parsed
        for field in (*_IDENTITY_FIELDS, "shape_zyx", "domain_clearance_diameters")
    )
    identity_equal = required_present and all(
        configuration.get(field) == baseline.get(field)
        for *_, configuration, _ in parsed[1:]
        for field in _IDENTITY_FIELDS
    )
    fixed_axes_equal = required_present and all(
        configuration["shape_zyx"][axis] == baseline["shape_zyx"][axis]
        for *_, configuration, _ in parsed[1:]
        for axis in (0, 2)
    )
    recorded_lateral_clearances = [
        float(configuration["domain_clearance_diameters"]["lateral_center_distance"])
        for *_, configuration, _ in parsed
    ] if required_present else []
    clearance_consistent = required_present and all(
        math.isclose(clearance, width / 2.0, rel_tol=0.0, abs_tol=1.0e-12)
        for clearance, width in zip(
            recorded_lateral_clearances, widths, strict=True,
        )
    )
    provenance_admitted = (
        schema_valid and required_present and identity_equal
        and fixed_axes_equal and clearance_consistent
    )

    cd_values = [item[1] for item in parsed]
    st_values = [item[2] for item in parsed]
    cd_change = _relative_change(cd_values[-2], cd_values[-1])
    st_change = _relative_change(st_values[-2], st_values[-1])
    cd_references = {float(item[4]["cd_reference"]) for item in parsed}
    st_references = {float(item[4]["strouhal_reference"]) for item in parsed}
    references_invariant = len(cd_references) == 1 and len(st_references) == 1
    cd_reference = next(iter(cd_references)) if len(cd_references) == 1 else math.nan
    st_reference = next(iter(st_references)) if len(st_references) == 1 else math.nan
    cd_reference_error = _reference_error(cd_values[-1], cd_reference)
    st_reference_error = _reference_error(st_values[-1], st_reference)
    cd_monotonic = _monotonic(cd_values)
    st_monotonic = _monotonic(st_values)
    domain_change_admitted = (
        cd_monotonic and st_monotonic
        and cd_change <= maximum_finest_domain_change_pct
        and st_change <= maximum_finest_domain_change_pct
    )
    admitted = (
        provenance_admitted and source_quality and references_invariant
        and domain_change_admitted
        and cd_reference_error <= maximum_reference_error_pct
        and st_reference_error <= maximum_reference_error_pct
    )
    return {
        "schema": "tensorlbm-cylinder-domain-convergence-v1",
        "lateral_width_diameters": widths,
        "cd_control_volume": cd_values,
        "strouhal": st_values,
        "configuration_identity": {
            "v4_schema": schema_valid,
            "required_fields_present": required_present,
            "identity_fields_equal": identity_equal,
            "spanwise_and_streamwise_cells_invariant": fixed_axes_equal,
            "recorded_lateral_clearance_diameters": recorded_lateral_clearances,
            "clearance_consistent_with_shape": clearance_consistent,
            "admitted": provenance_admitted,
        },
        "domain_convergence": {
            "drag_monotonic": cd_monotonic,
            "strouhal_monotonic": st_monotonic,
            "finest_drag_change_pct": cd_change,
            "finest_strouhal_change_pct": st_change,
            "maximum_change_pct": maximum_finest_domain_change_pct,
            "admitted": domain_change_admitted,
        },
        "reference": {
            "cd": cd_reference,
            "strouhal": st_reference,
            "finest_cd_error_pct": cd_reference_error,
            "finest_strouhal_error_pct": st_reference_error,
            "maximum_error_pct": maximum_reference_error_pct,
            "invariant": references_invariant,
        },
        "source_numerical_quality_admitted": source_quality,
        "physical_validation": admitted,
        "admitted": admitted,
    }


__all__ = ["assess_cylinder_domain_convergence"]
