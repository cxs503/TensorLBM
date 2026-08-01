"""Fail-closed multi-resolution assessment for flat-plate wall-model runs."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .spatial_convergence import assess_spatial_convergence

if TYPE_CHECKING:
    from collections.abc import Sequence


_IDENTITY_FIELDS = (
    "reynolds",
    "resolved_reynolds",
    "lattice_speed",
    "plate_start_fraction",
    "wall_law",
    "ramp_steps",
    "sponge_strength",
    "smagorinsky_cs",
    "positivity_limiter",
)


def assess_flat_plate_convergence(
    records: Sequence[dict[str, object]],
    *,
    maximum_finest_error_pct: float = 2.0,
    maximum_fit_rms_pct: float = 1.0,
    minimum_order: float = 0.5,
    maximum_reference_error_pct: float = 5.0,
) -> dict[str, object]:
    """Validate configuration identity, then fit the friction sequence."""
    if len(records) < 3:
        raise ValueError("flat-plate convergence requires at least three records")
    parsed: list[tuple[float, float, dict[str, object], dict[str, object]]] = []
    schema_valid = True
    single_grid_admitted = True
    for record in records:
        schema_valid &= record.get("schema") == "tensorlbm-flat-plate-wall-model-v3"
        configuration = record.get("configuration")
        result = record.get("result")
        acceptance = record.get("acceptance")
        if not isinstance(configuration, dict) or not isinstance(result, dict):
            raise ValueError("each record needs configuration and result mappings")
        if not isinstance(acceptance, dict):
            raise ValueError("each record needs an acceptance mapping")
        length = float(configuration["plate_length"])
        cf = float(result["friction_coefficient"])
        parsed.append((length, cf, configuration, result))
        single_grid_admitted &= acceptance.get("admitted") is True
    parsed.sort(key=lambda item: item[0])
    resolutions = [item[0] for item in parsed]
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("plate resolutions must be unique")
    baseline = parsed[0][2]
    configuration_identity = all(
        configuration.get(field) == baseline.get(field)
        for _, _, configuration, _ in parsed[1:]
        for field in _IDENTITY_FIELDS
    )
    required_fields_present = all(
        field in configuration
        for _, _, configuration, _ in parsed
        for field in _IDENTITY_FIELDS
    )
    exchange_ratios = [
        float(configuration["stress_exchange_distance"]) / length
        for length, _, configuration, _ in parsed
    ]
    exchange_ratio_invariant = (
        max(exchange_ratios) - min(exchange_ratios) <= 1e-12
    )
    streamwise_ratios = [
        float(configuration["shape_zyx"][2]) / length
        for length, _, configuration, _ in parsed
    ]
    transverse_ratios = [
        float(configuration["shape_zyx"][1]) / length
        for length, _, configuration, _ in parsed
    ]
    domain_ratio_invariant = (
        max(streamwise_ratios) - min(streamwise_ratios) <= 1e-12
        and max(transverse_ratios) - min(transverse_ratios) <= 1e-12
    )
    values = [item[1] for item in parsed]
    spatial = assess_spatial_convergence(resolutions, values)
    reference_values = {
        float(result["ittc_1957_reference"]) for _, _, _, result in parsed
    }
    reference_invariant = len(reference_values) == 1
    reference = next(iter(reference_values)) if reference_invariant else float("nan")
    extrapolated_reference_error = (
        abs(spatial.extrapolated_value - reference) / abs(reference) * 100.0
        if reference_invariant and reference != 0.0 else float("inf")
    )
    spatial_admitted = spatial.meets(
        maximum_finest_error_pct=maximum_finest_error_pct,
        maximum_fit_rms_pct=maximum_fit_rms_pct,
        minimum_order=minimum_order,
    )
    provenance_admitted = (
        schema_valid
        and required_fields_present
        and configuration_identity
        and exchange_ratio_invariant
        and domain_ratio_invariant
        and reference_invariant
    )
    admitted = (
        provenance_admitted
        and single_grid_admitted
        and spatial_admitted
        and extrapolated_reference_error <= maximum_reference_error_pct
    )
    return {
        "schema": "tensorlbm-flat-plate-convergence-v1",
        "resolutions": resolutions,
        "friction_coefficients": values,
        "configuration_identity": {
            "v3_schema": schema_valid,
            "required_fields_present": required_fields_present,
            "identity_fields_equal": configuration_identity,
            "exchange_distance_over_length": exchange_ratios,
            "exchange_ratio_invariant": exchange_ratio_invariant,
            "domain_ratio_invariant": domain_ratio_invariant,
            "reference_invariant": reference_invariant,
            "admitted": provenance_admitted,
        },
        "spatial_convergence": {
            "monotonic": spatial.monotonic,
            "observed_order": spatial.observed_order,
            "extrapolated_friction_coefficient": spatial.extrapolated_value,
            "finest_relative_error_pct": spatial.finest_relative_error_pct,
            "relative_fit_rms_pct": spatial.relative_fit_rms_pct,
            "admitted": spatial_admitted,
        },
        "reference": {
            "ittc_1957": reference,
            "extrapolated_reference_error_pct": extrapolated_reference_error,
            "maximum_reference_error_pct": maximum_reference_error_pct,
        },
        "single_grid_runs_admitted": single_grid_admitted,
        "admitted": admitted,
    }


__all__ = ["assess_flat_plate_convergence"]
