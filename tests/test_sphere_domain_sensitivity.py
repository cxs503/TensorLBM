from __future__ import annotations

import copy

import pytest

from tensorlbm.sphere_domain_sensitivity import (
    assess_sphere_domain_sensitivity_pair,
)


def _record(width_radii: float, cd: float) -> dict[str, object]:
    radius = 9.0
    return {
        "schema": "tensorlbm-sphere-bfl-control-volume-v3",
        "configuration": {
            "schema_version": 3,
            "shape_zyx": [width_radii * radius, width_radii * radius, 24 * radius],
            "radius": radius,
            "center_x_fraction": 0.3,
            "reynolds": 100.0,
            "lattice_speed": 0.06,
            "collision_model": "natural_kbc_d3q19",
            "collision_chunk_cells": 262144,
            "compile_natural_kbc": True,
            "steps": 7200,
            "warmup_steps": 4800,
            "ramp_steps": 720,
            "statistics_window_steps": 2400,
            "minimum_statistics_convective_times": 5.0,
            "report_interval": 360,
            "sponge_width": 18,
            "sponge_strength": 0.2,
            "sponge_inlet": True,
            "cv_margin": 6,
            "far_field_mode": "non_equilibrium_extrapolation",
        },
        "result": {
            "cd_control_volume": cd,
            "cd_reference_schiller_naumann": 1.0917310910948732,
        },
        "acceptance": {"numerical_quality_admitted": True},
    }


def test_sphere_domain_pair_admits_small_direct_drag_change() -> None:
    result = assess_sphere_domain_sensitivity_pair([
        _record(16.0, 1.1756), _record(20.0, 1.1700),
    ])

    assert result["configuration_identity"]["admitted"] is True
    assert result["domain_sensitivity"]["drag_change_pct"] < 1.0
    assert result["admitted_as_pair_sensitivity"] is True
    assert result["physical_validation"] is False
    assert result["pair_only_not_domain_convergence"] is True


def test_sphere_domain_pair_rejects_changed_streamwise_domain() -> None:
    baseline = _record(16.0, 1.1756)
    expanded = _record(20.0, 1.1700)
    expanded["configuration"]["shape_zyx"][2] += 1

    result = assess_sphere_domain_sensitivity_pair([baseline, expanded])

    assert result["configuration_identity"]["streamwise_cells_fixed"] is False
    assert result["admitted_as_pair_sensitivity"] is False


def test_sphere_domain_pair_rejects_numerically_rejected_source() -> None:
    expanded = _record(20.0, 1.1700)
    expanded["acceptance"]["numerical_quality_admitted"] = False
    result = assess_sphere_domain_sensitivity_pair([
        _record(16.0, 1.1756), expanded,
    ])

    assert result["source_numerical_quality_admitted"] is False
    assert result["admitted_as_pair_sensitivity"] is False


def test_sphere_domain_pair_rejects_changed_numerics() -> None:
    expanded = copy.deepcopy(_record(20.0, 1.1700))
    expanded["configuration"]["compile_natural_kbc"] = False
    result = assess_sphere_domain_sensitivity_pair([
        _record(16.0, 1.1756), expanded,
    ])

    assert result["configuration_identity"]["identity_fields_equal"] is False


def test_sphere_domain_pair_requires_exactly_two_records() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        assess_sphere_domain_sensitivity_pair([_record(16.0, 1.1756)])
