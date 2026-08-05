from __future__ import annotations

import copy

import pytest

from tensorlbm.sphere_boundary_sensitivity import assess_sphere_inlet_sponge_pair


def _record(sponge_inlet: bool, cd: float) -> dict[str, object]:
    return {
        "schema": "tensorlbm-sphere-bfl-control-volume-v3",
        "configuration": {
            "schema_version": 3,
            "shape_zyx": [144, 144, 216],
            "radius": 9.0,
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
            "sponge_inlet": sponge_inlet,
            "cv_margin": 6,
            "far_field_mode": "non_equilibrium_extrapolation",
        },
        "result": {
            "cd_control_volume": cd,
            "cd_reference_schiller_naumann": 1.0917310910948732,
        },
        "acceptance": {"numerical_quality_admitted": True},
    }


def test_sphere_inlet_sponge_pair_admits_small_matched_change() -> None:
    result = assess_sphere_inlet_sponge_pair([
        _record(True, 1.1756), _record(False, 1.1652),
    ])

    assert result["configuration_identity"]["admitted"] is True
    assert result["boundary_sensitivity"]["drag_change_pct"] < 1.0
    assert result["admitted_as_boundary_sensitivity"] is True
    assert result["physical_validation"] is False


def test_sphere_inlet_sponge_pair_rejects_changed_collision() -> None:
    enabled = copy.deepcopy(_record(True, 1.1756))
    enabled["configuration"]["compile_natural_kbc"] = False
    result = assess_sphere_inlet_sponge_pair([
        _record(False, 1.1652), enabled,
    ])

    assert result["configuration_identity"]["identity_fields_equal"] is False
    assert result["admitted_as_boundary_sensitivity"] is False


def test_sphere_inlet_sponge_pair_requires_both_switch_states() -> None:
    with pytest.raises(ValueError, match="disabled and one enabled"):
        assess_sphere_inlet_sponge_pair([
            _record(False, 1.1652), _record(False, 1.1653),
        ])


def test_sphere_inlet_sponge_pair_rejects_numerically_bad_source() -> None:
    disabled = _record(False, 1.1652)
    disabled["acceptance"]["numerical_quality_admitted"] = False
    result = assess_sphere_inlet_sponge_pair([
        disabled, _record(True, 1.1756),
    ])

    assert result["source_numerical_quality_admitted"] is False
    assert result["admitted_as_boundary_sensitivity"] is False
