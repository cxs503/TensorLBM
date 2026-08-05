from __future__ import annotations

import math

import pytest

from tensorlbm.yplus_guide import grid_quality_metrics


def test_grid_quality_uses_lattice_cell_geometry_without_domain_dx_alias() -> None:
    metrics = grid_quality_metrics(
        nx=450,
        ny=90,
        nz=90,
        hull_length=90.0,
        u_in=0.06,
        re=13_213_381.41322709,
    )
    radius = 90.0 / (2.0 * 8.57)
    expected_blockage = math.pi * radius**2 / (90.0 * 90.0)
    assert metrics["blockage_ratio"] == pytest.approx(expected_blockage)
    assert metrics["parameters"]["dx_lu"] == 1.0
    assert metrics["parameters"]["y_first_lu"] == 0.5
    assert metrics["cells_per_hull_length"] == 90.0
    assert metrics["streamwise_domain_lengths"] == 5.0
    assert metrics["transverse_domain_diameters"] == pytest.approx(8.57)
    assert metrics["blockage_ok"] is True
    assert metrics["quality_tier"] == "recommended"


def test_grid_quality_rejects_high_blockage_from_recommended_tier() -> None:
    metrics = grid_quality_metrics(
        nx=180,
        ny=20,
        nz=20,
        hull_length=90.0,
        u_in=0.06,
        re=1_000_000.0,
    )
    assert metrics["blockage_ratio"] > 0.05
    assert metrics["blockage_ok"] is False
    assert metrics["quality_tier"] != "recommended"


@pytest.mark.parametrize(
    "override",
    (
        {"nx": 0},
        {"ny": True},
        {"hull_length": 0.0},
        {"u_in": float("nan")},
        {"re": 100.0},
        {"hull_radius": -1.0},
    ),
)
def test_grid_quality_rejects_invalid_inputs(override) -> None:
    arguments = {
        "nx": 450,
        "ny": 90,
        "nz": 90,
        "hull_length": 90.0,
        "u_in": 0.06,
        "re": 13_213_381.41322709,
    } | override
    with pytest.raises(ValueError):
        grid_quality_metrics(**arguments)
