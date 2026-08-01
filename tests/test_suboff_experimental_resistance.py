from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "examples" / "suboff_experimental_resistance.py"
SPEC = importlib.util.spec_from_file_location("suboff_experimental_resistance", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

ENGINEERING_PATH = Path(__file__).parents[1] / "examples" / "suboff_engineering_resistance.py"
ENGINEERING_SPEC = importlib.util.spec_from_file_location(
    "suboff_engineering_resistance", ENGINEERING_PATH,
)
assert ENGINEERING_SPEC and ENGINEERING_SPEC.loader
engineering = importlib.util.module_from_spec(ENGINEERING_SPEC)
sys.modules[ENGINEERING_SPEC.name] = engineering
ENGINEERING_SPEC.loader.exec_module(engineering)


def test_primary_table_points_are_transcribed() -> None:
    assert module.experimental_point("bare_hull", 5.92).resistance_n == pytest.approx(87.40)
    assert module.experimental_point("full", 5.93).resistance_n == pytest.approx(102.3)
    assert module.experimental_point("bare_hull", 11.84).resistance_n == pytest.approx(332.9)
    assert module.experimental_point("full", 11.85).resistance_n == pytest.approx(389.2)


def test_experimental_appendage_ratio_matches_report() -> None:
    low = 102.3 / 87.40
    high = 389.2 / 332.9
    assert low == pytest.approx(1.1705, rel=1e-4)
    assert high == pytest.approx(1.1691, rel=1e-4)


def test_force_scale_is_positive_and_dimensionally_consistent() -> None:
    scale = module.force_scale_newton(
        rho_water=998.2, dx_m=4.356 / 80.0,
        speed_mps=5.92 * module.KNOT_TO_MPS, lattice_speed=0.06,
    )
    assert scale == pytest.approx(7614.44, rel=2e-3)


def test_non_table_speed_is_rejected() -> None:
    with pytest.raises(ValueError, match="primary Table 14"):
        module.experimental_point("bare_hull", 7.0)


def test_far_field_sponge_damps_six_faces_and_leaves_interior() -> None:
    sponge = module.build_far_field_sponge(
        100, 60, 50, width=10, strength=1.0, device=torch.device("cpu"),
    )
    assert sponge.shape == (50, 60, 100)
    assert sponge[25, 30, 40].item() == 0.0
    assert sponge[25, 30, 0].item() == 1.0
    assert sponge[25, 30, -1].item() == 1.0
    assert sponge[25, 0, 40].item() == 1.0
    assert sponge[0, 30, 40].item() == 1.0


def test_no_penetration_projection_removes_only_near_wall_normal_velocity() -> None:
    shape = (6, 6, 8)
    rho = torch.ones(shape)
    ux = torch.full(shape, 0.06)
    zero = torch.zeros(shape)
    f = module.equilibrium3d(rho, ux, zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[:, :, 4:] = True
    near = module.get_near_wall_3d(solid)
    projected = module.project_no_penetration(f, solid, near)
    _, ux_p, uy_p, uz_p = module.macroscopic3d(projected)
    nx_n, ny_n, nz_n = module.compute_wall_normal(solid, near)
    normal_speed = ux_p * nx_n + uy_p * ny_n + uz_p * nz_n
    assert normal_speed[near].abs().max().item() < 1e-6
    assert ux_p[~near & ~solid].mean().item() == pytest.approx(0.06, abs=1e-6)


def test_endpoint_calibrated_engineering_model_passes_four_holdouts() -> None:
    report = engineering.build_report()
    assert report["physical_validation"] is True
    assert report["cfd_validation"] is False
    assert report["all_holdout_targets_met"] is True
    assert len(report["cases"]) == 2
    for case in report["cases"]:
        assert sum(row["role"] == "calibration" for row in case["rows"]) == 2
        assert sum(row["role"] == "holdout" for row in case["rows"]) == 4
        assert case["holdout_max_absolute_error_pct"] <= 5.0
