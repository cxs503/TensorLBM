"""Unit tests for public, reusable ship-hydrodynamics utilities."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tensorlbm.hydrodynamics import ittc57_friction_coefficient, voxel_wetted_area


def test_ittc57_friction_coefficient_known_values() -> None:
    assert ittc57_friction_coefficient(1.0e5) == pytest.approx(1.0 / 120.0)
    assert ittc57_friction_coefficient(1.0e6) == pytest.approx(0.0046875)
    assert ittc57_friction_coefficient(1.0e7) == pytest.approx(0.003)


def test_ittc57_friction_coefficient_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="Reynolds number too low"):
        ittc57_friction_coefficient(100.0)


def test_voxel_wetted_area_counts_exposed_faces() -> None:
    mask = torch.zeros((3, 3, 3), dtype=torch.bool)
    mask[1, 1, 1] = True
    assert voxel_wetted_area(mask, 0.5) == pytest.approx(1.5)

    mask[1, 1, 2] = True
    assert voxel_wetted_area(mask, 0.5) == pytest.approx(2.5)


def test_hull_cases_do_not_import_suboff_private_hydrodynamics() -> None:
    root = Path(__file__).resolve().parents[1]
    case_files = (
        root / "src/tensorlbm/hull_free_surface_v2.py",
        root / "examples/hull_fs_d3q27_pf.py",
    )
    forbidden = ("_ittc57_friction_coefficient", "_voxel_wetted_area")
    for case_file in case_files:
        source = case_file.read_text(encoding="utf-8")
        assert "suboff_resistance import" not in source
        assert not any(name in source for name in forbidden)
