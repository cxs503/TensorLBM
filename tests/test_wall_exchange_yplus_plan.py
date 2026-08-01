from __future__ import annotations

import math

import pytest

from tensorlbm.hydrodynamics import ittc57_friction_coefficient
from tensorlbm.yplus_guide import (
    estimate_exchange_yplus,
    plan_exchange_yplus_refinement,
)


def test_exchange_yplus_uses_finest_body_resolution() -> None:
    reynolds = 13_213_381.41322709
    expected = (
        2.109375 / 180.0
        * reynolds
        * math.sqrt(ittc57_friction_coefficient(reynolds) / 2.0)
    )

    estimate = estimate_exchange_yplus(
        physical_reynolds=reynolds,
        characteristic_length_cells=180.0,
        exchange_distance_cells=2.109375,
    )

    assert estimate == pytest.approx(expected)
    assert estimate == pytest.approx(5855.3779636772915)


def test_refinement_plan_finds_one_extra_level_for_l150_minimum_sample() -> None:
    plan = plan_exchange_yplus_refinement(
        physical_reynolds=13_213_381.41322709,
        characteristic_length_cells=300.0,
        minimum_exchange_distance_cells=1.0,
        target_maximum_yplus=1000.0,
    )

    assert plan["required_characteristic_length_cells"] == pytest.approx(
        499.65891956712885,
    )
    assert plan["additional_refinement_levels"] == 1
    assert plan["planned_characteristic_length_cells"] == 600.0
    assert plan["planned_exchange_y_plus_estimate"] == pytest.approx(
        832.7648659452148,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"physical_reynolds": 0.0}, "must be positive"),
        ({"target_maximum_yplus": 0.0}, "must be positive"),
        ({"refinement_ratio": 1}, "at least two"),
    ],
)
def test_refinement_plan_rejects_invalid_inputs(kwargs: dict, message: str) -> None:
    arguments = {
        "physical_reynolds": 1.0e7,
        "characteristic_length_cells": 300.0,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        plan_exchange_yplus_refinement(**arguments)
