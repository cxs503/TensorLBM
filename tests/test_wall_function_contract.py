"""TDD acceptance tests for the deliberately narrow wall-function matrix."""

import pytest

from tensorlbm.wall_function_contract import (
    WITHHELD_UNVERIFIED_COMBINATION,
    WallFunctionCapability,
    WallFunctionCompatibilityError,
    WallFunctionRequest,
    ValidationLevel,
    assess_wall_function,
    require_wall_function,
    wall_function_capability_matrix,
)


def test_matrix_is_explicit_about_implementation_and_validation_levels() -> None:
    matrix = wall_function_capability_matrix()

    assert matrix[WallFunctionCapability.DISTANCE_FMM].validation is ValidationLevel.IMPLEMENTATION_ONLY
    log = matrix[WallFunctionCapability.LOG_LAW_BODY_FORCE]
    assert log.lattices == frozenset({"D3Q19"})
    assert log.collisions == frozenset({"MRT_SMAGORINSKY"})
    assert log.validation is ValidationLevel.IMPLEMENTATION_ONLY
    assert matrix[WallFunctionCapability.MOVING_BOUNCE_BACK].lattices == frozenset({"D3Q19"})


def test_documented_d3q19_log_law_runner_combination_is_recognised_but_not_promoted() -> None:
    result = assess_wall_function(
        WallFunctionRequest(
            capability=WallFunctionCapability.LOG_LAW_BODY_FORCE,
            lattice="D3Q19",
            physics="single_phase_incompressible",
            collision="MRT_SMAGORINSKY",
            geometry="static_voxel_solid",
            backend="torch",
        )
    )

    assert result.compatible
    assert result.validation is ValidationLevel.IMPLEMENTATION_ONLY
    assert "not physical validation" in result.note


def test_unverified_d3q27_or_free_surface_log_law_combinations_fail_closed() -> None:
    for request in (
        WallFunctionRequest(
            WallFunctionCapability.LOG_LAW_BODY_FORCE, "D3Q27", "single_phase_incompressible",
            "MRT_SMAGORINSKY", "static_voxel_solid", "torch",
        ),
        WallFunctionRequest(
            WallFunctionCapability.LOG_LAW_BODY_FORCE, "D3Q19", "free_surface",
            "MRT_SMAGORINSKY", "static_voxel_solid", "torch",
        ),
    ):
        with pytest.raises(WallFunctionCompatibilityError, match=WITHHELD_UNVERIFIED_COMBINATION):
            require_wall_function(request)


def test_validation_floor_fails_closed_when_only_implementation_exists() -> None:
    request = WallFunctionRequest(
        WallFunctionCapability.DISTANCE_FMM, "MASK_3D", "mask_only",
        "none", "static_voxel_solid", "torch",
    )
    with pytest.raises(WallFunctionCompatibilityError, match="WITHHELD_VALIDATION_LEVEL"):
        require_wall_function(request, minimum_validation=ValidationLevel.NUMERICAL_REGRESSION)


# ---------------------------------------------------------------------------
# Common wall-function capability (wall_function_common)
# ---------------------------------------------------------------------------

def test_common_wall_function_supports_d3q19_and_d3q27() -> None:
    matrix = wall_function_capability_matrix()
    record = matrix[WallFunctionCapability.COMMON_WALL_FUNCTION]
    assert record.lattices == frozenset({"D3Q19", "D3Q27"})
    assert record.validation is ValidationLevel.IMPLEMENTATION_ONLY
    assert "solver-agnostic" in record.note.lower()
    assert "wall_function(f, mask, u_tau, y_plus" in record.note


def test_common_wall_function_admits_d3q19_and_d3q27_at_implementation_level() -> None:
    for lattice in ("D3Q19", "D3Q27"):
        result = assess_wall_function(
            WallFunctionRequest(
                capability=WallFunctionCapability.COMMON_WALL_FUNCTION,
                lattice=lattice,
                physics="single_phase_incompressible",
                collision="BGK",
                geometry="static_voxel_solid",
                backend="torch",
            )
        )
        assert result.compatible
        assert result.validation is ValidationLevel.IMPLEMENTATION_ONLY


def test_common_wall_function_supports_multiple_collisions() -> None:
    """The common wall function is not bound to a specific collision."""
    for collision in ("BGK", "MRT", "CM", "KBC"):
        result = assess_wall_function(
            WallFunctionRequest(
                capability=WallFunctionCapability.COMMON_WALL_FUNCTION,
                lattice="D3Q27",
                physics="single_phase_incompressible",
                collision=collision,
                geometry="static_voxel_solid",
                backend="torch",
            )
        )
        assert result.compatible, f"collision={collision} should be admitted"


def test_common_wall_function_withholds_free_surface() -> None:
    with pytest.raises(WallFunctionCompatibilityError, match=WITHHELD_UNVERIFIED_COMBINATION):
        require_wall_function(
            WallFunctionRequest(
                capability=WallFunctionCapability.COMMON_WALL_FUNCTION,
                lattice="D3Q19",
                physics="free_surface",
                collision="BGK",
                geometry="static_voxel_solid",
                backend="torch",
            )
        )
