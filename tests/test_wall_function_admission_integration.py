"""Integration tests for the public wall-function admission boundary."""

import pytest

from tensorlbm.dg_lbm import DGLBMConfig, DGLBMSuboffConfig
from tensorlbm.suboff_resistance import SuboffResistanceBenchmarkConfig
from tensorlbm.wall_function_admission import (
    WallFunctionRunRequest,
    require_wall_function_run,
)
from tensorlbm.wall_function_contract import (
    WITHHELD_UNVERIFIED_COMBINATION,
    ValidationLevel,
    WallFunctionCapability,
    WallFunctionCompatibilityError,
)


def _log_law_request(**overrides: object) -> WallFunctionRunRequest:
    values: dict[str, object] = {
        "capability": WallFunctionCapability.LOG_LAW_BODY_FORCE,
        "lattice": "D3Q19",
        "physics": "single_phase_incompressible",
        "collision": "MRT_SMAGORINSKY",
        "geometry": "static_voxel_solid",
        "backend": "torch",
    }
    values.update(overrides)
    return WallFunctionRunRequest(**values)  # type: ignore[arg-type]


def test_public_admission_accepts_only_a_named_d3q19_tuple_at_implementation_level() -> None:
    record = require_wall_function_run(_log_law_request())

    assert record.validation is ValidationLevel.IMPLEMENTATION_ONLY


@pytest.mark.parametrize(
    "overrides",
    [
        {"lattice": "D3Q27"},
        {"free_surface": True},
        {"adaptive_mesh": True},
    ],
)
def test_public_admission_withholds_d3q27_free_surface_and_amr(overrides: dict[str, object]) -> None:
    with pytest.raises(WallFunctionCompatibilityError, match=WITHHELD_UNVERIFIED_COMBINATION):
        require_wall_function_run(_log_law_request(**overrides))


def test_public_admission_withholds_a_higher_evidence_floor() -> None:
    with pytest.raises(WallFunctionCompatibilityError, match="WITHHELD_VALIDATION_LEVEL"):
        require_wall_function_run(_log_law_request(
            minimum_validation=ValidationLevel.NUMERICAL_REGRESSION,
        ))


def test_dg_public_configs_withhold_incomplete_legacy_wall_requests() -> None:
    with pytest.raises(WallFunctionCompatibilityError, match=WITHHELD_UNVERIFIED_COMBINATION):
        DGLBMConfig(use_wall_model=True).validate()
    with pytest.raises(WallFunctionCompatibilityError, match=WITHHELD_UNVERIFIED_COMBINATION):
        DGLBMSuboffConfig(use_wall_model=True).validate()


def test_dg_suboff_public_log_law_request_is_admitted_only_at_implementation_level() -> None:
    config = DGLBMSuboffConfig(use_wall_function=True)
    config.validate()


def test_suboff_config_with_amr_wall_model_is_withheld_before_runner_execution() -> None:
    with pytest.raises(WallFunctionCompatibilityError, match=WITHHELD_UNVERIFIED_COMBINATION):
        SuboffResistanceBenchmarkConfig(use_wall_model=True, use_adaptive_mesh=True)


# ---------------------------------------------------------------------------
# HullFreeSurfaceV2Config — a D3Q19 free-surface hull runner whose default
# use_wall_function=True couples the log-law body force with a free-surface
# Körner step and a CG-KBC/cumulant/cascaded collision.  None of those tuples
# is in the audited D3Q19/MRT-Smagorinsky/single-phase matrix, so the public
# config entry must fail closed at construction time.
# ---------------------------------------------------------------------------

def test_hull_free_surface_v2_default_wall_function_is_withheld_before_run() -> None:
    from tensorlbm.hull_free_surface_v2 import HullFreeSurfaceV2Config

    with pytest.raises(WallFunctionCompatibilityError, match=WITHHELD_UNVERIFIED_COMBINATION):
        HullFreeSurfaceV2Config()


def test_hull_free_surface_v2_double_body_wall_function_is_also_withheld() -> None:
    """Even without free-surface, the CG-KBC collision is not the audited
    MRT+Smagorinsky tuple, so the wall function is still withheld."""
    from tensorlbm.hull_free_surface_v2 import HullFreeSurfaceV2Config

    with pytest.raises(WallFunctionCompatibilityError, match=WITHHELD_UNVERIFIED_COMBINATION):
        HullFreeSurfaceV2Config(use_wall_function=True, use_free_surface=False)


def test_hull_free_surface_v2_without_wall_function_is_not_gated() -> None:
    """Disabling the wall function bypasses admission entirely."""
    from tensorlbm.hull_free_surface_v2 import HullFreeSurfaceV2Config

    HullFreeSurfaceV2Config(use_wall_function=False)
