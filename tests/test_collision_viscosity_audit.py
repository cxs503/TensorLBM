from __future__ import annotations

import pytest

from tensorlbm.collision_viscosity_audit import (
    CollisionViscosityAuditConfig,
    run_collision_viscosity_audit,
)


@pytest.mark.parametrize(
    "collision_model", (
        "bgk", "cumulant", "planar_cumulant_d2q9",
        "cumulant_wale", "cumulant_vreman", "natural_kbc",
    ),
)
def test_audited_collision_recovers_target_viscosity(
    collision_model: str,
) -> None:
    result = run_collision_viscosity_audit(CollisionViscosityAuditConfig(
        collision_model=collision_model,
        wavelength_cells=24,
        steps=120,
        fit_start_step=15,
    ))

    assert result["acceptance"]["admitted"] is True
    assert result["result"]["relative_error_pct"] < 2.0


def test_current_kbc_is_withheld_when_viscosity_is_not_recovered() -> None:
    result = run_collision_viscosity_audit(CollisionViscosityAuditConfig(
        collision_model="entropic_kbc",
        wavelength_cells=24,
        steps=120,
        fit_start_step=15,
        kbc_max_iterations=12,
    ))

    assert result["acceptance"]["admitted"] is False
    assert result["result"]["relative_error_pct"] > 20.0


def test_planar_cylinder_tau_recovers_viscosity_in_float32() -> None:
    result = run_collision_viscosity_audit(CollisionViscosityAuditConfig(
        collision_model="planar_cumulant_d2q9",
        tau=0.5162,
        wavelength_cells=64,
        transverse_cells=3,
        amplitude=0.01,
        steps=800,
        fit_start_step=100,
        maximum_relative_error_pct=2.0,
        dtype="float32",
    ))
    assert result["acceptance"]["admitted"] is True
    assert result["result"]["relative_error_pct"] < 0.5


def test_mixed_precision_natural_kbc_recovers_near_half_tau() -> None:
    result = run_collision_viscosity_audit(CollisionViscosityAuditConfig(
        collision_model="natural_kbc",
        tau=0.5000162,
        wavelength_cells=16,
        transverse_cells=3,
        amplitude=0.02,
        steps=1200,
        fit_start_step=100,
        maximum_relative_error_pct=5.0,
        dtype="float32",
        natural_kbc_compute_dtype="float64",
    ))
    assert result["acceptance"]["admitted"] is True


def test_audit_rejects_an_under_resolved_decay_signal() -> None:
    result = run_collision_viscosity_audit(CollisionViscosityAuditConfig(
        collision_model="natural_kbc",
        tau=0.5000162,
        wavelength_cells=16,
        transverse_cells=3,
        amplitude=0.02,
        steps=1200,
        fit_start_step=100,
        maximum_relative_error_pct=5.0,
        minimum_fitted_log_decay=0.01,
        dtype="float32",
        natural_kbc_compute_dtype="float64",
    ))
    assert result["result"]["relative_error_pct"] < 5.0
    assert result["acceptance"]["decay_signal_admitted"] is False
    assert result["acceptance"]["admitted"] is False


def test_collision_viscosity_audit_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="collision_model"):
        CollisionViscosityAuditConfig(collision_model="mrt").validate()
    with pytest.raises(ValueError, match="tau"):
        CollisionViscosityAuditConfig(
            collision_model="bgk", tau=0.5,
        ).validate()
    with pytest.raises(ValueError, match="minimum_fitted_log_decay"):
        CollisionViscosityAuditConfig(
            collision_model="bgk", minimum_fitted_log_decay=-1.0,
        ).validate()


def test_collision_viscosity_audit_is_public() -> None:
    import tensorlbm

    assert tensorlbm.CollisionViscosityAuditConfig is CollisionViscosityAuditConfig
    assert tensorlbm.run_collision_viscosity_audit is run_collision_viscosity_audit
