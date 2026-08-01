from __future__ import annotations

import pytest

from tensorlbm.collision_viscosity_audit import (
    CollisionViscosityAuditConfig,
    run_collision_viscosity_audit,
)


@pytest.mark.parametrize(
    "collision_model", ("bgk", "cumulant", "natural_kbc"),
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


def test_collision_viscosity_audit_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="collision_model"):
        CollisionViscosityAuditConfig(collision_model="mrt").validate()
    with pytest.raises(ValueError, match="tau"):
        CollisionViscosityAuditConfig(
            collision_model="bgk", tau=0.5,
        ).validate()


def test_collision_viscosity_audit_is_public() -> None:
    import tensorlbm

    assert tensorlbm.CollisionViscosityAuditConfig is CollisionViscosityAuditConfig
    assert tensorlbm.run_collision_viscosity_audit is run_collision_viscosity_audit
