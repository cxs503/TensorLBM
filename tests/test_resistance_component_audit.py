from __future__ import annotations

import pytest

from tensorlbm.resistance_component_audit import audit_resistance_components


def test_resistance_component_audit_preserves_direct_observables() -> None:
    result = audit_resistance_components(
        total_resistance=110.0,
        pressure_resistance=25.0,
        wall_shear_resistance=86.0,
        experimental_total=88.0,
        friction_reference=80.0,
    )

    assert result.component_sum == pytest.approx(111.0)
    assert result.component_sum_vs_total_pct == pytest.approx(100.0 / 110.0)
    assert result.total_reference_error_pct == pytest.approx(25.0)
    assert result.wall_shear_vs_friction_reference_pct == pytest.approx(7.5)
    assert result.inferred_experimental_residual == pytest.approx(8.0)
    assert result.pressure_over_inferred_residual == pytest.approx(3.125)
    assert result.to_dict()["scope"] == "diagnostic_only_not_a_cfd_correction"


def test_resistance_component_audit_omits_nonpositive_inferred_residual() -> None:
    result = audit_resistance_components(
        total_resistance=10.0,
        pressure_resistance=4.0,
        wall_shear_resistance=6.0,
        experimental_total=8.0,
        friction_reference=9.0,
    )

    assert result.inferred_experimental_residual is None
    assert result.pressure_over_inferred_residual is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"experimental_total": 0.0},
        {"pressure_resistance": float("nan")},
        {"friction_reference": -1.0},
    ],
)
def test_resistance_component_audit_rejects_invalid_inputs(kwargs: dict) -> None:
    values = {
        "total_resistance": 10.0,
        "pressure_resistance": 4.0,
        "wall_shear_resistance": 6.0,
        "experimental_total": 8.0,
        "friction_reference": 5.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        audit_resistance_components(**values)
