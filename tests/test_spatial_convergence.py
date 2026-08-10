from __future__ import annotations

import pytest

from tensorlbm.spatial_convergence import assess_spatial_convergence


def test_manufactured_second_order_sequence_recovers_limit_and_order() -> None:
    resolutions = [20.0, 30.0, 45.0, 60.0]
    values = [1.25 + 3.0 / resolution**2 for resolution in resolutions]
    result = assess_spatial_convergence(resolutions, values)
    assert result.monotonic is True
    assert result.observed_order == pytest.approx(2.0, rel=2e-5)
    assert result.extrapolated_value == pytest.approx(1.25, rel=1e-8)
    assert result.relative_fit_rms_pct < 1e-8
    assert result.meets(
        maximum_finest_error_pct=0.1,
        maximum_fit_rms_pct=0.1,
    )


def test_nonmonotonic_sequence_is_not_admitted() -> None:
    result = assess_spatial_convergence([10, 20, 40], [1.0, 1.2, 1.1])
    assert result.monotonic is False
    assert not result.meets(
        maximum_finest_error_pct=10.0,
        maximum_fit_rms_pct=10.0,
    )


def test_resolution_contract_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least three"):
        assess_spatial_convergence([10, 20], [1.0, 1.1])
    with pytest.raises(ValueError, match="strictly increasing"):
        assess_spatial_convergence([10, 10, 20], [1.0, 1.1, 1.2])


def test_spatial_convergence_assessment_is_public() -> None:
    import tensorlbm

    assert tensorlbm.assess_spatial_convergence is assess_spatial_convergence
