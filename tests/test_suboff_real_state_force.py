"""State-backed SUBOFF D3Q19 link-wise force observer tests."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d
from tensorlbm.marine_geometry import GeometryAsset, compile_d3q19_wall_links
from tensorlbm.suboff_real_state_force import (
    SuboffRealStateForceConfig,
    observe_suboff_real_state_force_window,
)


def _asset(*, device: torch.device = torch.device("cpu")) -> GeometryAsset:
    mask = torch.zeros((5, 5, 5), dtype=torch.bool, device=device)
    mask[2, 2, 2] = True
    return GeometryAsset(mask, "suboff-real-state-test-body", (0.0, 0.0, 0.0), "lattice", "test")


def _equilibrium(asset: GeometryAsset, velocity: float) -> torch.Tensor:
    shape = asset.solid_mask.shape
    rho = torch.ones(shape, dtype=torch.float64, device=asset.solid_mask.device)
    ux = torch.full(shape, velocity, dtype=torch.float64, device=asset.solid_mask.device)
    zero = torch.zeros_like(rho)
    return equilibrium3d(rho, ux, zero, zero)


def _explicit_force(asset: GeometryAsset, state: torch.Tensor) -> tuple[float, float, float]:
    links = compile_d3q19_wall_links(asset)
    total = torch.zeros(3, dtype=state.dtype, device=state.device)
    for index in range(links.count):
        q = int(links.direction[index].item())
        z, y, x = (int(value.item()) for value in links.neighbor_zyx[index])
        total += -2.0 * state[int(OPPOSITE[q].item()), z, y, x] * C[q].to(state)
    return (float(total[0].item()), float(total[1].item()), float(total[2].item()))


def test_observer_consumes_caller_equilibrium_states_and_averages_window() -> None:
    asset = _asset()
    first = _equilibrium(asset, 0.02)
    second = _equilibrium(asset, 0.07)
    first_before, second_before = first.clone(), second.clone()

    result = observe_suboff_real_state_force_window(
        asset,
        [first, second],
        config=SuboffRealStateForceConfig(direction=(1.0, 0.0, 0.0)),
    )

    expected_first = _explicit_force(asset, first)
    expected_second = _explicit_force(asset, second)
    expected_mean = tuple((expected_first[i] + expected_second[i]) / 2.0 for i in range(3))
    assert result.link_count == 18
    assert result.windows == 2
    assert result.window_forces[0] == pytest.approx(expected_first)
    assert result.window_forces[1] == pytest.approx(expected_second)
    assert result.observation.force == pytest.approx(expected_mean)
    assert result.observation.status == "measured"
    assert result.observation.sample_phase == "post_stream_pre_bounce_back"
    assert result.contract.status == "measured_candidate"
    assert result.contract.validated is False
    assert result.physical_validation is False
    assert not torch.equal(first, second)
    assert torch.equal(first, first_before)
    assert torch.equal(second, second_before)


def test_observer_reads_actual_population_not_a_synthetic_velocity_proxy() -> None:
    asset = _asset()
    state = _equilibrium(asset, 0.03)
    altered = state.clone()
    links = compile_d3q19_wall_links(asset)
    q = int(links.direction[0].item())
    z, y, x = (int(value.item()) for value in links.neighbor_zyx[0])
    altered[int(OPPOSITE[q].item()), z, y, x] += 0.125

    baseline = observe_suboff_real_state_force_window(asset, [state])
    observed = observe_suboff_real_state_force_window(asset, [altered])
    expected_delta = -2.0 * 0.125 * C[q].to(torch.float64)
    actual_delta = torch.tensor(observed.observation.force) - torch.tensor(baseline.observation.force)
    assert actual_delta == pytest.approx(expected_delta)


@pytest.mark.parametrize(
    ("bad_state", "error"),
    [
        (torch.zeros((18, 5, 5, 5), dtype=torch.float64), "shape"),
        (torch.zeros((19, 5, 5, 5), dtype=torch.int64), "floating-point"),
    ],
)
def test_observer_rejects_invalid_population_shape_and_dtype(bad_state: torch.Tensor, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        observe_suboff_real_state_force_window(_asset(), [bad_state])


def test_observer_rejects_nonfinite_population() -> None:
    state = _equilibrium(_asset(), 0.03)
    state[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        observe_suboff_real_state_force_window(_asset(), [state])
