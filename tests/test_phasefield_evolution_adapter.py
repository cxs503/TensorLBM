import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.phasefield.evolution_adapter import (
    COLLISION_ONLY_STAGE,
    FreeEnergyCollisionOnlyConfig,
    FreeEnergyCollisionOnlyState,
    initialize_free_energy_collision_only_state,
    run_free_energy_collision_only,
)


def test_initialization_uses_production_g_initializer_and_declares_state_contract(monkeypatch):
    import tensorlbm.phasefield.evolution_adapter as adapter

    real_initializer = adapter.init_free_energy_g_3d
    calls: list[torch.Tensor] = []

    def recording_initializer(phi: torch.Tensor) -> torch.Tensor:
        calls.append(phi)
        return real_initializer(phi)

    monkeypatch.setattr(adapter, "init_free_energy_g_3d", recording_initializer)
    phi = torch.zeros((3, 4, 5), dtype=torch.float32)
    state = initialize_free_energy_collision_only_state(phi)

    assert len(calls) == 1
    assert state.f.shape == (19, 3, 4, 5)
    assert state.g.shape == (19, 3, 4, 5)
    assert state.f.device == state.g.device == phi.device
    assert state.f.dtype == state.g.dtype == phi.dtype
    assert COLLISION_ONLY_STAGE == "collision_only"


def test_runner_passes_updated_f_and_g_to_the_real_production_step_for_two_steps(monkeypatch):
    import tensorlbm.phasefield.evolution_adapter as adapter

    real_step = adapter.free_energy_step_3d
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def recording_step(f: torch.Tensor, g: torch.Tensor, **kwargs: object):
        calls.append((f, g))
        return real_step(f, g, **kwargs)

    monkeypatch.setattr(adapter, "free_energy_step_3d", recording_step)
    phi = torch.linspace(-0.2, 0.2, 3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)
    result = run_free_energy_collision_only(
        initialize_free_energy_collision_only_state(phi), FreeEnergyCollisionOnlyConfig(steps=2)
    )

    assert len(calls) == 2
    assert calls[1][0] is not calls[0][0]
    assert calls[1][1] is not calls[0][1]
    assert result.stage == "collision_only"
    assert result.status == "no_streaming_boundary_withheld"
    assert result.physical is False
    assert len(result.diagnostics) == 3
    assert [sample.step for sample in result.diagnostics] == [0, 1, 2]
    assert result.state.f.shape == (19, 3, 4, 5)
    assert result.state.g.shape == (19, 3, 4, 5)


def test_diagnostics_keep_phi_integral_f_mass_and_g_sum_separate():
    phi = torch.zeros((3, 3, 4), dtype=torch.float32)
    result = run_free_energy_collision_only(
        initialize_free_energy_collision_only_state(phi), FreeEnergyCollisionOnlyConfig(steps=2)
    )

    sample = result.diagnostics[0]
    assert sample.phi_integral == pytest.approx(0.0)
    assert sample.f_mass == pytest.approx(float(phi.numel()))
    assert sample.g_sum == pytest.approx(0.0)
    assert sample.phi_integral_name == "phi_integral=sum_x(phi), where phi=sum_i(g_i)"
    assert sample.f_mass_name == "f_mass=sum_i,x(f_i)"
    assert sample.g_sum_name == "g_sum=sum_i,x(g_i)"


@pytest.mark.parametrize(
    ("f", "g", "message"),
    [
        (torch.zeros((18, 3, 4, 5)), torch.zeros((19, 3, 4, 5)), "19"),
        (torch.zeros((19, 3, 4, 5)), torch.zeros((19, 3, 4, 4)), "same shape"),
        (torch.zeros((19, 3, 4, 5), dtype=torch.float32), torch.zeros((19, 3, 4, 5), dtype=torch.float64), "dtype"),
    ],
)
def test_state_rejects_invalid_distribution_shape_device_or_dtype(f, g, message):
    with pytest.raises((TypeError, ValueError), match=message):
        FreeEnergyCollisionOnlyState(f=f, g=g)


def test_initializer_rejects_non_3d_or_nonfloating_phi():
    with pytest.raises(ValueError, match="3-D"):
        initialize_free_energy_collision_only_state(torch.zeros((3, 4)))
    with pytest.raises(TypeError, match="floating"):
        initialize_free_energy_collision_only_state(torch.zeros((3, 4, 5), dtype=torch.int64))


def test_state_accepts_real_d3q19_equilibrium_distributions():
    scalar = torch.ones((3, 4, 5), dtype=torch.float32)
    zero = torch.zeros_like(scalar)
    state = FreeEnergyCollisionOnlyState(
        f=equilibrium3d(scalar, zero, zero, zero),
        g=initialize_free_energy_collision_only_state(zero).g,
    )
    assert state.f.shape == state.g.shape
