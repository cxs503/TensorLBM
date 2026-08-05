import pytest
import torch

from tensorlbm.phasefield.evolution_adapter import initialize_free_energy_collision_only_state
from tensorlbm.phasefield.evolution_stream_loop import (
    ADAPTER_STREAM_LOOP_STAGE,
    FreeEnergyAdapterStreamLoopConfig,
    collision_then_adapter_stream,
    run_free_energy_adapter_stream_loop,
)
from tensorlbm.phasefield.stream_boundary_contract import stream_free_energy_adapter


def test_one_step_uses_real_lazy_collision_then_applies_adapter_stream(monkeypatch):
    import tensorlbm.phasefield.evolution_stream_loop as loop

    real_collision = loop.free_energy_step_3d
    real_stream = loop.stream_free_energy_adapter
    collisions: list[tuple[torch.Tensor, torch.Tensor]] = []
    post_collisions: list[tuple[torch.Tensor, torch.Tensor]] = []
    streams: list[tuple[torch.Tensor, torch.Tensor]] = []

    def recording_collision(f: torch.Tensor, g: torch.Tensor, **kwargs: object):
        collisions.append((f, g))
        post = real_collision(f, g, **kwargs)
        post_collisions.append(post)
        return post

    def recording_stream(f: torch.Tensor, g: torch.Tensor, *, boundary: str):
        streams.append((f, g))
        return real_stream(f, g, boundary=boundary)  # type: ignore[arg-type]

    monkeypatch.setattr(loop, "free_energy_step_3d", recording_collision)
    monkeypatch.setattr(loop, "stream_free_energy_adapter", recording_stream)
    phi = torch.linspace(-0.2, 0.2, 3 * 4 * 5).reshape(3, 4, 5)
    initial = initialize_free_energy_collision_only_state(phi)

    result = collision_then_adapter_stream(
        initial, FreeEnergyAdapterStreamLoopConfig(steps=2, boundary="periodic")
    )

    assert len(collisions) == len(streams) == 2
    assert streams[0][0] is post_collisions[0][0]
    assert streams[0][1] is post_collisions[0][1]
    assert collisions[1][0] is result.step_states[0].f
    assert collisions[1][1] is result.step_states[0].g
    expected = real_stream(*post_collisions[0], boundary="periodic")
    assert torch.equal(result.step_states[0].f, expected.f)
    assert torch.equal(result.step_states[0].g, expected.g)
    assert result.stage == ADAPTER_STREAM_LOOP_STAGE == "collision_then_adapter_stream"


def test_two_steps_report_distinct_observables_inventory_and_withheld_physics():
    phi = torch.zeros((3, 3, 4), dtype=torch.float32)

    result = run_free_energy_adapter_stream_loop(
        initialize_free_energy_collision_only_state(phi),
        FreeEnergyAdapterStreamLoopConfig(steps=2, boundary="no_flux"),
    )

    assert [sample.step for sample in result.diagnostics] == [0, 1, 2]
    initial = result.diagnostics[0]
    assert initial.phi_integral == pytest.approx(0.0)
    assert initial.f_mass == pytest.approx(float(phi.numel()))
    assert initial.g_sum == pytest.approx(0.0)
    assert initial.distribution_inventory == pytest.approx(float(phi.numel()))
    assert initial.phi_integral_name != initial.f_mass_name != initial.g_sum_name
    assert initial.distribution_inventory_name == "distribution_inventory=sum_i,x(f_i)+sum_i,x(g_i)"
    assert result.state.f.shape == result.state.g.shape == (19, 3, 3, 4)
    assert result.boundary == "no_flux"
    assert result.physical is False
    assert result.phase_flux is None
    assert result.phase_flux_status == "withheld"


def test_no_flux_loop_does_not_wrap_the_adapter_stream_sentinel(monkeypatch):
    import tensorlbm.phasefield.evolution_stream_loop as loop

    def sentinel_collision(f: torch.Tensor, g: torch.Tensor, **kwargs: object):
        return f, g

    monkeypatch.setattr(loop, "free_energy_step_3d", sentinel_collision)
    f = torch.zeros((19, 2, 2, 3), dtype=torch.float64)
    g = torch.zeros_like(f)
    plus_x = 1
    minus_x = 2
    f[plus_x, 1, 1, -1] = 9.0
    state = initialize_free_energy_collision_only_state(torch.zeros((2, 2, 3), dtype=torch.float64))
    state = type(state)(f=f, g=g)

    result = run_free_energy_adapter_stream_loop(
        state, FreeEnergyAdapterStreamLoopConfig(steps=2, boundary="no_flux")
    )

    assert result.step_states[0].f[plus_x, 1, 1, 0].item() == 0.0
    assert result.step_states[0].f[minus_x, 1, 1, -1].item() == 9.0


@pytest.mark.parametrize("boundary", ["wetting", "open"])
def test_loop_rejects_non_adapter_boundary_policies(boundary):
    with pytest.raises(ValueError, match="boundary"):
        FreeEnergyAdapterStreamLoopConfig(boundary=boundary)
