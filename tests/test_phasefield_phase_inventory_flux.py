import pytest
import torch

from tensorlbm.phasefield.evolution_adapter import initialize_free_energy_collision_only_state
from tensorlbm.phasefield.evolution_stream_loop import (
    FreeEnergyAdapterStreamLoopConfig,
    run_free_energy_adapter_stream_loop,
)
from tensorlbm.phasefield.phase_inventory_flux import (
    ADAPTER_STREAM_DIAGNOSTIC_ONLY,
    diagnose_adapter_stream_phase_inventory_flux,
)


def _identity_collision(monkeypatch):
    import tensorlbm.phasefield.evolution_stream_loop as loop

    monkeypatch.setattr(loop, "free_energy_step_3d", lambda f, g, **kwargs: (f, g))


def test_periodic_inventory_flux_uses_actual_step_states_and_only_claims_zero_net_stream_transfer(monkeypatch):
    _identity_collision(monkeypatch)
    f = torch.zeros((19, 2, 2, 3), dtype=torch.float64)
    g = torch.zeros_like(f)
    g[1, 1, 1, -1] = 3.5  # +x wraps to x=0 in the adapter stream.
    initial = initialize_free_energy_collision_only_state(torch.zeros((2, 2, 3), dtype=torch.float64))
    initial = type(initial)(f=f, g=g)
    result = run_free_energy_adapter_stream_loop(
        initial, FreeEnergyAdapterStreamLoopConfig(steps=2, boundary="periodic")
    )

    diagnostic = diagnose_adapter_stream_phase_inventory_flux(result)

    assert diagnostic.status == ADAPTER_STREAM_DIAGNOSTIC_ONLY == "diagnostic_only"
    assert diagnostic.physical is False
    assert diagnostic.physical_phase_flux is None
    assert diagnostic.collision_contribution is None
    assert [sample.step for sample in diagnostic.steps] == [0, 1, 2]
    assert [sample.phi_integral for sample in diagnostic.steps] == pytest.approx([3.5, 3.5, 3.5])
    assert [sample.g_sum for sample in diagnostic.steps] == pytest.approx([3.5, 3.5, 3.5])
    assert diagnostic.steps[1].stream_boundary_outgoing_g == pytest.approx(3.5)
    assert diagnostic.steps[1].stream_boundary_incoming_g == pytest.approx(3.5)
    assert diagnostic.steps[1].stream_boundary_net_g == pytest.approx(0.0)
    assert diagnostic.steps[1].stream_boundary_crossing_status == "periodic_transfer_net_zero"
    assert "adapter-stream" in diagnostic.steps[1].stream_boundary_scope
    assert "not a total phase conservation claim" in diagnostic.steps[1].stream_boundary_scope


def test_no_flux_reports_structurally_zero_adapter_crossing_without_inferring_collision_or_physical_flux(monkeypatch):
    _identity_collision(monkeypatch)
    f = torch.zeros((19, 2, 2, 3), dtype=torch.float64)
    g = torch.zeros_like(f)
    g[1, 1, 1, -1] = 7.0
    initial = initialize_free_energy_collision_only_state(torch.zeros((2, 2, 3), dtype=torch.float64))
    initial = type(initial)(f=f, g=g)
    result = run_free_energy_adapter_stream_loop(
        initial, FreeEnergyAdapterStreamLoopConfig(steps=2, boundary="no_flux")
    )

    diagnostic = diagnose_adapter_stream_phase_inventory_flux(result)

    assert all(sample.stream_boundary_outgoing_g == pytest.approx(0.0) for sample in diagnostic.steps[1:])
    assert all(sample.stream_boundary_incoming_g == pytest.approx(0.0) for sample in diagnostic.steps[1:])
    assert all(sample.stream_boundary_net_g == pytest.approx(0.0) for sample in diagnostic.steps[1:])
    assert all(sample.stream_boundary_crossing_status == "no_flux_reflection_zero_crossing" for sample in diagnostic.steps[1:])
    assert diagnostic.collision_contribution is None
    assert diagnostic.physical_phase_flux is None
    assert diagnostic.physical is False


def test_diagnostic_rejects_non_loop_results():
    with pytest.raises(TypeError, match="FreeEnergyAdapterStreamLoopResult"):
        diagnose_adapter_stream_phase_inventory_flux(object())
