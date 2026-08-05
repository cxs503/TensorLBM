import pytest
import torch

from tensorlbm.d3q19 import C, OPPOSITE
from tensorlbm.phasefield.stream_boundary_contract import (
    ADAPTER_STREAM_STAGE,
    PHASE_FLUX_WITHHELD,
    PhaseBoundaryContract,
    stream_d3q19_adapter,
    stream_free_energy_adapter,
)


def test_periodic_adapter_stream_matches_each_authoritative_d3q19_shift():
    field = torch.arange(19 * 3 * 4 * 5, dtype=torch.float64).reshape(19, 3, 4, 5)

    streamed = stream_d3q19_adapter(field, boundary="periodic")

    for q, (cx, cy, cz) in enumerate(C.tolist()):
        expected = torch.roll(field[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
        assert torch.equal(streamed[q], expected)


def test_periodic_adapter_stream_preserves_constant_distributions():
    field = torch.full((19, 3, 4, 5), 2.75, dtype=torch.float32)

    assert torch.equal(stream_d3q19_adapter(field, boundary="periodic"), field)


def test_no_flux_never_wraps_an_outgoing_population_and_reflects_it_locally():
    field = torch.zeros((19, 3, 4, 5), dtype=torch.float64)
    plus_x = 1
    minus_x = int(OPPOSITE[plus_x].item())
    field[plus_x, 1, 2, -1] = 7.0

    streamed = stream_d3q19_adapter(field, boundary="no_flux")

    assert streamed[plus_x, 1, 2, 0].item() == 0.0
    assert streamed[minus_x, 1, 2, -1].item() == 7.0
    assert streamed.sum().item() == pytest.approx(field.sum().item())


def test_boundary_policy_is_mandatory_and_rejects_unknown_values():
    field = torch.zeros((19, 2, 2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="boundary"):
        stream_d3q19_adapter(field, boundary="wetting")
    with pytest.raises(TypeError, match="boundary"):
        stream_d3q19_adapter(field, boundary=None)  # type: ignore[arg-type]


def test_contract_is_explicitly_nonphysical_and_withholds_undefined_phase_flux():
    contract = PhaseBoundaryContract(boundary="no_flux")

    assert contract.stage == ADAPTER_STREAM_STAGE == "collision_then_adapter_stream"
    assert contract.physical is False
    assert contract.phase_flux_status == PHASE_FLUX_WITHHELD == "withheld"
    assert contract.phase_flux is None
    assert contract.wetting is False


def test_coupled_adapter_stream_reports_the_fail_closed_contract():
    f = torch.ones((19, 2, 3, 4), dtype=torch.float32)
    g = torch.full_like(f, 0.5)

    result = stream_free_energy_adapter(f, g, boundary="periodic")

    assert torch.equal(result.f, f)
    assert torch.equal(result.g, g)
    assert result.stage == "collision_then_adapter_stream"
    assert result.physical is False
    assert result.phase_flux_status == "withheld"
