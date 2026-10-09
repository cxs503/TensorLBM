"""Equivalence and contract tests for bounded-memory collision tiling."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import tensorlbm
from tensorlbm.chunked_collision import (
    NaturalKBCCollisionExecutor,
    collide_in_z_chunks,
)
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.entropic_kbc import collide_natural_kbc_d3q19


def _state() -> torch.Tensor:
    torch.manual_seed(20260802)
    shape = (7, 8, 9)
    rho = torch.ones(shape)
    ux = torch.full(shape, 0.035)
    zero = torch.zeros(shape)
    state = equilibrium3d(rho, ux, zero, zero)
    return state + 1.0e-5 * torch.randn_like(state)


def test_chunked_natural_kbc_matches_full_domain_collision_to_roundoff() -> None:
    state = _state()
    expected = collide_natural_kbc_d3q19(state, tau=0.73)

    actual = collide_in_z_chunks(
        state,
        lambda slab: collide_natural_kbc_d3q19(slab, tau=0.73),
        chunk_cells=2 * 8 * 9,
    )

    torch.testing.assert_close(actual, expected, rtol=5.0e-7, atol=1.0e-7)


def test_one_large_chunk_delegates_without_changing_values() -> None:
    state = _state()
    actual = collide_in_z_chunks(
        state,
        lambda slab: slab * 0.999,
        chunk_cells=state[0].numel(),
    )
    torch.testing.assert_close(actual, state * 0.999)


@pytest.mark.parametrize("chunk_cells", (0, -1, True))
def test_invalid_chunk_size_is_rejected(chunk_cells: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        collide_in_z_chunks(
            _state(),
            lambda slab: slab,
            chunk_cells=chunk_cells,
        )


def test_collision_cannot_change_the_slab_contract() -> None:
    state = _state()
    with pytest.raises(ValueError, match="preserve slab"):
        collide_in_z_chunks(
            state,
            lambda slab: slab[:, :-1],
            chunk_cells=2 * 8 * 9,
        )


def test_natural_kbc_executor_eager_path_is_exact() -> None:
    state = _state()
    expected = collide_natural_kbc_d3q19(state, 0.73)

    actual = NaturalKBCCollisionExecutor()(state, 0.73)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_natural_kbc_executor_can_compute_in_float64_with_float32_storage() -> None:
    state = _state()
    expected = collide_natural_kbc_d3q19(state.double(), 0.5000162).float()
    executor = NaturalKBCCollisionExecutor(compute_dtype="float64")

    actual = executor(state, 0.5000162)

    assert actual.dtype == state.dtype
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert executor.diagnostics()["compute_dtype"] == "float64"
    assert executor.diagnostics()["d3q19_weight_precision_scheme"] == (
        "rational_binary64_cast_to_runtime_dtype_v1"
    )


def test_natural_kbc_executor_rejects_unknown_compute_dtype() -> None:
    with pytest.raises(ValueError, match="compute_dtype"):
        NaturalKBCCollisionExecutor(compute_dtype="float16")


def test_bounded_collision_executor_is_part_of_public_api() -> None:
    assert tensorlbm.NaturalKBCCollisionExecutor is NaturalKBCCollisionExecutor
    assert tensorlbm.collide_in_z_chunks is collide_in_z_chunks


def test_compiled_executor_passes_tensor_tau_and_reuses_callable() -> None:
    state = _state()
    calls: list[torch.Tensor] = []

    def compiled(populations: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        calls.append(tau)
        return populations

    with patch("torch.compile", return_value=compiled) as compiler:
        executor = NaturalKBCCollisionExecutor(compile_enabled=True)
        assert executor(state, 0.73) is state
        assert executor(state, 0.71) is state

    compiler.assert_called_once()
    assert [float(value) for value in calls] == pytest.approx([0.73, 0.71])
    assert all(value.ndim == 0 and value.dtype == state.dtype for value in calls)
    diagnostics = executor.diagnostics()
    assert diagnostics["compile_enabled"] is True
    assert diagnostics["compute_dtype"] == "storage"
    assert diagnostics["collision_calls"] == 2
    assert diagnostics["minimum_tau"] == pytest.approx(0.71)
    assert diagnostics["maximum_tau"] == pytest.approx(0.73)
    assert diagnostics["input_signatures"] == [
        {
            "device": "cpu",
            "dtype": "torch.float32",
            "shape_qzyx": list(state.shape),
        }
    ]


@pytest.mark.parametrize("tau", (0.5, float("nan"), float("inf")))
def test_natural_kbc_executor_rejects_invalid_tau(tau: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        NaturalKBCCollisionExecutor(compile_enabled=True)(_state(), tau)
