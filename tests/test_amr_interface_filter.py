from __future__ import annotations

import pytest
import torch

from tensorlbm.amr_interface_filter import (
    damp_interface_nonequilibrium,
    interface_shell_blend,
)
from tensorlbm.d3q19 import C, equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27


def _perturbed_population() -> torch.Tensor:
    shape = (10, 12, 14)
    rho = torch.full(shape, 1.01, dtype=torch.float64)
    ux = torch.full(shape, 0.04, dtype=torch.float64)
    uy = torch.full(shape, -0.01, dtype=torch.float64)
    uz = torch.full(shape, 0.005, dtype=torch.float64)
    f = equilibrium3d(rho, ux, uy, uz)
    perturbation = torch.zeros_like(f)
    perturbation[7] = 2.0e-3
    perturbation[8] = 2.0e-3
    perturbation[9] = -2.0e-3
    perturbation[10] = -2.0e-3
    return f + perturbation


def test_interface_shell_excludes_ghosts_and_retains_core() -> None:
    blend = interface_shell_blend(
        (10, 12, 14), ghost=1, width=2, strength=0.4,
        device=torch.device("cpu"), dtype=torch.float64,
    )

    assert float(blend[0].max()) == 0.0
    assert float(blend[:, 0].max()) == 0.0
    assert float(blend[:, :, 0].max()) == 0.0
    assert float(blend[1, 1, 1]) == pytest.approx(0.4)
    assert float(blend[2, 2, 2]) == pytest.approx(0.2)
    assert float(blend[4, 5, 6]) == 0.0


def test_filter_preserves_density_and_momentum_and_reduces_nonequilibrium() -> None:
    f = _perturbed_population()
    blend = interface_shell_blend(
        f.shape[1:], ghost=1, width=2, strength=0.4,
        device=f.device, dtype=f.dtype,
    )
    rho_before, ux_before, uy_before, uz_before = macroscopic3d(f)
    filtered = damp_interface_nonequilibrium(f, blend)
    rho_after, ux_after, uy_after, uz_after = macroscopic3d(filtered)

    assert torch.allclose(rho_after, rho_before, rtol=0.0, atol=2.0e-15)
    assert torch.allclose(ux_after, ux_before, rtol=0.0, atol=2.0e-15)
    assert torch.allclose(uy_after, uy_before, rtol=0.0, atol=2.0e-15)
    assert torch.allclose(uz_after, uz_before, rtol=0.0, atol=2.0e-15)
    equilibrium = equilibrium3d(rho_before, ux_before, uy_before, uz_before)
    before_norm = torch.linalg.vector_norm((f - equilibrium)[:, 1, 1, 1])
    after_norm = torch.linalg.vector_norm((filtered - equilibrium)[:, 1, 1, 1])
    assert after_norm == pytest.approx(0.6 * before_norm)
    assert torch.equal(filtered[:, 4, 5, 6], f[:, 4, 5, 6])


def test_uniform_equilibrium_is_unchanged() -> None:
    rho = torch.ones((10, 12, 14), dtype=torch.float64)
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, zero, zero, zero)
    blend = interface_shell_blend(
        f.shape[1:], ghost=1, width=3, strength=1.0,
        device=f.device, dtype=f.dtype,
    )

    assert torch.allclose(
        damp_interface_nonequilibrium(f, blend), f, rtol=0.0, atol=2.0e-8,
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"ghost": 0, "width": 1, "strength": 0.2},
        {"ghost": 1, "width": -1, "strength": 0.2},
        {"ghost": 1, "width": 1, "strength": 1.1},
        {"ghost": 1, "width": 5, "strength": 0.2},
    ),
)
def test_invalid_filter_configuration_fails_closed(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        interface_shell_blend(
            (10, 12, 14), device=torch.device("cpu"), dtype=torch.float32,
            **kwargs,
        )


def test_filter_rejects_wrong_blend_shape() -> None:
    f = _perturbed_population()
    with pytest.raises(ValueError, match="spatial shape"):
        damp_interface_nonequilibrium(f, torch.zeros((2, 2, 2), dtype=f.dtype))


def test_population_momentum_is_directly_unchanged() -> None:
    f = _perturbed_population()
    blend = interface_shell_blend(
        f.shape[1:], ghost=1, width=2, strength=0.3,
        device=f.device, dtype=f.dtype,
    )
    filtered = damp_interface_nonequilibrium(f, blend)
    c = C.to(dtype=f.dtype)
    before = torch.einsum("qzyx,qd->dzyx", f, c)
    after = torch.einsum("qzyx,qd->dzyx", filtered, c)
    assert torch.allclose(after, before, rtol=0.0, atol=3.0e-15)


def test_d3q27_filter_preserves_macroscopic_state() -> None:
    torch.manual_seed(19)
    shape = (10, 12, 14)
    rho = torch.full(shape, 1.02, dtype=torch.float64)
    ux = torch.full(shape, 0.03, dtype=torch.float64)
    uy = torch.full(shape, -0.004, dtype=torch.float64)
    uz = torch.full(shape, 0.002, dtype=torch.float64)
    f = equilibrium27(rho, ux, uy, uz)
    f += 1.0e-5 * torch.randn_like(f)
    blend = interface_shell_blend(
        shape, ghost=1, width=2, strength=0.25,
        device=f.device, dtype=f.dtype,
    )
    before = macroscopic27(f)
    after = macroscopic27(damp_interface_nonequilibrium(f, blend))

    for actual, expected in zip(after, before, strict=True):
        assert torch.allclose(actual, expected, rtol=0.0, atol=3.0e-15)
