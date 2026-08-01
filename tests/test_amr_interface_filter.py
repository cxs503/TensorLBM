from __future__ import annotations

import pytest
import torch

from tensorlbm.amr_interface_filter import (
    assess_interface_filter_control_volume_clearance,
    damp_interface_nonequilibrium,
    interface_shell_blend,
)
from tensorlbm.amr_population_transfer import (
    regularize_nonequilibrium_second_order,
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
    c = C.to(dtype=f.dtype)
    moments = torch.stack((
        torch.ones(19, dtype=f.dtype),
        c[:, 0], c[:, 1], c[:, 2],
        c[:, 0].square(), c[:, 1].square(), c[:, 2].square(),
        c[:, 0] * c[:, 1], c[:, 0] * c[:, 2], c[:, 1] * c[:, 2],
    ))
    _, _, right = torch.linalg.svd(moments, full_matrices=True)
    kinetic_mode = right[-1] / torch.linalg.vector_norm(right[-1])
    return f + 2.0e-3 * kinetic_mode[:, None, None, None]


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
    before_neq = f - equilibrium
    after_neq = filtered - equilibrium
    before_kinetic = before_neq - regularize_nonequilibrium_second_order(before_neq)
    after_kinetic = after_neq - regularize_nonequilibrium_second_order(after_neq)
    before_norm = torch.linalg.vector_norm(before_kinetic[:, 1, 1, 1])
    after_norm = torch.linalg.vector_norm(after_kinetic[:, 1, 1, 1])
    assert after_norm == pytest.approx(0.6 * before_norm)
    assert torch.equal(filtered[:, 4, 5, 6], f[:, 4, 5, 6])

    c = C.to(dtype=f.dtype)
    for first_axis in range(3):
        for second_axis in range(first_axis, 3):
            before_stress = torch.einsum(
                "qzyx,q,q->zyx",
                before_neq,
                c[:, first_axis],
                c[:, second_axis],
            )
            after_stress = torch.einsum(
                "qzyx,q,q->zyx",
                after_neq,
                c[:, first_axis],
                c[:, second_axis],
            )
            assert torch.allclose(
                after_stress, before_stress, rtol=0.0, atol=3.0e-15,
            )


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


def test_interface_filter_is_publicly_exported() -> None:
    import tensorlbm

    assert tensorlbm.damp_interface_nonequilibrium is damp_interface_nonequilibrium
    assert tensorlbm.interface_shell_blend is interface_shell_blend
    assert (
        tensorlbm.assess_interface_filter_control_volume_clearance
        is assess_interface_filter_control_volume_clearance
    )


def test_control_volume_flux_stencil_requires_one_unfiltered_source_cell() -> None:
    safe = assess_interface_filter_control_volume_clearance(
        (40, 50, 60),
        bounds_xyz=(6, 54, 6, 44, 6, 34),
        ghost=1,
        filter_width=4,
    )
    touching = assess_interface_filter_control_volume_clearance(
        (40, 50, 60),
        bounds_xyz=(5, 55, 5, 45, 5, 35),
        ghost=1,
        filter_width=4,
    )

    assert safe.minimum_physical_interface_clearance_cells == 5
    assert safe.minimum_unfiltered_source_guard_cells == 1
    assert safe.flux_stencil_outside_filter is True
    assert touching.minimum_unfiltered_source_guard_cells == 0
    assert touching.flux_stencil_outside_filter is False


@pytest.mark.parametrize(
    "kwargs",
    (
        {"ghost": 0, "filter_width": 1},
        {"ghost": 1, "filter_width": -1},
        {"ghost": 1, "filter_width": 1, "streaming_stencil_radius": -1},
        {"ghost": 1, "filter_width": 1, "bounds_xyz": (0, 8, 2, 8, 2, 8)},
    ),
)
def test_invalid_control_volume_filter_clearance_fails_closed(kwargs: dict) -> None:
    options = {
        "shape": (10, 10, 10),
        "bounds_xyz": (2, 8, 2, 8, 2, 8),
        "ghost": 1,
        "filter_width": 1,
    }
    options.update(kwargs)
    with pytest.raises(ValueError):
        assess_interface_filter_control_volume_clearance(**options)
