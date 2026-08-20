from __future__ import annotations

import pytest
import torch

from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.solver3d import stream3d
from tensorlbm.sponge_layer import (
    apply_equilibrium_difference_sponge,
    build_anisotropic_sponge_sigma_3d,
    build_sponge_sigma_3d,
    smoothstep5,
)


def test_fifth_order_ramp_has_exact_endpoints() -> None:
    x = torch.tensor([0.0, 0.5, 1.0])
    y = smoothstep5(x)
    assert torch.equal(y[[0, 2]], torch.tensor([0.0, 1.0]))
    assert y[1].item() == 0.5


def test_sponge_is_zero_interior_and_strongest_at_selected_face() -> None:
    sigma = build_sponge_sigma_3d(
        (9, 11, 13),
        width=3,
        max_strength=0.2,
        faces=("x+",),
    )
    assert sigma[4, 5, 6].item() == 0.0
    assert torch.isclose(sigma[4, 5, -1], torch.tensor(0.2))
    assert sigma[4, 5, -2].item() < sigma[4, 5, -1].item()


def test_anisotropic_sponge_uses_independent_face_widths() -> None:
    sigma = build_anisotropic_sponge_sigma_3d(
        (15, 17, 41),
        face_widths={"x+": 12, "y-": 3},
        max_strength=0.2,
    )
    assert sigma[7, 8, 30].item() > 0.0
    assert sigma[7, 4, 20].item() == 0.0
    assert sigma[7, 0, 20].item() == pytest.approx(0.2)
    assert sigma[7, 8, 0].item() == 0.0


def test_target_equilibrium_is_fixed_point() -> None:
    shape = (5, 7, 9)
    rho = torch.ones(shape)
    ux = torch.full(shape, 0.04)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, ux, zero, zero)
    sigma = torch.rand(shape) * 0.2
    out = apply_equilibrium_difference_sponge(
        f,
        sigma,
        velocity_target=(0.04, 0.0, 0.0),
    )
    assert torch.allclose(out, f, atol=1e-7, rtol=0.0)


def test_sponge_reduces_macroscopic_velocity_perturbation() -> None:
    shape = (5, 7, 9)
    rho = torch.ones(shape)
    ux = torch.full(shape, 0.08)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, ux, zero, zero)
    sigma = torch.full(shape, 0.25)
    out = apply_equilibrium_difference_sponge(
        f,
        sigma,
        velocity_target=(0.04, 0.0, 0.0),
    )
    _, ux_out, _, _ = macroscopic3d(out)
    assert torch.allclose(ux_out, torch.full_like(ux_out, 0.07), atol=2e-7)


def test_equilibrium_difference_layer_absorbs_periodic_acoustic_return() -> None:
    """A pulse crossing x+ must not return through torch.roll at full energy."""
    from tensorlbm.solver3d import collide_bgk3d

    shape = (3, 3, 160)
    x = torch.arange(shape[2], dtype=torch.float32)
    pulse = 1e-3 * torch.exp(-((x - 40.0) / 6.0).square())
    rho = 1.0 + pulse.view(1, 1, -1).expand(shape)
    zero = torch.zeros(shape)
    initial = equilibrium3d(rho, zero, zero, zero)
    sigma = build_sponge_sigma_3d(
        shape,
        width=30,
        max_strength=0.2,
        faces=("x+",),
    )

    def returned_energy(use_sponge: bool) -> float:
        f = initial.clone()
        for _ in range(400):
            f = stream3d(collide_bgk3d(f, tau=0.56))
            if use_sponge:
                f = apply_equilibrium_difference_sponge(f, sigma)
        density, _, _, _ = macroscopic3d(f)
        return float((density[:, :, 10:90] - 1.0).square().sum())

    undamped = returned_energy(False)
    damped = returned_energy(True)
    assert damped < 0.2 * undamped


def test_production_cumulant_open_boundary_absorbs_acoustic_return() -> None:
    """The production collision/NEE chain must attenuate a returning pulse."""
    shape = (3, 3, 180)
    lattice_speed = 0.06
    x = torch.arange(shape[2], dtype=torch.float32)
    pulse = 1e-3 * torch.exp(-((x - 45.0) / 6.0).square())
    rho = (1.0 + pulse).view(1, 1, -1).expand(shape)
    ux = torch.full(shape, lattice_speed)
    zero = torch.zeros(shape)
    initial = equilibrium3d(rho, ux, zero, zero)
    sigma = build_sponge_sigma_3d(
        shape,
        width=30,
        max_strength=0.3,
        faces=("x+",),
    )

    def returned_energy(use_sponge: bool) -> float:
        f = initial.clone()
        maximum = 0.0
        for step in range(1, 701):
            f = stream3d(
                collide_cumulant_d3q19(
                    f,
                    tau=0.500324,
                    C_s=0.05,
                )
            )
            f = non_equilibrium_far_field_bc_3d(
                f,
                u_in=lattice_speed,
                faces=("x-", "x+"),
            )
            if use_sponge:
                f = apply_equilibrium_difference_sponge(
                    f,
                    sigma,
                    velocity_target=(lattice_speed, 0.0, 0.0),
                )
            f = non_equilibrium_far_field_bc_3d(
                f,
                u_in=lattice_speed,
                faces=("x-", "x+"),
            )
            if step >= 450:
                density, _, _, _ = macroscopic3d(f)
                energy = float((density[:, :, 10:90] - 1.0).square().mean())
                maximum = max(maximum, energy)
        return maximum

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        undamped = returned_energy(False)
        damped = returned_energy(True)
    finally:
        torch.set_num_threads(previous_threads)
    assert damped < 0.01 * undamped


def test_production_inlet_sponge_absorbs_upstream_acoustic_return() -> None:
    """Adding x- damping must attenuate a pulse returning from the inlet."""
    shape = (3, 3, 180)
    lattice_speed = 0.06
    x = torch.arange(shape[2], dtype=torch.float32)
    pulse = 1e-3 * torch.exp(-((x - 135.0) / 6.0).square())
    rho = (1.0 + pulse).view(1, 1, -1).expand(shape)
    ux = torch.full(shape, lattice_speed)
    zero = torch.zeros(shape)
    initial = equilibrium3d(rho, ux, zero, zero)
    outlet_only = build_sponge_sigma_3d(
        shape,
        width=30,
        max_strength=0.3,
        faces=("x+",),
    )
    both_streamwise_faces = build_sponge_sigma_3d(
        shape,
        width=30,
        max_strength=0.3,
        faces=("x-", "x+"),
    )

    def returned_energy(include_inlet: bool) -> float:
        f = initial.clone()
        maximum = 0.0
        sigma = both_streamwise_faces if include_inlet else outlet_only
        for step in range(1, 701):
            f = stream3d(
                collide_cumulant_d3q19(
                    f,
                    tau=0.500324,
                    C_s=0.05,
                )
            )
            f = non_equilibrium_far_field_bc_3d(
                f,
                u_in=lattice_speed,
                faces=("x-", "x+"),
            )
            f = apply_equilibrium_difference_sponge(
                f,
                sigma,
                velocity_target=(lattice_speed, 0.0, 0.0),
            )
            f = non_equilibrium_far_field_bc_3d(
                f,
                u_in=lattice_speed,
                faces=("x-", "x+"),
            )
            if step >= 450:
                density, _, _, _ = macroscopic3d(f)
                energy = float((density[:, :, 90:170] - 1.0).square().mean())
                maximum = max(maximum, energy)
        return maximum

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        undamped_inlet = returned_energy(False)
        damped_inlet = returned_energy(True)
    finally:
        torch.set_num_threads(previous_threads)
    assert damped_inlet < 0.01 * undamped_inlet
