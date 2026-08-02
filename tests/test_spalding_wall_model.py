from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.spalding_wall_model import (
    apply_spalding_exchange_wall_model,
    assess_wall_exchange_interface_clearance,
    effective_bfl_wall_distance,
    sample_wall_exchange_velocity,
    solve_spalding_friction_velocity,
    spalding_u_plus_from_y_plus,
    spalding_y_plus,
)


def _flat_boundary(shape=(5, 7, 9)):
    mask = torch.zeros((19,) + shape, dtype=torch.bool)
    q = torch.full((19,) + shape, 0.5)
    cell = (2, 3, 4)
    for direction in (4, 8, 9, 16, 18):  # c_y=-1, solid below fluid node
        mask[(direction,) + cell] = True
    nx = torch.zeros(shape)
    ny = torch.zeros(shape); ny[cell] = 1.0  # outward into fluid
    nz = torch.zeros(shape)
    return mask, q, (nx, ny, nz), cell


def test_spalding_inverse_round_trip() -> None:
    u_plus = torch.tensor([0.0, 1.0, 5.0, 12.0, 22.0], dtype=torch.float64)
    recovered = spalding_u_plus_from_y_plus(spalding_y_plus(u_plus))
    assert torch.allclose(recovered, u_plus, atol=2e-10, rtol=0.0)


def test_exchange_interface_clearance_accounts_for_trilinear_support() -> None:
    assessment = assess_wall_exchange_interface_clearance(
        exchange_distance_cells=8.4375,
        available_buffer_cells=14,
    )

    assert assessment.required_buffer_cells == 10
    assert assessment.remaining_clearance_cells == 4
    assert assessment.admitted is True


def test_exchange_interface_clearance_fails_closed() -> None:
    assessment = assess_wall_exchange_interface_clearance(
        exchange_distance_cells=8.4375,
        available_buffer_cells=9,
    )

    assert assessment.remaining_clearance_cells == -1
    assert assessment.admitted is False


def test_friction_velocity_recovers_manufactured_spalding_state() -> None:
    nu = 1.5e-4
    y2 = torch.tensor([2.5, 3.0, 4.0], dtype=torch.float64)
    expected = torch.tensor([0.003, 0.004, 0.005], dtype=torch.float64)
    u_plus = spalding_u_plus_from_y_plus(y2 * expected / nu)
    speed = expected * u_plus
    actual = solve_spalding_friction_velocity(speed, y2, nu)
    assert torch.allclose(actual, expected, rtol=2e-8, atol=1e-11)


def test_flat_halfway_links_recover_half_cell_normal_distance() -> None:
    mask, q, normals, cell = _flat_boundary()
    distance = effective_bfl_wall_distance(mask, q, normals)
    assert distance[cell].item() == pytest.approx(0.5)


def test_exchange_velocity_samples_linear_field_at_requested_wall_distance() -> None:
    mask, q, normals, cell = _flat_boundary()
    shape = q.shape[1:]
    z, y, x = torch.meshgrid(
        torch.arange(shape[0], dtype=torch.float64),
        torch.arange(shape[1], dtype=torch.float64),
        torch.arange(shape[2], dtype=torch.float64), indexing="ij",
    )
    samples = sample_wall_exchange_velocity(
        (2.0 * y + x, torch.zeros_like(x), torch.zeros_like(x)),
        mask, q.double(), tuple(component.double() for component in normals),
        exchange_distance=2.0,
    )
    assert int(samples.boundary.sum().item()) == 1
    assert samples.y1.item() == pytest.approx(0.5)
    assert samples.y2.item() == pytest.approx(2.0)
    # Boundary-node centre is y=3 and y2-y1=1.5 cells outward into fluid.
    assert samples.velocity_x.item() == pytest.approx(2.0 * 4.5 + cell[2])


def test_exchange_velocity_rejects_nonpositive_distance() -> None:
    mask, q, normals, _ = _flat_boundary()
    zero = torch.zeros(q.shape[1:])
    with pytest.raises(ValueError, match="exchange_distance"):
        sample_wall_exchange_velocity(
            (zero, zero, zero), mask, q, normals, exchange_distance=0.0,
        )


def test_exchange_velocity_excludes_points_outside_domain() -> None:
    mask, q, normals, _ = _flat_boundary()
    zero = torch.zeros(q.shape[1:])
    samples = sample_wall_exchange_velocity(
        (zero, zero, zero), mask, q, normals, exchange_distance=20.0,
    )
    assert not bool(samples.boundary.any())
    assert samples.velocity_x.numel() == 0


def test_exchange_velocity_excludes_solid_contaminated_stencil() -> None:
    mask, q, normals, cell = _flat_boundary()
    zero = torch.zeros(q.shape[1:])
    fluid = torch.ones(q.shape[1:], dtype=torch.bool)
    # y2=2 from a y1=.5 boundary node at y=3 samples y=4.5.
    fluid[cell[0], 5, cell[2]] = False
    samples = sample_wall_exchange_velocity(
        (zero, zero, zero), mask, q, normals,
        exchange_distance=2.0, fluid_mask=fluid,
    )
    assert not bool(samples.boundary.any())
    assert samples.velocity_x.numel() == 0


def test_zero_flow_is_unchanged_and_has_zero_shear() -> None:
    mask, q, normals, _ = _flat_boundary()
    shape = q.shape[1:]
    rho = torch.ones(shape)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, zero, zero, zero)
    out, diagnostics = apply_spalding_exchange_wall_model(
        f, mask, q, normals, nu=0.01, exchange_distance=2.0,
    )
    assert torch.allclose(out, f, atol=1e-7, rtol=0.0)
    assert diagnostics.boundary_nodes == 1
    assert diagnostics.shear_force == pytest.approx((0.0, 0.0, 0.0), abs=1e-14)


def test_assimilation_preserves_boundary_density() -> None:
    mask, q, normals, cell = _flat_boundary()
    shape = q.shape[1:]
    rho = torch.ones(shape)
    ux = torch.full(shape, 0.06)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, ux, zero, zero)
    out, diagnostics = apply_spalding_exchange_wall_model(
        f, mask, q, normals, nu=1e-4, exchange_distance=2.0,
    )
    rho_out, ux_out, _, _ = macroscopic3d(out)
    assert rho_out[cell].item() == pytest.approx(1.0, abs=2e-7)
    assert 0.0 < ux_out[cell].item() < 0.06
    assert diagnostics.mean_u_tau > 0.0
