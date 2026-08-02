from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    integrate_bfl_projected_pressure,
    reconstruct_bfl_wall_pressure,
)


def test_quadratic_wall_pressure_extrapolation_is_exact() -> None:
    shape = (3, 3, 7)
    x = torch.arange(shape[2], dtype=torch.float64).view(1, 1, -1)
    pressure = 1.0e-3 * x.square().expand(shape)
    rho = 1.0 + 3.0 * pressure
    zero = torch.zeros_like(rho)
    populations = equilibrium3d(rho, zero, zero, zero)
    near = torch.zeros(shape, dtype=torch.bool)
    near[1, 1, 2] = True
    nx = torch.zeros(shape, dtype=torch.float64)
    nx[near] = 1.0
    mesh = SurfaceMesh(
        near,
        nx,
        torch.zeros_like(nx),
        torch.zeros_like(nx),
    )
    solid = torch.zeros(shape, dtype=torch.bool)

    force_none = drag_pressure_integration(
        populations, mesh, 1.0, extrap="none",
        p0_method="inlet", solid=solid, p0_inlet_width=1,
    )[0]
    force_quadratic = drag_pressure_integration(
        populations, mesh, 1.0, extrap="quadratic",
        p0_method="inlet", solid=solid, p0_inlet_width=1,
    )[0]

    assert force_none == pytest.approx(-4.0e-3, abs=2.0e-8)
    assert force_quadratic == pytest.approx(-1.0e-3, abs=2.0e-8)


@pytest.mark.parametrize("q", (0.15, 0.5, 0.95))
def test_bfl_quadratic_wall_pressure_is_exact_at_actual_q(q: float) -> None:
    shape = (5, 5, 9)
    z, y, x = 2, 2, 3
    x_wall = x - q
    xx = torch.arange(shape[2], dtype=torch.float64).view(1, 1, -1)
    pressure = (
        0.013 + 0.007 * (xx - x_wall) + 0.002 * (xx - x_wall).square()
    ).expand(shape).clone()
    near = torch.zeros(shape, dtype=torch.bool)
    near[z, y, x] = True
    nx = torch.zeros(shape, dtype=torch.float64)
    nx[near] = 1.0
    zero = torch.zeros_like(nx)
    mesh = SurfaceMesh(near, nx, zero, zero)
    boundary = torch.zeros((19, *shape), dtype=torch.bool)
    q_field = torch.full((19, *shape), 0.5, dtype=torch.float64)
    from tensorlbm.d3q19 import C

    direction = int(((C[:, 0] == -1) & (C[:, 1] == 0) & (C[:, 2] == 0)).nonzero()[0])
    boundary[direction, z, y, x] = True
    q_field[direction, z, y, x] = q
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[z, y, x - 1] = True

    wall, diagnostics = reconstruct_bfl_wall_pressure(
        pressure, mesh, boundary, q_field, solid=solid,
    )

    assert wall[z, y, x].item() == pytest.approx(0.013, abs=2.0e-14)
    assert diagnostics.boundary_cells == 1
    assert diagnostics.requested_links == 1
    assert diagnostics.usable_links == 1
    assert diagnostics.fallback_cells == 0
    assert diagnostics.minimum_active_q == pytest.approx(q)
    assert diagnostics.maximum_active_q == pytest.approx(q)


def test_bfl_pressure_reconstruction_never_wraps_at_domain_edge() -> None:
    shape = (3, 3, 5)
    pressure = torch.arange(5, dtype=torch.float64).view(1, 1, 5).expand(shape)
    near = torch.zeros(shape, dtype=torch.bool)
    near[1, 1, 0] = True
    nx = torch.zeros(shape, dtype=torch.float64)
    nx[near] = -1.0
    zero = torch.zeros_like(nx)
    mesh = SurfaceMesh(near, nx, zero, zero)
    boundary = torch.zeros((19, *shape), dtype=torch.bool)
    q_field = torch.full((19, *shape), 0.5, dtype=torch.float64)
    from tensorlbm.d3q19 import C

    direction = int(((C[:, 0] == 1) & (C[:, 1] == 0) & (C[:, 2] == 0)).nonzero()[0])
    boundary[direction, 1, 1, 0] = True
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[1, 1, 1] = True

    wall, diagnostics = reconstruct_bfl_wall_pressure(
        pressure, mesh, boundary, q_field, solid=solid,
    )

    assert wall[1, 1, 0].item() == pytest.approx(0.0)
    assert diagnostics.usable_links == 0
    assert diagnostics.fallback_cells == 1


def test_bfl_pressure_integration_requires_link_geometry() -> None:
    shape = (3, 3, 5)
    rho = torch.ones(shape, dtype=torch.float64)
    zero = torch.zeros_like(rho)
    populations = equilibrium3d(rho, zero, zero, zero)
    near = torch.zeros(shape, dtype=torch.bool)
    near[1, 1, 2] = True
    mesh = SurfaceMesh(near, zero, zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)

    with pytest.raises(ValueError, match="fluid_boundary_mask and q_field"):
        drag_pressure_integration(
            populations, mesh, 1.0, extrap="bfl_quadratic",
            p0_method="inlet", solid=solid,
        )


def test_projected_bfl_pressure_closes_constant_and_linear_cube_fields() -> None:
    from tensorlbm.d3q19 import C

    shape = (7, 7, 7)
    center = (3, 3, 3)
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[center] = True
    boundary = torch.zeros((19, *shape), dtype=torch.bool)
    q_field = torch.full((19, *shape), 0.5, dtype=torch.float64)
    for direction in range(1, 19):
        cx, cy, cz = (int(value) for value in C[direction].tolist())
        if abs(cx) + abs(cy) + abs(cz) != 1:
            continue
        fluid = (center[0] - cz, center[1] - cy, center[2] - cx)
        boundary[(direction, *fluid)] = True
    constant = torch.full(shape, 2.75, dtype=torch.float64)
    force_constant, diagnostics = integrate_bfl_projected_pressure(
        constant, boundary, q_field, solid=solid,
    )
    assert force_constant == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-14)
    assert diagnostics.requested_links == 6
    assert diagnostics.usable_links == 6
    assert diagnostics.fallback_cells == 0

    x = torch.arange(shape[2], dtype=torch.float64).view(1, 1, -1)
    linear = (1.0 + 0.2 * x).expand(shape)
    force_linear, _ = integrate_bfl_projected_pressure(
        linear, boundary, q_field, solid=solid,
    )
    assert force_linear == pytest.approx((-0.2, 0.0, 0.0), abs=1.0e-14)

    force_linear_first_order, _ = integrate_bfl_projected_pressure(
        linear,
        boundary,
        q_field,
        solid=solid,
        reconstruction="linear",
    )
    assert force_linear_first_order == pytest.approx(
        (-0.2, 0.0, 0.0), abs=1.0e-14,
    )
    force_local, _ = integrate_bfl_projected_pressure(
        linear,
        boundary,
        q_field,
        solid=solid,
        reconstruction="local",
    )
    assert force_local == pytest.approx((-0.4, 0.0, 0.0), abs=1.0e-14)


def test_projected_bfl_pressure_rejects_unknown_reconstruction() -> None:
    shape = (3, 3, 3)
    pressure = torch.zeros(shape, dtype=torch.float64)
    boundary = torch.zeros((19, *shape), dtype=torch.bool)
    q_field = torch.full((19, *shape), 0.5, dtype=torch.float64)
    solid = torch.zeros(shape, dtype=torch.bool)
    with pytest.raises(ValueError, match="reconstruction"):
        integrate_bfl_projected_pressure(
            pressure,
            boundary,
            q_field,
            solid=solid,
            reconstruction="cubic",
        )


def test_bfl_pressure_reconstruction_is_public() -> None:
    import tensorlbm

    assert tensorlbm.reconstruct_bfl_wall_pressure is reconstruct_bfl_wall_pressure
    assert tensorlbm.integrate_bfl_projected_pressure is integrate_bfl_projected_pressure
