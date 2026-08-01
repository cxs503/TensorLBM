from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.drag_pressure import SurfaceMesh, drag_pressure_integration


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
