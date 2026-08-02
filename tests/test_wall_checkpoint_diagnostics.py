from __future__ import annotations

import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.wall_checkpoint_diagnostics import (
    diagnose_bfl_wall_exchange_state,
)


def _flat_wall_state() -> tuple[torch.Tensor, ...]:
    shape = (5, 5, 7)
    rho = torch.ones(shape)
    rho[..., 1:] += torch.arange(1, 7).view(1, 1, 6) * 1.0e-4
    ux = torch.full(shape, 0.06)
    zero = torch.zeros(shape)
    populations = equilibrium3d(rho, ux, zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[:, 0, :] = True
    links = torch.zeros_like(populations, dtype=torch.bool)
    links[3, :, 1, :] = True
    q = torch.zeros_like(populations)
    q[links] = 0.5
    near = torch.zeros_like(solid)
    near[:, 1, :] = True
    nx = torch.zeros(shape)
    ny = torch.zeros(shape)
    nz = torch.zeros(shape)
    ny[near] = 1.0
    return populations, solid, links, q, near, nx, ny, nz


def test_checkpoint_wall_diagnostic_is_read_only_and_reports_gradient() -> None:
    populations, solid, links, q, near, nx, ny, nz = _flat_wall_state()
    before = populations.clone()
    diagnostics = diagnose_bfl_wall_exchange_state(
        populations,
        solid,
        links,
        q,
        1.0e-4,
        near_mask=near,
        stress_exchange_distance=1.0,
        wall_normals=(nx, ny, nz),
    )
    assert torch.equal(populations, before)
    assert diagnostics.active_nodes > 0
    assert diagnostics.y_plus_summary is not None
    assert diagnostics.pressure_gradient_parameter_mean is not None
    assert diagnostics.pressure_gradient_parameter_mean > 0.0
    assert diagnostics.pressure_gradient_parameter_max >= (
        diagnostics.pressure_gradient_parameter_mean
    )


def test_checkpoint_wall_diagnostic_rejects_shape_mismatch() -> None:
    populations, solid, links, q, near, nx, ny, nz = _flat_wall_state()
    try:
        diagnose_bfl_wall_exchange_state(
            populations[:, :, :, :-1],
            solid,
            links,
            q,
            1.0e-4,
            near_mask=near,
            wall_normals=(nx, ny, nz),
        )
    except ValueError as error:
        assert "solid" in str(error)
    else:
        raise AssertionError("shape mismatch was accepted")
