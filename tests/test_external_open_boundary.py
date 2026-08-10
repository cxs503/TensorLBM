from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d


def test_far_field_equilibrium_is_fixed_point() -> None:
    shape = (7, 9, 11)
    rho = torch.ones(shape)
    ux = torch.full(shape, 0.04)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, ux, zero, zero)
    out = non_equilibrium_far_field_bc_3d(f, u_in=0.04)
    assert torch.allclose(out, f, atol=1e-7, rtol=0.0)


def test_outgoing_populations_are_not_overwritten() -> None:
    shape = (5, 7, 9)
    rho = torch.ones(shape)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, zero, zero, zero)
    # At x+, cx>0 populations leave the domain and must survive verbatim.
    outgoing = (C[:, 0] > 0).nonzero().flatten()
    f[outgoing, :, :, -1] += 0.0123
    out = non_equilibrium_far_field_bc_3d(f, u_in=0.0, faces=("x+",))
    assert torch.equal(out[outgoing, :, :, -1], f[outgoing, :, :, -1])


def test_incoming_outlet_nonequilibrium_is_extrapolated() -> None:
    shape = (5, 7, 9)
    rho = torch.ones(shape)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, zero, zero, zero)
    direction = int((C[:, 0] < 0).nonzero()[0])
    f[direction, :, :, -2] += 1e-3
    out = non_equilibrium_far_field_bc_3d(f, u_in=0.0, faces=("x+",))
    assert torch.all(out[direction, :, :, -1] > f[direction, :, :, -1])


def test_boundary_population_delta_ledger_closes_exactly() -> None:
    shape = (5, 7, 9)
    rho = torch.ones(shape)
    ux = torch.full(shape, 0.04)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, ux, zero, zero)
    f[:, :, :, 1] *= 1.001

    out, diagnostics = non_equilibrium_far_field_bc_3d(
        f,
        u_in=0.04,
        faces=("x-", "x+"),
        return_diagnostics=True,
    )

    delta = out - f
    expected_mass = float(delta.sum().item())
    expected_momentum = tuple(
        float(value.item())
        for value in (delta.sum(dim=(1, 2, 3))[:, None] * C.to(delta.dtype)).sum(dim=0)
    )
    assert diagnostics.mass_delta == pytest.approx(expected_mass, abs=1e-7)
    assert diagnostics.momentum_delta == pytest.approx(
        expected_momentum,
        abs=1e-7,
    )
    assert diagnostics.updated_populations > 0
    assert diagnostics.finite


def test_equilibrium_boundary_diagnostic_has_zero_population_delta() -> None:
    shape = (5, 7, 9)
    rho = torch.ones(shape)
    ux = torch.full(shape, 0.04)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, ux, zero, zero)

    out, diagnostics = non_equilibrium_far_field_bc_3d(
        f,
        u_in=0.04,
        return_diagnostics=True,
    )

    assert torch.allclose(out, f, atol=1e-7, rtol=0.0)
    assert diagnostics.mass_delta == pytest.approx(0.0, abs=2e-5)
    assert diagnostics.momentum_delta == pytest.approx((0.0, 0.0, 0.0), abs=2e-5)
