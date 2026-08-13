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


def test_outlet_density_is_extrapolated_from_interior() -> None:
    # Regression guard for the high-Re pressure-wave-reflection fix: at x+
    # (outlet) the incoming populations must be reconstructed with the
    # interior-extrapolated density (zero-gradient), NOT rho_far.  The outgoing
    # populations are left untouched, so we assert on the incoming direction
    # populations specifically.
    shape = (5, 7, 9)
    rho = torch.ones(shape)
    rho[:, :, -2] = 1.05  # perturb the interior plane adjacent to x+
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, zero, zero, zero)
    out = non_equilibrium_far_field_bc_3d(f, u_in=0.0, rho_far=1.0, faces=("x+",))
    incoming = (C[:, 0] < 0).nonzero().flatten().tolist()
    feq_interior = equilibrium3d(rho[:, :, -2], zero[:, :, -2], zero[:, :, -2], zero[:, :, -2])
    feq_far = equilibrium3d(torch.ones_like(rho[:, :, -2]), zero[:, :, -2], zero[:, :, -2], zero[:, :, -2])
    for d in incoming:
        # incoming population at the outlet == feq(rho=1.05) (interior extrapolation)
        assert torch.allclose(out[d, :, :, -1], feq_interior[d], atol=1e-7)
        # and crucially NOT feq(rho_far=1.0) (the old reflecting behaviour)
        assert not torch.allclose(out[d, :, :, -1], feq_far[d], atol=1e-7)
