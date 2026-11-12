"""Audit m16154 (2026-11-12): SuctionWallBC / ASBL defects are gated.

These tests do *not* claim accuracy. They pin down the three defects that were
found by independent audit of the shipped code and make sure the boundary
condition cannot be used silently as if it were verified.

Evidence table (measured in this audit) is in
``docs/AUDIT_m16154_SUCTION_ASBL.md``.
"""

import torch
import pytest

from tensorlbm.boundary.suction import (
    SuctionWallBC,
    SuctionWallBCUnverifiedError,
    apply_suction_wall,
)
from tensorlbm.boundary.solver2d import (
    collide_bgk2d,
    equilibrium2d,
    macroscopic2d,
    stream2d,
)
from tensorlbm.verification.asbl.simulation import ASBLSimulation


def _equilibrium_field(ny=16, nx=4, u=0.0, v=0.0, dtype=torch.float64):
    rho = torch.ones(ny, nx, dtype=dtype)
    ux = torch.full((ny, nx), float(u), dtype=dtype)
    uy = torch.full((ny, nx), float(v), dtype=dtype)
    return equilibrium2d(rho, ux, uy)


def test_withheld_by_default_bottom_and_top():
    f = _equilibrium_field()
    for wall_y in (0, 15):
        bc = SuctionWallBC(0.05, 1e-3, 0.56, wall_y=wall_y)
        with pytest.raises(SuctionWallBCUnverifiedError):
            bc.apply(f.clone())
    with pytest.raises(SuctionWallBCUnverifiedError):
        apply_suction_wall(f.clone(), 0.05, 1e-3, 0.56)


def test_top_wall_branch_no_longer_silent_noop():
    """Defect 2: wall_y != 0 used to return the populations untouched."""
    f = _equilibrium_field(ny=16, nx=4)
    top = SuctionWallBC(0.05, 0.0, 0.56, wall_y=15, u_wall=0.05)
    g = top.apply(f.clone(), allow_unverified=True)
    # the wall row must have changed, and only the wall row
    assert not torch.allclose(g[:, 15, :], f[:, 15, :])
    assert torch.allclose(g[:, :15, :], f[:, :15, :])


def test_prescribed_tangential_velocity_enters_reconstruction():
    """Defect 1/3: u_wall must at least influence the reconstructed row."""
    f = _equilibrium_field(ny=16, nx=4)
    a = SuctionWallBC(0.05, 0.0, 0.56, wall_y=15, u_wall=0.0)
    b = SuctionWallBC(0.05, 0.0, 0.56, wall_y=15, u_wall=0.05)
    ga = a.apply(f.clone(), allow_unverified=True)
    gb = b.apply(f.clone(), allow_unverified=True)
    assert not torch.allclose(ga[:, 15, :], gb[:, 15, :])
    _, ua, _ = macroscopic2d(ga)
    _, ub, _ = macroscopic2d(gb)
    # u_wall = 0.05 must raise the wall-row streamwise velocity
    assert ub[15].mean().item() > ua[15].mean().item()


def test_repaired_scheme_still_fails_couette_sanity_check():
    """Documented residual defect: steady Couette is NOT reproduced.

    Analytic plane Couette between row 0 (u=0) and row ny-1 (u=U) has a
    constant shear  U/(ny-1).  The repaired scheme gives about twice that
    slope plus a wall-adjacent undershoot, which is why the BC is WITHHELD.
    This test fails loudly if the number changes, forcing a re-audit.
    """
    ny, nx, U, nu = 16, 4, 0.05, 0.02
    tau = 0.5 + 3.0 * nu
    f = _equilibrium_field(ny=ny, nx=nx)
    bot = SuctionWallBC(U, 0.0, tau, wall_y=0, u_wall=0.0)
    top = SuctionWallBC(U, 0.0, tau, wall_y=ny - 1, u_wall=U)
    for _ in range(4000):
        f = collide_bgk2d(f, tau)
        f = stream2d(f)
        f = bot.apply(f, allow_unverified=True)
        f = top.apply(f, allow_unverified=True)
    rho, ux, uy = macroscopic2d(f)
    prof = ux[:, 0]
    assert torch.isfinite(prof).all()
    analytic_slope = U / (ny - 1)
    interior_slope = (prof[ny - 3] - prof[2]).item() / (ny - 5)
    assert interior_slope > 1.5 * analytic_slope, (
        f"repaired BC now matches Couette (slope {interior_slope:.5f} vs "
        f"analytic {analytic_slope:.5f}) -- re-audit and possibly promote"
    )
    assert prof[1].item() < 0.0, "wall-adjacent undershoot disappeared -- re-audit"


def test_asbl_driver_opts_in_explicitly():
    """The withdrawn ASBL driver must not silently pretend to be verified."""
    sim = ASBLSimulation(nx=8, ny=16, U_inf=0.05, V_suction=1e-3, nu=0.02,
                         max_steps=1)
    assert sim.wall_bc.u_wall == 0.0
