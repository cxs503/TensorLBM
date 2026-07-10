"""Regression tests for the rotating-cylinder moving-wall boundary."""

import importlib.util
from pathlib import Path

import torch

from tensorlbm.d3q19 import C, OPPOSITE, W


_SPEC = importlib.util.spec_from_file_location(
    "benchmark_magnus_cylinder",
    Path(__file__).parents[1] / "examples" / "benchmark_magnus_cylinder.py",
)
_MAGNUS = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MAGNUS)


def test_moving_wall_correction_is_applied_only_to_solid_boundary_nodes():
    """Rigid-cylinder interior must use stationary BB, not wall velocity."""
    nz = ny = nx = 7
    yy, xx = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
    yy = yy.unsqueeze(0).float()
    xx = xx.unsqueeze(0).float()
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[:, 2:5, 2:5] = True
    f = torch.arange(19 * nz * ny * nx, dtype=torch.float32).reshape(19, nz, ny, nx)

    result = _MAGNUS.apply_moving_bounceback(
        f, solid, OPPOSITE, C, W, omega_eff=0.02, cx=2.0, cy=2.0, yy=yy, xx=xx,
    )

    # (3, 3) is inside the disk but not on its fluid-facing boundary.  It has
    # nonzero rigid-body velocity for this off-centre rotation, so the legacy
    # whole-solid treatment changes it while a boundary-only treatment must not.
    assert torch.equal(result[:, 0, 3, 3], f[OPPOSITE, 0, 3, 3])
    assert not torch.equal(result[:, 0, 3, 4], f[OPPOSITE, 0, 3, 4])


def test_moving_wall_bounceback_uses_each_fluid_solid_link_and_fluid_density():
    """Diagonal links and their source-fluid density enter moving BB.

    A staircase cylinder has diagonal fluid--solid links.  Applying a wall
    correction merely to an axial-shell mask misses those links and makes the
    effective surface velocity grid-orientation dependent.  The link formula
    also requires the adjacent fluid density, rather than a hard-coded one.
    """
    f = torch.zeros((19, 1, 5, 5), dtype=torch.float32)
    solid = torch.zeros((1, 5, 5), dtype=torch.bool)
    solid[0, 2, 2] = True
    # q=7 (+x,+y) has arrived at the solid after streaming from the diagonal
    # fluid neighbour.  Give that source-fluid cell density two.
    q_in = 7
    f[q_in, 0, 2, 2] = 2.0
    f[0, 0, 1, 1] = 2.0
    yy, xx = torch.meshgrid(torch.arange(5), torch.arange(5), indexing="ij")
    result = _MAGNUS.apply_moving_bounceback(
        f, solid, OPPOSITE, C, W, omega_eff=0.1, cx=2.0, cy=1.0,
        yy=yy.unsqueeze(0).float(), xx=xx.unsqueeze(0).float(),
    )

    # At x_s=(2,2), u_wall=(-0.1, 0), rho_fluid=2, and the returned
    # population is f_q + 6*w_q*rho_fluid*(c_q.u_wall).
    expected = 2.0 + 6.0 * W[q_in] * 2.0 * (-0.1)
    assert torch.isclose(result[OPPOSITE[q_in], 0, 2, 2], expected)


def test_momentum_exchange_includes_moving_wall_impulse():
    """Force includes the wall momentum imparted by a moving reflected link."""
    f = torch.zeros((19, 1, 5, 5), dtype=torch.float32)
    solid = torch.zeros((1, 5, 5), dtype=torch.bool)
    solid[0, 2, 2] = True
    q_in = 7
    f[q_in, 0, 2, 2] = 2.0
    f[0, 0, 1, 1] = 2.0
    yy, xx = torch.meshgrid(torch.arange(5), torch.arange(5), indexing="ij")
    force = _MAGNUS.compute_force_momentum_exchange(
        f, solid, C, W, omega_eff=0.1, cx=2.0, cy=1.0,
        yy=yy.unsqueeze(0).float(), xx=xx.unsqueeze(0).float(),
    )
    # Link impulse on the solid: 2*c*f_q + 6*c*w*rho_f*(c.u_wall),
    # where c=(1,1,0), u_wall=(-0.1,0), rho_f=2.
    expected = torch.tensor([4.0, 4.0, 0.0]) + 6.0 * W[q_in] * 2.0 * (-0.1) * C[q_in]
    assert torch.allclose(force, expected)
