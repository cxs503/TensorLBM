"""Unit checks for the geometrical wetting BC (:mod:`tensorlbm.cg_wetting2d`).

The decisive test builds an *analytic* straight interface that meets the wall
under a known angle and checks the two things the boundary condition claims:

1. the ghost value equals the analytic ``phi`` one lattice step into the wall
   (i.e. the imposed one-sided wall-normal derivative is the rotated one), and
2. the angle read back from the imposed wall gradient
   (``wall_local_contact_angle``) reproduces the prescribed angle.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.cg_wetting2d import (
    geometrical_wetting_ghost,
    phase_field,
    wall_local_contact_angle,
)
from tensorlbm.color_gradient2d import color_gradient_two_tau_step


def _bottom_wall(ny: int = 64, nx: int = 64) -> torch.Tensor:
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True
    return solid


def _bottom_faces(ny: int, nx: int) -> torch.Tensor:
    face = torch.zeros(ny, nx, dtype=torch.bool)
    face[1, :] = True
    return face


def _analytic_interface(theta_deg: float, ny: int = 64, nx: int = 64, xi: float = 2.0):
    """``phi = tanh(s/xi)`` with ``s`` the signed distance to a line at ``theta``.

    The line passes through ``(nx/2, 0.5)`` with slope ``tan(theta)``; the
    normal ``(sin theta, -cos theta)`` points into the ``phi > 0`` (wetting)
    phase, so ``theta`` is the contact angle inside that phase.
    """
    th = math.radians(theta_deg)
    n = torch.tensor([math.sin(th), -math.cos(th)], dtype=torch.float64)
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    s = n[0] * (x - nx / 2.0) + n[1] * (y - 0.5)
    return torch.tanh(s / xi)


@pytest.mark.parametrize("theta", [30.0, 45.0, 60.0, 90.0, 120.0, 150.0])
def test_ghost_reproduces_rotated_normal(theta: float) -> None:
    ny, nx = 64, 64
    solid = _bottom_wall(ny, nx)
    phi = _analytic_interface(theta, ny, nx, xi=2.0)
    ghost = geometrical_wetting_ghost(phi, solid, theta)

    angles = wall_local_contact_angle(phi, ghost, solid, face_subset=_bottom_faces(ny, nx))
    assert angles.numel() >= 4
    med = float(angles.median())
    assert abs(med - theta) < 1.0, f"theta={theta} read back as {med:.2f}"

    # ghost value == analytic phi one lattice step into the wall (to first order)
    active = ghost[0, :].abs() > 1e-3
    assert int(active.sum()) > 5
    ref = _analytic_interface(theta, ny, nx, xi=2.0)[0, :]
    err = (ghost[0, active] - ref[active]).abs()
    assert float(err.median()) < 0.05


def test_neutral_angle_is_mirror() -> None:
    """``theta = 90`` -> zero wall-normal derivative, i.e. ghost == phi(face)."""
    solid = _bottom_wall()
    phi = _analytic_interface(45.0)
    ghost = geometrical_wetting_ghost(phi, solid, 90.0)
    assert torch.allclose(ghost[0, :], phi[1, :], atol=1e-9)


def test_no_interface_ghost_is_mirror() -> None:
    """Without an interface the constraint degenerates to a mirror value.

    The ghost then carries the bulk phase, so the solver sees a zero
    wall-normal phase derivative and no spurious wall force.
    """
    solid = _bottom_wall()
    phi = torch.full((64, 64), -1.0, dtype=torch.float64)
    ghost, diag = geometrical_wetting_ghost(phi, solid, 60.0, return_diagnostics=True)
    assert diag["n_interface_face_nodes"] == 0.0
    assert torch.allclose(ghost[0, :], torch.full((64,), -1.0, dtype=torch.float64))


def test_theta_out_of_range_rejected() -> None:
    solid = _bottom_wall()
    phi = _analytic_interface(60.0)
    for bad in (0.0, 180.0, -10.0, 200.0):
        with pytest.raises(ValueError):
            geometrical_wetting_ghost(phi, solid, bad)


def test_ghost_clamped_to_physical_range() -> None:
    solid = _bottom_wall()
    phi = _analytic_interface(5.0)
    ghost = geometrical_wetting_ghost(phi, solid, 5.0)
    assert float(ghost.max()) <= 1.0 + 1e-12
    assert float(ghost.min()) >= -1.0 - 1e-12


def test_theta_above_90_inverts_wall_wetting() -> None:
    """``theta > 90`` must drain the wetting phase away from the wall."""
    solid = _bottom_wall()
    phi = _analytic_interface(120.0)
    ghost = geometrical_wetting_ghost(phi, solid, 120.0)
    active = (ghost[0, :].abs() > 1e-3) & (phi[1, :].abs() < 0.9)
    # toward the wall the phase must decrease for a non-wetting wall
    assert int(active.sum()) > 3
    assert torch.all(ghost[0, active] < phi[1, active])


def test_color_gradient_step_backward_compatible() -> None:
    """``phi_solid=None`` must reproduce the stock masked-wall step bit-for-bit."""
    ny, nx, q = 16, 16, 9
    torch.manual_seed(0)
    f_r = torch.rand(q, ny, nx, dtype=torch.float64) * 0.1
    f_b = torch.rand(q, ny, nx, dtype=torch.float64) * 0.1
    solid = _bottom_wall(ny, nx)
    ref_r, ref_b = color_gradient_two_tau_step(f_r, f_b, solid_mask=solid)
    got_r, got_b = color_gradient_two_tau_step(f_r, f_b, solid_mask=solid, phi_solid=None)
    assert torch.equal(ref_r, got_r)
    assert torch.equal(ref_b, got_b)

    ghost = geometrical_wetting_ghost(phase_field(f_r.sum(0), f_b.sum(0)), solid, 60.0)
    wet_r, wet_b = color_gradient_two_tau_step(f_r, f_b, solid_mask=solid, phi_solid=ghost)
    # a non-neutral wall changes the post-collision state of fluid cells that
    # see the wall, and leaves solid cells untouched
    assert not torch.allclose(wet_r, ref_r)
    assert torch.equal(wet_r[:, solid], ref_r[:, solid])