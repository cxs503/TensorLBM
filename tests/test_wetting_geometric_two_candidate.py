"""Tests for the two-candidate geometrical wetting condition (openLBMPM recipe).

Locks the three statements the XVAL report relies on:

* with a *mirror*-filled solid field (a masked-wall colour-gradient solver) the
  two-candidate rule degenerates exactly to the shipped closed-form condition of
  :mod:`tensorlbm.cg_wetting2d` -- the sign ambiguity the reference resolves does
  not arise, because the measured one-sided wall-normal difference is zero;
* the rule does switch sign when the solid colour value is prescribed
  (openLBMPM ``SolidColorDiff``), which is where the reference's candidate
  branch gets its sign from -- implemented here as ``solid_color_offset``;
* ``theta = 90`` reduces to the neutral (mirror) wall, the interface gate leaves
  the mirror value plus ``SolidColorDiff``, and the values stay clamped.

The cap-geometry estimator (openLBMPM's own contact-angle recipe) is checked
against exact circular caps.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.cg_wetting2d import geometrical_wetting_ghost
from tensorlbm.wetting.geometric_two_candidate import (
    cap_geometry_contact_angle,
    geometric_wetting_ghost_candidates,
)

ANGLES = (30.0, 45.0, 60.0, 90.0, 120.0, 150.0)


def _flat_wall(ny: int = 96, nx: int = 128) -> torch.Tensor:
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    return solid


def _cap(ny: int, nx: int, theta_deg: float, xi: float = 2.0, radius: float = 24.0) -> torch.Tensor:
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    yc = 0.5 - radius * math.cos(math.radians(theta_deg))
    r = torch.sqrt((x - nx / 2.0) ** 2 + (y - yc) ** 2)
    return torch.tanh((radius - r) / xi)


@pytest.mark.parametrize("theta", ANGLES)
@pytest.mark.parametrize("xi", (1.5, 2.0))
def test_candidate_degenerates_to_closed_form_with_mirror_fill(theta, xi):
    """Mirror fill -> zero measured normal difference -> exact agreement."""
    ny, nx = 96, 128
    solid = _flat_wall(ny, nx)
    phi = _cap(ny, nx, theta, xi)
    a = geometrical_wetting_ghost(phi, solid, theta, periodic=(True, False))
    b, diag = geometric_wetting_ghost_candidates(
        phi, solid, theta, periodic=(True, False), return_diagnostics=True
    )
    assert diag["n_sign_flips"] == 0.0
    assert torch.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("theta", (45.0, 60.0))
def test_prescribed_solid_colour_switches_the_sign(theta):
    """``SolidColorDiff`` < 0 flips the imposed wall-normal difference."""
    ny, nx = 96, 128
    solid = _flat_wall(ny, nx)
    phi = _cap(ny, nx, theta)
    a = geometrical_wetting_ghost(phi, solid, theta, periodic=(True, False))
    b, diag = geometric_wetting_ghost_candidates(
        phi, solid, theta, periodic=(True, False),
        solid_color_offset=-0.5, return_diagnostics=True,
    )
    assert diag["n_sign_flips"] > 0.0
    d_a = a - phi
    d_b = b - phi
    m = (d_a.abs() > 1e-9) & (d_b.abs() > 1e-9)
    assert bool((torch.sign(d_a[m]) != torch.sign(d_b[m])).any())


def test_theta_ninety_is_mirror_wall():
    ny, nx = 32, 32
    solid = _flat_wall(ny, nx)
    torch.manual_seed(0)
    phi = torch.rand(ny, nx, dtype=torch.float64) * 2.0 - 1.0
    ghost = geometric_wetting_ghost_candidates(phi, solid, 90.0, periodic=(True, False))
    assert torch.allclose(ghost[0], phi[1], atol=1e-12)
    assert torch.allclose(ghost[-1], phi[-2], atol=1e-12)


def test_interface_gate_leaves_mirror_plus_solid_color_diff():
    ny, nx = 16, 16
    solid = _flat_wall(ny, nx)
    phi = torch.full((ny, nx), 0.3, dtype=torch.float64)
    ghost, diag = geometric_wetting_ghost_candidates(
        phi, solid, 60.0, solid_color_offset=0.5, periodic=(True, False), return_diagnostics=True
    )
    assert diag["n_interface_face_nodes"] == 0.0
    assert torch.allclose(ghost[0], torch.full((nx,), 0.8, dtype=torch.float64))
    ghost_hi = geometric_wetting_ghost_candidates(
        phi, solid, 60.0, solid_color_offset=0.9, periodic=(True, False)
    )
    assert float(ghost_hi.max()) <= 1.0


def test_rejects_bad_angle_stencil_and_shape():
    solid = _flat_wall(8, 8)
    phi = torch.zeros(8, 8)
    with pytest.raises(ValueError):
        geometric_wetting_ghost_candidates(phi, solid, 0.0)
    with pytest.raises(ValueError):
        geometric_wetting_ghost_candidates(phi, solid, 180.0)
    with pytest.raises(ValueError):
        geometric_wetting_ghost_candidates(phi, solid, 60.0, stencil="nope")
    with pytest.raises(ValueError):
        geometric_wetting_ghost_candidates(torch.zeros(4, 4), solid, 60.0)


@pytest.mark.parametrize("theta", ANGLES)
def test_cap_geometry_recovers_analytic_cap_angle(theta):
    """openLBMPM's (b, h) recipe returns the circular-cap angle to < 0.6 deg."""
    ny, nx = 96, 128
    phi = _cap(ny, nx, theta, xi=2.0)
    res = cap_geometry_contact_angle(phi, y_wall=0.5)
    assert res["status"] == "ok"
    assert abs(res["theta_deg"] - theta) < 0.6


def test_cap_geometry_reports_degenerate_field():
    res = cap_geometry_contact_angle(torch.zeros(16, 16))
    assert res["status"] != "ok"
    assert math.isnan(res["theta_deg"])