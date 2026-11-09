"""Unit tests for the contact-line angle measurement (IN100-N02).

The estimators are exercised against *synthetic* density fields whose contact
angle is known exactly, so the tests pin the geometric convention (which side is
advancing, how the tangent maps to the angle) rather than a fitted number.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.contact_line.engine import Channel, tanh_profile
from tensorlbm.contact_line.theta_fit import (
    angle_estimate,
    circle_fit,
    contact_angle_from_circle,
    interface_points,
    level_fraction,
    local_tangent_angle,
    wall_contact_angle,
)

RHO_L, RHO_V = 1.957, 0.1596
LEVEL = level_fraction(RHO_L, RHO_V)


def _cap_field(theta_deg: float, radius: float = 25.0, ny: int = 90, nx: int = 140):
    """Circular cap of known contact angle resting on the bottom wall."""
    ch = Channel(nx=nx, ny=ny, wall_rows=2)
    solid = ch.solid_mask()
    xx, yy = ch.grid()
    theta = math.radians(theta_deg)
    xc, yc = nx / 2.0, ch.y_wall() + radius * math.cos(theta)
    dist = torch.sqrt((xx - xc) ** 2 + (yy - yc) ** 2) - radius
    rho = tanh_profile(ch.shape, dist, RHO_L, RHO_V, 1.5)
    return ch, solid, rho


def _wedge_field(theta_front: float, theta_rear: float, ny: int = 120, nx: int = 160):
    """Sessile wedge with independent front/rear contact angles (exact tangents)."""
    ch = Channel(nx=nx, ny=ny, wall_rows=2)
    solid = ch.solid_mask()
    xx, yy = ch.grid()
    yw = ch.y_wall()

    def sig(z):
        return 0.5 * (1.0 + torch.tanh(z / 1.5))

    s_r = 1.0 / math.tan(math.radians(theta_front))
    s_l = -1.0 / math.tan(math.radians(theta_rear))
    x0r = nx / 2.0 + 12.0
    x0l = nx / 2.0 - 12.0
    ind = sig(yy - yw) * sig(x0r + s_r * (yy - yw) - xx) * sig(xx - (x0l + s_l * (yy - yw)))
    rho = RHO_V + (RHO_L - RHO_V) * ind
    return ch, solid, rho


class TestInterfacePoints:
    def test_circular_cap_contour(self):
        ch, solid, rho = _cap_field(90.0)
        pts = interface_points(rho, LEVEL, solid)
        assert pts.shape[1] == 2
        assert pts.shape[0] > 50
        # the level set of a tanh profile sits where the profile crosses LEVEL,
        # i.e. essentially on the radius; verify the mean radius
        xx, yy = ch.grid()
        r = torch.sqrt((pts[:, 0] - ch.nx / 2.0) ** 2 + (pts[:, 1] - (ch.y_wall() + 0.0)) ** 2)
        # centre height is y_wall for theta = 90 deg, cap radius = 25
        assert float(r.median()) == pytest.approx(25.0, abs=0.6)

    def test_solid_edges_excluded(self):
        ch, solid, rho = _cap_field(90.0)
        pts_noskip = interface_points(rho, LEVEL, None)
        pts_skip = interface_points(rho, LEVEL, solid)
        assert pts_skip.shape[0] < pts_noskip.shape[0]

    def test_no_crossing_returns_empty(self):
        rho = torch.full((10, 10), 1.0)
        pts = interface_points(rho, 1.5)
        assert pts.numel() == 0
        with pytest.raises(ValueError):
            angle_estimate(pts, 0.0)

    def test_1d_validation(self):
        with pytest.raises(ValueError):
            interface_points(torch.ones(3), 0.5)


class TestCircleFit:
    def test_exact_circle(self):
        t = torch.linspace(0.0, math.pi, 200)
        x = 5.0 + 3.0 * torch.cos(t)
        y = -2.0 + 3.0 * torch.sin(t)
        fit = circle_fit(torch.stack([x, y], dim=1))
        assert fit.xc == pytest.approx(5.0, abs=1e-4)
        assert fit.yc == pytest.approx(-2.0, abs=1e-4)
        assert fit.r == pytest.approx(3.0, abs=1e-4)
        assert fit.rms < 1e-4

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            circle_fit(torch.zeros((2, 2)))

    def test_contact_angle_convention(self):
        # circle of radius R whose centre is R*cos(theta) above the wall
        for theta in (20.0, 45.0, 90.0, 110.0, 150.0):
            th = math.radians(theta)
            R = 20.0
            yw = 1.5
            fit = circle_fit(
                torch.stack(
                    [
                        R * torch.sin(torch.linspace(0.2, math.pi - 0.2, 120))
                        + 30.0,
                        yw + R * math.cos(th)
                        + R * torch.cos(torch.linspace(0.2, math.pi - 0.2, 120)),
                    ],
                    dim=1,
                )
            )
            got, a = contact_angle_from_circle(fit, yw)
            assert got == pytest.approx(theta, abs=1e-3)
            assert a == pytest.approx(R * math.sin(th), abs=1e-3)

    def test_unreached_wall_gives_nan(self):
        t = torch.linspace(0.0, math.pi, 50)
        pts = torch.stack([torch.cos(t), 10.0 + torch.sin(t)], dim=1)
        fit = circle_fit(pts)
        got, a = contact_angle_from_circle(fit, 0.0)
        assert math.isnan(got) and math.isnan(a)


class TestAngleEstimate:
    @pytest.mark.parametrize("theta", [30.0, 60.0, 90.0, 120.0])
    def test_recovers_cap_angle(self, theta):
        ch, solid, rho = _cap_field(theta)
        pts = interface_points(rho, LEVEL, solid)
        est = angle_estimate(pts, ch.y_wall(), side="cap", bootstrap=16)
        assert est.theta_deg == pytest.approx(theta, abs=0.5)
        assert est.unc_deg > 0.0
        assert est.theta_by_window
        assert est.n_points >= 5

    def test_window_sweep_is_stable(self):
        ch, solid, rho = _cap_field(60.0)
        pts = interface_points(rho, LEVEL, solid)
        est = angle_estimate(pts, ch.y_wall(), side="cap", bootstrap=8)
        values = list(est.theta_by_window.values())
        assert max(values) - min(values) < 1.0

    def test_centre_x_required_for_side(self):
        ch, solid, rho = _cap_field(60.0)
        pts = interface_points(rho, LEVEL, solid)
        with pytest.raises(ValueError):
            angle_estimate(pts, ch.y_wall(), side="front")


class TestLocalTangent:
    @pytest.mark.parametrize("theta_f,theta_r", [(60.0, 30.0), (100.0, 80.0)])
    def test_recovers_wedge_angles(self, theta_f, theta_r):
        ch, solid, rho = _wedge_field(theta_f, theta_r)
        pts = interface_points(rho, LEVEL, solid)
        shift = torch.remainder(pts[:, 0] - ch.nx / 2.0 + ch.nx / 2, ch.nx) - ch.nx / 2
        ps = torch.stack([shift, pts[:, 1]], dim=1)
        front = local_tangent_angle(ps, ch.y_wall(), side="front", centre_x=0.0)
        rear = local_tangent_angle(ps, ch.y_wall(), side="rear", centre_x=0.0)
        assert front.theta_deg == pytest.approx(theta_f, abs=4.0)
        assert rear.theta_deg == pytest.approx(theta_r, abs=4.0)

    def test_symmetric_case_gives_equal_sides(self):
        ch, solid, rho = _cap_field(75.0)
        pts = interface_points(rho, LEVEL, solid)
        shift = torch.remainder(pts[:, 0] - ch.nx / 2.0 + ch.nx / 2, ch.nx) - ch.nx / 2
        ps = torch.stack([shift, pts[:, 1]], dim=1)
        front = local_tangent_angle(ps, ch.y_wall(), side="front", centre_x=0.0)
        rear = local_tangent_angle(ps, ch.y_wall(), side="rear", centre_x=0.0)
        assert front.theta_deg == pytest.approx(rear.theta_deg, abs=0.3)
        assert front.theta_deg == pytest.approx(75.0, abs=2.0)

    def test_bad_side(self):
        pts = torch.zeros((10, 2))
        with pytest.raises(ValueError):
            local_tangent_angle(pts, 0.0, side="up")


class TestWallContactAngle:
    def test_matches_front_tangent_on_wedge(self):
        ch, solid, rho = _wedge_field(80.0, 80.0)
        pts = interface_points(rho, LEVEL, solid)
        ps = torch.stack([pts[:, 0] - ch.nx / 2.0, pts[:, 1]], dim=1)
        bottom = wall_contact_angle(ps, ch.y_wall(), below=True, branch="max")
        assert bottom.theta_deg == pytest.approx(80.0, abs=4.0)
        assert bottom.side == "bottom"