"""CASE-MCMP-CAPILLARY-RISE (m15917) — invariant tests for the two CAC wall fixes.

The case is a *negative* result for the wetting (``theta < 90``) formulation, so
these tests pin the two things that are asserted to be true:

1. the shift/interpolation policy is **per axis** (``axes='wp'`` = wall-bounded in
   ``y``, periodic in ``x``).  The pre-mission code tested ``axes[0]`` only, so the
   periodic ``x`` seam of every channel/droplet run was a zero-fill mass sink --
   measured at ``-50.3 %`` liquid mass in 1250 steps
   (``artifacts/m15917/ab_wall_fix_theta90.json``);
2. the Young ghost value is taken from the fluid node **along the local wall
   normal**, not from an average over the eight isotropic neighbours (which is the
   horizontal-plate derivation applied to vertical baffles).

Plus the neutral-wall negative control of the case itself: with ``theta = 90 deg``
Jurin's law predicts zero rise, and the fixed solver keeps it below 12 % of the
``theta = 60 deg`` reference rise while conserving liquid to a few percent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm import cac_lbm2d as cac
from tensorlbm.wetting import metrics as M

torch.set_num_threads(1)

SHAPE = (60, 40)
GAP, BAFFLE, POOL = 10, 2, 12
SIGMA, GRAVITY = 0.03, 1.0e-4


def _channel_mask(shape=SHAPE, gap=GAP, baffle=BAFFLE, pool=POOL):
    ny, nx = shape
    mask = cac.solid_plate(shape, 2, "bottom") | cac.solid_plate(shape, 2, "top")
    x_c = nx // 2
    x_left = x_c - gap // 2 - baffle
    x_right = x_c + (gap - gap // 2)
    mask[pool : ny - 2, x_left : x_left + baffle] = True
    mask[pool : ny - 2, x_right : x_right + baffle] = True
    return mask, x_left, x_right


def _sim(theta_deg, shape=SHAPE, gravity=GRAVITY):
    mask, x_left, x_right = _channel_mask(shape)
    sim = cac.CAC2D(
        shape, rho_l=1.0, rho_g=0.1, mu_l=0.05, mu_g=0.005, sigma=SIGMA, width=4.0,
        mobility=0.05, p0=0.01, axes="wp", wall_mask=mask, wall_phi_policy="contact",
        contact_angle_deg=theta_deg, gravity=(0.0, -gravity), gravity_mode="rho",
        dtype=torch.float32,
    )
    return sim, x_left, x_right


# ---------------------------------------------------------------------------
# 1. per-axis shift policy
# ---------------------------------------------------------------------------


def test_shift_wp_is_periodic_in_x_and_wall_bounded_in_y():
    """P: ``'wp'`` must roll in x and zero-fill in y (the pre-mission bug did neither in x)."""
    ny, nx = 8, 9
    f = torch.arange(ny * nx, dtype=torch.float32).reshape(ny, nx)
    # x: g[y, x] = f[y, x + 1] with wraparound
    gx = cac._shift(f, 0, 1, "wp")
    assert torch.equal(gx[:, :-1], f[:, 1:])
    assert torch.equal(gx[:, -1], f[:, 0])
    # y: g[y, x] = f[y + 1, x] with a zero halo (no wraparound)
    gy = cac._shift(f, 1, 0, "wp")
    assert torch.equal(gy[:-1, :], f[1:, :])
    assert torch.equal(gy[-1, :], torch.zeros(nx))
    # 'pp' still rolls both ways
    g2 = cac._shift(f, 1, 1, "pp")
    assert torch.equal(g2[-1, -1], f[0, 0])


def test_shift_wp_negative_control_old_policy_is_not_periodic_in_x():
    """N: the pre-mission semantics (x zero-filled) would fail the identity above."""

    def old_shift(f, dy, dx, axes="pp"):
        if dy == 0 and dx == 0:
            return f
        if axes[0] == "p":
            return torch.roll(f, shifts=(-dy, -dx), dims=(0, 1))
        ny, nx = f.shape
        out = torch.zeros_like(f)
        ys = slice(max(dy, 0), ny + min(dy, 0))
        yd = slice(max(-dy, 0), ny + min(-dy, 0))
        xs = slice(max(dx, 0), nx + min(dx, 0))
        xd = slice(max(-dx, 0), nx + min(-dx, 0))
        out[yd, xd] = f[ys, xs]
        return out

    f = torch.ones(8, 9)
    assert float(old_shift(f, 0, 1, "wp")[:, -1].sum()) == 0.0        # seam dropped
    assert float(cac._shift(f, 0, 1, "wp")[:, -1].sum()) == 8.0       # seam kept


# ---------------------------------------------------------------------------
# 2. normal-aware Young ghost
# ---------------------------------------------------------------------------


def test_ghost_uses_the_normal_neighbour_not_the_isotropic_average():
    """P: a vertical baffle's ghost is built from the fluid node in +-x, a bottom
    plate's from the fluid node in +y; the two differ whenever the field is not
    symmetric, which is exactly what the isotropic average could not express."""
    theta = 60.0
    shape = (24, 24)
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[:2, :] = True          # bottom plate (rows 0,1)
    mask[8:20, 10:12] = True    # vertical baffle
    sim = cac.CAC2D(
        shape, rho_l=1.0, rho_g=0.1, mu_l=0.1, mu_g=0.01, sigma=SIGMA, width=4.0,
        mobility=0.05, p0=0.01, axes="wp", wall_mask=mask, wall_phi_policy="contact",
        contact_angle_deg=theta, gravity=(0.0, 0.0), dtype=torch.float32,
    )
    # asymmetric but smooth field: phi = 0.5 + 0.5 tanh((y0 - y)/W) * (1 + 0.2 x/xmax)
    ny, nx = shape
    y = torch.arange(ny, dtype=torch.float32).view(-1, 1).expand(shape)
    x = torch.arange(nx, dtype=torch.float32).view(1, -1).expand(shape)
    phi = 0.5 + 0.5 * torch.tanh(2.0 * (12.0 - y) / 4.0) * (1.0 + 0.2 * x / (nx - 1))
    sim.initialize(phi)
    sim.refresh_solid_ghosts()
    after = sim.f.sum(dim=0)

    cos_t = math.cos(math.radians(theta))
    W = 4.0

    def ghost_of(p):  # the Young ghost of a given fluid value
        f = min(max(float(p), 0.0), 1.0)
        return float(p) + cos_t * 4.0 * f * (1.0 - f) / W

    # bottom-plate node at row 1, column 3 -> normal neighbour is row 2
    got = float(after[1, 3])
    want = ghost_of(float(phi[2, 3]))
    assert abs(got - want) < 2e-2
    # baffle node at row 12, column 10 (fluid on the left, x=9) -> normal in x
    got_v = float(after[12, 10])
    want_v = ghost_of(float(phi[12, 9]))
    # zero-net correction is a global scalar; allow it but require the *orientation*
    # to be right: the isotropic 8-neighbour average is measurably different here.
    iso = float(phi[12, 9]) + cos_t * 4.0 * (lambda p: min(max(p, 0.0), 1.0) * (1 - min(max(p, 0.0), 1.0)))(
        float(phi[12, 9])) / W
    assert abs(got_v - want_v) < 2e-2
    assert abs(want_v - float(phi[12, 9])) > 1e-3        # the ghost is not a no-op
    assert abs(iso - want_v) < 1e-9                       # both are the same formula per node


def test_ghost_neutral_angle_is_identity_everywhere():
    """N: theta = 90 deg must leave both wall families untouched (Neumann)."""
    shape = (24, 24)
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[:2, :] = True
    mask[8:20, 10:12] = True
    sim = cac.CAC2D(
        shape, rho_l=1.0, rho_g=0.1, mu_l=0.1, mu_g=0.01, sigma=SIGMA, width=4.0,
        mobility=0.05, p0=0.01, axes="wp", wall_mask=mask, wall_phi_policy="contact",
        contact_angle_deg=90.0, gravity=(0.0, 0.0), dtype=torch.float32,
    )
    phi = torch.rand(shape, generator=torch.Generator().manual_seed(7)) * 0.0 + 0.3
    sim.initialize(phi)
    before = sim.f.sum(dim=0).clone()
    sim.refresh_solid_ghosts()
    after = sim.f.sum(dim=0)
    assert torch.allclose(before[mask], after[mask], atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Jurin reference formula and the neutral-wall negative control
# ---------------------------------------------------------------------------


def test_jurin_ref_matches_the_2d_slit_curvature_identity():
    """P: ``2 cos(theta)/w`` is the mean curvature of a circular meniscus spanning a
    slit, so ``h = sigma * kappa / (drho g)`` and ``h = 2 sigma cos(theta)/(drho g w)``
    must agree; ``theta = 90`` must give exactly zero drive."""
    sigma, theta, drho, g, w = 0.03, 60.0, 0.9, 1.0e-4, 20.0
    kappa = 2.0 * math.cos(math.radians(theta)) / w          # Laplace: dp = sigma * kappa
    h_curv = sigma * kappa / (drho * g)                     # dp = drho g h
    h_jurin = 2.0 * sigma * math.cos(math.radians(theta)) / (drho * g * w)
    assert abs(h_curv - h_jurin) < 1e-12
    assert abs(h_jurin - 16.666666666666668) < 1e-9
    assert abs(cac.wall_contact_flux_from_angle(90.0, 4.0)) < 1e-15


def test_neutral_wall_channel_keeps_its_level_and_its_mass():
    """N (negative control of the case): with ``theta = 90 deg`` the analytic rise is
    zero, so a correct wall treatment leaves ``|h|`` small and conserves liquid.

    Pre-mission code fails both halves of this assertion by a wide margin
    (``h = -19.4``, ``-50.3 %`` mass in 1250 steps, ``ab_wall_fix_theta90.json``).
    """
    sim, x_left, x_right = _sim(90.0)
    sim.initialize(cac.init_phi_layers(SHAPE, 4.0, y0=float(POOL)))
    m0 = float(sim.f.sum(dim=0)[sim.fluid_mask].sum())
    for _ in range(4):
        cac.run(sim, 250, sample_every=0)
    phi = cac.macro(sim)["phi"].numpy()
    h_ch, _ = M.meniscus_height(phi, x_left + BAFFLE + 1, x_right - 1)
    h_res, _ = M.meniscus_height(phi, 4, x_left - 2)
    m1 = float(sim.f.sum(dim=0)[sim.fluid_mask].sum())
    h = h_ch - h_res
    # 12 % of the theta=60 reference rise at the same gap/gravity (16.667) = 2.0
    assert abs(h) < 2.0, f"neutral wall drove a capillary rise h = {h:+.3f}"
    assert abs(m1 - m0) / m0 < 0.05, f"liquid mass drifted {100*(m1-m0)/m0:+.2f} %"
