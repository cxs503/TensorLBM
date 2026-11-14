"""Tests for the moving-wall (non-zero Dirichlet) term of the alpha-family operator.

Anchor ``XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME``.

The alpha form of linear interpolated bounce-back carries a Dirichlet term
``-alpha e^-_q(x_w)`` with ``e^-_q = (f^eq_q - f^eq_opp(q))/2 = 3 w_q rho_w (c_q . u_w)``
(quadratic equilibrium terms cancel between opposite directions).  It vanishes
only for a wall at rest, which is why the historical two-branch operator could
ignore it.  A previous mission's diff list claimed the term "vanishes for the
vertical cut links of this benchmark" -- true -- and treated that as "vanishes
for a wall-aligned channel", which is **false** for a *sliding* wall-aligned wall:
the diagonal cut links carry ``c_qx u_wx != 0``.  ``test_value_term_*`` below pin
both statements.

Reference for the closure: pylbm ``LinearBounceBack`` / lbmpy
``NoSlipLinearBouzidi``; benchmark geometry and exact solutions as in
``xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME/REPORT_V3.md``.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.boundary.interpolated_bounce_back import (
    alpha_bfl,
    alpha_cli,
    interpolated_bounce_back,
    interpolated_bounce_back_2d,
    interpolated_bounce_back_3d,
)
from tensorlbm.d2q9 import C as C9
from tensorlbm.d2q9 import OPPOSITE as OPP9
from tensorlbm.d2q9 import W as W9
from tensorlbm.d3q19 import C as C19
from tensorlbm.d3q19 import OPPOSITE as OPP19
from tensorlbm.d3q19 import W as W19

DT = torch.float64


def _channel(n_fluid=4, delta=0.3, nx=3, n_wall=2):
    ny = n_fluid + 2 * n_wall
    H = (n_fluid - 1) + 2 * delta
    y = torch.arange(ny, dtype=DT) - n_wall + delta
    sdf1d = torch.minimum(y, H - y)
    fluid1d = torch.zeros(ny, dtype=torch.bool)
    fluid1d[n_wall : n_wall + n_fluid] = True
    sdf = sdf1d.view(ny, 1).expand(ny, nx).contiguous()
    fluid = fluid1d.view(ny, 1).expand(ny, nx).contiguous()
    return sdf, fluid, ~fluid, n_wall


def _state(ny, nx, seed=0):
    g = torch.Generator().manual_seed(seed)
    rho = 1.0 + 0.1 * torch.rand(ny, nx, generator=g, dtype=DT)
    ux = 0.02 * torch.rand(ny, nx, generator=g, dtype=DT)
    uy = 0.01 * torch.rand(ny, nx, generator=g, dtype=DT)
    cu = C9[:, 0].view(9, 1, 1) * ux + C9[:, 1].view(9, 1, 1) * uy
    feq = W9.view(9, 1, 1) * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * (ux**2 + uy**2))
    f_pre = feq + 1e-4 * torch.rand(9, ny, nx, generator=g, dtype=DT)
    f_new = torch.roll(f_pre, shifts=1, dims=1)
    return f_new, f_pre


def _wall_velocity(ny, nx, ux, uy=0.0):
    wv = torch.zeros((ny, nx, 2), dtype=DT)
    wv[..., 0] = ux
    wv[..., 1] = uy
    return wv


# --------------------------------------------------------------------------- #
# 1. the term itself
# --------------------------------------------------------------------------- #
def test_value_term_equals_analytic_dirichlet_expression():
    """Delta between ``wall_velocity=u`` and ``wall_velocity=0`` is exactly ``-alpha e^-_q``."""
    sdf, fluid, solid, _ = _channel()
    ny, nx = sdf.shape
    f_new, f_pre = _state(ny, nx)
    u_w = 0.013
    out0 = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid, alpha=alpha_bfl,
                                       wall_velocity=_wall_velocity(ny, nx, 0.0))
    out1 = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid, alpha=alpha_bfl,
                                       wall_velocity=_wall_velocity(ny, nx, u_w))
    diff = out1 - out0
    assert torch.isfinite(diff).all()

    seen = 0
    for i in range(1, int(C9.shape[0])):
        cx, cy = int(C9[i, 0]), int(C9[i, 1])
        shift = (-cy, -cx)
        link = fluid & torch.roll(solid, shift, dims=(0, 1))
        if not bool(link.any()):
            continue
        # wall distance of the cut link and the alpha actually used
        s_f = sdf
        s_s = torch.roll(sdf, shift, dims=(0, 1))
        q = (s_f / (s_f - s_s)).clamp(0.05, 0.95)
        a = alpha_bfl(q)
        expected = -a * 3.0 * float(W9[i]) * cx * u_w           # e^- = 3 w c_x u_x
        d = int(OPP9[i].item())
        got = torch.roll(diff[d], shift, dims=(0, 1))           # delivered to x_s = x_f + c_i
        assert torch.allclose(got[link], expected[link], rtol=1e-12, atol=1e-15), (i, cx, cy)
        seen += 1
    assert seen >= 5          # vertical + 4 diagonals of the two cut rows


def test_value_term_is_zero_on_wall_normal_links_and_nonzero_on_diagonals():
    """A sliding *wall-aligned* wall still needs the term (diagonal cut links)."""
    sdf, fluid, solid, _ = _channel()
    ny, nx = sdf.shape
    f_new, f_pre = _state(ny, nx)
    u_w = 0.02
    out0 = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid, alpha=alpha_bfl)
    out1 = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid, alpha=alpha_bfl,
                                       wall_velocity=_wall_velocity(ny, nx, u_w))
    diff = (out1 - out0).abs()
    for i in range(1, int(C9.shape[0])):
        cx, cy = int(C9[i, 0]), int(C9[i, 1])
        d = int(OPP9[i].item())
        shift = (-cy, -cx)
        link = fluid & torch.roll(solid, shift, dims=(0, 1))
        if not bool(link.any()):
            continue
        got = torch.roll(diff[d], shift, dims=(0, 1))[link]
        if cx == 0:      # wall-normal cut link (c_x = 0): term vanishes
            assert torch.allclose(got, torch.zeros_like(got), atol=1e-20), i
        else:            # diagonal cut link: term is present
            assert float(got.abs().max()) > 1e-9, i


def test_zero_wall_velocity_is_bit_identical_to_none():
    """Backward compatibility: ``wall_velocity=0`` must not change a single bit."""
    sdf, fluid, solid, _ = _channel()
    ny, nx = sdf.shape
    f_new, f_pre = _state(ny, nx, seed=3)
    a = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid, alpha=alpha_cli)
    b = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid, alpha=alpha_cli,
                                    wall_velocity=_wall_velocity(ny, nx, 0.0))
    assert torch.equal(a, b)
    assert torch.equal(a, interpolated_bounce_back(f_new, f_pre, sdf, solid, C9, OPP9, alpha=alpha_cli))


def test_wall_density_scales_the_term():
    sdf, fluid, solid, _ = _channel()
    ny, nx = sdf.shape
    f_new, f_pre = _state(ny, nx, seed=5)
    kw = dict(alpha=alpha_bfl)
    ref = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid,
                                     wall_velocity=_wall_velocity(ny, nx, 0.0), **kw)
    one = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid,
                                      wall_velocity=_wall_velocity(ny, nx, 0.01),
                                      wall_density=1.0, **kw)
    two = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid,
                                      wall_velocity=_wall_velocity(ny, nx, 0.01),
                                      wall_density=2.0, **kw)
    assert torch.allclose(two - ref, 2.0 * (one - ref), rtol=1e-12, atol=1e-15)


# --------------------------------------------------------------------------- #
# 2. input validation
# --------------------------------------------------------------------------- #
def test_wall_velocity_requires_weights():
    sdf, _, solid, _ = _channel()
    ny, nx = sdf.shape
    f_new, f_pre = _state(ny, nx)
    with pytest.raises(ValueError, match="weights"):
        interpolated_bounce_back(f_new, f_pre, sdf, solid, C9, OPP9,
                                 wall_velocity=_wall_velocity(ny, nx, 0.01))


def test_wall_velocity_shape_is_validated():
    sdf, _, solid, _ = _channel()
    ny, nx = sdf.shape
    f_new, f_pre = _state(ny, nx)
    with pytest.raises(ValueError, match="broadcast"):
        interpolated_bounce_back_2d(f_new, f_pre, sdf, solid,
                                    wall_velocity=torch.zeros((ny, nx + 2, 2), dtype=DT))
    with pytest.raises(ValueError, match="finite"):
        bad = _wall_velocity(ny, nx, float("nan"))
        interpolated_bounce_back_2d(f_new, f_pre, sdf, solid, wall_velocity=bad)


# --------------------------------------------------------------------------- #
# 3. 3D wrapper: moving wall collapses onto the halfway closure at q = 1/2
# --------------------------------------------------------------------------- #
def test_3d_moving_wall_runs_and_collapses_at_half():
    nz, ny, nx = 3, 6, 4
    sdf = torch.full((nz, ny, nx), 0.5, dtype=DT)
    fluid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    fluid[:, 1:-1, :] = True
    sdf[:, 0, :] = -0.5
    sdf[:, -1, :] = -0.5
    solid = ~fluid
    g = torch.Generator().manual_seed(11)
    f_pre = 1e-3 * torch.rand(19, nz, ny, nx, generator=g, dtype=DT) + W19.view(19, 1, 1, 1)
    f_new = torch.roll(f_pre, shifts=1, dims=1)
    wv = torch.zeros((nz, ny, nx, 3), dtype=DT)
    wv[:, 1, :, 0] = 0.02
    out = interpolated_bounce_back_3d(f_new, f_pre, sdf, solid, alpha=alpha_bfl, wall_velocity=wv)
    assert out.shape == f_new.shape
    ref = interpolated_bounce_back_3d(f_new, f_pre, sdf, solid, alpha=alpha_bfl)
    # q = 1/2 for every cut link, so only the Dirichlet term can differ
    assert not torch.equal(out, ref)
    assert float((out - ref).abs().max()) < 0.01


# --------------------------------------------------------------------------- #
# 4. end-to-end: plane Couette with sub-grid walls
# --------------------------------------------------------------------------- #
def _couette_l2(*, delta: float, with_term: bool, carrier: str = "bgk", steps: int = 800,
                n_fluid: int = 8, u_wall: float = 0.02, nu: float = 0.1) -> float:
    """Small torch Couette solver driven through the production operator."""
    q = int(C9.shape[0])
    cx = C9[:, 0].to(DT).view(q, 1, 1)
    cy = C9[:, 1].to(DT).view(q, 1, 1)
    w = W9.to(DT).view(q, 1, 1)
    n_wall, nx = 2, 3
    ny = n_fluid + 2 * n_wall
    H = (n_fluid - 1) + 2 * delta
    y = torch.arange(ny, dtype=DT) - n_wall + delta
    sdf = torch.minimum(y, H - y).view(ny, 1).expand(ny, nx).contiguous()
    fluid1d = torch.zeros(ny, dtype=torch.bool)
    fluid1d[n_wall : n_wall + n_fluid] = True
    fluid = fluid1d.view(ny, 1).expand(ny, nx).contiguous()
    solid = ~fluid
    yj = y[n_wall : n_wall + n_fluid]
    u_ref = u_wall * (1.0 - yj / H)

    def eq(rho, ux, uy):
        cu = cx * ux + cy * uy
        return w * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * (ux * ux + uy * uy))

    u0 = torch.zeros((ny, nx), dtype=DT)
    u0[n_wall : n_wall + n_fluid, :] = u_ref.unsqueeze(1)
    zero = torch.zeros_like(u0)
    f = torch.where(fluid.unsqueeze(0), eq(torch.ones_like(u0), u0, zero), w)

    wv = None
    if with_term:
        wv = torch.zeros((ny, nx, 2), dtype=DT)
        wv[n_wall, :, 0] = u_wall
    tau = 0.5 + 3.0 * nu
    for _ in range(steps):
        rho = f.sum(0)
        mx = (cx * f).sum(0)
        my = (cy * f).sum(0)
        feq = eq(rho, mx / rho, my / rho)
        f_post = torch.where(fluid.unsqueeze(0), f - (1.0 / tau) * (f - feq), f)
        f_st = torch.empty_like(f_post)
        for i in range(q):
            f_st[i] = torch.roll(f_post[i], shifts=(int(C9[i, 1]), int(C9[i, 0])), dims=(0, 1))
        f = interpolated_bounce_back_2d(f_st, f_post, sdf, solid, alpha=alpha_bfl,
                                        wall_velocity=wv)
    prof = ((cx * f).sum(0) / f.sum(0))[n_wall : n_wall + n_fluid, 0]
    return float(torch.linalg.norm(prof - u_ref) / torch.linalg.norm(u_ref))


@pytest.mark.parametrize("delta", [0.1, 0.25, 0.5])
def test_couette_needs_the_dirichlet_term(delta):
    """Without the term the sliding wall is seen as at rest (100 % error)."""
    l2_term = _couette_l2(delta=delta, with_term=True)
    l2_none = _couette_l2(delta=delta, with_term=False)
    assert l2_term < 1e-3, (delta, l2_term)
    assert l2_none > 0.5, (delta, l2_none)
    assert l2_none / l2_term > 50.0


def test_couette_error_is_closure_independent_with_the_term():
    """The residual is a mass drift, not a wall-position error: it must not scale with delta."""
    vals = [_couette_l2(delta=d, with_term=True, steps=600) for d in (0.1, 0.5)]
    assert max(vals) < 1e-3
    assert max(vals) / min(vals) < 10.0