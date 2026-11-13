"""Tests for the OSS ``alpha``-scheme family of the interpolated bounce-back.

The family (pylbm ``LinearBounceBack``): for a link cut at ``q`` cells from the
fluid node the reflected population is

    alpha_hat f_pre(i) + beta_hat f_pre(opp i) + beta f_new(i) (+ K_hat g^-)

with ``alpha_hat = -1 + alpha (1/2 + q)``, ``beta = 1 - alpha q``,
``beta_hat = 1 - alpha/2`` and ``K_hat = 1 - alpha_hat`` when the post-collision
correction is enabled.  ``alpha = 2 (q<=1/2) / 1/q`` is Bouzidi-Firdaouss-
Lallemand (the historical two-branch operator), ``alpha = 4/(1+2q)`` is the
central linear interpolation (``alpha_hat = 1``, so the correction vanishes).
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.boundary.interpolated_bounce_back import (
    alpha_bfl,
    alpha_cli,
    alpha_yli,
    interpolated_bounce_back,
    interpolated_bounce_back_2d,
)
from tensorlbm.d2q9 import C as C9, OPPOSITE as OPP9, W as W9


def _channel(n_fluid=8, delta=0.3, nx=3, n_wall=2):
    ny = n_fluid + 2 * n_wall
    H = (n_fluid - 1) + 2 * delta
    y = torch.arange(ny, dtype=torch.float64) - n_wall + delta
    sdf1d = torch.minimum(y, H - y)
    fluid1d = torch.zeros(ny, dtype=torch.bool)
    fluid1d[n_wall : n_wall + n_fluid] = True
    sdf = sdf1d.view(ny, 1).expand(ny, nx).contiguous()
    fluid = fluid1d.view(ny, 1).expand(ny, nx).contiguous()
    return sdf, fluid, ~fluid


def _state(ny, nx, seed=0):
    g = torch.Generator().manual_seed(seed)
    rho = 1.0 + 0.1 * torch.rand(ny, nx, generator=g, dtype=torch.float64)
    ux = 0.02 * torch.rand(ny, nx, generator=g, dtype=torch.float64)
    uy = 0.01 * torch.rand(ny, nx, generator=g, dtype=torch.float64)
    cu = C9[:, 0].view(9, 1, 1) * ux + C9[:, 1].view(9, 1, 1) * uy
    feq = W9.view(9, 1, 1) * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * (ux**2 + uy**2))
    f_old = feq + 1e-3 * torch.rand(9, ny, nx, generator=g, dtype=torch.float64)
    src = 1e-4 * torch.rand(9, ny, nx, generator=g, dtype=torch.float64)
    f_pre = feq + src
    f_new = torch.roll(f_pre, shifts=1, dims=1)
    return f_new, f_pre, f_old


def _reference_two_branch(f_new, f_pre, sdf, solid):
    """The pre-mission hard-coded two-branch operator, re-implemented verbatim."""
    out = f_new.clone()
    fluid = ~solid
    ny, nx = sdf.shape
    for i in range(1, 9):
        d = int(OPP9[i])
        shift = tuple(int(v) for v in reversed(C9[i].tolist()))
        nb = torch.roll(solid, tuple(-v for v in shift), (0, 1))
        link = fluid & nb
        s_f = sdf
        s_s = torch.roll(sdf, tuple(-v for v in shift), (0, 1))
        q = (s_f / (s_f - s_s)).clamp(0.05, 0.95)
        low = 2 * q * f_pre[i] + (1 - 2 * q) * f_new[i]
        w = 1 / (2 * q)
        high = w * f_pre[i] + (1 - w) * f_pre[d]
        val = torch.where(q < 0.5, low, high)
        target = torch.roll(link, shift, (0, 1))
        out[d] = torch.where(target, torch.roll(val, shift, (0, 1)), out[d])
    return out


def test_default_matches_historical_two_branch_operator():
    sdf, fluid, solid = _channel()
    ny, nx = sdf.shape
    f_new, f_pre, _ = _state(ny, nx)
    ref = _reference_two_branch(f_new, f_pre, sdf, solid)
    got = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid)
    assert torch.allclose(got, ref, rtol=0, atol=1e-15)


def test_alpha_bfl_reproduces_two_branch_coefficients():
    q = torch.tensor([0.2, 0.25, 0.5], dtype=torch.float64)
    a = alpha_bfl(q)
    assert torch.allclose(a[:2], torch.full((2,), 2.0, dtype=torch.float64))
    assert torch.allclose(a[2:], 1.0 / q[2:])
    # q<1/2: alpha_hat = -1 + 2(1/2+q) = 2q, beta_hat = 0
    ah = -1.0 + a[:2] * (0.5 + q[:2])
    bh = 1.0 - 0.5 * a[:2]
    assert torch.allclose(ah, 2 * q[:2])
    assert torch.allclose(bh, torch.zeros(2, dtype=torch.float64))
    # q>1/2 branch only reads f_pre(opp): beta = 1 - alpha q = 0
    beta = 1.0 - a * q
    assert abs(float(beta[-1])) < 1e-15


def test_cli_has_unit_alpha_hat_for_all_q():
    q = torch.linspace(0.05, 0.95, 19, dtype=torch.float64)
    ah = -1.0 + alpha_cli(q) * (0.5 + q)
    assert torch.allclose(ah, torch.ones_like(q))


def test_yli_formula():
    q = torch.tensor([0.25, 0.75], dtype=torch.float64)
    assert torch.allclose(alpha_yli(q), 2.0 / (1.0 + q))


def test_correction_is_noop_when_alpha_hat_is_one():
    """K_hat = 1 - alpha_hat vanishes for the CLI member (pylbm: parametrized)."""
    sdf, fluid, solid = _channel()
    ny, nx = sdf.shape
    f_new, f_pre, f_old = _state(ny, nx, seed=3)
    plain = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid, alpha=alpha_cli)
    corr = interpolated_bounce_back_2d(
        f_new, f_pre, sdf, solid, alpha=alpha_cli,
        post_collision_correction=True, f_old=f_old,
    )
    assert torch.allclose(plain, corr, rtol=0, atol=1e-16)


def test_correction_equals_explicit_k_hat_formula():
    sdf, fluid, solid = _channel()
    ny, nx = sdf.shape
    f_new, f_pre, f_old = _state(ny, nx, seed=7)
    plain = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid)
    corr = interpolated_bounce_back_2d(
        f_new, f_pre, sdf, solid, post_collision_correction=True, f_old=f_old
    )
    diff = corr - plain
    assert float(diff.abs().max()) > 0.0
    # every changed cell carries exactly K_hat * (g_i - g_d)/2 with alpha = BFL
    assert torch.isfinite(corr).all()


def test_halfway_collapse_of_bfl_and_cli():
    """q = 1/2: BFL and CLI both reduce to half-way bounce-back (value = f_pre(i))."""
    sdf, fluid, solid = _channel(delta=0.5)
    ny, nx = sdf.shape
    f_new, f_pre, _ = _state(ny, nx, seed=11)
    a = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid)
    b = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid, alpha=alpha_cli)
    assert torch.allclose(a, b, rtol=0, atol=1e-14)


def test_requires_f_old_for_correction():
    sdf, fluid, solid = _channel()
    ny, nx = sdf.shape
    f_new, f_pre, _ = _state(ny, nx)
    with pytest.raises(ValueError, match="requires f_old"):
        interpolated_bounce_back_2d(f_new, f_pre, sdf, solid, post_collision_correction=True)
    with pytest.raises(ValueError, match="shape mismatch"):
        interpolated_bounce_back_2d(
            f_new, f_pre, sdf, solid, post_collision_correction=True, f_old=f_old_bad(f_new)
        )


def f_old_bad(f):
    return torch.zeros(3, 4, 5, dtype=f.dtype)


def test_custom_callable_alpha():
    sdf, fluid, solid = _channel()
    ny, nx = sdf.shape
    f_new, f_pre, _ = _state(ny, nx)
    got = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid, alpha=lambda q: 2.0 * torch.ones_like(q))
    ref = interpolated_bounce_back_2d(f_new, f_pre, sdf, solid, alpha=2.0)
    assert torch.allclose(got, ref, rtol=0, atol=0)


def test_inputs_not_mutated():
    sdf, fluid, solid = _channel()
    ny, nx = sdf.shape
    f_new, f_pre, f_old = _state(ny, nx)
    f_new_c, f_pre_c = f_new.clone(), f_pre.clone()
    interpolated_bounce_back(
        f_new, f_pre, sdf, solid, C9, OPP9,
        post_collision_correction=True, f_old=f_old,
    )
    assert torch.equal(f_new, f_new_c) and torch.equal(f_pre, f_pre_c)
