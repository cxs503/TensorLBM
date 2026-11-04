"""Planar-wall attribution gate for the momentum-exchange force estimators.

Debt being paid (mission 3596 / report #3740, section ⑤):

    On a *planar* channel wall ``momentum_exchange_standard`` returned
    exactly 1.833x the momentum-balanced value, and
    ``momentum_exchange_background_subtracted`` returned the *identical*
    number although its docstring claims the free-stream background
    "cancels across opposite-direction pairs" on flat walls.  ``wet_node``
    matched to 0.0065 %.  The attribution was left *unverified* ("we only
    record the hypothesis and do not touch library code").

The tests below bind the measured attribution, so it can no longer drift:

  R1  referee: in a plane channel forced by a body force g_x, steady state
      requires  F_wall = rho * g_x * V_fluid / 2  per wall, exactly, with no
      assumption on resolution, collision model or viscosity.
  R2  mechanism: ``standard`` = A + B where
        A = sum_links c_qx * f_q(x_f)      (post-stream value at the near
                                           fluid cell == post-collision value
                                           of the node ONE CELL UPSTREAM)
        B = sum_links c_qx * f_opp(x_s)    (population stored at the solid
                                           node, stale: NoDynamics restore)
      On a planar wall B == 0 *exactly* (the stale rest populations cancel
      over the +-c_x crossing pair), so 100 % of the deviation is the
      one-cell-off-wall sampling in A.
  R3  why ``bgsub`` == ``standard``: the term bgsub subtracts has c_qx = 0
      and cancels over the +-c_x pair, i.e. its streamwise projection is
      identically zero -- for *any* field and *any* u0.  The subtraction is
      wall-NORMAL only, so it can never remove the tangential bias.  The
      docstring's flat-wall claim is wrong for a different reason than the
      one it states.

Numbers measured by ``artifacts/mem_planar/run_study.py`` (ny=32, tau=0.8,
g=1e-5): standard/bgsub = 2.0818 x momentum balance, wet_node = 1.0103 x.
Across tau in {0.6, 0.8, 1.0, 1.5} the standard ratio reads
5.30 / 2.08 / 1.44 / 0.95 -- a *viscosity-dependent* bias that happens to
look harmless at tau = 1.5.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import C, OPPOSITE, W, equilibrium3d
from tensorlbm.drag_pressure import get_near_wall_3d
from tensorlbm.momentum_exchange import (
    momentum_exchange_background_subtracted,
    momentum_exchange_pair,
    momentum_exchange_standard,
    momentum_exchange_wet_node,
)
from tensorlbm.solver3d import collide_mrt3d_low_memory, stream3d

DEV = "cpu"
NY = 32
TAU = 0.8
G = 1.0e-5
NX = 2
NZ = 2


# ---------------------------------------------------------------------------
# Plane channel: solid y = 0 and y = NY-1, periodic x/z, body force g_x
# ---------------------------------------------------------------------------
def _channel(nz=NZ, ny=NY, nx=NX, device=DEV):
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, ny - 1, :] = True
    return solid


def _parabola(ny, g, nu, device):
    y = torch.arange(ny, device=device, dtype=torch.float64)
    return (g / (2.0 * nu) * (y - 0.5) * ((ny - 1.5) - y)).float()


@pytest.fixture(scope="module")
def channel():
    """Relaxed forced plane channel + the two single-wall masks."""
    dev = torch.device(DEV)
    shape = (NZ, NY, NX)
    solid = _channel()
    fluid = ~solid
    n_fluid = int(fluid.sum().item())
    nu = (TAU - 0.5) / 3.0

    # one-wall masks: the two walls carry opposite tangential force, so an
    # all-wall mask cancels it; the flow keeps both walls.
    solid_bot = solid & (torch.arange(NY, device=dev).view(1, NY, 1) == 0)
    near_bot = get_near_wall_3d(solid_bot)

    rho = torch.ones(shape, device=dev)
    zero = torch.zeros_like(rho)
    ux = _parabola(NY, G, nu, dev).view(1, NY, 1).expand(shape).contiguous()
    ux = torch.where(fluid, ux, torch.zeros_like(ux))
    f = equilibrium3d(rho, ux, zero, zero)

    w = W.to(dev).float()
    c = C.to(dev).float()
    force_term = 3.0 * w.view(19, 1, 1, 1) * c[:, 0].view(19, 1, 1, 1) * G

    prev = None
    steps = 0
    for it in range(1, 8001):
        f_solid = f[:, solid].clone()
        f = collide_mrt3d_low_memory(f, tau=TAU)
        f[:, solid] = f_solid
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        f = f + force_term
        steps = it
        if it % 500 == 0:
            fxw = momentum_exchange_wet_node(f, solid_bot, near_bot)[0]
            if prev is not None and abs(fxw - prev) <= 1e-5 * abs(fxw):
                break
            prev = fxw

    ux_f = (f * c[:, 0].view(19, 1, 1, 1)).sum(0) / f.sum(0)
    return {
        "f": f,
        "solid": solid,
        "solid_bot": solid_bot,
        "near_bot": near_bot,
        "n_fluid": n_fluid,
        "steps": steps,
        "nu": nu,
        "F_balance_one_wall": 0.5 * G * n_fluid,
        "ux": ux_f[0, :, 0],
        "ux_analytic": _parabola(NY, G, nu, dev),
    }


def _decompose(f, solid_bot, near_bot):
    """Return (A, B, wet_direct, bg_x, bg_y) for one wall."""
    c = C.to(f.device).float()
    opp = OPPOSITE.to(f.device)
    A = B = wet = bg_x = bg_y = 0.0
    for i in range(1, 19):
        di, dj, dk = (int(c[i][a].item()) for a in range(3))
        roll = lambda t: torch.roll(t, (-dk, -dj, -di), dims=(0, 1, 2))  # noqa: E731
        crossing = near_bot & roll(solid_bot)
        if not crossing.any():
            continue
        cf = crossing.float()
        A += float(c[i][0].item()) * float((f[i] * cf).sum().item())
        B += float(c[i][0].item()) * float((roll(f[int(opp[i].item())]) * cf).sum().item())
        full = (~solid_bot) & roll(solid_bot)
        wet += 2.0 * float(c[i][0].item()) * float((roll(f[i]) * full.float()).sum().item())
        bg = 2.0 * float(crossing.sum().item()) * float(W[i].item())  # u0 = 0
        bg_x += float(c[i][0].item()) * bg
        bg_y += float(c[i][1].item()) * bg
    return A, B, wet, bg_x, bg_y


# ---------------------------------------------------------------------------
# 1. the simulation itself is physical (otherwise nothing below means anything)
# ---------------------------------------------------------------------------
def test_field_matches_analytic_poiseuille(channel):
    u = channel["ux"]
    ua = channel["ux_analytic"]
    err = abs(u[NY // 2] - ua[NY // 2]) / abs(ua[NY // 2])
    assert err < 0.01, f"centreline error {err:.3%} -- field is not steady Poiseuille"
    # halfway bounce-back: the wall sits at y = 1/2, so u(y=0) is inside the wall
    assert abs(u[1] - channel["ux_analytic"][1]) / abs(channel["ux_analytic"][1]) < 0.05


# ---------------------------------------------------------------------------
# 2. wet_node is the estimator that satisfies the exact momentum balance
# ---------------------------------------------------------------------------
def test_wet_node_matches_momentum_balance_within_3pct(channel):
    fx = momentum_exchange_wet_node(channel["f"], channel["solid_bot"], channel["near_bot"])[0]
    ratio = fx / channel["F_balance_one_wall"]
    assert abs(ratio - 1.0) < 0.03, f"wet_node / momentum balance = {ratio:.4f} (measured 1.0103)"


def test_pair_matches_momentum_balance_within_2pct(channel):
    fx = momentum_exchange_pair(channel["f"], channel["solid_bot"], channel["near_bot"])[0]
    ratio = fx / channel["F_balance_one_wall"]
    assert abs(ratio - 1.0) < 0.02, f"pair / momentum balance = {ratio:.4f} (measured 0.9993)"


# ---------------------------------------------------------------------------
# 3. the audit finding, now bound: bgsub == standard on a planar wall
# ---------------------------------------------------------------------------
def test_bgsub_equals_standard_tangentially(channel):
    f, s, n = channel["f"], channel["solid_bot"], channel["near_bot"]
    fx_std = momentum_exchange_standard(f, s, n)[0]
    fx_bg = momentum_exchange_background_subtracted(f, s, n, rho0=1.0, u0=(0.0, 0.0, 0.0))[0]
    assert abs(fx_bg - fx_std) / abs(fx_std) < 1e-4, (
        "bgsub is supposed to be a tangential no-op on planar walls; if this "
        "fails the estimator changed and docs/mem_planar_wall_attribution.md "
        "must be re-derived"
    )


def test_standard_tangential_bias_is_systematic(channel):
    fx = momentum_exchange_standard(channel["f"], channel["solid_bot"], channel["near_bot"])[0]
    ratio = fx / channel["F_balance_one_wall"]
    # >15 % is the repo's unacceptable-error line; measured 2.0818 at nu=0.1.
    assert 1.9 < ratio < 2.3, (
        f"standard / momentum balance = {ratio:.4f}; the documented planar-wall "
        "bias (~2.08 at nu=0.1) no longer reproduces -- re-derive before "
        "changing the attribution"
    )


# ---------------------------------------------------------------------------
# 4. mechanism: stale solid-node population contributes exactly zero
# ---------------------------------------------------------------------------
def test_deviation_is_entirely_the_upstream_sampling(channel):
    f, s, n = channel["f"], channel["solid_bot"], channel["near_bot"]
    A, B, wet, _, _ = _decompose(f, s, n)
    fx_std = momentum_exchange_standard(f, s, n)[0]
    fx_wet = momentum_exchange_wet_node(f, s, n)[0]

    # identity standard == A + B (up to fp32 sum order)
    assert abs((A + B) - fx_std) / abs(fx_std) < 1e-4
    # identity wet_node == its own per-direction budget
    assert abs(wet - fx_wet) / abs(fx_wet) < 1e-6
    # the stale population dropped on the solid nodes cancels EXACTLY in x
    assert abs(B) / abs(A) < 1e-6, f"|B|/|A| = {abs(B)/abs(A):.2e}; expected ~0"
    # therefore A alone carries the whole bias
    assert abs(A - fx_std) / abs(fx_std) < 1e-4


def test_bg_term_has_zero_streamwise_projection(channel):
    f, s, n = channel["f"], channel["solid_bot"], channel["near_bot"]
    A, _, _, bg_x, bg_y = _decompose(f, s, n)
    # the subtracted background is purely wall-normal -- exactly zero in x,
    # large in y: this is why no tangential bias can ever be removed by it
    assert abs(bg_x) / abs(A) < 1e-9, f"bg streamwise projection {bg_x:.3e} should be 0"
    assert abs(bg_y) / abs(A) > 0.1, f"bg normal projection {bg_y:.3e} should be O(link count)"


# ---------------------------------------------------------------------------
# 5. the same identities hold for an arbitrary field (pure algebra, no run)
# ---------------------------------------------------------------------------
def test_identities_hold_for_arbitrary_field():
    """bgsub - standard has zero x-projection on ANY field: the flat-wall
    cancellation is a property of the crossing-set geometry, not of the flow."""
    dev = torch.device(DEV)
    nz, ny, nx = 2, 8, 4
    torch.manual_seed(7)
    f = torch.rand((19, nz, ny, nx), device=dev) * 0.1 + 0.01
    solid_bot = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    solid_bot[:, 0, :] = True
    near_bot = get_near_wall_3d(solid_bot)

    fx_std, fy_std, _ = momentum_exchange_standard(f, solid_bot, near_bot)
    fx_bg, fy_bg, _ = momentum_exchange_background_subtracted(
        f, solid_bot, near_bot, rho0=1.0, u0=(0.037, 0.0, 0.0)
    )
    scale = max(abs(fx_std), 1e-12)
    assert abs(fx_bg - fx_std) / scale < 1e-6  # exact in x, for any u0
    assert abs(fy_bg - fy_std) > 0.0  # ... but not in y

    A, B, _, _, _ = _decompose(f, solid_bot, near_bot)
    assert math.isfinite(A + B)
    assert abs((A + B) - fx_std) / scale < 1e-6