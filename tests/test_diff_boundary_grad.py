"""Differentiable boundary closures: exact anchors and AD-vs-FD gradient checks.

Covers ``tensorlbm.diff_boundary_grad`` (GAP-DIFFLBM-BOUNDARY-GRAD):

* plane Couette and plane Poiseuille profiles against the exact solutions
  (these pin the *value* of the momentum-exchange force probe, which an
  AD-vs-FD agreement can never pin on its own);
* reverse-mode AD versus central differences of (drag, lift) for the three
  closures -- static half-way bounce-back, moving (Couette) wall, volume
  penalisation solid -- on a small channel with an off-centre cylinder.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.diff_boundary_grad import DiffChannel2D


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(2)
    yield


def test_plane_couette_matches_exact_solution():
    ny, nx, uw, tau = 18, 4, 0.05, 0.8
    ch = DiffChannel2D(ny, nx, tau=tau, periodic_x=True)
    geom = ch.geometry(-500.0, -500.0, "bb_moving")      # walls only, no body
    f = ch.initial(0.0, 0.0)
    for _ in range(12000):
        f, femit, rho = ch.step(f, geom, u0=0.0, alpha=0.0, closure="bb_moving", uw=(uw, 0.0))
    _rho, ux, _uy = ch.velocity_profile(f)
    y = torch.arange(ny, dtype=torch.float64)
    exact = uw * (ny - 1.5 - y) / (ny - 2.0)
    assert float((ux[1:-1] - exact[1:-1, None]).abs().max() / uw) < 2e-3

    nu = (tau - 0.5) / 3.0
    f_bot, _ = ch.bb_force(femit, rho, geom, uw=(uw, 0.0), links9=geom.bot9.to(torch.float64))
    f_top, _ = ch.bb_force(femit, rho, geom, uw=(uw, 0.0), links9=geom.top9.to(torch.float64))
    f_exact = nu * uw / (ny - 2.0) * nx
    assert abs(abs(float(f_bot)) - f_exact) / f_exact < 5e-3
    # the fluid resists the moving wall and pulls the fixed wall along with it
    assert float(f_bot) < 0.0 < float(f_top)


def test_plane_poiseuille_matches_exact_solution():
    ny, nx, nu, force = 18, 4, 0.1, 2e-4
    ch = DiffChannel2D(ny, nx, tau=3 * nu + 0.5, periodic_x=True)
    geom = ch.geometry(-500.0, -500.0, "bb_static")
    f = ch.initial(0.0, 0.0)
    for _ in range(12000):
        f, femit, rho = ch.step(f, geom, u0=0.0, alpha=0.0, closure="bb_static", body_force=force)
    _rho, ux, _uy = ch.velocity_profile(f)
    y = torch.arange(ny, dtype=torch.float64)
    h = ny - 2.0
    exact = (force / (2 * nu)) * (y - 0.5) * (h - (y - 0.5))
    assert float((ux[1:-1] - exact[1:-1, None]).abs().max() / exact.max()) < 5e-3

    f_bot, _ = ch.bb_force(femit, rho, geom, uw=(0.0, 0.0), links9=geom.bot9.to(torch.float64))
    f_top, _ = ch.bb_force(femit, rho, geom, uw=(0.0, 0.0), links9=geom.top9.to(torch.float64))
    f_exact = force * h / 2.0 * nx
    assert abs(float(f_bot) - f_exact) / f_exact < 5e-3
    assert abs(float(f_top) - f_exact) / f_exact < 5e-3
    assert float(f_bot) > 0.0 and float(f_top) > 0.0


STEPS = 120


def _base():
    return {"xc": 30.0, "yc": 12.0, "u0": 0.2, "alpha": 0.06}


def _probe(ch, closure, theta, base, steps=STEPS):
    th = torch.tensor(float(theta), dtype=torch.float64, requires_grad=True)
    if closure == "penalization":
        drag, lift = ch.probe(xc=base["xc"], yc=th, u0=base["u0"], alpha=base["alpha"],
                              closure=closure, steps=steps, checkpoint_segment=40)
    elif closure == "bb_moving":
        drag, lift = ch.probe(xc=base["xc"], yc=base["yc"], u0=base["u0"], alpha=base["alpha"],
                              closure=closure, uw=(th, 0.0), steps=steps, checkpoint_segment=40)
    else:
        drag, lift = ch.probe(xc=base["xc"], yc=base["yc"], u0=base["u0"], alpha=th,
                              closure=closure, steps=steps, checkpoint_segment=40)
    (g_drag,) = torch.autograd.grad(drag, th, retain_graph=True)
    (g_lift,) = torch.autograd.grad(lift, th, retain_graph=False)
    return float(drag), float(lift), float(g_drag), float(g_lift)


def _fd(ch, closure, theta, base, h, steps=STEPS):
    def run(val):
        if closure == "penalization":
            d, l = ch.probe(xc=base["xc"], yc=val, u0=base["u0"], alpha=base["alpha"],
                            closure=closure, steps=steps)
        elif closure == "bb_moving":
            d, l = ch.probe(xc=base["xc"], yc=base["yc"], u0=base["u0"], alpha=base["alpha"],
                            closure=closure, uw=(val, 0.0), steps=steps)
        else:
            d, l = ch.probe(xc=base["xc"], yc=base["yc"], u0=base["u0"], alpha=val,
                            closure=closure, steps=steps)
        return float(d), float(l)

    dp, lp = run(theta + h)
    dm, lm = run(theta - h)
    return (dp - dm) / (2 * h), (lp - lm) / (2 * h)


@pytest.mark.parametrize(
    "closure,theta0,h",
    [("bb_static", 0.06, 1e-3), ("bb_moving", 0.06, 1e-3), ("penalization", 12.3, 1e-2)],
)
def test_ad_matches_central_difference(closure, theta0, h):
    ch = DiffChannel2D(34, 96, tau=0.65, radius=5.0)
    base = _base()
    drag, lift, g_drag, g_lift = _probe(ch, closure, theta0, base)
    fd_drag, fd_lift = _fd(ch, closure, theta0, base, h)
    assert math.isfinite(drag) and math.isfinite(lift)
    assert abs(g_drag - fd_drag) / max(abs(fd_drag), 1e-30) <= 1e-2
    assert abs(g_lift - fd_lift) / max(abs(fd_lift), 1e-30) <= 1e-2
    # both sensitivities must be non-degenerate (a zero-vs-zero comparison would
    # be vacuous), and the lift gradient must not be a copy of the drag one
    assert abs(fd_drag) > 1e-3 and abs(fd_lift) > 1e-3
    assert abs(g_drag - g_lift) > 1e-3


def test_penalization_geometry_gradient_is_not_detached():
    """The penalised body must stay in the graph: d(drag)/d(y_c) != 0."""
    ch = DiffChannel2D(34, 96, tau=0.65, radius=5.0)
    th = torch.tensor(12.3, dtype=torch.float64, requires_grad=True)
    drag, lift = ch.probe(xc=30.0, yc=th, u0=0.2, alpha=0.0, closure="penalization",
                          steps=60, checkpoint_segment=30)
    (g,) = torch.autograd.grad(drag, th)
    assert torch.isfinite(g) and abs(float(g)) > 1e-6
    assert abs(float(lift)) < 1e3


def test_closure_values_agree_in_magnitude():
    """Static bounce-back and volume penalisation see the same body at the same
    place: the two force probes must agree in magnitude (order of magnitude,
    same sign).  They are *not* expected to agree tightly at D = 10 cells --
    the diffuse penalisation interface (smoothing ~ 1 cell) is effectively
    larger than the staircase bounce-back body, which raises the drag
    monotonically with ``smoothing`` (documented in the GAP report)."""
    ch = DiffChannel2D(40, 120, tau=0.65, radius=5.0)
    kw = dict(xc=36.0, yc=15.0, u0=0.2, alpha=0.0, steps=600)
    d_bb, l_bb = ch.probe(closure="bb_static", **kw)
    d_pen, l_pen = ch.probe(closure="penalization", **kw)
    ratio = float(d_pen) / float(d_bb)
    assert 0.5 < ratio < 2.0, ratio
    assert float(d_bb) > 0.0 and float(d_pen) > 0.0