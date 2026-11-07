"""Verification cases: normal-shock Mach scan, double Mach reflection, step.

Each builder returns a configured solver plus a metadata dict, so a run can
persist the exact configuration next to the numbers (no hidden defaults).

Case conventions follow the standard literature definitions so the results are
comparable with published data:

``normal_shock_case``
    Two variants.  ``riemann`` is the non-circular one: the driver state is
    *inverted* from the shock-tube relations for a target shock Mach number, so
    the shock Mach number is a solver output.  ``jump`` prescribes the exact
    Rankine-Hugoniot jump and checks that the scheme sustains it.
``double_mach_reflection_case``
    Woodward & Colella (1984) Problem 2.  A Mach-10 shock inclined 30 deg to the
    wall runs into the wall at ``x = 1/6``; the state behind it is the exact
    oblique-shock state ``(rho, p, u, v) = (8, 116.5, 7.1447, -4.125)`` ahead of
    which the air ``(rho, p) = (1.4, 1)`` is at rest.
``forward_facing_step_case``
    Woodward & Colella (1984) Problem 1: Mach-3 flow over a step of height 0.2
    starting at ``x = 0.6`` in a unit-height channel, run to a steady state.
"""

from __future__ import annotations

import numpy as np

from .flux import GAMMA, conserved
from .rankine_hugoniot import moving_shock, normal_shock, shock_tube_driver, sound_speed
from .solver import EulerSolver1D, EulerSolver2D

__all__ = [
    "normal_shock_case",
    "oblique_shock_state",
    "double_mach_reflection_case",
    "forward_facing_step_case",
]


# ------------------------------------------------------ 1-D normal shock scan
def normal_shock_case(
    mach: float,
    n: int = 400,
    variant: str = "riemann",
    rho1: float = 1.0,
    p1: float = 1.0,
    gamma: float = GAMMA,
    cfl: float = 0.4,
    flux: str = "hllc",
    x_diaphragm: float | None = None,
    travel: float | None = None,
):
    """1-D normal shock at target shock Mach number ``mach``.

    Returns ``(solver, meta)``.  ``meta`` carries the exact reference states and
    the final time ``t_end`` (chosen so the shock crosses ``travel`` and stays
    clear of both boundaries).
    """
    ms = moving_shock(mach, rho1=rho1, p1=p1, gamma=gamma)
    pre = conserved(rho1, 0.0, 0.0, p1, gamma)
    solver = EulerSolver1D(n=n, length=1.0, gamma=gamma, flux=flux, cfl=cfl)

    if variant == "jump":
        post = conserved(ms["rho2"], ms["u2"], 0.0, ms["p2"], gamma)
        left, right = post, pre
        solver.bc_left = ("uniform", post)  # infinite piston: exact post-shock state
        solver.bc_right = ("transmissive",)
        x_diaphragm = 0.2 if x_diaphragm is None else x_diaphragm
        travel = 0.6 if travel is None else travel
        meta = {"variant": "jump", "driver": None}
    elif variant == "riemann":
        st = shock_tube_driver(mach, rho1=rho1, p1=p1, gamma=gamma)
        left = conserved(st["rho4"], 0.0, 0.0, st["p4"], gamma)
        right = pre
        solver.bc_left = ("transmissive",)
        solver.bc_right = ("transmissive",)
        x_diaphragm = 0.5 if x_diaphragm is None else x_diaphragm
        travel = 0.3 if travel is None else travel
        meta = {
            "variant": "riemann",
            "driver": {"rho4": st["rho4"], "p4": st["p4"], "a4_over_a1": st["a4"] / st["a1"]},
            "contact_state": {"rho3": st["rho3"], "u3": st["u3"], "p3": st["p3"]},
        }
    else:
        raise ValueError(f"unknown variant {variant!r}")

    solver.U = np.where((solver.xc < x_diaphragm)[:, None], left, right)
    solver.t = 0.0
    t_end = travel / ms["W"]
    meta.update(
        {
            "mach": mach,
            "n": n,
            "dx": solver.dx,
            "cfl": cfl,
            "flux": flux,
            "gamma": gamma,
            "x_diaphragm": x_diaphragm,
            "travel": travel,
            "t_end": t_end,
            "shock_speed_expected": ms["W"],
            "post_state": {"rho": ms["rho2"], "u": ms["u2"], "p": ms["p2"]},
            "pre_state": {"rho": rho1, "u": 0.0, "p": p1},
            "reference": ms,
        }
    )
    return solver, meta


# --------------------------------------------------------- 2-D oblique shock
def oblique_shock_state(
    mach: float = 10.0,
    angle_deg: float = 30.0,
    rho1: float = 1.4,
    p1: float = 1.0,
    gamma: float = GAMMA,
    sign_v: float = -1.0,
):
    """Exact state behind a Mach-``mach`` shock inclined ``angle_deg`` to the wall.

    ``angle_deg`` is the angle between the shock *normal* and the wall, so the
    normal-shock relations apply at the full shock Mach number (the
    Woodward-Colella convention).  Returns ``(rho2, u2, v2, p2, W, n, tan_theta)``
    with ``n`` the downstream-pointing shock normal.
    """
    sound_speed(rho1, p1, gamma)  # consistency check of the EOS inputs
    ms = moving_shock(mach, rho1=rho1, p1=p1, gamma=gamma)
    th = np.deg2rad(angle_deg)
    nx, ny = np.cos(th), sign_v * np.sin(th)
    return ms["rho2"], ms["u2"] * nx, ms["u2"] * ny, ms["p2"], ms["W"], (nx, ny), np.tan(th)


def double_mach_reflection_case(
    nx: int = 240,
    ny: int = 60,
    lx: float = 4.0,
    ly: float = 1.0,
    mach: float = 10.0,
    angle_deg: float = 30.0,
    rho1: float = 1.4,
    p1: float = 1.0,
    gamma: float = GAMMA,
    cfl: float = 0.4,
    flux: str = "hllc",
    t_end: float = 0.2,
):
    """Woodward-Colella (1984) double Mach reflection (Problem 2)."""
    rho2, u2, v2, p2, W, n, tan_th = oblique_shock_state(mach, angle_deg, rho1, p1, gamma)
    post = conserved(rho2, u2, v2, p2, gamma)
    pre = conserved(rho1, 0.0, 0.0, p1, gamma)
    solver = EulerSolver2D(nx=nx, ny=ny, lx=lx, ly=ly, gamma=gamma, flux=flux, cfl=cfl)
    X, Y = np.meshgrid(solver.xc, solver.yc)
    x_wall = lx / 24.0
    th = np.deg2rad(angle_deg)

    def x_front(y, t):
        """x of the shock line at height ``y`` and time ``t``."""
        return x_wall + W * t / np.cos(th) + tan_th * np.asarray(y)

    solver.U = np.ascontiguousarray(np.where((X < x_front(Y, 0.0))[:, :, None], post, pre))

    solver.bc_x0 = ("transmissive",)
    solver.bc_x1 = ("transmissive",)
    ghost_xs = lambda ncols: np.linspace(  # noqa: E731
        solver.x0 - 1.5 * solver.dx, solver.x0 + lx + 1.5 * solver.dx, ncols
    )

    def bottom(t, edge, axis):
        """x >= x_wall: reflecting wall; x < x_wall: exact post-shock inflow."""
        xs = ghost_xs(edge.shape[0])
        mirrored = edge.copy()
        mirrored[:, 2] *= -1.0
        wall = np.broadcast_to(mirrored[None, ...], (2,) + mirrored.shape)
        return np.where((xs >= x_wall)[None, :, None], wall, post[None, None, :])

    def top(t, edge, axis):
        """Exact moving shock imposed along y = ly."""
        xs = ghost_xs(edge.shape[0])
        ahead = xs >= x_front(ly, t)
        return np.where(ahead[None, :, None], pre[None, None, :], post[None, None, :])

    solver.bc_y0 = ("callable", bottom)
    solver.bc_y1 = ("callable", top)

    meta = {
        "case": "double_mach_reflection",
        "nx": nx,
        "ny": ny,
        "lx": lx,
        "ly": ly,
        "dx": solver.dx,
        "dy": solver.dy,
        "mach": mach,
        "angle_deg": angle_deg,
        "gamma": gamma,
        "flux": flux,
        "cfl": cfl,
        "t_end": t_end,
        "x_wall": x_wall,
        "post_shock": {"rho": rho2, "u": u2, "v": v2, "p": p2},
        "pre_shock": {"rho": rho1, "u": 0.0, "v": 0.0, "p": p1},
        "shock_normal_speed": W,
        "foot_speed_along_wall": W / np.cos(th),
        "incident_angle_to_x_deg": 90.0 - angle_deg,
        "reference": "Woodward & Colella 1984, J. Comput. Phys. 54, 115 (Problem 2)",
    }
    return solver, meta


# -------------------------------------------------------- 2-D forward-facing step
def forward_facing_step_case(
    nx: int = 240,
    ny: int = 80,
    lx: float = 3.0,
    ly: float = 1.0,
    mach: float = 3.0,
    step_x: float = 0.6,
    step_h: float = 0.2,
    rho_in: float = 1.4,
    p_in: float = 1.0,
    gamma: float = GAMMA,
    cfl: float = 0.4,
    flux: str = "hllc",
    t_end: float = 4.0,
):
    """Woodward-Colella (1984) forward-facing step (Problem 1) at Mach 3."""
    a_in = sound_speed(rho_in, p_in, gamma)
    u_in = mach * a_in
    inflow = conserved(rho_in, u_in, 0.0, p_in, gamma)
    solver = EulerSolver2D(nx=nx, ny=ny, lx=lx, ly=ly, gamma=gamma, flux=flux, cfl=cfl)
    X, Y = np.meshgrid(solver.xc, solver.yc)
    solver.U = np.broadcast_to(inflow, (ny, nx, 4)).copy()
    solver.solid = (X >= step_x) & (Y <= step_h)
    solver.bc_x0 = ("uniform", inflow)  # supersonic inflow: everything fixed
    solver.bc_x1 = ("transmissive",)  # supersonic outflow
    solver.bc_y0 = ("wall",)
    solver.bc_y1 = ("wall",)
    meta = {
        "case": "forward_facing_step",
        "nx": nx,
        "ny": ny,
        "lx": lx,
        "ly": ly,
        "dx": solver.dx,
        "dy": solver.dy,
        "mach": mach,
        "step_x": step_x,
        "step_h": step_h,
        "gamma": gamma,
        "flux": flux,
        "cfl": cfl,
        "t_end": t_end,
        "inflow": {"rho": rho_in, "u": u_in, "p": p_in, "a": a_in},
        "normal_shock_reference": normal_shock(mach, gamma),
        "reference": "Woodward & Colella 1984, J. Comput. Phys. 54, 115 (Problem 1)",
    }
    return solver, meta
