"""Minimal, verifiable fix + KPI extension applied by mission m18372 (XVAL anchor).

Two changes, both local to the force-driven channel reference used by the
single-component Poiseuille anchor:

1. ``analytic_profile`` is symmetric about the channel mid-plane: the far wall
   is the mirror of the near wall, ``y_far = (ny-1) - wall_origin``, so the
   reference width is ``ny-1-2*wall_origin`` (``= ny-2`` for the halfway
   bounce-back origin ``wall_origin = 0.5``).  The previous form
   ``s*(height+1-wall_origin-s)`` placed the second zero at the solid *node*
   ``y = ny-1`` instead of the bounce-back plane ``y = ny-1-wall_origin``,
   which inflated ``u_max`` by ``((height+1-wall_origin)/(height+...))``
   -> +3.3 % at ``ny=32``; that bias is comparable to a 3 % acceptance
   threshold and therefore not tolerable in a reference.
2. ``evaluate`` additionally fits ``u = A (y-y0)(y0+H-y)`` (linear least
   squares on the quadratic in ``y``) and reports ``nu_eff/nu`` and the fitted
   wall origin, so a viscosity error can be told apart from a wall-origin
   offset.
"""

from __future__ import annotations

import math

import numpy as np

C9 = np.array(
    [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]],
    dtype=np.int64,
)
W9 = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP9 = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7])
CS2 = 1.0 / 3.0
DEF_C = 0.57735026919  # kernel.cpp: velocity clamp (lattice c_s)


def analytic_walls(ny: int, wall_origin: float = 0.5) -> tuple[float, float]:
    """Lower and upper halfway-bounce-back wall planes (mirror symmetric)."""
    return wall_origin, (ny - 1) - wall_origin


def analytic_profile(ny: int, nu: float, force: float, wall_origin: float = 0.5):
    """Plane-Poiseuille profile evaluated at the fluid cell centres.

    ``u(y) = F/(2 nu) (y-y_w)(y_far-y)`` with ``y_w = wall_origin`` and
    ``y_far = (ny-1)-wall_origin``; exact solution of the incompressible
    Navier-Stokes equations for a constant streamwise body force between two
    no-slip planes.
    """
    lower, upper = analytic_walls(ny, wall_origin)
    y = np.arange(1, ny - 1, dtype=np.float64)
    return y, force / (2.0 * nu) * (y - lower) * (upper - y)


def analytic_umax(ny: int, nu: float, force: float, wall_origin: float = 0.5) -> float:
    """Continuum peak (between nodes for even ``ny``): ``F w^2/(8 nu)``."""
    lower, upper = analytic_walls(ny, wall_origin)
    width = upper - lower
    return force * width**2 / (8.0 * nu)


def _feq9_shifted(rho, ux, uy):
    """FluidX3D DDF-shifted equilibrium (kernel.cpp:1036-1059)."""
    usq = ux * ux + uy * uy
    out = np.empty((9,) + rho.shape, dtype=np.float64)
    for i in range(9):
        cu = C9[i, 0] * ux + C9[i, 1] * uy
        out[i] = W9[i] * (rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq) - 1.0)
    return out


def _forcing_terms9(ux, uy, fx, fy):
    """``calculate_forcing_terms`` (kernel.cpp:1124-1135), D2Q9 branch."""
    uF = -CS2 * (ux * fx + uy * fy)
    out = np.empty((9,) + ux.shape, dtype=np.float64)
    out[0] = 9.0 * W9[0] * uF
    for i in range(1, 9):
        cx, cy = C9[i]
        out[i] = 9.0 * W9[i] * ((cx * fx + cy * fy) * (cx * ux + cy * uy + CS2) + uF)
    return out


def _stream_pull_bb9(g, solid):
    """Pull-form streaming with mid-grid bounce-back (Esoteric-Pull equivalent).

    FluidX3D stores each post-collision population at the opposite index
    (kernel.cpp:1367-1373) and never stores into solid cells
    (kernel.cpp:1911-1913), which completes the halfway bounce-back
    implicitly; the explicit pull form below is the standard equivalent.
    """
    out = np.empty_like(g)
    out[0] = g[0]
    for i in range(1, 9):
        cx, cy = C9[i]
        pulled = np.roll(g[i], (cy, cx), axis=(0, 1))
        src_solid = np.roll(solid, (cy, cx), axis=(0, 1))
        out[i] = np.where(src_solid, g[OPP9[i]], pulled)
    out[:, solid] = 0.0
    return out


def oss_step(g, solid, tau, fx, fy, forcing="oss"):
    """One FluidX3D ``stream_collide`` step (D2Q9, SRT, VOLUME_FORCE)."""
    w = 1.0 / tau  # kernel.cpp:1615
    rho = 1.0 + g.sum(axis=0)
    ux = (C9[:, 0, None, None] * g).sum(axis=0) / rho
    uy = (C9[:, 1, None, None] * g).sum(axis=0) / rho

    if forcing == "oss":
        rho2 = 0.5 / rho
        usx = np.clip(ux + fx * rho2, -DEF_C, DEF_C)
        usy = np.clip(uy + fy * rho2, -DEF_C, DEF_C)
        fin = _forcing_terms9(usx, usy, fx, fy) * (1.0 - 0.5 * w)
        feq = _feq9_shifted(rho, usx, usy)
    elif forcing == "repo":
        usx = np.clip(ux, -DEF_C, DEF_C)
        usy = np.clip(uy, -DEF_C, DEF_C)
        fin = _forcing_terms9(usx, usy, fx, fy)
        feq = _feq9_shifted(rho, usx, usy)
    else:
        raise ValueError(f"unknown forcing {forcing!r}")

    post = (1.0 - w) * g + w * feq + fin  # kernel.cpp:1646 (SRT)
    post[:, solid] = 0.0
    return _stream_pull_bb9(post, solid)


def simulate_oss(ny, tau, force, nx=4, steps=200000, tol=1e-9, check_every=500, forcing="oss"):
    nu = (tau - 0.5) / 3.0
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    g = np.zeros((9, ny, nx), dtype=np.float64)  # rest state: feq = w -> g = 0

    u_prev, u, hist = None, None, []
    step_used = steps
    for step in range(1, steps + 1):
        g = oss_step(g, solid, tau, force, 0.0, forcing=forcing)
        if step % check_every == 0:
            rho = 1.0 + g.sum(axis=0)
            u = ((C9[:, 0, None, None] * g).sum(axis=0) / rho).mean(axis=1)
            if u_prev is not None:
                delta = float(np.max(np.abs(u - u_prev)) / max(np.max(np.abs(u)), 1e-30))
                hist.append((step, delta))
                if delta < tol:
                    step_used = step
                    break
            u_prev = u
    rho = 1.0 + g.sum(axis=0)
    ux = (C9[:, 0, None, None] * g).sum(axis=0) / rho
    uy = (C9[:, 1, None, None] * g).sum(axis=0) / rho
    u_stored = ux.mean(axis=1)
    bias = 0.5 * force / rho.mean(axis=1)  # FluidX3D stores the half-force-shifted u
    return {
        "arm": f"AO_{forcing.upper()}",
        "path": "oss_d2q9",
        "forcing": forcing,
        "ny": ny,
        "tau": tau,
        "nu": nu,
        "force": force,
        "y": np.arange(1, ny - 1, dtype=np.float64).tolist(),
        "u_stored": u_stored[1:-1].tolist(),
        "u_phys": (u_stored - bias)[1:-1].tolist(),
        "u_cross_max": float(np.max(np.abs(uy[1:-1, :]))),
        "rho_profile": rho.mean(axis=1)[1:-1].tolist(),
        "steps": step_used,
        "steady_tail": hist[-5:],
    }


def _repo_modules():
    from tensorlbm import channel_kernel_numpy as ck

    return ck


def repo_step_oss_forcing(f, solid, tau, force, *, collision="bgk", magic_lambda=3.0 / 16.0):
    """Repo-path step with the OSS forcing convention (shifted f_eq + (1-1/2tau))."""
    ck = _repo_modules()
    rho, ux, uy, uz = ck.macroscopic(f)
    rho2 = 0.5 / np.where(solid, 1.0, rho)
    usx = ux + force * rho2
    usy, usz = uy, uz
    feq = ck.equilibrium(rho, usx, usy, usz)
    fin = np.empty_like(f)
    uf = usx * force
    for i in range(19):
        cx, cy, cz = ck.C[i]
        c_dot_f = cx * force
        fin[i] = ck.W[i] * (3.0 * (c_dot_f - uf) + 9.0 * (cx * usx + cy * usy + cz * usz) * c_dot_f)

    s_plus = 1.0 / tau
    if collision == "bgk":
        post = f - s_plus * (f - feq) + (1.0 - 0.5 * s_plus) * fin
    elif collision == "trt":
        s_minus = 1.0 / (0.5 + magic_lambda / (tau - 0.5))
        f_plus = 0.5 * (f + f[ck.OPP])
        f_minus = 0.5 * (f - f[ck.OPP])
        eq_plus = 0.5 * (feq + feq[ck.OPP])
        eq_minus = 0.5 * (feq - feq[ck.OPP])
        fin_bar = fin[ck.OPP]
        fin_scaled = (0.5 - 0.25 * s_plus) * (fin + fin_bar) + (0.5 - 0.25 * s_minus) * (
            fin - fin_bar
        )  # kernel.cpp:1644-1652
        post = f - s_plus * (f_plus - eq_plus) - s_minus * (f_minus - eq_minus) + fin_scaled
    else:
        raise ValueError(f"unknown collision {collision!r}")

    post = np.where(solid[None, :, :, :], f, post)
    return ck.bounce_back_walls(ck.stream(post), solid)


def repo_step(f, solid, tau, force, *, forcing="repo", collision="bgk", magic_lambda=3.0 / 16.0):
    """One repo-path step; ``forcing='repo'`` is bit-identical to the repo ``step``."""
    if forcing == "oss":
        return repo_step_oss_forcing(f, solid, tau, force, collision=collision,
                                     magic_lambda=magic_lambda)
    ck = _repo_modules()
    collided = ck.collide(f, tau, collision=collision, magic_lambda=magic_lambda)
    post = np.where(solid[None, :, :, :], f, collided)
    rho, ux, uy, uz = ck.macroscopic(post)
    fx = np.where(solid, 0.0, rho * force)
    post = ck.apply_body_force(post, ux, uy, uz, fx)
    return ck.bounce_back_walls(ck.stream(post), solid)


def simulate_repo(ny, tau, force, *, nx=1, nz=1, collision="bgk", forcing="repo",
                  steps=200000, tol=1e-9, check_every=500):
    ck = _repo_modules()
    nu = (tau - 0.5) / 3.0
    solid = np.zeros((nx, ny, nz), dtype=bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    zeros = np.zeros((nx, ny, nz))
    f = ck.equilibrium(np.ones((nx, ny, nz)), zeros, zeros, zeros)

    u_prev, hist, step_used = None, [], steps
    for step in range(1, steps + 1):
        f = repo_step(f, solid, tau, force, forcing=forcing, collision=collision)
        if step % check_every == 0:
            rho, ux, uy, uz = ck.macroscopic(f)
            u = ux.mean(axis=(0, 2))
            if u_prev is not None:
                delta = float(np.max(np.abs(u - u_prev)) / max(np.max(np.abs(u)), 1e-30))
                hist.append((step, delta))
                if delta < tol:
                    step_used = step
                    break
            u_prev = u
    rho, ux, uy, uz = ck.macroscopic(f)
    prof = ux.mean(axis=(0, 2))
    return {
        "arm": f"AR_{forcing.upper()}" + ("" if collision == "bgk" else f"_{collision.upper()}"),
        "path": "repo_d3q19",
        "forcing": forcing,
        "collision": collision,
        "ny": ny,
        "tau": tau,
        "nu": nu,
        "force": force,
        "y": np.arange(1, ny - 1, dtype=np.float64).tolist(),
        "u_stored": prof[1:-1].tolist(),
        "u_phys": prof[1:-1].tolist(),
        "u_cross_max": float(np.max(np.abs(uy))),
        "rho_profile": rho.mean(axis=(0, 2))[1:-1].tolist(),
        "steps": step_used,
        "steady_tail": hist[-5:],
    }


def fit_parabola(ny: int, u) -> dict:
    """Least-squares fit ``u = A (y-y0)(y0+H-y)`` -> amplitude and wall origin.

    The right-hand side is a quadratic in ``y``, so the fit is a degree-2
    polynomial least-squares solve; ``nu_eff = F/(2A)`` separates a viscosity
    error from a wall-origin offset.
    """
    y = np.arange(1, ny - 1, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    a2, a1, a0 = np.polyfit(y, u, 2)
    if a2 >= 0:
        return {"ok": False}
    amp = -a2
    y_center = a1 / (2.0 * amp)
    height = float(ny - 2)
    return {"ok": True, "amplitude": float(amp), "y_center": float(y_center),
            "y0": float(y_center - height / 2.0)}


def evaluate(record: dict, wall_origin: float = 0.5) -> dict:
    """Score one run against the predeclared metrics."""
    ny, nu, force = record["ny"], record["nu"], record["force"]
    _, u_exact = analytic_profile(ny, nu, force, wall_origin=wall_origin)
    umax_exact_nodes = float(np.max(u_exact))
    u = np.asarray(record["u_phys"], dtype=np.float64)
    umax = float(np.max(u))
    u_mean = float(np.mean(u))
    rho = np.asarray(record["rho_profile"], dtype=np.float64)
    stored = np.asarray(record["u_stored"], dtype=np.float64)
    fit = fit_parabola(ny, u)
    return {
        "rel_err_u_max_pct": 100.0 * abs(umax - umax_exact_nodes) / umax_exact_nodes,
        "rel_err_u_max_vs_continuum_pct": 100.0
        * abs(umax - analytic_umax(ny, nu, force, wall_origin)) / analytic_umax(ny, nu, force, wall_origin),
        "profile_rel_l2_pct": 100.0
        * float(np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact)),
        "u_max": umax,
        "u_max_exact_nodes": umax_exact_nodes,
        "u_mean": u_mean,
        "umax_over_umean": umax / u_mean if u_mean else float("nan"),
        "nu_eff_over_nu": (force / (2.0 * fit["amplitude"]) / nu) if fit.get("ok") else None,
        "y0_fit": fit.get("y0"),
        "rho_deviation_max": float(np.max(np.abs(rho - 1.0))),
        "stored_velocity_bias_pct": 100.0 * abs(float(np.max(stored)) - umax) / umax,
        "steps": record["steps"],
        "Ma": umax / math.sqrt(CS2),
        "Re_H": umax * (ny - 2) / nu,
        "u_cross_max": record["u_cross_max"],
    }