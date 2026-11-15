"""Independent continuum reference for the quasi-steady Reynolds squeeze film.

Why this file exists
--------------------
``tensorlbm.squeeze_film_lbm`` was verified (2026-11-08) against the *leading-order*
Reynolds/Stefan closed form ``F = 8 mu V a^3 / h^3`` and came out **12.2 % - 37.2 %
high** over ``h/a in [0.19, 0.75]``.  That comparison cannot, by itself, decide
whether the lattice solver is wrong: the closed form is only the zeroth order of a
lubrication expansion whose next term is ``O((h/a)^2)``, and the reported window is
exactly where that term is of order unity.  Comparing a solver against an asymptotic
truncation is a **caliber** question, not a verification question.

This module is the independent second path: the *same* physical problem solved as
**2D incompressible Navier-Stokes in primitive variables** - a different
discretisation (staggered MAC finite volume + artificial compressibility, Chorin
1967) that never touches the lattice-Boltzmann formulation and never uses the
Reynolds closed form.

Reference problem (plate-fixed frame, quasi-steady)
---------------------------------------------------
* ``y = 0``  : ``u = 0``, ``v = 0``            (impermeable plate, no slip)
* ``y = h``  : ``u = 0``, ``v = -V``           (approaching plate: uniform blowing)
* ``x = 0``  : ``p = 0``, open (``du/dx = 0``)
* ``x = L``  : ``p = 0``, open (``du/dx = 0``), ``L = 2 a``

which is exactly the reduction ``simulate_squeeze`` implements (uniform zero-momentum
mass source on the plate-fixed top row, halfway bounce-back walls, fixed-density
pressure faces at ``x = 0`` and ``x = nx-1`` with plate span ``nx-1 = 2a``).

Measured quantities and why they are the right ones
---------------------------------------------------
The momentum-exchange load accumulated by halfway bounce-back is the integral of the
**total normal stress** on the plate,

    N = int ( p - 2 mu dv/dy ) dx ,

not of the pressure alone, while ``8 mu V a^3/h^3`` is a *pressure* integral.  Both are
reported here (``normal_load_top`` and ``pressure_load_top``) so the two calibers can
never be confused again.

Usage
-----
    from reference_ns import solve, reynolds_load
    ref = solve(a=16.0, h=6.0, nu=0.1, V=0.005)
    ref["normal_load_top"], ref["pressure_load_top"]

Everything is in lattice units (``nu = 0.1``, ``V = 0.005``,...) so the numbers are
directly comparable with ``simulate_squeeze`` output - no unit conversion anywhere.

References
----------
* Chorin, A. J. (1967) J. Comput. Phys. 2, 12-26 (artificial compressibility).
* Reynolds, O. (1886) Phil. Trans. R. Soc. Lond. 177, 157-234.
* Stefan, J. (1874) Sitzber. Akad. Wiss. Wien 69, 713.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["reynolds_load", "reynolds_pressure_profile", "solve"]


def reynolds_load(a: float, h: float, V: float, nu: float, rho: float = 1.0) -> float:
    """Leading-order plane-strain squeeze load (pressure only): ``8 rho nu V a^3 / h^3``."""
    return 8.0 * rho * nu * V * a**3 / h**3


def reynolds_pressure_profile(x, a: float, h: float, V: float, nu: float,
                              rho: float = 1.0):
    """Leading-order lubrication pressure ``p(x) = 6 rho nu V (a^2 - x^2) / h^3``.

    ``x`` is measured from the plate centre; the gauge is ``p(+-a) = 0``.
    """
    return 6.0 * rho * nu * V * (a**2 - np.asarray(x, dtype=float) ** 2) / h**3


def solve(*, a: float, h: float, nu: float, V: float, ny: int = 32,
          rho: float = 1.0, cfl: float = 0.25, max_iter: int = 400_000,
          tol: float = 1e-10, div_tol: float = 1e-4, check_every: int = 200,
          stokes: bool = False, seed: float = 1e-3) -> dict:
    """Steady 2D incompressible solution of the squeeze film by artificial compressibility.

    Args:
        a: plate half-width (lattice units); the plate span is ``2a``.
        h: film thickness (lattice units).
        nu: kinematic viscosity.
        V: approach speed of the approaching plate (its uniform blowing speed).
        ny: cells across the film; the streamwise spacing is ``dx = dy`` (uniform grid,
            no interpolation anywhere).
        rho: density.
        cfl: pseudo-time CFL factor.
        max_iter, tol, check_every: steady-state controls; convergence is monitored on
            the normalised change of the wall load *and* on the residual divergence.
        div_tol: admissible ``max|div u|`` scaled by ``V/h`` (the blowing divergence).
        stokes: drop the convective term (Stokes reference).  Used to *quantify* the
            inertia contribution, never to hide it.
        seed: amplitude of the symmetry-breaking pressure seed, in units of the viscous
            pressure scale ``mu V a^2/h^3``.  The converged solution must not depend on it
            (checked in the tests).

    Returns:
        dict with the loads (top/bottom, pressure/normal), the fields, the mass balance,
        the residual divergence and the convergence diagnostics.
    """
    if a <= 0 or h <= 0:
        raise ValueError("a and h must be positive")
    if nu <= 0:
        raise ValueError("nu must be positive")
    if ny < 4:
        raise ValueError("ny must be >= 4")

    Lx = 2.0 * a
    dy = h / ny
    nx = max(3, int(round(Lx / dy)))
    dx = Lx / nx

    # --- pseudo-time step and artificial sound speed ----------------------
    # p_t + c^2 div(u) = 0 is a wave equation in pseudo time: enforce its CFL and the
    # explicit viscous limit simultaneously.  c^2 = 4 nu puts both limits at the same dt,
    # and c ~ 0.6 is two orders of magnitude above the flow speed (~0.02) as required.
    c2 = 4.0 * nu
    hmin2 = min(dx, dy) ** 2
    dt = cfl * hmin2 / max(4.0 * nu, 2.0 * c2)

    # --- staggered fields -------------------------------------------------
    # u: (nx+1, ny) at ( i dx, (j+1/2) dy )        vertical faces
    # v: (nx, ny+1) at ( (i+1/2) dx, j dy )        horizontal faces
    # p: (nx, ny)   at ( (i+1/2) dx, (j+1/2) dy )  cell centres
    u = np.zeros((nx + 1, ny))
    v = np.zeros((nx, ny + 1))
    p = np.zeros((nx, ny))

    xc = (np.arange(nx) + 0.5) * dx - a
    yc = (np.arange(ny) + 0.5) * dy

    # The exactly symmetric state (u = 0, v linear in y) is a fixed point of the
    # iteration although it is not a steady solution: u is driven only by dp/dx and a
    # purely y-dependent divergence never creates one.  Seed the x-dependence at a tiny
    # fraction of the viscous pressure scale so the iteration can find the vented,
    # physically correct branch.
    p_scale = rho * nu * V * a**2 / h**3
    p += seed * p_scale * (1.0 - (xc / a) ** 2).reshape(-1, 1) * np.ones((1, ny))

    load_prev = None
    converged = False
    residual = math.inf
    div_scaled = math.inf
    it = 0
    mu = rho * nu

    for it in range(1, max_iter + 1):
        # ================= u momentum (vertical faces) =====================
        # u^2 flux, d(u u)/dx via face averages
        f_u = 0.25 * (u[:-1, :] + u[1:, :]) ** 2          # (nx, ny) faces
        convx_u = np.zeros_like(u)
        convx_u[1:-1, :] = (f_u[1:, :] - f_u[:-1, :]) / dx
        convx_u[0, :] = (f_u[0, :] - f_u[0, :]) / dx       # open: no streamwise flux
        convx_u[-1, :] = (f_u[-1, :] - f_u[-1, :]) / dx

        # d(u v)/dy: v interpolated to the u columns (nx+1, ny+1)
        v_at_u = np.empty((nx + 1, ny + 1))
        v_at_u[1:-1, :] = 0.5 * (v[:-1, :] + v[1:, :])
        v_at_u[0, :] = v[0, :]
        v_at_u[-1, :] = v[-1, :]
        uv = u * 0.5 * (v_at_u[:, :-1] + v_at_u[:, 1:])    # (nx+1, ny)
        convy_u = np.zeros_like(u)
        convy_u[:, 1:] = (uv[:, 1:] - uv[:, :-1]) / dy

        # viscous operator: central in x (interior only), parabolic-fit at the plates
        lap_u = np.zeros_like(u)
        lap_u[1:-1, :] = (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]) / dx**2
        lap_u[0, :] = (2.0 * u[1, :] - 2.0 * u[0, :]) / dx**2     # Neumann at the vent
        lap_u[-1, :] = (2.0 * u[-2, :] - 2.0 * u[-1, :]) / dx**2
        lap_u[:, 1:-1] += (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / dy**2
        # no-slip plate at y = 0 / h with u living dy/2 away: the (4/3)(u1 - 3u0) stencil
        # is exact for a parabola, i.e. for the lubrication profile (and equals the usual
        # ghost u_{-1} = -u_0 mirror).
        lap_u[:, 0] += (4.0 / 3.0) * (u[:, 1] - 3.0 * u[:, 0]) / dy**2
        lap_u[:, -1] += (4.0 / 3.0) * (u[:, -2] - 3.0 * u[:, -1]) / dy**2

        # pressure gradient: p = 0 on the vent plane x = 0 / x = L, which lies between the
        # first cell centre and a mirrored ghost cell -> dp/dx = 2 p[0] / dx at the vent
        dpdx = np.empty_like(u)
        dpdx[1:-1, :] = (p[1:, :] - p[:-1, :]) / dx
        dpdx[0, :] = 2.0 * p[0, :] / dx
        dpdx[-1, :] = -2.0 * p[-1, :] / dx

        rhs_u = -dpdx + nu * lap_u
        if not stokes:
            rhs_u -= (convx_u + convy_u)
        u += dt * rhs_u

        # ================= v momentum (horizontal faces) ===================
        # d(u v)/dx: u interpolated to the v nodes (average of the 4 surrounding values);
        # at the plates u is odd about the wall (u_wall = 0) so the wall average vanishes.
        u_at_v = np.zeros((nx, ny + 1))
        u_at_v[:, 1:-1] = 0.25 * (u[:-1, :-1] + u[1:, :-1] + u[:-1, 1:] + u[1:, 1:])
        uv_v = v * u_at_v
        convx_v = np.zeros_like(v)
        if nx > 2:
            convx_v[1:-1, :] = (uv_v[2:, :] - uv_v[:-2, :]) / (2.0 * dx)
            convx_v[0, :] = (uv_v[1, :] - uv_v[0, :]) / dx
            convx_v[-1, :] = (uv_v[-1, :] - uv_v[-2, :]) / dx

        f_v = 0.25 * (v[:, :-1] + v[:, 1:]) ** 2
        convy_v = np.zeros_like(v)
        convy_v[:, 1:-1] = (f_v[:, 1:] - f_v[:, :-1]) / dy

        lap_v = np.zeros_like(v)
        lap_v[1:-1, :] = (v[2:, :] - 2.0 * v[1:-1, :] + v[:-2, :]) / dx**2
        lap_v[0, :] = (2.0 * v[1, :] - 2.0 * v[0, :]) / dx**2
        lap_v[-1, :] = (2.0 * v[-2, :] - 2.0 * v[-1, :]) / dx**2
        lap_v[:, 1:-1] += (v[:, 2:] - 2.0 * v[:, 1:-1] + v[:, :-2]) / dy**2

        dpdy = np.zeros_like(v)
        dpdy[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dy

        rhs_v = -dpdy + nu * lap_v
        if not stokes:
            rhs_v -= (convx_v + convy_v)
        v[:, 1:-1] += dt * rhs_v[:, 1:-1]
        v[:, 0] = 0.0            # impermeable plate
        v[:, -1] = -V            # approaching plate (uniform blowing)

        # ================= artificial compressibility ======================
        div = (u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dy
        p = p - dt * c2 * div

        if it % check_every == 0:
            div_scaled = float(np.max(np.abs(div)) / (V / h))
            f_now = float(np.sum(p[:, -1]) * dx)
            if load_prev is not None:
                residual = abs(f_now - load_prev) / max(abs(f_now), 1e-300)
                if residual < tol and div_scaled < div_tol:
                    converged = True
                    break
            load_prev = f_now

    # ------------------------------------------------------------------ loads
    # top plate: pressure at the top cell row + normal viscous stress (2nd-order one-sided)
    dvdy_top = (3.0 * (-V) - 4.0 * v[:, -2] + v[:, -3]) / (2.0 * dy)
    dvdy_bot = (0.0 + 4.0 * v[:, 1] - v[:, 2]) / (2.0 * dy)
    p_top = p[:, -1]
    p_bot = p[:, 0]
    normal_top = float(np.sum(p_top - 2.0 * mu * dvdy_top) * dx)
    normal_bot = float(np.sum(p_bot - 2.0 * mu * dvdy_bot) * dx)

    # ------------------------------------------------------------- diagnostics
    injected = Lx * V
    out_right = float(np.sum(u[-1, :]) * dy)
    out_left = float(-np.sum(u[0, :]) * dy)
    mass_imbalance = abs(injected - (out_right + out_left)) / max(injected, 1e-300)

    i_mid = nx // 2
    return {
        "a": a, "h": h, "nu": nu, "V": V, "rho": rho,
        "nx": nx, "ny": ny, "dx": dx, "dy": dy, "dt": dt, "c2": c2,
        "stokes": stokes,
        "iterations": it, "converged": converged,
        "load_residual": float(residual), "div_scaled": float(div_scaled),
        "normal_load_top": normal_top, "normal_load_bottom": normal_bot,
        "pressure_load_top": float(np.sum(p_top) * dx),
        "pressure_load_bottom": float(np.sum(p_bot) * dx),
        "reynolds_load": reynolds_load(a, h, V, nu, rho),
        "mass_imbalance_rel": mass_imbalance,
        "out_right": out_right, "out_left": out_left, "injected": injected,
        "x": xc, "y": yc, "x_u": np.arange(nx + 1) * dx - a, "y_u": yc,
        "p": p, "u": u, "v": v,
        "p_top": p_top, "p_bot": p_bot,
        "u_mid_profile": u[i_mid, :],
        "dvdy_top": dvdy_top, "dvdy_bot": dvdy_bot,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    for (aa, hh) in [(16.0, 6.0), (16.0, 3.0)]:
        r = solve(a=aa, h=hh, nu=0.1, V=0.005, ny=32)
        print(f"a={aa} h={hh} conv={r['converged']} it={r['iterations']} "
              f"div={r['div_scaled']:.2e} mass={r['mass_imbalance_rel']:.2e}")
        print(f"  normal_top={r['normal_load_top']:.6e} "
              f"pressure_top={r['pressure_load_top']:.6e} rey={r['reynolds_load']:.6e}")