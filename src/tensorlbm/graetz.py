"""Circular-tube Graetz problem: exact series, axial-marching solver and a
lattice-BGK advection-diffusion solver (anchor 3 of the CHT verification
ladder).

Physical problem
----------------
Steady laminar flow in a circular tube (radius ``R``, diameter ``D = 2R``) with
a fully developed parabolic velocity profile ``u/u_m = 2(1-ρ²)``, ``ρ = r/R``.
The fluid enters at the uniform temperature ``T_in`` and the wall is held at
``T_w`` from ``x = 0`` onwards.  Neglecting axial conduction and viscous
dissipation the energy equation reads

    ``(1-ρ²)/2 · ∂Θ/∂x* = (1/ρ) ∂/∂ρ (ρ ∂Θ/∂ρ)``,
    ``Θ = (T - T_w)/(T_in - T_w)``,
    ``x* = x /(D·Re·Pr) = α x /(u_m D²)``.

Quantities
----------
Local Nusselt number ``Nu(x*) = q_w D /(k (T_w - T_m)) = -2 (∂Θ/∂ρ)|_{ρ=1} / Θ_m``
with the cup-mixing (bulk) temperature
``Θ_m = ∫₀¹ 4ρ(1-ρ²) Θ dρ``.  The fully developed limit is
``Nu → 3.6568`` (uniform wall temperature) and the classical entrance-region
solution is the Graetz/Shah-London series, reproduced here from its
Sturm-Liouville eigenproblem (no tabulated constants are used).

Three independent numerical routes are provided
-----------------------------------------------
* :func:`graetz_exact` -- the classical series solution, obtained by solving the
  Graetz eigen-ODE ``(ρR')' + (β/2)ρ(1-ρ²)R = 0``, ``R'(0)=0``, ``R(1)=0`` by
  RK4 shooting on β (this is the Shah-London solution; Shah & London 1978
  tabulate exactly this series).
* :func:`graetz_marching` -- conservative radial finite-volume + Crank-Nicolson
  axial march of the parabolic energy equation.
* :func:`graetz_lbm` -- lattice-BGK advection-diffusion (D2Q5 of
  :mod:`tensorlbm.thermal`) on a Cartesian ``(x, r)`` grid with the
  axisymmetric radial term supplied as a source; the velocity field is the
  prescribed Poiseuille profile.

References
----------
Shah, R.K. & London, A.L. (1978). *Laminar Flow Forced Convection in Ducts*,
    Adv. Heat Transfer Suppl. 1, Academic Press.
Graetz, L. (1885). *Ann. Phys.* 25, 337.
"""

from __future__ import annotations

import math

import numpy as np

from .cht_verification import solve_tridiag

__all__ = [
    "graetz_eigenpair_table",
    "graetz_exact",
    "graetz_marching",
    "graetz_lbm",
    "NU_TUBE_DEVELOPED",
]

NU_TUBE_DEVELOPED = 3.6568


# ===========================================================================
# exact series solution (Graetz / Shah-London)
# ===========================================================================


def _eigen_residual(betas: np.ndarray, n_steps: int = 500, eps: float = 1e-7) -> np.ndarray:
    """Vectorised RK4 integration of the Graetz eigen-ODE, returning R(1;β).

    ``(ρ R')' + (β/2) ρ(1-ρ²) R = 0`` with the regular start
    ``R(ε) = 1 - β ε²/8``, ``S(ε) = R'(ε) = -β ε/4``.
    """
    beta = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    rho = eps
    R = 1.0 - beta * eps**2 / 8.0
    S = -beta * eps / 4.0
    h = (1.0 - eps) / n_steps

    def rhs(rv, Rv, Sv):
        return Sv, -0.5 * beta * (1.0 - rv * rv) * Rv - Sv / rv

    for _ in range(n_steps):
        k1R, k1S = rhs(rho, R, S)
        k2R, k2S = rhs(rho + 0.5 * h, R + 0.5 * h * k1R, S + 0.5 * h * k1S)
        k3R, k3S = rhs(rho + 0.5 * h, R + 0.5 * h * k2R, S + 0.5 * h * k2S)
        k4R, k4S = rhs(rho + h, R + h * k3R, S + h * k3S)
        R = R + h * (k1R + 2.0 * k2R + 2.0 * k3R + k4R) / 6.0
        S = S + h * (k1S + 2.0 * k2S + 2.0 * k3S + k4S) / 6.0
        rho += h
    return R


def _eigen_functions(betas, rho: np.ndarray, n_steps: int = 900, eps: float = 1e-9) -> np.ndarray:
    """Eigenfunctions ``R_n(ρ)`` for several eigenvalues at once (vectorised RK4)."""
    beta = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    rs = np.linspace(eps, 1.0, n_steps + 1)
    Rs = np.empty((n_steps + 1, beta.size))
    R = 1.0 - beta * eps**2 / 8.0
    S = -beta * eps / 4.0
    Rs[0] = R
    h = (1.0 - eps) / n_steps

    def rhs(rv, Rv, Sv):
        return Sv, -0.5 * beta * (1.0 - rv * rv) * Rv - Sv / rv

    for i in range(n_steps):
        rv = rs[i]
        k1R, k1S = rhs(rv, R, S)
        k2R, k2S = rhs(rv + 0.5 * h, R + 0.5 * h * k1R, S + 0.5 * h * k1S)
        k3R, k3S = rhs(rv + 0.5 * h, R + 0.5 * h * k2R, S + 0.5 * h * k2S)
        k4R, k4S = rhs(rv + h, R + h * k3R, S + h * k3S)
        R = R + h * (k1R + 2.0 * k2R + 2.0 * k3R + k4R) / 6.0
        S = S + h * (k1S + 2.0 * k2S + 2.0 * k3S + k4S) / 6.0
        Rs[i + 1] = R
    rr = np.asarray(rho, dtype=np.float64)
    return np.stack([np.interp(rr, rs, Rs[:, j]) for j in range(beta.size)], axis=-1)


def graetz_eigenpair_table(n_modes: int = 10, beta_max: float = 4000.0) -> dict:
    """First ``n_modes`` Graetz eigenvalues and eigenfunction coefficients.

    The eigenvalues solve ``R(1;β) = 0``; they are bracketed by sign changes of
    ``R(1;β)`` on a uniform β grid and refined by vectorised bisection.  The
    coefficients ``C_n`` follow from the inlet condition ``Σ C_n R_n = 1`` using
    the orthogonality of the eigenfunctions with weight ``ρ(1-ρ²)/2``.
    """
    n_scan = max(400, int(beta_max / 4.0))
    betas = np.linspace(0.05, beta_max, n_scan)
    vals = _eigen_residual(betas, n_steps=600)
    idx = np.where(vals[:-1] * vals[1:] < 0.0)[0]
    if idx.size < n_modes:
        raise RuntimeError(f"only {idx.size} eigenvalues bracketed below beta={beta_max}")
    idx = idx[:n_modes]

    lo = betas[idx].copy()
    hi = betas[idx + 1].copy()
    flo = vals[idx].copy()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = _eigen_residual(mid, n_steps=600)
        same = flo * fm <= 0.0
        hi = np.where(same, mid, hi)
        lo = np.where(same, lo, mid)
        flo = np.where(same, flo, fm)
    roots = 0.5 * (lo + hi)

    rho = np.linspace(0.0, 1.0, 801)
    w = rho * (1.0 - rho**2) / 2.0
    modes = []
    Rall = _eigen_functions(roots, rho)
    for j, beta in enumerate(roots):
        R = Rall[:, j]
        num = float(np.trapezoid(R * w, rho))
        den = float(np.trapezoid(R * R * w, rho))
        drho = rho[-1] - rho[-2]
        # 2nd-order one-sided derivative at the wall (R(1) = 0)
        dR_wall = (3.0 * R[-1] - 4.0 * R[-2] + R[-3]) / (2.0 * drho)
        modes.append(
            {
                "beta": float(beta),
                "C": num / den,
                "R": R,
                "dR_wall": dR_wall,
                "norm": den,
            }
        )
    return {"rho": rho, "modes": modes, "beta": [m["beta"] for m in modes]}


def graetz_exact(
    x_star,
    n_modes: int = 10,
    table: dict | None = None,
    beta_max: float = 4000.0,
) -> dict:
    """Classical Graetz series solution for the local Nusselt number.

        ``Θ(ρ, x*) = Σ_n C_n R_n(ρ) exp(-β_n x*)``,
        ``Nu(x*) = -2 Σ_n C_n R_n'(1) exp(-β_n x*) / Θ_m``

    ``x* = x/(D Re Pr)``.  This is exactly the solution tabulated by Shah &
    London (1978) for the constant-wall-temperature circular duct.
    """
    if table is None:
        table = graetz_eigenpair_table(n_modes=n_modes, beta_max=beta_max)
    rho = table["rho"]
    modes = table["modes"]
    cup = 4.0 * rho * (1.0 - rho**2)

    xs = np.atleast_1d(np.asarray(x_star, dtype=np.float64))
    nu = np.empty(xs.size)
    theta_m = np.empty(xs.size)
    profiles = []
    for j, xsv in enumerate(xs):
        Theta = np.zeros_like(rho)
        dTheta_wall = 0.0
        for m in modes:
            e = math.exp(-m["beta"] * xsv)
            Theta += m["C"] * m["R"] * e
            dTheta_wall += m["C"] * m["dR_wall"] * e
        Tm = float(np.trapezoid(cup * Theta, rho))
        theta_m[j] = Tm
        nu[j] = -2.0 * dTheta_wall / Tm
        profiles.append(Theta)

    return {
        "x_star": xs,
        "Nu": nu,
        "theta_m": theta_m,
        "rho": rho,
        "profiles": profiles,
        "beta": table["beta"],
        "mode_coeffs": [(m["beta"], m["C"]) for m in modes],
    }


# ===========================================================================
# axial-marching solver (conservative radial FV + Crank-Nicolson in x*)
# ===========================================================================


def graetz_marching(
    n_r: int = 160,
    x_star_max: float = 0.30,
    x_star_stations: np.ndarray | None = None,
    n_x: int = 2400,
    theta_scheme: float = 0.5,
) -> dict:
    """Graetz energy equation by conservative radial FV + axial march.

    The radial operator is ``L_i = [ρ_{i+1/2}(Θ_{i+1}-Θ_i) - ρ_{i-1/2}(Θ_i-Θ_{i-1})]
    /(Δρ² ρ_i)`` (cell volume ``ρ Δρ``), and the march uses Crank–Nicolson,
    ``∂Θ/∂x* = L/[(1-ρ²)/2]``, with implicit Euler for the first few steps to
    absorb the inlet discontinuity.
    """
    drho = 1.0 / n_r
    rho = drho * (np.arange(n_r) + 0.5)
    rho_face = drho * (np.arange(1, n_r + 1))  # ρ_{i+1/2}, last == 1 (wall)

    a = 0.5 * (1.0 - rho**2)  # advection factor (1-ρ²)/2

    c_up = np.zeros(n_r)
    c_down = np.zeros(n_r)
    c_wall = np.zeros(n_r)
    c_up[:-1] = rho_face[:-1] / drho**2 / rho[:-1]  # coupling i -> i+1
    c_down[1:] = rho_face[:-1] / drho**2 / rho[1:]  # coupling i -> i-1
    c_wall[-1] = 2.0 * rho_face[-1] / drho**2 / rho[-1]  # wall Dirichlet (Θ=0), half-cell distance

    def apply_L(v: np.ndarray, out_diag_extra=None) -> np.ndarray:
        out = np.zeros_like(v)
        out[:-1] += c_up[:-1] * (v[1:] - v[:-1])
        out[1:] += c_down[1:] * (v[:-1] - v[1:])
        out[-1] += c_wall[-1] * (0.0 - v[-1])
        return out

    dxs = x_star_max / n_x
    Theta = np.ones(n_r)

    stations = (
        np.atleast_1d(np.asarray(x_star_stations, dtype=np.float64))
        if x_star_stations is not None
        else np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    )
    hist = np.empty((n_x + 1, n_r))
    hist[0] = Theta
    x_now = 0.0
    x_hist = np.empty(n_x + 1)
    x_hist[0] = 0.0

    for step in range(1, n_x + 1):
        th = 1.0 if step <= 5 else theta_scheme
        Ln = apply_L(Theta)
        lower = np.zeros(n_r)
        diag = np.ones(n_r)
        upper = np.zeros(n_r)
        diag += dxs * th * (c_up + c_down + c_wall) / a
        upper[:-1] += -dxs * th * c_up[:-1] / a[:-1]
        lower[1:] += -dxs * th * c_down[1:] / a[1:]
        rhs = Theta + dxs * (1.0 - th) * Ln / a
        Theta = solve_tridiag(lower, diag, upper, rhs)
        hist[step] = Theta
        x_now += dxs
        x_hist[step] = x_now

    cup = 4.0 * rho * (1.0 - rho**2)
    out_x = []
    out_nu = []
    out_tm = []
    out_prof = []
    for xsv in stations:
        idx = int(np.clip(np.searchsorted(x_hist, xsv), 1, n_x))
        # linear interpolation between the two bracketing marches
        x0, x1 = x_hist[idx - 1], x_hist[idx]
        w = 0.0 if x1 == x0 else float(np.clip((xsv - x0) / (x1 - x0), 0.0, 1.0))
        Th = (1.0 - w) * hist[idx - 1] + w * hist[idx]
        # 2nd-order one-sided derivative at the wall: nodes at u = Δρ/2, 3Δρ/2
        # (u = 1-ρ) together with the Dirichlet wall value Θ(1) = 0.
        dTdrho_wall = -(3.0 * Th[-1] - Th[-2] / 3.0) / drho
        Tm = float(np.sum(cup * Th * drho))
        out_x.append(float(xsv))
        out_nu.append(float(-2.0 * dTdrho_wall / Tm))
        out_tm.append(Tm)
        out_prof.append(Th)

    return {
        "x_star": np.asarray(out_x),
        "Nu": np.asarray(out_nu),
        "theta_m": np.asarray(out_tm),
        "profiles": out_prof,
        "rho": rho,
        "meta": {"n_r": n_r, "n_x": n_x, "dx_star": dxs, "theta_scheme": theta_scheme},
    }


# ===========================================================================
# lattice-BGK advection-diffusion solver (D2Q5 from tensorlbm.thermal)
# ===========================================================================


def graetz_lbm(
    n_r: int = 40,
    n_x: int = 320,
    u_mean: float = 0.05,
    alpha: float = 0.03,
    n_steps: int | None = None,
    T_in: float = 1.0,
    T_wall: float = 0.0,
    stations_x: np.ndarray | None = None,
) -> dict:
    """Graetz problem with the D2Q5 lattice-BGK thermal LBM.

    The solver is a NumPy mirror of the library's D2Q5 kernels
    (:func:`tensorlbm.thermal.temperature_equilibrium` /
    ``collide_thermal_bgk``, weights :data:`tensorlbm.thermal.W5`,
    ``alpha = c_s^2 (tau - 1/2) = (tau - 1/2)/3``); the mirror is checked
    element-wise against the torch kernels in ``tests/test_cht.py``
    (``test_graetz_lbm_kernel_matches_library``), so the two cannot drift.  The
    NumPy path is used because per-step launch overhead of the torch kernels
    dominates on the small ``(5, n_r, n_x)`` grids used here.

    The physical model is the Cartesian ``(x, r)`` form of
    ``∂_t T + u ∂_x T = alpha(∂_r²T + (1/r)∂_r T)``: the velocity field is the
    prescribed Poiseuille profile ``u_x(r) = 2 u_m (1-(r/R)²)`` (``y``-index =
    radius in lattice units, ``R = n_r - 1``) and the axisymmetric radial term
    enters as the source ``S = alpha (1/r) ∂_r T`` (limit ``2 ∂_r² T`` on the
    axis).  Wall Dirichlet uses anti-bounce-back, the axis uses specular
    reflection, the inlet a Dirichlet column and the outlet a zero-gradient
    (convective outflow) condition.

    Returns the local Nusselt number ``Nu = -2R (∂T/∂r)|_w / T_m`` at the
    requested axial stations in the same ``x* = alpha x/(u_m D²)``
    parameterisation as :func:`graetz_exact`.
    """
    from .thermal import W5 as _W5

    w = np.asarray(_W5, dtype=np.float64)  # (5,)
    wb = w.reshape(5, 1, 1)
    c = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)

    ny = n_r
    R = float(n_r - 1)
    tau = 0.5 + 3.0 * alpha

    yy = np.arange(ny, dtype=np.float64)
    ux = np.broadcast_to((2.0 * u_mean * (1.0 - (yy / R) ** 2)).reshape(ny, 1), (ny, n_x)).copy()
    uy = np.zeros_like(ux)

    T = np.full((ny, n_x), float(T_in))
    cu = c[:, 0].reshape(5, 1, 1) * ux[None] + c[:, 1].reshape(5, 1, 1) * uy[None]
    geq = wb * T[None] * (1.0 + 3.0 * cu)
    g = geq.copy()

    if n_steps is None:
        n_steps = int(1.6 * n_x / u_mean)
    if stations_x is None:
        stations_x = np.array([40, 80, 160, 240, 320], dtype=float)

    d_dim = 2.0 * R
    x_star = alpha * np.asarray(stations_x, dtype=float) / (u_mean * d_dim**2)

    inv_tau = 1.0 / tau
    for _ in range(n_steps):
        T = g.sum(axis=0)

        # --- axisymmetric radial source alpha (1/r) dT/dr ---
        grad = np.zeros_like(T)
        grad[1:-1] = 0.5 * (T[2:] - T[:-2])
        grad[-1] = float(T_wall) - T[-2]
        src = np.empty_like(T)
        src[1:] = alpha * grad[1:] / yy[1:].reshape(-1, 1)
        src[0] = alpha * 2.0 * (T[1] - T[0])  # 1/r dT/dr -> d2T/dr2 at the axis

        # --- BGK collision (library D2Q5 equilibrium) + source ---
        cu = c[:, 0].reshape(5, 1, 1) * ux[None] + c[:, 1].reshape(5, 1, 1) * uy[None]
        geq = wb * T[None] * (1.0 + 3.0 * cu)
        g_post = g - (g - geq) * inv_tau + wb * src[None]

        # --- streaming (explicit, so that boundaries can be overwritten) ---
        g = np.empty_like(g_post)
        g[0] = g_post[0]
        g[1, :, 1:] = g_post[1, :, :-1]
        g[2, :, :-1] = g_post[2, :, 1:]
        g[3, 1:, :] = g_post[3, :-1, :]
        g[4, :-1, :] = g_post[4, 1:, :]

        # inlet: Dirichlet column (anti-bounce-back)
        g[1, :, 0] = -g_post[2, :, 0] + 2.0 * w[1] * float(T_in)
        # outlet: zero-gradient
        g[2, :, -1] = g[2, :, -2]
        # wall: Dirichlet (anti-bounce-back)
        g[4, -1, :] = -g_post[3, -1, :] + 2.0 * w[4] * float(T_wall)
        # axis: specular reflection
        g[3, 0, :] = g_post[4, 0, :]

    T = g.sum(axis=0)
    rr = np.arange(ny, dtype=np.float64)
    u_prof = 2.0 * u_mean * (1.0 - (rr / R) ** 2)
    weight = u_prof * rr

    nu = []
    tm = []
    for xst in stations_x:
        col = int(round(float(xst)))
        col = min(max(col, 2), n_x - 1)
        Tw = T[-1, col]
        dTdr_wall = (3.0 * Tw - 4.0 * T[-2, col] + T[-3, col]) / 2.0
        Tm = float(np.sum(weight * T[:, col]) / np.sum(weight))
        tm.append(Tm)
        nu.append(float(-2.0 * R * dTdr_wall / Tm))

    return {
        "x_lattice": np.asarray(stations_x, dtype=float),
        "x_star": x_star,
        "Nu": np.asarray(nu),
        "theta_m": np.asarray(tm),
        "field": T,
        "r": rr,
        "meta": {
            "n_r": n_r,
            "n_x": n_x,
            "n_steps": n_steps,
            "u_mean": u_mean,
            "alpha": alpha,
            "tau": tau,
            "R": R,
            "mach": float(u_mean / math.sqrt(1.0 / 3.0)),
            "wall_temperature_offset": float(np.max(np.abs(T[-1] - T_wall))),
        },
    }
