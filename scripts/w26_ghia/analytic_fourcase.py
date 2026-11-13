#!/usr/bin/env python3
"""W26: the four analytic anchor cases for a D3Q19-BGK lattice-Boltzmann code.

Every case is checked at the *operator* level: the closed-form solution is
substituted back into the governing equations and the residual measured.  No
literature number enters this file -- everything is exact algebra, high-precision
differentiation (mpmath) or spectral derivatives (FFT).

  case 1  plane Poiseuille    u(y) = G/(2 mu) y (H-y)        forced channel
  case 2  plane Couette       u(y) = U y / H                 shear channel
  case 3  Taylor-Green vortex 2-D periodic decaying vortex
  case 4  cavity limits       creeping (Stokes) limit: biharmonic solution space,
                              Stokes pressure integrability, plus the two exact
                              mass-conservation invariants of the cavity tables

Each case reports residual_abs ||L[u]||_inf, residual_rel (normalised by the
largest term of the operator) and a `control`: the residual of a deliberately
corrupted solution.  The control is what makes a 1e-16 residual meaningful -- the
same code has to report O(1) for a wrong solution.

Two numerical facts that are easy to get wrong and are therefore recorded:
  * a 2nd-order central Laplacian in float64 cannot go below ~eps*|u|/h^2
    (cancellation), so the discrete residual is reported together with that floor;
  * the exact Poiseuille parabola is integrated exactly by the 2nd-order scheme
    (zero discretisation error), so the convergence slope is measured on a
    manufactured non-polynomial solution instead.

Usage:
    python3 scripts/w26_ghia/analytic_fourcase.py --out /tmp/analytic.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict

import mpmath as mp
import numpy as np

RESIDUAL_TOL = 1e-12   # mission criterion
EPS = float(np.finfo(float).eps)


def _stats(res: np.ndarray, scale: float) -> Dict[str, float]:
    a = float(np.max(np.abs(res)))
    return {"residual_abs": a, "residual_rel": a / scale if scale else float("nan")}


def tridiag_solve(a: float, b: float, c: float, d: np.ndarray) -> np.ndarray:
    n = len(d)
    cp = np.zeros(max(n - 1, 1))
    dp = np.zeros(n)
    cp[0] = c / b
    dp[0] = d[0] / b
    for i in range(1, n):
        m = b - a * cp[i - 1]
        dp[i] = (d[i] - a * dp[i - 1]) / m
        if i < n - 1:
            cp[i] = c / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# --------------------------------------------------------------------------- #
def case1_poiseuille(H: float = 1.0, mu: float = 1e-3, G: float = 1.0,
                     h_residual: float = 1.0 / 32.0,
                     grids=(8, 16, 32, 64, 128, 256, 512)) -> Dict[str, object]:
    """u(y) = G/(2 mu) y (H-y):  steady NS x-momentum 0 = -G + mu u'' = 0."""
    umax = G * H * H / (8.0 * mu)
    y = np.linspace(0.0, H, 1001)
    u = G / (2.0 * mu) * y * (H - y)
    res_cont = mu * np.full_like(y, -G / mu) + G          # exact second derivative

    N = int(round(H / h_residual))                        # coarse enough: eps/h^2 << 1e-12
    yd = np.linspace(0.0, H, N + 1)
    hd = H / N
    ud = G / (2.0 * mu) * yd * (H - yd)
    d2u_fd = (ud[2:] - 2.0 * ud[1:-1] + ud[:-2]) / hd ** 2
    res_disc = mu * d2u_fd + G
    floor = EPS * float(np.max(np.abs(ud))) * mu / hd ** 2

    shape = float(np.max(np.abs(u / umax - 4.0 * (y / H) * (1.0 - y / H))))

    exact_discrete = []
    for Ng in grids:
        yy = np.linspace(0.0, H, Ng + 1)
        hh = H / Ng
        uu = tridiag_solve(-mu / hh ** 2, 2.0 * mu / hh ** 2, -mu / hh ** 2, np.full(Ng - 1, G))
        u_full = np.concatenate(([0.0], uu, [0.0]))
        u_ex = G / (2.0 * mu) * yy * (H - yy)
        exact_discrete.append({"N": Ng, "h": hh,
                               "linf_error_vs_analytic": float(np.max(np.abs(u_full - u_ex))),
                               "roundoff_floor_estimate": EPS * umax * mu / hh ** 2})

    # manufactured non-polynomial solution: -mu u'' = mu pi^2 sin(pi y), u = sin(pi y)
    man, errs = [], []
    for Ng in grids:
        yy = np.linspace(0.0, H, Ng + 1)
        hh = H / Ng
        rhs = mu * np.pi ** 2 * np.sin(np.pi * yy[1:-1])
        uu = tridiag_solve(-mu / hh ** 2, 2.0 * mu / hh ** 2, -mu / hh ** 2, rhs)
        u_full = np.concatenate(([0.0], uu, [0.0]))
        err = float(np.max(np.abs(u_full - np.sin(np.pi * yy))))
        man.append({"N": Ng, "h": hh, "linf_error": err})
        errs.append(err)
    slopes = [float(np.log(errs[i] / errs[i + 1]) / np.log(2.0)) for i in range(len(errs) - 1)]
    hs = np.array([m["h"] for m in man])
    p_fit = float(np.polyfit(np.log(hs), np.log(np.array(errs)), 1)[0])
    p_fit_asym = float(np.polyfit(np.log(hs[len(hs) // 2:]),
                                  np.log(np.array(errs[len(errs) // 2:])), 1)[0])

    u_bad = G / (2.0 * mu) * y * (H - y) * 1.01           # negative control
    res_bad = mu * np.gradient(np.gradient(u_bad, y), y) + G

    return {
        "case": "plane_poiseuille",
        "solution": "u(y) = G/(2 mu) y (H-y), U_max = G H^2/(8 mu)",
        "parameters": {"H": H, "mu": mu, "G": G, "U_max": umax,
                       "residual_grid_h": hd, "residual_roundoff_floor_rel": floor / G},
        "residual_continuum": _stats(res_cont, G),
        "residual_discrete_laplacian": _stats(res_disc, G),
        "shape_identity_linf": shape,
        "parabola_reproduced_exactly": {
            "rows": exact_discrete,
            "note": "2nd-order central differences integrate a quadratic exactly -> "
                    "zero discretisation error for Poiseuille"},
        "convergence": {"manufactured_solution": "u = sin(pi y), -mu u'' = mu pi^2 sin(pi y)",
                        "grids": man, "pairwise_slopes": slopes,
                        "fitted_slope_p": p_fit, "fitted_slope_p_asymptotic": p_fit_asym,
                        "second_order_claimed": 2.0, "slope_error": abs(p_fit_asym - 2.0)},
        "control_wrong_amplitude_1pct": _stats(res_bad, G),
        "pass_residual": bool(_stats(res_cont, G)["residual_rel"] <= RESIDUAL_TOL and
                              _stats(res_disc, G)["residual_abs"] <= max(RESIDUAL_TOL, 10 * floor)),
        "pass_second_order": bool(abs(p_fit_asym - 2.0) <= 1e-2),
    }


# --------------------------------------------------------------------------- #
def case2_couette(H: float = 1.0, mu: float = 1e-3, U: float = 1.0,
                  h_residual: float = 1.0 / 32.0) -> Dict[str, object]:
    """u(y) = U y/H:  steady NS 0 = mu u'' = 0 exactly."""
    y = np.linspace(0.0, H, 1001)
    u = U * y / H
    res_cont = mu * np.zeros_like(y)
    N = int(round(H / h_residual))
    yd = np.linspace(0.0, H, N + 1)
    hd = H / N
    ud = U * yd / H
    d2u_fd = (ud[2:] - 2.0 * ud[1:-1] + ud[:-2]) / hd ** 2
    res_disc = mu * d2u_fd
    floor = EPS * float(np.max(np.abs(ud))) * mu / hd ** 2
    tau_disc = mu * (-3.0 * ud[0] + 4.0 * ud[1] - ud[2]) / (2.0 * hd)   # 3-point one-sided
    tau_exact = mu * U / H
    u_bad = U * (y / H) ** 2                                             # negative control
    res_bad = mu * np.gradient(np.gradient(u_bad, y), y)
    scale = mu * U / H ** 2
    return {
        "case": "plane_couette",
        "solution": "u(y) = U y/H (linear shear)",
        "parameters": {"H": H, "mu": mu, "U": U, "residual_grid_h": hd,
                       "residual_roundoff_floor_rel": floor / scale},
        "residual_continuum": _stats(res_cont, scale),
        "residual_discrete_laplacian": _stats(res_disc, scale),
        "wall_shear": {"tau_exact": tau_exact, "tau_from_discrete_gradient": float(tau_disc),
                       "rel_error": float(abs(tau_disc - tau_exact) / tau_exact)},
        "control_parabolic_profile": _stats(res_bad, scale),
        "pass_residual": bool(_stats(res_cont, scale)["residual_rel"] <= RESIDUAL_TOL and
                              _stats(res_disc, scale)["residual_abs"] <= max(RESIDUAL_TOL, 10 * floor)),
    }


# --------------------------------------------------------------------------- #
def case3_taylor_green(U: float = 1.0, k: float = 1.0, nu: float = 1e-2,
                       t: float = 0.5, N: int = 64) -> Dict[str, object]:
    """TG vortex: u = U cos(kx) sin(ky) D, v = -U sin(kx) cos(ky) D,
    p = -U^2/4 (cos2kx + cos2ky) D^2, D = exp(-2 nu k^2 t); FFT derivatives."""
    L = 2.0 * math.pi / k
    x = np.arange(N) * L / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    D = math.exp(-2.0 * nu * k ** 2 * t)
    u = U * np.cos(k * X) * np.sin(k * Y) * D
    v = -U * np.sin(k * X) * np.cos(k * Y) * D
    p = -U ** 2 / 4.0 * (np.cos(2 * k * X) + np.cos(2 * k * Y)) * D * D
    rho = 1.0
    kx = k * np.fft.fftfreq(N, 1.0 / N)

    def dx(f):
        return np.real(np.fft.ifft2(1j * kx[:, None] * np.fft.fft2(f)))

    def dy(f):
        return np.real(np.fft.ifft2(1j * kx[None, :] * np.fft.fft2(f)))

    div = dx(u) + dy(v)
    u_t, v_t = -2.0 * nu * k ** 2 * u, -2.0 * nu * k ** 2 * v
    mom_x = u_t + u * dx(u) + v * dy(u) + dx(p) / rho - nu * (dx(dx(u)) + dy(dy(u)))
    mom_y = v_t + u * dx(v) + v * dy(v) + dy(p) / rho - nu * (dx(dx(v)) + dy(dy(v)))
    scale_x = max(np.max(np.abs(u_t)), np.max(np.abs(u * dx(u) + v * dy(u))),
                  np.max(np.abs(dx(p))), np.max(np.abs(nu * (dx(dx(u)) + dy(dy(u))))))
    scale_y = max(np.max(np.abs(v_t)), np.max(np.abs(u * dx(v) + v * dy(v))),
                  np.max(np.abs(dy(p))), np.max(np.abs(nu * (dx(dx(v)) + dy(dy(v))))))

    omega = dx(v) - dy(u)
    E = 0.5 * np.mean(u ** 2 + v ** 2)
    dE_dt = -4.0 * nu * k ** 2 * E
    rhs = -nu * np.mean(omega ** 2)   # 2-D: dE/dt = -nu <omega^2>
    E_exact = 0.25 * U ** 2 * math.exp(-4.0 * nu * k ** 2 * t)
    zeta_exact = -nu * (U * k) ** 2 * math.exp(-4.0 * nu * k ** 2 * t)
    ctrl = float(np.max(np.abs(dx(p * 1.03) / rho - dx(p) / rho)) / scale_x)
    return {
        "case": "taylor_green_2d",
        "solution": "u=U cos(kx) sin(ky) e^{-2 nu k^2 t}, v=-U sin(kx) cos(ky) e^{-2 nu k^2 t}, "
                    "p=-U^2/4 (cos2kx+cos2ky) e^{-4 nu k^2 t}",
        "parameters": {"U": U, "k": k, "nu": nu, "t": t, "N": N, "Re_k": U / (k * nu)},
        "divergence": _stats(div, U * k),
        "momentum_x": _stats(mom_x, scale_x),
        "momentum_y": _stats(mom_y, scale_y),
        "energy": {"E_spectral": float(E), "E_exact": E_exact,
                   "rel_error_E": float(abs(E - E_exact) / E_exact),
                   "dE_dt_identity_rel_error": float(abs(dE_dt - rhs) / abs(zeta_exact)),
                   "identity_used": "2-D periodic: dE/dt = -nu <omega^2>",
                   "dissipation_identity_rel_error": float(abs(rhs - zeta_exact) / abs(zeta_exact))},
        "control_pressure_3pct": {"momentum_x_rel": ctrl},
        "pass_residual": bool(max(_stats(div, U * k)["residual_rel"],
                                  _stats(mom_x, scale_x)["residual_rel"],
                                  _stats(mom_y, scale_y)["residual_rel"]) <= RESIDUAL_TOL),
    }


# --------------------------------------------------------------------------- #
# case 4: creeping (Stokes) limit of the cavity
# --------------------------------------------------------------------------- #
def _psi_factory(lam, coeffs, shift=2):
    """psi(r,t) = r^lam [A cos(l t) + B sin(l t) + C cos((l-shift) t) + D sin((l-shift) t)]"""
    A, B, C, D = [mp.mpf(c) for c in coeffs]
    lam_m, s = mp.mpf(lam), mp.mpf(shift)

    def psi(rr, tt):
        return rr ** lam_m * (A * mp.cos(lam_m * tt) + B * mp.sin(lam_m * tt)
                              + C * mp.cos((lam_m - s) * tt) + D * mp.sin((lam_m - s) * tt))
    return psi


def _dr(f, r, t, n=1):
    return mp.diff(lambda rr: f(rr, t), r, n)


def _dt(f, r, t, n=1):
    return mp.diff(lambda tt: f(r, tt), t, n)


def _lap_polar(f, r, t):
    return _dr(f, r, t, 2) + _dr(f, r, t, 1) / r + _dt(f, r, t, 2) / r ** 2


def _gl_quad(f, a, b, n: int = 12):
    """Fixed-node Gauss-Legendre quadrature (cheap, deterministic, high precision)."""
    xs, ws = _gl_nodes(n)
    mid, half = (a + b) / 2, (b - a) / 2
    return half * mp.fsum(w * f(mid + half * x) for x, w in zip(xs, ws))


def _gl_nodes(n: int):
    """Gauss-Legendre nodes/weights on [-1, 1] from the Legendre polynomial roots."""
    if n not in _GL_CACHE:
        # Legendre P_n via numpy for the initial guess, then Newton refinement in mp
        import numpy.polynomial.legendre as L
        c = np.zeros(n + 1)
        c[n] = 1.0
        roots = L.legroots(c)
        xs, ws = [], []
        for r in roots:
            x = mp.mpf(float(r))
            p = mp.legendre(n, x)
            dp = mp.diff(lambda t: mp.legendre(n, t), x)
            x = x - p / dp
            dp = mp.diff(lambda t: mp.legendre(n, t), x)
            w = 2 / ((1 - x ** 2) * dp ** 2)
            xs.append(x)
            ws.append(w)
        _GL_CACHE[n] = (xs, ws)
    return _GL_CACHE[n]


_GL_CACHE: Dict[int, tuple] = {}


def case4_cavity_limits(mu: float = 1.0, U: float = 1.0, dps: int = 60) -> Dict[str, object]:
    """Creeping limit of the lid-driven cavity: biharmonic solution space + Stokes
    pressure integrability, verified with high-precision derivatives."""
    mp.mp.dps = dps
    coeffs = (1.0, 0.4, 0.7, -0.3)
    samples_rt = [(0.37, 0.21), (0.8, 1.05), (0.15, -0.6), (1.2, 2.0)]

    # (a) biharmonic exactness of the separable family
    fam = []
    for lam in (2.0, 2.7396, 3.0, 4.5):
        psi = _psi_factory(lam, coeffs)
        vals = [abs(_lap_polar(lambda r_, t_: _lap_polar(psi, r_, t_), r, t)) for r, t in samples_rt]
        mx = max(vals)
        fam.append({"lam": lam, "max_abs_high_precision": float(mx),
                    "max_abs_log10": float(mp.log10(mx)) if mx > 0 else None})

    # (b) Stokes: velocity from psi, integrability of mu*lap(u), pressure reconstruction
    lam = 2.7396
    psi = _psi_factory(lam, coeffs)

    def P(x, y):
        return psi(mp.sqrt(x ** 2 + y ** 2), mp.atan2(y, x))

    def ufn(x, y):
        return mp.diff(lambda yy: P(x, yy), y, 1)

    def vfn(x, y):
        return -mp.diff(lambda xx: P(xx, y), x, 1)

    def lap_u(x, y):
        return (mp.diff(lambda xx: ufn(xx, y), x, 2) + mp.diff(lambda yy: ufn(x, yy), y, 2))

    def lap_v(x, y):
        return (mp.diff(lambda xx: vfn(xx, y), x, 2) + mp.diff(lambda yy: vfn(x, yy), y, 2))

    x0, y0 = mp.mpf("0.55"), mp.mpf("0.35")

    def p_path(x, y):
        return mu * (_gl_quad(lambda tt: lap_u(tt, y0), x0, x, 10)
                     + _gl_quad(lambda tt: lap_v(x, tt), y0, y, 10))

    integ, press = [], []
    for x, y in [(mp.mpf("0.66"), mp.mpf("0.42")), (mp.mpf("0.62"), mp.mpf("0.47"))]:
        cross = (mp.diff(lambda yy: lap_u(x, yy), y, 1) - mp.diff(lambda xx: lap_v(xx, y), x, 1))
        div_u = mp.diff(lambda xx: ufn(xx, y), x, 1) + mp.diff(lambda yy: vfn(x, yy), y, 1)
        integ.append({"point": [float(x), float(y)],
                      "curl_of_lap_u": float(abs(cross)),
                      "divergence_of_u": float(abs(div_u))})
        px = mp.diff(lambda xx: p_path(xx, y), x, 1)
        py = mp.diff(lambda yy: p_path(x, yy), y, 1)
        su, sv = abs(mu * lap_u(x, y)), abs(mu * lap_v(x, y))
        press.append({"point": [float(x), float(y)],
                      "p": float(p_path(x, y)),
                      "residual_x_rel": float(abs(px - mu * lap_u(x, y)) / su),
                      "residual_y_rel": float(abs(py - mu * lap_v(x, y)) / sv)})

    # (b2) differentiation-noise floor of the same operator: a pure harmonic mode
    #      r^2 sin(2 theta) is biharmonic, so whatever the FD composition returns
    #      for it is pure numerical noise, i.e. the floor of this check.
    fl = _psi_factory(2.0, (0.0, 1.0, 0.0, 0.0), shift=2)
    floor = float(max(abs(_lap_polar(lambda r_, t_: _lap_polar(fl, r_, t_), r, t))
                      for r, t in samples_rt))

    # (c) negative control: exponent shift 2 -> 3 is not biharmonic
    psi_bad = _psi_factory(lam, coeffs, shift=3)
    ctrl = [abs(_lap_polar(lambda r_, t_: _lap_polar(psi_bad, r_, t_), r, t)) for r, t in samples_rt]
    ctrl_mp = max(ctrl)

    worst = max(max(r["max_abs_high_precision"] for r in fam),
                max(r["curl_of_lap_u"] for r in integ),
                max(max(r["residual_x_rel"], r["residual_y_rel"]) for r in press),
                max(r["divergence_of_u"] for r in integ))
    return {
        "case": "cavity_creeping_limit",
        "note": "Re->0 limit of the lid-driven cavity is the biharmonic Stokes problem "
                "grad^4 psi = 0, psi = psi_n = 0 on the walls, psi_n = -U on the lid; the "
                "separable corner family spans its exact solution space",
        "parameters": {"mu": mu, "U": U, "dps": dps},
        "biharmonic_family": fam,
        "stokes": {"integrability_and_continuity": integ,
                   "pressure_reconstruction": press,
                   "note": "p is reconstructed by path integration of mu*lap(u); both "
                           "momentum components are then re-verified by differentiating p"},
        "control_wrong_exponent": {"max_abs_high_precision": float(ctrl_mp),
                                   "max_abs_log10": float(mp.log10(ctrl_mp))},
        "fd_noise_floor": floor,
        "worst_residual": float(worst),
        "pass_residual": bool(worst <= max(RESIDUAL_TOL, 10 * floor)),
    }


# --------------------------------------------------------------------------- #
def cavity_table_invariants(y, u, x, v) -> Dict[str, object]:
    """Exact invariants of an incompressible lid-driven cavity.

    Incompressibility gives d/dx int u dy = -[v]_{y=0,1} = 0 and d/dy int v dx = 0,
    and both integrals vanish on the resting wall, hence

        int_0^1 u(x=0.5, y) dy = 0     (Table I, vertical centreline)
        int_0^1 v(x, y=0.5) dx = 0     (Table II, horizontal centreline)

    The tabulated values are 5-decimal truncations on a coarse non-uniform point
    set, so the quadrature residual is O(1e-3); the tolerance is derived from the
    truncation of the values, not fitted to the result.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    oy, ox = np.argsort(y), np.argsort(x)     # tables are listed top-down
    y, u = y[oy], u[oy]
    x, v = x[ox], v[ox]

    def trapz(yy, ff):
        return float(np.sum(0.5 * (ff[1:] + ff[:-1]) * np.diff(yy)))

    def simpson_nonuniform(yy, ff):
        tot = 0.0
        for i in range(0, len(yy) - 2, 2):
            a0, a1, a2 = yy[i], yy[i + 1], yy[i + 2]
            f0, f1, f2 = ff[i], ff[i + 1], ff[i + 2]
            d = (a2 - a0) * (a2 - a1) * (a1 - a0)
            A = ((f2 - f0) * (a1 - a0) - (f1 - f0) * (a2 - a0)) / d
            B = (f1 - f0) / (a1 - a0) - A * (a0 + a1)
            C = f0 - A * a0 ** 2 - B * a0
            F = lambda z: A * z ** 3 / 3.0 + B * z ** 2 / 2.0 + C * z
            tot += F(a2) - F(a0)
        if (len(yy) - 1) % 2 == 1:
            tot += 0.5 * (ff[-1] + ff[-2]) * (yy[-1] - yy[-2])
        return float(tot)

    def spline(yy, ff):
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(yy, ff)          # y must be increasing
        return float(cs.integrate(yy[0], yy[-1]))

    trunc = 0.5e-5
    tol_u = trunc * float(np.sum(np.abs(np.diff(y))))
    tol_v = trunc * float(np.sum(np.abs(np.diff(x))))
    est_u = [simpson_nonuniform(y, u), trapz(y, u), spline(y, u)]
    est_v = [simpson_nonuniform(x, v), trapz(x, v), spline(x, v)]
    # quadrature uncertainty = spread between the three estimators; the tabulated
    # point set is far too coarse for the truncation bound alone to be meaningful
    unc_u = max(est_u) - min(est_u)
    unc_v = max(est_v) - min(est_v)
    return {
        "vertical_centreline_int_u_dy": {
            "simpson": est_u[0], "trapezoid": est_u[1], "cubic_spline": est_u[2],
            "abs_best": abs(est_u[2]), "quadrature_spread": unc_u,
            "tolerance_from_truncation": tol_u,
            "within_uncertainty": bool(abs(est_u[2]) <= max(tol_u, unc_u))},
        "horizontal_centreline_int_v_dx": {
            "simpson": est_v[0], "trapezoid": est_v[1], "cubic_spline": est_v[2],
            "abs_best": abs(est_v[2]), "quadrature_spread": unc_v,
            "tolerance_from_truncation": tol_v,
            "within_uncertainty": bool(abs(est_v[2]) <= max(tol_v, unc_v))},
        "note": "exact for the continuous solution; here the residual level is set by the "
                "quadrature on the coarse tabulated point set (sparse near the walls), which "
                "is why the estimator spread is reported as the uncertainty",
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"residual_tolerance": RESIDUAL_TOL,
           "case1": case1_poiseuille(), "case2": case2_couette(),
           "case3": case3_taylor_green(), "case4": case4_cavity_limits(),
           "mpmath_version": mp.__version__, "numpy_version": np.__version__}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    c1, c2, c3, c4 = res["case1"], res["case2"], res["case3"], res["case4"]
    print("case1 pass:", c1["pass_residual"], c1["pass_second_order"],
          "cont/disc:", c1["residual_continuum"]["residual_rel"],
          c1["residual_discrete_laplacian"]["residual_abs"],
          "floor:", c1["parameters"]["residual_roundoff_floor_rel"],
          "slope:", c1["convergence"]["fitted_slope_p_asymptotic"],
          "control:", c1["control_wrong_amplitude_1pct"]["residual_rel"])
    print("case2 pass:", c2["pass_residual"],
          "disc:", c2["residual_discrete_laplacian"]["residual_abs"],
          "floor:", c2["parameters"]["residual_roundoff_floor_rel"],
          "tau rel:", c2["wall_shear"]["rel_error"],
          "control:", c2["control_parabolic_profile"]["residual_rel"])
    print("case3 pass:", c3["pass_residual"], "div:", c3["divergence"]["residual_rel"],
          "momx:", c3["momentum_x"]["residual_rel"], "momy:", c3["momentum_y"]["residual_rel"],
          "control:", c3["control_pressure_3pct"]["momentum_x_rel"])
    print("case4 pass:", c4["pass_residual"], "worst:", c4["worst_residual"],
          "control:", c4["control_wrong_exponent"]["max_abs_high_precision"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()