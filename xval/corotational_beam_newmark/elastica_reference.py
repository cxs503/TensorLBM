"""External references for the corotational beam cross-validation.

Three independent references, none of them a finite-element solution:

1. ``tip_load_elastica(lam)`` -- exact planar *elastica* of a clamped cantilever
   with a transverse point load at the free end, in the inextensible
   Euler-Bernoulli limit.  Two independent numerical routes are computed and
   must agree before the value is used:

   * quadrature route.  With ``theta`` measured from the beam axis and ``s``
     the arc length, moment equilibrium about the section gives
     ``EI theta'(s) = P (x_L - x(s))``; differentiating once gives
     ``EI theta'' = -P cos(theta)``, and the first integral with the
     free-end condition ``theta'(L) = 0`` is

         ds = dtheta / sqrt(2 P (sin(theta0) - sin(theta)) / EI),

     so with ``I0(theta0) = int_0^theta0 dtheta/sqrt(sin(theta0)-sin(theta))``

         lam = P L^2 / EI = I0(theta0)^2 / 2,
         x_L/L = (1/I0) int cos(theta) dtheta/sqrt(...),
         y_L/L = (1/I0) int sin(theta) dtheta/sqrt(...).

   * ODE route: shooting on ``k0 = theta'(0)`` for
     ``EI theta'' = -P cos(theta)``, ``theta(0) = 0``, ``theta'(L) = 0``,
     followed by quadrature of ``x = int cos(theta) ds``, ``y = int sin(theta) ds``.

2. ``pure_moment_arc(phi)`` -- exact solution for a pure end moment ``M`` on an
   inextensible Euler-Bernoulli cantilever: constant curvature
   ``1/R = M/EI``, i.e. a circular arc, tip at
   ``(R sin phi, R (1 - cos phi))`` with ``phi = M L / EI``.

3. ``clamped_free_omega()`` lives in the solver module
   (``tensorlbm.corotational_beam.euler_bernoulli_frequency``).

Everything here is derived from the governing equations; no tabulated number
is transcribed from the OSS project or from the literature.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_ivp

__all__ = [
    "tip_load_elastica",
    "elastica_shape",
    "pure_moment_arc",
    "elastica_theta0",
]


def _I0(theta0: float) -> float:
    """int_0^theta0 dtheta / sqrt(sin(theta0) - sin(theta)).

    The integrand has an inverse-square-root singularity at theta0; the
    substitution psi = t^2 (psi = theta0 - theta) makes it regular.
    """
    if theta0 <= 0.0:
        return 0.0

    def integrand(t: float) -> float:
        psi = t * t
        num = 2.0 * t
        den = np.sqrt(max(np.sin(theta0) - np.sin(theta0 - psi), 1e-300))
        return num / den if t > 0.0 else 0.0

    val, _ = quad(integrand, 0.0, np.sqrt(theta0), limit=400, epsabs=1e-13, epsrel=1e-13)
    return val


def _moments(theta0: float) -> tuple[float, float]:
    """int cos / sqrt(...) and int sin / sqrt(...) over [0, theta0]."""
    def mk(fn):
        def integrand(t: float) -> float:
            psi = t * t
            th = theta0 - psi
            den = np.sqrt(max(np.sin(theta0) - np.sin(th), 1e-300))
            return 2.0 * t * fn(th) / den if t > 0.0 else 0.0

        v, _ = quad(integrand, 0.0, np.sqrt(theta0), limit=400, epsabs=1e-13, epsrel=1e-13)
        return v

    return mk(np.cos), mk(np.sin)


def elastica_theta0(lam: float, tol: float = 1e-14) -> float:
    """Solve ``I0(theta0)^2 / 2 = lam`` for the tip rotation theta0."""
    target = np.sqrt(2.0 * lam)
    lo, hi = 1e-9, np.radians(179.9)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _I0(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def _elastica_ode(lam: float, theta0_guess: float) -> dict:
    """Independent ODE/shooting route for the same elastica (EI = L = 1)."""
    P = lam  # EI = 1, L = 1

    def resid(k0):
        def rhs(s, y):
            th, k = y
            return [k, -P * np.cos(th)]

        sol = solve_ivp(rhs, (0.0, 1.0), [0.0, k0], rtol=1e-12, atol=1e-14, dense_output=True)
        th, k = sol.y[0], sol.y[1]
        return k[-1], sol

    # bracket the initial curvature
    lo, hi = 0.0, 6.0  # k(L) is increasing in k0 -> bracket then bisect
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if resid(mid)[0] > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-14:
            break
    k0 = 0.5 * (lo + hi)
    _, sol = resid(k0)
    s = np.linspace(0.0, 1.0, 200001)
    th = sol.sol(s)[0]
    x = np.trapezoid(np.cos(th), s)
    y = np.trapezoid(np.sin(th), s)
    return {"theta0": float(th[-1]), "x": float(x), "y": float(y), "k0": float(k0)}


def tip_load_elastica(lam: float) -> dict:
    """Reference for a clamped cantilever under a free-end transverse load.

    ``lam = P L^2 / EI``.  Returns dimensionless ``dy_over_L`` (deflection in the
    direction of the load, positive), ``dx_over_L`` (tip x-position, < 1 shows
    the axial shortening) and ``theta0`` (tip rotation magnitude).

    Both routes are always evaluated and their difference is returned in
    ``route_disagreement`` (max abs difference of the three quantities).
    """
    th0 = elastica_theta0(lam)
    I0 = _I0(th0)
    cx, sx = _moments(th0)
    quad_res = {
        "theta0": th0,
        "x": cx / I0,
        "y": sx / I0,
    }
    ode_res = _elastica_ode(lam, th0)
    dis = max(
        abs(quad_res["theta0"] - ode_res["theta0"]),
        abs(quad_res["x"] - ode_res["x"]),
        abs(quad_res["y"] - ode_res["y"]),
    )
    out = {
        "lambda": float(lam),
        "theta0": quad_res["theta0"],
        "dx_over_L": quad_res["x"],
        "dy_over_L": quad_res["y"],
        "lambda_from_quadrature": I0 * I0 / 2.0,
        "route_disagreement": float(dis),
        "ode_route": ode_res,
    }
    if dis > 1e-6:
        raise RuntimeError(f"elastica reference routes disagree by {dis:.3e}")
    return out


def elastica_shape(lam: float, n_pts: int = 401) -> dict:
    """Full inextensible elastica shape (x(s), y(s), theta(s)) for plotting.

    Uses the ODE route (EI = L = 1) with the tip-rotation from the quadrature
    root, i.e. the two routes are coupled only through the scalar ``theta0``.
    """
    P = lam
    th0 = elastica_theta0(lam)

    def rhs(s, y):
        return [y[1], -P * np.cos(y[0])]

    lo, hi = 0.0, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        sol = solve_ivp(rhs, (0.0, 1.0), [0.0, mid], rtol=1e-12, atol=1e-14)
        if sol.y[1][-1] > 0.0:
            hi = mid
        else:
            lo = mid
    sol = solve_ivp(rhs, (0.0, 1.0), [0.0, 0.5 * (lo + hi)], rtol=1e-13, atol=1e-15,
                    dense_output=True, t_eval=np.linspace(0.0, 1.0, n_pts))
    th = sol.y[0]
    s = sol.t
    x = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * np.diff(s))])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * np.diff(s))])
    return {"s": s, "x": x, "y": y, "theta": th, "theta0": float(th[-1])}


def pure_moment_arc(phi: float) -> dict:
    """Exact circular-arc solution for a pure end moment (EI = L = 1).

    ``phi = M L / EI`` is the total turning angle, ``R = 1/phi`` the radius.
    """
    R = 1.0 / phi
    return {
        "phi": float(phi),
        "radius": R,
        "x_over_L": R * np.sin(phi),
        "y_over_L": R * (1.0 - np.cos(phi)),
        "theta0": float(phi),
    }