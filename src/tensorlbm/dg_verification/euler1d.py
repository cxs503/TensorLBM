"""1-D compressible Euler DG plus the exact Riemann references.

Two physics variants share one solver:

* ``isothermal``  ``p = cs^2 rho`` with ``cs^2 = 1/3`` -- the closure the D2Q9
  lattice-BGK solver of this repository obeys, so DG and LBM can be
  cross-examined on *identical* physics (density jump 4:1).
* ``ideal gas``   ``p = (gamma - 1)(E - rho u^2 / 2)`` with ``gamma = 1.4`` --
  the classical Sod shock tube, referenced against its exact Riemann solution.

Numerics: nodal DG (Gauss-Lobatto), local Lax-Friedrichs (Rusanov) interface
flux, SSP-RK3 in time, optional Cockburn-Shu minmod/TVB limiter applied to the
conservative variables.  State layout

    U[c, i, j]   component ``c``, cell ``i``, node ``j``
"""

from __future__ import annotations

import numpy as np

from .element import RefElement, ref_element
from .limiter import tvb_limit
from .scalar import apply_axis

__all__ = [
    "EulerDG1D",
    "exact_sod",
    "exact_isothermal_riemann",
    "shock_position",
    "rarefaction_tail",
]


# ---------------------------------------------------------------------------
# exact references
# ---------------------------------------------------------------------------


def _fk(p: float, rho_k: float, p_k: float, gamma: float) -> float:
    """Toro's pressure function ``f_K(p)``."""
    if p > p_k:  # shock branch
        a = 2.0 / ((gamma + 1.0) * rho_k)
        b = (gamma - 1.0) / (gamma + 1.0) * p_k
        return (p - p_k) * np.sqrt(a / (p + b))
    c_k = np.sqrt(gamma * p_k / rho_k)  # rarefaction branch
    return 2.0 * c_k / (gamma - 1.0) * (
        (p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0
    )


def exact_sod(
    x: np.ndarray,
    t: float,
    x0: float = 0.5,
    gamma: float = 1.4,
    rho_l: float = 1.0,
    u_l: float = 0.0,
    p_l: float = 1.0,
    rho_r: float = 0.125,
    u_r: float = 0.0,
    p_r: float = 0.1,
) -> dict:
    """Exact Riemann solution of the ideal-gas Sod problem (Toro, ch. 4).

    Returns ``{"rho", "u", "p", "p_star", "u_star"}`` sampled at ``x`` for
    time ``t``.
    """
    x = np.asarray(x, dtype=np.float64)
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)

    lo, hi = 1e-12, max(p_l, p_r) * 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = _fk(mid, rho_l, p_l, gamma) + _fk(mid, rho_r, p_r, gamma) + (u_r - u_l)
        if f > 0.0:
            hi = mid
        else:
            lo = mid
    p_star = 0.5 * (lo + hi)
    u_star = 0.5 * (u_l + u_r) + 0.5 * (
        _fk(p_star, rho_r, p_r, gamma) - _fk(p_star, rho_l, p_l, gamma)
    )

    # star densities: shock jump relation if the wave is a shock, isentropic
    # (rarefaction) relation otherwise -- using the shock branch for the Sod
    # rarefaction is a classic error (rho*_L becomes 0.4472 instead of 0.4263).
    if p_star > p_l:
        rho_sl = rho_l * (p_star / p_l + (gamma - 1.0) / (gamma + 1.0)) / (
            (gamma - 1.0) / (gamma + 1.0) * p_star / p_l + 1.0
        )
    else:
        rho_sl = rho_l * (p_star / p_l) ** (1.0 / gamma)
    if p_star > p_r:
        rho_sr = rho_r * (p_star / p_r + (gamma - 1.0) / (gamma + 1.0)) / (
            (gamma - 1.0) / (gamma + 1.0) * p_star / p_r + 1.0
        )
    else:
        rho_sr = rho_r * (p_star / p_r) ** (1.0 / gamma)
    c_sl = c_l * (p_star / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
    c_sr = c_r * (p_star / p_r) ** ((gamma - 1.0) / (2.0 * gamma))

    s = (x - x0) / max(t, 1e-300)
    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)

    for i, si in enumerate(s):
        if si <= u_star:  # left of the contact surface
            if p_star > p_l:  # left shock
                s_shock = u_l - c_l * np.sqrt(
                    (gamma + 1.0) / (2.0 * gamma) * p_star / p_l
                    + (gamma - 1.0) / (2.0 * gamma)
                )
                if si < s_shock:
                    rho[i], u[i], p[i] = rho_l, u_l, p_l
                else:
                    rho[i], u[i], p[i] = rho_sl, u_star, p_star
            else:  # left rarefaction fan
                if si < u_l - c_l:
                    rho[i], u[i], p[i] = rho_l, u_l, p_l
                elif si > u_star - c_sl:
                    rho[i], u[i], p[i] = rho_sl, u_star, p_star
                else:
                    fac = 2.0 / (gamma + 1.0) + (gamma - 1.0) / (
                        (gamma + 1.0) * c_l
                    ) * (u_l - si)
                    rho[i] = rho_l * fac ** (2.0 / (gamma - 1.0))
                    u[i] = 2.0 / (gamma + 1.0) * (c_l + 0.5 * (gamma - 1.0) * u_l + si)
                    p[i] = p_l * fac ** (2.0 * gamma / (gamma - 1.0))
        else:  # right of the contact surface
            if p_star > p_r:  # right shock
                s_shock = u_r + c_r * np.sqrt(
                    (gamma + 1.0) / (2.0 * gamma) * p_star / p_r
                    + (gamma - 1.0) / (2.0 * gamma)
                )
                if si > s_shock:
                    rho[i], u[i], p[i] = rho_r, u_r, p_r
                else:
                    rho[i], u[i], p[i] = rho_sr, u_star, p_star
            else:  # right rarefaction fan
                if si > u_r + c_r:
                    rho[i], u[i], p[i] = rho_r, u_r, p_r
                elif si < u_star + c_sr:
                    rho[i], u[i], p[i] = rho_sr, u_star, p_star
                else:
                    fac = 2.0 / (gamma + 1.0) - (gamma - 1.0) / (
                        (gamma + 1.0) * c_r
                    ) * (u_r - si)
                    rho[i] = rho_r * fac ** (2.0 / (gamma - 1.0))
                    u[i] = 2.0 / (gamma + 1.0) * (
                        -c_r + 0.5 * (gamma - 1.0) * u_r + si
                    )
                    p[i] = p_r * fac ** (2.0 * gamma / (gamma - 1.0))

    return {
        "rho": rho,
        "u": u,
        "p": p,
        "p_star": float(p_star),
        "u_star": float(u_star),
        "rho_star_l": float(rho_sl),
        "rho_star_r": float(rho_sr),
        "c_l": float(c_l),
        "c_r": float(c_r),
    }


def exact_isothermal_riemann(
    x: np.ndarray,
    t: float,
    x0: float = 0.5,
    rho_l: float = 1.0,
    rho_r: float = 0.25,
    cs: float = 1.0 / np.sqrt(3.0),
) -> dict:
    """Exact solution of the isothermal Riemann problem (``p = cs^2 rho``).

    Both states are at rest (``u = 0``).  The right wave is a shock whose density
    ratio ``r = rho* / rho_R`` solves the isothermal Rankine-Hugoniot relation

        ln(rho_L / (r rho_R)) = (r - 1) / sqrt(r) ,

    the left wave is a rarefaction fan spanning ``[-cs, u* - cs]``.  The shock
    speed is ``W = cs sqrt(r)``.  This is the same closure/initial data as the
    D2Q9 benchmark ``benchmarks/verified/sod_shock_tube`` so shock positions can
    be cross-examined without a physics mismatch.
    """
    x = np.asarray(x, dtype=np.float64)

    def resid(r: float) -> float:
        return np.log(rho_l / (r * rho_r)) - (r - 1.0) / np.sqrt(r)

    lo, hi = 1.0 + 1e-12, 50.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    rho_star = r * rho_r
    u_star = cs * (r - 1.0) / np.sqrt(r)
    w_shock = cs * np.sqrt(r)

    s = (x - x0) / max(t, 1e-300)
    rho = np.empty_like(x)
    u = np.empty_like(x)
    for i, si in enumerate(s):
        if si < -cs:
            rho[i], u[i] = rho_l, 0.0
        elif si < u_star - cs:
            u[i] = si + cs
            rho[i] = rho_l * np.exp(-(si + cs) / cs)
        elif si < w_shock:
            rho[i], u[i] = rho_star, u_star
        else:
            rho[i], u[i] = rho_r, 0.0
    return {
        "rho": rho,
        "u": u,
        "rho_star": float(rho_star),
        "u_star": float(u_star),
        "w_shock": float(w_shock),
        "rarefaction_head": float(-cs),
        "rarefaction_tail": float(u_star - cs),
    }


def rarefaction_tail(rho_l: float, rho_star: float, cs: float) -> float:
    """Isothermal rarefaction self-consistency: ``u* = cs ln(rho_L / rho*)``."""
    return cs * np.log(rho_l / rho_star)


# ---------------------------------------------------------------------------
# DG solver
# ---------------------------------------------------------------------------


class EulerDG1D:
    """Nodal DG solver for the 1-D (isothermal or ideal-gas) Euler equations."""

    def __init__(
        self,
        degree: int = 1,
        n_cells: int = 200,
        length: float = 1.0,
        gamma: float = 1.4,
        isothermal: bool = False,
        cs2: float = 1.0 / 3.0,
        use_limiter: bool = True,
        tvb_M: float = 0.0,
    ) -> None:
        self.degree = int(degree)
        self.elem: RefElement = ref_element(degree)
        self.n_cells = int(n_cells)
        self.length = float(length)
        self.dx = self.length / self.n_cells
        self.gamma = float(gamma)
        self.isothermal = bool(isothermal)
        self.cs2 = float(cs2)
        self.use_limiter = bool(use_limiter)
        self.tvb_M = float(tvb_M)
        self.n_comp = 2 if isothermal else 3

        self._n_positivity_fix = 0
        xc = (np.arange(self.n_cells) + 0.5) * self.dx
        self.x = xc[:, None] + 0.5 * self.dx * self.elem.nodes[None, :]  # (N, n)

    # -- state helpers ------------------------------------------------------
    def from_primitive(self, rho, u, p) -> np.ndarray:
        rho = np.asarray(rho, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        U = np.zeros((self.n_comp,) + rho.shape, dtype=np.float64)
        U[0] = rho
        U[1] = rho * u
        if not self.isothermal:
            U[2] = p / (self.gamma - 1.0) + 0.5 * rho * u * u
        return U

    def primitive(self, U: np.ndarray) -> tuple:
        rho = U[0]
        u = U[1] / rho
        if self.isothermal:
            p = self.cs2 * rho
        else:
            p = (self.gamma - 1.0) * (U[2] - 0.5 * rho * u * u)
        return rho, u, p

    def wave_speed(self, U: np.ndarray) -> np.ndarray:
        rho, u, p = self.primitive(U)
        if self.isothermal:
            c = np.sqrt(self.cs2)
        else:
            c = np.sqrt(self.gamma * p / np.maximum(rho, 1e-300))
        return np.abs(u) + c

    def flux(self, U: np.ndarray) -> np.ndarray:
        rho, u, p = self.primitive(U)
        F = np.zeros_like(U)
        F[0] = U[1]
        F[1] = U[1] * u + p
        if not self.isothermal:
            F[2] = u * (U[2] + p)
        return F

    # -- numerics -----------------------------------------------------------
    def limit(self, U: np.ndarray) -> np.ndarray:
        if not self.use_limiter:
            return U
        out = np.empty_like(U)
        for c in range(self.n_comp):
            out[c] = tvb_limit(U[c], self.elem, self.dx, self.tvb_M, axis=-2)
        return out

    def rhs(self, U: np.ndarray) -> np.ndarray:
        U = self.limit(U)
        F = self.flux(U)
        # volume term:  -int l_k' F  (every cell, node axis last)
        # weak form: M du/dt = +int F phi' - [F* phi]  (surface term with '-')
        dvol = apply_axis(F, self.elem.Gr, -1)
        # interface flux (local Lax-Friedrichs), face i between cell i and i+1
        F_L = F[..., -1]  # (C, N) right trace of cell i
        F_R = np.roll(F[..., 0], -1, axis=-1)  # left trace of cell i+1
        U_L = U[..., -1]
        U_R = np.roll(U[..., 0], -1, axis=-1)
        a = self.wave_speed(U)  # (N, n)
        a_face = np.maximum(a[..., -1], np.roll(a[..., 0], -1, axis=-1))  # (N,)
        F_star = 0.5 * (F_L + F_R) - 0.5 * a_face[None, :] * (U_R - U_L)
        F_star_left = np.roll(F_star, 1, axis=-1)
        surf = F_star_left[..., None] * self.elem.S[:, 0] + F_star[..., None] * self.elem.S[:, 1]
        return apply_axis(dvol - surf, self.elem.Minv, -1) * (2.0 / self.dx)

    def cfl_dt(self, cfl: float = 0.2) -> float:
        a = float(np.abs(self.wave_speed(self.U if hasattr(self, "U") else self._U)).max())
        return cfl * self.dx / ((2 * self.degree + 1) * max(a, 1e-12))

    def set_state(self, U: np.ndarray) -> None:
        self._U = np.array(U, dtype=np.float64, copy=True)

    @property
    def U(self) -> np.ndarray:
        return self._U

    def positivity_fix(self, U: np.ndarray) -> tuple[np.ndarray, int]:
        """Revert cells whose nodal density or pressure went non-positive.

        Admissible-state repair in the spirit of Zhang-Shu: a cell that violates
        ``rho > 0`` or ``p > 0`` at any node is replaced by its cell average (the
        first-order fallback), which is a valid physical state as long as the
        previously accepted state was.  This is what lets the *unlimited* DG run
        through the strong Sod jump; without it the pressure goes negative and
        the run turns into NaN within one step.
        """
        rho, u, p = self.primitive(U)
        bad = np.any((rho <= 0.0) | (p <= 0.0), axis=-1)
        if not np.any(bad):
            return U, 0
        out = U.copy()
        means = U.mean(axis=-1, keepdims=True)
        out[:, bad, :] = means[:, bad, :]
        return out, int(bad.sum())

    def step(self, dt: float, cfl: float = 0.2) -> None:
        U0 = self._U

        def f(U):
            return self.rhs(U)

        k1 = U0 + dt * f(U0)
        k2 = 0.75 * U0 + 0.25 * (k1 + dt * f(k1))
        Un = (1.0 / 3.0) * U0 + (2.0 / 3.0) * (k2 + dt * f(k2))
        Un = self.limit(Un)
        Un, nfixed = self.positivity_fix(Un)
        self._n_positivity_fix += nfixed
        self._U = Un

    def advance(self, t_end: float, cfl: float = 0.2, max_steps: int = 10_000_000) -> dict:
        """Advance to ``t_end`` with an adaptive (CFL-driven) explicit step."""
        t = 0.0
        steps = 0
        while t < t_end - 1e-15 and steps < max_steps:
            dt = min(self.cfl_dt(cfl), t_end - t)
            self.step(dt, cfl)
            t += dt
            steps += 1
        return {
            "t": t,
            "steps": steps,
            "dt_last": dt,
            "positivity_fixes": int(self._n_positivity_fix),
        }

    # -- diagnostics --------------------------------------------------------
    def cell_means(self, U: np.ndarray | None = None) -> np.ndarray:
        U = self._U if U is None else U
        return U.mean(axis=-1)

    def cell_centers(self) -> np.ndarray:
        return (np.arange(self.n_cells) + 0.5) * self.dx

    def node_x_flat(self) -> np.ndarray:
        return self.x.reshape(-1)

    def nodal_field(self, U: np.ndarray | None = None, comp: int = 0) -> np.ndarray:
        U = self._U if U is None else U
        return U[comp].reshape(-1)


def shock_position(x: np.ndarray, rho: np.ndarray) -> float:
    """Locate a right-moving shock from sampled ``rho`` by maximal gradient.

    Uses a parabolic refinement around the largest cell-to-cell density jump so
    the estimator is second-order accurate in the sampling interval.
    """
    x = np.asarray(x, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    d = np.diff(rho)
    i = int(np.argmax(np.abs(d)))
    xm = 0.5 * (x[i] + x[i + 1])
    if 0 < i < len(d) - 1:
        # parabolic refinement of |slope| peak
        f0, f1, f2 = abs(d[i - 1]), abs(d[i]), abs(d[i + 1])
        denom = f0 - 2.0 * f1 + f2
        if abs(denom) > 1e-300:
            shift = 0.5 * (f0 - f2) / denom
            shift = float(np.clip(shift, -0.5, 0.5))
            xm += shift * (x[i + 1] - x[i])
    return float(xm)