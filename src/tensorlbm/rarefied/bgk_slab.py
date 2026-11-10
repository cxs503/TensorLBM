"""Deterministic linearized-BGK discrete-velocity solver for a 1-D slab.

Geometry
--------
Gas between two infinite parallel plates at ``y = 0`` (moves with ``U_0``) and
``y = H`` (moves with ``U_H``).  Fully developed flow: every field depends on
``y`` only, so the velocity space is 2-D ``(v_x, v_y)`` (the ``v_z`` component
decouples exactly in the linearised problem and is integrated out).

Governing equation
------------------
Linearise ``f = f_M (1 + h)`` around a global Maxwellian at rest with
``R T = 1``.  With a body force ``g`` along ``x`` the linearised BGK equation is

.. math::

    v_y \\frac{\\partial h}{\\partial y}
        = \\frac{1}{\\tau}\\big( v_x \\tilde u(y) - h \\big),
    \\qquad \\tilde u = u + g\\tau ,

where the ``g\\tau`` term is the linearised body-force contribution
(``-g \\partial f/\\partial v_x`` at ``O(g)``), and

.. math::  u(y) = \\int v_x h f_M \\, d^2v ,\\qquad \\rho_0 = 1 .

Taking the first moment recovers the momentum balance
``d\\tau_{xy}/dy = \\rho_0 g``: force-driven Poiseuille flow.  ``g = 0`` and
moving walls is plane Couette flow.

Wall condition (Maxwell, TMAC = sigma)
--------------------------------------
For a wall whose inward normal points into the gas, on the incoming half of
velocity space::

    h_out(v) = (1 - sigma) h_inc(v_x, -v_y) + sigma [ v_x U_w + zeta_w ]

The specular part transfers no tangential momentum (and, for a *tangentially*
moving wall, no velocity shift either: ``R(v-U)+U`` keeps ``v_x``).  The diffuse
part carries the wall's tangential drift ``v_x U_w`` plus the Langmuir slip
constant ``zeta_w`` fixed by **zero net normal mass flux** through the wall::

    sum_{v_y>0} W v_y [ h_out - h_inc ] = 0 .

Numerical method
----------------
Exact integration of the transport along the characteristics.  In a cell of
width ``dy`` with a piecewise-constant source ``Q_i = v_x \\tilde u_i``::

    alpha = exp(-dy / (|v_y| tau)),
    h_i      = alpha h_{i-1} + (1 - alpha) Q_i        (v_y > 0, marching up)
    h_i      = alpha h_{i+1} + (1 - alpha) Q_i        (v_y < 0, marching down)

Velocity space uses a tensor-product Gauss-Hermite rule (exact for Maxwellian
moments of polynomials).  Because the problem is **linear**, the coupled
two-wall reflection is solved in closed form: for every ``v_y > 0`` node the pair
``(h at wall 0, h at wall H)`` obeys a 2x2 system in which the only global
coupling is the two scalars ``(zeta_0, zeta_H)``; those are obtained from the two
mass-flux constraints (a 2x2 linear solve per right-hand side).  The map
``u -> u`` (``(I - A) u = A g tau + B U``) is assembled by superposition and
solved directly, so there is **no Picard iteration** and no convergence-rate
dependence on Kn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .knudsen import (
    free_molecular_flow_rate,
    mean_free_path,
    ns_slip_flow_rate,
    tau_from_kn,
)


def _hermite(nv: int):
    """Gauss-Hermite nodes/weights for ``int e^{-x^2} f dx`` mapped to ``v = sqrt(2RT) x``."""
    x, w = np.polynomial.hermite.hermgauss(nv)
    return x, w


@dataclass
class SlabSolution:
    """Container for a converged slab solution."""

    kn: float
    sigma: float
    y: np.ndarray
    u: np.ndarray
    q: float
    flow: str
    H: float = 1.0
    tau: float = 0.0
    lam: float = 0.0
    g: float = 0.0
    u_walls: tuple = (0.0, 0.0)
    ny: int = 0
    nv: int = 0
    extra: dict = field(default_factory=dict)

    # -- derived quantities -------------------------------------------------
    def slip_length(self, bulk=(0.25, 0.75)) -> float:
        """Slip length from the bulk profile: ``l_s = u_extrap(0) / u'(0)``.

        For Couette (and to leading order for any slip flow) the bulk profile is
        linear; a least-squares line through the interior ``bulk`` fraction of
        the channel gives both the extrapolated wall velocity and the slope.
        """
        y, u = self.y, self.u
        m = (y >= bulk[0] * self.H) & (y <= bulk[1] * self.H)
        A = np.vstack([y[m], np.ones(m.sum())]).T
        slope, intercept = np.linalg.lstsq(A, u[m], rcond=None)[0]
        return intercept / slope

    def centre_velocity(self) -> float:
        i = int(np.argmin(np.abs(self.y - 0.5 * self.H)))
        return float(self.u[i])

    def mach(self) -> float:
        return float(np.max(np.abs(self.u)))


class LinearizedBGKSlab:
    """Force-driven Poiseuille and plane Couette flow for arbitrary Kn."""

    def __init__(self, kn: float, sigma: float = 1.0, ny: int = 128, nv: int = 40,
                 H: float = 1.0, RT: float = 1.0):
        if kn <= 0.0:
            raise ValueError("Kn must be > 0")
        if not (0.0 < sigma <= 1.0):
            raise ValueError("sigma (TMAC) must be in (0, 1]")
        if ny < 8 or nv < 8:
            raise ValueError("ny >= 8 and nv >= 8 required")
        self.kn = float(kn)
        self.sigma = float(sigma)
        self.H = float(H)
        self.RT = float(RT)
        self.ny = int(ny)
        self.nv = int(nv)
        self.dy = self.H / self.ny
        self.yc = (np.arange(self.ny) + 0.5) * self.dy
        self.lam = kn * H
        self.tau = tau_from_kn(kn, H, RT)
        self.nu = self.tau * RT

        x, w = _hermite(nv)
        v = math.sqrt(2.0 * RT) * x
        VX = np.repeat(v[:, None], nv, axis=1)
        VY = np.repeat(v[None, :], nv, axis=0)
        WT = np.outer(w, w) / math.pi          # sum(WT) = 1
        up = VY > 0.0
        self.vx_up = VX[up]
        self.vy_up = VY[up]
        self.W_up = WT[up]
        self.alpha = np.exp(-self.dy / (self.vy_up * self.tau))

        # precomputed powers for the separable source sums
        j = np.arange(self.ny)
        self._p_lo = self.alpha[:, None] ** j[None, :]              # alpha^j
        self._p_hi = self.alpha[:, None] ** (self.ny - 1 - j)[None, :]
        self._one_minus_alpha = (1.0 - self.alpha)[:, None]
        self._c = (1.0 - self.sigma) * self.alpha ** self.ny

        self._op_cache = None

    # ------------------------------------------------------------------ core
    def _sources(self, u_tilde: np.ndarray):
        """Separable source sums ``S0`` (into wall 0) and ``SH`` (into wall H)."""
        a1 = self._one_minus_alpha
        S0 = self.vx_up * ((self._p_lo * a1) @ u_tilde)
        SH = self.vx_up * ((self._p_hi * a1) @ u_tilde)
        return S0, SH

    def _wall_values(self, u_tilde, U0, UH, zeta0, zetaH):
        s = self.sigma
        S0, SH = self._sources(u_tilde)
        c = self._c
        den = 1.0 - c * c
        A1 = (1.0 - s) * S0 + s * (self.vx_up * U0 + zeta0)
        A2 = (1.0 - s) * SH + s * (self.vx_up * UH + zetaH)
        h0 = (c * A2 + A1) / den
        hH = (c * A1 + A2) / den
        hinc0 = self.alpha ** self.ny * hH + S0
        hincH = self.alpha * hH + (1.0 - self.alpha) * self.vx_up * u_tilde[-1]
        return h0, hH, hinc0, hincH, S0, SH

    def _flux_residuals(self, u_tilde, U0, UH, zeta0, zetaH):
        h0, hH, hinc0, hincH, _, _ = self._wall_values(
            u_tilde, U0, UH, zeta0, zetaH
        )
        r0 = float(np.sum(self.W_up * self.vy_up * (h0 - hinc0)))
        rH = float(np.sum(self.W_up * self.vy_up * (hincH - hH)))
        return r0, rH

    def _solve_zeta(self, u_tilde, U0, UH):
        """Langmuir slip constants from the two zero-net-mass-flux constraints."""
        base = np.array(self._flux_residuals(u_tilde, U0, UH, 0.0, 0.0))
        col0 = np.array(self._flux_residuals(u_tilde, U0, UH, 1.0, 0.0)) - base
        col1 = np.array(self._flux_residuals(u_tilde, U0, UH, 0.0, 1.0)) - base
        M = np.column_stack([col0, col1])
        rhs = -base
        zeta = np.linalg.solve(M, rhs)
        return float(zeta[0]), float(zeta[1])

    def _profile(self, u_tilde, U0, UH):
        """Velocity profile for a given driver field ``u_tilde``."""
        zeta0, zetaH = self._solve_zeta(u_tilde, U0, UH)
        h0, hH, _, _, _, _ = self._wall_values(u_tilde, U0, UH, zeta0, zetaH)
        a = self.alpha
        vx = self.vx_up
        # sequential march (ny steps, vectorised over velocity nodes)
        F_lo = np.zeros_like(h0)
        F_hi = np.zeros_like(h0)
        h_up = np.empty((self.ny, h0.size))
        for i in range(self.ny):
            F_lo = a * F_lo + u_tilde[i]
            h_up[i] = a ** (i + 1) * h0 + vx * (1.0 - a) * F_lo
        h_dn = np.empty((self.ny, h0.size))
        for i in range(self.ny - 1, -1, -1):
            F_hi = a * F_hi + u_tilde[i]
            h_dn[i] = a ** (self.ny - i) * hH + vx * (1.0 - a) * F_hi
        u = np.sum(self.W_up * vx * (h_up + h_dn), axis=1)
        return u, (zeta0, zetaH)

    # ------------------------------------------------------------- operators
    def _operators(self):
        """Assemble ``u = A u_tilde + B [U0, UH]`` by superposition."""
        if self._op_cache is not None:
            return self._op_cache
        ny = self.ny
        A = np.zeros((ny, ny))
        e = np.zeros(ny)
        for j in range(ny):
            e[:] = 0.0
            e[j] = 1.0
            A[:, j] = self._profile(e, 0.0, 0.0)[0]
        B = np.zeros((ny, 2))
        z = np.zeros(ny)
        B[:, 0] = self._profile(z, 1.0, 0.0)[0]
        B[:, 1] = self._profile(z, 0.0, 1.0)[0]
        self._op_cache = (A, B)
        return A, B

    # --------------------------------------------------------------- solvers
    def solve(self, g: float = 0.0, u_walls=(0.0, 0.0)) -> SlabSolution:
        """Solve the linearised problem (``g`` = body force, ``u_walls`` = (U0, UH))."""
        A, B = self._operators()
        U0, UH = float(u_walls[0]), float(u_walls[1])
        rhs = A @ (np.full(self.ny, g * self.tau)) + B @ np.array([U0, UH])
        lhs = np.eye(self.ny) - A
        u = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        u_tilde = u + g * self.tau
        u_check, _ = self._profile(u_tilde, U0, UH)
        resid = float(np.max(np.abs(u_check - u)) / max(np.max(np.abs(u)), 1e-300))
        q = float(np.sum(u) * self.dy)
        flow = "couette" if (g == 0.0 and (U0 != 0.0 or UH != 0.0)) else "poiseuille"
        return SlabSolution(
            kn=self.kn, sigma=self.sigma, y=self.yc.copy(), u=u, q=q, flow=flow,
            H=self.H, tau=self.tau, lam=self.lam, g=g, u_walls=(U0, UH),
            ny=self.ny, nv=self.nv,
            extra={"fixed_point_residual": resid,
                   "q_ns_slip": ns_slip_flow_rate(self.H, self.nu, g,
                                                 (2.0 - self.sigma) / self.sigma * self.lam),
                   "q_ns": ns_slip_flow_rate(self.H, self.nu, g, 0.0),
                   "q_fm": free_molecular_flow_rate(self.H, g, self.RT)},
        )

    def solve_poiseuille(self, g: float = 1.0) -> SlabSolution:
        return self.solve(g=g, u_walls=(0.0, 0.0))

    def solve_couette(self, U: float = 1.0) -> SlabSolution:
        return self.solve(g=0.0, u_walls=(0.0, U))

    # ------------------------------------------------------------ diagnostics
    def knudsen_layer_thickness(self, g: float = 1.0, bulk=(0.30, 0.70)) -> dict:
        """Thickness of the near-wall Knudsen layer of Poiseuille flow.

        The kinetic profile is compared with the parabola extrapolated from the
        channel interior (which is the NS+slip asymptote)::

            Delta(y) = u_kin(y) - u_fit(y),   u_fit from a quadratic LSQ on ``bulk``

        ``Delta`` decays over the Knudsen layer; two independent estimators are
        returned: the 1/e decay length of ``|Delta|`` fitted over the near-wall
        band ``0.02 H <= y <= 0.35 H`` and the distance where ``|Delta|`` drops
        below 1/e of its wall value.  Both are compared with ``lambda``, the only
        length scale of the layer.
        """
        sol = self.solve_poiseuille(g)
        y, u = sol.y, sol.u
        m = (y >= bulk[0] * self.H) & (y <= bulk[1] * self.H)
        phi = np.column_stack([np.ones(m.sum()), (y[m] - self.H / 2) ** 2])
        coef = np.linalg.lstsq(phi, u[m], rcond=None)[0]
        u_fit = coef[0] + coef[1] * (y - self.H / 2) ** 2
        d = u - u_fit
        band = (y >= 0.02 * self.H) & (y <= 0.35 * self.H) & (np.abs(d) > 0)
        yy, dd = y[band], np.abs(d[band])
        slope = np.polyfit(yy, np.log(dd), 1)[0]
        delta_fit = -1.0 / slope
        # 1/e crossing estimator
        d_wall = np.abs(d[0])
        idx = np.argmax(np.abs(d) < d_wall / math.e)
        delta_cross = y[idx] if idx > 0 else float("nan")
        return {
            "upstream": sol,
            "delta_fit": float(delta_fit),
            "delta_cross": float(delta_cross),
            "lambda": self.lam,
            "kn_layer_ratio_fit": float(delta_fit / self.lam),
            "kn_layer_ratio_cross": float(delta_cross / self.lam),
            "u_fit": u_fit,
            "deviation": d,
        }