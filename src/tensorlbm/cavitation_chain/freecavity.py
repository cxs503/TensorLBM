"""Closed (constant-pressure) sheet-cavity solver -- free boundary by least squares.

Mission P151W1-N04.  The sheet cavity on the suction side of a 2-D section is
modelled as a *constant-pressure* free streamline (Riabouchinsky closure): on
the cavity boundary the surface speed is fixed by Bernoulli,

.. math::   q_c = \\sqrt{1 + \\sigma}, \\qquad \\sigma = (p_\\infty - p_v)/(\\tfrac12 \\rho U^2)

with ``p_inf = 0``, ``U = 1``.  The cavity *shape* (thickness at ``n_cav - 1``
interior stations) and its *length* ``L`` are the unknowns of a free-boundary
problem; they are found here by a damped Gauss-Newton (trust-region least
squares) iteration on the normal-velocity residual of the cavity panels, which
is exactly zero for the true cavity streamline.

Why least squares instead of the classic transpiration sweep
------------------------------------------------------------
The transpiration ("move the wall by ``v_n dt``") sweep used by
:mod:`tensorlbm.cavitation_free_streamline` is a fixed-point map whose gain
grows with the cavity length; on NACA0015 it either crawls or oscillates.  The
least-squares form uses the *same* flow engine and residuals but computes the
shape increment from a Jacobian (finite differences, ``n_cav`` extra flow
solves per iteration, each an ``O(N^3)`` dense solve with ``N ~ 2 n_foil``), so
the iteration converges quadratically and needs no hand-tuned relaxation.

Declared approximations
-----------------------
* inviscid, irrotational, incompressible; no boundary layer, so no viscous
  ``-C_p,min`` limitation -- the predicted inception number is the *inviscid*
  one and is therefore an upper bound on the water-tunnel value;
* closed (Riabouchinsky) closure: no wake/vapour transport, so this module
  yields ``L/c`` and the section force but *not* the cloud-shedding frequency
  (that comes from the re-entrant-jet model in
  :mod:`tensorlbm.cavitation_chain.sheet_cloud`);
* cavity detachment fixed at the leading edge (``x_detach = 0``);
* cavity thickness measured along the chord normal (geometry update only; the
  flow solve is exact for the panel geometry used).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..cavitation_free_streamline import (
    _surface_y,
    flow_solve,
    foil_contour,
    panel_geometry,
)

__all__ = [
    "FreeCavityResult",
    "cavity_sweep",
    "solve_free_cavity",
    "wetted_state",
]


@dataclass
class FreeCavityResult:
    """Free-boundary sheet-cavity solution at fixed ``(section, alpha, sigma)``."""

    section: str
    alpha_deg: float
    sigma: float
    l_cav: float
    delta_cav: float
    cl: float
    cd: float
    q_c: float
    residual_norm: float
    converged: bool
    iterations: int
    n_cav: int
    x_cav: list[float] = field(default_factory=list)
    thickness: list[float] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "section": self.section,
            "alpha_deg": self.alpha_deg,
            "sigma": self.sigma,
            "l_cav": self.l_cav,
            "delta_cav": self.delta_cav,
            "cl": self.cl,
            "cd": self.cd,
            "q_c": self.q_c,
            "residual_norm": self.residual_norm,
            "converged": self.converged,
            "iterations": self.iterations,
            "n_cav": self.n_cav,
            "x_cav": list(self.x_cav),
            "thickness": list(self.thickness),
            "notes": self.notes,
        }


def wetted_state(section: str = "naca0015", alpha_deg: float = 5.0, n_foil: int = 200):
    """Wetted (non-cavitating) panel solution: ``(geom, FlowState)``."""
    m = n_foil // 2
    xl = np.linspace(1.0, 0.0, m + 1)
    lower = np.stack([xl, _surface_y(section, xl, upper=False)], axis=1)
    xu = np.linspace(0.0, 1.0, m + 1)
    upper = np.stack([xu, _surface_y(section, xu, upper=True)], axis=1)
    nodes = np.vstack([lower[:-1], upper])
    geom = panel_geometry(nodes)
    return geom, flow_solve(geom, alpha_deg)


def _cavity_geom(section: str, alpha_deg: float, n_foil: int, n_cav: int, l_cav: float, thick: np.ndarray):
    nodes = foil_contour(
        section, alpha_deg, n_foil=n_foil, n_cav=n_cav, l_cav=l_cav, thickness=thick
    )
    return panel_geometry(nodes)


def _residual(section, alpha_deg, n_foil, n_cav, l_cav, thick, sigma, ret_state=False):
    """Normal-velocity residual on the cavity panels (``n_cav`` values)."""
    geom = _cavity_geom(section, alpha_deg, n_foil, n_cav, l_cav, thick)
    q_c = math.sqrt(1.0 + sigma)
    i0 = n_foil // 2
    i1 = i0 + n_cav
    cav = np.zeros(geom["mid"].shape[0], dtype=bool)
    cav[i0:i1] = True
    st = flow_solve(geom, alpha_deg, q_c=q_c, cavity_mask=cav)
    res = np.asarray(st.n_residual[i0:i1], dtype=float)
    if ret_state:
        return res, st, geom, i0, i1
    return res


def _pack(thick, L):
    return np.concatenate([np.asarray(thick, dtype=float)[1:-1], [float(L)]])


def solve_free_cavity(
    section: str = "naca0015",
    alpha_deg: float = 5.0,
    sigma: float = 1.0,
    n_foil: int = 200,
    n_cav: int = 20,
    max_iter: int = 60,
    tol: float = 5.0e-4,
    l_init: float | None = None,
    thick_init: np.ndarray | None = None,
) -> FreeCavityResult:
    """Solve the closed constant-pressure cavity (damped Gauss-Newton / LM).

    Unknowns: ``n_cav - 1`` interior cavity thicknesses (the end stations are
    pinned to zero -- detachment and closure both lie on the foil surface) plus
    the cavity length ``L/c``.  Equations: zero normal velocity on the ``n_cav``
    cavity panels (the constant-pressure condition itself is imposed exactly as
    a Dirichlet condition on the panel speed, ``q_t = q_c``).  The system is
    square; the increment is a Levenberg-Marquardt step with bound clipping and
    backtracking, which keeps the iteration stable in the long-cavity regime
    where the plain Newton step overshoots.
    """
    if sigma <= 0.0:
        msg = "sigma must be positive"
        raise ValueError(msg)
    i0 = n_foil // 2
    n_cav = int(n_cav)
    xi = np.linspace(0.0, 1.0, n_cav + 1)
    if l_init is None:
        L = float(np.clip(0.55 / (1.0 + sigma) ** 1.4, 0.02, 0.9))
    else:
        L = float(np.clip(l_init, 0.01, 0.95))
    if thick_init is None:
        thick = 0.08 * L * np.sin(math.pi * xi) ** 0.8
    else:
        thick = np.array(thick_init, dtype=float).copy()
        thick = np.interp(xi, np.linspace(0.0, 1.0, thick.size), thick)
    thick[0] = 0.0
    thick[-1] = 0.0
    x = _pack(thick, L)
    best = None
    it = 0
    mu = 1.0e-3
    for it in range(1, max_iter + 1):
        thick = _unpack(x)
        res, st, geom, _, i1 = _residual(
            section, alpha_deg, n_foil, n_cav, x[-1], thick, sigma, ret_state=True
        )
        rnorm = float(np.linalg.norm(res))
        if best is None or rnorm < best[0]:
            best = [rnorm, x.copy(), st, geom, i1]
        if rnorm < tol:
            break
        jac = np.zeros((n_cav, n_cav))
        dp = 5.0e-4
        for j in range(n_cav):
            xp = x.copy()
            xp[j] += dp
            if j < n_cav - 1 and _unpack(xp) is None:
                xp[j] = x[j] - dp
            rp = _residual(section, alpha_deg, n_foil, n_cav, xp[-1], _unpack(xp), sigma)
            jac[:, j] = (rp - res) / (xp[j] - x[j])
        # Levenberg-Marquardt with an adaptive damping
        step = None
        for _ in range(8):
            gram = jac.T @ jac
            damp = np.diag(np.diag(gram)) * mu + 1.0e-12 * np.eye(n_cav)
            try:
                cand = np.linalg.solve(gram + damp, -(jac.T @ res))
            except np.linalg.LinAlgError:  # pragma: no cover
                cand = np.linalg.lstsq(gram + damp, -(jac.T @ res), rcond=None)[0]
            # trust region: the cavity increments must stay small
            scale = max(1.0, np.max(np.abs(cand[:-1])) / 0.05, abs(cand[-1]) / 0.10)
            cand = cand / scale
            xt = x + cand
            xt[-1] = float(np.clip(xt[-1], 0.01, 0.95))
            tt = _unpack(xt)
            if tt is not None:
                rt = _residual(section, alpha_deg, n_foil, n_cav, xt[-1], tt, sigma)
                if np.linalg.norm(rt) < rnorm:
                    x = xt
                    mu = max(1.0e-6, 0.3 * mu)
                    step = cand
                    break
            mu = min(1.0e3, 10.0 * mu)
        if step is None:
            break
    rnorm, x, st, geom, i1 = best
    thick = _unpack(x)
    q_c = math.sqrt(1.0 + sigma)
    x_cav = np.array(geom["mid"][i0:i1, 0], dtype=float)
    return FreeCavityResult(
        section=section,
        alpha_deg=alpha_deg,
        sigma=sigma,
        l_cav=float(x[-1]),
        delta_cav=float(np.max(thick)),
        cl=float(st.cl),
        cd=float(st.cd),
        q_c=q_c,
        residual_norm=float(rnorm),
        converged=bool(rnorm < tol),
        iterations=it,
        n_cav=n_cav,
        x_cav=x_cav.tolist(),
        thickness=np.asarray(thick, dtype=float).tolist(),
        notes="closed (Riabouchinsky) constant-pressure cavity; inviscid; LE detachment",
    )


def cavity_sweep(
    section: str = "naca0015",
    alpha_deg: float = 5.0,
    sigmas=(2.0, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5),
    n_foil: int = 200,
    n_cav: int = 20,
    max_iter: int = 60,
    tol: float = 5.0e-4,
) -> list[FreeCavityResult]:
    """Continuation sweep in ``sigma`` (high -> low), warm-starting the shape.

    Continuation is not cosmetic: the long-cavity solutions at low ``sigma`` are
    unreachable from a cold parabolic start, but are reached in a few LM steps
    from the neighbouring (slightly shorter) cavity.
    """
    out: list[FreeCavityResult] = []
    prev = None
    for sig in sorted(sigmas, reverse=True):
        kwargs = {}
        if prev is not None:
            kwargs["l_init"] = prev.l_cav
            kwargs["thick_init"] = np.asarray(prev.thickness)
        r = solve_free_cavity(
            section, alpha_deg, sig, n_foil=n_foil, n_cav=n_cav,
            max_iter=max_iter, tol=tol, **kwargs,
        )
        out.append(r)
        prev = r
    out.sort(key=lambda r: r.sigma)
    return out


def _unpack(x):
    """Expand the free unknowns back into the ``n_cav + 1`` thickness array."""
    n_cav = x.size
    thick = np.zeros(n_cav + 1)
    thick[1:-1] = x[:-1]
    if np.any(thick < -1e-6):
        return None
    thick = np.clip(thick, 0.0, 0.6)
    thick[0] = 0.0
    thick[-1] = 0.0
    return thick
