"""Free-surface interface tracking: 2-D vertical-plane Navier--Stokes + VOF.

Why this module exists
----------------------
The repository's historic free-surface "verification" face
(:mod:`tensorlbm.free_surface_verification`) compared a collapsing liquid
column's front against ``X* = 1 + t*`` and *called that curve* Martin & Moyce
(1952).  It is not the experiment: with aspect ratio ``H/a = 2`` the measured
front reaches ``X = 3.67`` at ``T = 2.96`` whereas the ideal (Ritter, shallow
water) front would be ``X = 2T = 5.92`` -- a deep-column collapse is *not* a
shallow-water problem, and the old gate ``RMSE <= 2.5`` passes any roughly
linear curve.  A dedicated audit of four existing LBM phase-field models
(``benchmarks/pending/dam_break_2d_multimodel``) found 4/4 FAIL against the real
table.  This module supplies the missing *independent* solver instead of
another gate tweak.

What is implemented here
------------------------
A self-contained staggered (MAC) incompressible two-phase solver with a
Volume-Of-Fluid interface:

* **Interface tracking** -- conservative flux-form VOF transport
  ``dF/dt + div(u F) = 0`` with MUSCL/superbee interface reconstruction.  The
  projection below enforces ``div(u) = 0`` *discretely*, so total liquid volume
  is conserved by construction; a conservative clip/redistribute repair and an
  optional global volume correction are kept for long runs and are *measured*
  (never silently applied) in :class:`MassLedger`.
* **Pressure projection** -- variable-density Poisson equation
  ``div((1/rho) grad p) = div(u*)/dt`` solved with a geometric multigrid
  V-cycle (red--black Gauss--Seidel smoother); the 1000:1 water/air density
  jump is therefore not approximated by a constant-coefficient operator.
* **Surface tension** -- continuum-surface-force (CSF) model on a smoothed
  colour function; :func:`laplace_pressure` reproduces ``dp = sigma/R`` (the
  analytic anchor of the force model).
* **Wall boundary conditions** -- free-slip or no-slip floor/side walls, open
  (``p = 0``) top, half-cell consistent staggered placement.
* **Mass ledger** -- :class:`MassLedger` records the liquid volume every step,
  so ``|M(t)-M(0)|/M(0)`` is a measured quantity.

Verification anchors shipped with the module
--------------------------------------------
* :func:`martin_moyce_table` -- 14 digitalised points of Martin & Moyce (1952,
  Phil. Trans. R. Soc. A **244**, 312) on the *correct* axis
  ``T = t sqrt(2 g / a)``, ``X = x_front/a`` (the convention used by the Lethe
  ``dam-break-2d`` post-processing ``time*sqrt(2g/L1)`` / ``x/L1``; provenance
  in ``benchmarks/pending/dam_break_sc/result.json``).  The last tabulated
  point is ``T = 2.96``; nothing here extrapolates beyond it.
* :class:`DamBreakMoyce` -- ``H/a = 2`` collapse driver, floor-row front
  tracking, RMSE against those 14 points.
* :func:`advect_interface_convergence` -- L2 convergence of the interface
  transport against an RK4-computed flow map on a divergence-free vortex field
  (the order-of-accuracy anchor of the tracking scheme).
* :func:`laplace_pressure` -- static droplet Laplace law.

Units
-----
Non-dimensionalised with the dam width ``a``: ``g = 1``, ``a = 1``, ``H = 2``,
velocity unit ``sqrt(g a)``, time unit ``sqrt(a/g)``.  The M&M axis then reads
``T = sqrt(2) t`` and ``X = x``.

Evidence level: REAL -- every number reported by the solver comes from the
discrete equations above; no tabulated or fitted value is returned as a
simulation output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                  # tiny dense solves are far faster here
    from scipy.linalg import solve as _dense_solve
except Exception:                     # pragma: no cover - scipy is a hard dep
    _dense_solve = np.linalg.solve

__all__ = [
    "martin_moyce_table",
    "VOFConfig",
    "FreeSurfaceVOF2D",
    "MassLedger",
    "DamBreakMoyce",
    "VariableCoeffPoisson2D",
    "advect_interface_convergence",
    "laplace_pressure",
    "fit_convergence_order",
    "wigley_halfbreadth",
]

_EPS = 1.0e-30


# ---------------------------------------------------------------------------
# literature anchor
# ---------------------------------------------------------------------------

def martin_moyce_table() -> Dict[str, np.ndarray]:
    """Martin & Moyce (1952) collapse front for aspect ratio ``H/a = 2``.

    Digitised table as distributed with the Lethe ``dam-break-2d`` example
    (``time_list = t sqrt(2g/L1)``, ``x_list = x/L1``).  The ``sqrt(2)`` axis
    factor is exactly what separates this table from the retired
    ``T = t sqrt(g/a)`` axis; mixing the two is the documented root cause of the
    retracted "+50 ... +82 %" claims (see
    ``benchmarks/pending/dam_break_sc/result.json``,
    ``convention_and_table_fix``).

    Returns
    -------
    dict with ``T`` (dimensionless time) and ``X`` (front position ``x/a``).
    """
    T = np.array([0.00, 0.41, 0.84, 1.19, 1.43, 1.63, 1.82, 1.97,
                  2.20, 2.32, 2.50, 2.64, 2.82, 2.96], dtype=float)
    X = np.array([1.00, 1.11, 1.23, 1.44, 1.67, 1.89, 2.11, 2.33,
                  2.56, 2.78, 3.00, 3.22, 3.44, 3.67], dtype=float)
    return {"T": T, "X": X}


def wigley_halfbreadth(x: np.ndarray, beam: float = 0.1) -> np.ndarray:
    """Half-breadth of the ITTC Wigley hull, ``y(x) = (B/2)(1-(2x/L)^2)``.

    ``x`` is measured from midship, reference length ``L = 1``.  Declared here so
    that hull *geometry* has one definition shared with
    :mod:`tensorlbm.free_surface_waves`.
    """
    xi = 2.0 * np.asarray(x, dtype=float)
    return 0.5 * beam * np.clip(1.0 - xi ** 2, 0.0, None)


def fit_convergence_order(h: Sequence[float], err: Sequence[float]) -> Dict[str, float]:
    """Least-squares slope of ``log(err)`` vs ``log(h)`` plus fit quality."""
    h = np.asarray(h, float)
    e = np.maximum(np.asarray(err, float), _EPS)
    A = np.vstack([np.log(h), np.ones_like(h)]).T
    sol, *_ = np.linalg.lstsq(A, np.log(e), rcond=None)
    pred = A @ sol
    ss_tot = float(np.sum((np.log(e) - np.log(e).mean()) ** 2))
    ss_res = float(np.sum((np.log(e) - pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return {"order": float(sol[0]), "r2": float(max(r2, 0.0)), "constant": float(sol[1])}


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass
class VOFConfig:
    """Discretisation and physics knobs of :class:`FreeSurfaceVOF2D`."""

    nx: int = 120
    ny: int = 60
    lx: float = 6.0
    ly: float = 3.0
    g: float = 1.0
    rho_l: float = 1.0
    rho_g: float = 1.0e-3
    nu_l: float = 1.0e-3
    nu_g: float = 1.0e-2
    sigma: float = 0.0
    cfl: float = 0.25
    wall: str = "free-slip"
    mass_correction: bool = False
    csf_smoothing: int = 3
    kappa_limit: float = 1.0
    max_poisson_cycles: int = 30
    poisson_tol: float = 1.0e-9
    smoother_sweeps: int = 2

    def __post_init__(self) -> None:
        if self.nx < 8 or self.ny < 8:
            raise ValueError("grid must be at least 8 x 8")
        if not 0.0 < self.rho_g < self.rho_l:
            raise ValueError("need 0 < rho_g < rho_l")
        if self.wall not in ("free-slip", "no-slip"):
            raise ValueError("wall must be 'free-slip' or 'no-slip'")
        if not 0.0 < self.cfl <= 0.5:
            raise ValueError("cfl must lie in (0, 0.5]")
        if self.nu_l <= 0.0 or self.nu_g <= 0.0:
            raise ValueError("viscosities must be positive")

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny


# ---------------------------------------------------------------------------
# geometric multigrid for div(beta grad p) = f
# ---------------------------------------------------------------------------

class VariableCoeffPoisson2D:
    """``div(beta grad p) = f`` on a cell-centred grid, geometric MG V-cycle.

    ``beta`` lives on faces: ``bxp`` has shape ``(ny, nx+1)`` (x-faces per cell
    row) and ``byp`` shape ``(ny+1, nx)`` (y-faces per cell column).  Boundary
    conditions are encoded in the coefficients:

    * ``bxp[:, 0] = bxp[:, -1] = 0`` -> impermeable side walls (Neumann)
    * ``byp[0, :] = 0``              -> impermeable floor (Neumann)
    * ``byp[ny, :] > 0``             -> open top, ``p = 0`` outside (Dirichlet)
    """

    def __init__(self, bxp: np.ndarray, byp: np.ndarray, dx: float, dy: float,
                 min_size: int = 8):
        self.levels: List[dict] = []
        self._build(np.array(bxp, float, copy=True), np.array(byp, float, copy=True),
                    dx, dy, min_size)

    @staticmethod
    def _pack(bxp: np.ndarray, byp: np.ndarray, dx: float, dy: float) -> dict:
        ny, nx = bxp.shape[0], bxp.shape[1] - 1
        if byp.shape != (ny + 1, nx):
            raise ValueError(f"face coefficient shapes disagree: {bxp.shape} / {byp.shape}")
        aE = bxp[:, 1:] / dx ** 2
        aW = bxp[:, :-1] / dx ** 2
        aN = byp[1:, :] / dy ** 2
        aS = byp[:-1, :] / dy ** 2
        diag = aE + aW + aN + aS
        diag = np.where(diag > 0.0, diag, 1.0)
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        colour = (ii + jj) % 2
        return {"nx": nx, "ny": ny, "aE": aE, "aW": aW, "aN": aN, "aS": aS,
                "diag": diag, "dx": dx, "dy": dy, "colour": colour}

    def _build(self, bxp, byp, dx, dy, min_size):
        self.levels.append(self._pack(bxp, byp, dx, dy))
        while True:
            ny, nx = bxp.shape[0], bxp.shape[1] - 1
            if min(ny, nx) <= min_size or nx % 2 or ny % 2:
                break
            bxp = self._restrict_faces_x(bxp)
            byp = self._restrict_faces_y(byp)
            dx *= 2.0
            dy *= 2.0
            self.levels.append(self._pack(bxp, byp, dx, dy))

    @staticmethod
    def _restrict_faces_x(bxp: np.ndarray) -> np.ndarray:
        """Coarse x-face coefficients: harmonic (series) mean of the fine pair.

        Arithmetic averaging of the conductance across a 1000:1 density jump
        makes the coarse operator inconsistent with the fine one and the
        V-cycle diverges; the harmonic mean is the series connection of the two
        half-cell resistances and keeps the correction bounded.
        """
        ny, nxf = bxp.shape
        out = np.zeros((ny // 2, nxf // 2 + 1))
        out[:, 0] = 0.0
        out[:, -1] = 0.0
        if nxf > 3:
            b1 = bxp[:, 1:-2:2]
            b2 = bxp[:, 2:-1:2]
            hm = 2.0 * b1 * b2 / np.maximum(b1 + b2, 1.0e-300)
            out[:, 1:-1] = 0.5 * (hm[0::2, :] + hm[1::2, :])
        return out

    @staticmethod
    def _restrict_faces_y(byp: np.ndarray) -> np.ndarray:
        """Coarse y-face coefficients: harmonic mean of the fine pair."""
        nyf, nx = byp.shape
        out = np.zeros((nyf // 2 + 1, nx // 2))
        out[0, :] = 0.0
        out[-1, :] = byp[-1, 0::2] * 0.0 + 0.5 * (byp[-1, 0::2] + byp[-1, 1::2])
        if nyf > 3:
            b1 = byp[1:-2:2, :]
            b2 = byp[2:-1:2, :]
            hm = 2.0 * b1 * b2 / np.maximum(b1 + b2, 1.0e-300)
            out[1:-1, :] = 0.5 * (hm[:, 0::2] + hm[:, 1::2])
        return out

    @staticmethod
    def apply(lvl: dict, p: np.ndarray) -> np.ndarray:
        pp = np.zeros((lvl["ny"] + 2, lvl["nx"] + 2))
        pp[1:-1, 1:-1] = p
        return (lvl["aE"] * pp[1:-1, 2:] + lvl["aW"] * pp[1:-1, :-2]
                + lvl["aN"] * pp[2:, 1:-1] + lvl["aS"] * pp[:-2, 1:-1]
                - lvl["diag"] * p)

    def _smooth(self, lvl: dict, p: np.ndarray, f: np.ndarray, sweeps: int,
                omega: float = 1.6) -> None:
        """Red--black Gauss--Seidel with SOR over-relaxation (``omega``)."""
        ny, nx = lvl["ny"], lvl["nx"]
        pp = np.zeros((ny + 2, nx + 2))
        colour = lvl["colour"]
        for _ in range(sweeps):
            for c in (0, 1):
                pp[1:-1, 1:-1] = p
                num = (lvl["aE"] * pp[1:-1, 2:] + lvl["aW"] * pp[1:-1, :-2]
                       + lvl["aN"] * pp[2:, 1:-1] + lvl["aS"] * pp[:-2, 1:-1] - f)
                mask = colour == c
                p[mask] = p[mask] + omega * (num[mask] / lvl["diag"][mask] - p[mask])

    @staticmethod
    def _restrict(r: np.ndarray) -> np.ndarray:
        rp = np.zeros((r.shape[0] + 2, r.shape[1] + 2))
        rp[1:-1, 1:-1] = r
        return (4.0 * rp[1:-1:2, 1:-1:2]
                + 2.0 * (rp[:-2:2, 1:-1:2] + rp[2::2, 1:-1:2]
                         + rp[1:-1:2, :-2:2] + rp[1:-1:2, 2::2])
                + (rp[:-2:2, :-2:2] + rp[:-2:2, 2::2]
                   + rp[2::2, :-2:2] + rp[2::2, 2::2])) / 16.0

    @staticmethod
    def _prolong(e: np.ndarray) -> np.ndarray:
        ey = np.concatenate([e, e[-1:, :]], axis=0)
        ex = np.concatenate([e, e[:, -1:]], axis=1)
        eyx = np.concatenate([ey, ey[:, -1:]], axis=1)
        out = np.zeros((2 * e.shape[0], 2 * e.shape[1]))
        out[0::2, 0::2] = e
        out[1::2, 0::2] = 0.5 * (ey[:-1, :] + ey[1:, :])
        out[0::2, 1::2] = 0.5 * (ex[:, :-1] + ex[:, 1:])
        out[1::2, 1::2] = 0.25 * (eyx[:-1, :-1] + eyx[:-1, 1:]
                                  + eyx[1:, :-1] + eyx[1:, 1:])
        return out

    @staticmethod
    def _dense(lvl: dict) -> np.ndarray:
        ny, nx = lvl["ny"], lvl["nx"]
        n = ny * nx
        m = np.zeros((n, n))
        idx = np.arange(n).reshape(ny, nx)
        m[idx, idx] = -lvl["diag"]
        m[idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)] += lvl["aE"][:, :-1].reshape(-1)
        m[idx[:, 1:].reshape(-1), idx[:, :-1].reshape(-1)] += lvl["aW"][:, 1:].reshape(-1)
        m[idx[:-1, :].reshape(-1), idx[1:, :].reshape(-1)] += lvl["aN"][:-1, :].reshape(-1)
        m[idx[1:, :].reshape(-1), idx[:-1, :].reshape(-1)] += lvl["aS"][1:, :].reshape(-1)
        return m

    def _vcycle(self, k: int, p: np.ndarray, f: np.ndarray, sweeps: int,
                damping: float = 0.8) -> None:
        lvl = self.levels[k]
        self._smooth(lvl, p, f, sweeps)
        if k == len(self.levels) - 1:
            if lvl["ny"] * lvl["nx"] <= 600:
                p[:] = _dense_solve(self._dense(lvl), f.reshape(-1)).reshape(p.shape)
            else:
                # coarsest grid still too large for a dense factorisation ->
                # many Gauss-Seidel sweeps (cheap in 2-D)
                self._smooth(lvl, p, f, 60)
            return
        r = f - self.apply(lvl, p)
        rc = self._restrict(r)
        ec = np.zeros(rc.shape)
        self._vcycle(k + 1, ec, rc, sweeps, damping)
        p += damping * self._prolong(ec)

    def solve(self, p: np.ndarray, f: np.ndarray, tol: float = 1.0e-9,
              max_cycles: int = 30, sweeps: int = 2) -> Dict[str, float]:
        """Iterate in place; returns residual diagnostics."""
        scale = max(float(np.abs(f).max()), _EPS)
        res = float(np.abs(f).max())
        for cyc in range(1, max_cycles + 1):
            self._vcycle(0, p, f, sweeps)
            res = float(np.abs(f - self.apply(self.levels[0], p)).max())
            if res <= tol * scale:
                return {"cycles": float(cyc), "residual": res, "relative": res / scale}
        return {"cycles": float(max_cycles), "residual": res, "relative": res / scale}


# ---------------------------------------------------------------------------
# mass ledger
# ---------------------------------------------------------------------------

@dataclass
class MassLedger:
    """Liquid volume bookkeeping, ``M(t) = sum(F) dx dy``."""

    volume0: float = 0.0
    volumes: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    repair_clip: float = 0.0
    repair_global: float = 0.0

    def record(self, t: float, volume: float) -> None:
        self.times.append(float(t))
        self.volumes.append(float(volume))

    def relative_drift(self) -> np.ndarray:
        v = np.asarray(self.volumes, float)
        if self.volume0 == 0.0:
            return np.zeros_like(v)
        return np.abs(v - self.volume0) / abs(self.volume0)

    def max_drift(self) -> float:
        d = self.relative_drift()
        return float(d.max()) if d.size else 0.0

    def as_dict(self) -> dict:
        return {
            "volume0": self.volume0,
            "max_relative_drift": self.max_drift(),
            "final_relative_drift": float(self.relative_drift()[-1]) if self.volumes else 0.0,
            "repair_clip": self.repair_clip,
            "repair_global": self.repair_global,
            "n_samples": len(self.volumes),
        }


# ---------------------------------------------------------------------------
# low-level VOF transport helpers
# ---------------------------------------------------------------------------

def _vanleer(dl: np.ndarray, dr: np.ndarray) -> np.ndarray:
    """van Leer harmonic slope limiter (TVD, second order on smooth data)."""
    same = dl * dr > 0.0
    den = dl + dr
    out = np.zeros_like(dl)
    np.divide(2.0 * dl * dr, den, out=out, where=same & (np.abs(den) > _EPS))
    return out


def _superbee(dl: np.ndarray, dr: np.ndarray) -> np.ndarray:
    """Superbee slope limiter (compressive -- keeps VOF interfaces sharp)."""
    same = dl * dr > 0.0
    out = np.zeros_like(dl)
    a = np.minimum(2.0 * np.abs(dl), np.abs(dr))
    b = np.minimum(np.abs(dl), 2.0 * np.abs(dr))
    out[same] = np.sign(dl[same]) * np.maximum(a[same], b[same])
    return out


def muscl_derivative(a: np.ndarray, vel: np.ndarray, h: float, axis: int) -> np.ndarray:
    """Slope-limited second-order upwind derivative of ``a`` along ``axis``."""
    if axis == 1:
        pad = np.concatenate([a[:, :1], a, a[:, -1:]], axis=1)
        dl = a - pad[:, :-2]
        dr = pad[:, 2:] - a
        s = _vanleer(dl, dr)
        sp = np.concatenate([s[:, :1], s, s[:, -1:]], axis=1)
        fwd = (a - pad[:, :-2]) / h + 0.5 * (s - sp[:, :-2]) / h
        bwd = (pad[:, 2:] - a) / h - 0.5 * (sp[:, 2:] - s) / h
    else:
        pad = np.concatenate([a[:1, :], a, a[-1:, :]], axis=0)
        dl = a - pad[:-2, :]
        dr = pad[2:, :] - a
        s = _vanleer(dl, dr)
        sp = np.concatenate([s[:1, :], s, s[-1:, :]], axis=0)
        fwd = (a - pad[:-2, :]) / h + 0.5 * (s - sp[:-2, :]) / h
        bwd = (pad[2:, :] - a) / h - 0.5 * (sp[2:, :] - s) / h
    return np.where(vel > 0.0, fwd, bwd)


def vof_face_values_1d(f: np.ndarray, vf: np.ndarray, axis: int,
                       periodic: bool = False) -> np.ndarray:
    """MUSCL face values of ``f`` upwinded by the face velocity ``vf``."""
    if axis == 1:
        pad = (np.concatenate([f[:, -1:], f, f[:, :1]], axis=1) if periodic
               else np.concatenate([f[:, :1], f, f[:, -1:]], axis=1))
        dl = pad[:, 1:-1] - pad[:, :-2]
        dr = pad[:, 2:] - pad[:, 1:-1]
        s = _superbee(dl, dr)
        fL = np.clip(pad[:, 1:-1] + 0.5 * s, 0.0, 1.0)
        fR = np.clip(pad[:, 1:-1] - 0.5 * s, 0.0, 1.0)
        out = np.empty_like(vf)
        out[:, 0] = (np.where(vf[:, 0] > 0.0, fL[:, -1], fR[:, 0]) if periodic
                     else f[:, 0])
        out[:, -1] = f[:, -1]
        out[:, 1:-1] = np.where(vf[:, 1:-1] > 0.0, fL[:, :-1], fR[:, 1:])
        return out
    pad = (np.concatenate([f[-1:, :], f, f[:1, :]], axis=0) if periodic
           else np.concatenate([f[:1, :], f, f[-1:, :]], axis=0))
    dl = pad[1:-1, :] - pad[:-2, :]
    dr = pad[2:, :] - pad[1:-1, :]
    s = _superbee(dl, dr)
    fL = np.clip(pad[1:-1, :] + 0.5 * s, 0.0, 1.0)
    fR = np.clip(pad[1:-1, :] - 0.5 * s, 0.0, 1.0)
    out = np.empty_like(vf)
    out[0, :] = (np.where(vf[0, :] > 0.0, fL[-1, :], fR[0, :]) if periodic
                 else f[0, :])
    out[-1, :] = f[-1, :]
    out[1:-1, :] = np.where(vf[1:-1, :] > 0.0, fL[:-1, :], fR[1:, :])
    return out


def vof_sweep_x(f: np.ndarray, uf: np.ndarray, dt: float, dx: float,
                periodic: bool = False) -> np.ndarray:
    """Conservative x-sweep ``f -= dt/dx (u f)|_faces``."""
    flux = uf * vof_face_values_1d(f, uf, axis=1, periodic=periodic)
    return f - dt * (flux[:, 1:] - flux[:, :-1]) / dx


def vof_sweep_y(f: np.ndarray, vf: np.ndarray, dt: float, dy: float,
                periodic: bool = False) -> np.ndarray:
    """Conservative y-sweep ``f -= dt/dy (v f)|_faces``."""
    flux = vf * vof_face_values_1d(f, vf, axis=0, periodic=periodic)
    return f - dt * (flux[1:, :] - flux[:-1, :]) / dy


def clip_redistribute(f: np.ndarray, iters: int = 6) -> Tuple[np.ndarray, float]:
    """Clamp ``f`` into ``[0, 1]`` and push the excess into neighbour capacity.

    Returns the repaired field and the *net volume change* the repair caused
    (``sum(f_in) - sum(f_out)``): volume moved between cells is conserved, so a
    non-zero return value is volume the neighbourhood could not absorb.  It is
    logged in :class:`MassLedger` -- never silently dropped.
    """
    f_in = np.array(f, dtype=float, copy=True)
    f = f_in.copy()
    for _ in range(iters):
        excess = np.clip(f - 1.0, 0.0, None) - np.clip(-f, 0.0, None)
        if np.abs(excess).sum() <= 1.0e-14:
            break
        f = np.clip(f, 0.0, 1.0)
        cap = np.where(excess > 0.0, 1.0 - f, np.where(excess < 0.0, f, 0.0))
        pad_c = np.zeros((f.shape[0] + 2, f.shape[1] + 2))
        pad_e = np.zeros_like(pad_c)
        pad_c[1:-1, 1:-1] = cap
        pad_e[1:-1, 1:-1] = excess
        w = pad_c[1:-1, 2:] + pad_c[1:-1, :-2] + pad_c[2:, 1:-1] + pad_c[:-2, 1:-1]
        e = pad_e[1:-1, 2:] + pad_e[1:-1, :-2] + pad_e[2:, 1:-1] + pad_e[:-2, 1:-1]
        share = np.zeros_like(f)
        np.divide(e, w, out=share, where=w > 1.0e-12)
        f = np.clip(f + share, 0.0, 1.0)
    return f, float(abs(f_in.sum() - f.sum()))


def _to_faces_x(a: np.ndarray) -> np.ndarray:
    """Cell-centred scalar -> x-face array of shape ``(ny, nx+1)``."""
    out = np.empty((a.shape[0], a.shape[1] + 1))
    out[:, 1:-1] = 0.5 * (a[:, :-1] + a[:, 1:])
    out[:, 0] = a[:, 0]
    out[:, -1] = a[:, -1]
    return out


def _to_faces_y(a: np.ndarray) -> np.ndarray:
    """Cell-centred scalar -> y-face array of shape ``(ny+1, nx)``."""
    out = np.empty((a.shape[0] + 1, a.shape[1]))
    out[1:-1, :] = 0.5 * (a[:-1, :] + a[1:, :])
    out[0, :] = a[0, :]
    out[-1, :] = a[-1, :]
    return out


# ---------------------------------------------------------------------------
# main solver
# ---------------------------------------------------------------------------

class FreeSurfaceVOF2D:
    """Staggered two-phase Navier--Stokes solver with a VOF free surface.

    Layout (cell-centred scalars, staggered velocities)::

        F, p   : (ny, nx)          cell centres
        u      : (ny, nx+1)        u[j, i] at (i dx, (j + 1/2) dy)
        v      : (ny+1, nx)        v[j, i] at ((i + 1/2) dx, j dy)
    """

    def __init__(self, cfg: VOFConfig):
        self.cfg = cfg
        self.nx, self.ny = cfg.nx, cfg.ny
        self.dx, self.dy = cfg.dx, cfg.dy
        self.F = np.zeros((self.ny, self.nx))
        self.p = np.zeros((self.ny, self.nx))
        self.u = np.zeros((self.ny, self.nx + 1))
        self.v = np.zeros((self.ny + 1, self.nx))
        self.t = 0.0
        self.n_steps = 0
        self.ledger = MassLedger(volume0=0.0)
        self.poisson_info: Dict[str, float] = {}
        self.div_max = 0.0
        self.dt_last = 0.0

    # -- derived fields ----------------------------------------------------
    @property
    def rho(self) -> np.ndarray:
        c = self.cfg
        return c.rho_g + (c.rho_l - c.rho_g) * np.clip(self.F, 0.0, 1.0)

    @property
    def mu(self) -> np.ndarray:
        c = self.cfg
        return (c.rho_g * c.nu_g
                + (c.rho_l * c.nu_l - c.rho_g * c.nu_g) * np.clip(self.F, 0.0, 1.0))

    def volume(self) -> float:
        return float(np.clip(self.F, 0.0, 1.0).sum() * self.dx * self.dy)

    def liquid_mass(self) -> float:
        """Liquid mass = ``rho_l * volume`` (liquid fraction, not mixture)."""
        return self.cfg.rho_l * self.volume()

    # -- initialisation ----------------------------------------------------
    def set_liquid(self, rects: Sequence[Tuple[float, float, float, float]]) -> None:
        """Fill ``[(x0, x1, y0, y1), ...]`` rectangles with liquid (``F = 1``)."""
        x = (np.arange(self.nx) + 0.5) * self.dx
        y = (np.arange(self.ny) + 0.5) * self.dy
        X, Y = np.meshgrid(x, y)
        self.F = np.zeros((self.ny, self.nx))
        for x0, x1, y0, y1 in rects:
            self.F[(X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)] = 1.0
        self.t = 0.0
        self.n_steps = 0
        self.u[:] = 0.0
        self.v[:] = 0.0
        self.ledger = MassLedger(volume0=self.volume())
        self._hydrostatic_pressure()

    def set_droplet(self, radius: float, centre: Optional[Tuple[float, float]] = None,
                    width: Optional[float] = None) -> None:
        """Circular liquid patch with a one-cell tanh interface (capillary tests)."""
        cx, cy = centre if centre else (0.5 * self.cfg.lx, 0.5 * self.cfg.ly)
        x = (np.arange(self.nx) + 0.5) * self.dx
        y = (np.arange(self.ny) + 0.5) * self.dy
        X, Y = np.meshgrid(x, y)
        r = np.hypot(X - cx, Y - cy)
        w = width if width else 0.5 * max(self.dx, self.dy)
        self.F = np.clip(0.5 * (1.0 - np.tanh((r - radius) / w)), 0.0, 1.0)
        self.t = 0.0
        self.n_steps = 0
        self.u[:] = 0.0
        self.v[:] = 0.0
        self.p[:] = 0.0
        self.ledger = MassLedger(volume0=self.volume())

    def _hydrostatic_pressure(self) -> None:
        y = (np.arange(self.ny) + 0.5) * self.dy
        wet = self.F > 0.5
        self.p = np.zeros((self.ny, self.nx))
        for i in range(self.nx):
            rows = np.flatnonzero(wet[:, i])
            if rows.size:
                hcol = (rows.max() + 1) * self.dy
                col = self.cfg.rho_l * self.cfg.g * (hcol - y)
                self.p[:, i] = np.where(wet[:, i], np.clip(col, 0.0, None), 0.0)

    # -- boundary conditions -----------------------------------------------
    def _apply_velocity_bc(self) -> None:
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[0, :] = 0.0
        if self.cfg.wall == "no-slip":
            self.u[0, :] = 0.0
            self.v[:, 0] = 0.0
            self.v[:, -1] = 0.0

    # -- discrete operators -------------------------------------------------
    def _lap(self, a: np.ndarray, axis: int) -> np.ndarray:
        h = self.dx if axis == 1 else self.dy
        out = np.zeros_like(a)
        if axis == 1:
            out[:, 1:-1] = (a[:, 2:] - 2.0 * a[:, 1:-1] + a[:, :-2]) / h ** 2
            out[:, 0] = (a[:, 1] - 2.0 * a[:, 0]) / h ** 2
            out[:, -1] = (a[:, -2] - 2.0 * a[:, -1]) / h ** 2
            return out
        ap = np.empty((a.shape[0] + 2, a.shape[1]))
        ap[1:-1, :] = a
        ap[0, :] = a[0, :] if self.cfg.wall == "free-slip" else -a[0, :]
        ap[-1, :] = a[-1, :]
        out[1:-1, :] = (ap[2:-2, :] - 2.0 * ap[1:-3, :] + ap[:-4, :]) / h ** 2
        out[0, :] = (ap[2, :] - 2.0 * ap[1, :] + ap[0, :]) / h ** 2
        out[-1, :] = (ap[-1, :] - ap[-2, :]) / h ** 2
        return out

    @staticmethod
    def _smooth3(a: np.ndarray) -> np.ndarray:
        p = np.zeros((a.shape[0] + 2, a.shape[1] + 2))
        p[1:-1, 1:-1] = a
        out = (4.0 * p[1:-1, 1:-1]
               + 2.0 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
               + (p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:])) / 16.0
        return out

    def _csf_acceleration(self) -> Tuple[np.ndarray, np.ndarray]:
        """CSF surface-tension acceleration on the staggered velocities."""
        cfg = self.cfg
        f = np.clip(self.F, 0.0, 1.0)
        fs = f
        for _ in range(max(cfg.csf_smoothing, 0)):
            fs = self._smooth3(fs)
        gy_fx = np.zeros_like(f)
        gy_fy = np.zeros_like(f)
        gy_fx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * self.dx)
        gy_fx[:, 0] = (f[:, 1] - f[:, 0]) / self.dx
        gy_fx[:, -1] = (f[:, -1] - f[:, -2]) / self.dx
        gy_fy[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * self.dy)
        gy_fy[0, :] = (f[1, :] - f[0, :]) / self.dy
        gy_fy[-1, :] = (f[-1, :] - f[-2, :]) / self.dy
        nx_f = np.zeros_like(f)
        ny_f = np.zeros_like(f)
        nx_f[:, 1:-1] = (fs[:, 2:] - fs[:, :-2]) / (2.0 * self.dx)
        ny_f[1:-1, :] = (fs[2:, :] - fs[:-2, :]) / (2.0 * self.dy)
        mag = np.sqrt(nx_f ** 2 + ny_f ** 2) + 1.0e-12
        nxu, nyu = nx_f / mag, ny_f / mag
        # curvature = div(n)
        kx = np.zeros_like(f)
        ky = np.zeros_like(f)
        kx[:, 1:-1] = (nxu[:, 2:] - nxu[:, :-2]) / (2.0 * self.dx)
        ky[1:-1, :] = (nyu[2:, :] - nyu[:-2, :]) / (2.0 * self.dy)
        # n = grad(F)/|grad(F)| points *into* the liquid, so the (positive for a
        # convex liquid body) curvature carries a minus sign; getting this sign
        # wrong turns surface tension anti-diffusive and the run diverges.
        kappa = -(kx + ky)
        # The curvature of a 1--2 cell interface is noise dominated: bound it by
        # the finest curvature the grid can represent (|kappa| <= 1/h).  Without
        # the bound the CSF term amplifies reconstruction noise and the run
        # diverges, which is a discretisation artefact, not physics.
        kappa = np.clip(kappa, -cfg.kappa_limit / max(self.dx, self.dy),
                        cfg.kappa_limit / max(self.dx, self.dy))
        fx = cfg.sigma * kappa * gy_fx
        fy = cfg.sigma * kappa * gy_fy
        # The CSF force is divided by the *mean* two-fluid density rather than
        # the local mixture density: on the gas side the local density would
        # amplify the capillary acceleration by rho_l/rho_g and destabilise the
        # air layer (classic bracketed-CSF practice).
        rho_st = 0.5 * (cfg.rho_l + cfg.rho_g)
        acc_u = _to_faces_x(fx) / rho_st
        acc_v = _to_faces_y(fy) / rho_st
        return acc_u, acc_v

    # -- time step ----------------------------------------------------------
    def _stable_dt(self) -> float:
        cfg = self.cfg
        h = min(self.dx, self.dy)
        umax = max(np.abs(self.u).max(), np.abs(self.v).max())
        depth = float(np.clip(self.F, 0.0, 1.0).sum(axis=0).max() * self.dy)
        cwave = np.sqrt(cfg.g * max(depth, 1.0e-8)) if cfg.g > 0.0 else 0.0
        dt = cfg.cfl * h / max(umax + cwave, 1.0e-12)
        nu_max = max(cfg.nu_l, cfg.nu_g) * 1.0
        dt = min(dt, 0.2 * h ** 2 / nu_max)
        if cfg.sigma > 0.0:
            dt = min(dt, 0.5 * np.sqrt(cfg.rho_l * h ** 3 / max(cfg.sigma, 1.0e-30)))
        return float(dt)

    def step(self, dt: Optional[float] = None) -> Dict[str, float]:
        cfg = self.cfg
        rho = self.rho
        mu = self.mu
        rho_u = _to_faces_x(rho)
        rho_v = _to_faces_y(rho)
        mu_u = _to_faces_x(mu)
        mu_v = _to_faces_y(mu)
        if dt is None:
            dt = self._stable_dt()

        u, v = self.u, self.v
        # cross-plane interpolations to the staggered locations
        vc = 0.5 * (v[:-1, :] + v[1:, :])
        uc = 0.5 * (u[:, :-1] + u[:, 1:])
        v_u = np.empty_like(u)
        v_u[:, 1:-1] = 0.5 * (vc[:, :-1] + vc[:, 1:])
        v_u[:, 0] = vc[:, 0]
        v_u[:, -1] = vc[:, -1]
        u_v = np.empty_like(v)
        u_v[1:-1, :] = 0.5 * (uc[:-1, :] + uc[1:, :])
        u_v[0, :] = uc[0, :]
        u_v[-1, :] = uc[-1, :]

        adv_u = u * muscl_derivative(u, u, self.dx, 1) \
            + v_u * muscl_derivative(u, v_u, self.dy, 0)
        adv_v = u_v * muscl_derivative(v, u_v, self.dx, 1) \
            + v * muscl_derivative(v, v, self.dy, 0)
        visc_u = (mu_u / rho_u) * (self._lap(u, 1) + self._lap(u, 0))
        visc_v = (mu_v / rho_v) * (self._lap(v, 0) + self._lap(v, 1))

        u_star = u + dt * (-adv_u + visc_u)
        v_star = v + dt * (-adv_v + visc_v - cfg.g)
        if cfg.sigma > 0.0:
            au, av = self._csf_acceleration()
            u_star = u_star + dt * au
            v_star = v_star + dt * av

        u_star[:, 0] = 0.0
        u_star[:, -1] = 0.0
        v_star[0, :] = 0.0
        if cfg.wall == "no-slip":
            u_star[0, :] = 0.0
            v_star[:, 0] = 0.0
            v_star[:, -1] = 0.0

        div = (u_star[:, 1:] - u_star[:, :-1]) / self.dx \
            + (v_star[1:, :] - v_star[:-1, :]) / self.dy

        bxp = 1.0 / rho_u
        bxp[:, 0] = 0.0
        bxp[:, -1] = 0.0
        byp = 1.0 / rho_v
        byp[0, :] = 0.0
        byp[-1, :] = 1.0 / cfg.rho_g
        poisson = VariableCoeffPoisson2D(bxp, byp, self.dx, self.dy)
        info = poisson.solve(self.p, div / dt, tol=cfg.poisson_tol,
                             max_cycles=cfg.max_poisson_cycles,
                             sweeps=cfg.smoother_sweeps)
        self.poisson_info = info

        p_x = np.concatenate([self.p[:, :1], self.p, np.zeros((self.ny, 1))], axis=1)
        p_y = np.concatenate([self.p[:1, :], self.p, np.zeros((1, self.nx))], axis=0)
        self.u = u_star - dt * bxp * (p_x[:, 1:] - p_x[:, :-1]) / self.dx
        self.v = v_star - dt * byp * (p_y[1:, :] - p_y[:-1, :]) / self.dy
        self._apply_velocity_bc()

        div_new = (self.u[:, 1:] - self.u[:, :-1]) / self.dx \
            + (self.v[1:, :] - self.v[:-1, :]) / self.dy
        self.div_max = float(np.abs(div_new).max())

        # VOF transport with the projected (discretely divergence-free) velocity
        F = vof_sweep_x(self.F, self.u, dt, self.dx)
        F = vof_sweep_y(F, self.v, dt, self.dy)
        F, leftover = clip_redistribute(F)
        self.ledger.repair_clip += leftover * self.dx * self.dy
        if cfg.mass_correction and self.ledger.volume0 > 0.0:
            target = self.ledger.volume0
            cur = float(F.sum() * self.dx * self.dy)
            excess = target - cur
            cap = np.where(excess > 0.0, 1.0 - F, F)
            total_cap = float(cap.sum() * self.dx * self.dy)
            if total_cap > 1.0e-16:
                F = np.clip(F + excess / total_cap * cap, 0.0, 1.0)
                self.ledger.repair_global += excess
        self.F = F

        self.t += dt
        self.dt_last = dt
        self.n_steps += 1
        self.ledger.record(self.t, self.volume())
        return {"dt": dt, "div_max": self.div_max, "poisson": info}

    def run(self, t_end: float, callback=None, max_steps: int = 2_000_000) -> None:
        """Integrate to ``t_end`` (adaptive ``dt``); ``callback(self)`` per step."""
        while self.t < t_end - 1.0e-12 and self.n_steps < max_steps:
            dt = min(self._stable_dt(), t_end - self.t)
            self.step(dt)
            if callback is not None:
                callback(self)

    # -- measurements -------------------------------------------------------
    def front_position(self, threshold: float = 0.5, rows: int = 2) -> float:
        """Toe of the surge: rightmost wet cell within ``rows`` of the floor."""
        cols = np.flatnonzero(np.any(self.F[:rows, :] > threshold, axis=0))
        if cols.size == 0:
            return 0.0
        return float((cols.max() + 1) * self.dx)

    def column_height(self) -> np.ndarray:
        """Depth-integrated liquid height per column (``int F dy``)."""
        return np.clip(self.F, 0.0, 1.0).sum(axis=0) * self.dy

    def interface_height(self, threshold: float = 0.5) -> np.ndarray:
        """Topmost ``F = threshold`` crossing per column, sub-cell interpolated."""
        ny, nx = self.F.shape
        h = np.zeros(nx)
        for i in range(nx):
            col = self.F[:, i]
            above = np.flatnonzero(col > threshold)
            if above.size == 0:
                continue
            j = above.max()
            if j + 1 < ny:
                f0, f1 = col[j], col[j + 1]
                frac = (f0 - threshold) / max(f0 - f1, 1.0e-12)
                h[i] = (j + 0.5 + np.clip(frac, 0.0, 1.0)) * self.dy
            else:
                h[i] = (j + 0.5) * self.dy
        return h

    def probe(self, x: float) -> Dict[str, float]:
        """Surface elevation at abscissa ``x`` plus the local horizontal velocity."""
        i = int(np.clip(round(x / self.dx - 0.5), 0, self.nx - 1))
        return {"height": float(self.interface_height()[i]),
                "u": float(0.5 * (self.u[:, i].mean() + self.u[:, i + 1].mean()))}


# ---------------------------------------------------------------------------
# verification drivers
# ---------------------------------------------------------------------------

class DamBreakMoyce:
    """``H/a = 2`` liquid-column collapse with the front compared to M&M (1952).

    The column is ``a`` wide and ``aspect * a`` tall (``aspect = 2`` reproduces
    the experimental series the digitalised table belongs to).  Non-dimensional
    units make ``a = g = 1``, so the experimental axis ``T = t sqrt(2 g / a)``
    becomes ``T = sqrt(2) t``.
    """

    def __init__(self, cells_per_a: int = 60, aspect: float = 2.0, lx: float = 6.0,
                 ly: float = 3.0, nu_l: float = 1.0e-3, nu_g: float = 1.0e-2,
                 wall: str = "free-slip", cfl: float = 0.25,
                 rho_ratio: float = 1000.0, mass_correction: bool = False,
                 sigma: float = 0.0):
        dx = 1.0 / float(cells_per_a)
        # multiples of 16 let the geometric multigrid coarsen down to a tiny
        # direct solve; the physical domain changes by well under one percent.
        nx = max(16, 16 * int(round(lx / dx / 16.0)))
        ny = max(16, 16 * int(round(ly / dx / 16.0)))
        self.cfg = VOFConfig(nx=nx, ny=ny, lx=nx * dx, ly=ny * dx, g=1.0,
                             rho_g=1.0 / rho_ratio, nu_l=nu_l, nu_g=nu_g,
                             wall=wall, cfl=cfl, sigma=sigma,
                             mass_correction=mass_correction)
        self.aspect = aspect
        self.sim = FreeSurfaceVOF2D(self.cfg)
        self.sim.set_liquid([(0.0, 1.0, 0.0, aspect)])
        self.result: Dict[str, object] = {}

    def run(self, t_end_mm: float = 2.96, callback=None) -> Dict[str, object]:
        """Integrate and sample the front at every tabulated experimental time."""
        table = martin_moyce_table()
        t_nd = table["T"] / np.sqrt(2.0)      # T = sqrt(2) t in these units
        fronts = []
        for tt in t_nd:
            if tt > self.sim.t + 1.0e-12:
                self.sim.run(float(tt), callback=callback)
            fronts.append(self.sim.front_position())
        X_sim = np.asarray(fronts, float)
        X_exp = table["X"]
        err = X_sim - X_exp
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rel = float(np.sqrt(np.mean((err / X_exp) ** 2)) * 100.0)
        self.result = {
            "T": table["T"].tolist(),
            "X_sim": X_sim.tolist(),
            "X_exp": X_exp.tolist(),
            "error": err.tolist(),
            "error_percent": (err / X_exp * 100.0).tolist(),
            "rmse": rmse,
            "rmse_relative_percent": rel,
            "max_abs_error": float(np.abs(err).max()),
            "mass": self.sim.ledger.as_dict(),
            "config": {"cells_per_a": int(round(1.0 / self.cfg.dx)),
                       "nx": self.cfg.nx, "ny": self.cfg.ny,
                       "aspect": self.aspect, "nu_l": self.cfg.nu_l,
                       "wall": self.cfg.wall, "rho_ratio": 1.0 / self.cfg.rho_g,
                       "t_end_mm": float(t_end_mm)},
            "n_steps": self.sim.n_steps,
            "div_max": self.sim.div_max,
        }
        return self.result


def advect_interface_convergence(sizes: Sequence[int] = (40, 80, 160),
                                 t_end: float = 1.0, radius: float = 0.15,
                                 centre: Tuple[float, float] = (0.5, 0.75),
                                 width: float = 0.02, cfl: float = 0.3,
                                 n_rk: int = 400,
                                 width_cells: float = 0.0,
                                 splitting: str = "sequential") -> Dict[str, object]:
    """L2 convergence of the VOF transport against an RK4 flow map.

    Single-vortex field on the unit torus (divergence free, analytic)

        u = -sin^2(pi x) sin(2 pi y),   v = sin(2 pi x) sin^2(pi y)

    which stretches and folds the interface at O(1) strain for one time unit.
    The "exact" field is obtained by integrating the analytic velocity
    *backwards* in time from every cell centre with RK4 (an independent
    discretisation of the flow map, not the advection scheme under test), then
    evaluating the initial interface profile there.

    ``width_cells > 0`` makes the initial tanh thickness ``width_cells`` cells
    instead of a fixed physical width.  A grid-*resolved* interface isolates the
    discretisation error of the transport scheme from the interface-thickness
    model error, which is the standard way to quote a formal order of accuracy
    for a VOF transport: with a fixed physical thickness the measured order
    saturates at ~1 because interface smearing (an h-proportional thickness
    growth) dominates the L2 norm whatever the order of the flux.
    """
    cx, cy = centre
    sizes = [int(s) for s in sizes]
    errors: List[float] = []
    for n in sizes:
        dx = 1.0 / n
        xc = (np.arange(n) + 0.5) * dx
        X, Y = np.meshgrid(xc, xc)
        w = width_cells * dx if width_cells > 0.0 else width
        F = 0.5 * (1.0 - np.tanh((np.hypot(X - cx, Y - cy) - radius) / w))

        xf = np.arange(n + 1) * dx
        uf = -(np.sin(np.pi * xf)[None, :] ** 2) * np.sin(2.0 * np.pi * xc)[:, None]
        vf = np.sin(2.0 * np.pi * xc)[None, :] * (np.sin(np.pi * xf)[:, None] ** 2)
        umax = max(np.abs(uf).max(), np.abs(vf).max())
        dt = cfl * dx / umax
        n_sub = int(np.ceil(t_end / dt))
        dt = t_end / n_sub
        for _ in range(n_sub):
            if splitting == "strang":
                # half x-sweep, full y-sweep, half x-sweep
                F = vof_sweep_x(F, uf, 0.5 * dt, dx, periodic=True)
                F = vof_sweep_y(F, vf, dt, dx, periodic=True)
                F = vof_sweep_x(F, uf, 0.5 * dt, dx, periodic=True)
            elif splitting == "sequential":
                # identical to FreeSurfaceVOF2D.step (the shipped scheme)
                F = vof_sweep_x(F, uf, dt, dx, periodic=True)
                F = vof_sweep_y(F, vf, dt, dx, periodic=True)
            else:
                raise ValueError("splitting must be 'sequential' or 'strang'")
            F = np.clip(F, 0.0, 1.0)

        # backward flow map (RK4), independent of the VOF scheme
        x0, y0 = X.copy(), Y.copy()
        h = -t_end / n_rk
        for _ in range(n_rk):
            k1x, k1y = _vortex(x0, y0)
            k2x, k2y = _vortex(x0 + 0.5 * h * k1x, y0 + 0.5 * h * k1y)
            k3x, k3y = _vortex(x0 + 0.5 * h * k2x, y0 + 0.5 * h * k2y)
            k4x, k4y = _vortex(x0 + h * k3x, y0 + h * k3y)
            x0 = x0 + h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
            y0 = y0 + h * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        F_exact = 0.5 * (1.0 - np.tanh((np.hypot(x0 - cx, y0 - cy) - radius) / w))
        errors.append(float(np.sqrt(np.mean((F - F_exact) ** 2))))

    fit = fit_convergence_order([1.0 / n for n in sizes], errors)
    return {"sizes": sizes, "h": [1.0 / n for n in sizes], "l2": errors,
            "order": fit["order"], "r2": fit["r2"], "t_end": t_end}


def _vortex(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return (-np.sin(np.pi * x) ** 2 * np.sin(2.0 * np.pi * y),
            np.sin(2.0 * np.pi * x) * np.sin(np.pi * y) ** 2)


def laplace_pressure(radius: float = 0.2, sigma: float = 0.01, nx: int = 48,
                     n_steps: int = 600, nu: float = 0.05,
                     rho_g: float = 1.0e-3, poisson_tol: float = 1.0e-6,
                     max_poisson_cycles: int = 15) -> Dict[str, float]:
    """Static droplet: measure ``dp = p_inside - p_outside`` against ``sigma/R``.

    Analytic anchor of the CSF surface-tension model, **not part of the verified
    set**: gravity is off and the pressure jump is averaged over the last 40 % of
    the run.  As documented in ``docs/free_surface_dambreak_vof_report.md`` the
    current CSF discretisation produces spurious capillary currents that make
    this measurement unreliable (honest-FAIL); the dam-break verification runs
    with ``sigma = 0`` and is unaffected.
    """
    cfg = VOFConfig(nx=nx, ny=nx, lx=1.0, ly=1.0, g=0.0, sigma=sigma,
                    nu_l=nu, nu_g=nu, rho_g=rho_g, cfl=0.25, csf_smoothing=2,
                    poisson_tol=poisson_tol, max_poisson_cycles=max_poisson_cycles)
    sim = FreeSurfaceVOF2D(cfg)
    sim.set_droplet(radius)
    x = (np.arange(nx) + 0.5) * cfg.dx
    X, Y = np.meshgrid(x, x)
    r = np.hypot(X - 0.5, Y - 0.5)
    inside = r < 0.5 * radius
    outside = (r > 1.6 * radius) & (r < 2.2 * radius)
    samples = []
    for k in range(n_steps):
        sim.step()
        if k >= int(0.6 * n_steps):
            samples.append(float(sim.p[inside].mean() - sim.p[outside].mean()))
    dp = float(np.mean(samples)) if samples else 0.0
    r_eff = float(np.sqrt(sim.volume() / np.pi))
    dp_theory = sigma / r_eff
    return {"dp_measured": dp, "dp_theory": dp_theory, "radius_initial": radius,
            "radius_effective": r_eff,
            "relative_error_percent": float((dp - dp_theory) / dp_theory * 100.0),
            "n_samples": len(samples),
            "mass": sim.ledger.as_dict()}


# ---------------------------------------------------------------------------
# generic comparison helpers (domain-free: no hull/case vocabulary here)
# ---------------------------------------------------------------------------

def rmse(observed: Sequence[float], reference: Sequence[float]) -> float:
    """Root-mean-square difference between two equally sized series."""
    o = np.asarray(observed, float)
    r = np.asarray(reference, float)
    if o.shape != r.shape:
        raise ValueError(f"series shapes differ: {o.shape} vs {r.shape}")
    if o.size == 0:
        raise ValueError("empty series")
    return float(np.sqrt(np.mean((o - r) ** 2)))


def relative_rmse_percent(observed: Sequence[float], reference: Sequence[float]) -> float:
    """Relative RMSE in percent, normalised pointwise by the reference values."""
    o = np.asarray(observed, float)
    r = np.asarray(reference, float)
    if o.shape != r.shape:
        raise ValueError(f"series shapes differ: {o.shape} vs {r.shape}")
    if np.any(r == 0.0):
        raise ValueError("reference contains zeros; relative RMSE undefined")
    return float(np.sqrt(np.mean(((o - r) / r) ** 2)) * 100.0)
