"""Ice-interface momentum exchange: sub-grid wetted fraction, interface
mass-flux accounting and four-level conservation gates.

Mission AG27W2-N01 (parent: IN37-N05 "ice surface books only ~50 % of the
momentum budget").

Why this module exists
----------------------
IN37-N05 reported that the Ladd momentum-exchange method (MEM) applied to the
*ice* surface of the level-ice channel reproduced only about half of the global
momentum budget, while on the straight channel walls the same estimator was
exact to <1e-3 (``docs/IN37_N05_ICE_RESISTANCE_FACTOR_CERTIFICATION.md`` §5.2).
That observation has several candidate causes (see the mission report for the
competing-hypotheses table).  This module supplies the three instruments needed
to separate them, plus the repair:

1. :func:`subgrid_wetted_fraction` -- sub-grid geometry of an analytic ice
   interface (level set): per-cell water fraction ``alpha``, per-link wetted
   fraction ``chi`` (length fraction of the lattice link that lies in ice) and
   the Ladd immersion weight ``w = clip(2*chi, 0, 1)``.  A staircase closure
   silently sets ``chi = 1`` on every blocked link.
2. :class:`InterfaceMassFlux` -- explicit bookkeeping of every mass / momentum /
   angular-momentum / work exchange crossing the ice-water interface, including
   the *moving-interface mass injection* of the Ladd bounce-back
   ``mdot = 2 w_q rho (c_q.u_w)/cs^2``.  The static Ladd estimator drops it; in
   lattice units its size is ``3 u_w`` relative to the reflected population,
   i.e. O(10-50 %) for ice speeds 0.03-0.17.
3. :class:`ConservationGates` -- C1 mass (<1e-10), C2 linear momentum (<1e-9),
   C3 angular momentum (<1e-9), C4 energy (<1 %).  Every gate compares a
   *link-wise ledger* against a *volume integral of the macroscopic moments*,
   i.e. two independent routes, so a gate cannot pass by construction.
4. :class:`IceInterfaceChannel` -- D2Q9 BGK + Guo forcing channel with a
   sub-grid (BFL-type) bounce-back closure, an optional analytic ice floe, and
   defect switches (``subgrid=False``, ``moving_term=False``,
   ``corner_links=False``) that reproduce the candidate root causes.

Exact discrete identities used by the gates
-------------------------------------------
For a closure slot ``(x, k)`` -- fluid node ``x`` whose upwind neighbour
``x - c_k`` is ice, so the population returned by the closure arrives in
direction ``k`` -- the exact mass and momentum exchanged in one step are

    dm = f_new[k](x) - f_post[opp(k)](x)
    dp = c_k * ( f_new[k](x) + f_post[opp(k)](x) )

The population ``f_post[opp(k)]`` leaves ``x`` towards the ice and is discarded;
``f_new[k]`` arrives from the closure.  For the static half-way rule
``f_new[k] = f_post[opp(k)]`` this reduces to ``dm = 0`` and to the classical
Ladd form ``dp = 2 c_q f_q`` with ``c_q = -c_k`` pointing into the wall.
Summing ``dp`` over *all* closure slots reproduces the total momentum change of
the water, so a missed corner link, a wrong wetted weight or a missing
moving-interface term shows up as a residual -- which is what makes C2
discriminating rather than tautological.

Evidence grade: REAL (executable D2Q9 solver, analytic references, no hard-coded
force values).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "CX", "CY", "W", "OPP", "CS2",
    "HalfPlane", "Disk", "InterfaceSet",
    "subgrid_wetted_fraction",
    "InterfaceMassFlux", "ConservationGates", "IceInterfaceChannel",
    "run_poiseuille_subgrid", "run_couette_moving_wall",
    "run_disk_momentum_budget", "run_conservation_suite", "main",
]

# ---------------------------------------------------------------------------
# D2Q9 lattice
# ---------------------------------------------------------------------------
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CS2 = 1.0 / 3.0


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium ``(9, ny, nx)`` from ``(ny, nx)`` macroscopic fields."""
    cu = CX[:, None, None] * ux + CY[:, None, None] * uy
    usq = ux * ux + uy * uy
    return W[:, None, None] * rho * (
        1.0 + cu / CS2 + 0.5 * cu * cu / CS2 ** 2 - 0.5 * usq / CS2
    )


def _shift(a: np.ndarray, k: int, sign: int = 1) -> np.ndarray:
    """``out[..., j, i] = a[..., j - sign*CY[k], i - sign*CX[k]]``.

    ``sign = +1`` is the upwind neighbour ``x - c_k`` (pull streaming);
    ``sign = -1`` is the downwind neighbour ``x + c_k``.
    """
    out = a
    if CY[k]:
        out = np.roll(out, int(sign * CY[k]), axis=-2)
    if CX[k]:
        out = np.roll(out, int(sign * CX[k]), axis=-1)
    return out


# ---------------------------------------------------------------------------
# 1. analytic ice interface (level set)
# ---------------------------------------------------------------------------
class _Surface:
    """Analytic surface: signed distance (>0 in water) and interface velocity."""

    def phi(self, X, Y):  # pragma: no cover - interface
        raise NotImplementedError

    def velocity(self, X, Y):
        return np.zeros_like(X, dtype=float), np.zeros_like(Y, dtype=float)


@dataclass
class HalfPlane(_Surface):
    """Wall ``n.(x - p) = 0``, ice where ``n.(x - p) < 0`` (``|n| = 1``)."""

    normal: Tuple[float, float]
    point: Tuple[float, float]
    u_wall: Tuple[float, float] = (0.0, 0.0)

    def phi(self, X, Y):
        nx, ny = self.normal
        return nx * (X - self.point[0]) + ny * (Y - self.point[1])

    def velocity(self, X, Y):
        return (np.full_like(X, self.u_wall[0], dtype=float),
                np.full_like(Y, self.u_wall[1], dtype=float))


@dataclass
class Disk(_Surface):
    """Ice floe: disk of radius ``R`` at sub-grid centre ``(cx, cy)``."""

    cx: float
    cy: float
    R: float
    u_wall: Tuple[float, float] = (0.0, 0.0)

    def phi(self, X, Y):
        return np.hypot(X - self.cx, Y - self.cy) - self.R

    def velocity(self, X, Y):
        return (np.full_like(X, self.u_wall[0], dtype=float),
                np.full_like(Y, self.u_wall[1], dtype=float))


@dataclass
class InterfaceSet:
    """Union of surfaces: a point is water iff it is outside every ice body."""

    surfaces: List[_Surface] = field(default_factory=list)

    def add(self, s: _Surface) -> "InterfaceSet":
        self.surfaces.append(s)
        return self

    def phi_and_owner(self, X, Y):
        best = None
        owner = None
        for i, s in enumerate(self.surfaces):
            p = s.phi(X, Y)
            if best is None:
                best, owner = p, np.zeros_like(p, dtype=np.int64)
            else:
                m = p < best
                best = np.where(m, p, best)
                owner = np.where(m, i, owner)
        if best is None:
            raise ValueError("empty interface set")
        return best, owner

    def phi(self, X, Y):
        return self.phi_and_owner(X, Y)[0]

    def velocity_at(self, owner, X, Y):
        ux = np.zeros(np.shape(X), dtype=float)
        uy = np.zeros(np.shape(Y), dtype=float)
        for i, s in enumerate(self.surfaces):
            m = owner == i
            if not m.any():
                continue
            sx, sy = s.velocity(X, Y)
            ux[m] = np.asarray(sx)[m]
            uy[m] = np.asarray(sy)[m]
        return ux, uy


# ---------------------------------------------------------------------------
# 1b. sub-grid wetted fraction
# ---------------------------------------------------------------------------
def subgrid_wetted_fraction(
    iface: InterfaceSet,
    nx: int,
    ny: int,
    *,
    n_link: int = 32,
    n_cell: int = 6,
    periodic_x: bool = True,
) -> Dict[str, np.ndarray]:
    """Sub-grid geometry of the ice interface on the ``(ny, nx)`` lattice.

    Returns
    -------
    ``phi``    ``(ny, nx)`` signed distance at cell centres (>0 water)
    ``alpha``  ``(ny, nx)`` water fraction per cell (mid-point quadrature)
    ``chi``    ``(9, ny, nx)`` per-link wetted fraction: the fraction of the
               segment ``[x, x + c_q]`` that lies inside ice (``chi[0]`` holds
               the cell ice fraction for convenience)
    ``w``      ``(9, ny, nx)`` Ladd immersion weight ``clip(2*chi, 0, 1)``;
               ``w = 1`` is the staircase / half-way limit
    ``normal`` ``(2, ny, nx)`` level-set normal at cell centres
    """
    jj, ii = np.mgrid[0:ny, 0:nx]
    Xc = ii + 0.5
    Yc = jj + 0.5
    phi_c = iface.phi(Xc, Yc)

    sub = (np.arange(n_cell) + 0.5) / n_cell - 0.5
    alpha = np.zeros((ny, nx))
    for dy in sub:
        for dx in sub:
            alpha += iface.phi(Xc + dx, Yc + dy) > 0.0
    alpha /= float(n_cell * n_cell)

    chi = np.zeros((9, ny, nx))
    chi[0] = 1.0 - alpha
    ts = (np.arange(n_link) + 0.5) / n_link
    for k in range(1, 9):
        solid = np.zeros((ny, nx))
        for t in ts:
            Xs = Xc + t * float(CX[k])
            Ys = Yc + t * float(CY[k])
            if periodic_x:
                Xs = np.mod(Xs, nx)
            solid += iface.phi(Xs, Ys) < 0.0
        chi[k] = solid / float(n_link)
    w = np.clip(2.0 * chi, 0.0, 1.0)

    gy, gx = np.gradient(phi_c)
    norm = np.maximum(np.hypot(gx, gy), 1e-12)
    normal = np.stack([gx / norm, gy / norm])
    return {"phi": phi_c, "alpha": alpha, "chi": chi, "w": w, "normal": normal}


# ---------------------------------------------------------------------------
# 2. interface mass-flux / momentum-flux accounting
# ---------------------------------------------------------------------------
@dataclass
class InterfaceMassFlux:
    """Ledger of everything that crosses the ice-water interface.

    ``mass_in`` is the sum of the moving-interface injections
    ``2 w rho (c.u_w)/cs^2`` plus any sub-grid closure mass defect (both are
    identically zero for the static half-way rule); ``px_in``/``py_in``/
    ``lz_in`` are the corresponding momentum / angular-momentum fluxes and
    ``work_in`` the work done by the ice on the water.
    """

    mass_in: float = 0.0
    px_in: float = 0.0
    py_in: float = 0.0
    lz_in: float = 0.0
    work_in: float = 0.0
    mass_abs: float = 0.0
    momentum_conv: float = 0.0
    steps: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)

    def record(self, dm, dpx, dpy, rx, ry, uwx, uwy, step: int) -> None:
        self.mass_in += float(dm.sum())
        self.px_in += float(dpx.sum())
        self.py_in += float(dpy.sum())
        self.lz_in += float((rx * dpy - ry * dpx).sum())
        self.work_in += float((dpx * uwx + dpy * uwy).sum())
        self.mass_abs += float(np.abs(dm).sum())
        self.momentum_conv += float((np.abs(dm) * np.hypot(uwx, uwy)).sum())
        self.steps += 1
        if step % 250 == 249:
            self.history.append(dict(step=float(step), mass_in=self.mass_in,
                                     mass_abs=self.mass_abs, px_in=self.px_in,
                                     lz_in=self.lz_in, work_in=self.work_in))

    def as_dict(self) -> Dict[str, float]:
        return dict(mass_in=self.mass_in, px_in=self.px_in, py_in=self.py_in,
                    lz_in=self.lz_in, work_in=self.work_in,
                    mass_abs=self.mass_abs, momentum_conv=self.momentum_conv,
                    steps=float(self.steps))


# ---------------------------------------------------------------------------
# 3. four-level conservation gates
# ---------------------------------------------------------------------------
@dataclass
class ConservationGates:
    """C1 mass / C2 momentum / C3 angular momentum / C4 energy gates."""

    tol_mass: float = 1e-10
    tol_momentum: float = 1e-9
    tol_angular: float = 1e-9
    tol_energy: float = 1e-2

    @staticmethod
    def c1_mass(mass0: float, mass1: float, booked: float) -> Dict[str, float]:
        resid = (mass1 - mass0) - booked
        scale = max(abs(mass1), abs(mass0), abs(booked), 1e-30)
        return dict(residual=resid, rel=abs(resid) / scale, scale=scale,
                    d_volume=mass1 - mass0, booked=booked)

    @staticmethod
    def c2_momentum(p0, p1, impulse, ledger) -> Dict[str, float]:
        dP = np.asarray(p1, float) - np.asarray(p0, float)
        rhs = np.asarray(impulse, float) + np.asarray(ledger, float)
        resid = dP - rhs
        scale = max(float(np.abs(dP).max()), float(np.abs(rhs).max()), 1e-30)
        return dict(residual_x=float(resid[0]), residual_y=float(resid[1]),
                    rel=float(np.abs(resid).max()) / scale, scale=scale,
                    dP=float(np.abs(dP).max()), rhs=float(np.abs(rhs).max()))

    @staticmethod
    def c3_angular(l0: float, l1: float, torque_impulse: float,
                   ledger: float) -> Dict[str, float]:
        resid = (l1 - l0) - (torque_impulse + ledger)
        scale = max(abs(l1), abs(l0), abs(torque_impulse), abs(ledger), 1e-30)
        return dict(residual=resid, rel=abs(resid) / scale, scale=scale,
                    dL=l1 - l0, rhs=torque_impulse + ledger)

    @staticmethod
    def c4_energy(ke0: float, ke1: float, work_ext: float, work_wall: float,
                  dissipation: float) -> Dict[str, float]:
        resid = (ke1 - ke0) - (work_ext + work_wall - dissipation)
        scale = max(abs(ke1), abs(ke0), abs(work_ext), abs(work_wall),
                    abs(dissipation), 1e-30)
        return dict(residual=resid, rel=abs(resid) / scale, scale=scale,
                    dKE=ke1 - ke0, rhs=work_ext + work_wall - dissipation)

    def evaluate(self, **gates) -> Dict[str, object]:
        tol_of = {"c1_mass": self.tol_mass, "c2_momentum": self.tol_momentum,
                  "c3_angular": self.tol_angular, "c4_energy": self.tol_energy}
        out: Dict[str, object] = {}
        for name, payload in gates.items():
            if payload is None:
                out[name] = dict(passed=False, rel=float("nan"),
                                 reason="not_evaluated")
                continue
            payload = dict(payload)
            payload["tol"] = tol_of[name]
            payload["passed"] = bool(payload["rel"] <= tol_of[name])
            out[name] = payload
        out["all_passed"] = bool(all(out[n]["passed"] for n in tol_of if n in out))
        return out

# ---------------------------------------------------------------------------
# 4. executable bench
# ---------------------------------------------------------------------------
C2Q = (CX ** 2 + CY ** 2).astype(float)   # |c_q|^2


class IceInterfaceChannel:
    """D2Q9 BGK + Guo forcing, periodic in ``x``, analytic ice interface.

    Nodes are classified by the cell-centre level set (staircase), but each
    closure link carries its sub-grid wall position ``q`` from the level set and
    the closure is the Bouzidi-Firdaouss-Lallemand interpolation

    ``q >= 1/2``: ``f_new = (1/2q) f_opp(x) + (1 - 1/2q) f_opp(x + c_k)``
    ``q <  1/2``: ``f_new = 2q f_opp(x) + (1 - 2q) f_opp(x + c_k)``

    so ``q = 1/2`` is exactly the half-way rule.  A moving interface adds the
    Ladd velocity term ``+ 2 w_k rho (c_k.u_w)/cs^2`` to the returned
    population; the mass it injects is booked by :class:`InterfaceMassFlux`.

    Defect switches (all default ``True``):

    ``subgrid``      use the level-set ``q`` (``False``: staircase ``q = 1/2``)
    ``moving_term``  add the Ladd moving-interface term
    ``book_flux``    book ``dm``/``dp`` into :class:`InterfaceMassFlux`
    ``corner_links`` include the diagonal closure links

    Ledgers (all *exact* discrete identities, see the module docstring):

    ==================  ============================================================
    ``ledger``          mass / momentum / angular-momentum exchange with the ice
    ``impulse``         accumulated body-force impulse ``sum rho g``
    ``torque_impulse``  accumulated body-force torque about the origin
    ``seam_lz``         angular-momentum transport across the ``x``-periodic seam
    ``e_coll_loss``     lattice-energy lost in the BGK collision
    ``e_source``        lattice energy injected by the Guo forcing term
    ``e_closure``       lattice energy exchanged through the closure links
    ==================  ============================================================
    """

    def __init__(self, iface: InterfaceSet, ny: int, nx: int, tau: float, *,
                 body_force: Tuple[float, float] = (0.0, 0.0),
                 subgrid: bool = True, moving_term: bool = True,
                 book_flux: bool = True, corner_links: bool = True,
                 u0: float = 0.0) -> None:
        self.iface = iface
        self.ny, self.nx = int(ny), int(nx)
        self.tau = float(tau)
        self.omega = 1.0 / self.tau
        self.fx_ext, self.fy_ext = float(body_force[0]), float(body_force[1])
        self.subgrid = subgrid
        self.moving_term = moving_term
        self.book_flux = book_flux
        self.corner_links = corner_links

        geo = subgrid_wetted_fraction(iface, self.nx, self.ny)
        self.phi, self.alpha = geo["phi"], geo["alpha"]
        self.chi, self.w = geo["chi"], geo["w"]
        self.solid = self.phi < 0.0
        self.fluid = ~self.solid
        self.n_open = int(self.fluid.sum())

        jj, ii = np.mgrid[0:self.ny, 0:self.nx]
        Xc, Yc = ii + 0.5, jj + 0.5
        _, owner_c = iface.phi_and_owner(Xc, Yc)
        self.owner = owner_c
        self.n_surf = len(iface.surfaces)
        self.ux_w, self.uy_w = iface.velocity_at(owner_c, Xc, Yc)

        self.close: Dict[int, np.ndarray] = {}
        self.q: Dict[int, np.ndarray] = {}
        for k in range(1, 9):
            if k >= 5 and not corner_links:
                continue
            up_solid = _shift(self.solid, k, +1) & self.fluid
            if not up_solid.any():
                continue
            ph_x = self.phi
            ph_up = _shift(self.phi, k, +1)
            denom = ph_x - ph_up
            ok = np.abs(denom) > 1e-12
            qk = np.where(ok, ph_x / np.where(ok, denom, 1.0), 0.5)
            self.close[k] = up_solid
            self.q[k] = np.where(up_solid, np.clip(qk, 1e-3, 1.0), 0.5)

        rho0 = np.ones((self.ny, self.nx))
        self.f = equilibrium(rho0, np.full_like(rho0, u0), np.zeros_like(rho0))
        self.f[:, self.solid] = W[:, None]
        self.steps = 0

        self.ledger = InterfaceMassFlux()
        self.impulse = np.zeros(2)
        self.torque_impulse = 0.0
        self.seam_lz = 0.0
        self.work_ext = 0.0
        self.work_wall = 0.0
        self.e_coll_loss = 0.0
        self.e_source = 0.0
        self.e_closure = 0.0
        self.dissipation = 0.0
        self.wall_history: List[Dict[str, float]] = []
        # per-surface force, accumulated and averaged over steps
        self.force_acc = np.zeros((self.n_surf, 2))
        self.force_steps = 0

    # -- diagnostics -------------------------------------------------------
    def moments(self):
        rho = self.f.sum(axis=0)
        ux = (self.f * CX[:, None, None]).sum(axis=0) / rho
        uy = (self.f * CY[:, None, None]).sum(axis=0) / rho
        return rho, ux, uy

    def mass(self) -> float:
        return float(self.f.sum(axis=0)[self.fluid].sum())

    def momentum(self) -> np.ndarray:
        rho, ux, uy = self.moments()
        return np.array([float((rho * ux)[self.fluid].sum()),
                         float((rho * uy)[self.fluid].sum())])

    def angular_momentum(self, x0: float = 0.0, y0: float = 0.0) -> float:
        rho, ux, uy = self.moments()
        jj, ii = np.mgrid[0:self.ny, 0:self.nx]
        rx, ry = ii + 0.5 - x0, jj + 0.5 - y0
        return float((rho * (rx * uy - ry * ux))[self.fluid].sum())

    def kinetic_energy(self) -> float:
        rho, ux, uy = self.moments()
        return float((0.5 * rho * (ux * ux + uy * uy))[self.fluid].sum())

    def lattice_energy(self) -> float:
        """``sum_q f_q |c_q|^2 / 2`` over water cells (linear in ``f``)."""
        return float((0.5 * (self.f * C2Q[:, None, None]).sum(axis=0))[self.fluid].sum())

    def force(self, i: int) -> np.ndarray:
        """Time-averaged force **on ice surface ``i``** (per unit depth)."""
        if self.force_steps == 0:
            return np.zeros(2)
        return self.force_acc[i] / self.force_steps

    # -- one step ----------------------------------------------------------
    def step(self) -> Dict[str, float]:
        rho, ux, uy = self.moments()
        fext_x = np.where(self.fluid, self.fx_ext * rho, 0.0)
        fext_y = np.where(self.fluid, self.fy_ext * rho, 0.0)
        ux = ((self.f * CX[:, None, None]).sum(axis=0) + 0.5 * fext_x) / rho
        uy = ((self.f * CY[:, None, None]).sum(axis=0) + 0.5 * fext_y) / rho
        ux[~self.fluid] = 0.0
        uy[~self.fluid] = 0.0
        feq = equilibrium(rho, ux, uy)
        cu = CX[:, None, None] * ux + CY[:, None, None] * uy
        src_x = (1.0 - 0.5 * self.omega) * W[:, None, None] * (
            (CX[:, None, None] - ux) / CS2 + cu * CX[:, None, None] / CS2 ** 2)
        src_y = (1.0 - 0.5 * self.omega) * W[:, None, None] * (
            (CY[:, None, None] - uy) / CS2 + cu * CY[:, None, None] / CS2 ** 2)
        f_post = self.f - self.omega * (self.f - feq) \
            + src_x * fext_x + src_y * fext_y

        # --- exact lattice-energy bookkeeping of collision + forcing ------
        fneq = self.f - feq
        e_coll = float((self.omega * 0.5 * (fneq * C2Q[:, None, None]).sum()) *
                       np.count_nonzero(self.fluid) / max(self.n_open, 1)) \
            if False else float((0.5 * (self.omega * fneq * C2Q[:, None, None])
                                 .sum(axis=0))[self.fluid].sum())
        e_src = float((0.5 * ((src_x * fext_x + src_y * fext_y) *
                              C2Q[:, None, None]).sum(axis=0))[self.fluid].sum())
        diss = float(((1.0 - 0.5 * self.omega) * fneq * fneq
                      / np.maximum(feq, 1e-30)).sum())

        # --- streaming + sub-grid / moving closure -----------------------
        f_new = np.empty_like(f_post)
        f_new[0] = f_post[0]
        for k in range(1, 9):
            f_new[k] = _shift(f_post[k], k, +1)
        returned: Dict[int, np.ndarray] = {}
        for k, m in self.close.items():
            f_opp = f_post[int(OPP[k])]
            if self.subgrid:
                qk = self.q[k]
                w0 = np.where(qk >= 0.5, 1.0 / (2.0 * qk), 2.0 * qk)
                val = w0 * f_opp + (1.0 - w0) * _shift(f_opp, k, -1)
            else:
                val = f_opp
            if self.moving_term:
                val = val + 2.0 * W[k] * rho * (
                    CX[k] * self.ux_w + CY[k] * self.uy_w) / CS2
            f_new[k] = np.where(m, val, f_new[k])
            returned[k] = np.where(m, val, 0.0)

        # --- exact link-wise exchange ledgers ----------------------------
        dm = np.zeros((self.ny, self.nx))
        dpx = np.zeros_like(dm)
        dpy = np.zeros_like(dm)
        de = np.zeros_like(dm)
        f_step = np.zeros((self.n_surf, 2))
        work = 0.0
        for k, m in self.close.items():
            f_opp = f_post[int(OPP[k])]
            ret = returned[k]
            cx, cy = float(CX[k]), float(CY[k])
            px = np.where(m, cx * (ret + f_opp), 0.0)
            py = np.where(m, cy * (ret + f_opp), 0.0)
            dm += np.where(m, ret - f_opp, 0.0)
            dpx += px
            dpy += py
            de += np.where(m, 0.5 * C2Q[k] * (ret - f_opp), 0.0)
            work += float(-(px * self.ux_w + py * self.uy_w).sum())
            if m.any():
                own = self.owner[m]
                for i in range(self.n_surf):
                    sel = own == i
                    if sel.any():
                        f_step[i, 0] -= float(px[m][sel].sum())
                        f_step[i, 1] -= float(py[m][sel].sum())

        jj, ii = np.mgrid[0:self.ny, 0:self.nx]
        rx, ry = ii + 0.5, jj + 0.5
        if self.book_flux:
            self.ledger.record(dm, dpx, dpy, rx, ry, self.ux_w, self.uy_w,
                               self.steps)

        # angular-momentum transport across the x-periodic seam
        seam = 0.0
        for k in range(1, 9):
            if CX[k] == 0:
                continue
            src = self.nx - 1 if CX[k] > 0 else 0
            p_y = float((CY[k] * f_new[k][:, src]).sum())
            seam += -float(self.nx) * float(CX[k]) * p_y
        self.seam_lz += seam

        # external torque about the origin
        self.torque_impulse += float((rx * fext_y - ry * fext_x).sum())

        self.force_acc += f_step
        self.force_steps += 1
        self.work_wall += work
        self.work_ext += float((fext_x * ux + fext_y * uy).sum())
        self.e_coll_loss += e_coll
        self.e_source += e_src
        self.e_closure += float(de.sum())
        self.dissipation += diss
        self.impulse[0] += float(fext_x.sum())
        self.impulse[1] += float(fext_y.sum())

        self.f = f_new
        self.steps += 1
        if self.steps % 50 == 0:
            self.wall_history.append(dict(
                step=float(self.steps), f_water_x=float(dpx.sum()),
                f_ice_x=float(-dpx.sum())))
        return dict(mass=self.mass(), ke=self.kinetic_energy(),
                    px=float(self.momentum()[0]), py=float(self.momentum()[1]),
                    diss=diss)


# ---------------------------------------------------------------------------
# 5. bench cases with analytic references
# ---------------------------------------------------------------------------
def _gates_for(s: "IceInterfaceChannel", m0, p0, l0, ke0, e0) -> Dict[str, object]:
    """Evaluate C1-C4 from the "before" snapshot and the accumulated ledgers."""
    m1, p1 = s.mass(), s.momentum()
    l1, ke1 = s.angular_momentum(), s.kinetic_energy()
    e1 = s.lattice_energy()
    c1 = ConservationGates.c1_mass(m0, m1, s.ledger.mass_in)
    c2 = ConservationGates.c2_momentum(p0, p1, s.impulse,
                                       np.array([s.ledger.px_in, s.ledger.py_in]))
    c3 = ConservationGates.c3_angular(l0 + s.ledger.lz_in + s.seam_lz,
                                      l1, s.torque_impulse, 0.0)
    # C4 as the exact lattice-energy balance:  dE_lat = e_source + e_closure
    # - e_coll_loss, and KE = E_lat - cs^2 M.
    c4 = ConservationGates.c4_energy(
        ke0 + CS2 * m0, ke1 + CS2 * m1,
        s.e_source, s.e_closure, s.e_coll_loss)
    c4["physical_work_ext"] = s.work_ext
    c4["physical_work_wall"] = s.work_wall
    c4["physical_ke_balance_rel"] = abs((ke1 - ke0) - (s.work_ext + s.work_wall
                                                       - s.dissipation)) / max(
        abs(ke1), abs(ke0), 1e-30)
    return ConservationGates().evaluate(c1_mass=c1, c2_momentum=c2,
                                        c3_angular=c3, c4_energy=c4)


def run_poiseuille_subgrid(*, y_bottom: float = 1.37, y_top: float = 18.63,
                           nx: int = 16, g: float = 1e-5, tau: float = 0.8,
                           steps: int = 9000,
                           subgrid: bool = True) -> Dict[str, object]:
    """Plane Poiseuille between two *sub-grid* walls (wetted-fraction test).

    Analytic: ``u(y) = g/(2 nu) (y-y1)(y2-y)`` and the wall momentum budget
    ``F_wall = g A`` with ``A = y2 - y1`` the true open height.  Staircase
    bookkeeping cannot represent ``y1 = 1.37`` and quantises the height, so its
    force (and profile) is biased by the wall-position error.
    """
    ny = int(math.ceil(y_top)) + 3
    iface = InterfaceSet().add(HalfPlane((0.0, 1.0), (0.0, y_bottom))).add(
        HalfPlane((0.0, -1.0), (0.0, y_top)))
    nu = (tau - 0.5) / 3.0
    s = IceInterfaceChannel(iface, ny, nx, tau, body_force=(g, 0.0),
                            subgrid=subgrid)
    m0, p0 = s.mass(), s.momentum()
    l0, ke0, e0 = s.angular_momentum(), s.kinetic_energy(), s.lattice_energy()
    for _ in range(steps):
        s.step()
    rho, ux, uy = s.moments()
    jj = np.arange(ny) + 0.5
    u_true = g / (2.0 * nu) * (jj - y_bottom) * (y_top - jj)
    err = ux[s.fluid] - np.broadcast_to(u_true[:, None], ux.shape)[s.fluid]
    u_rms = float(np.sqrt((err ** 2).mean()) / max(np.abs(u_true).max(), 1e-30))

    A_true = y_top - y_bottom
    f_analytic = g * A_true * nx          # total body-force impulse
    f_total = float(s.force(0)[0] + s.force(1)[0])
    f_budget_discrete = g * s.n_open
    gates = _gates_for(s, m0, p0, l0, ke0, e0)
    return dict(case="poiseuille_subgrid", subgrid=subgrid,
                y_bottom=y_bottom, y_top=y_top, A_true=A_true, nx=nx, ny=ny,
                tau=tau, nu=nu, g=g, steps=steps, n_open=s.n_open,
                wall_force_measured=abs(f_total),
                wall_force_analytic=f_analytic,
                wall_force_rel_error=abs(abs(f_total) - f_analytic) / f_analytic,
                wall_force_discrete_budget=f_budget_discrete,
                mem_over_discrete_budget=(abs(f_total) / f_budget_discrete
                                          if f_budget_discrete else float("nan")),
                u_profile_rel_rms=u_rms,
                ubar_measured=float(ux[s.fluid].mean()),
                ubar_analytic=float(u_true[u_true > 0].mean()),
                ledger=s.ledger.as_dict(), gates=gates)


def run_couette_moving_wall(*, y_bottom: float = 1.23, y_top: float = 18.77,
                            nx: int = 12, u_wall: float = 0.06,
                            tau: float = 0.8, steps: int = 6000,
                            moving_term: bool = True,
                            subgrid: bool = True,
                            corner_links: bool = True) -> Dict[str, object]:
    """Couette flow driven by a *moving* ice surface at a sub-grid position.

    Analytic: shear ``du/dy = U/H``, wall stress ``rho nu U/H``, so the force on
    the plate is ``nu U nx / H`` per unit depth.  The moving-interface Ladd term
    is what makes the exchange equal to this; the static estimator under-books
    it by ``3 U`` in lattice units (``2 w rho U/cs^2`` relative to ``2 w rho``).
    """
    ny = int(math.ceil(y_top)) + 3
    iface = InterfaceSet().add(
        HalfPlane((0.0, 1.0), (0.0, y_bottom), u_wall=(u_wall, 0.0))).add(
        HalfPlane((0.0, -1.0), (0.0, y_top)))
    nu = (tau - 0.5) / 3.0
    s = IceInterfaceChannel(iface, ny, nx, tau, subgrid=subgrid,
                            moving_term=moving_term, corner_links=corner_links)
    m0, p0 = s.mass(), s.momentum()
    l0, ke0, e0 = s.angular_momentum(), s.kinetic_energy(), s.lattice_energy()
    for _ in range(steps):
        s.step()
    H = y_top - y_bottom
    f_analytic = nu * u_wall / H * nx
    f_plate = abs(float(s.force(0)[0]))
    rho, ux, uy = s.moments()
    jj = np.arange(ny) + 0.5
    u_true = u_wall * (y_top - jj) / H
    err = ux[s.fluid] - np.broadcast_to(u_true[:, None], ux.shape)[s.fluid]
    u_rms = float(np.sqrt((err ** 2).mean()) / max(abs(u_wall), 1e-30))
    gates = _gates_for(s, m0, p0, l0, ke0, e0)
    return dict(case="couette_moving_wall", moving_term=moving_term,
                subgrid=subgrid, corner_links=corner_links,
                y_bottom=y_bottom, y_top=y_top, H=H, u_wall=u_wall, nx=nx,
                tau=tau, nu=nu, steps=steps,
                wall_force_measured=f_plate, wall_force_analytic=f_analytic,
                wall_force_rel_error=abs(f_plate - f_analytic) / f_analytic,
                implied_open_height=(nu * u_wall * nx / f_plate
                                     if f_plate else float("nan")),
                u_profile_rel_rms=u_rms,
                ubar_measured=float(ux[s.fluid].mean()),
                ledger=s.ledger.as_dict(), gates=gates)


def run_disk_momentum_budget(*, R: float = 4.37, cy: float = 10.42,
                             y_bottom: float = 1.81, y_top: float = 19.19,
                             nx: int = 24, g: float = 2e-5, tau: float = 0.9,
                             steps: int = 4000,
                             subgrid: bool = True) -> Dict[str, object]:
    """Ice floe in a periodic channel: MEM force vs. global momentum budget.

    Two-dimensional analogue of the IN37-N05 level-ice bench.  At steady state
    the momentum budget of the periodic cell is ``F_wall + F_ice = g n_open``
    *exactly*, so the MEM/budget ratio of the ice force is a direct measure of
    the estimator defect.  ``subgrid=False`` reproduces the staircase MEM.
    """
    ny = int(math.ceil(y_top)) + 3
    iface = InterfaceSet().add(HalfPlane((0.0, 1.0), (0.0, y_bottom))).add(
        HalfPlane((0.0, -1.0), (0.0, y_top))).add(Disk(nx * 0.5, cy, R))
    nu = (tau - 0.5) / 3.0
    s = IceInterfaceChannel(iface, ny, nx, tau, body_force=(g, 0.0),
                            subgrid=subgrid)
    m0, p0 = s.mass(), s.momentum()
    l0, ke0, e0 = s.angular_momentum(), s.kinetic_energy(), s.lattice_energy()
    hist = []
    for n in range(steps):
        s.step()
        if n % 100 == 99:
            hist.append(float(s.force(2)[0]))
    f_ice = float(s.force(2)[0])
    f_budget = g * s.n_open - float(s.force(0)[0] + s.force(1)[0])
    gates = _gates_for(s, m0, p0, l0, ke0, e0)
    return dict(case="disk_momentum_budget", subgrid=subgrid, R=R, cy=cy,
                nx=nx, ny=ny, y_bottom=y_bottom, y_top=y_top, g=g, tau=tau,
                nu=nu, steps=steps, n_open=s.n_open,
                force_ice=f_ice, force_budget=f_budget,
                wall_force_x=float(s.force(0)[0] + s.force(1)[0]),
                mem_over_budget=f_ice / f_budget if f_budget else float("nan"),
                ubar=float(s.moments()[1][s.fluid].mean()),
                budget_history=hist, ledger=s.ledger.as_dict(), gates=gates)


def run_conservation_suite(out_dir: Optional[str] = None,
                           *, steps_scale: float = 1.0) -> Dict[str, object]:
    """Run every bench case (repair + defect variants) and assemble the report."""
    n_po = max(200, int(9000 * steps_scale))
    n_co = max(200, int(6000 * steps_scale))
    n_dk = max(200, int(4000 * steps_scale))
    suite = {
        "mission": "AG27W2-N01",
        "parent": "IN37-N05",
        "poiseuille_subgrid": run_poiseuille_subgrid(subgrid=True, steps=n_po),
        "poiseuille_staircase": run_poiseuille_subgrid(subgrid=False, steps=n_po),
        "couette_moving": run_couette_moving_wall(moving_term=True,
                                                  subgrid=True, steps=n_co),
        "couette_static_mem": run_couette_moving_wall(moving_term=False,
                                                      subgrid=True, steps=n_co),
        "couette_corner_off": run_couette_moving_wall(moving_term=True,
                                                      subgrid=True,
                                                      steps=n_co,
                                                      corner_links=False),
        "disk_subgrid": run_disk_momentum_budget(subgrid=True, steps=n_dk),
        "disk_staircase": run_disk_momentum_budget(subgrid=False, steps=n_dk),
    }
    suite["verdict"] = _suite_verdict(suite)
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "conservation_suite.json").write_text(
            json.dumps(suite, indent=2, default=float))
    return suite


def _suite_verdict(s: Dict[str, object]) -> Dict[str, object]:
    po, ps = s["poiseuille_subgrid"], s["poiseuille_staircase"]
    co, cs = s["couette_moving"], s["couette_static_mem"]
    ck = s["couette_corner_off"]
    dk, dst = s["disk_subgrid"], s["disk_staircase"]
    return dict(
        subgrid_wetted_fraction_improves_poiseuille=bool(
            po["wall_force_rel_error"] < ps["wall_force_rel_error"]),
        poiseuille_error_subgrid=po["wall_force_rel_error"],
        poiseuille_error_staircase=ps["wall_force_rel_error"],
        poiseuille_improvement_factor=(
            ps["wall_force_rel_error"] / po["wall_force_rel_error"]
            if po["wall_force_rel_error"] > 0 else float("inf")),
        moving_wall_term_required=bool(
            co["wall_force_rel_error"] < 0.5 * cs["wall_force_rel_error"]),
        couette_error_with_moving_term=co["wall_force_rel_error"],
        couette_error_static_mem=cs["wall_force_rel_error"],
        couette_error_ratio=(cs["wall_force_rel_error"]
                             / co["wall_force_rel_error"]
                             if co["wall_force_rel_error"] > 0 else float("inf")),
        corner_links_required=bool(
            co["gates"]["c2_momentum"]["passed"]
            and not ck["gates"]["c2_momentum"]["passed"]),
        gates_all_passed_subgrid=bool(po["gates"]["all_passed"]
                                      and co["gates"]["all_passed"]
                                      and dk["gates"]["all_passed"]),
        disk_mem_over_budget_subgrid=dk["mem_over_budget"],
        disk_mem_over_budget_staircase=dst["mem_over_budget"],
    )


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="artifacts/ag27w2_n01")
    ap.add_argument("--steps-scale", type=float, default=1.0)
    args = ap.parse_args(argv)
    suite = run_conservation_suite(args.out, steps_scale=args.steps_scale)
    print(json.dumps(suite.get("verdict", {}), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
