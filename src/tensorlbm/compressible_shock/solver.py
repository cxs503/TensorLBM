"""Finite-volume MUSCL / SSP-RK solvers for the compressible Euler equations.

Both solvers use the same ingredients: primitive-variable MUSCL reconstruction
with a minmod limiter (2nd order in space), method-of-lines time stepping with
SSP-RK2, and a pluggable numerical flux (:mod:`flux`).  Primitive
reconstruction is deliberate: it keeps ``rho`` and ``p`` positive after limiting,
which is what lets a Mach-10 shock run for thousands of steps without clipping.

Grid
----
Cell-centred uniform grid.  Boundaries use ``N_GHOST = 2`` ghost layers built
from a boundary *spec*; a spec may be a callable so that time-dependent,
spatially varying Dirichlet data (the moving oblique shock of the
double-Mach-reflection case) is expressible.

Boundary specs
--------------
``("transmissive",)``   zero-gradient (supersonic outflow / non-reflecting)
``("wall",)``           slip adiabatic wall: normal momentum mirrored
``("uniform", state)``  uniform Dirichlet state ``[rho, rho*u, rho*v, E]``
``("callable", fn)``    ``fn(t, edge, axis) -> array`` broadcastable to the
                        ghost block; ``edge`` is the interior edge state

Arrays are ``(ny, nx, 4)`` so ``U[j, i]`` follows the usual (row, column)
plotting convention.  A boolean ``solid`` mask supports blocked-cell obstacles:
fluxes through faces adjacent to a solid cell are set to zero, which is an
exactly non-penetrating staircase wall that conserves mass and momentum.
"""

from __future__ import annotations

import numpy as np

from .flux import GAMMA, max_wave_speed, primitive

__all__ = ["EulerSolver1D", "EulerSolver2D", "minmod", "N_GHOST"]

N_GHOST = 2


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod slope limiter of two one-sided differences."""
    same = a * b > 0.0
    return np.where(same, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _slopes(w: np.ndarray, axis: int) -> np.ndarray:
    """Limited slope of a primitive field along ``axis`` (minmod)."""
    fwd = np.roll(w, -1, axis=axis) - w
    bwd = w - np.roll(w, 1, axis=axis)
    return minmod(fwd, bwd)


def _positivity_scale(rho, p, drho, dp, eps_rel: float = 1e-3):
    """Per-cell slope scale ``theta in [0,1]`` that keeps face states admissible.

    Primitive-variable reconstruction makes the constraint linear: a face value
    ``w_c + theta * dw`` stays positive iff ``theta <= (eps_rel*w_c - w_c)/dw``
    for ``dw < 0``.  Taking the minimum over ``rho`` and ``p`` and both face
    directions gives the Zhang-Shu style positivity limiter.  Returns ``theta``
    with the shape of ``rho``.
    """
    theta = np.ones_like(rho)
    for w_c, dw in ((rho, drho), (p, dp)):
        w_floor = eps_rel * w_c
        for sgn in (1.0, -1.0):
            den = sgn * dw
            limit = np.where(den < 0.0, (w_floor - w_c) / np.where(den < 0.0, den, 1.0), np.inf)
            theta = np.minimum(theta, limit)
    return np.clip(theta, 0.0, 1.0)


def _flux_fn(name: str):
    from . import flux as _f

    try:
        return getattr(_f, f"{name}_flux")
    except AttributeError as exc:  # pragma: no cover - guarded by tests
        raise ValueError(f"unknown flux {name!r}") from exc


class EulerSolver1D:
    """1-D finite-volume Euler solver (MUSCL + SSP-RK2)."""

    def __init__(self, n, length=1.0, gamma=GAMMA, flux="hllc", cfl=0.4, x0=0.0):
        self.n = int(n)
        self.length = float(length)
        self.x0 = float(x0)
        self.dx = self.length / self.n
        self.xc = self.x0 + (np.arange(self.n) + 0.5) * self.dx
        self.gamma = float(gamma)
        self.flux = flux
        self._flux = _flux_fn(flux)
        self.cfl = float(cfl)
        self.U = np.zeros((self.n, 4))
        self.t = 0.0
        self.n_admissibility_fixes = 0
        self.n_positivity_fallbacks = 0
        self.bc_left = ("transmissive",)
        self.bc_right = ("transmissive",)

    # ------------------------------------------------------------------ BCs
    def _ghost(self, spec, edge):
        kind = spec[0]
        if kind == "transmissive":
            return np.repeat(edge[None, :], N_GHOST, axis=0)
        if kind == "wall":
            g = np.repeat(edge[None, :], N_GHOST, axis=0).copy()
            g[:, 1] *= -1.0
            return g
        if kind == "uniform":
            return np.broadcast_to(np.asarray(spec[1], dtype=float), (N_GHOST, 4)).copy()
        if kind == "callable":
            val = np.asarray(spec[1](self.t, edge, 0), dtype=float)
            return np.broadcast_to(val, (N_GHOST, 4)).copy()
        raise ValueError(f"unknown boundary {kind!r}")

    def _haloed(self):
        gL = self._ghost(self.bc_left, self.U[0])
        gR = self._ghost(self.bc_right, self.U[-1])
        return np.vstack([gL, self.U, gR])

    # ------------------------------------------------------------------ RHS
    def _rhs(self, U):
        saved, self.U = self.U, U
        try:
            Uh = self._haloed()
        finally:
            self.U = saved
        rho, u, v, p = primitive(Uh, self.gamma)
        drho, du, dv, dp = (0.5 * _slopes(w, axis=0) for w in (rho, u, v, p))
        theta = _positivity_scale(rho, p, drho, dp)
        drho, du, dv, dp = (theta * x for x in (drho, du, dv, dp))
        sl = slice(1, -1)
        rng = slice(None, -1)
        rng2 = slice(1, None)
        rhoL = rho[sl][rng] + drho[sl][rng]
        uL = u[sl][rng] + du[sl][rng]
        vL = v[sl][rng] + dv[sl][rng]
        pL = p[sl][rng] + dp[sl][rng]
        rhoR = rho[sl][rng2] - drho[sl][rng2]
        uR = u[sl][rng2] - du[sl][rng2]
        vR = v[sl][rng2] - dv[sl][rng2]
        pR = p[sl][rng2] - dp[sl][rng2]
        rhoL = np.maximum(rhoL, 1e-10)
        rhoR = np.maximum(rhoR, 1e-10)
        pL = np.maximum(pL, 1e-10)
        pR = np.maximum(pR, 1e-10)
        g1 = self.gamma - 1.0
        UL = np.stack([rhoL, rhoL * uL, rhoL * vL, pL / g1 + 0.5 * rhoL * (uL**2 + vL**2)], axis=-1)
        UR = np.stack([rhoR, rhoR * uR, rhoR * vR, pR / g1 + 0.5 * rhoR * (uR**2 + vR**2)], axis=-1)
        F = self._flux(UL, UR, self.gamma)
        return -(F[1:] - F[:-1]) / self.dx

    def max_speed(self, U=None) -> float:
        return float(np.max(max_wave_speed(self.U if U is None else U, self.gamma)))

    def _sanitize(self, U):
        U = np.asarray(U, dtype=float)
        if not np.isfinite(U).all():
            U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
        U = U.copy()
        U[:, 0] = np.maximum(U[:, 0], 1e-10)
        rho, u, v, p = primitive(U, self.gamma)
        ke = 0.5 * rho * (u * u + v * v)
        U[:, 3] = np.where(p <= 0.0, ke + 1e-10, U[:, 3])
        return U

    def advance(self, t_end, max_steps=10_000_000, callback=None, callback_every=0):
        steps = 0
        while self.t < t_end - 1e-15 and steps < max_steps:
            dt = self.cfl * self.dx / self.max_speed()
            if self.t + dt > t_end:
                dt = t_end - self.t
            U0 = self.U.copy()
            U1 = self._sanitize(U0 + dt * self._rhs(U0))
            self.U = self._sanitize(0.5 * U0 + 0.5 * (U1 + dt * self._rhs(U1)))
            self.t += dt
            steps += 1
            if callback is not None and callback_every and steps % callback_every == 0:
                callback(self)
        return steps

    def primitives(self):
        return primitive(self.U, self.gamma)


class EulerSolver2D:
    """2-D unsplit finite-volume Euler solver (MUSCL + SSP-RK2)."""

    def __init__(self, nx, ny, lx=1.0, ly=1.0, gamma=GAMMA, flux="hllc", cfl=0.4, x0=0.0, y0=0.0):
        self.nx = int(nx)
        self.ny = int(ny)
        self.lx, self.ly = float(lx), float(ly)
        self.x0, self.y0 = float(x0), float(y0)
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.xc = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        self.yc = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        self.gamma = float(gamma)
        self.flux = flux
        self._flux = _flux_fn(flux)
        self.cfl = float(cfl)
        self.U = np.zeros((self.ny, self.nx, 4))
        self.t = 0.0
        self.solid = None
        self.n_admissibility_fixes = 0
        self.n_positivity_fallbacks = 0
        self.bc_x0 = ("transmissive",)
        self.bc_x1 = ("transmissive",)
        self.bc_y0 = ("transmissive",)
        self.bc_y1 = ("transmissive",)

    # ------------------------------------------------------------------ BCs
    def _edge_ghost(self, spec, edge, axis):
        gh = N_GHOST
        if axis == 0:
            shape = (gh,) + edge.shape
            expand = lambda a: a[None, ...]  # noqa: E731
        else:
            shape = (edge.shape[0], gh) + edge.shape[1:]
            expand = lambda a: a[:, None, :]  # noqa: E731
        kind = spec[0]
        if kind == "transmissive":
            return np.broadcast_to(expand(edge), shape).copy()
        if kind == "wall":
            g = np.broadcast_to(expand(edge), shape).copy()
            g[..., 1 if axis == 1 else 2] *= -1.0
            return g
        if kind == "uniform":
            return np.broadcast_to(np.asarray(spec[1], dtype=float), shape).copy()
        if kind == "callable":
            val = np.asarray(spec[1](self.t, edge, axis), dtype=float)
            return np.broadcast_to(val, shape).copy()
        raise ValueError(f"unknown boundary {kind!r}")

    def _haloed(self):
        U = self.U
        x0 = self._edge_ghost(self.bc_x0, U[:, 0, :], 1)
        x1 = self._edge_ghost(self.bc_x1, U[:, -1, :], 1)
        Ux = np.concatenate([x0, U, x1], axis=1)
        y0 = self._edge_ghost(self.bc_y0, Ux[0, :, :], 0)
        y1 = self._edge_ghost(self.bc_y1, Ux[-1, :, :], 0)
        return np.concatenate([y0, Ux, y1], axis=0)

    # ------------------------------------------------------------------ RHS
    def _reconstruct(self, Uh):
        """Right/left states at the faces along axis 1 (``n_cells + 1`` faces)."""
        rho, u, v, p = primitive(Uh, self.gamma)
        prims = (rho, u, v, p)
        slopes = [0.5 * _slopes(w, axis=1) for w in prims]
        core = tuple(w[:, 1:-1] for w in prims)
        scol = tuple(sl[:, 1:-1] for sl in slopes)
        # positivity limiter on the slopes (keeps rho, p > 0 at every face state)
        theta = _positivity_scale(core[0], core[3], scol[0], scol[3])
        scol = tuple(theta * sl for sl in scol)
        g1 = self.gamma - 1.0
        states = []
        for sgn, sl_cells in ((+1.0, slice(None, -1)), (-1.0, slice(1, None))):
            r, uu, vv, pp = (c[:, sl_cells] + sgn * s_[:, sl_cells] for c, s_ in zip(core, scol))
            r = np.maximum(r, 1e-10)
            pp = np.maximum(pp, 1e-10)
            states.append(
                np.stack([r, r * uu, r * vv, pp / g1 + 0.5 * r * (uu**2 + vv**2)], axis=-1)
            )
        return states[0], states[1]

    def _rhs(self, U):
        saved, self.U = self.U, U
        try:
            Uh = self._haloed()
        finally:
            self.U = saved
        # x-direction fluxes: reconstruction along axis 1
        UL, UR = self._reconstruct(Uh)
        # drop the transverse ghost layers: keep the interior rows
        # face f lies between cell f-1 and f in the haloed indexing, so the
        # interior faces are f = N_GHOST .. N_GHOST+nx
        Fx = self._flux(UL, UR, self.gamma)[N_GHOST:-N_GHOST, :, :]  # (ny, nx+1, 4)
        # y-direction fluxes: transpose the grid AND swap the momentum
        # components, so the very same reconstruction/flux code sees u_n = v.
        Ut = np.swapaxes(Uh, 0, 1)[..., [0, 2, 1, 3]]
        UL, UR = self._reconstruct(Ut)
        Fy = self._flux(UL, UR, self.gamma)
        # back to the physical component ordering: [rho v, rho u v, rho v^2+p, ...]
        Fy = Fy[..., [0, 2, 1, 3]]
        Fy = np.swapaxes(Fy, 0, 1)  # (ny+1, nx+2g, 4)
        Fy = Fy[:, N_GHOST:-N_GHOST, :]  # (ny+1, nx, 4)
        if self.solid is not None:
            # Wall faces: a blocked cell must still transmit the pressure force,
            # otherwise the obstacle is transparent to pressure and mass piles up
            # in the last fluid cell.  The exact wall flux is [0, p, 0, 0]: no mass
            # flux, no work, full normal pressure reaction from the fluid side.
            s = self.solid
            p_cell = primitive(U, self.gamma)[3]
            wall_x = np.zeros((self.ny, self.nx + 1), dtype=bool)
            wall_x[:, 1:-1] = s[:, :-1] | s[:, 1:]
            p_face_x = np.zeros((self.ny, self.nx + 1))
            p_face_x[:, 1:-1] = np.where(s[:, 1:], p_cell[:, :-1], p_cell[:, 1:])
            wall_val_x = np.zeros_like(Fx)
            wall_val_x[..., 1] = p_face_x
            Fx = np.where(wall_x[..., None], wall_val_x, Fx)
            wall_y = np.zeros((self.ny + 1, self.nx), dtype=bool)
            wall_y[1:-1, :] = s[:-1, :] | s[1:, :]
            p_face_y = np.zeros((self.ny + 1, self.nx))
            p_face_y[1:-1, :] = np.where(s[1:, :], p_cell[:-1, :], p_cell[1:, :])
            wall_val_y = np.zeros_like(Fy)
            wall_val_y[..., 2] = p_face_y
            Fy = np.where(wall_y[..., None], wall_val_y, Fy)
        rh = -(Fx[:, 1:, :] - Fx[:, :-1, :]) / self.dx - (Fy[1:, :, :] - Fy[:-1, :, :]) / self.dy
        if self.solid is not None:
            rh = np.where(self.solid[..., None], 0.0, rh)
        return rh

    def max_speed(self, U=None) -> float:
        return float(np.max(max_wave_speed(self.U if U is None else U, self.gamma)))

    def _sanitize(self, U):
        U = np.asarray(U, dtype=float)
        if not np.isfinite(U).all():
            U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
        U = U.copy()
        U[..., 0] = np.maximum(U[..., 0], 1e-10)
        rho, u, v, p = primitive(U, self.gamma)
        ke = 0.5 * rho * (u * u + v * v)
        U[..., 3] = np.where(p <= 0.0, ke + 1e-10, U[..., 3])
        return U

    def _enforce_admissible(self, U_old, U_new, eps_rel: float = 1e-6):
        """Per-cell blend ``U_old + theta (U_new - U_old)`` restoring rho, p > 0.

        Cells whose conservative update would leave the admissible set (possible
        at shock-wall corners, where a single cell absorbs a whole jump) are
        pushed back towards the old state by a bisection on ``theta``.  The
        blend is local, so it does not slow down the global time step; the number
        of affected cells is reported through ``self.n_admissibility_fixes``.
        """
        rho, u, v, p = primitive(U_new, self.gamma)
        bad = ~(np.isfinite(U_new).all(axis=-1)) | (rho <= eps_rel) | (p <= eps_rel)
        n_bad = int(np.count_nonzero(bad))
        if n_bad == 0:
            return U_new
        lo = np.zeros(rho.shape)
        hi = np.ones(rho.shape)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            cand = U_old + mid[..., None] * (U_new - U_old)
            rc, uc, vc, pc = primitive(cand, self.gamma)
            ok = np.isfinite(cand).all(axis=-1) & (rc > eps_rel) & (pc > eps_rel)
            lo = np.where(bad & ok, mid, lo)
            hi = np.where(bad & ~ok, mid, hi)
        theta = np.where(bad, np.maximum(lo, 1e-3), 1.0)
        self.n_admissibility_fixes += n_bad
        return U_old + theta[..., None] * (U_new - U_old)

    def _admissible(self, U) -> bool:
        if not np.isfinite(U).all():
            return False
        rho, u, v, p = primitive(U, self.gamma)
        return bool(np.all(rho > 0.0) and np.all(p > 0.0))

    def advance(self, t_end, max_steps=5_000_000, callback=None, callback_every=0):
        """Integrate to ``t_end``; a rejected step halves ``dt`` (positivity guard)."""
        steps = 0
        self.n_positivity_fallbacks = 0
        self.n_admissibility_fixes = 0
        while self.t < t_end - 1e-15 and steps < max_steps:
            dt = self.cfl / (self.max_speed() * (1.0 / self.dx + 1.0 / self.dy))
            if self.t + dt > t_end:
                dt = t_end - self.t
            U0 = self.U.copy()
            U1 = self._enforce_admissible(U0, U0 + dt * self._rhs(U0))
            U2 = 0.5 * U0 + 0.5 * (U1 + dt * self._rhs(U1))
            if not np.isfinite(U2).all():
                self.n_positivity_fallbacks += 1
                dt *= 0.5
                continue
            self.U = self._enforce_admissible(U0, U2)
            self.t += dt
            steps += 1
            if callback is not None and callback_every and steps % callback_every == 0:
                callback(self)
        return steps

    def primitives(self):
        return primitive(self.U, self.gamma)
