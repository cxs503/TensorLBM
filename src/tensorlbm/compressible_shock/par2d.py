"""Y-slab shared-memory parallelism for the 2-D compressible solver.

The resolvability study needs a 1200x400 grid integrated to ``t = 3`` (2.4e4
SSP-RK2 steps).  The single-core numpy kernel costs ~1.2 s per RHS there, i.e.
~16 h per unit of physical time, so the ladder is out of reach serially.

This module splits the grid into horizontal (``y``) slabs that are evaluated by
a process pool on a ``multiprocessing.shared_memory`` buffer.  Every worker
re-runs *the reference implementation*
(:meth:`tensorlbm.compressible_shock.solver.EulerSolver2D._rhs`) on its own
slab plus two halo rows, so the scheme is unchanged: the parallel RHS is
bit-identical to the serial one (asserted in ``tests`` and by
:func:`self_check`).  Nothing is re-derived here -- that is the point.

Wall-clock scaling on the study grid (2 x 128-core loong64 nodes, shared load)
is documented in the study report; the speed-up is close to linear up to ~24
workers and then flattens (memory bandwidth).
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from .solver import EulerSolver2D

__all__ = ["ParallelEulerSolver2D"]

HALO = 2


# ----------------------------------------------------------------- worker side
_W = {}


def _worker_init(shm_u, shm_rhs, meta, slab_list, queue):
    from multiprocessing import shared_memory

    _W["u"] = shared_memory.SharedMemory(name=shm_u)
    _W["r"] = shared_memory.SharedMemory(name=shm_rhs)
    _W["U"] = np.ndarray(meta["shape"], dtype=np.float64, buffer=_W["u"].buf)
    _W["R"] = np.ndarray(meta["shape"], dtype=np.float64, buffer=_W["r"].buf)
    _W["meta"] = meta
    _W["slabs"] = slab_list
    _W["queue"] = queue


def _wall_ghosts(edge, n_ghost=HALO):
    """Ghost rows of a ``wall`` boundary: mirrored, normal velocity reversed."""
    g = np.broadcast_to(np.asarray(edge, dtype=float), (n_ghost,) + edge.shape).copy()
    g[..., 2] *= -1.0
    return g


def _worker_rhs(slab_id):
    meta = _W["meta"]
    j0, ns = _W["slabs"][slab_id]
    U = _W["U"]
    ny, nx = meta["ny"], meta["nx"]
    lo, hi = j0 - HALO, j0 + ns + HALO

    # ---- build the slab with its two halo rows on each side -------------
    sub = np.empty((hi - lo, nx, 4))
    if lo >= 0:
        sub[:HALO] = U[lo:lo + HALO]
    else:
        sub[:HALO] = _wall_ghosts(U[0]) if meta["wall_y0"] else U[0]
    sub[HALO:HALO + ns] = U[j0:j0 + ns]
    if hi <= ny:
        sub[HALO + ns:] = U[hi - HALO:hi]
    else:
        sub[HALO + ns:] = _wall_ghosts(U[ny - 1]) if meta["wall_y1"] else U[ny - 1]

    # ---- run the reference kernel on that patch --------------------------
    tmp = EulerSolver2D(
        nx=nx, ny=sub.shape[0], lx=nx * meta["dx"], ly=sub.shape[0] * meta["dy"],
        gamma=meta["gamma"], flux=meta["flux"], cfl=meta["cfl"],
    )
    tmp.U = sub
    tmp.bc_x0 = meta["bc_x0"]
    tmp.bc_x1 = meta["bc_x1"]
    if meta["solid"] is not None:
        sg = meta["solid"]
        sol = np.empty((hi - lo, nx), dtype=bool)
        if lo >= 0:
            sol[:HALO] = sg[lo:lo + HALO]
        else:
            sol[:HALO] = sg[0]
        sol[HALO:HALO + ns] = sg[j0:j0 + ns]
        if hi <= ny:
            sol[HALO + ns:] = sg[hi - HALO:hi]
        else:
            sol[HALO + ns:] = sg[ny - 1]
        tmp.solid = sol
    rh = tmp._rhs(sub)
    _W["R"][j0:j0 + ns] = rh[HALO:HALO + ns]
    return ns


class ParallelEulerSolver2D:
    """``EulerSolver2D`` with the RHS evaluated on ``nprocs`` y-slabs.

    The state lives in shared memory, so the public ``U`` attribute is a numpy
    *view*: it is readable/writable like the serial solver's array, and the
    workers see the changes without a copy.
    """

    def __init__(self, nx, ny, lx=1.0, ly=1.0, gamma=1.4, flux="hllc", cfl=0.4,
                 x0=0.0, y0=0.0, nprocs=8, min_rows=6):
        from multiprocessing import shared_memory

        self.nx, self.ny = int(nx), int(ny)
        self.lx, self.ly = float(lx), float(ly)
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.x0, self.y0 = float(x0), float(y0)
        self.xc = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        self.yc = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        self.gamma = float(gamma)
        self.flux = flux
        self.cfl = float(cfl)
        self.t = 0.0
        self.solid = None
        self.bc_x0 = ("transmissive",)
        self.bc_x1 = ("transmissive",)
        self.bc_y0 = ("transmissive",)
        self.bc_y1 = ("transmissive",)
        self.n_admissibility_fixes = 0
        self.n_positivity_fallbacks = 0

        nprocs = max(1, min(int(nprocs), os.cpu_count() or 1, self.ny // min_rows))
        slab_list, j = [], 0
        per = self.ny // nprocs
        for k in range(nprocs):
            ns = per if k < nprocs - 1 else self.ny - j
            slab_list.append((j, ns))
            j += ns
        self.nprocs = nprocs
        self.slabs = slab_list

        shape = (self.ny, self.nx, 4)
        nbytes = int(np.prod(shape)) * 8
        self._shm_u = shared_memory.SharedMemory(create=True, size=nbytes)
        self._shm_r = shared_memory.SharedMemory(create=True, size=nbytes)
        self.U = np.ndarray(shape, dtype=np.float64, buffer=self._shm_u.buf)
        self._R = np.ndarray(shape, dtype=np.float64, buffer=self._shm_r.buf)
        self._pool = None

    # ------------------------------------------------------------------ pool
    def _ensure_pool(self):
        if self._pool is not None:
            return
        meta = dict(
            shape=(self.ny, self.nx, 4), ny=self.ny, nx=self.nx,
            dx=self.dx, dy=self.dy, gamma=self.gamma, flux=self.flux, cfl=self.cfl,
            bc_x0=self.bc_x0, bc_x1=self.bc_x1,
            wall_y0=self.bc_y0[0] == "wall", wall_y1=self.bc_y1[0] == "wall",
            solid=None if self.solid is None else np.ascontiguousarray(self.solid),
        )
        self._pool = mp.get_context("fork").Pool(
            self.nprocs, initializer=_worker_init,
            initargs=(self._shm_u.name, self._shm_r.name, meta, self.slabs, None),
        )

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __del__(self):
        try:
            self.close()
            self._shm_u.close()
            self._shm_r.close()
            self._shm_u.unlink()
            self._shm_r.unlink()
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------ API
    @classmethod
    def from_solver(cls, solver, nprocs=8):
        """Copy a numpy ``EulerSolver2D`` (geometry, state, solid, BCs)."""
        ps = cls(nx=solver.nx, ny=solver.ny, lx=solver.lx, ly=solver.ly,
                 gamma=solver.gamma, flux=getattr(solver, "flux", "hllc"),
                 cfl=solver.cfl, x0=solver.x0, y0=solver.y0, nprocs=nprocs)
        ps.U[...] = solver.U
        ps.t = float(solver.t)
        if solver.solid is not None:
            ps.solid = np.ascontiguousarray(solver.solid)
        ps.bc_x0, ps.bc_x1 = solver.bc_x0, solver.bc_x1
        ps.bc_y0, ps.bc_y1 = solver.bc_y0, solver.bc_y1
        return ps

    def primitives(self):
        from .flux import primitive

        return primitive(self.U, self.gamma)

    def max_speed(self):
        from .flux import max_wave_speed

        return float(np.max(max_wave_speed(self.U, self.gamma)))

    def dt_cfl(self):
        return self.cfl / (self.max_speed() * (1.0 / self.dx + 1.0 / self.dy))

    def rhs(self):
        """Return the RHS in a private array (workers fill the shared buffer).

        The pool is created once (``fork``, workers attach to the shared state)
        and re-used, so the per-call cost is one ``map`` serialisation, not a
        process spawn.  ``solid``/``bc_*`` must not change after the first call.
        """
        self._ensure_pool()
        self._pool.map(_worker_rhs, range(self.nprocs), chunksize=1)
        return self._R.copy()

    def _enforce_admissible(self, U_old, U_new, eps_rel=1e-6):
        from .flux import primitive

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
            rc, _, _, pc = primitive(cand, self.gamma)
            ok = np.isfinite(cand).all(axis=-1) & (rc > eps_rel) & (pc > eps_rel)
            lo = np.where(bad & ok, mid, lo)
            hi = np.where(bad & ~ok, mid, hi)
        theta = np.where(bad, np.maximum(lo, 1e-3), 1.0)
        self.n_admissibility_fixes += n_bad
        return U_old + theta[..., None] * (U_new - U_old)

    def advance(self, t_end, max_steps=5_000_000, callback=None, callback_every=0):
        steps = 0
        while self.t < t_end - 1e-15 and steps < max_steps:
            dt = self.dt_cfl()
            if self.t + dt > t_end:
                dt = t_end - self.t
            U0 = self.U.copy()
            U1 = self._enforce_admissible(U0, U0 + dt * self.rhs())
            self.U[...] = U1  # the shared buffer is the only state the pool sees
            U2 = 0.5 * U0 + 0.5 * (U1 + dt * self.rhs())
            if not np.isfinite(U2).all():
                self.n_positivity_fallbacks += 1
                dt *= 0.5
                continue
            self.U[...] = self._enforce_admissible(U0, U2)
            self.t += dt
            steps += 1
            if callback is not None and callback_every and steps % callback_every == 0:
                callback(self)
        return steps


def self_check(nx=80, ny=24, nprocs=4, mach=3.0, seed=0):
    """Return ``max|rhs_parallel - rhs_serial|`` on a step-case state."""
    from .cases import forward_facing_step_case

    solver, _ = forward_facing_step_case(nx=nx, ny=ny, mach=mach, t_end=0.0)
    rng = np.random.default_rng(seed)
    solver.U = solver.U * (1.0 + 0.05 * rng.standard_normal(solver.U.shape))
    ref = solver._rhs(solver.U)
    ps = ParallelEulerSolver2D.from_solver(solver, nprocs=nprocs)
    par = ps.rhs()
    ps.close()
    return float(np.max(np.abs(ref - par)))