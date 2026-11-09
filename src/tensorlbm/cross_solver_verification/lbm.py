"""D2Q9 lattice-Boltzmann reference solver for the cross-solver benchmark.

NumPy only (no torch) so that the *same* physics can be run in a process next to
the finite-volume and finite-element reference solvers without a GPU stack.

Two configurations are provided:

``run_lbm_cavity``
    Unit-square lid-driven cavity with a Zou/He-free, half-way bounce-back wall
    treatment and a *moving-wall* momentum correction on the lid, which is the
    standard second-order LBM wall closure.  Cell centres sit at
    ``((i+0.5)/N, (j+0.5)/N)`` so the field can be compared cell-by-cell with
    the FVM/FEM references.

``run_lbm_mms``
    Fully periodic box with the manufactured solution of :mod:`.mms` injected
    through the Guo (2002) force scheme.  Used for the observed-order study.

Conventions: ``rho = 1``, ``c_s^2 = 1/3``, lattice units ``dx = dt = 1``; the
physical domain ``[0,1]^2`` is mapped to ``N x N`` cells, so
``nu_lattice = nu_physical * N``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["LBMResult", "run_lbm_cavity", "run_lbm_mms", "C", "W", "OPP"]

C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CS2 = 1.0 / 3.0

_ENTER_LEFT = (1, 5, 8)     # c_x > 0: enter through x = 0
_ENTER_RIGHT = (3, 6, 7)    # c_x < 0: enter through x = 1
_ENTER_BOTTOM = (2, 5, 6)   # c_y > 0: enter through y = 0
_ENTER_TOP = (4, 7, 8)      # c_y < 0: enter through y = 1


@dataclass
class LBMResult:
    """Container for a converged (or truncated) LBM run."""

    u: np.ndarray
    v: np.ndarray
    rho: np.ndarray
    n: int
    steps: int
    converged: bool
    residual: float
    tau: float
    nu_lattice: float
    history: list[dict] = field(default_factory=list)

    @property
    def speed(self) -> np.ndarray:
        return np.sqrt(self.u * self.u + self.v * self.v)


class _D2Q9:
    """Minimal vectorised D2Q9 BGK + Guo forcing kernel."""

    def __init__(self, n: int, tau: float, periodic: bool):
        self.n = n
        self.tau = tau
        self.periodic = periodic
        self.omega = 1.0 / tau

    # ------------------------------------------------------------------ basics
    def equilibrium(self, rho, u, v):
        cu = 3.0 * (C[:, 0, None, None] * u + C[:, 1, None, None] * v)
        usq = u * u + v * v
        return W[:, None, None] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)

    def macroscopic(self, f, fx=None, fy=None):
        rho = f.sum(axis=0)
        u = (C[:, 0, None, None] * f).sum(axis=0) / rho
        v = (C[:, 1, None, None] * f).sum(axis=0) / rho
        if fx is not None:
            u = u + 0.5 * fx / rho
            v = v + 0.5 * fy / rho
        return rho, u, v

    def stream(self, f):
        g = np.empty_like(f)
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            gi = f[i]
            if cy:
                gi = np.roll(gi, cy, axis=0)
            if cx:
                gi = np.roll(gi, cx, axis=1)
            gi = gi.copy()
            if not self.periodic:
                if cx > 0:
                    gi[:, 0] = 0.0
                elif cx < 0:
                    gi[:, -1] = 0.0
                if cy > 0:
                    gi[0, :] = 0.0
                elif cy < 0:
                    gi[-1, :] = 0.0
            g[i] = gi
        return g

    def collide(self, f, rho, u, v, fx=None, fy=None):
        feq = self.equilibrium(rho, u, v)
        out = f - self.omega * (f - feq)
        if fx is not None:
            cf = C[:, 0, None, None] * fx + C[:, 1, None, None] * fy
            cu = C[:, 0, None, None] * u + C[:, 1, None, None] * v
            uf = u * fx + v * fy
            src = W[:, None, None] * (3.0 * cf + 9.0 * cu * cf - 3.0 * uf)
            out = out + (1.0 - 0.5 * self.omega) * src
        return out

    def walls(self, f, f_post, u_lid: float):
        """Half-way bounce-back with the *post-collision* populations.

        The reflected population at a boundary node is its own post-collision
        value in the opposite direction (the wall sits half a cell away, so the
        particle returns to the same node within one time step).  Using the
        post-*streaming* value instead mixes in the neighbour's population and
        is unconditionally unstable for the driven cavity -- this was caught by
        the Re=100 cavity run (blow-up at n=32) before any result was recorded.
        """
        for i in _ENTER_LEFT:
            f[i][:, 0] = f_post[OPP[i]][:, 0]
        for i in _ENTER_RIGHT:
            f[i][:, -1] = f_post[OPP[i]][:, -1]
        for i in _ENTER_BOTTOM:
            f[i][0, :] = f_post[OPP[i]][0, :]
        for i in _ENTER_TOP:  # moving lid: u_wall = (u_lid, 0)
            f[i][-1, :] = f_post[OPP[i]][-1, :] + 6.0 * W[i] * (C[i, 0] * u_lid)
        return f


def _prolongate_velocity(u_coarse, n_fine):
    """Bilinear prolongation of a coarse cell-centred field (multigrid seed)."""
    n_coarse = u_coarse.shape[0]
    xc = (np.arange(n_coarse) + 0.5) / n_coarse
    xf = (np.arange(n_fine) + 0.5) / n_fine

    def interp_1d(field):
        tmp = np.empty((n_fine, n_coarse), dtype=field.dtype)
        for j in range(n_coarse):
            tmp[:, j] = np.interp(xf, xc, field[:, j])
        out = np.empty((n_fine, n_fine), dtype=field.dtype)
        for i in range(n_fine):
            out[i, :] = np.interp(xf, xc, tmp[i, :])
        return out

    return interp_1d(u_coarse)


def run_lbm_cavity(
    n: int,
    re: float,
    u_lid: float = 0.1,
    max_steps: int = 200_000,
    tol: float = 1e-7,
    check_every: int = 200,
    init_velocity: tuple[np.ndarray, np.ndarray] | None = None,
) -> LBMResult:
    """Run the 2-D lid-driven cavity until the velocity field stops changing.

    Parameters
    ----------
    n:
        Number of cells per side (cell centres at ``(i+0.5)/n``).
    re:
        Reynolds number ``re = u_lid * L / nu`` with ``L = 1``.
    u_lid:
        Lid speed in lattice units (keep ``<= 0.1`` for compressibility).
    tol:
        Steady-state tolerance on ``max|u_new - u_old|`` (in lattice units).
    init_velocity:
        Optional ``(u, v)`` seed on the same grid (used for grid sequencing).
    """
    if re <= 0:
        raise ValueError("Reynolds number must be positive")
    # u_lid = 0 is the zero-driving control: Re is then undefined, so a nominal
    # velocity scale of 0.1 is used for nu only (the field stays identically 0).
    u_scale = u_lid if u_lid > 0 else 0.1
    nu_lat = u_scale * n / re
    tau = 0.5 + 3.0 * nu_lat
    if tau <= 0.5:
        raise ValueError(f"unstable tau={tau:.4f} (increase n or u_lid)")
    k = _D2Q9(n, tau, periodic=False)
    rho = np.ones((n, n))
    if init_velocity is None:
        u = np.zeros((n, n))
        v = np.zeros((n, n))
    else:
        u, v = init_velocity[0].copy(), init_velocity[1].copy()
    f = k.equilibrium(rho, u, v)
    history: list[dict] = []
    conv = False
    residual = np.inf
    steps = 0
    u_prev = u.copy()
    v_prev = v.copy()
    for step in range(1, max_steps + 1):
        steps = step
        rho, u, v = k.macroscopic(f)
        f = k.collide(f, rho, u, v)
        f_post = f
        f = k.stream(f)
        f = k.walls(f, f_post, u_lid)
        if step % check_every == 0:
            rho, u, v = k.macroscopic(f)
            residual = float(max(np.abs(u - u_prev).max(), np.abs(v - v_prev).max()))
            history.append({"step": step, "du": residual, "max_speed": float(np.hypot(u, v).max())})
            u_prev, v_prev = u.copy(), v.copy()
            if residual < tol:
                conv = True
                break
    rho, u, v = k.macroscopic(f)
    return LBMResult(
        u=u, v=v, rho=rho, n=n, steps=steps, converged=conv, residual=residual,
        tau=tau, nu_lattice=nu_lat, history=history,
    )


def run_lbm_mms(
    mms,
    n: int,
    max_steps: int = 60_000,
    tol: float = 1e-9,
    check_every: int = 100,
    seed_exact: bool = True,
) -> LBMResult:
    """Periodic-box MMS run: ``nu_lattice = nu * n``, force = ``rho * f``."""
    nu_lat = mms.nu * n
    tau = 0.5 + 3.0 * nu_lat
    if tau <= 0.5:
        raise ValueError(f"unstable tau={tau:.4f}")
    k = _D2Q9(n, tau, periodic=True)
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")  # (n, n) with axis0=y
    # lattice scaling: nu_lat = nu * n  =>  a_lat = a_phys / n (diffusive scaling),
    # so the manufactured source must be divided by n to keep the same Re.
    fx, fy = mms.source(xx, yy)
    fx = fx / n
    fy = fy / n
    rho = np.ones((n, n))
    if seed_exact:
        u, v = mms.velocity(xx, yy)
        u = np.ascontiguousarray(u)
        v = np.ascontiguousarray(v)
    else:
        u = np.zeros((n, n))
        v = np.zeros((n, n))
    f = k.equilibrium(rho, u, v)
    u_prev, v_prev = u.copy(), v.copy()
    conv, residual, steps = False, np.inf, 0
    history: list[dict] = []
    for step in range(1, max_steps + 1):
        steps = step
        rho, u, v = k.macroscopic(f, fx, fy)
        f = k.collide(f, rho, u, v, fx, fy)
        f = k.stream(f)
        if step % check_every == 0:
            rho, u, v = k.macroscopic(f, fx, fy)
            residual = float(max(np.abs(u - u_prev).max(), np.abs(v - v_prev).max()))
            history.append({"step": step, "du": residual})
            u_prev, v_prev = u.copy(), v.copy()
            if residual < tol:
                conv = True
                break
    rho, u, v = k.macroscopic(f, fx, fy)
    return LBMResult(
        u=u, v=v, rho=rho, n=n, steps=steps, converged=conv, residual=residual,
        tau=tau, nu_lattice=nu_lat, history=history,
    )