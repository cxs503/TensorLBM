"""Torch mirror of :class:`~.solver.EulerSolver2D` (same discretisation, threaded).

Why this file exists
--------------------
The resolvability study of the bow-shock stand-off ahead of a step needs a
four-grid ladder topping out at 1200x400 cells run to ``t = 4`` (``~3.2e4``
SSP-RK2 steps).  The reference numpy kernel is single-threaded and memory bound
(``~1.6 us`` per cell per RHS on this hardware), i.e. ``~4 h`` for the finest
grid alone.  This module re-implements *exactly the same discretisation*
(MUSCL/minmod primitive reconstruction, Zhang-Shu positivity limiter, HLLC
flux, SSP-RK2, staircase solid with the ``[0, p, 0, 0]`` wall flux) on torch
tensors, which releases the 128 cores of a compute node.

It is a *port*, not a new scheme: equivalence to the numpy kernel is a test
(``tests/test_shock_resolvability.py::test_torch_mirror_matches_numpy``), and
every production number of the study is reproducible with ``backend="numpy"``
(slower, identical to the port's round-off).

Notes
-----
* ``torch`` is an optional dependency of TensorLBM.  Importing this module
  without torch raises a clear ``ImportError`` only when a solver is built, so
  the rest of the package keeps working.
* ``EulerSolver2D`` remains the reference implementation; this class carries
  the same public surface (``U``, ``dx``, ``dy``, ``advance``, ``primitives``,
  ``max_speed``) plus :meth:`face_fluxes`, which the mass-flux audit uses.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised by the torch-availability branch
    import torch
except Exception:  # pragma: no cover
    torch = None

from .flux import GAMMA

__all__ = ["TorchEulerSolver2D", "TORCH_AVAILABLE"]

TORCH_AVAILABLE = torch is not None
N_GHOST = 2


def _require_torch():
    if torch is None:  # pragma: no cover
        raise ImportError(
            "tensorlbm.compressible_shock.fast2d needs torch; use "
            "tensorlbm.compressible_shock.solver.EulerSolver2D instead"
        )


def _minmod(a, b):
    same = a * b > 0.0
    return torch.where(same, torch.sign(a) * torch.minimum(a.abs(), b.abs()),
                       torch.zeros_like(a))


def _slopes(w, axis):
    fwd = torch.roll(w, -1, dims=axis) - w
    bwd = w - torch.roll(w, 1, dims=axis)
    return _minmod(fwd, bwd)


def _primitive(U, gamma):
    rho = U[..., 0]
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    p = (gamma - 1.0) * (U[..., 3] - 0.5 * rho * (u * u + v * v))
    return rho, u, v, p


def _euler_flux(U, gamma):
    rho, u, v, p = _primitive(U, gamma)
    F = torch.empty_like(U)
    F[..., 0] = rho * u
    F[..., 1] = rho * u * u + p
    F[..., 2] = rho * u * v
    F[..., 3] = u * (U[..., 3] + p)
    return F


def _positivity_scale(rho, p, drho, dp, eps_rel=1e-3):
    theta = torch.ones_like(rho)
    for w_c, dw in ((rho, drho), (p, dp)):
        w_floor = eps_rel * w_c
        for sgn in (1.0, -1.0):
            den = sgn * dw
            live = den < 0.0
            lim = (w_floor - w_c) / torch.where(live, den, torch.ones_like(den))
            theta = torch.minimum(theta, torch.where(live, lim, torch.full_like(lim, float("inf"))))
    return torch.clamp(theta, 0.0, 1.0)


def _hllc_flux(UL, UR, gamma):
    """HLLC with Einfeldt/Roe wave speeds -- line-by-line port of ``flux.hllc_flux``."""
    rhoL, uL, vL, pL = _primitive(UL, gamma)
    rhoR, uR, vR, pR = _primitive(UR, gamma)
    aL = torch.sqrt(gamma * pL / rhoL)
    aR = torch.sqrt(gamma * pR / rhoR)
    sL = torch.sqrt(rhoL)
    sR = torch.sqrt(rhoR)
    den = sL + sR
    u_t = (sL * uL + sR * uR) / den
    v_t = (sL * vL + sR * vR) / den
    H = (sL * (UL[..., 3] + pL) / rhoL + sR * (UR[..., 3] + pR) / rhoR) / den
    a_t = torch.sqrt(torch.clamp((gamma - 1.0) * (H - 0.5 * (u_t * u_t + v_t * v_t)), min=1e-300))
    SL = torch.minimum(uL - aL, u_t - a_t)
    SR = torch.maximum(uR + aR, u_t + a_t)

    num = pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)
    den2 = rhoL * (SL - uL) - rhoR * (SR - uR)
    den2 = torch.where(den2.abs() < 1e-300, torch.full_like(den2, 1e-300), den2)
    SM = num / den2

    FL = _euler_flux(UL, gamma)
    FR = _euler_flux(UR, gamma)

    def star(U, rho, u, v, p, S):
        fac = rho * (S - u) / (S - SM)
        out = torch.empty_like(U)
        out[..., 0] = fac
        out[..., 1] = fac * SM
        out[..., 2] = fac * v
        out[..., 3] = fac * (U[..., 3] / rho + (SM - u) * (SM + p / (rho * (S - u))))
        return out

    USL = star(UL, rhoL, uL, vL, pL, SL)
    USR = star(UR, rhoR, uR, vR, pR, SR)
    FSL = FL + SL[..., None] * (USL - UL)
    FSR = FR + SR[..., None] * (USR - UR)

    out = torch.where((SL >= 0.0)[..., None], FL, FSL)
    out = torch.where(((SL < 0.0) & (SM >= 0.0))[..., None], FSL, out)
    out = torch.where(((SM < 0.0) & (SR > 0.0))[..., None], FSR, out)
    out = torch.where((SR <= 0.0)[..., None], FR, out)
    return out


class TorchEulerSolver2D:
    """Threaded mirror of ``solver.EulerSolver2D`` (same numerics, torch tensors)."""

    def __init__(self, nx, ny, lx=1.0, ly=1.0, gamma=GAMMA, cfl=0.4, x0=0.0, y0=0.0,
                 dtype=None, threads=None):
        _require_torch()
        if threads is not None:
            torch.set_num_threads(int(threads))
        self.threads_used = int(torch.get_num_threads())
        self.dtype = torch.float64 if dtype is None else dtype
        self.nx, self.ny = int(nx), int(ny)
        self.lx, self.ly = float(lx), float(ly)
        self.gamma = float(gamma)
        self.cfl = float(cfl)
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.x0, self.y0 = float(x0), float(y0)
        self.xc = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        self.yc = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        self.U = torch.zeros((self.ny, self.nx, 4), dtype=self.dtype)
        self.t = 0.0
        self.solid = None
        self.n_positivity_fallbacks = 0
        self.n_admissibility_fixes = 0
        self.bc_x0 = ("transmissive",)
        self.bc_x1 = ("transmissive",)
        self.bc_y0 = ("transmissive",)
        self.bc_y1 = ("transmissive",)

    # ------------------------------------------------------------- utilities
    def primitives(self):
        return tuple(t.detach().cpu().numpy() for t in _primitive(self.U, self.gamma))

    def set_uniform(self, U_vec):
        self.U = torch.as_tensor(np.asarray(U_vec, dtype=float), dtype=self.dtype)
        self.U = self.U.expand((self.ny, self.nx, 4)).clone()

    def max_speed(self):
        rho, u, v, p = _primitive(self.U, self.gamma)
        a = torch.sqrt(self.gamma * torch.clamp(p, min=1e-300) / rho)
        return float((u.abs() + a).max().item())

    def dt_cfl(self, safety=1.0):
        return safety * self.cfl / (self.max_speed() * (1.0 / self.dx + 1.0 / self.dy))

    # ------------------------------------------------------------------ BCs
    def _edge_ghost(self, spec, edge, axis):
        gh = N_GHOST
        if axis == 0:
            shape = (gh,) + tuple(edge.shape)
            expand = lambda a: a.unsqueeze(0)  # noqa: E731
        else:
            shape = (edge.shape[0], gh) + tuple(edge.shape[1:])
            expand = lambda a: a.unsqueeze(1)  # noqa: E731
        kind = spec[0]
        if kind == "transmissive":
            return expand(edge).expand(shape).clone()
        if kind == "wall":
            g = expand(edge).expand(shape).clone()
            g[..., 1 if axis == 1 else 2] *= -1.0
            return g
        if kind == "uniform":
            return torch.as_tensor(np.asarray(spec[1], dtype=float), dtype=self.dtype).expand(shape).clone()
        if kind == "callable":
            val = np.asarray(spec[1](self.t, edge.detach().cpu().numpy(), axis), dtype=float)
            return torch.as_tensor(val, dtype=self.dtype).expand(shape).clone()
        raise ValueError(f"unknown boundary {kind!r}")

    def _haloed(self):
        U = self.U
        x0 = self._edge_ghost(self.bc_x0, U[:, 0, :], 1)
        x1 = self._edge_ghost(self.bc_x1, U[:, -1, :], 1)
        Ux = torch.cat([x0, U, x1], dim=1)
        y0 = self._edge_ghost(self.bc_y0, Ux[0, :, :], 0)
        y1 = self._edge_ghost(self.bc_y1, Ux[-1, :, :], 0)
        return torch.cat([y0, Ux, y1], dim=0)

    # -------------------------------------------------------------- kernels
    def _reconstruct(self, Uh, axis):
        rho, u, v, p = _primitive(Uh, self.gamma)
        prims = (rho, u, v, p)
        slopes = [0.5 * _slopes(w, axis) for w in prims]
        core = tuple(w.narrow(axis, 1, w.shape[axis] - 2) for w in prims)
        scol = tuple(sl.narrow(axis, 1, sl.shape[axis] - 2) for sl in slopes)
        theta = _positivity_scale(core[0], core[3], scol[0], scol[3])
        scol = tuple(theta * sl for sl in scol)
        g1 = self.gamma - 1.0
        states = []
        for sgn in (+1.0, -1.0):
            n = core[0].shape[axis] - 1
            r, uu, vv, pp = (
                c.narrow(axis, 0 if sgn > 0 else 1, n) + sgn * s_.narrow(axis, 0 if sgn > 0 else 1, n)
                for c, s_ in zip(core, scol)
            )
            r = torch.clamp(r, min=1e-10)
            pp = torch.clamp(pp, min=1e-10)
            states.append(torch.stack([r, r * uu, r * vv, pp / g1 + 0.5 * r * (uu ** 2 + vv ** 2)], dim=-1))
        return states[0], states[1]

    def face_fluxes(self, U=None):
        """Return ``(Fx, Fy)`` -- the x/y interface fluxes of the current state.

        Shapes ``Fx: (ny, nx+1, 4)``, ``Fy: (ny+1, nx, 4)``; this is exactly the
        flux the scheme conserves with, so a control-volume mass budget built
        from it closes to round-off (used by
        :func:`tensorlbm.compressible_shock.resolvability.blockage_mass_audit`).
        """
        U_save = None
        if U is not None:
            U_save, self.U = self.U, U
        try:
            Uh = self._haloed()
            UL, UR = self._reconstruct(Uh, 1)
            Fx = _hllc_flux(UL, UR, self.gamma)[N_GHOST:-N_GHOST, :, :]
            Ut = torch.swapaxes(Uh, 0, 1)[..., [0, 2, 1, 3]]
            UL, UR = self._reconstruct(Ut, 1)
            Fy = _hllc_flux(UL, UR, self.gamma)
            Fy = Fy[..., [0, 2, 1, 3]]
            Fy = torch.swapaxes(Fy, 0, 1)[:, N_GHOST:-N_GHOST, :]
            if self.solid is not None:
                s = self.solid
                p_cell = _primitive(self.U if U is None else U, self.gamma)[3]
                wall_x = torch.zeros((self.ny, self.nx + 1), dtype=torch.bool)
                wall_x[:, 1:-1] = s[:, :-1] | s[:, 1:]
                p_face_x = torch.zeros((self.ny, self.nx + 1), dtype=self.dtype)
                p_face_x[:, 1:-1] = torch.where(s[:, 1:], p_cell[:, :-1], p_cell[:, 1:])
                wall_val_x = torch.zeros_like(Fx)
                wall_val_x[..., 1] = p_face_x
                Fx = torch.where(wall_x.unsqueeze(-1), wall_val_x, Fx)
                wall_y = torch.zeros((self.ny + 1, self.nx), dtype=torch.bool)
                wall_y[1:-1, :] = s[:-1, :] | s[1:, :]
                p_face_y = torch.zeros((self.ny + 1, self.nx), dtype=self.dtype)
                p_face_y[1:-1, :] = torch.where(s[1:, :], p_cell[:-1, :], p_cell[1:, :])
                wall_val_y = torch.zeros_like(Fy)
                wall_val_y[..., 2] = p_face_y
                Fy = torch.where(wall_y.unsqueeze(-1), wall_val_y, Fy)
            return Fx, Fy
        finally:
            if U_save is not None:
                self.U = U_save

    def rhs(self, U):
        Fx, Fy = self.face_fluxes(U)
        rh = -(Fx[:, 1:, :] - Fx[:, :-1, :]) / self.dx - (Fy[1:, :, :] - Fy[:-1, :, :]) / self.dy
        if self.solid is not None:
            rh = torch.where(self.solid.unsqueeze(-1), torch.zeros_like(rh), rh)
        return rh

    # ------------------------------------------------------------ integrator
    def _enforce_admissible(self, U_old, U_new, eps_rel=1e-6):
        rho, u, v, p = _primitive(U_new, self.gamma)
        bad = ~torch.isfinite(U_new).all(dim=-1) | (rho <= eps_rel) | (p <= eps_rel)
        n_bad = int(bad.sum().item())
        if n_bad == 0:
            return U_new
        lo = torch.zeros_like(rho)
        hi = torch.ones_like(rho)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            cand = U_old + mid.unsqueeze(-1) * (U_new - U_old)
            rc, _, _, pc = _primitive(cand, self.gamma)
            ok = torch.isfinite(cand).all(dim=-1) & (rc > eps_rel) & (pc > eps_rel)
            lo = torch.where(bad & ok, mid, lo)
            hi = torch.where(bad & ~ok, mid, hi)
        theta = torch.where(bad, torch.clamp(lo, min=1e-3), torch.ones_like(lo))
        self.n_admissibility_fixes += n_bad
        return U_old + theta.unsqueeze(-1) * (U_new - U_old)

    def step(self, dt):
        U0 = self.U.clone()
        U1 = self._enforce_admissible(U0, U0 + dt * self.rhs(U0))
        U2 = 0.5 * U0 + 0.5 * (U1 + dt * self.rhs(U1))
        self.U = self._enforce_admissible(U0, U2)
        self.t += dt
        return dt

    def advance(self, t_end, max_steps=5_000_000, callback=None, callback_every=0):
        """Integrate to ``t_end``; identical stepping logic to the numpy solver."""
        steps = 0
        while self.t < t_end - 1e-15 and steps < max_steps:
            dt = self.dt_cfl()
            if self.t + dt > t_end:
                dt = t_end - self.t
            U0 = self.U.clone()
            U1 = self._enforce_admissible(U0, U0 + dt * self.rhs(U0))
            U2 = 0.5 * U0 + 0.5 * (U1 + dt * self.rhs(U1))
            if not bool(torch.isfinite(U2).all().item()):
                self.n_positivity_fallbacks += 1
                dt *= 0.5
                continue
            self.U = self._enforce_admissible(U0, U2)
            self.t += dt
            steps += 1
            if callback is not None and callback_every and steps % callback_every == 0:
                callback(self)
        return steps

def to_torch(solver, threads=None, dtype=None):
    """Copy a numpy :class:`EulerSolver2D` (state, solid mask, BCs) into the mirror.

    The numpy solver stays the reference: cases are always *built* by
    ``cases.py`` and then handed over here, so a case definition can never
    diverge between the two backends.
    """
    _require_torch()
    ts = TorchEulerSolver2D(
        nx=solver.nx, ny=solver.ny, lx=solver.lx, ly=solver.ly,
        gamma=solver.gamma, cfl=solver.cfl, x0=solver.x0, y0=solver.y0,
        dtype=dtype, threads=threads,
    )
    ts.U = torch.as_tensor(np.ascontiguousarray(solver.U), dtype=ts.dtype).clone()
    ts.t = float(solver.t)
    if solver.solid is not None:
        ts.solid = torch.as_tensor(np.ascontiguousarray(solver.solid), dtype=torch.bool).clone()
    ts.bc_x0, ts.bc_x1, ts.bc_y0, ts.bc_y1 = solver.bc_x0, solver.bc_x1, solver.bc_y0, solver.bc_y1
    return ts


__all__.append("to_torch")
