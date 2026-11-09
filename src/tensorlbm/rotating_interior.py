"""Rotating-reference-frame *interior* flows and their analytic anchors.

Two verification surfaces that the plate-Ekman / Taylor-column benchmarks in
:mod:`tensorlbm.rotating_frame` do not cover:

1. **Rotating-frame Taylor-Couette flow** (annular, curved, closed geometry).
   In the frame rotating with ``Omega_f`` about the cylinder axis the steady
   axisymmetric azimuthal momentum balance is *purely viscous* -- the Coriolis
   acceleration ``-2 Omega x u`` of a purely azimuthal field is radial -- so

   .. math:: u_\\theta(r) = A r + B / r, \\quad
             A = \\frac{\\Omega_o r_o^2-\\Omega_i r_i^2}{r_o^2-r_i^2},\\quad
             B = \\frac{r_i^2 r_o^2 (\\Omega_i-\\Omega_o)}{r_o^2-r_i^2}

   is *unchanged* by the frame rotation, while the reduced pressure is not:

   .. math::
      \\frac{dp}{dr} = \\rho\\Big(\\frac{u_\\theta^2}{r}
                       + 2\\Omega_f u_\\theta + \\Omega_f^2 r\\Big)
      \\;\\Longrightarrow\\;
      p(r)-p(r_i) = \\rho\\Big[(A^2+2\\Omega_f A+\\Omega_f^2)\\frac{r^2-r_i^2}{2}
        + 2B(A+\\Omega_f)\\ln\\frac{r}{r_i}
        - \\frac{B^2}{2}\\Big(\\frac{1}{r^2}-\\frac{1}{r_i^2}\\Big)\\Big] .

   The **pressure anchor is the discriminating one**: a wrong Coriolis sign or
   factor, or a missing centrifugal term, leaves the velocity profile almost
   intact but moves ``p(r)`` by an amount of the same order as the physical
   effect.  (The centrifugal part alone is a pure gradient and therefore only a
   pressure, never a velocity effect -- see :mod:`tensorlbm.rotating_frame`.)

2. **Rotating plane channel -> Ekman secondary flow.**  A plane channel with
   walls at ``y = +-h``, driven by a streamwise body force ``G e_x``, rotating
   about the *wall-normal* axis ``Omega e_y``.  Coriolis couples the two
   in-plane components and the exact steady solution is

   .. math:: w(y) \\equiv u_x - i u_z = w_p\\big[1-\\cosh(ky)/\\cosh(kh)\\big],
             \\quad w_p = -\\frac{i G}{2\\Omega},\\quad
             k = \\frac{1+i}{\\delta},\\ \\delta=\\sqrt{\\nu/\\Omega},

   i.e. **two Ekman layers of thickness** ``delta = sqrt(nu/Omega) =
   sqrt(2 nu/f)`` (``f = 2 Omega`` is the Coriolis parameter; the campaign
   convention ``delta_E = sqrt(2 nu / Omega)`` is this same formula with
   ``Omega`` read as the Coriolis parameter ``f``), separated by a geostrophic
   interior where ``u_x -> 0`` and ``u_z -> G/(2 Omega)``: rotation blocks the
   forced direction (Taylor-Proudman) and pushes the transport into the
   cross-stream geostrophic jet.  No free amplitude -- ``w_p`` is fixed by the
   inputs -- so fitting ``delta`` is a one-parameter blind measurement.

The annular solver is deliberately ``numpy`` (float64): the same step in
``torch`` costs ~13 ms/step at 60x60 because every lattice direction is a
separate kernel launch, while numpy does the whole 9-direction sweep in one
vectorised pass (~0.3 ms/step).  Equivalence of the force assembly with the
library's ``coriolis_accel`` / ``centrifugal_accel`` is asserted in
``tests/test_rotating_interior.py`` instead of being assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .rotating_frame import RotatingFrameLBM, centrifugal_accel, coriolis_accel

__all__ = [
    "CS2",
    "circular_couette",
    "couette_pressure_rotating_frame",
    "fit_couette_AB",
    "invert_effective_radii",
    "RotatingAnnulusD2Q9",
    "rotating_channel_analytic",
    "fd_rotating_channel",
    "fit_channel_ekman_delta",
    "run_rotating_channel_lbm",
    "stream_function_field",
    "inertial_angular_momentum",
    "angular_momentum_residual",
]

CS2 = 1.0 / 3.0

_C9 = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
_W9 = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
_OPP9 = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


# --------------------------------------------------------------------------- #
# 1. Analytic anchors                                                         #
# --------------------------------------------------------------------------- #
def circular_couette(r, ri: float, ro: float, omega_i: float, omega_o: float = 0.0) -> dict:
    """Steady circular Couette flow ``u_theta = A r + B/r``.

    ``omega_i``/``omega_o`` are the wall angular velocities *in the frame used
    by the solver*; the frame rotation does not enter the azimuthal solution
    (Coriolis is purely radial there) -- which is exactly the statement
    verified here.
    """
    r = np.asarray(r, dtype=float)
    A = (omega_o * ro**2 - omega_i * ri**2) / (ro**2 - ri**2)
    B = ri**2 * ro**2 * (omega_i - omega_o) / (ro**2 - ri**2)
    return {"A": float(A), "B": float(B), "u_theta": A * r + B / r}


def couette_pressure_rotating_frame(
    r,
    ri: float,
    ro: float,
    omega_i: float,
    omega_o: float = 0.0,
    omega_frame: float = 0.0,
    rho: float = 1.0,
) -> np.ndarray:
    """Reduced pressure ``p(r) - p(r_i)`` in the frame rotating at ``omega_frame``.

    Radial balance ``dp/dr = rho (u^2/r + 2 Omega u + Omega^2 r)`` integrated in
    closed form.  ``2 Omega u`` is the Coriolis contribution and ``Omega^2 r``
    the centrifugal one; both vanish at ``omega_frame = 0``, reducing the
    expression to the classical inertial-frame Couette pressure.
    """
    r = np.asarray(r, dtype=float)
    c = circular_couette(r, ri, ro, omega_i, omega_o)
    A, B, W = c["A"], c["B"], float(omega_frame)
    k1 = A * A + 2.0 * W * A + W * W
    return rho * (
        k1 * (r**2 - ri**2) / 2.0
        + 2.0 * B * (A + W) * np.log(r / ri)
        - 0.5 * B * B * (1.0 / r**2 - 1.0 / ri**2)
    )


def fit_couette_AB(r, u_theta, weight=None) -> dict:
    """Weighted least squares ``u = A r + B/r`` (two parameters, exact family)."""
    r = np.asarray(r, dtype=float)
    u = np.asarray(u_theta, dtype=float)
    w = np.ones_like(r) if weight is None else np.asarray(weight, dtype=float)
    M = np.stack([r, 1.0 / r], axis=1)
    sw = np.sqrt(w)[:, None]
    sol, *_ = np.linalg.lstsq(M * sw, u * np.sqrt(w), rcond=None)
    pred = M @ sol
    resid = float(np.sqrt(np.sum(w * (pred - u) ** 2) / np.sum(w)))
    return {"A": float(sol[0]), "B": float(sol[1]), "residual_rms": resid}


def invert_effective_radii(A: float, B: float, omega_i: float) -> dict:
    """Effective (staircase-corrected) radii from the no-slip conditions.

    ``u(r_o,eff) = 0`` gives ``r_o,eff = sqrt(-B/A)``; ``u(r_i,eff) =
    omega_i r_i,eff`` gives ``r_i,eff = sqrt(B/(omega_i - A))``.  Same inversion
    as the verified ``benchmarks/verified/taylor_couette`` case: for a staircase
    annulus the nominal radii are *not* the effective ones, so comparing against
    them measures the staircase, not the solver.
    """
    ri_eff = math.sqrt(B / (omega_i - A)) if omega_i - A > 0 and B > 0 else float("nan")
    ro_eff = math.sqrt(-B / A) if (B > 0 and A < 0) else float("nan")
    return {"ri_eff": ri_eff, "ro_eff": ro_eff}


# --------------------------------------------------------------------------- #
# 2. Rotating-frame annular Taylor-Couette solver (D2Q9, moving walls)         #
# --------------------------------------------------------------------------- #
class RotatingAnnulusD2Q9:
    """D2Q9 BGK in a frame rotating at ``omega_frame`` about the z axis.

    * pseudo-forces ``a_cor = -2 Omega x u`` and ``a_cf = Omega^2 r``, assembled
      exactly as :func:`tensorlbm.rotating_frame.coriolis_accel` /
      :func:`~tensorlbm.rotating_frame.centrifugal_accel` do (equivalence
      asserted in the unit tests);
    * Guo (2002) second-order forcing with the half-step velocity from a
      two-iteration fixed point (same scheme as ``RotatingFrameLBM``);
    * walls: half-way bounce-back with the Ladd momentum correction
      ``f_k(x) = f_opp(k)(x) + 2 w_k rho (c_k . u_wall)/cs^2``, so the cylinders
      rotate with ``omega_i`` / ``omega_o`` *in the rotating frame*.  A sign
      error here reverses the drag and is caught by the analytic profile test.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        tau: float,
        ri: float,
        ro: float,
        omega_i: float,
        omega_o: float = 0.0,
        omega_frame: float = 0.0,
        center: tuple[float, float] | None = None,
        centrifugal: bool = True,
        n_guo_iter: int = 2,
        rho0: float = 1.0,
    ) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.tau = float(tau)
        self.nu = (self.tau - 0.5) / 3.0
        self.ri, self.ro = float(ri), float(ro)
        self.omega_i, self.omega_o = float(omega_i), float(omega_o)
        self.omega_frame = float(omega_frame)
        self.centrifugal = bool(centrifugal)
        self.n_guo_iter = int(n_guo_iter)
        self.rho0 = float(rho0)
        self.steps = 0
        ny, nx = self.shape
        self.yc, self.xc = (
            ((ny - 1) / 2.0, (nx - 1) / 2.0) if center is None else (float(center[0]), float(center[1]))
        )
        self.yy, self.xx = np.meshgrid(
            np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij"
        )
        self.dx = self.xx - self.xc
        self.dy = self.yy - self.yc
        self.rad = np.sqrt(self.dx**2 + self.dy**2)

        inner = self.rad < self.ri
        outer = self.rad > self.ro
        self.solid = inner | outer
        uwx = np.where(inner, -self.omega_i * self.dy, np.where(outer, -self.omega_o * self.dy, 0.0))
        uwy = np.where(inner, self.omega_i * self.dx, np.where(outer, self.omega_o * self.dx, 0.0))
        self._uwx, self._uwy = uwx, uwy

        self._shifts = [(int(c[1]), int(c[0])) for c in _C9]  # (dy, dx)
        self._nb_solid = [np.roll(self.solid, self._shifts[k], axis=(0, 1)) for k in range(9)]
        self._nb_uwx = [np.roll(uwx, self._shifts[k], axis=(0, 1)) for k in range(9)]
        self._nb_uwy = [np.roll(uwy, self._shifts[k], axis=(0, 1)) for k in range(9)]
        self._cx = _C9[:, 0].reshape(9, 1, 1).astype(float)
        self._cy = _C9[:, 1].reshape(9, 1, 1).astype(float)
        self._w = _W9.reshape(9, 1, 1)

        self.f = self._equilibrium(
            np.full(self.shape, self.rho0), np.zeros(self.shape), np.zeros(self.shape)
        )

    def _equilibrium(self, rho, ux, uy) -> np.ndarray:
        cu = self._cx * ux[None] + self._cy * uy[None]
        usq = ux * ux + uy * uy
        return rho[None] * self._w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])

    def _macroscopic(self):
        f = self.f
        rho = f.sum(axis=0)
        ux = (f * self._cx).sum(axis=0) / rho
        uy = (f * self._cy).sum(axis=0) / rho
        return rho, ux, uy

    def _force(self, ux, uy):
        """Pseudo-force assembly (same algebra as the library accel helpers)."""
        W = self.omega_frame
        ax = 2.0 * W * uy
        ay = -2.0 * W * ux
        if self.centrifugal and W != 0.0:
            ax = ax + W * W * self.dx
            ay = ay + W * W * self.dy
        return ax, ay

    def _guo(self, rho, ux, uy, Fx, Fy) -> np.ndarray:
        cu = self._cx * ux[None] + self._cy * uy[None]
        cF = self._cx * Fx[None] + self._cy * Fy[None]
        uF = ux * Fx + uy * Fy
        return (1.0 - 0.5 / self.tau) * self._w * (3.0 * cF - 3.0 * uF[None] + 9.0 * cu * cF)

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            rho, ux_r, uy_r = self._macroscopic()
            inv_rho = 1.0 / rho
            Fx, Fy = self._force(ux_r, uy_r)
            for _it in range(max(0, self.n_guo_iter - 1)):
                ux = ux_r + 0.5 * Fx * inv_rho
                uy = uy_r + 0.5 * Fy * inv_rho
                Fx, Fy = self._force(ux, uy)
            ux = ux_r + 0.5 * Fx * inv_rho
            uy = uy_r + 0.5 * Fy * inv_rho

            feq = self._equilibrium(rho, ux, uy)
            f_post = self.f + (feq - self.f) * (1.0 / self.tau) + self._guo(rho, ux, uy, Fx, Fy)
            f_post[:, self.solid] = self.f[:, self.solid]

            f_new = np.empty_like(f_post)
            for k in range(9):
                streamed = np.roll(f_post[k], self._shifts[k], axis=(0, 1))
                cu_w = self._cx[k, 0, 0] * self._nb_uwx[k] + self._cy[k, 0, 0] * self._nb_uwy[k]
                refl = f_post[_OPP9[k]] + 2.0 * _W9[k] * rho * cu_w * 3.0
                f_new[k] = np.where(self._nb_solid[k], refl, streamed)
            eq0 = self._equilibrium(
                np.full(self.shape, self.rho0), np.zeros(self.shape), np.zeros(self.shape)
            )
            f_new[:, self.solid] = eq0[:, self.solid]
            self.f = f_new
            self.steps += 1

    # -- observables --------------------------------------------------------- #
    def fields(self):
        return self._macroscopic()

    def initialize_from_velocity(self, ux, uy, rho: float | None = None) -> None:
        r = self.rho0 if rho is None else float(rho)
        self.f = self._equilibrium(np.full(self.shape, r), np.asarray(ux), np.asarray(uy))

    def azimuthal_profile(self) -> dict:
        """Shell-averaged ``(r, u_theta, u_r, p)`` over the fluid annulus.

        Bins are ``floor(d)`` shells, so the wall-adjacent staircase shells are
        resolved individually (they are excluded from the fit by the caller).
        """
        rho, ux, uy = self.fields()
        fluid = ~self.solid
        r_safe = np.where(self.rad > 0, self.rad, 1.0)
        u_th = (-self.dy * ux + self.dx * uy) / r_safe
        u_rad = (self.dx * ux + self.dy * uy) / r_safe
        kk = np.floor(self.rad).astype(np.int64)
        rs, uth, urad, ps, cnt = [], [], [], [], []
        for k in range(int(self.ro) + 2):
            sel = fluid & (kk == k)
            n = int(sel.sum())
            rs.append(k + 0.5)
            cnt.append(n)
            if n == 0:
                uth.append(float("nan"))
                urad.append(float("nan"))
                ps.append(float("nan"))
            else:
                uth.append(float(u_th[sel].mean()))
                urad.append(float(u_rad[sel].mean()))
                ps.append(float((CS2 * rho[sel]).mean()))
        return {
            "r_bin": rs,
            "count": cnt,
            "u_theta": uth,
            "u_rad": urad,
            "p": ps,
            "Ma": float(np.max(np.sqrt(ux[fluid] ** 2 + uy[fluid] ** 2)) / math.sqrt(CS2)),
        }

    def run_until_steady(self, tol=1e-8, max_steps=200_000, check_every=200, period=None) -> dict:
        """Integrate until the (period-averaged) velocity stops changing."""
        fluid = ~self.solid
        history = []
        residual = float("inf")
        if period is None:
            prev = np.stack(self.fields()[1:3])
            while self.steps < max_steps:
                self.step(check_every)
                cur = np.stack(self.fields()[1:3])
                residual = float(np.max(np.abs(cur - prev)[:, fluid]))
                history.append({"steps": self.steps, "max_du": residual})
                prev = cur
                if residual < tol:
                    break
        else:
            period = max(int(period), 2)
            check_every = max(1, int(round(period / 8)))
            buf = [(self.steps, np.stack(self.fields()[1:3]))]
            while self.steps < max_steps:
                self.step(check_every)
                cur = np.stack(self.fields()[1:3])
                buf.append((self.steps, cur.copy()))
                target = self.steps - period
                while len(buf) > 1 and buf[1][0] <= target:
                    buf.pop(0)
                residual = float(np.max(np.abs(cur - buf[0][1])[:, fluid]))
                history.append({"steps": self.steps, "compare_to": buf[0][0], "max_du": residual})
                if residual < tol and self.steps >= period:
                    break
        return {
            "steps": self.steps,
            "residual": residual,
            "tol": tol,
            "period": period,
            "converged": bool(residual < tol),
            "history": history,
        }


# --------------------------------------------------------------------------- #
# 3. Rotating plane channel: exact solution, FD reference, delta measurement  #
# --------------------------------------------------------------------------- #
def rotating_channel_analytic(y, h: float, G: float, Omega: float, nu: float):
    """Exact steady rotating-channel solution ``(u_x, u_z)`` for walls at ``y=+-h``."""
    y = np.asarray(y, dtype=float)
    w_p = -1j * G / (2.0 * Omega)
    k = (1.0 + 1.0j) * math.sqrt(Omega / nu)
    w = w_p * (1.0 - np.cosh(k * y) / np.cosh(k * h))
    return w.real, -w.imag


def fd_rotating_channel(G: float, Omega: float, nu: float, h: float, n: int = 1201) -> dict:
    """Independent FD reference ("numerical DNS") of the same rotating NS problem.

    Solves the coupled system

        nu u_x'' - 2 Omega u_z = -G
        nu u_z'' + 2 Omega u_x = 0,      u_x(+-h) = u_z(+-h) = 0

    with second-order central differences and a direct solve.  It validates the
    closed form (algebra slip-through guard) and serves as the direct numerical
    reference for the LBM comparison.
    """
    ys = np.linspace(-h, h, n)
    dy = ys[1] - ys[0]
    m = n - 2
    D2 = (
        np.diag(np.full(m, -2.0))
        + np.diag(np.full(m - 1, 1.0), 1)
        + np.diag(np.full(m - 1, 1.0), -1)
    ) / dy**2
    I = np.eye(m)
    A = np.block([[nu * D2, -2.0 * Omega * I], [2.0 * Omega * I, nu * D2]])
    b = np.zeros(2 * m)
    b[:m] = -G
    sol = np.linalg.solve(A, b)
    ux = np.zeros(n)
    uz = np.zeros(n)
    ux[1:-1] = sol[:m]
    uz[1:-1] = sol[m:]
    return {"y": ys, "ux": ux, "uz": uz, "n": n}


def fit_channel_ekman_delta(
    y, uz, h: float, G: float, Omega: float, nu: float, n_scan: int = 400
) -> dict:
    """Blind one-parameter fit of the Ekman thickness to the measured ``u_z(y)``.

    The amplitude ``w_p = -iG/(2 Omega)`` is fixed by the inputs, so ``delta``
    is the only free parameter (``delta`` enters through ``k = (1+i)/delta``,
    i.e. an effective viscosity ``nu_eff = delta^2 Omega``): a 1-D scan plus
    golden-section refinement.
    """
    y = np.asarray(y, dtype=float)
    uz = np.asarray(uz, dtype=float)
    d_th = math.sqrt(nu / Omega)
    lo, hi = 0.25 * d_th, 3.0 * d_th

    def cost(delta: float) -> float:
        nu_eff = Omega * delta * delta
        _, uz_m = rotating_channel_analytic(y, h, G, Omega, nu_eff)
        return float(np.sqrt(np.mean((uz_m - uz) ** 2)))

    gr = (math.sqrt(5.0) - 1.0) / 2.0
    grid = np.linspace(lo, hi, n_scan)
    costs = [cost(float(d)) for d in grid]
    k = int(np.argmin(costs))
    a, bb = float(grid[max(k - 1, 0)]), float(grid[min(k + 1, n_scan - 1)])
    c1, c2 = bb - gr * (bb - a), a + gr * (bb - a)
    f1, f2 = cost(c1), cost(c2)
    for _ in range(100):
        if abs(bb - a) < 1e-12:
            break
        if f1 < f2:
            bb, c2, f2 = c2, c1, f1
            c1 = bb - gr * (bb - a)
            f1 = cost(c1)
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + gr * (bb - a)
            f2 = cost(c2)
    delta = 0.5 * (a + bb)
    return {
        "delta_fit": float(delta),
        "delta_theory": float(d_th),
        "rms": float(cost(delta)),
        "bounds": [lo, hi],
    }


def run_rotating_channel_lbm(
    h: int = 24,
    nu: float = 0.1,
    delta_nodes: float = 8.0,
    G: float | None = None,
    uz_core_target: float = 0.02,
    nx: int = 4,
    nz: int = 3,
    init: str = "analytic",
    threads: int | None = 1,
    tol: float = 1e-7,
    max_steps: int = 40_000,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """LBM rotating plane channel (D3Q19, rotation about the wall-normal axis).

    Walls at ``y = 0`` and ``y = ny-1`` (half-way bounce-back => nominal wall
    positions ``0.5`` and ``ny-1.5``, half width ``h``); periodic in ``x`` and
    ``z``; uniform body force ``G`` in ``x`` (equivalent to a pressure
    gradient); frame angular velocity ``Omega = nu/delta_nodes**2`` about
    ``y``.  ``init="analytic"`` starts from the exact solution (fixed-point
    test: the discrete steady state must not drift away from it); ``init="rest"``
    is the independent start-from-rest recomputation.

    ``threads=1`` is set by default: on a 128-core host the per-op thread
    overhead dominates these 10^3-cell grids by an order of magnitude.
    """
    if threads is not None:
        torch.set_num_threads(int(threads))
    Omega = nu / delta_nodes**2
    if G is None:
        G = 2.0 * Omega * uz_core_target
    ny = 2 * h + 2
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[:, 0, :] = True
    solid[:, ny - 1, :] = True
    sim = RotatingFrameLBM(
        shape=(nz, ny, nx),
        tau=3.0 * nu + 0.5,
        omega=(0.0, Omega, 0.0),
        body_force=(G, 0.0, 0.0),
        solid=solid,
        centrifugal=False,  # gradient field: absorbed into the reduced pressure
        n_guo_iter=3,
        dtype=dtype,
        rho0=1.0,
    )
    if init == "analytic":
        y = np.arange(ny, dtype=float) - (ny - 1) / 2.0
        ux0, uz0 = rotating_channel_analytic(y, h, G, Omega, nu)
        shape = (nz, ny, nx)
        sim.initialize_from_velocity(
            torch.as_tensor(np.broadcast_to(ux0[None, :, None], shape).copy(), dtype=dtype),
            torch.zeros(shape, dtype=dtype),
            torch.as_tensor(np.broadcast_to(uz0[None, :, None], shape).copy(), dtype=dtype),
        )
    elif init != "rest":
        raise ValueError("init must be 'analytic' or 'rest'")
    period = int(round(2.0 * math.pi / (2.0 * Omega)))
    conv = sim.run_until_steady(tol=tol, max_steps=max_steps, period=period)
    rho, ux, uy, uz = sim.fields()
    return {
        "h": h,
        "nu": nu,
        "Omega": Omega,
        "G": G,
        "delta_theory": math.sqrt(nu / Omega),
        "ny": ny,
        "init": init,
        "steps": conv["steps"],
        "residual": conv["residual"],
        "converged": conv["converged"],
        "converge_tol": tol,
        "period": period,
        "y": [j - h for j in range(1, ny - 1)],
        "ux": [float(v) for v in ux[:, 1:-1, :].mean(dim=(0, 2))],
        "uy": [float(v) for v in uy[:, 1:-1, :].mean(dim=(0, 2))],
        "uz": [float(v) for v in uz[:, 1:-1, :].mean(dim=(0, 2))],
        "rho": [float(v) for v in rho[:, 1:-1, :].mean(dim=(0, 2))],
    }


# --------------------------------------------------------------------------- #
# 4. Angular-momentum bookkeeping of the rotation force operator               #
# --------------------------------------------------------------------------- #
def stream_function_field(
    ny: int, nx: int, seed: int = 0, k_max: int = 3, shear: float = 0.0
):
    """Discretely divergence-free periodic velocity field from a stream function.

    Central differences commute on a periodic grid, so
    ``div u = d_x d_y psi - d_y d_x psi = 0`` *exactly* (to round-off).  That is
    what makes the angular-momentum identity below a machine-precision test and
    not a truncation-error observation.  ``shear`` adds ``u_x += shear * y``,
    which is also exactly divergence-free and gives the field a non-zero
    angular-momentum moment (a random superposition can cancel it).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    psi = np.zeros((ny, nx))
    for kx in range(1, k_max + 1):
        for ky in range(1, k_max + 1):
            amp = rng.normal() / (kx * ky)
            psi += (
                amp
                * np.sin(2 * np.pi * kx * xx / nx + rng.uniform(0, 2 * np.pi))
                * np.cos(2 * np.pi * ky * yy / ny + rng.uniform(0, 2 * np.pi))
            )
    ux = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / 2.0
    uy = -(np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / 2.0
    uy = uy + shear * (1.0 + np.cos(2.0 * np.pi * xx / nx))
    return ux, uy


def inertial_angular_momentum(rho, ux, uy, x, y, Omega: float) -> float:
    """Angular momentum about the z axis of the *inertial* velocity field.

    ``L = sum rho [ (x u_y - y u_x) + Omega (x^2 + y^2) ]``; the second term is
    the rigid-body contribution of the frame rotation.
    """
    return float(np.sum(rho * ((x * uy - y * ux) + Omega * (x * x + y * y))))


def angular_momentum_residual(rho, ux, uy, x, y, Omega: float) -> dict:
    """Torque budget of the pseudo-forces about the rotation axis.

    The Coriolis force exerts an axial torque ``T_c = sum rho (x F_y - y F_x)``
    with ``F = -2 Omega x u``; the centrifugal force is radial, hence
    ``T_cf = sum rho (r x Omega^2 r) = 0`` exactly.  For a divergence-free
    field the inertial angular momentum of the *same* discrete field changes at
    ``dI/dt = 2 sum rho (x u_x + y u_y)``, so the exact identity is

        T_c + Omega * dI/dt = 0.

    This is a functional test of the *implementation* (the force fields come
    from :func:`tensorlbm.rotating_frame.coriolis_accel` /
    ``centrifugal_accel``): a sign flip or a wrong factor of 2 breaks it at
    O(1), a correct one leaves round-off.
    """
    ux_t = torch.as_tensor(ux, dtype=torch.float64)
    uy_t = torch.as_tensor(uy, dtype=torch.float64)
    Fx_c, Fy_c, _ = coriolis_accel(ux_t, uy_t, torch.zeros_like(ux_t), (0.0, 0.0, float(Omega)))
    Fx_cf, Fy_cf, _ = centrifugal_accel(
        torch.as_tensor(x, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        None,
        (0.0, 0.0, float(Omega)),
    )
    rho = np.asarray(rho, dtype=float)
    ux = np.asarray(ux, dtype=float)
    uy = np.asarray(uy, dtype=float)
    Fx = (Fx_c + Fx_cf).numpy()
    Fy = (Fy_c + Fy_cf).numpy()
    T_c = float(np.sum(rho * (x * Fy - y * Fx)))
    dI = 2.0 * float(np.sum(rho * (x * ux + y * uy)))
    balance = T_c + float(Omega) * dI
    scale = max(abs(T_c), abs(float(Omega) * dI), 1e-300)
    return {
        "torque_coriolis_plus_centrifugal": T_c,
        "Omega_dIdt": float(Omega) * dI,
        "balance": balance,
        "residual_rel": abs(balance) / scale,
        "L_inertial": inertial_angular_momentum(rho, ux, uy, x, y, float(Omega)),
    }