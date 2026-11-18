"""Two-distribution D2Q9 solver for air-layer (gas/liquid) drag reduction.

Model
-----
Hydrodynamics: pressure-based variable-density incompressible lattice Boltzmann
(He-Luo equilibria with a pressure zeroth moment) plus Guo forcing.  The mixture
density and viscosity are slaved to the order parameter ``phi`` (+1 liquid,
-1 gas), which is standard for phase-field LBM at high density ratio: mass
conservation is then carried by the *conservative* phase-transport equation.

Phase transport: conservative Allen-Cahn equation

    d_t phi + div(phi u) = div[ M ( grad phi - (1-phi^2)/delta * n ) ],  n = grad phi/|grad phi|

solved with a second BGK population.  The RHS is a flux form, so the scheme is
telescoping-conservative: sum(phi) is conserved to round-off (no phase source
inside the domain).  The `delta` parameter is tied to the tanh interface width
``W = 4 delta`` so that ``grad phi = (1-phi^2)/delta * n`` holds at equilibrium.

Surface tension: Cahn-Hilliard (Korteweg) force ``F_s = mu_phi grad phi`` with
``mu_phi = 4 beta (phi^3 - phi) - kappa lap(phi)``.  For the double-well
``f0 = beta (phi^2-1)^2`` the equilibrium interface is ``phi = tanh(s/delta)``,
``delta = sqrt(kappa/(2 beta))`` and the surface tension is

    sigma = (4/3) sqrt(2 kappa beta) = (8/3) beta delta.

Lattice units: cs^2 = 1/3, dx = dt = 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float64)
W9 = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float64)
OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)
CS2 = 1.0 / 3.0

# direction sets that hit the bottom (cy<0) / top (cy>0) wall
_TO_BOTTOM = (4, 7, 8)
_TO_TOP = (2, 5, 6)


def _opp(i: int) -> int:
    return OPP[i]


@dataclass(frozen=True)
class PhaseFieldParams:
    """Physical parameters (lattice units)."""

    rho_l: float = 1.0
    rho_g: float = 0.01          # density ratio 100 (>=100 required by the gate)
    nu_l: float = 0.05
    nu_g: float = 0.35           # nu_g/nu_l = 7; mu_g/mu_l = 0.07
    sigma: float = 0.01
    interface_width: float = 4.0  # 4*delta, must be >= 3 lattice nodes
    mobility: float = 0.05
    inertia_relax: float = 1.0

    @property
    def delta(self) -> float:
        return 0.25 * self.interface_width

    @property
    def beta(self) -> float:
        # sigma = (8/3) beta delta
        return 3.0 * self.sigma / (8.0 * self.delta)

    @property
    def kappa(self) -> float:
        return 2.0 * self.beta * self.delta ** 2

    @property
    def tau_phi(self) -> float:
        return self.mobility / CS2 + 0.5

    @property
    def density_ratio(self) -> float:
        return self.rho_l / self.rho_g

    @property
    def mu_g_over_mu_l(self) -> float:
        return (self.rho_g * self.nu_g) / (self.rho_l * self.nu_l)


@dataclass
class StepDiagnostics:
    step: int
    phi_total: float
    mass_total: float
    u_max: float
    p_mean: float
    ke: float


def _grad(field2d: np.ndarray, periodic_y: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradient: periodic in x, Neumann or periodic in y."""
    gx = 0.5 * (np.roll(field2d, -1, axis=1) - np.roll(field2d, 1, axis=1))
    if periodic_y:
        gy = 0.5 * (np.roll(field2d, -1, axis=0) - np.roll(field2d, 1, axis=0))
        return gx, gy
    gy = np.zeros_like(field2d)
    gy[1:-1, :] = 0.5 * (field2d[2:, :] - field2d[:-2, :])
    gy[0, :] = field2d[1, :] - field2d[0, :]
    gy[-1, :] = field2d[-1, :] - field2d[-2, :]
    return gx, gy


def _lap(field2d: np.ndarray, periodic_y: bool = False) -> np.ndarray:
    """5-point Laplacian: periodic in x, Neumann or periodic in y."""
    xm = np.roll(field2d, 1, axis=1)
    xp = np.roll(field2d, -1, axis=1)
    if periodic_y:
        return xp + xm + np.roll(field2d, 1, axis=0) + np.roll(field2d, -1, axis=0) - 4.0 * field2d
    ym = np.empty_like(field2d)
    yp = np.empty_like(field2d)
    ym[1:, :] = field2d[:-1, :]
    ym[0, :] = field2d[0, :]
    yp[:-1, :] = field2d[1:, :]
    yp[-1, :] = field2d[-1, :]
    return xp + xm + yp + ym - 4.0 * field2d


def _stream(dist: np.ndarray) -> np.ndarray:
    """Streaming; y-shifts wrap, the wrapped (solid) rows are overwritten by BC."""
    out = np.empty_like(dist)
    for i in range(9):
        out[i] = np.roll(np.roll(dist[i], int(CY[i]), axis=0), int(CX[i]), axis=1)
    return out


class _BaseChannel:
    """Shared D2Q9 machinery (hydrodynamic populations + wall bookkeeping)."""

    def __init__(self, nx: int, ny: int, *, walls: bool = True, x_periodic: bool = True):
        if nx < 8 or ny < 8:
            raise ValueError("need at least 8x8 lattice nodes")
        self.nx = nx
        self.ny = ny
        self.walls = walls
        self.x_periodic = x_periodic
        self.f = np.zeros((9, ny, nx), dtype=np.float64)
        self.j, self.i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

    # -- wall handling ---------------------------------------------------
    def _apply_wall_bc(self, streamed: np.ndarray, post: np.ndarray) -> np.ndarray:
        """On-site bounce-back: replace the wrapped (outside) links by reflections
        of the post-collision outgoing populations at the same wall node."""
        if not self.walls:
            return streamed
        for i in _TO_TOP:               # bottom row: links entering from below
            streamed[i, 0, :] = post[_opp(i), 0, :]
        for i in _TO_BOTTOM:            # top row: links entering from above
            streamed[i, -1, :] = post[_opp(i), -1, :]
        return streamed

    def _wall_shear(self, post: np.ndarray) -> tuple[float, float]:
        """Momentum-exchange wall shear per unit length (force/length)."""
        fx_bottom = 2.0 * sum(CX[i] * post[i, 0, :] for i in _TO_BOTTOM)
        fx_top = 2.0 * sum(CX[i] * post[i, -1, :] for i in _TO_TOP)
        return float(np.sum(fx_bottom)) / self.nx, float(np.sum(fx_top)) / self.nx

    @staticmethod
    def _equilibrium_f(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray,
                       pbar: np.ndarray) -> np.ndarray:
        """Pressure-based variable-density equilibrium (sum = pbar, momentum = rho*u)."""
        cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
        u2 = ux ** 2 + uy ** 2
        feq = (W9[:, None, None] * pbar[None]
               + W9[:, None, None] * rho[None] * (3.0 * cu + 4.5 * cu ** 2 - 1.5 * u2[None]))
        return feq

    @staticmethod
    def _guo_source(fx: np.ndarray, fy: np.ndarray, ux: np.ndarray, uy: np.ndarray
                    ) -> np.ndarray:
        cx = CX[:, None, None]
        cy = CY[:, None, None]
        cuf = cx * fx[None] + cy * fy[None]
        cu = cx * ux[None] + cy * uy[None]
        return W9[:, None, None] * (3.0 * cuf - 3.0 * (ux[None] * fx[None] + uy[None] * fy[None])
                                    + 9.0 * cu * cuf)

    def momentum(self, force_x: np.ndarray, force_y: np.ndarray, rho: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Velocity and pressure from the current populations (half-force step)."""
        mx = sum(CX[i] * self.f[i] for i in range(9)) + 0.5 * force_x
        my = sum(CY[i] * self.f[i] for i in range(9)) + 0.5 * force_y
        pbar = sum(self.f[i] for i in range(9))
        ux = mx / rho
        uy = my / rho
        return ux, uy, pbar

    def collide_stream(self, ux, uy, pbar, rho, nu, fx, fy) -> tuple[float, float]:
        feq = self._equilibrium_f(rho, ux, uy, pbar)
        tau = 3.0 * np.clip(nu, 1e-6, None) + 0.5
        src = self._guo_source(fx, fy, ux, uy) * (1.0 - 0.5 / tau)[None]
        post = self.f - (self.f - feq) / tau[None] + src
        shear = self._wall_shear(post) if self.walls else (0.0, 0.0)
        self.f = self._apply_wall_bc(_stream(post), post)
        return shear


class SinglePhaseChannel(_BaseChannel):
    """Single-phase D2Q9 channel used as the analytic friction calibration."""

    def __init__(self, nx: int, ny: int, *, rho: float = 1.0, nu: float = 0.05,
                 body_force_x: float = 0.0):
        super().__init__(nx, ny, walls=True)
        self.rho = float(rho)
        self.nu = float(nu)
        self.gx = float(body_force_x)
        feq = self._equilibrium_f(np.full((ny, nx), rho), np.zeros((ny, nx)),
                                  np.zeros((ny, nx)), np.full((ny, nx), rho * CS2))
        self.f[:] = feq
        self.ux = np.zeros((ny, nx))
        self._rho_arr = np.full((ny, nx), self.rho)
        self._zero = np.zeros((ny, nx))

    def step(self, n: int = 1) -> tuple[float, float]:
        tau = 3.0 * self.nu + 0.5
        fx = np.full((self.ny, self.nx), self.gx)
        for _ in range(n):
            ux, uy, pbar = self.momentum(fx, self._zero, self._rho_arr)
            self.ux = ux
            self.shear = self.collide_stream(ux, uy, pbar, self._rho_arr,
                                             np.full_like(ux, self.nu), fx, self._zero)
        return self.shear

    # -- observables -----------------------------------------------------
    @property
    def height(self) -> float:
        """Gap between the two no-slip planes (half-way bounce-back: ny cells)."""
        return float(self.ny)

    def bulk_velocity(self) -> float:
        """Mean velocity over the full gap (trapezoid with zero at both walls)."""
        return float(np.sum(self.ux) / (self.nx * self.height))

    def wall_shear_mean(self) -> float:
        return 0.5 * (abs(self.shear[0]) + abs(self.shear[1]))

    def analytic_cf(self) -> float:
        """Cf = 12/Re for plane Poiseuille flow (Re based on gap and bulk velocity)."""
        re = self.rho * self.bulk_velocity() * self.height / self.nu
        return 12.0 / re

    def simulated_cf(self) -> float:
        ub = self.bulk_velocity()
        return 2.0 * self.wall_shear_mean() / (self.rho * ub ** 2)

    def profile_error(self) -> float:
        """Relative L2 error of u(y) against the analytic parabola."""
        ub = self.bulk_velocity()
        h = 0.5 * self.height
        y = (np.arange(self.ny) + 0.5) - h
        # wall planes at y = +-h; u = (3 ub / (2 h^2)) (h^2 - y^2)
        ana = (3.0 * ub / (2.0 * h ** 2)) * (h ** 2 - y ** 2)
        num = self.ux[:, :].mean(axis=1)
        denom = np.sqrt(np.mean(ana ** 2))
        return float(np.sqrt(np.mean((num - ana) ** 2)) / denom)


class PhaseFieldChannel(_BaseChannel):
    """Two-phase channel (or periodic box when ``walls=False``)."""

    def __init__(self, nx: int, ny: int, params: PhaseFieldParams, phi_init: np.ndarray,
                 *, body_force_x: float = 0.0, walls: bool = True,
                 p_init: np.ndarray | None = None, ramp_steps: int = 400,
                 damp_steps: int = 0, damp_gamma: float = 0.0):
        super().__init__(nx, ny, walls=walls)
        self.p = params
        self.gx = float(body_force_x)
        self.ramp_steps = int(ramp_steps)
        self.damp_steps = int(damp_steps)
        self.damp_gamma = float(damp_gamma)
        self._step_count = 0
        if phi_init.shape != (ny, nx):
            raise ValueError("phi_init must be shaped (ny, nx)")
        if not np.all(np.isfinite(phi_init)):
            raise ValueError("phi_init must be finite")
        self.phi = np.clip(np.asarray(phi_init, dtype=np.float64), -1.0, 1.0)
        self.rho = self._density(self.phi)
        pbar0 = np.full((ny, nx), CS2)
        if p_init is not None:
            if p_init.shape != (ny, nx):
                raise ValueError("p_init must be shaped (ny, nx)")
            pbar0 = np.asarray(p_init, dtype=np.float64) / CS2
        feq = self._equilibrium_f(self.rho, np.zeros((ny, nx)), np.zeros((ny, nx)), pbar0)
        self.f[:] = feq
        self.g = np.zeros((9, ny, nx))
        geq = self._phase_equilibrium(self.phi, np.zeros((ny, nx)), np.zeros((ny, nx)))
        self.g[:] = geq
        self.ux = np.zeros((ny, nx))
        self.uy = np.zeros((ny, nx))
        self.pbar = pbar0
        self.injected_mass = 0.0
        self.vented_mass = 0.0
        self.injected_volume = 0.0
        self.vented_volume = 0.0
        self.shear = (0.0, 0.0)
        self.history: list[StepDiagnostics] = []

    # -- helpers ---------------------------------------------------------
    def _density(self, phi: np.ndarray) -> np.ndarray:
        a = 0.5 * (1.0 + phi)
        return self.p.rho_g + a * (self.p.rho_l - self.p.rho_g)

    def _viscosity(self, phi: np.ndarray) -> np.ndarray:
        a = 0.5 * (1.0 + phi)
        return self.p.nu_g + a * (self.p.nu_l - self.p.nu_g)

    def _phase_equilibrium(self, phi, ux, uy) -> np.ndarray:
        cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
        return W9[:, None, None] * phi[None] * (1.0 + 3.0 * cu)

    def gas_fraction(self) -> np.ndarray:
        return 0.5 * (1.0 - self.phi)

    def total_gas_volume(self) -> float:
        return float(np.sum(self.gas_fraction()))

    def total_mass(self) -> float:
        return float(np.sum(self._density(self.phi)))

    # -- phase forcing ---------------------------------------------------
    def _drive_ramp(self) -> float:
        if self.ramp_steps <= 0:
            return 1.0
        return min(1.0, (self._step_count + 1.0) / (2.0 * self.ramp_steps))

    def _ramp(self) -> float:
        """Soft start: ramp the capillary force so the pressure field can track it."""
        if self.ramp_steps <= 0:
            return 1.0
        return min(1.0, (self._step_count + 1.0) / self.ramp_steps)

    def _surface_force(self) -> tuple[np.ndarray, np.ndarray]:
        phi = self.phi
        per_y = not self.walls
        lap = _lap(phi, per_y)
        mu = 4.0 * self.p.beta * (phi ** 3 - phi) - self.p.kappa * lap
        gx, gy = _grad(phi, per_y)
        ramp = self._ramp()
        fx = ramp * mu * gx
        fy = ramp * mu * gy
        if self.walls:  # zero normal force on solid walls (neutral wetting)
            fy[0, :] = 0.0
            fy[-1, :] = 0.0
        return fx, fy

    def add_gas(self, mask: np.ndarray, dphi: float) -> None:
        """Inject gas with an explicit, exactly accounted volumetric source."""
        mask = np.asarray(mask, dtype=bool)
        if dphi <= 0.0:
            raise ValueError("dphi must be positive for injection")
        old = self.phi[mask]
        new = np.clip(old - dphi, -1.0, 1.0)
        d = old - new
        self.phi[mask] = new
        # keep the transported distributions consistent with the new phase field
        delta_phi = np.zeros_like(self.phi)
        delta_phi[mask] = new - old
        self.g += W9[:, None, None] * delta_phi[None]
        self.injected_volume += float(np.sum(0.5 * d))
        self.injected_mass += float(np.sum(0.5 * d * (self.p.rho_l - self.p.rho_g)))

    def vent_gas(self, mask: np.ndarray, dphi: float) -> None:
        """Remove gas downstream (vent / dissolution sink), exactly accounted."""
        mask = np.asarray(mask, dtype=bool)
        if dphi <= 0.0:
            raise ValueError("dphi must be positive for venting")
        old = self.phi[mask]
        new = np.clip(old + dphi, -1.0, 1.0)
        d = new - old
        self.phi[mask] = new
        delta_phi = np.zeros_like(self.phi)
        delta_phi[mask] = new - old
        self.g += W9[:, None, None] * delta_phi[None]
        self.vented_volume += float(np.sum(0.5 * d))
        self.vented_mass += float(np.sum(0.5 * d * (self.p.rho_l - self.p.rho_g)))

    # -- time stepping ---------------------------------------------------
    def step(self, n: int = 1, *, inject: tuple[np.ndarray, float] | None = None,
             vent: tuple[np.ndarray, float] | None = None,
             record: bool = False, record_every: int = 0) -> None:
        tau_phi = self.p.tau_phi
        delta = self.p.delta
        per_y = not self.walls
        for k in range(n):
            if inject is not None:
                self.add_gas(inject[0], inject[1])
            if vent is not None:
                self.vent_gas(vent[0], vent[1])

            rho = self._density(self.phi)
            nu = self._viscosity(self.phi)
            fsx, fsy = self._surface_force()
            fx = fsx + self.gx * self._drive_ramp()
            fy = fsy
            if self._step_count < self.damp_steps and self.damp_gamma > 0.0:
                # startup damping: pull the momentum to rest so the capillary
                # pressure field can establish itself before the drive ramps in
                w = self.damp_gamma * (1.0 - self._step_count / max(self.damp_steps, 1))
                # velocity damping referenced to the heavy phase: the light phase
                # (rho_g = 1/100) would otherwise be damped 100x too weakly
                fx = fx - w * self.p.rho_l * self.ux
                fy = fy - w * self.p.rho_l * self.uy
            ux, uy, pbar = self.momentum(fx, fy, rho)

            # conservative Allen-Cahn collision + sharpening source
            geq = self._phase_equilibrium(self.phi, ux, uy)
            pgx, pgy = _grad(self.phi, per_y)
            norm = np.sqrt(pgx ** 2 + pgy ** 2) + 1e-12
            sharp = (1.0 - self.phi ** 2) / delta
            amp = (self.p.mobility * sharp / tau_phi) / norm
            sx = amp * pgx
            sy = amp * pgy
            cu = CX[:, None, None] * sx[None] + CY[:, None, None] * sy[None]
            gsrc = W9[:, None, None] * 3.0 * cu
            gpost = self.g - (self.g - geq) / tau_phi + gsrc
            self.g = self._apply_wall_bc(_stream(gpost), gpost)
            raw = sum(self.g[i] for i in range(9))
            phi_c = np.clip(raw, -1.0, 1.0)
            excess = phi_c - raw
            if np.any(excess != 0.0):
                # node-wise clamp, then hand the clamped mass back globally so that
                # sum(phi) stays conserved to round-off (no phantom phase source)
                phi_c = phi_c - float(excess.sum()) / float(excess.size)
                self.g += W9[:, None, None] * (phi_c - raw)[None]
            self.phi = phi_c

            self.shear = self.collide_stream(ux, uy, pbar, rho, nu, fx, fy)
            self._step_count += 1
            self.ux, self.uy, self.pbar = ux, uy, pbar
            self.rho = rho

            if record and (record_every <= 0 or (k + 1) % record_every == 0):
                self.history.append(self.diagnostics(k + 1))

    def diagnostics(self, step: int) -> StepDiagnostics:
        return StepDiagnostics(
            step=step,
            phi_total=float(np.sum(self.phi)),
            mass_total=self.total_mass(),
            u_max=float(np.max(np.abs(self.ux))),
            p_mean=float(np.mean(CS2 * self.pbar)),
            ke=float(np.sum(0.5 * self.rho * (self.ux ** 2 + self.uy ** 2))),
        )

    # -- observables -----------------------------------------------------
    def pressure(self) -> np.ndarray:
        return CS2 * self.pbar

    def stress_tensor(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Second moment Pi_ab = sum_i c_ia c_ib f_i (equilibrium part is isotropic).

        The anisotropy ``Pi_yy - Pi_xx`` integrated across a flat interface gives
        the model surface tension, independently of the nominal sigma formula.
        """
        pxx = sum(CX[i] ** 2 * self.f[i] for i in range(9))
        pyy = sum(CY[i] ** 2 * self.f[i] for i in range(9))
        pxy = sum(CX[i] * CY[i] * self.f[i] for i in range(9))
        return pxx, pyy, pxy

    def wall_shear_mean(self) -> float:
        return 0.5 * (abs(self.shear[0]) + abs(self.shear[1]))

    def bulk_velocity(self) -> float:
        return float(np.sum(self.ux) / (self.nx * self.ny))


# ---------------------------------------------------------------------------
# initial conditions
# ---------------------------------------------------------------------------
def gas_patches(nx: int, ny: int, *, coverage: float, thickness: float,
                bottom: bool = True, top: bool = False) -> np.ndarray:
    """Gas (phi=-1) patches of ``thickness`` nodes covering ``coverage`` of the wall."""
    phi = np.ones((ny, nx), dtype=np.float64)
    n_cov = int(round(coverage * nx))
    if n_cov <= 0:
        return phi
    rows_b = int(round(thickness))
    if bottom:
        phi[1:1 + rows_b, :n_cov] = -1.0
    if top:
        phi[ny - 1 - rows_b:ny - 1, nx - n_cov:] = -1.0
    return smooth_phi(phi)


def smooth_phi(phi: np.ndarray, passes: int = 2) -> np.ndarray:
    """Light smoothing so the sharp initial mask relaxes onto the tanh profile."""
    out = phi.copy()
    for _ in range(passes):
        out = 0.25 * (np.roll(out, 1, axis=0) + np.roll(out, -1, axis=0)
                      + np.roll(out, 1, axis=1) + np.roll(out, -1, axis=1))
    return np.clip(out, -1.0, 1.0)


def initial_pressure(phi: np.ndarray, params: PhaseFieldParams, *,
                     p_out: float = CS2) -> np.ndarray:
    """Capillary-consistent pressure field (exact for a flat interface).

    Integrating ``dp/dn = mu_phi dphi/dn`` across an interface gives
    ``p = p_out + 4 beta (phi^3/3 - phi)``; using it removes the violent start-up
    transient that otherwise destroys high-density-ratio runs.
    """
    return p_out + 4.0 * params.beta * (phi ** 3 / 3.0 - phi)


def laplace_pressure(nx: int, ny: int, radius: float, sigma: float, *,
                     center: Sequence[float] | None = None, delta: float = 1.0,
                     p_out: float = CS2) -> np.ndarray:
    """Pressure field matching a 2D bubble (dP = sigma/R) so the start is quiet."""
    cx, cy = (0.5 * nx, 0.5 * ny) if center is None else center
    x = np.arange(nx)[None, :] - cx
    y = np.arange(ny)[:, None] - cy
    r = np.sqrt(x ** 2 + y ** 2)
    inside = np.clip(0.5 * (1.0 - np.tanh((r - radius) / delta)), 0.0, 1.0)
    return p_out + sigma / radius * inside


def circular_bubble(nx: int, ny: int, radius: float, *, center: Sequence[float] | None = None,
                    delta: float = 1.0) -> np.ndarray:
    """Static bubble with the equilibrium tanh profile (periodic box)."""
    cx, cy = (0.5 * nx, 0.5 * ny) if center is None else center
    x = np.arange(nx)[None, :] - cx
    y = np.arange(ny)[:, None] - cy
    r = np.sqrt(x ** 2 + y ** 2)
    return -np.tanh((r - radius) / delta)