"""Two-phase incompressible Navier-Stokes solver with conservative VOF transport.

Design notes (kept deliberately compact so the verification ladder stays
auditable):

* staggered MAC grid.  ``u`` lives on the ``nx`` periodic x-faces (face ``i``
  sits between cell ``i-1`` and cell ``i``), ``v`` on the ``ny+1`` y-faces
  (faces ``0`` and ``ny`` are the solid walls), scalars live at cell centres;
* momentum: minmod-limited second-order upwind advection, explicit centred
  viscous diffusion, gravity, continuum-surface-force (CSF) surface tension;
* pressure projection: variable-coefficient Poisson equation
  ``div(dt/rho grad p) = div(u*)`` solved with a preconditioned conjugate
  gradient, preconditioned by the exact constant-density FFT/DCT solver (the
  DCT-II basis diagonalises the Neumann wall condition);
* interface: direction-split CICSAM (Ubbink & Issa 1999) in flux form with a
  discrete divergence-correction term.

All quantities are SI.  ``F = 1`` is liquid, ``F = 0`` is gas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct

__all__ = ["VOFChannelFlow", "VOFConfig", "still_water_level"]


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class VOFConfig:
    """Discretisation and fluid properties for :class:`VOFChannelFlow`."""

    nx: int = 256
    ny: int = 64
    length_x: float = 3.0
    length_y: float = 0.75
    rho_liquid: float = 1000.0
    rho_gas: float = 1.0
    nu_liquid: float = 1.0e-6
    nu_gas: float = 1.5e-5
    sigma: float = 0.0728
    gravity: float = 9.81
    bottom_noslip: bool = True
    surface_tension: bool = True
    cfl: float = 0.30
    capillary_cfl: float = 0.50
    pressure_tol: float = 1.0e-5
    max_pressure_iterations: int = 80

    def __post_init__(self) -> None:
        if self.nx < 8 or self.ny < 8:
            raise ValueError("grid is too small")
        self.dx = self.length_x / self.nx
        self.dy = self.length_y / self.ny
        if abs(self.dx - self.dy) > 1.0e-9 * max(self.dx, self.dy):
            raise ValueError("square cells are required (choose ny accordingly)")
        if self.rho_liquid <= self.rho_gas:
            raise ValueError("liquid must be heavier than gas")

    def dt_limit(self, speed: float) -> float:
        """Stable explicit time step for the current maximum velocity."""
        dx = self.dx
        candidates = [
            self.cfl * dx / max(abs(speed), 1.0e-8),
            0.25 * math.sqrt(dx / self.gravity),
            0.40 * dx * dx / max(self.nu_liquid, 1.0e-12),
        ]
        if self.surface_tension:
            candidates.append(
                self.capillary_cfl
                * math.sqrt(
                    (self.rho_liquid + self.rho_gas) * dx**3 / (2.0 * math.pi * self.sigma)
                )
            )
        return min(candidates)


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def still_water_level(fraction: np.ndarray, dy: float) -> np.ndarray:
    """Column integral of ``F`` in length units, i.e. the free-surface height."""
    return fraction.sum(axis=1) * dy


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #
class VOFChannelFlow:
    """MAC projection solver with conservative VOF interface transport."""

    def __init__(self, cfg: VOFConfig, fraction: np.ndarray | None = None) -> None:
        self.cfg = cfg
        nx, ny = cfg.nx, cfg.ny
        if fraction is None:
            fraction = np.zeros((nx, ny))
        fraction = np.asarray(fraction, dtype=float)
        if fraction.shape != (nx, ny):
            raise ValueError("fraction array shape does not match the grid")
        self.F = np.clip(fraction, 0.0, 1.0)
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny + 1))
        self.p = np.zeros((nx, ny))
        self.t = 0.0
        self.pressure_iterations = 0
        self.time_steps = 0
        self.clipped_volume = 0.0
        self.reference_volume = float(self.F.sum()) * cfg.dx * cfg.dy
        self._kx, self._ky = self._wavenumbers()

    # -- staggered-grid helpers ------------------------------------------- #
    def _wavenumbers(self) -> tuple[np.ndarray, np.ndarray]:
        nx, ny = self.cfg.nx, self.cfg.ny
        m = np.arange(nx)
        kx = 2.0 / self.cfg.dx * np.sin(math.pi * np.minimum(m, nx - m) / nx)
        n = np.arange(ny)
        ky = 2.0 / self.cfg.dy * np.sin(math.pi * n / (2.0 * ny))
        return kx, ky

    @property
    def rho(self) -> np.ndarray:
        cfg = self.cfg
        return cfg.rho_gas + (cfg.rho_liquid - cfg.rho_gas) * self.F

    @property
    def nu(self) -> np.ndarray:
        cfg = self.cfg
        return cfg.nu_gas + (cfg.nu_liquid - cfg.nu_gas) * self.F

    def _face_density_x(self) -> np.ndarray:
        rho = self.rho
        return 0.5 * (rho + np.roll(rho, 1, axis=0))

    def _face_density_y(self) -> np.ndarray:
        rho = self.rho
        face = np.empty((self.cfg.nx, self.cfg.ny + 1))
        face[:, 1:-1] = 0.5 * (rho[:, 1:] + rho[:, :-1])
        face[:, 0] = rho[:, 0]
        face[:, -1] = rho[:, -1]
        return face

    def divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        dx, dy = self.cfg.dx, self.cfg.dy
        return (np.roll(u, -1, axis=0) - u) / dx + (v[:, 1:] - v[:, :-1]) / dy

    def max_speed(self) -> float:
        return float(max(np.abs(self.u).max(), np.abs(self.v).max()))

    def _at_u_faces(self, field: np.ndarray) -> np.ndarray:
        return 0.5 * (field + np.roll(field, 1, axis=0))

    def _at_v_faces(self, field: np.ndarray) -> np.ndarray:
        out = np.empty((self.cfg.nx, self.cfg.ny + 1))
        out[:, 1:-1] = 0.5 * (field[:, 1:] + field[:, :-1])
        out[:, 0] = field[:, 0]
        out[:, -1] = field[:, -1]
        return out

    # -- momentum ---------------------------------------------------------- #
    def _advect_u(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """minmod-limited upwind advection of the x-momentum (face centred)."""
        dx, dy = self.cfg.dx, self.cfg.dy
        vv = 0.25 * (
            np.roll(v, 1, axis=0)[:, :-1]
            + v[:, :-1]
            + np.roll(v, 1, axis=0)[:, 1:]
            + v[:, 1:]
        )
        fm1 = np.roll(u, 1, axis=0)
        fm2 = np.roll(u, 2, axis=0)
        fp1 = np.roll(u, -1, axis=0)
        d1 = u - fm1
        curv_l = d1 - (fm1 - fm2)
        curv_r = fp1 - u - d1
        du_dx = np.where(
            u > 0.0,
            (d1 + 0.5 * _minmod(curv_l, curv_r)) / dx,
            (fp1 - u - 0.5 * _minmod(curv_r, curv_l)) / dx,
        )
        du_dy = np.empty_like(u)
        du_dy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dy)
        du_dy[:, 0] = (u[:, 1] - u[:, 0]) / dy
        du_dy[:, -1] = (u[:, -1] - u[:, -2]) / dy
        return u * du_dx + vv * du_dy

    def _advect_v(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """minmod-limited upwind advection of the y-momentum (face centred)."""
        dx, dy = self.cfg.dx, self.cfg.dy
        uc = 0.5 * (u + np.roll(u, -1, axis=0))
        uu = np.empty((self.cfg.nx, self.cfg.ny + 1))
        uu[:, 1:-1] = 0.5 * (uc[:, :-1] + uc[:, 1:])
        uu[:, 0] = uc[:, 0]
        uu[:, -1] = uc[:, -1]
        d1 = np.zeros_like(v)
        d1[:, 1:] = v[:, 1:] - v[:, :-1]
        d2 = np.zeros_like(v)
        d2[:, 2:] = d1[:, 1:-1]
        d3 = np.zeros_like(v)
        d3[:, :-1] = d1[:, 1:]
        lim = _minmod(d3 - d1, d1 - d2)
        adv = np.zeros_like(v)
        inner = slice(1, -1)
        adv[:, inner] = np.where(
            v[:, inner] > 0.0,
            v[:, inner] * (d1[:, inner] + 0.5 * lim[:, inner]) / dy,
            v[:, inner] * (d3[:, inner] - 0.5 * lim[:, inner]) / dy,
        )
        dv_dx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dx)
        return adv + uu * dv_dx

    def _viscous_u(self, u: np.ndarray) -> np.ndarray:
        dx, dy = self.cfg.dx, self.cfg.dy
        nu_u = self._at_u_faces(self.nu)
        lap_y = np.pad(
            (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / dy**2, ((0, 0), (1, 1))
        )
        lap = (np.roll(u, -1, axis=0) - 2.0 * u + np.roll(u, 1, axis=0)) / dx**2 + lap_y
        return nu_u * lap

    def _viscous_v(self, v: np.ndarray) -> np.ndarray:
        dx, dy = self.cfg.dx, self.cfg.dy
        nu_v = self._at_v_faces(self.nu)
        lap = np.zeros_like(v)
        lap[:, 1:-1] = (v[:, 2:] - 2.0 * v[:, 1:-1] + v[:, :-2]) / dy**2
        lap += (np.roll(v, -1, axis=0) - 2.0 * v + np.roll(v, 1, axis=0)) / dx**2
        return nu_v * lap

    # -- surface tension ---------------------------------------------------- #
    def _csf_acceleration(self) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = self.cfg.dx, self.cfg.dy
        F = self.F
        kern = np.array([0.25, 0.5, 0.25])
        Fs = np.apply_along_axis(lambda m: np.convolve(m, kern, mode="same"), 0, F)
        Fs = np.apply_along_axis(lambda m: np.convolve(m, kern, mode="same"), 1, Fs)
        gx = (np.roll(Fs, -1, axis=0) - np.roll(Fs, 1, axis=0)) / (2.0 * dx)
        gy = np.zeros_like(F)
        gy[:, 1:-1] = (Fs[:, 2:] - Fs[:, :-2]) / (2.0 * dy)
        gy[:, 0] = (Fs[:, 1] - Fs[:, 0]) / dy
        gy[:, -1] = (Fs[:, -1] - Fs[:, -2]) / dy
        mag = np.sqrt(gx * gx + gy * gy) + 1.0e-12
        nx = gx / mag
        ny = gy / mag
        div = (np.roll(nx, -1, axis=0) - np.roll(nx, 1, axis=0)) / (2.0 * dx)
        div[:, 1:-1] += (ny[:, 2:] - ny[:, :-2]) / (2.0 * dy)
        div[:, 0] += (ny[:, 1] - ny[:, 0]) / dy
        div[:, -1] += (ny[:, -1] - ny[:, -2]) / dy
        scale = -self.cfg.sigma * div / (0.5 * (self.cfg.rho_liquid + self.cfg.rho_gas))
        return scale * gx, scale * gy

    # -- pressure ----------------------------------------------------------- #
    def _poisson_fft(self, rhs: np.ndarray) -> np.ndarray:
        """Exact constant-density Neumann/periodic Poisson solve (DCT-II + FFT)."""
        f = rhs - rhs.mean()
        y = dct(f, type=2, axis=1, norm="ortho")
        yh = np.fft.fft(y, axis=0, norm="ortho")
        lam = -(self._kx[:, None] ** 2 + self._ky[None, :] ** 2)
        lam[0, 0] = 1.0
        qh = yh / lam
        qh[0, 0] = 0.0
        qy = np.fft.ifft(qh, axis=0, norm="ortho")
        return idct(qy.real, type=2, axis=1, norm="ortho")

    def _apply_pressure_operator(
        self, p: np.ndarray, ax: np.ndarray, ay: np.ndarray
    ) -> np.ndarray:
        dx, dy = self.cfg.dx, self.cfg.dy
        fx = ax * ((p - np.roll(p, 1, axis=0)) / dx)
        div = (np.roll(fx, -1, axis=0) - fx) / dx
        gy = np.zeros((self.cfg.nx, self.cfg.ny + 1))
        gy[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dy
        fy = ay * gy
        div += (fy[:, 1:] - fy[:, :-1]) / dy
        return div

    def _solve_pressure(
        self, rhs: np.ndarray, ax: np.ndarray, ay: np.ndarray
    ) -> np.ndarray:
        cfg = self.cfg
        rhs = rhs - rhs.mean()
        scale = float(np.mean(1.0 / ax))
        p = self.p - self.p.mean() if self.p.shape == rhs.shape else np.zeros_like(rhs)
        r = rhs - self._apply_pressure_operator(p, ax, ay) if p.any() else rhs.copy()
        z = self._poisson_fft(r / scale) / scale
        d = z.copy()
        rz = float(np.sum(r * z))
        norm0 = float(np.sqrt(np.sum(r * r))) or 1.0
        for it in range(1, cfg.max_pressure_iterations + 1):
            Ad = self._apply_pressure_operator(d, ax, ay)
            denom = float(np.sum(d * Ad))
            if denom == 0.0:
                self.pressure_iterations = it
                break
            alpha = rz / denom
            p = p + alpha * d
            r = r - alpha * Ad
            if float(np.sqrt(np.sum(r * r))) <= cfg.pressure_tol * norm0:
                self.pressure_iterations = it
                break
            z = self._poisson_fft(r / scale) / scale
            rz_new = float(np.sum(r * z))
            d = z + (rz_new / rz) * d
            rz = rz_new
        else:
            self.pressure_iterations = cfg.max_pressure_iterations
        return p

    # -- VOF transport ------------------------------------------------------ #
    def _interface_cos(self, axis: int) -> np.ndarray:
        dx, dy = self.cfg.dx, self.cfg.dy
        Fs = self.F
        gx = (np.roll(Fs, -1, axis=0) - np.roll(Fs, 1, axis=0)) / (2.0 * dx)
        gy = np.zeros_like(Fs)
        gy[:, 1:-1] = (Fs[:, 2:] - Fs[:, :-2]) / (2.0 * dy)
        gy[:, 0] = (Fs[:, 1] - Fs[:, 0]) / dy
        gy[:, -1] = (Fs[:, -1] - Fs[:, -2]) / dy
        mag = np.sqrt(gx * gx + gy * gy) + 1.0e-12
        return np.abs(gx) / mag if axis == 0 else np.abs(gy) / mag

    def _cicsam_face(
        self, F: np.ndarray, vel: np.ndarray, axis: int, dt: float, cos_theta: np.ndarray
    ) -> np.ndarray:
        """CICSAM face values for one sweep direction."""
        dx = self.cfg.dx
        c = np.clip(vel * dt / dx, -0.95, 0.95)
        down = np.roll(F, 1, axis=axis)
        up = np.roll(F, 2, axis=axis)
        up2 = np.roll(F, -1, axis=axis)
        pos = vel > 0.0
        Fd = np.where(pos, down, F)
        Fa = np.where(pos, F, down)
        Fu = np.where(pos, up, up2)
        denom = Fa - Fu
        safe = np.abs(denom) > 1.0e-12
        FtD = np.clip((Fd - Fu) / np.where(safe, denom, 1.0), 0.0, 1.0)
        cabs = np.abs(c)
        FtUQ = np.clip((8.0 * cabs * FtD + (1.0 - cabs) * (6.0 * FtD + 3.0)) / 8.0, FtD, 1.0)
        FtHC = np.clip(np.where(FtD < cabs, FtD / np.maximum(cabs, 1.0e-12), 1.0), FtD, 1.0)
        gamma = np.clip(cos_theta, 0.0, 1.0) ** 2
        Ft = np.clip(gamma * FtHC + (1.0 - gamma) * FtUQ, FtD, 1.0)
        return np.where(safe, Fu + Ft * denom, Fd)

    def _sweep(self, dt: float, axis: int) -> None:
        dx = self.cfg.dx
        F = self.F
        nxc, nyc = self.cfg.nx, self.cfg.ny
        cos = self._interface_cos(axis)
        if axis == 0:
            face_F = self._cicsam_face(F, self.u, 0, dt, cos)
            flux = self.u * face_F
            div = (np.roll(flux, -1, axis=0) - flux) / dx
            div_corr = (np.roll(self.u, -1, axis=0) - self.u) / dx
        else:
            vint = self.v[:, :nyc]
            face_F = np.empty((nxc, nyc + 1))
            face_F[:, :nyc] = self._cicsam_face(F, vint, 1, dt, cos)
            face_F[:, nyc] = face_F[:, 0]
            flux = self.v * face_F
            div = (flux[:, 1:] - flux[:, :-1]) / dx
            div_corr = (self.v[:, 1:] - self.v[:, :-1]) / dx
        F_new = F - dt * div + dt * F * div_corr
        clipped = np.clip(F_new, 0.0, 1.0)
        delta = float((F_new - clipped).sum())
        if abs(delta) > 0.0:
            # Algebraic VOF can overshoot by the amount that the donor-cell
            # bound allows at the CFL limit.  Redistributing the (tiny) excess
            # over the interface band keeps the transport exactly conservative
            # instead of silently losing liquid volume.
            self.clipped_volume += abs(delta) * dx * self.cfg.dy
            weight = clipped * (1.0 - clipped)
            total = float(weight.sum())
            if total > 1.0e-12:
                clipped = np.clip(clipped + delta * weight / total, 0.0, 1.0)
            else:
                clipped = np.clip(clipped + delta / clipped.size, 0.0, 1.0)
        self.F = clipped

    # -- boundary conditions ------------------------------------------------- #
    def _apply_walls(self) -> None:
        cfg = self.cfg
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0
        if cfg.bottom_noslip:
            self.u[:, 0] = 0.0
        else:
            self.u[:, 0] = self.u[:, 1]
        self.u[:, -1] = self.u[:, -2]

    # -- public API ---------------------------------------------------------- #
    def step(self, dt: float | None = None) -> float:
        cfg = self.cfg
        if dt is None:
            dt = cfg.dt_limit(self.max_speed())
        rho_x = self._face_density_x()
        rho_y = self._face_density_y()
        ax = dt / rho_x
        ay = np.zeros_like(rho_y)
        ay[:, 1:-1] = dt / rho_y[:, 1:-1]

        adv_u = self._advect_u(self.u, self.v)
        adv_v = self._advect_v(self.u, self.v)
        visc_u = self._viscous_u(self.u)
        visc_v = self._viscous_v(self.v)
        u_star = self.u + dt * (-adv_u + visc_u)
        v_star = self.v + dt * (-adv_v + visc_v - cfg.gravity)
        if cfg.surface_tension:
            fx, fy = self._csf_acceleration()
            u_star = u_star + dt * self._at_u_faces(fx)
            v_star = v_star + dt * self._at_v_faces(fy)

        # Solid-wall normal velocities are imposed on the predictor as well: a
        # uniform gravity-induced divergence would be annihilated by the
        # Poisson compatibility condition and the hydrostatic pressure would be
        # lost, so the wall rows are zeroed before the divergence is formed.
        # Wall conditions are imposed on the predictor and are *not* re-imposed
        # after the projection: overwriting a face after the pressure solve
        # would destroy the discrete divergence-free property that the VOF
        # transport relies on for mass conservation.
        v_star[:, 0] = 0.0
        v_star[:, -1] = 0.0
        if cfg.bottom_noslip:
            u_star[:, 0] = 0.0
        else:
            u_star[:, 0] = u_star[:, 1]
        u_star[:, -1] = u_star[:, -2]
        div = self.divergence(u_star, v_star)
        p = self._solve_pressure(div, ax, ay)
        u_new = u_star - ax * ((p - np.roll(p, 1, axis=0)) / cfg.dx)
        v_new = v_star.copy()
        v_new[:, 1:-1] = v_star[:, 1:-1] - ay[:, 1:-1] * (p[:, 1:] - p[:, :-1]) / cfg.dy
        v_new[:, 0] = 0.0
        v_new[:, -1] = 0.0
        self.u, self.v, self.p = u_new, v_new, p

        self._sweep(0.5 * dt, 0)
        self._sweep(dt, 1)
        self._sweep(0.5 * dt, 0)
        self.t += dt
        self.time_steps += 1
        return dt

    def run(self, t_end: float, *, max_steps: int = 400000) -> None:
        while self.t < t_end - 1.0e-12 and self.time_steps < max_steps:
            dt = min(self.cfg.dt_limit(self.max_speed()), t_end - self.t)
            self.step(dt)

    # -- diagnostics --------------------------------------------------------- #
    def volume(self) -> float:
        return float(self.F.sum()) * self.cfg.dx * self.cfg.dy

    def mass_error(self) -> float:
        return (self.volume() - self.reference_volume) / max(self.reference_volume, 1e-30)

    def surface_height(self) -> np.ndarray:
        return still_water_level(self.F, self.cfg.dy)